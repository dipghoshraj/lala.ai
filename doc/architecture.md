# lala.ai — System Architecture

> **Current state:** Phase 0 in progress — two-binary system (`lala` CLI client + `LLML` inference server) communicating over HTTP. LLML now supports multiple model roles (`reasoning` / `decision`) with per-request temperature overrides. PostgreSQL/pgvector is provisioned but not yet wired into the live request loop.

---

## 1. Repository Layout

```
lala.ai/
├── ai-config.yaml          # Shared model configuration (read by LLML at startup)
├── psql.Dockerfile         # PostgreSQL 18 + pgvector image
├── lala/                   # Binary 1 — CLI client
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs         # Entry point
│       ├── cli.rs          # REPL loop + spinner + history
│       └── agent/
│           ├── mod.rs
│           └── model.rs    # ApiClient — HTTP wrapper for LLML
└── LLML/                   # Binary 2 — Inference server
    ├── Cargo.toml
    └── src/
        ├── main.rs         # Entry point — loads config, starts Axum
        ├── api/
        │   └── mod.rs      # Router, handlers, prompt builder, OpenAI-compatible types
        ├── model/
        │   ├── mod.rs
        │   ├── model.rs    # ModelRunner — wraps llama_cpp
        │   └── registry.rs # ModelRegistry — role → ModelRunner map
        └── loalYaml/
            ├── mod.rs
            └── loadYaml.rs # Deserializes ai-config.yaml
```

---

## 2. High-Level System Diagram

```mermaid
graph TD
    User["👤 User (terminal)"]
    Lala["lala\nCLI binary"]
    LLML["LLML\nInference server\n:3000"]
    Config["ai-config.yaml\n(2 model roles)"]
    GGUF["*.gguf model files\n(local filesystem)"]
    DB[("PostgreSQL\n+ pgvector\n:5432")]
    Registry["ModelRegistry\nrole → ModelRunner"]

    User -->|"stdin / rustyline"| Lala
    Lala -->|"POST /v1/chat/completions\n{model: 'reasoning'|'decision', ...}"| LLML
    LLML -->|"response JSON"| Lala
    Lala -->|"print reply"| User

    Config -->|"read on startup"| LLML
    GGUF -->|"mmap via llama_cpp"| Registry
    LLML --> Registry

    DB -.->|"planned — RAG + memory"| Lala

    style DB stroke-dasharray: 5 5
```

Solid lines = live today. Dashed = provisioned, not yet in the request loop.

---

## 3. Binary Entry Points

### 3.1 `lala` — CLI client

**Crate:** `lala/`  
**Entry:** `lala/src/main.rs`

```
cargo run [-- <LLML_API_URL>]
```

URL resolution order:
1. First positional CLI argument
2. `LLML_API_URL` environment variable
3. Fallback: `http://localhost:3000`

`main()` calls `cli::run(&api_url)` — that's the only thing it does.

---

### 3.2 `LLML` — inference server

**Crate:** `LLML/`  
**Entry:** `LLML/src/main.rs`

```
cargo run          # reads ../ai-config.yaml, binds 0.0.0.0:3000
```

Startup sequence (see §5 for detail):
1. Init `tracing` subscriber (level from `RUST_LOG`)
2. `load_config("../ai-config.yaml")`
3. For each model in config → `ModelRunner::load(path, params)` → register in `ModelRegistry`
4. Wrap registry in `Arc`, build Axum router
5. `TcpListener::bind("0.0.0.0:3000")` → `axum::serve()`

---

## 4. lala — CLI Client Flow

```mermaid
sequenceDiagram
    participant User
    participant rustyline
    participant cli as cli::run()
    participant spinner as spinner thread
    participant ApiClient

    User->>rustyline: type input
    rustyline-->>cli: line string

    alt /clear
        cli->>cli: truncate history to system prompt
    else /exit or Ctrl-C
        cli->>cli: break loop
    else normal message
        cli->>cli: push ChatMessage{role:"user"} to history
        cli->>spinner: spawn background thread
        cli->>ApiClient: chat(&history, None, None)\n[model_role=None → server default]
        ApiClient->>LLML: POST /v1/chat/completions\n{messages:[...]}
        LLML-->>ApiClient: ChatResponse JSON
        ApiClient-->>cli: Ok(reply) or Err(e)
        cli->>spinner: set running=false
        spinner-->>cli: join (erase spinner line)

        alt Ok(reply)
            cli->>cli: push ChatMessage{role:"assistant"} to history
            cli->>User: println!(reply)
        else Err(e)
            cli->>cli: pop pending user message (keep history consistent)
            cli->>User: eprintln!(error)
        end
    end
```

**Conversation history** is a `Vec<ChatMessage>` that permanently holds the system prompt at index 0:

```
index 0   { role: "system",    content: SYSTEM_PROMPT }
index 1   { role: "user",      content: "..." }
index 2   { role: "assistant", content: "..." }
...
```

The entire vector is sent on every request so the model maintains multi-turn context. `/clear` truncates back to `len == 1`.

### ApiClient — model role selection

`ApiClient` in `lala/src/agent/model.rs` exposes three call paths:

| Method | `"model"` field sent | Temperature used |
|--------|---------------------|-----------------|
| `chat(&msgs, max_tokens, None)` | omitted → server picks first registered | model config default |
| `reason(&msgs, max_tokens)` | `"reasoning"` | 0.7 (from config) |
| `decide(&msgs, max_tokens)` | `"decision"` | 0.3 (from config) |

`cli.rs` currently calls `chat(…, None, None)` (server default). The `reason()` / `decide()` methods are ready for the Phase 0 planner loop — see §6.

---

## 5. LLML — Inference Server Flow

### 5.1 Multi-Model Configuration (`ai-config.yaml`)

LLML loads **all** models declared in `ai-config.yaml` at startup and stores them in a `ModelRegistry` keyed by `role`. Two roles are defined by default:

| Role | Type | Temperature | `n_ctx` | Purpose |
|------|------|-------------|---------|---------|
| `reasoning` | Reasoning | 0.7 | 2048 | Deep analysis, multi-step thinking |
| `decision` | Decision | 0.3 | 512 | Short, deterministic action selection |

The `role` field in each model entry is the API-facing key. Both roles currently load the same GGUF file — swap `modelPath` independently to use different checkpoints.

### 5.2 Startup

```mermaid
flowchart TD
    A[main: parse RUST_LOG] --> B[init tracing subscriber]
    B --> C[load_config ai-config.yaml]
    C --> D{models empty?}
    D -- yes --> E[return Err — exit]
    D -- no --> F[for each model entry]
    F --> G[params_from_config\nreads temperature, max_tokens,\nn_gpu_layers, n_threads, n_ctx, n_batch]
    G --> H[ModelRunner::load GGUF file]
    H --> I["registry.register(role, runner)"]
    I --> F
    F --> J[Arc wrap registry]
    J --> K[api::create_router registry]
    K --> L[TcpListener bind 0.0.0.0:3000]
    L --> M[axum::serve — event loop]
```

### 5.3 Request: POST /v1/chat/completions

```mermaid
sequenceDiagram
    participant lala as lala ApiClient
    participant axum as Axum handler
    participant registry as ModelRegistry
    participant blocking as spawn_blocking thread
    participant runner as ModelRunner

    lala->>axum: POST /v1/chat/completions\n{model?, messages, max_tokens?, temperature?}

    axum->>axum: validate messages not empty
    axum->>registry: resolve role\n(req.model → exact lookup, or first())
    registry-->>axum: &ModelRunner  OR  400 unknown role / 500 empty registry

    axum->>axum: build_prompt(messages)\nMistral [INST]...[/INST] format

    axum->>blocking: spawn_blocking closure\n(prompt, max_tokens, temperature)
    blocking->>runner: generate_from_prompt(prompt, max_tokens, temperature)
    note over runner: temperature = req.temperature\n  ?? model config default
    runner->>runner: create_session(SessionParams)
    runner->>runner: advance_context(prompt)
    runner->>runner: StandardSampler with resolved temperature
    runner->>runner: collect tokens until max or [/INST] stop
    runner-->>blocking: Ok(output_string)
    blocking-->>axum: output_string

    axum-->>lala: ChatResponse JSON\n{choices:[{message:{content}}], usage, ...}
```

### 5.3 Request: GET /v1/models

Returns all registered role names (e.g. `"reasoning"`, `"decision"`) in OpenAI list format. No inference involved.

---

## 6. Configuration — `ai-config.yaml`

Parsed by `LLML/src/loalYaml/loadYaml.rs` into typed structs on startup.

```mermaid
classDiagram
    class AiConfig {
        +u32 version
        +ModelTypes model_types
        +Vec~Model~ models
    }
    class ModelTypes {
        +Vec~String~ types
    }
    class Model {
        +String name
        +String description
        +String model_type
        +String role
        +Vec~Parameter~ parameters
        +String model_path
    }
    class Parameter {
        +String name
        +String description
        +String param_type
        +serde_yaml::Value default
    }
    AiConfig --> ModelTypes
    AiConfig --> Model
    Model --> Parameter
```

**Registered models (current `ai-config.yaml`):**

| Role | Model name | Temperature | max_tokens | n_ctx |
|------|-----------|-------------|------------|-------|
| `reasoning` | mistral-reasoning | 0.7 | 512 | 2048 |
| `decision` | mistral-decision | 0.3 | 256 | 512 |

Both point to the same GGUF file (`mistral-7b-v0.1.Q4_K_M.gguf`). Different `ModelRunner` instances are loaded with different params.

---

## 7. Model Layer Internals

```mermaid
classDiagram
    class ModelRegistry {
        -HashMap~String, ModelRunner~ models
        +register(role, runner)
        +get(role) Option~ModelRunner~
        +roles() Vec~String~
        +first() Option~str,ModelRunner~
    }
    class ModelRunner {
        +LlamaModel model
        -ModelParams params
        +load(path, params) ModelRunner
        +generate_from_prompt(prompt, max_tokens, temperature) String
    }
    class ModelParams {
        +f32 temperature
        +usize max_tokens
        +u32 n_gpu_layers
        +u32 n_threads
        +u32 n_ctx
        +u32 n_batch
    }
    ModelRegistry "1" --> "*" ModelRunner : contains
    ModelRunner --> ModelParams : owns
```

**Thread safety:** `ModelRunner` is `Send + Sync` (llama_cpp C++ objects are thread-safe). Each HTTP request runs inference in `tokio::task::spawn_blocking` — no async executor blocking. Each call creates a fresh `LlamaSession` so there is no context bleed between concurrent requests.

---

## 8. Prompt Format

`build_prompt()` in `api/mod.rs` converts the OpenAI `messages` array into the Mistral/Llama instruction format:

```
<s>[INST] {system_prompt}

{first_user_message} [/INST] {assistant_reply} </s>[INST] {next_user} [/INST]...
```

The output stream is cut early when a `[/INST]` marker appears in generated tokens, preventing prompt leakage.

---

## 9. HTTP API Reference

Both endpoints live on `LLML` at port `3000`.

### POST `/v1/chat/completions`

**Request:**
```json
{
  "model": "reasoning",
  "messages": [
    { "role": "system",    "content": "You are a helpful assistant." },
    { "role": "user",      "content": "What is Rust?" }
  ],
  "max_tokens": 200
}
```

| Field | Required | Notes |
|-------|----------|-------|
| `messages` | yes | Non-empty array. First element may be `system`. |
| `model` | no | Role key from registry. Omit to use first registered model. |
| `max_tokens` | no | Overrides the config default for this request. |
| `temperature` | no | Overrides the model config default for this request (0.0–2.0). |

**Response:** OpenAI-compatible `ChatResponse` with `choices[0].message.content`.

#### curl examples

**1. Default model (server picks first registered — `reasoning`):**
```sh
curl -s http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      { "role": "user", "content": "What is Rust?" }
    ]
  }' | jq '.choices[0].message.content'
```

**2. Explicit `reasoning` model with a system prompt and multi-turn history:**
```sh
curl -s http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "reasoning",
    "messages": [
      { "role": "system",    "content": "You are a helpful AI assistant named lala." },
      { "role": "user",      "content": "Explain ownership in Rust." }
    ],
    "max_tokens": 512
  }' | jq '.choices[0].message.content'
```

**3. `decision` model — short, deterministic output:**
```sh
curl -s http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "decision",
    "messages": [
      { "role": "user", "content": "Should I use Vec or LinkedList for a stack in Rust? Answer in one sentence." }
    ],
    "max_tokens": 64
  }' | jq '.choices[0].message.content'
```

**4. Override temperature at request time:**
```sh
curl -s http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "reasoning",
    "messages": [
      { "role": "user", "content": "Write a creative haiku about memory safety." }
    ],
    "max_tokens": 100,
    "temperature": 1.2
  }' | jq '.choices[0].message.content'
```

**5. Multi-turn conversation (pass full history on each request):**
```sh
curl -s http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "reasoning",
    "messages": [
      { "role": "system",    "content": "You are lala, a concise technical assistant." },
      { "role": "user",      "content": "What is a borrow checker?" },
      { "role": "assistant", "content": "The borrow checker is a Rust compiler component that enforces memory safety rules at compile time." },
      { "role": "user",      "content": "How does it relate to lifetimes?" }
    ],
    "max_tokens": 256
  }' | jq '.choices[0].message.content'
```

### GET `/v1/models`

Returns all registered roles. Example:
```json
{
  "object": "list",
  "data": [
    { "id": "decision", "object": "model" },
    { "id": "reasoning", "object": "model" }
  ]
}
```

#### curl example

```sh
curl -s http://localhost:3000/v1/models | jq .
```

---

## 10. ApiClient — `lala/src/agent/model.rs`

The `ApiClient` struct is the sole boundary between `lala` and `LLML`. It uses `reqwest::blocking::Client` (no timeout — CPU inference can be slow).

| Method | Description |
|--------|-------------|
| `chat(messages, max_tokens, model_role)` | Core — sends full history, returns reply string |
| `reason(messages, max_tokens)` | Shortcut — selects `ModelRole::Reasoning` |
| `decide(messages, max_tokens)` | Shortcut — selects `ModelRole::Decision` |

`ModelRole::as_str()` maps enum variants to the string keys expected by the LLML server (`"reasoning"`, `"decision"`).

---

## 11. Infrastructure — PostgreSQL + pgvector

```mermaid
flowchart LR
    dockerfile["psql.Dockerfile\npostgres:18 + pgvector"]
    container["Docker container\n:5432"]
    init["init.sql\n(to be created)"]
    lala_db["lala/src/db/\nconnection.rs\n(sqlx PgPool)"]

    dockerfile --> container
    init -->|"auto-run on first start"| container
    lala_db -.->|"DATABASE_URL env var"| container

    style lala_db stroke-dasharray: 5 5
    style init stroke-dasharray: 5 5
```

Docker setup:
```sh
docker build -f psql.Dockerfile -t lala-postgres .
docker run -e POSTGRES_PASSWORD=postgres -p 5432:5432 lala-postgres
```

`DATABASE_URL=postgres://postgres:postgres@localhost:5432/lala` must be set for DB operations. The `sqlx` dependency is declared in `lala/Cargo.toml` but DB code is not yet wired into the live request loop.

**Planned tables** (from `doc/future/design.md`):

| Table | Purpose |
|-------|---------|
| `sessions` | Conversation session metadata |
| `messages` | Per-turn message content + vector embeddings |
| `documents` | Ingested source documents |
| `document_chunks` | Chunked text + `bge-small` embeddings (pgvector) |
| `queries` | Per-turn query log |
| `retrieval_results` | Which chunks were retrieved for which query |
| `answers` | Generated answer text |
| `answer_citations` | Which chunks were cited in each answer |

---

## 12. Current vs. Target Architecture

```mermaid
flowchart TD
    subgraph current ["Current (implemented)"]
        U1[User] --> CLI1[lala CLI]
        CLI1 -->|HTTP| LLML1[LLML server]
        LLML1 --> LM1[llama_cpp model]
    end

    subgraph target ["Target — Phase 0 (doc/phase0.md)"]
        U2[User] --> IF[Interface Layer\ncli.rs]
        IF --> AG[Agent Layer\nPlanner / Reasoner / Executor]
        AG --> RAG[RAG Layer\nretrieve / store / embed / chunk]
        AG --> LLM[Model Layer\ngenerate]
        RAG --> DB2[(PostgreSQL\n+ pgvector)]
    end
```

**Gap summary:**

| Concern | Current | Target (Phase 0) |
|---------|---------|-----------------|
| REPL / input | `cli.rs` direct loop | Interface Layer calls `Agent::run()` |
| Multi-model routing (server) | `ModelRegistry` + role-based routing ✅ | Done |
| Per-request temperature (server) | Wired through `generate_from_prompt` ✅ | Done |
| Role selection (client) | `reason()` / `decide()` methods exist ✅ | Needs planner to call them |
| Two-step agent loop | Single `chat(…, None, None)` call | Agent: `reason()` → parse → `decide()` |
| Prompt building | `api/mod.rs build_prompt()` in LLML | Agent Layer (Executor/Reasoner) |
| Retrieval | None | RAG Layer `retrieve(query, k)` |
| Document ingestion | None | RAG Layer `store(text)` |
| DB access | `db/connection.rs` declared but unused | RAG Layer only via `PgPool` |

---

## 13. Key Dependencies

### lala
| Crate | Purpose |
|-------|---------|
| `rustyline` | Readline-style REPL with history navigation |
| `reqwest` (blocking + json) | HTTP client for LLML API |
| `serde` / `serde_json` | JSON serialization of ChatMessage arrays |
| `anyhow` | Error propagation |
| `sqlx` | PostgreSQL client (declared, not yet active in loop) |

### LLML
| Crate | Purpose |
|-------|---------|
| `llama_cpp` | Local GGUF model loading and token generation |
| `axum` | Async HTTP server and router |
| `tokio` | Async runtime |
| `serde` / `serde_yaml` / `serde_json` | YAML config + JSON API serialization |
| `tracing` / `tracing-subscriber` | Structured logging (level via `RUST_LOG`) |
| `anyhow` | Error propagation |
| `uuid` | Response IDs |
