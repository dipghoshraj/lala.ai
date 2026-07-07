# lala.ai — System Architecture

> **Current state:** Phase 0 complete — five-layer architecture (Interface → Agent → RAG → Model + DB). `lala` Rust CLI + `LLML` Python inference server + `telegram` Python bot connected over HTTP. RAG retrieval is **live** on every query via PostgreSQL FTS (`tsvector` + GIN) + pgvector (`vector(384)` IVFFlat) + memory blocks. LLML serves an OpenAI-compatible API with configurable work models and a query-classification endpoint (`/v1/classify`).

---

## 1. Repository Layout

```
lala.ai/
├── Cargo.toml              # Workspace root: members = ["lala", "rag"]
├── ai-config.yaml          # Shared model configuration (read by LLML at startup)
├── LLML.Dockerfile         # LLML inference server Docker image (CPU; GPU-ready)
├── psql.Dockerfile         # PostgreSQL 18 + pgvector image
├── lala/                   # Rust CLI client (binary crate)
│   ├── Cargo.toml          # deps: reqwest, rustyline, serde, anyhow, rag (path)
│   └── src/
│   ├── main.rs         # Entry point — resolves API URL, DATABASE_URL, SMART_ROUTER; inits RagStore
│       ├── cli/
│       │   ├── mod.rs      # REPL loop, animated banner, command/chat dispatch
│       │   ├── chat.rs     # Chat struct — history, retrieve+inject context, spinner
│       │   ├── commands.rs # Command dispatch (/ingest, /search, /memory-search, /ingest-news)
│       │   ├── ingest.rs   # Batch + single-file + RSS ingestion
│       │   └── display.rs  # Spinner, ANSI colors, print helpers
│       └── agent/
│           ├── mod.rs
│           ├── model.rs    # ApiClient — HTTP wrapper (chat, classify); RouteDecision enum
│           └── planner.rs  # Agent — query router, reasoning→decision pipeline
├── rag/                    # Standalone RAG library crate
│   ├── Cargo.toml          # deps: postgres, pgvector, uuid (v4), anyhow, reqwest, rss, regex
│   └── src/
│       ├── lib.rs          # RagStore, Chunk, EmbeddingSearchResult, MemoryBlock — public API
│       ├── store.rs        # PostgreSQL FTS + pgvector implementation
│       ├── migrate.rs      # run_migrations() — idempotent SQL file runner
│       ├── chunker.rs      # chunk(text, chunk_size, overlap) → Vec<String>
│       └── news.rs         # ingest_news_feed(store, rss_url, delay_ms) RSS ingestion
├── migrations/             # SQL files applied in lex order on RagStore::open()
│   ├── 001_initial_schema.sql  # documents, chunks (tsvector GIN), memory_blocks
│   └── 002_pgvector.sql        # vector extension + chunk_embeddings (vector(384) IVFFlat)
├── LLML/                   # Python inference server (FastAPI + llama-cpp-python)
│   ├── main.py             # Entry point — loads config, starts uvicorn on :3000
│   ├── config.py           # Deserializes ai-config.yaml → ModelParams
│   ├── requirements.txt
│   ├── model/
│   │   ├── runner.py       # ModelRunner — generate() + stream() via asyncio.to_thread
│   │   └── registry.py     # ModelRegistry: model name (str) → ModelRunner
│   └── api/
│       ├── routes.py       # Router: /v1/chat/completions, /v1/models, /v1/classify
│       └── classifier.py   # Shared heuristic + CLASSIFIER_SYSTEM prompt constant
└── telegram/               # Telegram bot client
    ├── app.py              # Entry point — wires handlers, starts long-polling
    ├── config.py           # Config from environment variables (incl. SMART_ROUTER)
    ├── requirements.txt
    ├── agent/
│   ├── client.py       # LLMLClient — reason(), decide(), classify()
│   └── conversation.py # Per-user rolling conversation history (thread-safe)
    └── bot/
        ├── handlers.py     # Pipeline: classify → direct | reason→decide; spoiler formatting
        └── middleware.py   # Auth guard
```

---

## 2. High-Level System Diagram

```mermaid
graph TD
    User["👤 User (terminal)"]
    TGUser["👤 User (Telegram)"]
    Lala["lala\nRust CLI"]
    TGBot["telegram/\nPython bot"]
    LLML["LLML\nPython/FastAPI\n:3000"]
    Config["ai-config.yaml\n(default_work_model + work_models)"]
    GGUF["*.gguf model files\n(local filesystem)"]
    DB[("PostgreSQL\n+ pgvector\n:5432")]
    Registry["ModelRegistry\nmodel name → ModelRunner"]
    Classifier["classifier.py\nheuristic + LLM"]

    User -->|"stdin / rustyline"| Lala
    Lala -->|"POST /v1/classify"| LLML
    Lala -->|"POST /v1/chat/completions {model:'reasoning'}"| LLML
    Lala -->|"POST /v1/chat/completions {model:'decision'}"| LLML
    LLML -->|"JSON responses"| Lala
    Lala -->|"print reply"| User

    TGUser -->|"Telegram message"| TGBot
    TGBot -->|"POST /v1/classify"| LLML
    TGBot -->|"POST /v1/chat/completions"| LLML
    LLML -->|"JSON responses"| TGBot
    TGBot -->|"reply (spoiler + answer)"| TGUser

    Config -->|"read on startup"| LLML
    GGUF -->|"mmap via llama-cpp-python"| Registry
    LLML --> Registry
    LLML --> Classifier

    RAGCrate["rag/\nRust library crate\nPostgreSQL FTS + pgvector"] -->|"use rag::RagStore"| Lala
    RAGCrate -->|"TCP postgres"| DB
    Lala -->|"retrieve → inject context"| RAGCrate

    DB[("PostgreSQL + pgvector\n:5432\ndocker-compose db service")]
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

`main()` also reads `LALA_SMART_ROUTER` (enabled by default; set to "0" to disable LLM-based classification) and `DATABASE_URL` (libpq connection string; default matches the `docker-compose.yml` `db` service). When `lala serve` is used, service URLs are persisted to a temp file `lala-serve-env.json` and may be consumed by subsequent CLI runs. `RagStore::open()` is called before `cli::run()` and runs any pending migrations automatically.

---

### 3.2 `LLML` — inference server

**Location:** `LLML/`  
**Entry:** `LLML/main.py`

```
python main.py [--config PATH] [--port PORT]
# reads ../ai-config.yaml by default, binds 0.0.0.0:3000
```

Startup sequence (see §5 for detail):
1. Parse CLI args (`--config`, `--port`)
2. `load_config("../ai-config.yaml")`
3. For each model in config → `ModelRunner(path, params)` → register in `ModelRegistry`
4. Build FastAPI app with registry in `app.state`, mount API router
5. `uvicorn.run(app, host="0.0.0.0", port=3000)`

---

## 4. lala — CLI Client Flow

```mermaid
sequenceDiagram
    participant User
    participant rustyline
    participant cli as cli::run()
    participant spinner as spinner thread
    participant Agent as agent::planner::Agent
    participant ApiClient

    User->>rustyline: type input
    rustyline-->>cli: line string

    alt /clear
        cli->>cli: truncate history to system prompt
    else /exit or Ctrl-C
        cli->>cli: break loop
    else normal message
        cli->>cli: push ChatMessage{role:"user"} to history

        alt LALA_SMART_ROUTER=1
            cli->>Agent: classify_query(input, history)
            Agent->>ApiClient: classify(query, context=last_2_turns)
            ApiClient->>LLML: POST /v1/classify
            LLML-->>ApiClient: {route, confidence}
            ApiClient-->>Agent: RouteDecision
            Agent-->>cli: RouteDecision
        else heuristic (default)
            cli->>cli: needs_reasoning(input) → RouteDecision
        end

        alt RouteDecision::Direct
            cli->>spinner: spawn
            cli->>Agent: run_direct(&history)
            Agent->>ApiClient: decide(decision_messages)
            ApiClient->>LLML: POST /v1/chat/completions {model:"decision"}
            LLML-->>ApiClient: answer
            Agent-->>cli: answer
            cli->>spinner: stop
            cli->>User: print_section("Answer")
        else RouteDecision::Reasoning
            cli->>spinner: spawn
            cli->>Agent: run_reasoning(&history)
            Agent->>ApiClient: reason(reasoning_history)
            ApiClient->>LLML: POST /v1/chat/completions {model:"reasoning"}
            LLML-->>ApiClient: analysis
            Agent-->>cli: analysis
            cli->>User: print_section("Reasoning")

            cli->>Agent: run_decision(&history, analysis)
            Agent->>ApiClient: decide(decision_messages)
            ApiClient->>LLML: POST /v1/chat/completions {model:"decision"}
            LLML-->>ApiClient: final answer
            Agent-->>cli: answer
            cli->>spinner: stop
            cli->>User: print_section("Answer")
        end

        cli->>cli: push ChatMessage{role:"assistant"} to history
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

### ApiClient — model role selection and classification

`ApiClient` in `lala/src/agent/model.rs` exposes four call paths:

| Method | Endpoint | Notes |
|--------|----------|-------|
| `chat(&msgs, max_tokens, None)` | `POST /v1/chat/completions` | Server picks first registered model |
| `reason(&msgs, max_tokens)` | `POST /v1/chat/completions` | `model: "reasoning"`, temp 0.7 |
| `decide(&msgs, max_tokens)` | `POST /v1/chat/completions` | `model: "decision"`, temp 0.3 |
| `classify(query, context)` | `POST /v1/classify` | Returns `RouteDecision::{Direct,Reasoning}` |

`planner.rs` exposes `Agent::classify_query(input, history)` which wraps `client.classify()` with a local `needs_reasoning()` heuristic fallback. `cli.rs` calls the method when `smart_router=true`, otherwise resolves directly via `needs_reasoning()`.

---

## 5. LLML — Inference Server Flow

### 5.1 Multi-Model Configuration (`ai-config.yaml`)

LLML loads **all** work models declared in `ai-config.yaml` at startup and stores them in a `ModelRegistry` keyed by model name. Two default work models are configured by example:

| Model | Purpose | Temperature | `n_ctx` | Notes |
|------|---------|-------------|---------|-------|
| `reasoning` | Deep analysis, multi-step thinking | 0.7 | 2048 | Uses a larger context window |
| `decision` | Short, deterministic output | 0.3 | 512 | Used to produce the final answer |

The `name` field in each work model entry is the API-facing key. Both models currently use the same GGUF file by default — swap `model_path` independently to use different checkpoints.

### 5.2 Startup

```mermaid
flowchart TD
    A[main.py: parse --config, --port] --> C[load_config ai-config.yaml]
    C --> D{models empty?}
    D -- yes --> E[exit with error]
    D -- no --> F[for each model entry]
    F --> G[params_from_config\nreads temperature, max_tokens,\nn_gpu_layers, n_threads, n_ctx, n_batch]
    G --> H[ModelRunner GGUF file]
    H --> I["registry.register(role, runner)"]
    I --> F
    F --> J[app.state.registry = registry]
    J --> K[mount api.routes router]
    K --> L[uvicorn.run app 0.0.0.0:3000]
```

### 5.3 Request: POST /v1/chat/completions

```mermaid
sequenceDiagram
    participant lala as lala ApiClient
    participant fastapi as FastAPI handler
    participant registry as ModelRegistry
    participant thread as asyncio.to_thread
    participant runner as ModelRunner

    lala->>fastapi: POST /v1/chat/completions\n{model?, messages, max_tokens?, temperature?}

    fastapi->>fastapi: validate messages not empty
    fastapi->>registry: resolve model name\n(req.model → exact lookup, or first())
    registry-->>fastapi: ModelRunner  OR  400 unknown model / 500 empty registry

    fastapi->>fastapi: slide_messages(messages, n_ctx budget)
    fastapi->>fastapi: build_prompt(messages)\nMistral [INST]...[/INST] format

    fastapi->>thread: asyncio.to_thread\n(runner.generate, prompt, max_tokens, temperature)
    thread->>runner: _model(prompt, max_tokens, temperature, stop=["[/INST]"])
    runner-->>thread: output_string
    thread-->>fastapi: output_string

    fastapi-->>lala: ChatResponse JSON\n{choices:[{message:{content}}], usage, ...}
```

### 5.3 Request: GET /v1/models

Returns all registered work model names (e.g. `"reasoning"`, `"decision"`) in OpenAI list format. No inference involved.

---

## 6. Agent — Two-Step Planner Loop

`lala/src/agent/planner.rs` implements the `Agent` struct that drives every user turn.

### How it works

```
user query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1 — Reason  (model: "reasoning", temp: 0.7)       │
│  Input:  full conversation history                      │
│          + REASONING_SYSTEM prompt                      │
│  Output: internal analysis string  (never shown)        │
└────────────────────────────┬────────────────────────────┘
                             │ analysis
                             ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2 — Decide  (model: "decision", temp: 0.3)        │
│  Input:  DECISION_SYSTEM prompt                         │
│          + analysis appended as hidden [system] message │
│          + last user message                            │
│  Output: final answer shown to user                     │
└─────────────────────────────────────────────────────────┘
```

### Why two steps?

| Concern | Reasoning model | Decision model |
|---------|----------------|----------------|
| Temperature | 0.7 — creative, exploratory | 0.3 — deterministic, concise |
| Context window | 2048 — can hold full history | 512 — tight, focused on final answer |
| Job | Think through the problem | Turn analysis into a clean reply |
| Visible to user | No | Yes |

### Message arrays

**Step 1 input** — full history with the REPL system prompt swapped for `REASONING_SYSTEM`:
```
[{role:"system", content:REASONING_SYSTEM}, {role:"user",...}, {role:"assistant",...}, ...]
```

**Step 2 input** — condensed array (keeps the decision model within its 512-token window):
```
[{role:"system", content:DECISION_SYSTEM},
 {role:"system", content:"[Internal analysis — do not quote this]\n{analysis}"},
 {role:"user",   content:"{last_user_message}"}]
```

### `Agent` API

```rust
pub struct Agent<'a> { client: &'a ApiClient }
impl<'a> Agent<'a> {
    pub fn new(client: &'a ApiClient) -> Self
    pub fn classify_query(&self, input: &str, history: &[ChatMessage]) -> RouteDecision
    pub fn run_direct(&self, history: &[ChatMessage]) -> anyhow::Result<String>
    pub fn run_reasoning(&self, history: &[ChatMessage]) -> anyhow::Result<String>
    pub fn run_decision(&self, history: &[ChatMessage], analysis: &str) -> anyhow::Result<String>
    fn replace_system(history: &[ChatMessage], new_system: &str) -> Vec<ChatMessage>
}
```

`classify_query` calls `client.classify()` and falls back to `needs_reasoning()` on error. `cli.rs` branches on the returned `RouteDecision` to pick the direct or reasoning path.

The reasoning output (`analysis`) is displayed to the user in the CLI under a `▷ Reasoning` section with yellow ANSI colouring. In the Telegram bot it is wrapped in a `<tg-spoiler>` so users can tap to reveal it.

---

## 7. Configuration — `ai-config.yaml`

Parsed by `LLML/config.py` (`load_config()`) into an `AiConfig` dataclass on startup. Each configured work model defines a `name`, GGUF path, and inference parameters.

```mermaid
classDiagram
    class ModelParams {
        +float temperature
        +int max_tokens
        +int n_gpu_layers
        +int n_threads
        +int n_threads_batch
        +int n_ctx
        +int n_batch
        +bool use_mlock
        +bool embedding
    }
    class ModelConfig {
        +str name
        +str model_path
        +ModelParams params
    }
    class AiConfig {
        +int version
        +str default_work_model
        +list~ModelConfig~ work_models
        +ModelConfig? embedding_model
    }
    AiConfig --> ModelConfig
    ModelConfig --> ModelParams
```

**Registered models (current `ai-config.yaml`):**

| Role | Model name | Temperature | max_tokens | n_ctx |
|------|-----------|-------------|------------|-------|
| `reasoning` | mistral-reasoning | 0.7 | 512 | 2048 |
| `decision` | mistral-decision | 0.3 | 256 | 512 |

Both point to the same GGUF file (`mistral-7b-v0.1.Q4_K_M.gguf`). Different `ModelRunner` instances are loaded with different params.

---

## 8. Model Layer Internals

```mermaid
classDiagram
    class ModelRegistry {
        -dict~str, ModelRunner~ _models
        +register(role, runner)
        +get(role) ModelRunner
        +roles() list~str~
        +first() tuple~str, ModelRunner~
    }
    class ModelRunner {
        -Llama _llm
        -ModelParams _params
        +generate(prompt, max_tokens, temperature) str
        +stream(prompt, max_tokens, temperature) Iterator
    }
    ModelRegistry "1" --> "*" ModelRunner : contains
    ModelRunner --> ModelParams : configured by
```

**Thread safety:** `ModelRunner` wraps `llama-cpp-python`'s `Llama` object. Each HTTP request runs inference via `asyncio.to_thread()` so the async event loop is never blocked. There is no shared mutable session state across concurrent requests.

---

## 9. Prompt Format

`build_prompt()` in `LLML/api/routes.py` converts the OpenAI `messages` array into the Mistral/Llama instruction format:

```
<s>[INST] {system_prompt}

{first_user_message} [/INST] {assistant_reply} </s>[INST] {next_user} [/INST]...
```

The output stream is cut early when a `[/INST]` marker appears in generated tokens, preventing prompt leakage.

---

## 10. HTTP API Reference

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

### POST `/v1/classify`

Classifies a query as requiring reasoning or a direct answer. Used by `lala` CLI (when `LALA_SMART_ROUTER=1`) and by the Telegram bot (when `SMART_ROUTER=1`).

**Request:**
```json
{
  "query": "what's the weather like today?",
  "context": [
    { "role": "user",      "content": "hi" },
    { "role": "assistant", "content": "Hello! How can I help?" }
  ],
  "model": "reasoning"
}
```

| Field | Required | Notes |
|-------|----------|-------|
| `query` | yes | The raw user message to classify |
| `context` | no | Last 1–2 conversation turns for context |
| `model` | no | Which model to use for LLM classification; defaults to `"reasoning"` |

**Response:**
```json
{ "route": "direct", "confidence": "heuristic" }
```

| Field | Values | Notes |
|-------|--------|-------|
| `route` | `"direct"` \| `"reasoning"` | Destination path |
| `confidence` | `"heuristic"` \| `"llm"` | Whether LLM or fast-path heuristic decided |

The heuristic fast-path fires first (social/greeting patterns → `"direct"` immediately, no LLM call). On error, the endpoint returns 200 with a heuristic fallback — never 5xx.

#### curl example

```sh
curl -s http://localhost:3000/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"query": "explain transformers in ML"}' | jq .
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

## 11. ApiClient — `lala/src/agent/model.rs`

The `ApiClient` struct is the sole boundary between `lala` and `LLML`. It uses `reqwest::blocking::Client` (no timeout — CPU inference can be slow).

| Method | Endpoint | Description |
|--------|----------|-------------|
| `chat(messages, max_tokens, model_role)` | `/v1/chat/completions` | Core — sends full history, returns reply string |
| `reason(messages, max_tokens)` | `/v1/chat/completions` | Shortcut — selects `ModelRole::Reasoning` |
| `decide(messages, max_tokens)` | `/v1/chat/completions` | Shortcut — selects `ModelRole::Decision` |
| `classify(query, context)` | `/v1/classify` | Returns `RouteDecision::{Direct,Reasoning}` |

`RouteDecision::from_str()` maps the `"route"` string from the server response to the enum, defaulting to `Reasoning` on any unrecognised value (fail-closed).

---

## 12. Infrastructure

### PostgreSQL + pgvector (Active)

**RAG Storage Engine:** PostgreSQL FTS for keyword retrieval and pgvector for dense embeddings.
- Accessed via `postgres` + `pgvector` in the `rag` crate
- Connection string: `DATABASE_URL` (default `postgres://postgres:mysecretpassword@localhost:5432/vector_db`)
- `docker-compose.yml` provides a `db` service on port `5432`
- Schema: `documents`, `chunks` (`tsvector` GIN index), `memory_blocks`, `chunk_embeddings`
- Migrations are applied automatically by `run_migrations()` from `migrations/*.sql`

See [RAG.md](RAG.md) for the full RAG implementation details.

### Phase 1 semantic search (Planned)

Phase 1 may add a semantic search path over pgvector embeddings. The public `RagStore` API will remain unchanged; only the backend implementation and vector store wiring change.

**Planned changes:**
- Add an embedding endpoint in LLML or a local embedding model
- Store chunk vectors in `chunk_embeddings` via `RagStore::store_embedding()`
- Use `retrieve_by_embedding()` for semantic recall and reranking
- Keep `retrieve()` and `retrieve_memory_blocks()` as the keyword fallback

**Docker setup (future):**
```sh
# LLML inference server
docker build -f LLML.Dockerfile -t lala-llml .

docker run -p 3000:3000 \
  -v /path/to/models:/models \
  -v ./ai-config.yaml:/app/ai-config.yaml \
  lala-llml
```

`psql.Dockerfile` remains available for future PostgreSQL-based persistence work.

---

## 13. Current vs. Target Architecture

```mermaid
flowchart TD
    subgraph current ["Current (implemented)"]
        U1[User] --> CLI1[lala CLI]
        CLI1 -->|HTTP| LLML1[LLML server]
        LLML1 --> LM1[llama_cpp model]
    end

    subgraph target ["Target — implemented architecture"]
        U2[User] --> IF[CLI\ncli.rs]
        IF --> AG[Agent Layer\nPlanner / Reasoner]
        AG --> RAG["rag/ crate\nRagStore: retrieve / store / chunk"]
        AG --> LLM[Model Layer\nApiClient → LLML]
        RAG --> DB2[(PostgreSQL + pgvector)]
    end
```

**Phase 0 completion status:**

| Concern | Phase 0 Status |
|---------|---------------|
| REPL / input | `cli/` directory with submodules for chat, commands, ingest, display ✅ |
| RAG Layer | Standalone `rag/` crate with PostgreSQL FTS + pgvector + memory blocks, live context injection ✅ |
| Multi-model routing (server) | `ModelRegistry` + role-based routing ✅ |
| Per-request temperature (server) | Wired through request/response ✅ |
| Two-step agent loop | `Agent::run_reasoning()` + `run_decision()` with context injection ✅ |
| Query routing | `POST /v1/classify` + `RouteDecision` + `LALA_SMART_ROUTER` ✅ |
| Telegram bot | classify → direct | reason→decide + spoiler formatting + context injection ✅ |
| Document ingestion | `/ingest`, `/ingest-file`, `/ingest-news` CLI commands ✅ |
| Keyword retrieval | `/search`, `/memory-search` CLI commands (BM25 via PostgreSQL FTS) ✅ |
| Context injection | Auto-retrieve top-5 chunks + memory blocks, inject into system prompt on every query ✅ |
| Planned: `/v1/embed` | Not yet — for Phase 1 Qdrant integration |
| Planned: pgvector | Deferred — provisioned but not used in Phase 0 |

---

## 14. Key Dependencies

### lala
| Crate | Purpose |
|-------|---------|
| `rustyline` | Readline-style REPL with history navigation |
| `reqwest` (blocking + json) | HTTP client for LLML API |
| `serde` / `serde_json` | JSON serialization of ChatMessage arrays |
| `anyhow` | Error propagation |
| `rag` (path dep) | Standalone RAG crate — PostgreSQL FTS + pgvector store, retrieve, embed |

### rag (Phase 0)
| Crate | Purpose |
|-------|---------|
| `postgres` | Blocking PostgreSQL client (sync, no async runtime required) |
| `pgvector` (postgres feature) | `Vector` type for `vector(384)` column, cosine `<=>` operator |
| `uuid` | Document/chunk ID generation |
| `anyhow` | Error propagation |

### LLML (Python)
| Package | Purpose |
|---------|--------|
| `llama-cpp-python` | Local GGUF model loading and token generation |
| `fastapi` | Async HTTP server and router |
| `uvicorn` | ASGI server |
| `pydantic` | Request/response validation and serialization |
| `pyyaml` | `ai-config.yaml` parsing |
| `uuid` | Response IDs |

### Telegram bot (Python)
| Package | Purpose |
|---------|--------|
| `python-telegram-bot` | Telegram API client + async long-polling |
| `requests` | Blocking HTTP client for LLML API |
| `python-dotenv` | Load `.env` files into environment |
