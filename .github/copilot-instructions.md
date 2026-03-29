# lala.ai — Agentic RAG System

> Full architecture reference: [doc/architecture.md](../doc/architecture.md)

Rust-based local **Agentic RAG** system. Two binaries communicate over HTTP:

- **`lala`** — interactive CLI client (terminal REPL, conversation history, spinner)
- **`LLML`** — local LLM inference server (loads GGUF models, serves OpenAI-compatible API)

PostgreSQL + pgvector is provisioned for RAG storage but is **not yet wired into the live request loop** (Phase 0 target).

---

## System Map

```
User
 │ stdin (rustyline)
 ▼
lala/src/main.rs          resolves LLML_API_URL, calls cli::run()
lala/src/cli.rs           REPL loop, conversation Vec<ChatMessage>, spinner thread
lala/src/agent/model.rs   ApiClient — reqwest::blocking POST /v1/chat/completions
 │ HTTP JSON
 ▼
LLML/src/main.rs          reads ai-config.yaml, loads models, starts Axum on :3000
LLML/src/loalYaml/        deserializes ai-config.yaml → AiConfig / Model / Parameter
LLML/src/model/registry.rs  ModelRegistry: role (String) → ModelRunner
LLML/src/model/model.rs   ModelRunner::load(gguf_path) + generate_from_prompt()
LLML/src/api/mod.rs       Router: POST /v1/chat/completions, GET /v1/models
                          build_prompt() → Mistral [INST]...[/INST] format
 │ llama_cpp C FFI
 ▼
*.gguf model file (local filesystem, path from ai-config.yaml)
```

---

## Build & Run

```sh
# Start the inference server (reads ../ai-config.yaml)
cd LLML && cargo run

# Start the CLI client (connects to http://localhost:3000 by default)
cd lala && cargo run
# or with a custom server URL:
cd lala && cargo run -- http://192.168.1.10:3000
# or via env:
LLML_API_URL=http://192.168.1.10:3000 cargo run

# Database (PostgreSQL 18 + pgvector)
docker build -f psql.Dockerfile -t lala-postgres .
docker run -e POSTGRES_PASSWORD=postgres -p 5432:5432 lala-postgres
DATABASE_URL=postgres://postgres:postgres@localhost:5432/lala
```

---

## Two-Binary Architecture

### lala (CLI client) — `lala/`

| File | Role |
|------|------|
| `src/main.rs` | Entry — resolves API URL (arg → `LLML_API_URL` env → `http://localhost:3000`), calls `cli::run()` |
| `src/cli.rs` | REPL: `rustyline` input, `Vec<ChatMessage>` history (system prompt at index 0), braille spinner on background thread, `/clear` and `/exit` commands |
| `src/agent/model.rs` | `ApiClient` wrapping `reqwest::blocking::Client`; `ChatMessage`, `ModelRole` enum (`Reasoning`/`Decision`); methods: `chat()`, `reason()`, `decide()` |

Conversation history format sent on every request:
```
[{role:"system", content:SYSTEM_PROMPT}, {role:"user",...}, {role:"assistant",...}, ...]
```

### LLML (inference server) — `LLML/`

| File | Role |
|------|------|
| `src/main.rs` | Startup: init tracing, `load_config()`, loop models → `ModelRunner::load()` → `registry.register(role, runner)`, `Arc<ModelRegistry>`, Axum bind `:3000` |
| `src/loalYaml/loadYaml.rs` | `AiConfig` / `Model` / `Parameter` structs; `load_config(path)` — reads + deserializes YAML |
| `src/model/model.rs` | `ModelRunner`: `load(path, params)` once at startup; `generate_from_prompt(prompt, max_tokens)` — creates fresh `LlamaSession` per call (no context bleed) |
| `src/model/registry.rs` | `ModelRegistry`: `HashMap<String, ModelRunner>`; `register(role, runner)`, `get(role)`, `first()` |
| `src/api/mod.rs` | `create_router()` → `POST /v1/chat/completions` + `GET /v1/models`; `build_prompt()` → Mistral format; inference runs in `tokio::task::spawn_blocking` |

---

## API (LLML server)

```
POST /v1/chat/completions
{
  "model": "reasoning" | "decision",   // optional — defaults to first registered
  "messages": [{role, content}, ...],
  "max_tokens": 200                     // optional — overrides config default
}
→ { choices: [{ message: { content: "..." } }], usage, ... }

GET /v1/models
→ { object: "list", data: [{ id: "reasoning" }, { id: "decision" }] }
```

---

## Configuration — `ai-config.yaml`

Read by **LLML only** at startup. Defines model roles, GGUF paths and inference parameters.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `role` | (model name) | Key used by `lala` in `"model"` field: `"reasoning"` or `"decision"` |
| `temperature` | 0.7 / 0.3 | Sampling temperature |
| `max_tokens` | 512 / 256 | Per-request token limit (overridable in API request) |
| `n_gpu_layers` | 0 | `0` = CPU-only; `99` = all layers to GPU (needs CUDA/Metal build) |
| `n_threads` | 4 | Physical core count; `0` = auto-detect |
| `n_ctx` | 2048 / 512 | Context window in tokens |
| `n_batch` | 512 | Prompt evaluation batch size |
| `modelPath` | (absolute path) | Path to `.gguf` file — currently both roles share the same file |

---

## Prompt Format

`build_prompt()` in `LLML/src/api/mod.rs` produces Mistral/Llama instruction format:

```
<s>[INST] {system_prompt}\n\n{first_user_msg} [/INST] {assistant_reply} </s>[INST] {next_user} [/INST]...
```

Generation stops early if `[/INST]` appears in output tokens (prevents prompt leakage).

---

## Key Conventions

- **Error handling**: propagate with `anyhow::Result`; no `.unwrap()` in new code.
- **Thread safety**: `ModelRunner` is `Send + Sync` (llama_cpp C++ objects are thread-safe). Each request spawns a fresh `LlamaSession` — no shared mutable session state.
- **Blocking inference**: always run `ModelRunner::generate_from_prompt` inside `tokio::task::spawn_blocking` — never block the async executor directly.
- **Embeddings** (planned): `Vec<f32>`, pgvector columns, model `"bge-small"`, cosine distance `<=>` operator.
- **Async DB**: all `sqlx` calls are `async` — wire through `tokio` runtime; do not call from sync context without `spawn_blocking`.
- **Config is LLML's concern**: `lala` never reads `ai-config.yaml`; it selects models by role string via the API.
- **Role strings**: `"reasoning"` and `"decision"` — must match keys registered in `ModelRegistry`; defined under `role:` in `ai-config.yaml`.

---

## Phase Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | In progress | Layered architecture: Interface → Agent → RAG → Model + DB layers |
| 1 | Planned | Query rewriting, multi-step planning, session history, streaming |
| 2 | Planned | Reranking, hybrid search, grounding/citation validation |
| 3 | Planned | HTTP/gRPC interface, metadata filtering |

Target module layout (Phase 0) — see [doc/phase0.md](../doc/phase0.md):
```
lala/src/
  interface/cli.rs       # readline → agent::run()
  agent/                 # Planner, Reasoner, Executor
  rag/                   # retrieve(), store(), embed(), chunk()
  model/wrapper.rs       # generate(prompt) → String
  db/connection.rs       # PgPool (unchanged)
```

---

## Infrastructure

- **PostgreSQL 18 + pgvector**: `psql.Dockerfile` — build and run with `docker run -e POSTGRES_PASSWORD=postgres -p 5432:5432 lala-postgres`
- **`init.sql`**: place at repo root; auto-executed by Docker on first start (file not yet created)
- **Planned DB tables**: `sessions`, `messages`, `documents`, `document_chunks`, `queries`, `retrieval_results`, `answers`, `answer_citations` — see [doc/future/design.md](../doc/future/design.md)

---

## Dependencies

### lala
| Crate | Purpose |
|-------|---------|
| `rustyline` | Readline REPL with history navigation |
| `reqwest` (blocking + json) | HTTP client for LLML API |
| `serde` / `serde_json` | ChatMessage serialization |
| `anyhow` | Error propagation |
| `sqlx` | PostgreSQL client (declared, not active in live loop yet) |

### LLML
| Crate | Purpose |
|-------|---------|
| `llama_cpp` | GGUF model loading + token generation (requires C++ toolchain) |
| `axum` | Async HTTP server |
| `tokio` | Async runtime |
| `serde` / `serde_yaml` / `serde_json` | YAML config + JSON API |
| `tracing` / `tracing-subscriber` | Structured logging (`RUST_LOG` env) |
| `anyhow` | Error propagation |
| `uuid` | Response IDs |
