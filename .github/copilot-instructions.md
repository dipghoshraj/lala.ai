# lala.ai — Agentic RAG System

> Full architecture reference: [doc/architecture.md](../doc/architecture.md)

Rust-based local **Agentic RAG** system. Three components communicate over HTTP:

- **`lala`** — interactive CLI client (Rust: terminal REPL, conversation history, spinner)
- **`LLML`** — local LLM inference server (Python/FastAPI: loads GGUF models, serves OpenAI-compatible API)
- **`telegram`** — Telegram bot client (Python: classify → route → spoiler-formatted reply)
- **`rag`** — standalone RAG library crate (Rust: PostgreSQL FTS + pgvector, keyword + vector retrieval)

The project uses a **Cargo workspace** (`lala.ai/Cargo.toml`) with members `lala` and `rag`. PostgreSQL + pgvector is the RAG storage engine: `chunks` table with `tsvector` GIN index for keyword retrieval, `chunk_embeddings` table with `vector(384)` IVFFlat index for semantic search.

---

## System Map

```
User
 │ stdin (rustyline)
 ▼
lala/src/main.rs          resolves LLML_API_URL + LALA_SMART_ROUTER, calls cli::run()
lala/src/cli.rs           REPL loop, conversation Vec<ChatMessage>, spinner thread
lala/src/agent/model.rs   ApiClient — reqwest::blocking POST /v1/chat/completions, /v1/classify
lala/src/agent/planner.rs Agent — query router, reasoning→decision pipeline
 │ HTTP JSON
 ▼
LLML/main.py              reads ai-config.yaml, loads models, starts uvicorn on :3000
LLML/config.py            deserializes ai-config.yaml → AiConfig / Model / ModelParams
LLML/model/registry.py    ModelRegistry: role (str) → ModelRunner
LLML/model/runner.py      ModelRunner: generate() + stream() via asyncio.to_thread()
LLML/api/routes.py        Router: POST /v1/chat/completions, GET /v1/models, POST /v1/classify
                          build_prompt() → Mistral [INST]...[/INST] format
                          slide_messages() → context window management
LLML/api/classifier.py    Heuristic + LLM-based query classifier
 │ llama-cpp-python (C FFI)
 ▼
*.gguf model file (local filesystem, path from ai-config.yaml)

rag/src/lib.rs            RagStore — PostgreSQL FTS store()/retrieve() + pgvector store_embedding()/retrieve_by_embedding()
rag/src/chunker.rs        chunk(text, size, overlap) → Vec<String>
rag/src/migrate.rs        run_migrations(client, dir) — applies migrations/*.sql in order
migrations/               SQL migration files (001_initial_schema.sql, 002_pgvector.sql)
 │ postgres (TCP)
 ▼
PostgreSQL + pgvector (docker-compose db service, :5432)
```
```

---

## Build & Run

```sh
# Start the inference server (reads /ai-config.yaml)
cd LLML && pip install -r requirements.txt && python main.py

# Start the CLI client (connects to http://localhost:3000 by default)
cd lala && cargo run
# or with a custom server URL:
cd lala && cargo run -- http://192.168.1.10:3000
# or via env:
LLML_API_URL=http://192.168.1.10:3000 cargo run

# Enable LLM-based smart query router
LALA_SMART_ROUTER=1 cargo run

# Start PostgreSQL + pgvector (docker-compose)
docker compose up -d db
# Connect lala to PostgreSQL (default matches docker-compose)
DATABASE_URL=postgres://postgres:mysecretpassword@localhost:5432/vector_db cargo run
# Override migrations directory (default: ./migrations)
LALA_MIGRATIONS_DIR=./migrations cargo run
```

---

## Two-Binary Architecture

### lala (CLI client) — `lala/`

| File | Role |
|------|------|
| `src/main.rs` | Entry — resolves API URL (arg → `LLML_API_URL` env → `http://localhost:3000`) + `LALA_SMART_ROUTER` flag + `DATABASE_URL`, calls `RagStore::open()` then `cli::run()` |
| `src/cli.rs` | REPL: `rustyline` input, `Vec<ChatMessage>` history (system prompt at index 0), braille spinner on background thread, `/clear` and `/exit` commands |
| `src/agent/model.rs` | `ApiClient` wrapping `reqwest::blocking::Client`; `ChatMessage`, `ModelRole` enum (`Reasoning`/`Decision`), `RouteDecision` enum; methods: `chat()`, `reason()`, `decide()`, `classify()` |
| `src/agent/planner.rs` | `Agent` — `classify_query()`, `run_direct()`, `run_reasoning()`, `run_decision()`, local `needs_reasoning()` heuristic |

Conversation history format sent on every request:
```
[{role:"system", content:SYSTEM_PROMPT}, {role:"user",...}, {role:"assistant",...}, ...]
```

### LLML (inference server) — `LLML/`

| File | Role |
|------|------|
| `main.py` | Startup: parse args, `load_config()`, loop models → `ModelRunner()` → `registry.register(role, runner)`, mount FastAPI router, `uvicorn.run()` on `:3000` |
| `config.py` | `AiConfig` / `Model` / `ModelParams` dataclasses; `load_config(path)` — reads + deserializes YAML |
| `model/runner.py` | `ModelRunner`: wraps `llama_cpp.Llama`; `generate(prompt, max_tokens, temperature)` via `asyncio.to_thread()`; `stream()` for SSE |
| `model/registry.py` | `ModelRegistry`: `dict[str, ModelRunner]`; `register(role, runner)`, `get(role)`, `roles()`, `first()` |
| `api/routes.py` | Router: `POST /v1/chat/completions` + `GET /v1/models` + `POST /v1/classify`; `build_prompt()` → Mistral format; `slide_messages()` for context window management |
| `api/classifier.py` | Heuristic fast-path (greeting/keyword patterns) + LLM classifier system prompt |

---

## API (LLML server)

```
POST /v1/chat/completions
{
  "model": "reasoning" | "decision",   // optional — defaults to first registered
  "messages": [{role, content}, ...],
  "max_tokens": 200,                   // optional — overrides config default
  "temperature": 0.7,                  // optional — overrides config default
  "stream": false                      // optional — true for SSE streaming
}
→ { choices: [{ message: { content: "..." } }], usage, ... }

POST /v1/classify
{
  "query": "explain transformers",
  "context": [{role, content}, ...]     // optional — last 1–2 turns
}
→ { route: "direct" | "reasoning", confidence: "heuristic" | "llm" }

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

`build_prompt()` in `LLML/api/routes.py` produces Mistral/Llama instruction format:

```
<s>[INST] {system_prompt}\n\n{first_user_msg} [/INST] {assistant_reply} </s>[INST] {next_user} [/INST]...
```

Generation stops early if `[/INST]` appears in output tokens (prevents prompt leakage).

---

## Key Conventions

- **Error handling**: propagate with `anyhow::Result` in Rust; no `.unwrap()` in new code.
- **Thread safety (LLML)**: `ModelRunner` wraps `llama-cpp-python`'s `Llama` object. Each HTTP request runs inference via `asyncio.to_thread()` so the async event loop is never blocked.
- **Blocking inference**: always run model inference inside `asyncio.to_thread()` in LLML — never block the FastAPI event loop directly.
- **Embeddings**: `Vec<f32>` (384-dim, `all-MiniLM-L6-v2`), stored in `chunk_embeddings.embedding vector(384)`, cosine distance `<=>` via pgvector. `RagStore::store_embedding()` / `retrieve_by_embedding()` are the write/read APIs.
- **RAG storage**: PostgreSQL + pgvector. Keyword search uses `tsvector` GIN index + `websearch_to_tsquery` / `ts_rank_cd`. Vector search uses `ivfflat` cosine index. Schema applied by `run_migrations()` from `migrations/*.sql` on `RagStore::open()`.
- **Config is LLML's concern**: `lala` never reads `ai-config.yaml`; it selects models by role string via the API.
- **Role strings**: `"reasoning"` and `"decision"` — must match keys registered in `ModelRegistry`; defined under `role:` in `ai-config.yaml`.
- **RAG crate independence**: `rag/` is a standalone library crate with zero dependencies on `lala`, agent, CLI, or model layers. Consumers depend on it via `rag = { path = "../rag" }` and call the `RagStore` public API.
- **Cargo workspace**: The repo root `Cargo.toml` defines `members = ["lala", "rag"]`. Both crates share a workspace lockfile.

---

## Phase Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | In progress | Layered architecture: Interface → Agent → RAG → Model + DB layers |
| 1 | Planned | Query rewriting, multi-step planning, session history, streaming |
| 2 | Planned | Reranking, hybrid search, grounding/citation validation |
| 3 | Planned | HTTP/gRPC interface, metadata filtering |

Current module layout:
```
rag/                        # Standalone RAG library crate
  Cargo.toml                # deps: postgres, pgvector, uuid (v4), anyhow, reqwest, rss, regex
  src/
    lib.rs                  # RagStore, Chunk, EmbeddingSearchResult, MemoryBlock — public API
    store.rs                # PostgreSQL impl: store/retrieve (FTS) + store_embedding/retrieve_by_embedding (pgvector)
    migrate.rs              # run_migrations(client, dir) — idempotent SQL file runner
    chunker.rs              # chunk(text, size, overlap) → Vec<String>
    news.rs                 # RSS ingestion
migrations/                 # SQL files applied in lex order on startup
  001_initial_schema.sql    # documents, chunks (tsvector GIN), memory_blocks, schema_migrations
  002_pgvector.sql          # CREATE EXTENSION vector, chunk_embeddings (vector(384) IVFFlat)

lala/src/                   # CLI + Agent binary crate
  main.rs                   # Startup: resolve API URL + DATABASE_URL, init RagStore, start CLI
  cli/                      # Readline loop, /ingest-file, /search, /memory-search commands
  agent/                    # Planner, Reasoner
```

---

## Infrastructure

- **PostgreSQL + pgvector**: managed via `docker-compose.yml` (`db` service, `pgvector/pgvector:pg17`, port 5432). Start with `docker compose up -d db`.
- **Migrations**: `migrations/001_initial_schema.sql` + `migrations/002_pgvector.sql`. Applied automatically by `run_migrations()` on every `RagStore::open()`. Tracked in `schema_migrations` table.
- **ChromaDB**: `chroma` service in `docker-compose.yml`. Python vector store for LLML endpoints (`/v1/vector/*`). Embedded mode (no server) by default; switch to server mode in `ai-config.yaml`.

---

## Dependencies

### lala
| Crate | Purpose |
|-------|---------|
| `rustyline` | Readline REPL with history navigation |
| `reqwest` (blocking + json) | HTTP client for LLML API |
| `serde` / `serde_json` | ChatMessage serialization |
| `anyhow` | Error propagation |
| `rag` (path dep) | Standalone RAG crate — PostgreSQL FTS + pgvector store, retrieve, embed |

### rag
| Crate | Purpose |
|-------|---------||
| `postgres` | Blocking PostgreSQL client (sync, no async runtime required) |
| `pgvector` (postgres feature) | `Vector` type for `vector(384)` column, cosine `<=>` operator |
| `uuid` | Document/chunk ID generation |
| `anyhow` | Error propagation |

### LLML
| Package | Purpose |
|---------|---------|
| `fastapi` | Async HTTP server and router |
| `uvicorn` | ASGI server |
| `llama-cpp-python` | GGUF model loading + token generation via llama.cpp C FFI |
| `pyyaml` | YAML config parsing |
