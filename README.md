# lala.ai

A local **Agentic RAG** system built in Rust and Python. Two services communicate over HTTP to deliver a multi-step reasoning pipeline:

- **`lala`** — interactive Rust CLI (terminal REPL, conversation history, braille spinner)
- **`LLML`** — local LLM inference server (Python/FastAPI, loads GGUF models, OpenAI-compatible API)
- **`telegram/`** — Telegram bot client (Python, same inference pipeline over HTTP)

PostgreSQL + pgvector is provisioned for RAG storage and is active in the current implementation.

> Full architecture reference: [doc/architecture.md](doc/architecture.md)

---

## Quick Start

### What `lala.ai` does

`lala.ai` is an agentic RAG system that combines:
- `lala` (Rust CLI): local REPL with clever query routing (`direct` vs `reasoning`), document search, and ingestion workflow
- `LLML` (Python FastAPI): GGUF model inference (`/v1/chat/completions`, `/v1/classify`, `/v1/models`)
- `telegram` bot (optional): same pipeline exposed via Telegram

It routes questions through `decision` only for fast replies, or through `reasoning` + `decision` for long-form multi-step answers.

### lala CLI commands (enter `/help` in REPL for this list)

- `/project create --name <name> [--description <description>]` : create a new project and select it
- `/project select <name-or-id>` : select an existing project
- `/project deselect` : clear the selected project and use the LLM directly
- `/project list` : list available projects
- `/project current` : show the currently selected project
- `/ingest [dir]`       : batch ingest text/docs from directory (default `./ingest/`)
- `/ingest-file <path>` : ingest a single file
- `/ingest-news <url>`  : ingest RSS articles
- `/search <query>`     : search document memory with BM25
- `/memory-search <query>` : search structured memory blocks
- `/status`             : show store stats (`documents`, `chunks`, `ingest dir`)
- `/clear`              : reset conversation history
- `/help`               : show commands
- `/exit` or `/quit`    : exit the REPL

### 1. Start the inference server

**Option A — Docker Compose (recommended)**

A consolidated `docker-compose.yml` is included to start the inference server, PostgreSQL, and optional ChromaDB together.

```sh
docker compose up -d
```

The compose file defines these services:
- `db` — PostgreSQL with pgvector for RAG storage
- `llm-inference` — LLML inference server built from `LLML/Dockerfile.llm-inference`
- `chroma` — optional ChromaDB server for future vector store use

By default the compose file maps:
- `3000:3000` for LLML
- `5432:5432` for PostgreSQL
- `8000:8000` for ChromaDB

If your models or config live outside the repo, override the paths before `docker compose up`:

```sh
MODELS_DIR=/path/to/models CONFIG_PATH=./ai-config.yaml docker compose up -d
```

**Option B — Local Python**

```sh
cd LLML
pip install -r requirements.txt
python main.py                          # reads ../ai-config.yaml, serves :3000
python main.py --config /path/to/ai-config.yaml --port 3000
```

### 2. Start the CLI client

```sh
cd lala
cargo run                               # connects to http://localhost:3000
cargo run -- http://192.168.1.10:3000   # custom server URL
LLML_API_URL=http://192.168.1.10:3000 cargo run
```

If you are using PostgreSQL from Docker Compose, set `DATABASE_URL` if needed:

```sh
DATABASE_URL=postgres://postgres:mysecretpassword@localhost:5432/vector_db cargo run
```

### 3. (Optional) Telegram bot

```sh
cd telegram
pip install -r requirements.txt
cp .env.example .env                    # set TOKEN, USERID, LLML_API_URL
python app.py
```

### 4. (Optional) PostgreSQL + pgvector

The `db` service in `docker-compose.yml` already provisions PostgreSQL with `pgvector`.
If you want a standalone PostgreSQL container instead of Compose, you can still build from `psql.Dockerfile`:

```sh
docker build -f psql.Dockerfile -t lala-postgres .
docker run -e POSTGRES_PASSWORD=postgres -p 5432:5432 lala-postgres
```

### Running all services together

```sh
# Start everything with Docker Compose
docker compose up -d

# Follow logs for the inference service
docker compose logs -f llm-inference

# Start the CLI client locally
cd lala && cargo run
```

---

## Repository Layout

```
lala.ai/
├── ai-config.yaml          # Model configuration shared by all components
├── LLML.Dockerfile         # LLML inference server Docker image
├── psql.Dockerfile         # PostgreSQL 18 + pgvector image
├── lala/                   # Rust CLI client
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs         # Entry point — resolves API URL + smart-router flag
│       ├── cli.rs          # REPL loop, spinner, conversation history
│       └── agent/
│           ├── mod.rs
│           ├── model.rs    # ApiClient — HTTP wrapper (chat, classify)
│           └── planner.rs  # Agent — query router + reasoning→decision pipeline
├── LLML/                   # Python inference server
│   ├── main.py             # Entry point — loads config, starts uvicorn on :3000
│   ├── config.py           # Deserializes ai-config.yaml
│   ├── requirements.txt
│   ├── model/
│   │   ├── runner.py       # ModelRunner wrapping llama-cpp-python
│   │   └── registry.py     # ModelRegistry: role → ModelRunner
│   └── api/
│       ├── routes.py       # FastAPI router: /v1/chat/completions, /v1/models, /v1/classify
│       └── classifier.py   # Shared heuristic + LLM classifier logic
└── telegram/               # Telegram bot
    ├── app.py              # Entry point — wires handlers and starts long-polling
    ├── config.py           # Config from environment variables
    ├── requirements.txt
    ├── agent/
    │   ├── client.py       # LLMLClient — HTTP wrapper (reason, decide, classify)
    │   └── conversation.py # Per-user rolling conversation history
    └── bot/
        ├── handlers.py     # Message pipeline: classify → direct or reason→decide
        └── middleware.py   # Auth guard
```

---

## System Map

```mermaid
flowchart TD
    User(["👤 User\nstdin / rustyline"])
    TGUser(["👤 User\nTelegram"])

    subgraph lala ["lala — Rust CLI"]
        CLI["cli.rs\nREPL loop · LALA_SMART_ROUTER"]
        Planner["agent/planner.rs\nclassify_query()\nrun_direct() | run_reasoning()+run_decision()"]
        ApiClient["agent/model.rs\nApiClient\nreqwest::blocking"]
    end

    subgraph TGBot ["telegram — Python bot"]
        TGClient["agent/client.py\nLLMLClient"]
    end

    subgraph LLML ["LLML — Python/FastAPI :3000"]
        Routes["api/routes.py\nPOST /v1/chat/completions\nPOST /v1/classify\nGET  /v1/models"]
        Classifier["api/classifier.py\nheuristic_route()"]
        Runner["model/runner.py\nModelRunner.generate() / .stream()\nasyncio.to_thread"]
    end

    GGUF[("*.gguf\nmodel file")]
    Config["ai-config.yaml"]

    User --> CLI
    CLI --> Planner
    Planner --> ApiClient
    ApiClient -->|"HTTP"| Routes
    Routes --> Classifier
    Routes --> Runner
    Runner -->|"llama-cpp-python FFI"| GGUF
    Config -->|"read on startup"| LLML

    TGUser --> TGClient
    TGClient -->|"HTTP"| Routes
```

---

## Query Router

Every query is classified before inference to decide whether multi-step reasoning is needed.

```mermaid
flowchart TD
    Q(["incoming query"])
    Classify["POST /v1/classify"]
    Heuristic["heuristic fast-path\ngreetings → direct\nno LLM call"]
    LLM["LLM classifier\nreasoning model"]
    Direct["run_direct()\ndecision model only"]
    Reason["run_reasoning()\nthen run_decision()"]
    Out(["reply to user"])

    Q --> Classify
    Classify --> Heuristic
    Classify --> LLM
    Heuristic -->|"direct"| Direct
    LLM -->|"direct"| Direct
    LLM -->|"reasoning"| Reason
    Direct --> Out
    Reason --> Out
```

| Route | Path | Use case |
|-------|------|----------|
| `direct` | decision model only | Greetings, simple factual questions, short conversational replies |
| `reasoning` | reasoning → decision | Analysis, code, comparisons, multi-step questions |

**lala CLI:** enable LLM classification with `LALA_SMART_ROUTER=1`. Default uses the local heuristic (no extra network call).

**Telegram bot:** enable with `SMART_ROUTER=1` in `.env`. Default routes every message through the full reasoning pipeline.

---

## API Reference

All endpoints served by LLML on port `3000`.

### `POST /v1/chat/completions`

```json
{
  "model": "reasoning",
  "messages": [{ "role": "user", "content": "Explain Rust lifetimes." }],
  "max_tokens": 512,
  "temperature": 0.7,
  "stream": false
}
```

| Field | Required | Notes |
|-------|----------|-------|
| `messages` | yes | Non-empty. First element may be `system`. |
| `model` | no | `"reasoning"` or `"decision"`. Omit → first registered model. |
| `max_tokens` | no | Overrides config default for this request. |
| `temperature` | no | Overrides model config default (0.0–2.0). |
| `stream` | no | `true` → SSE token stream. Default `false`. |

Response: OpenAI `ChatResponse` — `choices[0].message.content`.

### `POST /v1/classify`

```json
{
  "query": "explain how Rust lifetimes work",
  "context": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

Response:
```json
{ "route": "reasoning", "confidence": "llm" }
```

`confidence` is `"heuristic"` when the fast-path fired (social patterns, no LLM call), `"llm"` when the reasoning model classified the query.

### `GET /v1/models`

```json
{ "object": "list", "data": [{ "id": "reasoning" }, { "id": "decision" }] }
```

---

## Configuration — `ai-config.yaml`

Read by LLML only. Defines model roles, GGUF paths, and inference parameters.

```yaml
models:
  - name: mistral-reasoning
    role: reasoning
    modelPath: /path/to/mistral-7b-v0.1.Q4_K_M.gguf
    parameters:
      - { name: temperature,  default: 0.7  }
      - { name: max_tokens,   default: 512  }
      - { name: n_ctx,        default: 2048 }
      - { name: n_gpu_layers, default: 0    }
      - { name: n_threads,    default: 4    }
      - { name: n_batch,      default: 512  }

  - name: mistral-decision
    role: decision
    modelPath: /path/to/mistral-7b-v0.1.Q4_K_M.gguf
    parameters:
      - { name: temperature,  default: 0.3 }
      - { name: max_tokens,   default: 256 }
      - { name: n_ctx,        default: 512 }
      - { name: n_gpu_layers, default: 0   }
      - { name: n_threads,    default: 4   }
      - { name: n_batch,      default: 512 }
```

| Parameter | Notes |
|-----------|-------|
| `role` | API key used by clients (`"reasoning"` / `"decision"`) |
| `n_gpu_layers` | `0` = CPU-only; `99` = all layers to GPU (needs CUDA/Metal build) |
| `n_threads` | Set to physical core count; `0` = auto-detect |
| `modelPath` | Absolute path to `.gguf` file — both roles can share the same file |

---

## Environment Variables

### lala (Rust CLI)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLML_API_URL` | `http://localhost:3000` | LLML server URL (overridden by CLI arg) |
| `LALA_SMART_ROUTER` | `0` | Set to `1` to enable LLM-based query classification |

### LLML (Python inference server)

| Variable | Default | Description |
|----------|---------|-------------|
| `RUST_LOG` | `info` | Log level |

### Telegram bot

| Variable | Required | Description |
|----------|----------|-------------|
| `TOKEN` | yes | Telegram bot token |
| `USERID` | yes | Authorized user ID |
| `LLML_API_URL` | no | LLML server URL (default `http://localhost:3000`) |
| `SMART_ROUTER` | no | Set to `1` to enable LLM-based classification |
| `REASONING_MAX_TOKENS` | no | Default `512` |
| `DECISION_MAX_TOKENS` | no | Default `256` |
| `MAX_HISTORY_TURNS` | no | Default `10` |

---

## Phase Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | Complete | Layered architecture: CLI → Agent → RAG → LLML server; PostgreSQL + pgvector active |
| 1 | Planned | Session history persistence, query rewriting, streaming to CLI |
| 2 | Planned | Hybrid keyword + vector retrieval, embedding-based reranking |
| 3 | Planned | Learned router (embedding similarity + user feedback loop) |
| 4 | Planned | HTTP/gRPC interface, metadata filtering, citation grounding |

---

## Dependencies

### lala (Rust)

| Crate | Purpose |
|-------|---------|
| `rustyline` | Readline REPL with history |
| `reqwest` (blocking + json) | HTTP client for LLML API |
| `serde` / `serde_json` | ChatMessage serialization |
| `anyhow` | Error propagation |

### LLML (Python)

| Package | Purpose |
|---------|---------|
| `fastapi` + `uvicorn` | Async HTTP server |
| `llama-cpp-python` | GGUF model loading + inference (C++ backend) |
| `pyyaml` | Config deserialization |
| `pydantic` | Request/response validation |

### Telegram bot (Python)

| Package | Purpose |
|---------|---------|
| `python-telegram-bot` | Bot framework |
| `requests` | Blocking HTTP client for LLML |
| `python-dotenv` | `.env` loading |

---

## Key Conventions

### Error handling
- Rust: propagate with `anyhow::Result`; avoid `.unwrap()` in new code
- Python: raise meaningful exceptions with context; catch and log in API routes

### Thread safety (LLML)
- `ModelRunner` wraps `llama-cpp-python`'s `Llama` object
- **Never block the FastAPI event loop**: always run inference inside `asyncio.to_thread()`
- Each HTTP request runs on a thread pool, not blocking the async loop

### Configuration scope
- `ai-config.yaml` is LLML's concern only — `lala` CLI never reads it
- Clients select models by **role string** via the API: `"reasoning"` and `"decision"`
- Role strings must match keys registered in `ModelRegistry`

### Role strings (fixed)
- `"reasoning"` — multi-step analysis, code review, comparisons
- `"decision"` — fast replies, social/API patterns, conversational replies

### RAG
- Standalone `rag/` library crate with zero dependencies on `lala`, agent, or model layers
- Consumers depend via `rag = { path = "../rag" }` and call `RagStore` public API
- PostgreSQL FTS for BM25 keyword retrieval and pgvector for vector storage
- Embeddings are stored in `chunk_embeddings`; semantic retrieval and hybrid search are future work

### Cargo workspace
- Root `Cargo.toml` defines `members = ["lala", "rag"]`
- Both crates share a workspace lockfile
- Build both: `cargo build --workspace`

### Prompt format (Mistral)
`build_prompt()` in `LLML/api/routes.py` produces Mistral/Llama instruction format:
```
<s>[INST] {system_prompt}\n\n{first_user_msg} [/INST] {assistant_reply} </s>[INST] {next_user} [/INST]...
```
Generation stops early if `[/INST]` appears (prevents prompt leakage).

### Message sliding (context window management)
- `slide_messages()` in `LLML/api/routes.py` evicts oldest user/assistant pairs if token count exceeds `n_ctx`
- System prompt always pinned at index 0