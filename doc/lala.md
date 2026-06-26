# lala — CLI Client

> **Location:** `lala.ai/lala/`  
> **Role:** User-facing layer — interactive terminal REPL that sends conversation turns to the LLML API and displays responses with a live spinner.

---

## Overview

`lala` is the front-end of the system. It owns the user experience: readline input, multi-turn conversation history, a spinner animation during inference, folder-based document ingestion, RSS news ingestion, BM25 keyword search, structured memory block search, and clean error recovery. It has no direct knowledge of the model — all LLM communication goes through HTTP to the LLML server.

RAG context is automatically retrieved on **every** query (both direct and reasoning paths) and injected into the model prompt as retrieved context, up to a 800-token budget.

```
User (terminal)
      │
  rustyline REPL  →  command dispatch (/ingest, /ingest-news, /search, /memory-search, ...)
      │                         │
      │                    rag::RagStore  ──►  PostgreSQL FTS + pgvector + memory_blocks
      │
  conversation history (in-memory)
      │
  agent pipeline:
    ├── retrieve_context(query)        → top-5 keyword chunks
    ├── retrieve_memory_context(query) → top-5 structured memory blocks
    ├── inject context into system prompt (token budget: 800)
    └── POST /v1/chat/completions  ──►  LLML API server
      │
  spinner thread  (while waiting)
      │
  print response
```

---

## Source Layout

```
lala/src/
  main.rs              # Entry point — resolves API URL, DB path, SMART_ROUTER; inits RagStore
  cli/
    mod.rs             # REPL loop, animated banner, command/chat dispatch
    chat.rs            # Chat struct — history, classify→route, retrieve_context→inject, spinner
    commands.rs        # Command dispatch (/help, /status, /search, /memory-search, /ingest, /ingest-news, /clear, /exit)
    ingest.rs          # Batch + single-file + RSS news ingestion with progress output
    display.rs         # Spinner, ANSI colours, print_section(), print_sources(), info/success/warn/error helpers
  agent/
    mod.rs
    model.rs           # ApiClient — HTTP wrapper (chat, reason, decide, classify); RouteDecision enum
    planner.rs         # Agent — classify_query(), run_direct(), run_reasoning(), run_decision(), retrieve_context(), retrieve_memory_context()
```

---

## Running

```sh
cd lala

# Default — connects to http://localhost:3000
cargo run

# Custom server URL via argument
cargo run -- http://192.168.1.10:3000

# Custom server URL via environment variable
LLML_API_URL=http://192.168.1.10:3000 cargo run

# Enable LLM-based smart query router (requires LLML server)
LALA_SMART_ROUTER=1 cargo run

# Database URL (default: postgres://postgres:mysecretpassword@localhost:5432/vector_db)
DATABASE_URL=postgres://postgres:mysecretpassword@localhost:5432/vector_db cargo run

# Custom ingest directory (default: ./ingest)
LALA_INGEST_DIR=/path/to/docs cargo run
```

URL resolution priority: **CLI argument → `LLML_API_URL` env var → `http://localhost:3000`**

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|--------|
| `LLML_API_URL` | `http://localhost:3000` | LLML inference server URL |
| `LALA_SMART_ROUTER` | unset | Set to `1` to enable LLM-based query classification |
| `DATABASE_URL` | `postgres://postgres:mysecretpassword@localhost:5432/vector_db` | PostgreSQL connection string for RAG storage |
| `LALA_INGEST_DIR` | `./ingest` | Directory scanned by `/ingest` for batch ingestion |
| `LALA_QDRANT_URL` | _(planned)_ | Qdrant endpoint URL — Phase 1 vector search migration |

---

## CLI Commands

| Input | Action |
|-------|--------|
| Any text | Send as a user message to the LLM (RAG context auto-injected) |
| `/ingest` | Batch-ingest all files in `./ingest/` (or `LALA_INGEST_DIR`) |
| `/ingest-file <path>` | Ingest a single file by explicit path |
| `/ingest-news <rss_url>` | Fetch an RSS feed and ingest all articles |
| `/search <query>` | BM25 full-text search over ingested documents (top 5 results) |
| `/memory-search <query>` | BM25 search over structured memory blocks (facts / capabilities / constraints) |
| `/status` | Show document count, chunk count, ingest directory |
| `/help` | Show available commands |
| `/clear` | Reset conversation history (keeps system prompt) |
| `/exit` | Quit |
| Ctrl-C / Ctrl-D | Quit |

Arrow-key history navigation (up/down) is provided by `rustyline`.

### Ingestion

Place files in the `./ingest/` directory and run `/ingest` to batch-process all of them. Each file is read, chunked into 512-character overlapping windows (64-char overlap), and stored in PostgreSQL with FTS-enabled chunk indexing and auto-extracted memory blocks. Duplicate files (same source path) are skipped. Progress and a summary are displayed:

```
>> /ingest
  ℹ Found 3 file(s) in ./ingest/
  [1/3] architecture.md
  ✓ architecture.md → 12 chunks
  [2/3] design.md
  ✓ design.md → 8 chunks
  [3/3] notes.txt
  ⚠ notes.txt: Already ingested: ./ingest/notes.txt
  ────────────────────────────────────────────────────────────
  Ingested: 2  Skipped: 1  Failed: 0  Chunks: 20
  ────────────────────────────────────────────────────────────
```

For one-off files outside the ingest directory, use `/ingest-file <path>`.

**RSS News Ingestion**

Fetch an RSS feed and automatically ingest all linked articles:

```
>> /ingest-news https://feeds.bbci.co.uk/news/rss.xml
  ℹ Ingesting news from: https://feeds.bbci.co.uk/news/rss.xml
  ────────────────────────────────────────────────────────────
  Ingested: 14  Skipped: 3  Failed: 1
  ────────────────────────────────────────────────────────────
```

Articles are fetched with a 1-second polite delay between requests. A CORS-proxy fallback is tried automatically on HTTP 403 responses. Each article is chunked and deduplicated by URL (source field).

### Search

```
>> /search layered architecture
  [1] score: -3.2140  chunk #2
      The system is structured as five distinct layers…
  [2] score: -1.8700  chunk #0
      lala.ai — Agentic RAG System…
```

Returns top 5 chunks ranked by BM25 relevance. More negative score = better match.

### Memory Search

Search across structured memory blocks extracted from ingested documents:

```
>> /memory-search system constraints
  [1] chunk #3  source: ./ingest/architecture.md
    FACTS: The system processes queries through a layered pipeline…
    CAPABILITIES: Supports multi-turn conversation with RAG context injection…
    CONSTRAINTS: Retrieval is limited to 800 tokens per query…
```

Memory blocks store the same text as chunks (Phase 0 placeholder). In Phase 1, these fields will be populated by LLM-based extraction, exposing structured `facts`, `capabilities`, and `constraints` per chunk.

---

## Conversation History

The full message history is maintained in memory for the duration of the session. Each turn appends a `user` and `assistant` message:

```
[ system prompt, user1, assistant1, user2, assistant2, ... ]
```

The entire history is sent with every request so the model maintains context across turns. `/clear` trims the list back to just the system prompt, starting a fresh conversation without restarting the process.

If the API returns an error, the pending `user` message is removed from history so the context stays consistent.

---

## Spinner

While waiting for the API response, a braille spinner runs on a background thread:

```
  ⠼ thinking...
```

The spinner is stopped and the line is erased before the response is printed, so the output is clean:

```
>> What is Rust?
  ⠼ thinking...          ← live, while waiting
Rust is a systems programming language...   ← replaces the spinner line
```

Implementation: `Arc<AtomicBool>` stop flag shared between the main thread and the spinner thread. When the API call returns (success or error), the flag is set to `false`, the spinner thread exits, and the cursor line is cleared with spaces before any output is printed.

---

## HTTP Client — `agent/model.rs`

### `ApiClient`

```rust
ApiClient::new(base_url: &str) -> ApiClient
```

Wraps a `reqwest::blocking::Client` configured with **no timeout** — CPU inference can take tens of seconds and must not be aborted prematurely.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `chat(messages, max_tokens, model)` | `/v1/chat/completions` | Core call — sends full history, returns reply string |
| `reason(messages, max_tokens)` | `/v1/chat/completions` | Shortcut with `model: "reasoning"` |
| `decide(messages, max_tokens)` | `/v1/chat/completions` | Shortcut with `model: "decision"` |
| `classify(query, context)` | `/v1/classify` | Returns `RouteDecision::{Direct,Reasoning}` |

### `ChatMessage`

```rust
pub struct ChatMessage {
    pub role: String,    // "system" | "user" | "assistant"
    pub content: String,
}
```

Shared between the CLI history and all HTTP request bodies.

### `RouteDecision`

```rust
pub enum RouteDecision { Direct, Reasoning }
```

Returned by `ApiClient::classify()` and by `Agent::classify_query()`. Defaults to `Reasoning` on any parse error (fail-closed).

---

## RAG Context Injection — `cli/chat.rs`

Every query (both direct and reasoning paths) automatically retrieves context from the RAG store before calling the model:

```
input query
    │
    ▼
 retrieve_context(query)         → top-5 BM25 chunks   (800-token budget)
 retrieve_memory_context(query)  → top-5 memory blocks (fits within same budget)
    │
    ▼
 inject into system prompt as "--- Retrieved Context ---" block
    │  (context is omitted if store is empty or no results)
    ▼
 display print_sources() / memory blocks to terminal
    │
    ▼
 pass context_str to run_direct() / run_reasoning() / run_decision()
```

The token budget (`CONTEXT_TOKEN_BUDGET = 800`) is enforced by `limit_chunks_by_tokens()` and `limit_memory_by_tokens()` in `planner.rs` so the combined context never overflows the model's context window.

## Query Router — `agent/planner.rs`

Every user turn goes through a routing decision before inference:

```
input query
    │
    ▼
 classify_query(input, history)
    │
    ├── LALA_SMART_ROUTER=1  →  POST /v1/classify  → RouteDecision
    └── heuristic (default)  →  needs_reasoning(input) → RouteDecision
    │
    ├── Direct     →  run_direct(history, context)             → decision model only
    └── Reasoning  →  run_reasoning(history, context)          → reasoning model
                        run_decision(history, analysis, context) → decision model
```

- `needs_reasoning()` — local keyword + word-count heuristic; used as fallback when server is unreachable or smart router is off.
- `classify_query()` — calls `client.classify()`, falls back to `needs_reasoning()` on any `Err`.
- `retrieve_context()` — sanitises query, calls `store.retrieve()`, returns `Vec<Chunk>`.
- `retrieve_memory_context()` — same flow via `store.retrieve_memory_blocks()`.
- Reasoning output is displayed to the user under a `▷ Reasoning` section (yellow ANSI).

---

## System Prompt

The system prompt is hardcoded in `cli/mod.rs` and always occupies index 0 of the history:

```
You are a friendly AI assistant named lala.
Explain things clearly and naturally.
Respond in full sentences.
```

This can be edited in `cli/mod.rs` under `const SYSTEM_PROMPT`.

---

## Dependencies

| Crate       | Purpose |
|-------------|---------|
| `reqwest`   | Blocking HTTP client for LLML API calls |
| `rustyline` | Readline-style input with history and arrow-key navigation |
| `serde` / `serde_json` | HTTP request/response serialization |
| `anyhow`    | Error propagation |
| `rag` (path dep) | Standalone RAG crate — PostgreSQL FTS + pgvector store, retrieve, embed |

---

## System Architecture

`lala` communicates with the LLML server over HTTP. Both the query classification and inference happen server-side.

```
┌─────────────┐          POST /v1/classify            ┌──────────────────┐
│   lala CLI  │  ──────────────────────────────────►   LLML server    │
│  (lala/)    │  POST /v1/chat/completions         │   (LLML/)         │
│             │  ◄──────────────────────────────────   │                  │
│  User REPL  │  JSON response                     │  llama-cpp-python │
└─────────────┘                                   └──────────────────┘
```

See [LLML.md](LLML.md) for the server-side documentation.

---

## Planned: Vector Search (Phase 1)

The `rag` crate currently uses PostgreSQL FTS for keyword retrieval and stores vector embeddings in pgvector. Phase 1 will extend the backend to add semantic recall over `chunk_embeddings` while preserving the `RagStore` API (`store()`, `retrieve()`, `retrieve_memory_blocks()`).

Required additions (Phase 1):
- Embedding endpoint in LLML or a local embedding model to generate chunk/query vectors
- `RagStore::store_embedding()` calls to persist chunk embeddings into `chunk_embeddings`
- `RagStore::retrieve_by_embedding()` for semantic similarity search
- Optional runtime config for embedding model and retrieval strategy

See [RAG.md](RAG.md) for the full RAG implementation details and storage design.
