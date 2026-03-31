# lala — CLI Client

> **Location:** `lala.ai/lala/`  
> **Role:** User-facing layer — interactive terminal REPL that sends conversation turns to the LLML API and displays responses with a live spinner.

---

## Overview

`lala` is the front-end of the system. It owns the user experience: readline input, multi-turn conversation history, a spinner animation during inference, folder-based document ingestion, BM25 search over ingested content, and clean error recovery. It has no direct knowledge of the model — all LLM communication goes through HTTP to the LLML server.

```
User (terminal)
      │
  rustyline REPL  →  command dispatch (/ingest, /search, /status, /help, ...)
      │                         │
      │                    rag::RagStore  ──►  SQLite FTS5 (lala.db)
      │
  conversation history (in-memory)
      │
  POST /v1/chat/completions  ──►  LLML API server
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
    mod.rs             # REPL loop, welcome banner, chat routing
    commands.rs        # Command dispatch (/help, /status, /search, /ingest, /clear, /exit)
    ingest.rs          # Batch + single-file ingestion with progress output
    display.rs         # Spinner, ANSI colours, print_section(), info/success/warn/error helpers
  agent/
    mod.rs
    model.rs           # ApiClient — HTTP wrapper (chat, reason, decide, classify); RouteDecision enum
    planner.rs         # Agent — classify_query(), run_direct(), run_reasoning(), run_decision()
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

# Custom database path (default: ./lala.db)
LALA_DB_PATH=/path/to/my.db cargo run

# Custom ingest directory (default: ./ingest)
LALA_INGEST_DIR=/path/to/docs cargo run
```

URL resolution priority: **CLI argument → `LLML_API_URL` env var → `http://localhost:3000`**

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|--------|
| `LLML_API_URL` | `http://localhost:3000` | LLML inference server URL |
| `LALA_SMART_ROUTER` | unset | Set to `1` to enable LLM-based query classification |
| `LALA_DB_PATH` | `./lala.db` | SQLite database file path for RAG storage |
| `LALA_INGEST_DIR` | `./ingest` | Directory scanned by `/ingest` for batch ingestion |

---

## CLI Commands

| Input | Action |
|-------|--------|
| Any text | Send as a user message to the LLM |
| `/ingest` | Batch-ingest all files in `./ingest/` (or `LALA_INGEST_DIR`) |
| `/ingest-file <path>` | Ingest a single file by explicit path |
| `/search <query>` | BM25 full-text search over ingested documents (top 5 results) |
| `/status` | Show document count, chunk count, ingest directory |
| `/help` | Show available commands |
| `/clear` | Reset conversation history (keeps system prompt) |
| `/exit` | Quit |
| Ctrl-C / Ctrl-D | Quit |

Arrow-key history navigation (up/down) is provided by `rustyline`.

### Ingestion

Place files in the `./ingest/` directory and run `/ingest` to batch-process all of them. Each file is read, chunked into 512-character overlapping windows (64-char overlap), and stored in SQLite FTS5. Duplicate files (same source path) are skipped. Progress and a summary are displayed:

```
>> /ingest
  ℹ Found 3 file(s) in ./ingest/
  [1/3] architecture.md
  ✓ architecture.md → 12 chunks
  [2/3] phase0.md
  ✓ phase0.md → 8 chunks
  [3/3] notes.txt
  ⚠ notes.txt: Already ingested: ./ingest/notes.txt
  ────────────────────────────────────────────────────────────
  Ingested: 2  Skipped: 1  Failed: 0  Chunks: 20
  ────────────────────────────────────────────────────────────
```

For one-off files outside the ingest directory, use `/ingest-file <path>`.

### Search

```
>> /search layered architecture
  [1] score: -3.2140  chunk #2
      The system is structured as five distinct layers…
  [2] score: -1.8700  chunk #0
      lala.ai — Agentic RAG System…
```

Returns top 5 chunks ranked by BM25 relevance. More negative score = better match.

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
    ├── Direct     →  run_direct(history)             → decision model only
    └── Reasoning  →  run_reasoning(history)          → reasoning model
                        run_decision(history, analysis) → decision model
```

- `needs_reasoning()` — local keyword + word-count heuristic; used as fallback when server is unreachable or smart router is off.
- `classify_query()` — calls `client.classify()`, falls back to `needs_reasoning()` on any `Err`.
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
| `rag` (path dep) | Standalone RAG crate — SQLite FTS5 store, retrieve, document/chunk counts |

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

See [LLML-py.md](LLML-py.md) for the server-side documentation.
