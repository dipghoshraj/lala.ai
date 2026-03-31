# Phase 0 — RAG Layer

> **Status:** Done
> **Depends on:** [phase0.md](phase0.md) — layered architecture scaffold
> **Goal:** Standalone Rust `rag` crate using SQLite FTS5 for keyword (BM25) retrieval, consumed by the `lala` CLI. No neural embeddings. Store + Retrieve only — agent wiring is Phase 1.

---

## 1. How to Use

### Start the CLI

```sh
cd lala && cargo run                          # connects to http://localhost:3000
LLML_API_URL=http://192.168.1.10:3000 cargo run   # custom server
```

RAG is built into the REPL. Type `/help` to see all commands.

### Ingest Documents

**Option A — Batch ingest (recommended)**

Place files in the `./ingest/` directory and run:

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

The ingest directory defaults to `./ingest/` and is created automatically on first run. Override with `LALA_INGEST_DIR`.

**Option B — Single file**

```
>> /ingest-file doc/architecture.md
  ✓ architecture.md → 12 chunks
```

### Search

```
>> /search layered architecture
  ────────────────────────────────────────────────────────────
  [1] score: -3.2140  chunk #2
      The system is structured as five distinct layers. Data only flows between adjacent…

  [2] score: -1.8700  chunk #0
      lala.ai — Agentic RAG System. Rust-based local Agentic RAG system…
  ────────────────────────────────────────────────────────────
```

Returns the top 5 chunks ranked by BM25 relevance. More negative score = better match.

### Check Status

```
>> /status
  ────────────────────────────────────────────────────────────
  Documents: 3    Chunks: 28
  Ingest dir: ./ingest
  ────────────────────────────────────────────────────────────
```

### All CLI Commands

| Command | Behaviour |
|---------|-----------|
| `/ingest` | Batch-ingest all files in `./ingest/` (or `LALA_INGEST_DIR`) |
| `/ingest-file <path>` | Ingest a single file by explicit path |
| `/search <query>` | BM25 full-text search over ingested chunks (top 5) |
| `/status` | Show document count, chunk count, ingest directory |
| `/help` | Show available commands |
| `/clear` | Reset conversation history |
| `/exit` | Quit |

---

## 2. How It Works

### Data Flow — Ingestion

```
/ingest  or  /ingest-file doc/architecture.md
      │
      ▼
  cli/ingest.rs — scan directory or read single file
      │  std::fs::read_to_string(path)
      ▼
  rag::chunk(text, 512, 64)  →  Vec<String>   (N chunks)
      │  512-char windows, 64-char overlap
      │
      ├─► INSERT INTO documents (id, title, source, created_at)
      │
      └─► INSERT INTO chunks_fts (chunk_id, document_id, chunk_index, chunk_text, char_count)
              × N rows   (inside a single transaction)
```

### Data Flow — Retrieval

```
/search layered architecture
      │
      ▼
  SELECT chunk_id, document_id, chunk_index, chunk_text, bm25(chunks_fts) AS score
  FROM   chunks_fts
  WHERE  chunk_text MATCH 'layered architecture'
  ORDER  BY bm25(chunks_fts)
  LIMIT  5
      │
      ▼
  Vec<Chunk>  →  printed as ranked previews
```

FTS5's `bm25()` returns a **negative** float — more negative = better match. A score of `-3.2` is more relevant than `-1.1`.

### Error Handling

| Scenario | Behaviour |
|----------|-----------|
| File not found | `✗ cannot read file: ...`, continue REPL |
| Empty file | `⚠ file is empty`, skipped |
| Duplicate source | `⚠ Already ingested: <path>`, skipped |
| Empty search query | `Usage: /search <query>` |
| No search results | `⚠ No results found for: <query>` |
| Unknown command | `⚠ Unknown command: ...` + hint to use `/help` |

---

## 3. Architecture

### Module Layout

```
lala.ai/
  Cargo.toml                  ← Workspace root: members = ["lala", "rag"], resolver = "3"
  Cargo.lock                  ← Shared lockfile (auto-generated)

  rag/                        ← Standalone RAG library crate
    Cargo.toml                ← deps: rusqlite (bundled), uuid (v4), anyhow
    src/
      lib.rs                  ← RagStore, Chunk, store(), retrieve(), document_count(), chunk_count()
      chunker.rs              ← chunk(text, chunk_size, overlap) → Vec<String>

  lala/                       ← CLI + Agent binary crate
    Cargo.toml                ← deps include: rag = { path = "../rag" }
    src/
      main.rs                 ← Startup: resolve API URL + DB path, init RagStore, start CLI
      cli/
        mod.rs                ← REPL loop, welcome banner, chat routing
        commands.rs           ← Command dispatch (/help, /status, /search, /ingest, /clear, /exit)
        ingest.rs             ← Batch + single-file ingestion with progress output
        display.rs            ← Spinner, colours, print_section(), info/success/warn/error helpers
      agent/
        mod.rs
        model.rs              ← ApiClient — HTTP wrapper
        planner.rs            ← Agent — query router, reasoning→decision pipeline
```

### Decisions

| Concern | Decision | Rationale |
|---------|----------|-----------|
| Embedding | None (Phase 0) | Avoid model dependency; FTS5 BM25 delivers sufficient keyword recall |
| Retrieval engine | SQLite FTS5 | Bundled with `rusqlite`; no external service; FTS5 exposes native BM25 ranking |
| DB file | `./lala.db` (next to binary) | Simple default; overridable via `LALA_DB_PATH` env var |
| DB crate | `rusqlite` with `bundled` feature | Compiles SQLite + FTS5 in; no system SQLite install required |
| ID generation | `uuid` v4 | Collision-free identifiers for documents and chunks |
| Chunker | Character-based sliding window | 512-char chunks, 64-char overlap — simple, no tokeniser dependency |
| Duplicate handling | Skip if `(source)` already exists in `documents` | Prevents duplicate chunks |
| Ingest directory | `./ingest/` (configurable) | Batch ingestion of multiple files from a known location |
| CLI modularity | `cli/` directory with submodules | Each file is focused, testable, maintainable |

### Schema

```sql
CREATE TABLE IF NOT EXISTS documents (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    source     TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id    UNINDEXED,
    document_id UNINDEXED,
    chunk_index UNINDEXED,
    chunk_text,
    char_count  UNINDEXED
);
```

Both created via `CREATE ... IF NOT EXISTS` inside `RagStore::open()`.

### Public API (rag crate)

```rust
use rag::RagStore;

let store = RagStore::open("./lala.db")?;

// Ingest: chunk text and store in SQLite FTS5
let count = store.store("title", "source_path", "text content")?;

// Retrieve: BM25 full-text search, top-k results
let chunks = store.retrieve("search query", 5)?;

// Stats
let docs = store.document_count()?;
let chunks = store.chunk_count()?;
```

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LALA_DB_PATH` | `./lala.db` | SQLite database file path |
| `LALA_INGEST_DIR` | `./ingest` | Directory scanned by `/ingest` |
| `LLML_API_URL` | `http://localhost:3000` | LLML inference server URL |
| `LALA_SMART_ROUTER` | unset | Set to `1` to enable LLM-based query classification |

---

## 4. Crate Independence

The `rag` crate has zero dependencies on `lala`, the agent, CLI, or model layers. Any application in the workspace can depend on it:

- `lala` consumes it via `rag = { path = "../rag" }` in `Cargo.toml`
- Future applications (HTTP API, background indexer, telegram Rust port) can independently depend on the `rag` crate
- The retrieval backend can be swapped (e.g. to pgvector) without changing any consumer code, as long as the `RagStore` method signatures are preserved

---

## 5. Phase 1 Wiring (deferred)

Retrieved chunks will be injected as a hidden `[system]` message in the agent's reasoning pipeline inside `lala/src/agent/planner.rs`:

```rust
ChatMessage {
    role: "system".to_string(),
    content: format!(
        "[Retrieved context]\n{}",
        chunks.iter().map(|c| c.chunk_text.clone()).collect::<Vec<_>>().join("\n---\n")
    ),
}
```

No structural changes needed — it is a data injection, not an architectural change.

---

## 6. Out of Scope

| Feature | Target phase |
|---------|-------------|
| Neural embeddings | Phase 1 |
| Vector similarity search | Phase 1 |
| Agent loop integration (inject chunks into prompt) | Phase 1 |
| Hybrid BM25 + vector reranking | Phase 2 |
| Metadata filtering | Phase 2 |
| Session-scoped retrieval | Phase 1 |
| `/ingest-url` or recursive directory ingestion | Phase 1 |
