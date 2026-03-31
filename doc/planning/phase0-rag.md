# Phase 0 — RAG Layer Implementation Plan

> **Status:** Planned
> **Depends on:** [phase0.md](phase0.md) — layered architecture scaffold
> **Goal:** Implement the RAG Layer as a **standalone Rust crate** (`rag/` at repo root) using SQLite FTS5 for keyword (BM25) retrieval. No neural embeddings. Store + Retrieve only — agent wiring is Phase 1. This is an independent library crate that any consumer (`lala`, future HTTP API, background indexer) can import via `use rag::RagStore`. The focus is on building a robust, self-contained retrieval crate that can be easily wired into the agent in Phase 1.

---

## 1. Decisions

| Concern | Decision | Rationale |
|---------|----------|-----------|
| Embedding | None (Phase 0) | Avoid model dependency; FTS5 BM25 delivers sufficient keyword recall |
| Retrieval engine | SQLite FTS5 | Bundled with `rusqlite`; no external service; FTS5 exposes native BM25 ranking |
| Scope | Store + Retrieve only | Verify the layer in isolation before wiring into the agent loop |
| DB file | `./lala.db` (next to binary) | Simple default; overridable via `LALA_DB_PATH` env var |
| DB crate | `rusqlite` with `bundled` feature | Compiles SQLite + FTS5 in; no system SQLite install required |
| ID generation | `uuid` v4 | Stable, sortable, collision-free identifiers for documents and chunks |
| Chunker | Character-based sliding window | 512-char chunks, 64-char overlap — simple, no tokeniser dependency |
| Duplicate handling | Skip if `(source)` already exists in `documents` | Prevents duplicate chunks; re-ingest by deleting first |
| Agent wiring | Deferred to Phase 1 | Retrieved chunks injected as a `[system]` context message in `planner.rs` |

---

## 2. Schema

Two objects — one regular table and one FTS5 virtual table:

```sql
-- Parent document record
CREATE TABLE IF NOT EXISTS documents (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    source     TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Full-text search index over chunk text
-- UNINDEXED columns are stored but excluded from the BM25 index
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id    UNINDEXED,
    document_id UNINDEXED,
    chunk_index UNINDEXED,
    chunk_text,            -- indexed for BM25
    char_count  UNINDEXED  -- character length of chunk_text (not token count)
);
```

Both objects are created via `CREATE ... IF NOT EXISTS` inside `RagStore::open()` — no migration tooling needed at this stage.

---

## 3. Crate Structure

The RAG layer lives in its own crate at the repository root, sibling to `lala/` and `LLML/`:

```
rag/
  Cargo.toml      ← [package] name = "rag"; deps: rusqlite (bundled), uuid, anyhow
  src/
    lib.rs         ← RagStore, Chunk, store(), retrieve(), re-exports chunker::chunk
    chunker.rs     ← chunk(text, chunk_size, overlap) -> Vec<String>
```

A Cargo workspace file at the repo root ties the crates together:

```toml
# lala.ai/Cargo.toml
[workspace]
members = ["lala", "rag"]
```

`lib.rs` re-exports the `chunk` function from `chunker` to keep external callers to a single import path.

---

## 4. Public API

### `Chunk`

```rust
pub struct Chunk {
    pub id:           String,
    pub document_id:  String,
    pub chunk_index:  usize,
    pub chunk_text:   String,
    pub score:        f64,   // BM25 rank from SQLite: negative float, more negative = better match
}
```

### `RagStore`

```rust
pub struct RagStore { /* conn: rusqlite::Connection */ }

impl RagStore {
    /// Open (or create) the SQLite DB at `path` and initialise the schema.
    pub fn open(path: &str) -> anyhow::Result<Self>;

    /// Chunk `text`, insert into `documents` + `chunks_fts`, return chunk count.
    pub fn store(&self, title: &str, source: &str, text: &str) -> anyhow::Result<usize>;

    /// BM25 full-text search — return top `k` chunks ordered by relevance.
    pub fn retrieve(&self, query: &str, k: usize) -> anyhow::Result<Vec<Chunk>>;
}
```

### `chunker::chunk`

```rust
/// Split `text` into overlapping character windows.
/// Default call: chunk(text, 512, 64)
///
/// Edge cases:
///   - Empty text → empty Vec
///   - Text shorter than `chunk_size` → single-element Vec containing the full text
///   - `overlap >= chunk_size` → treated as `overlap = 0` (no overlap)
pub fn chunk(text: &str, chunk_size: usize, overlap: usize) -> Vec<String>;
```

---

## 5. Retrieve Query

```sql
SELECT chunk_id, document_id, chunk_index, chunk_text, bm25(chunks_fts) AS score
FROM   chunks_fts
WHERE  chunk_text MATCH ?1
ORDER  BY bm25(chunks_fts)
LIMIT  ?2
```

FTS5's `bm25()` returns a **negative** float — more negative means a better match. `ORDER BY bm25(chunks_fts)` (ascending) therefore returns best-first results. A score of `-3.2` is more relevant than `-1.1`.

**Note on the `MATCH` clause:** FTS5 requires a valid query expression. If the user's query contains special FTS5 characters (`*`, `"`, `OR`, etc.), they should be escaped or the query should be quoted. Phase 0 passes the raw query string directly — this is acceptable for the CLI proof-of-concept, but Phase 1 should sanitise input.

---

## 6. Files Changed

| File | Change |
|------|--------|
| `Cargo.toml` (repo root) | **New** — Cargo workspace: `members = ["lala", "rag"]` |
| `rag/Cargo.toml` | **New** — `[package] name = "rag"`; deps: `rusqlite` (bundled), `uuid` (v4), `anyhow` |
| `rag/src/lib.rs` | **New** — `RagStore`, `Chunk`, `store()`, `retrieve()`, unit tests |
| `rag/src/chunker.rs` | **New** — `chunk()` pure function |
| `lala/Cargo.toml` | Add `rag = { path = "../rag" }` dependency; remove `sqlx` (unused) |
| `lala/src/main.rs` | Add `use rag::RagStore;`, resolve `LALA_DB_PATH` env var, init `RagStore`, thread it to `cli::run()` |
| `lala/src/cli.rs` | Accept `RagStore` (owned or `Arc`), add `/ingest-file <path>` and `/search <query>` commands |

---

## 7. CLI Commands (additions)

| Command | Behaviour |
|---------|-----------|
| `/ingest-file <path>` | Read file at `path`, call `rag.store(filename, path, content)`, print chunk count |
| `/search <query>` | Call `rag.retrieve(query, 5)`, print each result's index, BM25 score, and a 100-char preview |

### Error handling

| Scenario | Behaviour |
|----------|-----------|
| `/ingest-file` — path does not exist | Print `"Error: file not found: <path>"`, continue REPL |
| `/ingest-file` — empty file | Print `"Warning: file is empty, nothing to ingest"`, continue REPL |
| `/ingest-file` — file already ingested (same `source`) | Print `"Already ingested: <path> (N chunks). Use /delete-doc <path> first to re-ingest."` |
| `/search` — empty query | Print `"Usage: /search <query>"`, continue REPL |
| `/search` — no results | Print `"No results found for: <query>"` |

---

## 8. Data Flow

```
/ingest-file doc/architecture.md
      │
      ▼
  std::fs::read_to_string(path)
      │
      ▼
  chunker::chunk(text, 512, 64)  →  Vec<String>   (N chunks)
      │
      ├─► INSERT INTO documents (id, title, source, created_at)
      │
      └─► INSERT INTO chunks_fts (chunk_id, document_id, chunk_index, chunk_text, char_count)
              × N rows


/search layered architecture
      │
      ▼
  SELECT … FROM chunks_fts WHERE chunk_text MATCH ? ORDER BY bm25() LIMIT 5
      │
      ▼
  Vec<Chunk>  →  print preview table
```

---

## 9. Phase 1 Wiring (deferred)

When the agent loop is ready to consume retrieved context, the integration point is already defined in `lala/src/agent/planner.rs`. Retrieved chunks will be injected as a hidden `[system]` message between `REASONING_SYSTEM` and the user turn — mirroring the existing analysis injection in `run_decision()`:

```rust
// Phase 1 addition inside Agent::run_reasoning() or a new run_with_rag()
use rag::{RagStore, Chunk};

ChatMessage {
    role: "system".to_string(),
    content: format!(
        "[Retrieved context]\n{}",
        chunks.iter().map(|c| c.chunk_text.clone()).collect::<Vec<_>>().join("\n---\n")
    ),
}
```

No structural changes to `planner.rs` are needed for this — it is a data injection, not an architectural change. The `rag` crate is already a workspace dependency of `lala`.

---

## 10. Acceptance Criteria

- [ ] `cargo check -p rag` — zero errors, zero warnings
- [ ] `cargo check -p lala` — zero errors, zero warnings
- [ ] `cargo test -p rag` — unit tests:
  - Store one document, retrieve with a term present in it, assert ≥1 chunk with a negative BM25 score
  - Store a document, attempt to store with same `source` again — assert duplicate is rejected
  - Chunker: text shorter than `chunk_size` → returns exactly 1 chunk
  - Chunker: empty text → returns empty vec
  - Retrieve with a query matching nothing → returns empty vec
- [ ] `/ingest-file doc/architecture.md` in the REPL prints the number of chunks stored
- [ ] `/search layered architecture` returns ranked chunk previews from the ingested file
- [ ] `/search` with no arguments prints a usage hint
- [ ] DB file is created at `./lala.db` (or `LALA_DB_PATH`) and persists between REPL sessions

---

## 11. Out of Scope for Phase 0

| Feature | Target phase |
|---------|-------------|
| Neural embeddings (bge-small or similar) | Phase 1 |
| Vector similarity search (pgvector / sqlite-vec) | Phase 1 |
| Agent loop integration (inject chunks into prompt) | Phase 1 |
| Hybrid BM25 + vector reranking | Phase 2 |
| Metadata filtering | Phase 2 |
| Session-scoped retrieval | Phase 1 |
| `/ingest-url` or directory ingestion | Phase 1 |

---

## 12. Crate Independence

The RAG layer is a **standalone Rust crate** (`rag/` at repo root) with zero dependencies on `lala`, the agent, CLI, or model layers. Any application in the workspace can depend on it:

```rust
// Any crate in the workspace can use the RAG layer:
use rag::RagStore;

let store = RagStore::open("./lala.db")?;
let count = store.store("title", "source", "text content")?;
let chunks = store.retrieve("search query", 5)?;
```

This means:
- `lala` (the agentic layer) adds `rag = { path = "../rag" }` to its `Cargo.toml` and calls `use rag::RagStore;`
- The agent layer (Phase 1) can call `store.retrieve()` without knowing about SQLite or FTS5 internals
- Future applications (HTTP API, background indexer, telegram Rust port) can independently depend on the `rag` crate
- The retrieval backend can be swapped (e.g. to pgvector) without changing any consumer code, as long as the `RagStore` method signatures are preserved

**Phase 1 consideration:** If multiple retrieval backends are needed (e.g. FTS5 + vector), introduce a `RagRetriever` trait at that point. For Phase 0, the concrete `RagStore` struct is sufficient.
