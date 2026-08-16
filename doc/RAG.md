# RAG Layer - lala.ai

> PostgreSQL FTS keyword retrieval + pgvector semantic search.

---

## 1. Overview

The **RAG Layer** is a standalone Rust library crate (`rag/`) handling knowledge storage,
chunking, retrieval, and memory block operations. It injects retrieved context into every
agent query so the model can cite evidence from ingested documents.

**Current state:** PostgreSQL FTS (`tsvector` + GIN index, `ts_rank_cd` scoring) for keyword
retrieval; pgvector (`vector(384)`, IVFFlat cosine index) for dense semantic search.

**Phase 1 (Planned):** Query rewriting, LLM-based memory extraction, hybrid FTS+vector reranking.

The public `RagStore` API is stable across phases — only the backing SQL changes.

---

## 2. Crate Structure

```
rag/
├── Cargo.toml             # deps: postgres, pgvector, uuid, anyhow
└── src/
    ├── lib.rs             # Public API: RagStore, Chunk, EmbeddingSearchResult, MemoryBlock
    ├── store.rs           # PostgreSQL implementation of all RagStore methods
    ├── migrate.rs         # run_migrations() — idempotent SQL file runner
    ├── chunker.rs         # chunk(text, size, overlap) -> Vec<String>
    └── model/             # Document, chunk, memory block, project SQL models
```

### Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `postgres` | ~0.19 | Blocking PostgreSQL client — no async runtime required in `rag` |
| `pgvector` (postgres feature) | ~0.4 | `Vector` type for `vector(384)` column; `<=>` cosine operator |
| `uuid` | ~1.0 | Document / chunk ID generation (v4 random) |
| `anyhow` | ~1.0 | Error propagation via `?` |

---

## 3. Database Schema

Schema is applied automatically by `run_migrations()` from `migrations/*.sql` on every
`RagStore::open()`. Migration versions are tracked in `schema_migrations`.

### Migration 001 — `migrations/001_initial_schema.sql`

#### `documents`

```sql
CREATE TABLE documents (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    source     TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX idx_documents_source ON documents (source);
```

One row per ingested file/URL. `source` is unique — re-ingesting the same path is a no-op.

#### `chunks`

```sql
CREATE TABLE chunks (
    chunk_id    TEXT    PRIMARY KEY,
    document_id TEXT    NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text  TEXT    NOT NULL,
    char_count  INTEGER NOT NULL,
    fts_vector  tsvector GENERATED ALWAYS AS (
                    to_tsvector('english', chunk_text)
                ) STORED
);
CREATE INDEX idx_chunks_fts        ON chunks USING GIN (fts_vector);
CREATE INDEX idx_chunks_document_id ON chunks (document_id);
```

Replaces the former SQLite `chunks_fts` FTS5 virtual table. `fts_vector` is auto-maintained
by PostgreSQL; the GIN index makes `@@` lookups fast.

#### `memory_blocks`

```sql
CREATE TABLE memory_blocks (
    id            TEXT    PRIMARY KEY,
    document_id   TEXT    NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    chunk_text    TEXT    NOT NULL,
    facts         TEXT    NOT NULL,        -- placeholder: = chunk_text
    capabilities  TEXT    NOT NULL,        -- placeholder: = chunk_text
    constraints   TEXT    NOT NULL,        -- placeholder: = chunk_text
    created_at    TEXT    NOT NULL
);
```

One row per chunk. `facts`/`capabilities`/`constraints` are placeholders (= `chunk_text`)
until Phase 1 LLM extraction replaces them.

#### `schema_migrations`

```sql
CREATE TABLE schema_migrations (
    version    TEXT        PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Tracks applied migration files to prevent double-application.

---

### Migration 002 — `migrations/002_pgvector.sql`

#### `chunk_embeddings`

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunk_embeddings (
    id          TEXT        PRIMARY KEY,
    chunk_id    TEXT        NOT NULL REFERENCES chunks (chunk_id) ON DELETE CASCADE,
    document_id TEXT        NOT NULL REFERENCES documents (id)    ON DELETE CASCADE,
    embedding   vector(384),                     -- all-MiniLM-L6-v2 dimensions
    model_name  TEXT        NOT NULL DEFAULT 'all-MiniLM-L6-v2',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_chunk_embeddings_cosine
    ON chunk_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE UNIQUE INDEX idx_chunk_embeddings_chunk ON chunk_embeddings (chunk_id);
```

Stores one 384-dim float vector per chunk. IVFFlat index (`lists = 100`; increase as dataset
grows — rule of thumb: `sqrt(n_rows)`). Cosine distance `<=>` in `[0, 2]`; 0 = identical.

---

## 4. Public API Reference

### Initialization

```rust
pub fn open(database_url: &str) -> Result<RagStore>
```

Connects to PostgreSQL, runs pending migrations, returns a ready store.

```sh
# Default URL (matches docker-compose `db` service):
DATABASE_URL=postgres://postgres:mysecretpassword@localhost:5432/vector_db cargo run

# When `lala serve` is used, `DATABASE_URL` may instead be populated from `lala-serve-env.json`.

# Override migrations directory (default: ./migrations):
LALA_MIGRATIONS_DIR=./custom/migrations cargo run
```

---

### Write Methods

#### `store` / `ingest`

```rust
pub fn store(&self, title: &str, source: &str, text: &str) -> Result<usize>
pub fn ingest(&self, title: &str, source: &str, text: &str) -> Result<usize>  // alias
```

Chunks `text` (512 chars, 64-char overlap) and inserts into `documents`, `chunks`, and
`memory_blocks` in a single transaction. Returns chunk count. Errors if `source` already exists.

#### `store_embedding`

```rust
pub fn store_embedding(
    &self,
    chunk_id: &str,
    document_id: &str,
    embedding: Vec<f32>,   // 384 dimensions required
    model_name: &str,
) -> Result<()>
```

Upserts a precomputed vector into `chunk_embeddings`. The caller computes the vector (e.g. via
an embedding model call). Existing embedding for `chunk_id` is replaced on conflict.

---

### Read Methods

#### `retrieve` — keyword search

```rust
pub fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Chunk>>
```

`websearch_to_tsquery('english', query)` matched against `fts_vector`, ranked by `ts_rank_cd`
(higher = more relevant). Returns up to `k` results.

#### `retrieve_by_embedding` — vector search

```rust
pub fn retrieve_by_embedding(
    &self,
    embedding: Vec<f32>,
    k: usize,
) -> Result<Vec<EmbeddingSearchResult>>
```

Cosine nearest-neighbour search via pgvector `<=>`. Returns up to `k` results sorted by
ascending distance. Returns empty if no embeddings have been stored.

#### `retrieve_memory_blocks`

```rust
pub fn retrieve_memory_blocks(&self, query: &str, k: usize) -> Result<Vec<MemoryBlock>>
```

Same FTS as `retrieve()`, joined with `memory_blocks`. Returns structured metadata per chunk.

#### `memory_blocks_for_document` / `memory_blocks_for_source`

```rust
pub fn memory_blocks_for_document(&self, doc_id: &str) -> Result<Vec<MemoryBlock>>
pub fn memory_blocks_for_source(&self, source_path: &str) -> Result<Vec<MemoryBlock>>
```

All memory blocks for a document (by ID or source path), ordered by `chunk_index`.

#### `document_count` / `chunk_count`

```rust
pub fn document_count(&self) -> Result<usize>
pub fn chunk_count(&self) -> Result<usize>
```

Row counts used by the CLI `/status` command.

---

## 5. Data Structures

### `Chunk`

```rust
pub struct Chunk {
    pub id: String,
    pub document_id: String,
    pub chunk_index: usize,
    pub chunk_text: String,
    /// ts_rank_cd score — higher is more relevant (PostgreSQL FTS).
    pub score: f64,
    pub title: String,
    pub source: String,
}
```

### `EmbeddingSearchResult`

```rust
pub struct EmbeddingSearchResult {
    pub chunk_id: String,
    pub document_id: String,
    pub chunk_text: String,
    pub title: String,
    pub source: String,
    /// Cosine distance in [0, 2]; 0.0 = identical vectors.
    pub distance: f64,
}
```

### `MemoryBlock`

```rust
pub struct MemoryBlock {
    pub id: String,
    pub document_id: String,
    pub chunk_index: usize,
    pub chunk_text: String,
    pub facts: String,           // placeholder: = chunk_text
    pub capabilities: String,    // placeholder: = chunk_text
    pub constraints: String,     // placeholder: = chunk_text
    pub title: String,
    pub source: String,
}
```

---

## 6. Migration Runner

```rust
// rag/src/migrate.rs
pub fn run_migrations(client: &mut postgres::Client, migrations_dir: &str) -> Result<Vec<String>>
```

- Reads `*.sql` files in `migrations_dir` lexicographically (`001_`, `002_`, ...)
- Skips versions already in `schema_migrations`
- Each file runs in a transaction; version is recorded atomically on commit
- Returns the versions applied this run; empty list = already up to date
- Called by `RagStore::open()` automatically on every startup

---

## 7. Chunking

Fixed-size sliding window, character-based:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `chunk_size` | 512 chars | Tune for the embedding model token budget |
| `overlap` | 64 chars | Preserves sentence context at chunk boundaries |

---

## 8. Request Flow

```
user query (lala REPL)
      |
      v  cli/chat.rs
  store.retrieve(query, 5)            <- PostgreSQL FTS (tsvector GIN + ts_rank_cd)
  store.retrieve_memory_blocks(q, 3)  <- same FTS, joined to memory_blocks
      |
      v  inject top chunks as context prefix into conversation history
  ApiClient::chat(history_with_context)
      |  HTTP POST /v1/chat/completions
      v  LLML inference server
  model generates grounded answer
```

`retrieve_by_embedding()` is available but not yet wired into the automatic context injection
path — that is Phase 1 work.
