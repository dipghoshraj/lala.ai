-- Migration 001 — initial schema
-- Equivalent of the SQLite FTS5 schema, now in PostgreSQL.
-- FTS5 BM25 virtual table → real table with tsvector column + GIN index.

-- Track applied migrations (self-referential; created first so 001 can insert itself).
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    TEXT        PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Documents ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS documents (
    id         TEXT        PRIMARY KEY,
    title      TEXT        NOT NULL,
    source     TEXT        NOT NULL,
    created_at TEXT        NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_source
    ON documents (source);

-- ── Chunks (replaces chunks_fts FTS5 virtual table) ─────────────────────────
-- fts_vector is a generated tsvector column; a GIN index makes full-text
-- queries fast (equivalent to SQLite's hidden BM25 index on chunks_fts).

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT    PRIMARY KEY,
    document_id TEXT    NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text  TEXT    NOT NULL,
    char_count  INTEGER NOT NULL,
    fts_vector  tsvector GENERATED ALWAYS AS (
                    to_tsvector('english', chunk_text)
                ) STORED
);

CREATE INDEX IF NOT EXISTS idx_chunks_fts
    ON chunks USING GIN (fts_vector);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id
    ON chunks (document_id);

-- ── Memory blocks ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS memory_blocks (
    id            TEXT    PRIMARY KEY,
    document_id   TEXT    NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    chunk_text    TEXT    NOT NULL,
    facts         TEXT    NOT NULL,
    capabilities  TEXT    NOT NULL,
    constraints   TEXT    NOT NULL,
    created_at    TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_blocks_document_id
    ON memory_blocks (document_id);

CREATE INDEX IF NOT EXISTS idx_memory_blocks_chunk
    ON memory_blocks (document_id, chunk_index);
