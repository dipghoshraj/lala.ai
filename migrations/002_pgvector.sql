-- Migration 002 — pgvector extension + chunk_embeddings table
-- Stores dense vector embeddings alongside keyword chunks.
-- Default embedding dimension: 384 (all-MiniLM-L6-v2).
-- Cosine similarity index via IVFFlat (tune `lists` after data grows).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    id          TEXT        PRIMARY KEY,
    chunk_id    TEXT        NOT NULL REFERENCES chunks (chunk_id) ON DELETE CASCADE,
    document_id TEXT        NOT NULL REFERENCES documents (id)    ON DELETE CASCADE,
    embedding   vector(384),                      -- all-MiniLM-L6-v2 dimensions
    model_name  TEXT        NOT NULL DEFAULT 'all-MiniLM-L6-v2',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- IVFFlat index for approximate cosine nearest-neighbour search.
-- lists = 100 is a reasonable default; increase once the collection grows
-- (rule of thumb: sqrt(n_rows)).
CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_cosine
    ON chunk_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE UNIQUE INDEX IF NOT EXISTS idx_chunk_embeddings_chunk
    ON chunk_embeddings (chunk_id);
