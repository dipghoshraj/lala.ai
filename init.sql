-- Enable pgvector extension (make sure it's installed)
CREATE EXTENSION IF NOT EXISTS vector;

-- 1️⃣ Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2️⃣ Messages table
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,         -- e.g., "user" or "assistant"
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3️⃣ Documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    title TEXT,
    source TEXT,
    content TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4️⃣ Document chunks table
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(1536),
    token_count INT
);

-- 5️⃣ Queries table
CREATE TABLE IF NOT EXISTS queries (
    id UUID PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    query_text TEXT NOT NULL,
    query_embedding VECTOR(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- -- 6️⃣ Retrieval results table
-- CREATE TABLE IF NOT EXISTS retrieval_results (
--     id UUID PRIMARY KEY,
--     query_id UUID NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
--     chunk_id UUID NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
--     score FLOAT,
--     rank INT
-- );

-- 7️⃣ Answers table
CREATE TABLE IF NOT EXISTS answers (
    id UUID PRIMARY KEY,
    query_id UUID NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    message_id UUID REFERENCES messages(id),
    answer_text TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- -- 8️⃣ Answer citations table (many-to-many between answers and chunks)
-- CREATE TABLE IF NOT EXISTS answer_citations (
--     answer_id UUID NOT NULL REFERENCES answers(id) ON DELETE CASCADE,
--     chunk_id UUID NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
--     PRIMARY KEY(answer_id, chunk_id)
-- );