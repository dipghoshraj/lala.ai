


pub const DOCUMENT_EXISTS: &str = "SELECT EXISTS(SELECT 1 FROM documents WHERE source = $1)";
pub const INSERT_DOCUMENT: &str =
    "INSERT INTO documents (id, title, source, created_at) VALUES ($1, $2, $3, $4)";
pub const INSERT_CHUNK: &str = "INSERT INTO chunks (chunk_id, document_id, chunk_index, chunk_text, char_count) VALUES ($1, $2, $3, $4, $5)";
pub const INSERT_MEMORY_BLOCK: &str = "INSERT INTO memory_blocks (id, document_id, chunk_index, chunk_text, facts, capabilities, constraints, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)";
pub const UPSERT_CHUNK_EMBEDDING: &str =
    "INSERT INTO chunk_embeddings (id, chunk_id, document_id, embedding, model_name)
 VALUES ($1, $2, $3, $4, $5)
 ON CONFLICT (chunk_id) DO UPDATE
 SET embedding = EXCLUDED.embedding,
     model_name = EXCLUDED.model_name,
     created_at = now()";

pub const SEARCH_CHUNKS: &str = "WITH query AS (SELECT websearch_to_tsquery('english', $1) AS tsq)
 SELECT c.chunk_id,
        c.document_id,
        c.chunk_index,
        c.chunk_text,
        ts_rank_cd(c.fts_vector, query.tsq) AS score,
        d.title,
        d.source
 FROM chunks c
 JOIN documents d ON d.id = c.document_id
 CROSS JOIN query
 WHERE query.tsq <> ''::tsquery
   AND c.fts_vector @@ query.tsq
 ORDER BY score DESC, c.chunk_index ASC
 LIMIT $2";

pub const SEARCH_EMBEDDINGS: &str = "SELECT e.chunk_id,
        e.document_id,
        c.chunk_text,
        d.title,
        d.source,
        (e.embedding <=> $1) AS distance
 FROM chunk_embeddings e
 JOIN chunks c ON c.chunk_id = e.chunk_id
 JOIN documents d ON d.id = e.document_id
 ORDER BY distance ASC
 LIMIT $2";

pub const SEARCH_MEMORY_BLOCKS: &str =
    "WITH query AS (SELECT websearch_to_tsquery('english', $1) AS tsq)
 SELECT b.id,
        b.document_id,
        b.chunk_index,
        b.chunk_text,
        b.facts,
        b.capabilities,
        b.constraints,
        d.title,
        d.source
 FROM chunks c
 JOIN memory_blocks b ON b.document_id = c.document_id AND b.chunk_index = c.chunk_index
 JOIN documents d ON d.id = c.document_id
 CROSS JOIN query
 WHERE query.tsq <> ''::tsquery
   AND c.fts_vector @@ query.tsq
 ORDER BY ts_rank_cd(c.fts_vector, query.tsq) DESC, b.chunk_index ASC
 LIMIT $2";

pub const DOCUMENT_COUNT: &str = "SELECT COUNT(*) FROM documents";
pub const CHUNK_COUNT: &str = "SELECT COUNT(*) FROM chunks";

pub const MEMORY_BLOCKS_BY_DOCUMENT: &str = "SELECT b.id, b.document_id, b.chunk_index, b.chunk_text, b.facts, b.capabilities, b.constraints, d.title, d.source
 FROM memory_blocks b
 JOIN documents d ON d.id = b.document_id
 WHERE b.document_id = $1
 ORDER BY b.chunk_index ASC";

pub const MEMORY_BLOCKS_BY_SOURCE: &str = "SELECT b.id, b.document_id, b.chunk_index, b.chunk_text, b.facts, b.capabilities, b.constraints, d.title, d.source
 FROM memory_blocks b
 JOIN documents d ON d.id = b.document_id
 WHERE d.source = $1
 ORDER BY b.chunk_index ASC";