use std::sync::Mutex;

use anyhow::{bail, Result};
use pgvector::Vector;
use postgres::{Client, NoTls};
use uuid::Uuid;

use crate::chunker::chunk;
use crate::migrate::run_migrations;
use crate::types::{build_memory_block, Chunk, EmbeddingSearchResult, MemoryBlock};
use crate::model::{memory, document, chunk};

/// Default directory where migration SQL files are discovered.
const DEFAULT_MIGRATIONS_DIR: &str = "./migrations";

pub struct RagStore {
    /// Interior-mutable client so all public methods can take `&self`,
    /// matching the original SQLite-based API.
    client: Mutex<Client>,
}

impl RagStore {
    /// Connect to PostgreSQL, run any pending migrations, and return a store.
    ///
    /// `database_url` is a standard libpq connection string, e.g.
    /// `"postgres://postgres:postgres@localhost:5432/lala"`.
    ///
    /// Migrations are loaded from `./migrations/` by default; override with
    /// the `LALA_MIGRATIONS_DIR` environment variable.
    pub fn open(database_url: &str) -> Result<Self> {
        let mut client = Client::connect(database_url, NoTls)
            .map_err(|e| anyhow::anyhow!("PostgreSQL connection failed: {e}\nURL: {database_url}"))?;

        let migrations_dir = std::env::var("LALA_MIGRATIONS_DIR")
            .unwrap_or_else(|_| DEFAULT_MIGRATIONS_DIR.to_string());

        run_migrations(&mut client, &migrations_dir)?;

        Ok(Self {
            client: Mutex::new(client),
        })
    }

    // ── Write ─────────────────────────────────────────────────────────────────

    /// Chunk `text`, insert into `documents` + `chunks`, return chunk count.
    ///
    /// Skips (returns an error) if a document with the same `source` already exists.
    pub fn store(&self, title: &str, source: &str, text: &str) -> Result<usize> {
        let exists: bool = document::Document::exist(source)?;
        if exists {
            bail!("Already ingested: {source}");
        }

        let chunks = chunk(text, 512, 64);
        if chunks.is_empty() {
            return Ok(0);
        }

        let doc = document::Document::new(title, source);
        doc.insert()?;

        for (i, chunk_text) in chunks.iter().enumerate() {
            let chunk_idx = i as i32;

            let chunk = chunk::DocumentChunk::new(&doc.id, chunk_idx, chunk_text.clone());
            chunk.insert()?;

            let memory_block = memory::MemoryBlockRecord::from_chunk(&chunk);
            memory_block.insert(&self)?;
        }
        Ok(chunks.len())
    }

    /// Alias for `store()` — ingest a document without LLM memory extraction.
    pub fn ingest(&self, title: &str, source: &str, text: &str) -> Result<usize> {
        self.store(title, source, text)
    }

    /// Store a precomputed embedding for a chunk.
    ///
    /// Upserts — if an embedding for `chunk_id` already exists it is replaced.
    ///
    /// # Arguments
    /// * `chunk_id`    — must reference an existing row in `chunks`
    /// * `document_id` — must reference an existing row in `documents`
    /// * `embedding`   — dense float vector (must be 384 dimensions for the
    ///                   default `all-MiniLM-L6-v2` model)
    /// * `model_name`  — name of the embedding model used
    pub fn store_embedding(
        &self,
        chunk_id: &str,
        document_id: &str,
        embedding: Vec<f32>,
        model_name: &str,
    ) -> Result<()> {
        let id = Uuid::new_v4().to_string();
        let vec = Vector::from(embedding);

        let mut client = self.client.lock().unwrap();
        client.execute(
            "INSERT INTO chunk_embeddings (id, chunk_id, document_id, embedding, model_name)
             VALUES ($1, $2, $3, $4, $5)
             ON CONFLICT (chunk_id) DO UPDATE
                 SET embedding  = EXCLUDED.embedding,
                     model_name = EXCLUDED.model_name,
                     created_at = now()",
            &[&id, &chunk_id, &document_id, &vec, &model_name],
        )?;
        Ok(())
    }

    // ── Read ──────────────────────────────────────────────────────────────────

    /// Full-text search using PostgreSQL `websearch_to_tsquery`.
    ///
    /// Returns up to `k` chunks ranked by `ts_rank_cd` (higher = more relevant).
    pub fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Chunk>> {
        let k_i64 = k as i64;
        let mut client = self.client.lock().unwrap();

        let rows = client.query(
            "SELECT c.chunk_id,
                    c.document_id,
                    c.chunk_index,
                    c.chunk_text,
                    ts_rank_cd(c.fts_vector, websearch_to_tsquery('english', $1)) AS score,
                    d.title,
                    d.source
             FROM   chunks c
             JOIN   documents d ON d.id = c.document_id
             WHERE  c.fts_vector @@ websearch_to_tsquery('english', $1)
             ORDER  BY score DESC
             LIMIT  $2",
            &[&query, &k_i64],
        )?;

        let results = rows
            .into_iter()
            .map(|row| Chunk {
                id: row.get(0),
                document_id: row.get(1),
                chunk_index: row.get::<_, i32>(2) as usize,
                chunk_text: row.get(3),
                score: row.get(4),
                title: row.get(5),
                source: row.get(6),
            })
            .collect();

        Ok(results)
    }

    /// Vector similarity search using pgvector cosine distance (`<=>`).
    ///
    /// `embedding` must have the same dimensions as stored embeddings (384).
    /// Returns up to `k` results sorted by ascending cosine distance
    /// (closer = more similar).
    pub fn retrieve_by_embedding(
        &self,
        embedding: Vec<f32>,
        k: usize,
    ) -> Result<Vec<EmbeddingSearchResult>> {
        let k_i64 = k as i64;
        let vec = Vector::from(embedding);
        let mut client = self.client.lock().unwrap();

        let rows = client.query(
            "SELECT e.chunk_id,
                    e.document_id,
                    c.chunk_text,
                    d.title,
                    d.source,
                    (e.embedding <=> $1) AS distance
             FROM   chunk_embeddings e
             JOIN   chunks    c ON c.chunk_id  = e.chunk_id
             JOIN   documents d ON d.id        = e.document_id
             ORDER  BY distance ASC
             LIMIT  $2",
            &[&vec, &k_i64],
        )?;

        let results = rows
            .into_iter()
            .map(|row| EmbeddingSearchResult {
                chunk_id: row.get(0),
                document_id: row.get(1),
                chunk_text: row.get(2),
                title: row.get(3),
                source: row.get(4),
                distance: row.get(5),
            })
            .collect();

        Ok(results)
    }

    /// Retrieve structured memory blocks for a full-text query.
    pub fn retrieve_memory_blocks(&self, query: &str, k: usize) -> Result<Vec<MemoryBlock>> {
        let k_i64 = k as i64;
        let mut client = self.client.lock().unwrap();

        let rows = client.query(
            "SELECT b.id,
                    b.document_id,
                    b.chunk_index,
                    b.chunk_text,
                    b.facts,
                    b.capabilities,
                    b.constraints,
                    d.title,
                    d.source
             FROM   chunks c
             JOIN   memory_blocks b
                    ON  b.document_id = c.document_id
                    AND b.chunk_index = c.chunk_index
             JOIN   documents d ON d.id = c.document_id
             WHERE  c.fts_vector @@ websearch_to_tsquery('english', $1)
             ORDER  BY ts_rank_cd(c.fts_vector, websearch_to_tsquery('english', $1)) DESC
             LIMIT  $2",
            &[&query, &k_i64],
        )?;

        let results = rows
            .into_iter()
            .map(|row| MemoryBlock {
                id: row.get(0),
                document_id: row.get(1),
                chunk_index: row.get::<_, i32>(2) as usize,
                chunk_text: row.get(3),
                facts: row.get(4),
                capabilities: row.get(5),
                constraints: row.get(6),
                title: row.get(7),
                source: row.get(8),
            })
            .collect();

        Ok(results)
    }

    /// Count of documents in the store.
    pub fn document_count(&self) -> Result<usize> {
        let mut client = self.client.lock().unwrap();
        let row = client.query_one("SELECT COUNT(*) FROM documents", &[])?;
        Ok(row.get::<_, i64>(0) as usize)
    }

    /// Count of chunks in the store.
    pub fn chunk_count(&self) -> Result<usize> {
        let mut client = self.client.lock().unwrap();
        let row = client.query_one("SELECT COUNT(*) FROM chunks", &[])?;
        Ok(row.get::<_, i64>(0) as usize)
    }

    /// Retrieve all memory blocks for a given document_id.
    pub fn memory_blocks_for_document(&self, doc_id: &str) -> Result<Vec<MemoryBlock>> {
        let mut client = self.client.lock().unwrap();

        let rows = client.query(
            "SELECT b.id,
                    b.document_id,
                    b.chunk_index,
                    b.chunk_text,
                    b.facts,
                    b.capabilities,
                    b.constraints,
                    d.title,
                    d.source
             FROM   memory_blocks b
             JOIN   documents d ON d.id = b.document_id
             WHERE  b.document_id = $1
             ORDER  BY b.chunk_index ASC",
            &[&doc_id],
        )?;

        Ok(rows
            .into_iter()
            .map(|row| MemoryBlock {
                id: row.get(0),
                document_id: row.get(1),
                chunk_index: row.get::<_, i32>(2) as usize,
                chunk_text: row.get(3),
                facts: row.get(4),
                capabilities: row.get(5),
                constraints: row.get(6),
                title: row.get(7),
                source: row.get(8),
            })
            .collect())
    }

    /// Retrieve all memory blocks for a given source path.
    pub fn memory_blocks_for_source(&self, source_path: &str) -> Result<Vec<MemoryBlock>> {
        let mut client = self.client.lock().unwrap();

        let rows = client.query(
            "SELECT b.id,
                    b.document_id,
                    b.chunk_index,
                    b.chunk_text,
                    b.facts,
                    b.capabilities,
                    b.constraints,
                    d.title,
                    d.source
             FROM   memory_blocks b
             JOIN   documents d ON d.id = b.document_id
             WHERE  d.source = $1
             ORDER  BY b.chunk_index ASC",
            &[&source_path],
        )?;

        Ok(rows
            .into_iter()
            .map(|row| MemoryBlock {
                id: row.get(0),
                document_id: row.get(1),
                chunk_index: row.get::<_, i32>(2) as usize,
                chunk_text: row.get(3),
                facts: row.get(4),
                capabilities: row.get(5),
                constraints: row.get(6),
                title: row.get(7),
                source: row.get(8),
            })
            .collect())
    }
}
