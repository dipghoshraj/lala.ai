use std::sync::Mutex;

use anyhow::{Result, bail};
use pgvector::Vector;
use postgres::{Client, NoTls};
use uuid::Uuid;

use crate::chunker::chunk;
use crate::migrate::run_migrations;
use crate::models::{
    ChunkSearchRow, Document, DocumentChunk, EmbeddingSearchRow, MemoryBlockRecord, MemoryBlockRow,
};
use crate::sql;
use crate::types::{Chunk, EmbeddingSearchResult, MemoryBlock};

/// Default directory where migration SQL files are discovered.
const DEFAULT_MIGRATIONS_DIR: &str = "./migrations";
const DEFAULT_CHUNK_SIZE: usize = 512;
const DEFAULT_CHUNK_OVERLAP: usize = 64;

pub struct RagStore {
    /// Interior-mutable client so all public methods can take `&self`,
    /// matching the original SQLite-based API.
    client: Mutex<Client>,
}

impl RagStore {
    /// Connect to PostgreSQL, run any pending migrations, and return a store.
    pub fn open(database_url: &str) -> Result<Self> {
        let mut client = Client::connect(database_url, NoTls).map_err(|e| {
            anyhow::anyhow!("PostgreSQL connection failed: {e}\nURL: {database_url}")
        })?;

        let migrations_dir = std::env::var("LALA_MIGRATIONS_DIR")
            .unwrap_or_else(|_| DEFAULT_MIGRATIONS_DIR.to_string());

        run_migrations(&mut client, &migrations_dir)?;

        Ok(Self {
            client: Mutex::new(client),
        })
    }

    // ── Write ─────────────────────────────────────────────────────────────────

    /// Chunk `text`, insert into `documents`, `chunks`, and `memory_blocks`, then return chunk count.
    ///
    /// This method intentionally routes all persisted rows through small model
    /// structs before binding them to prepared SQL statements. That keeps the
    /// insertion logic typed and in one place while avoiding a heavyweight ORM
    /// for the small, Postgres-specific RAG schema.
    pub fn store(&self, title: &str, source: &str, text: &str) -> Result<usize> {
        let mut client = self.client.lock().unwrap();

        let exists: bool = client.query_one(sql::DOCUMENT_EXISTS, &[&source])?.get(0);
        if exists {
            bail!("Already ingested: {source}");
        }

        let document = Document::new(title, source);
        let chunks = chunk(text, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP);
        if chunks.is_empty() {
            return Ok(0);
        }

        let chunk_rows: Vec<DocumentChunk> = chunks
            .into_iter()
            .enumerate()
            .map(|(index, text)| DocumentChunk::new(&document.id, index, text))
            .collect();
        let memory_rows: Vec<MemoryBlockRecord> = chunk_rows
            .iter()
            .map(|chunk| MemoryBlockRecord::from_chunk(chunk, &document.created_at))
            .collect();

        let mut tx = client.transaction()?;
        let insert_document = tx.prepare(sql::INSERT_DOCUMENT)?;
        let insert_chunk = tx.prepare(sql::INSERT_CHUNK)?;
        let insert_memory_block = tx.prepare(sql::INSERT_MEMORY_BLOCK)?;

        tx.execute(
            &insert_document,
            &[
                &document.id,
                &document.title,
                &document.source,
                &document.created_at,
            ],
        )?;

        for row in &chunk_rows {
            tx.execute(
                &insert_chunk,
                &[
                    &row.id,
                    &row.document_id,
                    &row.index,
                    &row.text,
                    &row.char_count,
                ],
            )?;
        }

        for row in &memory_rows {
            tx.execute(
                &insert_memory_block,
                &[
                    &row.id,
                    &row.document_id,
                    &row.chunk_index,
                    &row.chunk_text,
                    &row.facts,
                    &row.capabilities,
                    &row.constraints,
                    &row.created_at,
                ],
            )?;
        }

        tx.commit()?;
        Ok(chunk_rows.len())
    }

    /// Alias for `store()` — ingest a document without LLM memory extraction.
    pub fn ingest(&self, title: &str, source: &str, text: &str) -> Result<usize> {
        self.store(title, source, text)
    }

    /// Store a precomputed embedding for a chunk.
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
            sql::UPSERT_CHUNK_EMBEDDING,
            &[&id, &chunk_id, &document_id, &vec, &model_name],
        )?;
        Ok(())
    }

    // ── Read ──────────────────────────────────────────────────────────────────

    /// Full-text search using PostgreSQL `websearch_to_tsquery`.
    ///
    /// The caller can pass natural language directly; the SQL layer builds the
    /// tsquery once in a CTE and reuses it for filtering and ranking, preventing
    /// mismatched retrieval/ranking behavior.
    pub fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Chunk>> {
        if query.trim().is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let k_i64 = k as i64;
        let mut client = self.client.lock().unwrap();
        let rows = client.query(sql::SEARCH_CHUNKS, &[&query, &k_i64])?;
        Ok(rows.into_iter().map(ChunkSearchRow::from_row).collect())
    }

    /// Vector similarity search using pgvector cosine distance (`<=>`).
    pub fn retrieve_by_embedding(
        &self,
        embedding: Vec<f32>,
        k: usize,
    ) -> Result<Vec<EmbeddingSearchResult>> {
        if k == 0 {
            return Ok(Vec::new());
        }

        let k_i64 = k as i64;
        let vec = Vector::from(embedding);
        let mut client = self.client.lock().unwrap();
        let rows = client.query(sql::SEARCH_EMBEDDINGS, &[&vec, &k_i64])?;
        Ok(rows.into_iter().map(EmbeddingSearchRow::from_row).collect())
    }

    /// Retrieve structured memory blocks for a full-text query.
    pub fn retrieve_memory_blocks(&self, query: &str, k: usize) -> Result<Vec<MemoryBlock>> {
        if query.trim().is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let k_i64 = k as i64;
        let mut client = self.client.lock().unwrap();
        let rows = client.query(sql::SEARCH_MEMORY_BLOCKS, &[&query, &k_i64])?;
        Ok(rows.into_iter().map(MemoryBlockRow::from_row).collect())
    }

    /// Count of documents in the store.
    pub fn document_count(&self) -> Result<usize> {
        let mut client = self.client.lock().unwrap();
        let row = client.query_one(sql::DOCUMENT_COUNT, &[])?;
        Ok(row.get::<_, i64>(0) as usize)
    }

    /// Count of chunks in the store.
    pub fn chunk_count(&self) -> Result<usize> {
        let mut client = self.client.lock().unwrap();
        let row = client.query_one(sql::CHUNK_COUNT, &[])?;
        Ok(row.get::<_, i64>(0) as usize)
    }

    /// Retrieve all memory blocks for a given document_id.
    pub fn memory_blocks_for_document(&self, doc_id: &str) -> Result<Vec<MemoryBlock>> {
        let mut client = self.client.lock().unwrap();
        let rows = client.query(sql::MEMORY_BLOCKS_BY_DOCUMENT, &[&doc_id])?;
        Ok(rows.into_iter().map(MemoryBlockRow::from_row).collect())
    }

    /// Retrieve all memory blocks for a given source path.
    pub fn memory_blocks_for_source(&self, source_path: &str) -> Result<Vec<MemoryBlock>> {
        let mut client = self.client.lock().unwrap();
        let rows = client.query(sql::MEMORY_BLOCKS_BY_SOURCE, &[&source_path])?;
        Ok(rows.into_iter().map(MemoryBlockRow::from_row).collect())
    }
}
