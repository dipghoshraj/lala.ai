use anyhow::{bail, Result};
use rusqlite::Connection;
use uuid::Uuid;

use crate::chunker::chunk;
use crate::types::{build_memory_block, Chunk, MemoryBlock};

pub struct RagStore {
    conn: Connection,
}

impl RagStore {
    /// Open (or create) the SQLite DB at `path` and initialise the schema.
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS documents (
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

            CREATE TABLE IF NOT EXISTS memory_blocks (
                id            TEXT PRIMARY KEY,
                document_id   TEXT NOT NULL,
                chunk_index   INTEGER NOT NULL,
                chunk_text    TEXT NOT NULL,
                facts         TEXT NOT NULL,
                capabilities  TEXT NOT NULL,
                constraints   TEXT NOT NULL,
                created_at    TEXT NOT NULL
            );",
        )?;

        Ok(Self { conn })
    }

    /// Chunk `text`, insert into `documents` + `chunks_fts`, return chunk count.
    ///
    /// Skips if a document with the same `source` already exists.
    pub fn store(&self, title: &str, source: &str, text: &str) -> Result<usize> {
        let exists: bool = self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM documents WHERE source = ?1)",
            [source],
            |row| row.get(0),
        )?;

        if exists {
            bail!("Already ingested: {source}");
        }

        let doc_id = Uuid::new_v4().to_string();
        let created_at = crate::types::chrono_now();

        let chunks = chunk(text, 512, 64);
        if chunks.is_empty() {
            return Ok(0);
        }

        let tx = self.conn.unchecked_transaction()?;

        tx.execute(
            "INSERT INTO documents (id, title, source, created_at) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![doc_id, title, source, created_at],
        )?;

        for (i, chunk_text) in chunks.iter().enumerate() {
            let chunk_id = Uuid::new_v4().to_string();
            tx.execute(
                "INSERT INTO chunks_fts (chunk_id, document_id, chunk_index, chunk_text, char_count)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![chunk_id, doc_id, i, chunk_text, chunk_text.len()],
            )?;

            let (facts, capabilities, constraints) = build_memory_block(chunk_text);
            let memory_id = Uuid::new_v4().to_string();
            tx.execute(
                "INSERT INTO memory_blocks (id, document_id, chunk_index, chunk_text, facts, capabilities, constraints, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                rusqlite::params![
                    memory_id,
                    doc_id,
                    i as i64,
                    chunk_text,
                    facts,
                    capabilities,
                    constraints,
                    created_at,
                ],
            )?;
        }

        tx.commit()?;

        Ok(chunks.len())
    }

    /// Ingest a document and store chunks with raw text in memory fields.
    ///
    /// No LLM memory extraction is invoked; all data stays as text chunks.
    pub fn ingest(&self, title: &str, source: &str, text: &str) -> Result<usize> {
        self.store(title, source, text)
    }

    /// BM25 full-text search — return top `k` chunks ordered by relevance.
    pub fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Chunk>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
                    bm25(chunks_fts) AS score, d.title, d.source
             FROM   chunks_fts c
             JOIN   documents d ON d.id = c.document_id
             WHERE  c.chunk_text MATCH ?1
             ORDER  BY bm25(chunks_fts)
             LIMIT  ?2",
        )?;

        let rows = stmt.query_map(rusqlite::params![query, k], |row| {
            Ok(Chunk {
                id: row.get(0)?,
                document_id: row.get(1)?,
                chunk_index: row.get::<_, i64>(2)? as usize,
                chunk_text: row.get(3)?,
                score: row.get(4)?,
                title: row.get(5)?,
                source: row.get(6)?,
            })
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Retrieve structured memory blocks for the given query.
    pub fn retrieve_memory_blocks(&self, query: &str, k: usize) -> Result<Vec<MemoryBlock>> {
        let mut stmt = self.conn.prepare(
            "SELECT b.id, b.document_id, b.chunk_index, b.chunk_text,
                    b.facts, b.capabilities, b.constraints,
                    d.title, d.source
             FROM   chunks_fts c
             JOIN   memory_blocks b ON b.document_id = c.document_id AND b.chunk_index = c.chunk_index
             JOIN   documents d ON d.id = c.document_id
             WHERE  c.chunk_text MATCH ?1
             ORDER  BY bm25(chunks_fts)
             LIMIT  ?2",
        )?;

        let rows = stmt.query_map(rusqlite::params![query, k], |row| {
            Ok(MemoryBlock {
                id: row.get(0)?,
                document_id: row.get(1)?,
                chunk_index: row.get::<_, i64>(2)? as usize,
                chunk_text: row.get(3)?,
                facts: row.get(4)?,
                capabilities: row.get(5)?,
                constraints: row.get(6)?,
                title: row.get(7)?,
                source: row.get(8)?,
            })
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Count of documents in the store.
    pub fn document_count(&self) -> Result<usize> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    /// Count of chunks in the store.
    pub fn chunk_count(&self) -> Result<usize> {
        let count: i64 = self.conn.query_row("SELECT COUNT(*) FROM chunks_fts", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    /// Retrieve all memory blocks for a given document_id.
    pub fn memory_blocks_for_document(&self, doc_id: &str) -> Result<Vec<MemoryBlock>> {
        let mut stmt = self.conn.prepare(
            "SELECT b.id, b.document_id, b.chunk_index, b.chunk_text,
                    b.facts, b.capabilities, b.constraints,
                    d.title, d.source
             FROM   memory_blocks b
             JOIN   documents d ON d.id = b.document_id
             WHERE  b.document_id = ?1
             ORDER  BY b.chunk_index ASC",
        )?;

        let rows = stmt.query_map(rusqlite::params![doc_id], |row| {
            Ok(MemoryBlock {
                id: row.get(0)?,
                document_id: row.get(1)?,
                chunk_index: row.get::<_, i64>(2)? as usize,
                chunk_text: row.get(3)?,
                facts: row.get(4)?,
                capabilities: row.get(5)?,
                constraints: row.get(6)?,
                title: row.get(7)?,
                source: row.get(8)?,
            })
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Retrieve all memory blocks for a given source path.
    pub fn memory_blocks_for_source(&self, source_path: &str) -> Result<Vec<MemoryBlock>> {
        let mut stmt = self.conn.prepare(
            "SELECT b.id, b.document_id, b.chunk_index, b.chunk_text,
                    b.facts, b.capabilities, b.constraints,
                    d.title, d.source
             FROM   memory_blocks b
             JOIN   documents d ON d.id = b.document_id
             WHERE  d.source = ?1
             ORDER  BY b.chunk_index ASC",
        )?;

        let rows = stmt.query_map(rusqlite::params![source_path], |row| {
            Ok(MemoryBlock {
                id: row.get(0)?,
                document_id: row.get(1)?,
                chunk_index: row.get::<_, i64>(2)? as usize,
                chunk_text: row.get(3)?,
                facts: row.get(4)?,
                capabilities: row.get(5)?,
                constraints: row.get(6)?,
                title: row.get(7)?,
                source: row.get(8)?,
            })
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Update the facts, capabilities, and constraints for a single memory block.
    pub fn update_memory_block(
        &self,
        block_id: &str,
        facts: &str,
        capabilities: &str,
        constraints: &str,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE memory_blocks SET facts = ?1, capabilities = ?2, constraints = ?3 WHERE id = ?4",
            rusqlite::params![facts, capabilities, constraints, block_id],
        )?;
        Ok(())
    }
}
