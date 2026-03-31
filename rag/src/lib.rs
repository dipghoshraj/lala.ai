mod chunker;

pub use chunker::chunk;

use anyhow::{Result, bail};
use rusqlite::Connection;
use uuid::Uuid;

/// A retrieved chunk with its BM25 relevance score.
pub struct Chunk {
    pub id: String,
    pub document_id: String,
    pub chunk_index: usize,
    pub chunk_text: String,
    /// BM25 rank from SQLite — negative float, more negative = better match.
    pub score: f64,
}

/// SQLite FTS5-backed document store for keyword (BM25) retrieval.
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
            );",
        )?;

        Ok(Self { conn })
    }

    /// Chunk `text`, insert into `documents` + `chunks_fts`, return chunk count.
    ///
    /// Skips if a document with the same `source` already exists.
    pub fn store(&self, title: &str, source: &str, text: &str) -> Result<usize> {
        // Check for duplicate source
        let exists: bool = self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM documents WHERE source = ?1)",
            [source],
            |row| row.get(0),
        )?;

        if exists {
            bail!("Already ingested: {source}");
        }

        let doc_id = Uuid::new_v4().to_string();
        let created_at = chrono_now();

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
        }

        tx.commit()?;

        Ok(chunks.len())
    }

    /// BM25 full-text search — return top `k` chunks ordered by relevance.
    pub fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Chunk>> {
        let mut stmt = self.conn.prepare(
            "SELECT chunk_id, document_id, chunk_index, chunk_text, bm25(chunks_fts) AS score
             FROM   chunks_fts
             WHERE  chunk_text MATCH ?1
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
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM chunks_fts",
            [],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }
}

/// Simple ISO-8601 timestamp without pulling in chrono.
fn chrono_now() -> String {
    // Use a fixed format: SQLite-friendly datetime string.
    // In production you'd use chrono, but we avoid the dependency here.
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    // Convert to a simple readable timestamp.
    // Not a full ISO-8601 parser, but sufficient for Phase 0 ordering.
    format!("{secs}")
}

