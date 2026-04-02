mod chunker;

pub use chunker::chunk;

use anyhow::{Result, bail};
use rusqlite::Connection;
use uuid::Uuid;

/// Trait for LLM-based memory extraction.
/// Implementations extract FACTS, CAPABILITIES, CONSTRAINTS from text via LLM.
pub trait MemoryExtractor {
    /// Extract structured memory from chunk text.
    /// Returns (facts, capabilities, constraints) as semicolon-delimited strings.
    fn extract_memory(&self, chunk_text: &str) -> Result<(String, String, String)>;
}

/// A retrieved chunk with its BM25 relevance score.
pub struct Chunk {
    pub id: String,
    pub document_id: String,
    pub chunk_index: usize,
    pub chunk_text: String,
    /// BM25 rank from SQLite — negative float, more negative = better match.
    pub score: f64,
    /// Title of the parent document.
    pub title: String,
    /// Source path of the parent document.
    pub source: String,
}

/// A structured memory block extracted from a chunk of text.
pub struct MemoryBlock {
    pub id: String,
    pub document_id: String,
    pub chunk_index: usize,
    pub chunk_text: String,
    pub facts: String,
    pub capabilities: String,
    pub constraints: String,
    pub title: String,
    pub source: String,
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

    /// Ingest a document and optionally run memory extraction from an LLM extractor.
    ///
    /// This keeps CLI ingestion as a thin interface and moves chunk creation,
    /// document persist, and memory block saving into the Rag layer.
    pub fn ingest(&self, title: &str, source: &str, text: &str, extractor: Option<&dyn MemoryExtractor>) -> Result<usize> {
        let count = self.store(title, source, text)?;

        if let Some(extractor) = extractor {
            // best effort; do not fail ingestion if extraction has an issue
            let _ = self.extract_memory_from_source(source, extractor);
        }

        Ok(count)
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
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM chunks_fts",
            [],
            |row| row.get(0),
        )?;
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

    /// Extract memory blocks for a source document using an LLM extractor.
    /// Filters prose content, calls LLM for each chunk, and updates memory blocks in database.
    pub fn extract_memory_from_source(
        &self,
        source_path: &str,
        extractor: &dyn MemoryExtractor,
    ) -> Result<(usize, usize)> {
        // Retrieve memory blocks for this source
        let blocks = self.memory_blocks_for_source(source_path)?;

        let mut prose_count = 0;
        let mut filtered_count = 0;

        for block in blocks {
            // Pre-filter: only process prose content
            if !is_prose_content(&block.chunk_text) {
                filtered_count += 1;
                continue;
            }
            prose_count += 1;

            // Call LLM to extract structured memory
            match extractor.extract_memory(&block.chunk_text) {
                Ok((facts, capabilities, constraints)) => {
                    let _ = self.update_memory_block(&block.id, &facts, &capabilities, &constraints);
                }
                Err(_e) => {
                    // Silently skip extraction errors
                }
            }
        }

        Ok((prose_count, filtered_count))
    }
}

/// Detect if text is prose (descriptive, explanatory) vs code or structured data.
pub fn is_prose_content(text: &str) -> bool {
    let text = text.trim();
    
    // Skip empty or whitespace-only text
    if text.is_empty() {
        return false;
    }
    
    // Count heuristic markers
    let mut prose_score = 0i32;
    let mut code_score = 0i32;
    
    // Prose indicators
    if text.len() > 100 {
        prose_score += 1; // Substantial length suggests prose
    }
    if text.contains(|c: char| c.is_alphabetic()) && text.matches(' ').count() > 10 {
        prose_score += 2; // Multiple words with spaces = prose-like
    }
    if text.contains("the ") || text.contains("is ") || text.contains("are ") || text.contains("and ") {
        prose_score += 2; // Natural language connectives
    }
    if text.contains(".") && text.matches('.').count() > 2 {
        prose_score += 2; // Multiple sentences = prose
    }
    if text.to_lowercase().contains("architecture") || 
       text.to_lowercase().contains("description") ||
       text.to_lowercase().contains("explanation") ||
       text.to_lowercase().contains("why ") ||
       text.to_lowercase().contains("how ") {
        prose_score += 3; // Explicit prose keywords
    }
    
    // Code/structured data indicators
    if text.contains('{') && text.contains('}') {
        code_score += 3; // JSON/JS objects
    }
    if text.contains('[') && text.contains(']') && (text.contains(',') || text.contains(':')) {
        code_score += 2; // Arrays or lists with structure
    }
    if text.contains("function ") || text.contains("def ") || text.contains("class ") {
        code_score += 3; // Function/class definitions
    }
    if text.contains("=>") || text.contains("->") {
        code_score += 2; // Arrow functions/types
    }
    if text.contains("import ") || text.contains("from ") && text.contains("import") {
        code_score += 2; // Import statements
    }
    if text.contains("```") || text.contains("    ") && text.matches("    ").count() > 3 {
        code_score += 3; // Code blocks or deep indentation
    }
    if text.contains("|") && text.contains("-") && text.matches('-').count() > 10 {
        code_score += 3; // Markdown tables
    }
    // Detect YAML/key-value structured data
    if text.contains(": ") && !text.contains("description") && text.matches(':').count() > 3 {
        code_score += 2;
    }
    
    // Line break patterns
    let line_count = text.matches('\n').count();
    let avg_line_len = text.len() / (line_count.max(1) + 1);
    
    // Very short lines often indicate code or lists
    if avg_line_len < 30 && line_count > 5 {
        code_score += 2;
    }
    
    prose_score >= code_score
}

/// Placeholder memory block builder — stores chunk text as fallback.
/// Real extraction is done via LLM in the lala CLI ingest pipeline.
///
/// Returns (facts, capabilities, constraints).
fn build_memory_block(chunk: &str) -> (String, String, String) {
    // Placeholder: store the chunk text in all fields.
    // LLM extraction will update these fields during ingest.
    let chunk_text = chunk.to_string();
    (chunk_text.clone(), chunk_text.clone(), chunk_text)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn build_memory_block_placeholder() {
        let chunk = "Test chunk content";
        let (facts, capabilities, constraints) = build_memory_block(chunk);

        // Placeholder stores chunk in all fields
        assert_eq!(facts, chunk);
        assert_eq!(capabilities, chunk);
        assert_eq!(constraints, chunk);
    }

    #[test]
    fn store_and_retrieve_memory_blocks() {
        let tmp_dir = std::env::temp_dir();
        let db_path = tmp_dir.join(format!("rag_test_{}.db", Uuid::new_v4()));
        let db_path_str = db_path.to_str().unwrap();

        let store = RagStore::open(db_path_str).expect("open store");
        let text = "Test chunk text. This system can store facts. It should respect constraints.";
        let count = store.store("test", "source-path", text).expect("store");
        assert!(count > 0);

        let blocks = store.retrieve_memory_blocks("Test OR store", 5).expect("retrieve");
        assert!(!blocks.is_empty());

        let b = &blocks[0];
        assert!(b.facts.len() > 0);
        assert!(b.capabilities.len() > 0);
        assert!(b.constraints.len() > 0);

        // Clean up DB file.
        let _ = fs::remove_file(db_path_str);
    }

    #[test]
    fn update_memory_block() {
        let tmp_dir = std::env::temp_dir();
        let db_path = tmp_dir.join(format!("rag_test_{}.db", Uuid::new_v4()));
        let db_path_str = db_path.to_str().unwrap();

        let store = RagStore::open(db_path_str).expect("open store");
        let text = "Original chunk text";
        store.store("test", "source-path", text).expect("store");

        let blocks = store.retrieve_memory_blocks("Original", 5).expect("retrieve");
        assert!(!blocks.is_empty());
        let block_id = blocks[0].id.clone();

        // Update the memory block
        store.update_memory_block(&block_id, "New facts", "New capabilities", "New constraints")
            .expect("update");

        // Re-fetch by chunk_text (FTS searches chunk_text, not facts/capabilities/constraints)
        let updated_blocks = store.retrieve_memory_blocks("Original", 5).expect("retrieve updated");
        assert!(!updated_blocks.is_empty());
        assert_eq!(updated_blocks[0].facts, "New facts");
        assert_eq!(updated_blocks[0].capabilities, "New capabilities");
        assert_eq!(updated_blocks[0].constraints, "New constraints");

        // Clean up DB file.
        let _ = fs::remove_file(db_path_str);
    }
}

