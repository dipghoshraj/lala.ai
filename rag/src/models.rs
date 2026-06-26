use postgres::Row;
use uuid::Uuid;

use crate::types::{Chunk, EmbeddingSearchResult, MemoryBlock, build_memory_block, chrono_now};

#[derive(Debug, Clone)]
pub struct Document {
    pub id: String,
    pub title: String,
    pub source: String,
    pub created_at: String,
}

impl Document {
    pub fn new(title: &str, source: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            title: title.to_string(),
            source: source.to_string(),
            created_at: chrono_now(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DocumentChunk {
    pub id: String,
    pub document_id: String,
    pub index: i32,
    pub text: String,
    pub char_count: i32,
}

impl DocumentChunk {
    pub fn new(document_id: &str, index: usize, text: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            document_id: document_id.to_string(),
            index: index as i32,
            char_count: text.chars().count() as i32,
            text,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryBlockRecord {
    pub id: String,
    pub document_id: String,
    pub chunk_index: i32,
    pub chunk_text: String,
    pub facts: String,
    pub capabilities: String,
    pub constraints: String,
    pub created_at: String,
}

impl MemoryBlockRecord {
    pub fn from_chunk(chunk: &DocumentChunk, created_at: &str) -> Self {
        let (facts, capabilities, constraints) = build_memory_block(&chunk.text);
        Self {
            id: Uuid::new_v4().to_string(),
            document_id: chunk.document_id.clone(),
            chunk_index: chunk.index,
            chunk_text: chunk.text.clone(),
            facts,
            capabilities,
            constraints,
            created_at: created_at.to_string(),
        }
    }
}

pub struct ChunkSearchRow;

impl ChunkSearchRow {
    pub fn from_row(row: Row) -> Chunk {
        Chunk {
            id: row.get(0),
            document_id: row.get(1),
            chunk_index: row.get::<_, i32>(2) as usize,
            chunk_text: row.get(3),
            score: row.get(4),
            title: row.get(5),
            source: row.get(6),
        }
    }
}

pub struct EmbeddingSearchRow;

impl EmbeddingSearchRow {
    pub fn from_row(row: Row) -> EmbeddingSearchResult {
        EmbeddingSearchResult {
            chunk_id: row.get(0),
            document_id: row.get(1),
            chunk_text: row.get(2),
            title: row.get(3),
            source: row.get(4),
            distance: row.get(5),
        }
    }
}

pub struct MemoryBlockRow;

impl MemoryBlockRow {
    pub fn from_row(row: Row) -> MemoryBlock {
        MemoryBlock {
            id: row.get(0),
            document_id: row.get(1),
            chunk_index: row.get::<_, i32>(2) as usize,
            chunk_text: row.get(3),
            facts: row.get(4),
            capabilities: row.get(5),
            constraints: row.get(6),
            title: row.get(7),
            source: row.get(8),
        }
    }
}
