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
