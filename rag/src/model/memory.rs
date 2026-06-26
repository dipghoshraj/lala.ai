
use uuid::Uuid;
use crate::model::chunk::DocumentChunk;

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

impl  MemoryBlockRecord {
    
    pub fn from_chunk(chunk: &DocumentChunk, created_at: &str) -> Self {
        let (facts, capabilities, constraints) = crate::model::build_memory_block(&chunk.text);
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