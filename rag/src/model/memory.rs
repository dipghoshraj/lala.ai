
use uuid::Uuid;
use crate::{RagStore, model::chunk::DocumentChunk};
use crate::model::chrono_now;

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
    
    pub fn from_chunk(chunk: &DocumentChunk) -> Self {
        let (facts, capabilities, constraints) = crate::model::build_memory_block(&chunk.text);
        Self {
            id: Uuid::new_v4().to_string(),
            document_id: chunk.document_id.clone(),
            chunk_index: chunk.index,
            chunk_text: chunk.text.clone(),
            facts,
            capabilities,
            constraints,
            created_at: chrono_now(),
        }
    }

    pub fn insert(&self, store: &RagStore) -> anyhow::Result<()> {
        let db = crate::model::db();
        let mut client = db.client();
        client.execute(
            crate::model::sql::INSERT_MEMORY_BLOCK,
            &[&self.id, &self.document_id, &self.chunk_index, &self.chunk_text, 
            &self.facts, &self.capabilities, &self.constraints, &self.created_at],
        )?;
        Ok(())
    }
}