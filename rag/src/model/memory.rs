
use uuid::Uuid;
use crate::{model::chunk::DocumentChunk};
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


/// A structured memory block extracted from a chunk of text.
#[derive(Clone)]
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

    pub fn insert(&self) -> anyhow::Result<()> {
        let db = crate::model::db();
        let mut client = db.client();
        client.execute(
            crate::model::sql::INSERT_MEMORY_BLOCK,
            &[&self.id, &self.document_id, &self.chunk_index, &self.chunk_text, 
            &self.facts, &self.capabilities, &self.constraints, &self.created_at],
        )?;
        
        Ok(())
    }

    pub fn fetch_by_documents(query: &str, project_id: &str, limit: i64) -> anyhow::Result<Vec<MemoryBlock>> {
        let db = crate::model::db();
        let mut client = db.client();
        let rows = client.query(
            crate::model::sql::SEARCH_MEMORY_BLOCKS,
            &[&query, &limit, &project_id],
        )?;
        let memory_blocks = rows
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
        Ok(memory_blocks)
    }

    
}