use anyhow::{bail, Result};
use std::sync::Mutex;

use crate::chunker::chunk;
use crate::model::{memory, document, chunk};
use crate::model::chunk::ChunkRow;
use crate::model::memory::MemoryBlock;

pub struct RagStore {
    current_project_id: Mutex<Option<String>>,
}

impl Default for RagStore {
    fn default() -> Self {
        Self {
            current_project_id: Mutex::new(None),
        }
    }
}

impl RagStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn select_project(&self, project_id: &str) {
        let mut current = self.current_project_id.lock().unwrap();
        *current = Some(project_id.to_string());
    }

    pub fn deselect_project(&self) {
        let mut current = self.current_project_id.lock().unwrap();
        *current = None;
    }

    pub fn current_project_id(&self) -> Option<String> {
        self.current_project_id.lock().unwrap().clone()
    }

    fn require_selected_project(&self) -> Result<String> {
        self.current_project_id()
            .ok_or_else(|| anyhow::anyhow!("No project selected. Use /project select <name-or-id> or /project create <name>."))
    }

    /// Chunk `text`, insert into `documents` + `chunks`, return chunk count.
    ///
    /// Skips (returns an error) if a document with the same `source` already exists.
    pub fn store(&self, title: &str, source: &str, text: &str) -> Result<usize> {
        println!("Storing document: {title} ({source})");
        let exists: bool = document::Document::exist(source)?;
        if exists {
            bail!("Already ingested: {source}");
        }

        let chunks = chunk(text, 512, 64);
        if chunks.is_empty() {
            return Ok(0);
        }
        let project_id = self.require_selected_project()?;

        let doc = document::Document::new(title, source, project_id);
        doc.insert()?;

        for (i, chunk_text) in chunks.iter().enumerate() {
            let chunk_idx = i as i32;
            let chunk = chunk::DocumentChunk::new(&doc.id, chunk_idx, chunk_text.clone());
            chunk.insert()?;

            let memory_block = memory::MemoryBlockRecord::from_chunk(&chunk);
            memory_block.insert()?;
        }
        Ok(chunks.len())
    }

    /// Alias for `store()` — ingest a document without LLM memory extraction.
    pub fn ingest(&self, title: &str, source: &str, text: &str) -> Result<usize> {
        self.store(title, source, text)
    }

   
    pub fn retrieve(&self, query: &str, k: usize) -> Result<Vec<ChunkRow>> {
        if query.trim().is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let project_id = match self.current_project_id() {
            Some(id) => id,
            None => return Ok(Vec::new()),
        };
        let k_i64 = k as i64;
        let document_chunks = chunk::DocumentChunk::fetch_by_documents(query, &project_id, k_i64)?;
        Ok(document_chunks)
    }

    
    /// Retrieve structured memory blocks for a full-text query.
    pub fn retrieve_memory_blocks(&self, query: &str, k: usize) -> Result<Vec<MemoryBlock>> {
        if query.trim().is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let project_id = match self.current_project_id() {
            Some(id) => id,
            None => return Ok(Vec::new()),
        };
        let k_i64 = k as i64;
        let results = memory::MemoryBlockRecord::fetch_by_documents(query, &project_id, k_i64)?;
        Ok(results)
    }

    /// Count of documents in the store.
    pub fn document_count(&self) -> Result<usize> {
        let count = document::Document::count()?;
        Ok(count)
    }

    /// Count of chunks in the store.
    pub fn chunk_count(&self) -> Result<usize> {
        let count = chunk::DocumentChunk::count()?;
        Ok(count)
    }
}
