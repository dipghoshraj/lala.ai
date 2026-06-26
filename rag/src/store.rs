use std::sync::Mutex;

use anyhow::{bail, Result};
use pgvector::Vector;
use postgres::{Client, NoTls};
use uuid::Uuid;

use crate::chunker::chunk;
use crate::migrate::run_migrations;
use crate::types::{build_memory_block, EmbeddingSearchResult};
use crate::model::{memory, document, chunk};
use crate::model::chunk::ChunkRow;
use crate::model::memory::MemoryBlock;

pub struct RagStore{}

impl RagStore {

    /// Chunk `text`, insert into `documents` + `chunks`, return chunk count.
    ///
    /// Skips (returns an error) if a document with the same `source` already exists.
    pub fn store(&self, title: &str, source: &str, text: &str) -> Result<usize> {
        let exists: bool = document::Document::exist(source)?;
        if exists {
            bail!("Already ingested: {source}");
        }

        let chunks = chunk(text, 512, 64);
        if chunks.is_empty() {
            return Ok(0);
        }

        let doc = document::Document::new(title, source);
        doc.insert()?;

        for (i, chunk_text) in chunks.iter().enumerate() {
            let chunk_idx = i as i32;
            let chunk = chunk::DocumentChunk::new(&doc.id, chunk_idx, chunk_text.clone());
            chunk.insert()?;

            let memory_block = memory::MemoryBlockRecord::from_chunk(&chunk);
            memory_block.insert(&self)?;
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
        let k_i64 = k as i64;
        let document_chunks =  chunk::DocumentChunk::fetch_by_documents(query, k_i64)?;
        Ok(document_chunks)
    }

    
    /// Retrieve structured memory blocks for a full-text query.
    pub fn retrieve_memory_blocks(&self, query: &str, k: usize) -> Result<Vec<MemoryBlock>> {
        let k_i64 = k as i64;
        let resuluts = memory::MemoryBlockRecord::fetch_by_documents(self, query, k_i64)?;
        Ok(resuluts)
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
