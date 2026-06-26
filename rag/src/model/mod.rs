pub mod document;
pub mod chunk;
pub mod memory;

pub fn chrono_now() -> String {
    let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    let secs = d.as_secs();
    format!("{secs}")
}


/// Placeholder memory block builder — stores chunk text as fallback.
/// Real extraction is done via LLM in the lala CLI ingest pipeline.
///
/// Returns (facts, capabilities, constraints).
pub fn build_memory_block(chunk: &str) -> (String, String, String) {
    let chunk_text = chunk.to_string();
    (chunk_text.clone(), chunk_text.clone(), chunk_text)
}