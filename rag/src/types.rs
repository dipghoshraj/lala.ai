use anyhow::Result;

/// A retrieved chunk with its BM25 relevance score.
#[derive(Clone)]
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

/// SQLite FTS5-backed document store for keyword (BM25) retrieval.
/// Trait for LLM-based memory extraction (unused in standard ingestion path).
///
/// Kept for compatibility, but CLI ingestion no longer invokes it.
pub trait MemoryExtractor {
    fn extract_memory(&self, chunk_text: &str) -> Result<(String, String, String)>;
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
        text.to_lowercase().contains("how ")
    {
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
    if text.contains("import ") || (text.contains("from ") && text.contains("import")) {
        code_score += 2; // Import statements
    }
    if text.contains("```") || (text.contains("    ") && text.matches("    ").count() > 3) {
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
pub fn build_memory_block(chunk: &str) -> (String, String, String) {
    let chunk_text = chunk.to_string();
    (chunk_text.clone(), chunk_text.clone(), chunk_text)
}

/// Simple ISO-8601 timestamp without pulling in chrono.
pub fn chrono_now() -> String {
    let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    let secs = d.as_secs();
    format!("{secs}")
}
