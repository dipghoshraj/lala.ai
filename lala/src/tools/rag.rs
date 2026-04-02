use tools::Tool;
use anyhow::{Result, anyhow, Context};
use serde_json::Value;
use rag::RagStore;

/// Tool for searching the RAG knowledge base
pub struct RagSearchTool {
    pub db_path: String,
}

impl RagSearchTool {
    pub fn new(db_path: String) -> Self {
        RagSearchTool { db_path }
    }
}

impl Tool for RagSearchTool {
    fn name(&self) -> &str {
        "search_knowledge_base"
    }

    fn description(&self) -> &str {
        "Search the knowledge base for relevant documents and context"
    }

    fn keywords(&self) -> &[&str] {
        &["search", "find", "retrieve", "lookup", "documents", "knowledge", "query"]
    }

    fn execute(&self, input: Value) -> Result<String> {
        // Extract query from input
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing required field: query (must be a string)"))?;

        // Extract limit if provided, default to 5
        let limit = input
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as usize;

        // Open RAG store and retrieve
        let store = RagStore::open(&self.db_path)
            .with_context(|| format!("Failed to open RAG store at: {}", self.db_path))?;

        let results = store.retrieve(query, limit)
            .with_context(|| format!("Failed to search knowledge base for: {}", query))?;

        if results.is_empty() {
            return Ok("No relevant documents found in knowledge base.".to_string());
        }

        // Format results as markdown-style text
        let formatted = results
            .iter()
            .enumerate()
            .map(|(idx, chunk)| {
                format!(
                    "**Document {}:**\n{}\n",
                    idx + 1,
                    chunk.chunk_text
                )
            })
            .collect::<Vec<_>>()
            .join("\n---\n\n");

        Ok(formatted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rag_search_tool_name() {
        let tool = RagSearchTool::new("test.db".to_string());
        assert_eq!(tool.name(), "search_knowledge_base");
    }

    #[test]
    fn test_rag_search_tool_keywords() {
        let tool = RagSearchTool::new("test.db".to_string());
        let keywords = tool.keywords();
        assert!(keywords.contains(&"search"));
        assert!(keywords.contains(&"find"));
    }
}
