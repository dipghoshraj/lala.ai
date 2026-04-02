use crate::agent::model::{ApiClient, ChatMessage};
use rag::MemoryExtractor;

/// LLM-based memory extractor using LLML for structured knowledge extraction.
pub struct LlmMemoryExtractor<'a> {
    client: &'a ApiClient,
}

impl<'a> LlmMemoryExtractor<'a> {
    /// Create a new LLM memory extractor.
    pub fn new(client: &'a ApiClient) -> Self {
        Self { client }
    }
}

impl<'a> MemoryExtractor for LlmMemoryExtractor<'a> {
    /// Extract FACTS, CAPABILITIES, CONSTRAINTS from chunk text via LLM.
    fn extract_memory(&self, chunk_text: &str) -> anyhow::Result<(String, String, String)> {
        // Early exit: skip whitespace-only chunks
        let trimmed = chunk_text.trim();
        if trimmed.is_empty() {
            eprintln!("[LLM Memory Extraction] SKIPPED: chunk is empty/whitespace-only");
            return Ok((String::new(), String::new(), String::new()));
        }

        const MEMORY_SYSTEM: &str = 
"You are a deterministic memory extraction engine.

Extract structured knowledge from the input text.

RULES:
- Use ONLY information explicitly present in the text
- Do NOT infer or add external knowledge
- Keep each item atomic (one idea per string)
- Keep items short and precise
- No duplicates

IGNORE:
- Code blocks
- Shell commands
- File trees
- Config snippets

If the input is not meaningful, return empty arrays.

OUTPUT:
Return ONLY valid JSON. No explanation. No extra text.

SCHEMA:
{
  \"facts\": [\"string\"],
  \"capabilities\": [\"string\"],
  \"constraints\": [\"string\"]
}";

        // Log the request
        eprintln!("\n[LLM Memory Extraction] ==== REQUEST ====");
        eprintln!("[SYSTEM PROMPT]");
        for (idx, line) in MEMORY_SYSTEM.lines().enumerate() {
            eprintln!("  {}", line);
            if idx >= 15 {
                eprintln!("  ... (truncated, {} total lines)", MEMORY_SYSTEM.lines().count());
                break;
            }
        }
        
        eprintln!("\n[USER INPUT] (first 200 chars)");
        let preview: String = chunk_text.chars().take(200).collect();
        eprintln!("  {}{}", preview, if chunk_text.len() > 200 { "..." } else { "" });
        eprintln!("  [total chunk size: {} chars]", chunk_text.len());

        let messages: Vec<ChatMessage> = vec![
            ChatMessage {
                role: "system".to_string(),
                content: MEMORY_SYSTEM.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: format!("Extract memory blocks from:\n{}", chunk_text),
            },
        ];

        // Call LLM
        eprintln!("\n[CALLING LLML]");
        let response = self.client.reason(&messages, Some(512))?;
        eprintln!("[LLM RESPONSE] ------ Raw Output ------");
        for (_line_idx, line) in response.lines().enumerate() {
            eprintln!("  {}", line);
        }
        eprintln!("------ End Raw Output ------\n");

        // Parse JSON response
        let mut facts = String::new();
        let mut capabilities = String::new();
        let mut constraints = String::new();

        eprintln!("[PARSING JSON RESPONSE]");
        
        // Try to parse as JSON
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&response) {
            // Extract facts array
            if let Some(facts_arr) = parsed.get("facts").and_then(|v| v.as_array()) {
                let fact_strs: Vec<String> = facts_arr
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                facts = fact_strs.join("; ");
                eprintln!("  ✓ Parsed {} facts", fact_strs.len());
            }

            // Extract capabilities array
            if let Some(cap_arr) = parsed.get("capabilities").and_then(|v| v.as_array()) {
                let cap_strs: Vec<String> = cap_arr
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                capabilities = cap_strs.join("; ");
                eprintln!("  ✓ Parsed {} capabilities", cap_strs.len());
            }

            // Extract constraints array
            if let Some(con_arr) = parsed.get("constraints").and_then(|v| v.as_array()) {
                let con_strs: Vec<String> = con_arr
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                constraints = con_strs.join("; ");
                eprintln!("  ✓ Parsed {} constraints", con_strs.len());
            }
        } else {
            eprintln!("  ✗ Failed to parse JSON, attempting fallback parse");
        }

        // Fallback: if JSON parsing entirely failed, use chunk text
        if facts.is_empty() && capabilities.is_empty() && constraints.is_empty() {
            eprintln!("  ⚠ All fields empty after parse, using chunk text as fallback");
            facts = chunk_text.to_string();
            capabilities = "(none identified)".to_string();
            constraints = "(none identified)".to_string();
        }

        eprintln!("\n[EXTRACTION RESULT]");
        eprintln!("  FACTS ({} chars): {}", facts.len(), if facts.len() > 100 { format!("{}...", facts.chars().take(100).collect::<String>()) } else { facts.clone() });
        eprintln!("  CAPABILITIES ({} chars): {}", capabilities.len(), if capabilities.len() > 100 { format!("{}...", capabilities.chars().take(100).collect::<String>()) } else { capabilities.clone() });
        eprintln!("  CONSTRAINTS ({} chars): {}", constraints.len(), if constraints.len() > 100 { format!("{}...", constraints.chars().take(100).collect::<String>()) } else { constraints.clone() });
        eprintln!("[END Memory Extraction] ==================\n");

        Ok((facts, capabilities, constraints))
    }
}
