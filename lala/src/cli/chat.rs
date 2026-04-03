use crate::agent::model::{ApiClient, ChatMessage, RouteDecision};
use crate::agent::planner::{Agent, needs_reasoning, limit_chunks_by_tokens, limit_memory_by_tokens};
use rag::RagStore;

use super::display;

const SYSTEM_PROMPT: &str =
    "You are a friendly AI assistant named lala. \
     Explain things clearly and naturally. \
     Respond in full sentences.";

/// Token budget for RAG context injection (out of ~2048 context window).
/// Leave room for: system prompt (~50), conversation (~400-500), generation budget (~256).
const CONTEXT_TOKEN_BUDGET: usize = 800;


/// Owns conversation history and drives the chat pipeline.
pub struct Chat<'a> {
    agent: Agent<'a>,
    smart_router: bool,
    history: Vec<ChatMessage>,
}

impl<'a> Chat<'a> {
    pub fn new(client: &'a ApiClient, smart_router: bool, store: &'a RagStore) -> Self {
        let history = vec![ChatMessage {
            role: "system".to_string(),
            content: SYSTEM_PROMPT.to_string(),
        }];

        Self {
            agent: Agent::new(client, store),
            smart_router,
            history,
        }
    }

    /// Clear conversation history, keeping only the system prompt.
    pub fn clear(&mut self) {
        self.history.truncate(1);
        display::success("Conversation cleared.");
        println!();
    }

    /// Process a user message through the routing → inference pipeline.
    pub fn handle(&mut self, input: &str) {
        self.history.push(ChatMessage {
            role: "user".to_string(),
            content: input.to_string(),
        });

        let route = self.classify(input);

        match route {
            RouteDecision::Direct => self.run_direct(),
            RouteDecision::Reasoning => self.run_reasoning(),
        }
    }

    // ── Internal ──────────────────────────────────────────────────────────

    fn classify(&self, input: &str) -> RouteDecision {
        if self.smart_router {
            self.agent.classify_query(input, &self.history)
        } else if needs_reasoning(input) {
            RouteDecision::Reasoning
        } else {
            RouteDecision::Direct
        }
    }

    fn run_direct(&mut self) {
        // Retrieve context (same as reasoning path).
        let input = match self.history.iter().rfind(|m| m.role == "user") {
            Some(m) => m.content.clone(),
            None => {
                display::error("No user message found.");
                return;
            }
        };

        let (context_str, limited_chunks, limited_memory) = self.retrieve_and_limit_context(&input);

        // Display retrieved sources if any.
        if !limited_chunks.is_empty() {
            display::print_sources(&limited_chunks);
        }
        if !limited_memory.is_empty() {
            let sep = "─".repeat(display::SECTION_WIDTH);
            println!("{}{}{}", display::DIM, sep, display::RESET);
            println!("  {}Structured Memory Blocks:{}", display::BOLD_GREEN, display::RESET);
            for block in &limited_memory {
                println!("    {}- source:{} {} chunk #{}", display::CYAN, display::RESET, block.source, block.chunk_index);
                println!("      {}FACTS:{} {}", display::CYAN, display::RESET, block.facts);
                println!("      {}CAPABILITIES:{} {}", display::CYAN, display::RESET, block.capabilities);
                println!("      {}CONSTRAINTS:{} {}", display::CYAN, display::RESET, block.constraints);
            }
            println!("{}{}{}", display::DIM, sep, display::RESET);
        }

        let result = display::with_spinner("thinking", || {
            self.agent.run_direct(&self.history, context_str.as_deref())
        });

        match result {
            Ok(reply) => {
                display::print_section("Answer", display::BOLD_CYAN, display::CYAN, &reply);
                self.history.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: reply,
                });
            }
            Err(e) => {
                display::error(&format!("Error: {e}"));
                self.history.pop();
            }
        }
    }

    fn run_reasoning(&mut self) {
        // Retrieve context from RAG store.
        let input = match self.history.iter().rfind(|m| m.role == "user") {
            Some(m) => m.content.clone(),
            None => {
                display::error("No user message found.");
                return;
            }
        };

        let (context_str, limited_chunks, limited_memory) = self.retrieve_and_limit_context(&input);

        // Display retrieved sources and memory if any were found.
        if !limited_chunks.is_empty() {
            display::print_sources(&limited_chunks);
        }
        if !limited_memory.is_empty() {
            let sep = "─".repeat(display::SECTION_WIDTH);
            println!("{}{}{}", display::DIM, sep, display::RESET);
            println!("  {}Structured Memory Blocks:{}", display::BOLD_GREEN, display::RESET);
            for block in &limited_memory {
                println!("    {}- source:{} {} chunk #{}", display::CYAN, display::RESET, block.source, block.chunk_index);
                println!("      {}FACTS:{} {}", display::CYAN, display::RESET, block.facts);
                println!("      {}CAPABILITIES:{} {}", display::CYAN, display::RESET, block.capabilities);
                println!("      {}CONSTRAINTS:{} {}", display::CYAN, display::RESET, block.constraints);
            }
            println!("{}{}{}", display::DIM, sep, display::RESET);
        }

        let ctx_ref = context_str.as_deref();

        let reasoning_result = display::with_spinner("reasoning", || {
            self.agent.run_reasoning(&self.history, ctx_ref)
        });

        match reasoning_result {
            Err(e) => {
                display::error(&format!("Reasoning failed: {e}"));
                self.history.pop();
            }
            Ok(analysis) => {
                display::print_section(
                    "Reasoning",
                    display::BOLD_YELLOW,
                    display::DIM_YELLOW,
                    &analysis,
                );

                let decision_result = display::with_spinner("deciding", || {
                    self.agent.run_decision(&self.history, &analysis, ctx_ref)
                });

                match decision_result {
                    Ok(reply) => {
                        display::print_section(
                            "Answer",
                            display::BOLD_CYAN,
                            display::CYAN,
                            &reply,
                        );
                        self.history.push(ChatMessage {
                            role: "assistant".to_string(),
                            content: reply,
                        });
                    }
                    Err(e) => {
                        display::error(&format!("Decision failed: {e}"));
                        self.history.pop();
                    }
                }
            }
        }
    }

    /// Retrieve and limit RAG context by token budget.
    /// Returns: (context_string, limited_chunks, limited_memory_blocks)
    fn retrieve_and_limit_context(&self, query: &str) -> (Option<String>, Vec<rag::Chunk>, Vec<rag::MemoryBlock>) {
        let chunks = match display::with_spinner("retrieving", || {
            self.agent.retrieve_context(query)
        }) {
            Ok(c) => c,
            Err(e) => {
                display::warn(&format!("Retrieval error: {e} — proceeding without context."));
                Vec::new()
            }
        };

        // Also retrieve structured memory blocks.
        let memory_blocks = match display::with_spinner("retrieving memory", || {
            self.agent.retrieve_memory_context(query)
        }) {
            Ok(m) => m,
            Err(e) => {
                display::warn(&format!("Memory retrieval error: {e} — proceeding without context."));
                Vec::new()
            }
        };

        // Limit both by token budget to fit context window.
        let limited_chunks = limit_chunks_by_tokens(chunks.clone(), CONTEXT_TOKEN_BUDGET / 2);
        let limited_memory = limit_memory_by_tokens(memory_blocks.clone(), CONTEXT_TOKEN_BUDGET / 2);

        // Build context string for LLM injection using token-limited results.
        let context_str = if limited_chunks.is_empty() && limited_memory.is_empty() {
            None
        } else {
            let mut ctx = String::new();
            if !limited_chunks.is_empty() {
                ctx.push_str("--- Retrieved Chunks ---\n");
                ctx.push_str(
                    &limited_chunks
                        .iter()
                        .map(|c| c.chunk_text.as_str())
                        .collect::<Vec<_>>()
                        .join("\n---\n"),
                );
                ctx.push_str("\n");
            }
            if !limited_memory.is_empty() {
                ctx.push_str("--- Retrieved Structured Memory Blocks ---\n");
                for block in &limited_memory {
                    ctx.push_str(&format!(
                        "FACTS: {}\nCAPABILITIES: {}\nCONSTRAINTS: {}\nTEXT: {}\n---\n",
                        block.facts,
                        block.capabilities,
                        block.constraints,
                        block.chunk_text
                    ));
                }
            }
            Some(ctx)
        };

        (context_str, limited_chunks, limited_memory)
    }
}
