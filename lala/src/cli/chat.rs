use crate::agent::model::{ApiClient, ChatMessage, RouteDecision};
use crate::agent::planner::{Agent, needs_reasoning, limit_chunks_by_tokens, limit_memory_by_tokens};
use crate::config::LalaConfig;
use rag::RagStore;
use std::collections::HashMap;

use super::display;

/// Owns conversation history and drives the chat pipeline.
pub struct Chat<'a> {
    agent: Agent<'a>,
    smart_router: bool,
    history: Vec<ChatMessage>,
    project_histories: HashMap<Option<String>, Vec<ChatMessage>>,
    current_project_id: Option<String>,
    system_prompt: String,
}

impl<'a> Chat<'a> {
    fn context_token_budget() -> usize {
        std::env::var("LALA_CONTEXT_TOKEN_BUDGET")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(|| Agent::context_token_budget())
    }

    fn chunk_token_budget(context_budget: usize) -> usize {
        // Keep roughly 2/3 for chunk text and 1/3 for structured memory.
        (context_budget * 2) / 3
    }

    fn memory_token_budget(context_budget: usize) -> usize {
        context_budget - Self::chunk_token_budget(context_budget)
    }

    pub fn new(client: &'a ApiClient, smart_router: bool, store: &'a RagStore, config: LalaConfig) -> Self {
        let current_project_id = store.current_project_id();
        let system_prompt = config.system_prompt.clone();
        let history = vec![ChatMessage {
            role: "system".to_string(),
            content: system_prompt.clone(),
        }];
        let mut project_histories = HashMap::new();
        project_histories.insert(current_project_id.clone(), history.clone());

        Self {
            agent: Agent::new(client, store, config),
            smart_router,
            history,
            project_histories,
            current_project_id,
            system_prompt,
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
        self.sync_project_history();

        let is_plan = Self::strip_plan_prefix(input).is_some();
        let user_content = if let Some(stripped) = Self::strip_plan_prefix(input) {
            stripped.to_string()
        } else {
            input.to_string()
        };

        self.history.push(ChatMessage {
            role: "user".to_string(),
            content: user_content.clone(),
        });

        if is_plan {
            if self.current_project_id.is_none() {
                display::error("Plan mode requires a selected project. Use /project select <name-or-id> first.");
                self.history.pop();
                return;
            }
            self.run_planning();
            return;
        }

        let route = self.classify(input);

        match route {
            RouteDecision::Direct => self.run_direct(),
            RouteDecision::Reasoning => self.run_reasoning(),
            RouteDecision::Metadata => self.run_metadata(),
        }
    }

    fn strip_plan_prefix(input: &str) -> Option<&str> {
        let trimmed = input.trim_start();
        if trimmed.len() >= 5 && trimmed[..5].eq_ignore_ascii_case("plan:") {
            let remainder = trimmed[5..].trim_start();
            if remainder.is_empty() {
                None
            } else {
                Some(remainder)
            }
        } else {
            None
        }
    }

    // ── Internal ──────────────────────────────────────────────────────────

    fn classify(&self, input: &str) -> RouteDecision {
        if self.smart_router {
            self.agent.classify_query(input, &self.history)
        } else if self.agent.current_project_id().is_some() && self.is_metadata_query(input) {
            RouteDecision::Metadata
        } else if needs_reasoning(input) {
            RouteDecision::Reasoning
        } else {
            RouteDecision::Direct
        }
    }

    fn is_metadata_query(&self, input: &str) -> bool {
        let lower = input.to_lowercase();
        [
            "how many projects",
            "how many documents",
            "what documents",
            "what projects",
            "list documents",
            "list projects",
            "documents in this project",
            "projects i have",
            "current project",
            "selected project",
        ]
        .iter()
        .any(|pat| lower.contains(pat))
    }

    fn sync_project_history(&mut self) {
        let active_project_id = self.agent.current_project_id();
        if active_project_id == self.current_project_id {
            return;
        }

        self.project_histories
            .insert(self.current_project_id.clone(), self.history.clone());

        self.history = self
            .project_histories
            .remove(&active_project_id)
            .unwrap_or_else(|| vec![ChatMessage {
                role: "system".to_string(),
                content: self.system_prompt.clone(),
            }]);

        self.current_project_id = active_project_id;
    }

    fn preserve_current_history(&mut self) {
        self.project_histories
            .insert(self.current_project_id.clone(), self.history.clone());
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

        match self.agent.run_direct_stream(&self.history, context_str.as_deref()) {
            Ok(stream) => match display::print_section_stream("Answer", display::BOLD_CYAN, display::CYAN, stream) {
                Ok(reply) => {
                    self.history.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: reply,
                    });
                    self.preserve_current_history();
                }
                Err(e) => {
                    display::error(&format!("Error streaming answer: {e}"));
                    self.history.pop();
                }
            },
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

        match self.agent.run_reasoning_stream(&self.history, ctx_ref) {
            Err(e) => {
                display::error(&format!("Reasoning failed: {e}"));
                self.history.pop();
            }
            Ok(stream) => match display::print_section_stream("Reasoning", display::BOLD_YELLOW, display::DIM_YELLOW, stream) {
                Err(e) => {
                    display::error(&format!("Error streaming reasoning: {e}"));
                    self.history.pop();
                }
                Ok(analysis) => {
                    match self.agent.run_decision_stream(&self.history, &analysis, ctx_ref) {
                        Err(e) => {
                            display::error(&format!("Decision failed: {e}"));
                            self.history.pop();
                        }
                        Ok(stream) => match display::print_section_stream("Answer", display::BOLD_CYAN, display::CYAN, stream) {
                            Ok(reply) => {
                                self.history.push(ChatMessage {
                                    role: "assistant".to_string(),
                                    content: reply,
                                });
                                self.preserve_current_history();
                            }
                            Err(e) => {
                                display::error(&format!("Error streaming answer: {e}"));
                                self.history.pop();
                            }
                        },
                    }
                }
            },
        }
    }

    fn run_metadata(&mut self) {
        let input = match self.history.iter().rfind(|m| m.role == "user") {
            Some(m) => m.content.clone(),
            None => {
                display::error("No user message found.");
                return;
            }
        };

        let lower = input.to_lowercase();
        if self.agent.current_project_id().is_none()
            && (lower.contains("this project")
                || lower.contains("current project")
                || lower.contains("selected project"))
        {
            display::error("No project selected. Use /project select <name-or-id> or /project create <name> first.");
            self.history.pop();
            return;
        }

        let metadata = match self.agent.build_metadata_facts(&input) {
            Ok(facts) => facts,
            Err(e) => {
                display::error(&format!("Failed to build metadata facts: {e}"));
                self.history.pop();
                return;
            }
        };

        match self.agent.run_metadata_stream(&self.history, &metadata) {
            Ok(stream) => match display::print_section_stream("Answer", display::BOLD_CYAN, display::CYAN, stream) {
                Ok(reply) => {
                    self.history.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: reply,
                    });
                    self.preserve_current_history();
                }
                Err(e) => {
                    display::error(&format!("Error streaming answer: {e}"));
                    self.history.pop();
                }
            },
            Err(e) => {
                display::error(&format!("Metadata answer failed: {e}"));
                self.history.pop();
            }
        }
    }

    fn run_planning(&mut self) {
        let input = match self.history.iter().rfind(|m| m.role == "user") {
            Some(m) => m.content.clone(),
            None => {
                display::error("No user message found for planning.");
                return;
            }
        };

        let (context_str, limited_chunks, limited_memory) = self.retrieve_and_limit_context(&input);

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
        match self.agent.run_planning_stream(&self.history, ctx_ref) {
            Err(e) => {
                display::error(&format!("Plan generation failed: {e}"));
                self.history.pop();
            }
            Ok(stream) => match display::print_section_stream("Plan", display::BOLD_YELLOW, display::DIM_YELLOW, stream) {
                Ok(plan_text) => {
                    self.history.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: plan_text,
                    });
                    self.preserve_current_history();
                }
                Err(e) => {
                    display::error(&format!("Error streaming plan: {e}"));
                    self.history.pop();
                }
            },
        }
    }

    /// Retrieve and limit RAG context by token budget.
    /// Returns: (context_string, limited_chunks, limited_memory_blocks)
    fn retrieve_and_limit_context(&self, query: &str) -> (Option<String>, Vec<rag::model::chunk::ChunkRow>, Vec<rag::model::memory::MemoryBlock>) {
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

        let context_budget = Self::context_token_budget();
        let limited_chunks = limit_chunks_by_tokens(chunks.clone(), Self::chunk_token_budget(context_budget));
        let limited_memory = limit_memory_by_tokens(memory_blocks.clone(), Self::memory_token_budget(context_budget));

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
