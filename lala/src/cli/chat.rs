use crate::agent::model::{ApiClient, ChatMessage, RouteDecision};
use crate::agent::planner::{Agent, needs_reasoning};
use rag::RagStore;
use tools::ToolRegistry;

use super::display;

const SYSTEM_PROMPT: &str =
    "You are a friendly AI assistant named lala. \
     Explain things clearly and naturally. \
     Respond in full sentences.";

/// Owns conversation history and drives the chat pipeline.
pub struct Chat<'a> {
    agent: Agent<'a>,
    smart_router: bool,
    history: Vec<ChatMessage>,
}

impl<'a> Chat<'a> {
    pub fn new(
        client: &'a ApiClient,
        smart_router: bool,
        store: &'a RagStore,
        tool_registry: &'a ToolRegistry,
    ) -> Self {
        let history = vec![ChatMessage {
            role: "system".to_string(),
            content: SYSTEM_PROMPT.to_string(),
        }];

        Self {
            agent: Agent::with_tools(client, store, tool_registry),
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
        // First, try to run tools if applicable
        let user_input = match self.history.iter().rfind(|m| m.role == "user") {
            Some(m) => m.content.clone(),
            None => {
                display::error("No user message found.");
                return;
            }
        };

        let tool_result = display::with_spinner("", || {
            self.agent.run_tool_calls(&user_input, &self.history)
        });

        let tool_output = match tool_result {
            Ok(Some(output)) => {
                display::print_section("Retrieved Data", display::BOLD_YELLOW, display::DIM_YELLOW, &output);
                Some(output)
            }
            Ok(None) => None,
            Err(e) => {
                display::warn(&format!("Tool execution error: {e} — proceeding without tools."));
                None
            }
        };

        // Build context string that combines tool output if available
        let context_str = tool_output;
        let ctx_ref = context_str.as_deref();

        let result = display::with_spinner("thinking", || {
            let mut history_with_context = self.history.clone();
            
            // If we have tool output, inject it as a system message
            if let Some(ctx) = ctx_ref {
                history_with_context.push(ChatMessage {
                    role: "system".to_string(),
                    content: format!("[Tool output]\n{}", ctx),
                });
            }

            self.agent.run_direct(&history_with_context)
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

        // Try to run tools if applicable
        let tool_result = display::with_spinner("", || {
            self.agent.run_tool_calls(&input, &self.history)
        });

        let tool_output = match tool_result {
            Ok(Some(output)) => {
                display::print_section("Retrieved Data", display::BOLD_YELLOW, display::DIM_YELLOW, &output);
                Some(output)
            }
            Ok(None) => None,
            Err(e) => {
                display::warn(&format!("Tool execution error: {e} — proceeding without tools."));
                None
            }
        };

        let chunks = match display::with_spinner("retrieving", || {
            self.agent.retrieve_context(&input)
        }) {
            Ok(c) => c,
            Err(e) => {
                display::warn(&format!("Retrieval error: {e} — proceeding without context."));
                Vec::new()
            }
        };

        // Display retrieved sources if any were found.
        if !chunks.is_empty() {
            display::print_sources(&chunks);
        }

        // Build context string combining RAG chunks and tool output
        let mut context_parts = Vec::new();

        if let Some(tool_out) = &tool_output {
            context_parts.push(format!("[Tool output]\n{}", tool_out));
        }

        for chunk in &chunks {
            context_parts.push(chunk.chunk_text.clone());
        }

        let context_str = if context_parts.is_empty() {
            None
        } else {
            Some(context_parts.join("\n---\n"))
        };
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
}
