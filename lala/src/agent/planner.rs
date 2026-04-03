use crate::agent::model::{ApiClient, ChatMessage, RouteDecision};
use crate::config::LalaConfig;
use rag::RagStore;

// ── Token estimation ──────────────────────────────────────────────────────────

/// Estimate tokens using ~3 bytes per token (same as LLML server).
fn estimate_tokens(text: &str) -> usize {
    (text.len() as f32 / 3.0).ceil() as usize
}

/// The fallback context token budget when no env var is set.
pub const DEFAULT_CONTEXT_TOKEN_BUDGET: usize = 800;

/// Default maximum chunk results to retrieve from RAG (before token limiting).
pub const DEFAULT_RAG_FETCH_LIMIT: usize = 32;

/// Minimum chunk results to retrieve.
const MIN_RAG_FETCH_LIMIT: usize = 5;

/// Maximum chunk results to retrieve to avoid excessive query payload.
const MAX_RAG_FETCH_LIMIT: usize = 200;

/// Limit retrieved chunks to fit within a token budget.
/// Returns the chunks that fit, ordered by relevance (highest first).
pub fn limit_chunks_by_tokens(chunks: Vec<rag::Chunk>, max_tokens: usize) -> Vec<rag::Chunk> {
    let mut result = Vec::new();
    let mut used_tokens = 0;

    for chunk in chunks {
        let chunk_tokens = estimate_tokens(&chunk.chunk_text);
        if used_tokens + chunk_tokens > max_tokens {
            break;
        }
        used_tokens += chunk_tokens;
        result.push(chunk);
    }

    result
}

/// Limit retrieved memory blocks to fit within a token budget.
pub fn limit_memory_by_tokens(blocks: Vec<rag::MemoryBlock>, max_tokens: usize) -> Vec<rag::MemoryBlock> {
    let mut result = Vec::new();
    let mut used_tokens = 0;

    for block in blocks {
        let block_tokens = estimate_tokens(&block.facts)
            + estimate_tokens(&block.capabilities)
            + estimate_tokens(&block.constraints);
        if used_tokens + block_tokens > max_tokens {
            break;
        }
        used_tokens += block_tokens;
        result.push(block);
    }

    result
}

// ── Query classifier ──────────────────────────────────────────────────────────

/// Greeting / social phrases that never warrant reasoning.
const DIRECT_PATTERNS: &[&str] = &[
    "hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye",
    "good morning", "good evening", "good night", "good afternoon",
    "ok", "okay", "sure", "yes", "no", "great", "perfect", "nice",
    "cool", "awesome", "got it", "understood",
];

/// Keywords that signal the query needs multi-step reasoning.
const REASONING_TRIGGERS: &[&str] = &[
    "why", "how", "explain", "analyze", "analyse", "compare",
    "difference", "what if", "implement", "write", "debug", "fix",
    "code", "algorithm", "calculate", "evaluate", "pros", "cons",
    "summarize", "summarise", "describe", "define", "plan", "design",
    "architecture", "step", "process", "reasoning", "derive", "prove",
    "optimise", "optimize", "refactor", "suggest", "recommend",
];

/// Returns `true` if the query warrants running through the reasoning step.
///
/// Routing logic (in priority order):
/// 1. Matches a direct/social pattern → `false`
/// 2. ≤ 3 words and no reasoning trigger → `false`
/// 3. Contains a reasoning trigger keyword → `true`
/// 4. ≤ 8 words and no trigger → `false`
/// 5. Longer queries default to reasoning → `true`
pub fn needs_reasoning(input: &str) -> bool {
    let lower = input.trim().to_lowercase();

    // 1 — social / greeting patterns are always direct
    for pat in DIRECT_PATTERNS {
        if lower == *pat || lower.starts_with(&format!("{} ", pat)) {
            return false;
        }
    }

    let word_count = input.split_whitespace().count();

    // 2 — very short queries without a trigger go direct
    if word_count <= 3 && !REASONING_TRIGGERS.iter().any(|t| lower.contains(t)) {
        return false;
    }

    // 3 — explicit reasoning trigger present
    if REASONING_TRIGGERS.iter().any(|t| lower.contains(t)) {
        return true;
    }

    // 4 — medium queries with no trigger go direct
    if word_count <= 8 {
        return false;
    }

    // 5 — longer queries default to reasoning
    true
}

// ─────────────────────────────────────────────────────────────────────────────

/// Drives a single user turn through the two-step reasoning→decision pipeline.
///
/// `run_reasoning` and `run_decision` are exposed as separate public steps so
/// the CLI can display each phase (with its own spinner) as it completes.
/// `run` is a convenience wrapper that executes both steps in sequence.
pub struct Agent<'a> {
    client: &'a ApiClient,
    store: &'a RagStore,
    config: LalaConfig,
}

impl<'a> Agent<'a> {
    pub fn new(client: &'a ApiClient, store: &'a RagStore, config: LalaConfig) -> Self {
        Self { client, store, config }
    }

    /// Returns the context token budget used by the CLI for RAG context injection.
    /// Can be overridden via env var `LALA_CONTEXT_TOKEN_BUDGET`.
    pub fn context_token_budget() -> usize {
        std::env::var("LALA_CONTEXT_TOKEN_BUDGET")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_CONTEXT_TOKEN_BUDGET)
    }

    /// Compute the number of chunk results to request from RAG based on context size.
    /// Can be overridden via `LALA_RAG_FETCH_LIMIT`.
    pub fn rag_fetch_limit() -> usize {
        std::env::var("LALA_RAG_FETCH_LIMIT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .or_else(|| {
                // Use context window (including 2048 etc) to choose a safer retrieval size.
                let context_tokens = std::env::var("LALA_N_CTX")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok());
                context_tokens.map(|ct| (ct / 64).max(MIN_RAG_FETCH_LIMIT).min(MAX_RAG_FETCH_LIMIT))
            })
            .unwrap_or(DEFAULT_RAG_FETCH_LIMIT)
    }

    /// Retrieve relevant chunks from the RAG store for the given query.
    /// Returns the matched chunks, or an empty vec if nothing matched.
    pub fn retrieve_context(&self, query: &str) -> anyhow::Result<Vec<rag::Chunk>> {
        // Strip characters that are special in FTS5 query syntax.
        let sanitized: String = query
            .chars()
            .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
            .collect();
        // Join terms with OR so FTS5 matches any word (not all).
        // This gives much better recall for conversational queries like
        // "explain me about Lala front end" → "explain OR me OR about OR Lala OR front OR end".
        let terms: Vec<&str> = sanitized.split_whitespace().collect();
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        let fts_query = terms.join(" OR ");
        self.store.retrieve(&fts_query, Self::rag_fetch_limit())
    }

    /// Retrieve structured memory blocks for the given query.
    pub fn retrieve_memory_context(&self, query: &str) -> anyhow::Result<Vec<rag::MemoryBlock>> {
        let sanitized: String = query
            .chars()
            .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
            .collect();
        let terms: Vec<&str> = sanitized.split_whitespace().collect();
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        let fts_query = terms.join(" OR ");
        self.store
            .retrieve_memory_blocks(&fts_query, Self::rag_fetch_limit())
    }

    /// Step 1 — send the full history to the reasoning model.
    /// Returns the internal analysis string; does not modify history.
    /// When `context` is provided, it is appended to the reasoning system prompt.
    pub fn run_reasoning(
        &self,
        history: &[ChatMessage],
        context: Option<&str>,
    ) -> anyhow::Result<String> {
        let base = &self.config.reasoning_system_prompt;
        let system = match context {
            Some(ctx) => format!(
                "{}\n\n--- Retrieved Context ---\n{}\n--- End Context ---\n\n\
                 Use the retrieved context above to inform your analysis when relevant.",
                base, ctx
            ),
            None => base.clone(),
        };
        let reasoning_history = Self::replace_system(history, &system);
        self.client.reason(&reasoning_history, Some(512))
    }

    /// Step 2 — send a compact context (system + analysis + last user message)
    /// to the decision model. Returns the final answer string.
    /// When `context` is provided, it is appended to the decision system prompt.
    pub fn run_decision(
        &self,
        history: &[ChatMessage],
        analysis: &str,
        context: Option<&str>,
    ) -> anyhow::Result<String> {
        let base = &self.config.decision_system_prompt;
        let system = match context {
            Some(ctx) => format!(
                "{}\n\n--- Retrieved Context ---\n{}\n--- End Context ---\n\n\
                 Use the retrieved context above to inform your answer when relevant.",
                base, ctx
            ),
            None => base.clone(),
        };

        let last_user = history
            .iter()
            .rfind(|m| m.role == "user")
            .map(|m| m.content.as_str())
            .unwrap_or("");

        let decision_messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: system,
            },
            ChatMessage {
                role: "system".to_string(),
                content: format!("[Internal analysis — do not quote this]\n{}", analysis),
            },
            ChatMessage {
                role: "user".to_string(),
                content: last_user.to_string(),
            },
        ];

        self.client.decide(&decision_messages, Some(256))
    }

    /// Ask the LLML server to classify the query via `POST /v1/classify`.
    ///
    /// Passes the last two history turns as context so the server can handle
    /// follow-up queries correctly (e.g. "why?" after a complex answer).
    ///
    /// Falls back to the local heuristic (`needs_reasoning`) on any error so
    /// the REPL keeps working even when the server is temporarily unreachable.
    pub fn classify_query(
        &self,
        input: &str,
        history: &[ChatMessage],
    ) -> RouteDecision {
        // Extract last ≤2 non-system turns as context for the server.
        let context: Vec<ChatMessage> = history
            .iter()
            .filter(|m| m.role != "system")
            .rev()
            .take(2)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        match self.client.classify(input, &context) {
            Ok(decision) => decision,
            Err(e) => {
                // Log silently and fall back to the local heuristic.
                eprintln!("[classify] server error, falling back to heuristic: {e}");
                if needs_reasoning(input) {
                    RouteDecision::Reasoning
                } else {
                    RouteDecision::Direct
                }
            }
        }
    }

    /// Direct path — skips reasoning and sends the full conversation history
    /// straight to the decision model under its normal system prompt.
    /// Used for simple or conversational queries classified by `classify_query()`.
    /// When `context` is provided, it is appended to the decision system prompt.
    pub fn run_direct(&self, history: &[ChatMessage], context: Option<&str>) -> anyhow::Result<String> {
        let base = &self.config.decision_system_prompt;
        let system = match context {
            Some(ctx) => format!(
                "{}\n\n--- Retrieved Context ---\n{}\n--- End Context ---\n\n\
                 Use the retrieved context above to inform your answer when relevant.",
                base, ctx
            ),
            None => base.clone(),
        };
        let decision_messages = Self::replace_system(history, &system);
        self.client.decide(&decision_messages, Some(256))
    }

    /// Returns a copy of `history` with the first `system` message replaced
    /// by `new_system`. If no system message is present, prepends it.
    fn replace_system(history: &[ChatMessage], new_system: &str) -> Vec<ChatMessage> {
        let mut out = history.to_vec();
        let new_msg = ChatMessage {
            role: "system".to_string(),
            content: new_system.to_string(),
        };
        if out.first().map(|m| m.role.as_str()) == Some("system") {
            out[0] = new_msg;
        } else {
            out.insert(0, new_msg);
        }
        out
    }
}
