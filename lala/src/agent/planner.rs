use crate::agent::model::{ApiClient, ChatMessage, RouteDecision};

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

/// System prompt injected for the reasoning step.
///
/// The reasoning model's job is to silently think through the query —
/// it never speaks directly to the user.
const REASONING_SYSTEM: &str =
    "You are an internal reasoning engine. \
     Analyse the user's query carefully. \
     Think step by step: what is the user asking, what context matters, \
     and what would make the best answer. \
     Output your analysis concisely — this will be used to guide the final response, \
     not shown to the user.";

/// System prompt for the decision step.
///
/// The decision model receives the original conversation history plus the
/// reasoning output as extra context, and produces the reply the user sees.
const DECISION_SYSTEM: &str =
    "You are lala, a friendly and concise AI assistant. \
     You have been given an internal analysis to guide you. \
     Use it to inform your answer but do NOT repeat or quote it. \
     Respond directly to the user in clear, natural language.";

/// Drives a single user turn through the two-step reasoning→decision pipeline.
///
/// `run_reasoning` and `run_decision` are exposed as separate public steps so
/// the CLI can display each phase (with its own spinner) as it completes.
/// `run` is a convenience wrapper that executes both steps in sequence.
pub struct Agent<'a> {
    client: &'a ApiClient,
}

impl<'a> Agent<'a> {
    pub fn new(client: &'a ApiClient) -> Self {
        Self { client }
    }

    /// Step 1 — send the full history to the reasoning model.
    /// Returns the internal analysis string; does not modify history.
    pub fn run_reasoning(&self, history: &[ChatMessage]) -> anyhow::Result<String> {
        let reasoning_history = Self::replace_system(history, REASONING_SYSTEM);
        self.client.reason(&reasoning_history, Some(512))
    }

    /// Step 2 — send a compact context (system + analysis + last user message)
    /// to the decision model. Returns the final answer string.
    pub fn run_decision(&self, history: &[ChatMessage], analysis: &str) -> anyhow::Result<String> {
        let last_user = history
            .iter()
            .rfind(|m| m.role == "user")
            .map(|m| m.content.as_str())
            .unwrap_or("");

        let decision_messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: DECISION_SYSTEM.to_string(),
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
    pub fn run_direct(&self, history: &[ChatMessage]) -> anyhow::Result<String> {
        let decision_messages = Self::replace_system(history, DECISION_SYSTEM);
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
