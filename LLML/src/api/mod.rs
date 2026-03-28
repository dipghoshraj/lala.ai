use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tracing::{error, info, instrument, warn};
use axum::{
    Router,
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json,
};
use serde::{Deserialize, Serialize};

use crate::model::ModelRegistry;

// ── OpenAI-compatible request/response types ─────────────────────────────────

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    /// "system" | "user" | "assistant"
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    /// The model role to use: "reasoning" | "decision" (or any registered role).
    /// Defaults to the first registered model when not supplied.
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    /// Overrides the model-config default when provided.
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: ChatUsage,
}

// ── /v1/models response ───────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
}

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

// ── Prompt builder ────────────────────────────────────────────────────────────

/// Conservative token estimator: ~3 bytes per token.
/// Actual BPE token counts are usually lower, so this errs on the safe side.
#[inline]
fn estimate_tokens(text: &str) -> usize {
    (text.len() + 2) / 3
}

/// Trim the oldest non-system turns until the estimated prompt token count
/// fits within the context budget: `n_ctx - max_tokens - safety_margin`.
///
/// Rules:
///   - System messages are always kept.
///   - Turns are dropped oldest-first in pairs (user + assistant) to keep
///     the conversation structure syntactically valid.
///   - The final user message is never dropped — at worst the system prompt
///     + one user message is all that remains.
fn slide_messages(
    messages: &[ChatMessage],
    n_ctx: u32,
    max_tokens: usize,
) -> Vec<ChatMessage> {
    // Reserve space for generation output + a small safety margin.
    const SAFETY_MARGIN: usize = 32;
    let budget = (n_ctx as usize)
        .saturating_sub(max_tokens)
        .saturating_sub(SAFETY_MARGIN);

    let (system, mut turns): (Vec<_>, Vec<_>) = messages
        .iter()
        .cloned()
        .partition(|m| m.role == "system");

    loop {
        let candidate: Vec<ChatMessage> = system
            .iter()
            .chain(turns.iter())
            .cloned()
            .collect();

        let prompt = build_prompt(&candidate);
        let estimated = estimate_tokens(&prompt);

        if estimated <= budget {
            return candidate;
        }

        // Nothing left to drop — return what we have and let the model
        // report the error rather than silently mangling the request.
        if turns.len() <= 1 {
            warn!(
                estimated,
                budget,
                "context still over budget after full slide; proceeding anyway"
            );
            return candidate;
        }

        // Drop the oldest pair (user + assistant) when possible, otherwise
        // just the oldest single message, to preserve conversation structure.
        if turns[0].role == "user"
            && turns.len() > 1
            && turns[1].role == "assistant"
        {
            let dropped = turns.drain(0..2).count();
            warn!(dropped, remaining = turns.len(), "sliding context window: dropped turn pair");
        } else {
            turns.drain(0..1);
            warn!(remaining = turns.len(), "sliding context window: dropped single message");
        }
    }
}

/// Converts an OpenAI `messages` array into a Mistral/Llama `[INST]` prompt.
pub fn build_prompt(messages: &[ChatMessage]) -> String {
    let mut result = String::new();

    let (system_opt, turns) = match messages.first() {
        Some(m) if m.role == "system" => (Some(m.content.as_str()), &messages[1..]),
        _ => (None, messages),
    };

    let mut iter = turns.iter();
    let mut first_user = true;

    while let Some(msg) = iter.next() {
        match msg.role.as_str() {
            "user" => {
                if first_user {
                    match system_opt {
                        Some(sys) => result.push_str(&format!(
                            "<s>[INST] {}\n\n{} [/INST]",
                            sys, msg.content
                        )),
                        None => result
                            .push_str(&format!("<s>[INST] {} [/INST]", msg.content)),
                    }
                    first_user = false;
                } else {
                    result.push_str(&format!("[INST] {} [/INST]", msg.content));
                }
            }
            "assistant" => {
                result.push_str(&format!(" {} </s>", msg.content));
            }
            _ => {}
        }
    }

    result
}

// ── Router ────────────────────────────────────────────────────────────────────

pub fn create_router(registry: Arc<ModelRegistry>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .with_state(registry)
}

// ── Handlers ──────────────────────────────────────────────────────────────────

/// Returns the list of registered model roles (OpenAI-compatible format).
async fn list_models(
    State(registry): State<Arc<ModelRegistry>>,
) -> Json<ModelListResponse> {
    let data = registry
        .roles()
        .into_iter()
        .map(|id| ModelInfo { id, object: "model".to_string() })
        .collect();

    Json(ModelListResponse {
        object: "list".to_string(),
        data,
    })
}

#[instrument(skip(registry, req), fields(model = ?req.model, message_count = req.messages.len(), max_tokens = ?req.max_tokens))]
async fn chat_completions(
    State(registry): State<Arc<ModelRegistry>>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (StatusCode, String)> {
    if req.messages.is_empty() {
        warn!("rejected request: messages array is empty");
        return Err((StatusCode::BAD_REQUEST, "messages must not be empty".into()));
    }

    let resolved_role = match &req.model {
        Some(role) => {
            if registry.get(role).is_none() {
                let available = registry.roles().join(", ");
                warn!(role, available, "unknown model role requested");
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!("unknown model role '{}'. Available: {}", role, available),
                ));
            }
            role.clone()
        }
        None => {
            registry
                .first()
                .map(|(r, _)| r.to_string())
                .ok_or_else(|| {
                    error!("no models registered in the registry");
                    (StatusCode::INTERNAL_SERVER_ERROR, "no models available".into())
                })?
        }
    };

    let max_tokens = req.max_tokens;
    let messages = req.messages;
    let temperature = req.temperature;

    info!(
        model = %resolved_role,
        message_count = messages.len(),
        max_tokens = ?max_tokens,
        "received chat completion request"
    );

    let registry_clone = Arc::clone(&registry);
    let role_clone = resolved_role.clone();
    let output = tokio::task::spawn_blocking(move || {
        let runner = registry_clone
            .get(&role_clone)
            .ok_or_else(|| anyhow::anyhow!("model '{}' disappeared from registry", role_clone))?;

        // Resolve the generation budget, then slide the context window so the
        // prompt always fits within the model's n_ctx.
        let gen_budget = max_tokens.unwrap_or_else(|| runner.max_tokens_default());
        let slid = slide_messages(&messages, runner.n_ctx(), gen_budget);
        let prompt = build_prompt(&slid);

        info!(
            original_turns = messages.len(),
            slid_turns = slid.len(),
            prompt_len = prompt.len(),
            "prompt built after context slide"
        );

        runner.generate_from_prompt(&prompt, max_tokens, temperature)
    })
    .await
    .map_err(|e| {
        error!(error = %e, "spawn_blocking join error");
        (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })?
    .map_err(|e| {
        error!(error = %e, "inference error");
        (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })?;

    info!(output_len = output.len(), model = %resolved_role, "inference complete");

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Ok(Json(ChatResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created,
        model: resolved_role,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: output,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: ChatUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    }))
}
