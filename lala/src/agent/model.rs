use serde::{Deserialize, Serialize};

/// A single message in the OpenAI-style conversation.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    /// The model role to invoke on the LLML server: "reasoning" | "decision".
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    messages: &'a [ChatMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: AssistantMessage,
}

#[derive(Debug, Deserialize)]
struct AssistantMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

/// Routing decision returned by the LLML classify endpoint or the local fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteDecision {
    Direct,
    Reasoning,
}

impl RouteDecision {
    /// Convert the raw string returned by the server ("direct" | "reasoning").
    /// Anything not recognised defaults to `Reasoning` (safe fail-closed).
    fn from_str(s: &str) -> Self {
        if s.trim().eq_ignore_ascii_case("direct") {
            RouteDecision::Direct
        } else {
            RouteDecision::Reasoning
        }
    }
}

/// Logical model roles exposed by the LLML server.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelRole {
    Reasoning,
    Decision,
}

impl ModelRole {
    pub fn as_str(self) -> &'static str {
        match self {
            ModelRole::Reasoning => "reasoning",
            ModelRole::Decision => "decision",
        }
    }
}

#[derive(Debug, Serialize)]
struct ClassifyRequest<'a> {
    query: &'a str,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    context: &'a [ChatMessage],
}

#[derive(Debug, Deserialize)]
struct ClassifyResponse {
    route: String,
    // confidence field exists in the response but we don't need it client-side
    #[allow(dead_code)]
    confidence: String,
}

/// HTTP client that talks to the LLML API server.
pub struct ApiClient {
    client: reqwest::blocking::Client,
    base_url: String,
}

impl ApiClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            client: reqwest::blocking::Client::builder()
                // No timeout — inference can take a while on CPU.
                .timeout(None)
                .build()
                .expect("failed to build HTTP client"),
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// Send a chat request with the full conversation history and return
    /// the assistant's reply text.
    ///
    /// `model_role` selects which model to use on the LLML server.
    /// Pass `None` to let the server choose the default (first registered) model.
    pub fn chat(
        &self,
        messages: &[ChatMessage],
        max_tokens: Option<usize>,
        model_role: Option<ModelRole>,
    ) -> anyhow::Result<String> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let role_str = model_role.map(|r| r.as_str());
        let body = ChatRequest { model: role_str, messages, max_tokens };

        let resp: ChatResponse = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .map_err(|e| anyhow::anyhow!("request failed: {e}"))?
            .error_for_status()
            .map_err(|e| anyhow::anyhow!("server error: {e}"))?
            .json()
            .map_err(|e| anyhow::anyhow!("invalid response: {e}"))?;

        resp.choices
            .into_iter()
            .next()
            .map(|c| c.message.content.trim().to_string())
            .ok_or_else(|| anyhow::anyhow!("empty choices in API response"))
    }

    /// Convenience wrapper — uses the `reasoning` model.
    pub fn reason(
        &self,
        messages: &[ChatMessage],
        max_tokens: Option<usize>,
    ) -> anyhow::Result<String> {
        self.chat(messages, max_tokens, Some(ModelRole::Reasoning))
    }

    /// Convenience wrapper — uses the `decision` model.
    pub fn decide(
        &self,
        messages: &[ChatMessage],
        max_tokens: Option<usize>,
    ) -> anyhow::Result<String> {
        self.chat(messages, max_tokens, Some(ModelRole::Decision))
    }

    /// Call the LLML `/v1/classify` endpoint to get a routing decision.
    ///
    /// `context` should be the last ≤ 2 conversation turns so the server can
    /// handle follow-up queries correctly (e.g. "why?" after a complex answer).
    ///
    /// On any network or parse error the caller should fall back to the local
    /// heuristic via `needs_reasoning()` — this method does *not* swallow the
    /// error so the caller controls the fallback strategy.
    pub fn classify(
        &self,
        query: &str,
        context: &[ChatMessage],
    ) -> anyhow::Result<RouteDecision> {
        let url = format!("{}/v1/classify", self.base_url);
        let body = ClassifyRequest { query, context };

        let resp: ClassifyResponse = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .map_err(|e| anyhow::anyhow!("classify request failed: {e}"))?
            .error_for_status()
            .map_err(|e| anyhow::anyhow!("classify server error: {e}"))?
            .json()
            .map_err(|e| anyhow::anyhow!("classify invalid response: {e}"))?;

        Ok(RouteDecision::from_str(&resp.route))
    }
}
