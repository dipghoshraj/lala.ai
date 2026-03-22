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
}
