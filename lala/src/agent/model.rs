use serde::{Deserialize, Serialize};

/// A single message in the OpenAI-style conversation.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
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
    pub fn chat(
        &self,
        messages: &[ChatMessage],
        max_tokens: Option<usize>,
    ) -> anyhow::Result<String> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = ChatRequest { messages, max_tokens };

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
}
