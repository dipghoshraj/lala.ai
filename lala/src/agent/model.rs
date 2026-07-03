use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, BufReader};

/// A single message in the OpenAI-style conversation.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    /// Optional model name to invoke on the LLML server.
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    messages: &'a [ChatMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
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

#[derive(Debug, Deserialize)]
struct SseChoiceDelta {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SseChoice {
    delta: SseChoiceDelta,
}

#[derive(Debug, Deserialize)]
struct SseChatChunk {
    choices: Vec<SseChoice>,
}

/// Blocking SSE reader over a streaming chat completion response, or a direct
/// non-streaming fallback payload.
pub struct ChatStream {
    source: ChatStreamSource,
}

enum ChatStreamSource {
    Sse(io::Lines<Box<dyn BufRead>>),
    Plain { content: String, index: usize },
}

impl ChatStream {
    fn parse_event(&mut self) -> anyhow::Result<Option<String>> {
        match &mut self.source {
            ChatStreamSource::Sse(lines) => {
                let mut event_lines = Vec::new();

                loop {
                    let line = match lines.next() {
                        Some(Ok(line)) => line,
                        Some(Err(err)) => return Err(anyhow::anyhow!("stream read error: {err}")),
                        None => return Ok(None),
                    };

                    let trimmed = line.trim_end();
                    if trimmed.is_empty() {
                        if !event_lines.is_empty() {
                            break;
                        }
                        continue;
                    }

                    if let Some(payload) = trimmed.strip_prefix("data: ") {
                        if payload == "[DONE]" {
                            return Ok(None);
                        }
                        event_lines.push(payload.to_string());
                    }
                }

                let payload = event_lines.join("\n");
                let chunk: SseChatChunk = serde_json::from_str(&payload)
                    .map_err(|e| anyhow::anyhow!("invalid SSE chunk JSON: {e} payload={payload}"))?;

                let content = chunk
                    .choices
                    .get(0)
                    .and_then(|choice| choice.delta.content.clone())
                    .unwrap_or_default();

                if content.is_empty() {
                    self.parse_event()
                } else {
                    Ok(Some(content))
                }
            }
            ChatStreamSource::Plain { content, index } => {
                if *index >= content.len() {
                    return Ok(None);
                }
                let remainder = content[*index..].to_string();
                *index = content.len();
                Ok(Some(remainder))
            }
        }
    }
}

impl Iterator for ChatStream {
    type Item = anyhow::Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.parse_event() {
            Ok(Some(token)) => Some(Ok(token)),
            Ok(None) => None,
            Err(err) => Some(Err(err)),
        }
    }
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

    /// Create a streaming chat completion response.
    pub fn chat_stream<'a>(
        &self,
        messages: &[ChatMessage],
        max_tokens: Option<usize>,
        temperature: Option<f32>,
        model: Option<&'a str>,
    ) -> anyhow::Result<ChatStream> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = ChatRequest { model, messages, max_tokens, temperature, stream: true };

        let resp = self
            .client
            .post(&url)
            .header("Accept", "text/event-stream")
            .json(&body)
            .send()
            .map_err(|e| anyhow::anyhow!("request failed: {e}"))?
            .error_for_status()
            .map_err(|e| anyhow::anyhow!("server error: {e}"))?;

        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        if !content_type.contains("text/event-stream") {
            let response: ChatResponse = resp
                .json()
                .map_err(|e| anyhow::anyhow!("invalid response while falling back from stream: {e}"))?;
            let content = response
                .choices
                .into_iter()
                .next()
                .map(|c| c.message.content.trim().to_string())
                .ok_or_else(|| anyhow::anyhow!("empty choices in fallback response"))?;
            return Ok(ChatStream {
                source: ChatStreamSource::Plain { content, index: 0 },
            });
        }

        let reader: Box<dyn BufRead> = Box::new(BufReader::new(resp));
        Ok(ChatStream {
            source: ChatStreamSource::Sse(reader.lines()),
        })
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
