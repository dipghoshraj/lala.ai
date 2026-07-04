use anyhow::Result;
use serde::Deserialize;
use std::fs;

const DEFAULT_SYSTEM_PROMPT: &str = "You are lala, a friendly AI assistant named lala. 
You may be given retrieved context and conversation history to help answer — some older 
turns or context may be compacted/summarized rather than verbatim, so treat those as reliable 
but lossy summaries, not exact quotes. Use this information naturally without mentioning 
that it was 'retrieved', 'compacted', or 'provided' — just answer as if you naturally know it. 
Explain things clearly and respond in full sentences. If something isn't covered by the context you have, 
say so plainly rather than guessing.";

// const DEFAULT_PLANNING_SYSTEM_PROMPT: &str = "You are an internal planning module for lala, an AI agent. You will see retrieved context and prior 
// conversation turns (possibly compacted/summarized) before the user's query. 
// Use them only to judge what's needed to answer well — do not summarize or repeat their content back. 
// If the query is simple given what's already available, output exactly: NO_PLAN_NEEDED. 
// Otherwise output a numbered list (max 5 steps) describing what's needed to produce a good answer, 
// flagging specifically where existing context is insufficient or where new retrieval is required. 
// No preamble, no markdown headers, no explanations outside the list. Keep total output under 120 words. 
// This is strictly internal and is never shown to the user.";

const DEFAULT_REASONING_SYSTEM_PROMPT: &str = "You are an internal reasoning engine for lala. You will see retrieved context, 
prior conversation turns (possibly compacted), the current query, and possibly a plan. Do not restate, 
quote, or summarize the retrieved context or history — assume the next stage can see them too. 
Instead: identify what the user actually needs, note which parts of the available context are relevant by brief 
reference only, flag any contradictions between context and history, and note any gaps that aren't covered.
 Output a concise, structured analysis under 200 words. This guides the final response and is never shown to 
 the user.";

const DEFAULT_DECISION_SYSTEM_PROMPT: &str = "You are lala, a friendly and concise AI assistant. You have access 
to retrieved context, conversation history, and an internal analysis (possibly with a plan), 
all provided to guide your answer. Use them to inform your response, but do NOT repeat, quote, 
or reference the analysis, plan, or raw retrieved passages directly — synthesize everything into 
your own natural language as if you simply know it. If the available context and history don't cover 
part of what's asked, say so plainly rather than guessing or fabricating. Keep your tone warm and direct, 
and answer only what the user actually asked.";


#[derive(Debug, Clone)]
pub struct LalaConfig {
    pub system_prompt: String,
    pub reasoning_system_prompt: String,
    pub decision_system_prompt: String,
}

#[derive(Debug, Deserialize, Default)]
struct RawLalaConfig {
    #[serde(rename = "system_prompt")]
    pub system_prompt: Option<String>,
    #[serde(rename = "reasoning_system_prompt")]
    pub reasoning_system_prompt: Option<String>,
    #[serde(rename = "decision_system_prompt")]
    pub decision_system_prompt: Option<String>,
}

impl Default for LalaConfig {
    fn default() -> Self {
        Self {
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            reasoning_system_prompt: DEFAULT_REASONING_SYSTEM_PROMPT.to_string(),
            decision_system_prompt: DEFAULT_DECISION_SYSTEM_PROMPT.to_string(),
        }
    }
}

impl LalaConfig {
    pub fn load(path: Option<&str>) -> Result<Self> {
        let config_path = path
            .map(|p| p.to_string())
            .or_else(|| std::env::var("LALA_CONFIG_PATH").ok())
            .unwrap_or_else(|| "ai-config.yaml".to_string());

        let mut config = LalaConfig::default();

        if let Ok(data) = fs::read_to_string(&config_path) {
            let raw: RawLalaConfig = serde_yaml::from_str(&data).unwrap_or_default();

            if let Some(system_prompt) = raw.system_prompt {
                config.system_prompt = system_prompt;
            }
            if let Some(reasoning_system_prompt) = raw.reasoning_system_prompt {
                config.reasoning_system_prompt = reasoning_system_prompt;
            }
            if let Some(decision_system_prompt) = raw.decision_system_prompt {
                config.decision_system_prompt = decision_system_prompt;
            }
        }

        Ok(config)
    }
}
