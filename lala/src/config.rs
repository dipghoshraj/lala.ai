use anyhow::Result;
use serde::Deserialize;
use std::fs;

const DEFAULT_SYSTEM_PROMPT: &str = "You are a friendly AI assistant named lala. Explain things clearly and naturally. Respond in full sentences.";
const DEFAULT_REASONING_SYSTEM_PROMPT: &str = "You are an internal reasoning engine. Analyse the user's query carefully. Think step by step: what is the user asking, what context matters, and what would make the best answer. Output your analysis concisely — this will be used to guide the final response, not shown to the user.";
const DEFAULT_DECISION_SYSTEM_PROMPT: &str = "You are lala, a friendly and concise AI assistant. You have been given an internal analysis to guide you. Use it to inform your answer but do NOT repeat or quote it. Respond directly to the user in clear, natural language.";

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
