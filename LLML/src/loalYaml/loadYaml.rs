use serde::Deserialize;
use std::{fs, path::Path};

/// Top-level structure matching `ai-config.yaml`.
#[derive(Debug, Deserialize)]
pub struct AiConfig {
    pub version: u32,
    #[serde(rename = "Modeltypes")]
    pub model_types: ModelTypes,
    #[serde(rename = "Models")]
    pub models: Vec<Model>,
}

#[derive(Debug, Deserialize)]
pub struct ModelTypes {
    pub types: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct Model {
    pub name: String,
    pub description: String,
    #[serde(rename = "type")]
    pub model_type: String,
    /// Logical role for this model, e.g. "reasoning" or "decision".
    /// Falls back to the model `name` when absent in the YAML.
    #[serde(default)]
    pub role: String,
    pub parameters: Vec<Parameter>,
    #[serde(rename = "modelPath")]
    pub model_path: String,
}

/// A single model parameter. `default` can be a float or integer in the YAML,
/// so it is stored as a generic `serde_yaml::Value`.
#[derive(Debug, Deserialize, Clone)]
pub struct Parameter {
    pub name: String,
    pub description: String,
    #[serde(rename = "type")]
    pub param_type: String,
    pub default: serde_yaml::Value,
}

/// Reads and deserializes the YAML config file at `path`.
/// Returns an `AiConfig` on success or an error string on failure.
pub fn load_config(path: impl AsRef<Path>) -> Result<AiConfig, String> {
    let content = fs::read_to_string(path.as_ref())
        .map_err(|e| format!("Failed to read config file: {e}"))?;
    let config: AiConfig = serde_yaml::from_str(&content)
        .map_err(|e| format!("Failed to parse YAML: {e}"))?;
    Ok(config)
}