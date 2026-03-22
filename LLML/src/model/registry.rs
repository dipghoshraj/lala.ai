use std::collections::HashMap;
use tracing::info;

use super::model::{ModelParams, ModelRunner};

/// Holds all loaded model runners, keyed by their logical role
/// (e.g. "reasoning", "decision").
pub struct ModelRegistry {
    models: HashMap<String, ModelRunner>,
}

// ModelRunner is already Send + Sync (backed by thread-safe llama_cpp C++ objects).
unsafe impl Send for ModelRegistry {}
unsafe impl Sync for ModelRegistry {}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    /// Register a runner under the given role key.
    pub fn register(&mut self, role: String, runner: ModelRunner) {
        info!(role, "registering model runner");
        self.models.insert(role, runner);
    }

    /// Retrieve a runner by role. Returns `None` when the role is unknown.
    pub fn get(&self, role: &str) -> Option<&ModelRunner> {
        self.models.get(role)
    }

    /// Returns all registered role names, sorted for stable output.
    pub fn roles(&self) -> Vec<String> {
        let mut roles: Vec<_> = self.models.keys().cloned().collect();
        roles.sort();
        roles
    }

    /// Convenience: returns the first available runner (useful as a default).
    pub fn first(&self) -> Option<(&str, &ModelRunner)> {
        self.models.iter().next().map(|(k, v)| (k.as_str(), v))
    }
}

/// Extract `ModelParams` from a config model entry's parameter list.
pub fn params_from_config(
    parameters: &[crate::loalYaml::loadYaml::Parameter],
) -> ModelParams {
    let get_f64 = |name: &str, fallback: f64| -> f64 {
        parameters
            .iter()
            .find(|p| p.name == name)
            .and_then(|p| p.default.as_f64())
            .unwrap_or(fallback)
    };

    ModelParams {
        temperature: get_f64("temperature", 0.7) as f32,
        max_tokens: get_f64("max_tokens", 100.0) as usize,
        n_gpu_layers: get_f64("n_gpu_layers", 0.0) as u32,
        n_threads: get_f64("n_threads", 4.0) as u32,
        n_ctx: get_f64("n_ctx", 512.0) as u32,
        n_batch: get_f64("n_batch", 512.0) as u32,
    }
}
