use llama_cpp::{LlamaModel, LlamaParams, SessionParams};
use llama_cpp::standard_sampler::StandardSampler;
use tracing::{debug, error, info, instrument, warn};

/// Parameters extracted from `ai-config.yaml` for a model entry.
#[derive(Debug)]
pub struct ModelParams {
    pub temperature: f32,
    pub max_tokens: usize,
    /// Layers to offload to GPU. 99 = all layers. 0 = CPU-only.
    pub n_gpu_layers: u32,
    /// CPU threads for generation. 0 = auto-detect.
    pub n_threads: u32,
    /// Context window in tokens.
    pub n_ctx: u32,
    /// Batch size for prompt evaluation.
    pub n_batch: u32,
}

/// Owns the loaded `LlamaModel`. Load once with [`ModelRunner::load`],
/// then call [`ModelRunner::generate_from_prompt`] for each request.
pub struct ModelRunner {
    pub(crate) model: LlamaModel,
    params: ModelParams,
}

// llama_cpp's LlamaModel is backed by a thread-safe C++ object.
// Sessions are created per-call so no shared mutable state exists.
unsafe impl Send for ModelRunner {}
unsafe impl Sync for ModelRunner {}

impl ModelRunner {
    /// Load the GGUF model from `path`. Called exactly once at startup.
    #[instrument(fields(path))]
    pub fn load(path: &str, params: ModelParams) -> anyhow::Result<Self> {
        info!(
            path,
            n_gpu_layers = params.n_gpu_layers,
            n_threads = params.n_threads,
            n_ctx = params.n_ctx,
            n_batch = params.n_batch,
            "loading GGUF model"
        );
        let llama_params = LlamaParams {
            n_gpu_layers: params.n_gpu_layers,
            ..LlamaParams::default()
        };
        let model = LlamaModel::load_from_file(path, llama_params)
            .map_err(|e| {
                error!(path, error = %e, "failed to load model from file");
                e
            })?;
        info!(path, "model loaded successfully");
        Ok(Self { model, params })
    }

    /// Run inference on a fully-formed prompt string.
    /// `max_tokens` and `temperature` override the config defaults when supplied by the caller.
    #[instrument(skip(self, prompt), fields(prompt_len = prompt.len(), max_tokens, temperature))]
    pub fn generate_from_prompt(
        &self,
        prompt: &str,
        max_tokens: Option<usize>,
        temperature: Option<f32>,
    ) -> anyhow::Result<String> {
        let max = max_tokens.unwrap_or(self.params.max_tokens);
        let temp = temperature.unwrap_or(self.params.temperature);
        debug!(prompt_len = prompt.len(), max_tokens = max, temperature = temp, "creating inference session");

        // Resolve thread count: 0 means auto-detect from available CPU cores.
        let n_threads = if self.params.n_threads == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(4)
        } else {
            self.params.n_threads
        };

        let session_params = SessionParams {
            n_ctx: self.params.n_ctx,
            n_batch: self.params.n_batch,
            n_threads,
            n_threads_batch: n_threads,
            ..SessionParams::default()
        };

        let mut session = self.model.create_session(session_params)
            .map_err(|e| {
                error!(error = %e, "failed to create llama session");
                e
            })?;

        session.advance_context(prompt)
            .map_err(|e| {
                error!(error = %e, "failed to advance context");
                e
            })?;

        let sampler = StandardSampler::new_softmax(
            vec![
                llama_cpp::standard_sampler::SamplerStage::RepetitionPenalty {
                    repetition_penalty: 1.1,
                    frequency_penalty: 0.0,
                    presence_penalty: 0.0,
                    last_n: 64,
                },
                llama_cpp::standard_sampler::SamplerStage::TopK(40),
                llama_cpp::standard_sampler::SamplerStage::TopP(0.95),
                llama_cpp::standard_sampler::SamplerStage::MinP(0.05),
                llama_cpp::standard_sampler::SamplerStage::Temperature(temp),
            ],
            1,
        );

        let mut stream = session
            .start_completing_with(sampler, max)
            .map_err(|e| {
                error!(error = %e, "failed to start completion stream");
                e
            })?;

        info!(max_tokens = max, temperature = temp, "inference started");
        let mut output = String::new();
        let mut token_count: usize = 0;

        while let Some(token) = stream.next_token() {
            let piece = self.model.token_to_piece(token);
            if piece.contains("[/INST]") {
                warn!("[/INST] marker found in output — stopping early");
                break;
            }
            output.push_str(&piece);
            token_count += 1;
        }

        info!(tokens_generated = token_count, output_len = output.len(), "inference complete");
        Ok(output)
    }
}
