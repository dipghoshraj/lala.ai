use llama_cpp::{LlamaModel, LlamaParams, SessionParams};

pub struct ModelWrapper {
    pub model: LlamaModel,
}

impl ModelWrapper {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let model = LlamaModel::load_from_file(path, LlamaParams::default())?;
        Ok(Self { model })
    }

    pub fn create_session(&self) -> anyhow::Result<SessionWrapper> {
        let session = self.model.create_session(SessionParams::default())?;
        Ok(SessionWrapper { session })
    }
}

pub struct SessionWrapper {
    pub session: llama_cpp::LlamaSession, // <-- fixed type
}