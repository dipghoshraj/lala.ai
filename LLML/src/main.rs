mod loalYaml;
mod model;
mod api;

use std::sync::Arc;
use tracing::info;
use tracing_subscriber::{EnvFilter, fmt};
use loalYaml::loadYaml::load_config;
use model::{ModelRunner, ModelRegistry, params_from_config};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialise structured logging.
    // Control verbosity via RUST_LOG, e.g.:
    //   RUST_LOG=info          — info and above (default)
    //   RUST_LOG=LLML=debug    — debug for this crate only
    //   RUST_LOG=debug         — everything including deps
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(true)
        .with_thread_ids(true)
        .init();

    let config_path = "../ai-config.yaml";
    let config = load_config(config_path).map_err(|e| anyhow::anyhow!(e))?;

    if config.models.is_empty() {
        return Err(anyhow::anyhow!("No models defined in ai-config.yaml"));
    }

    // Load every model declared in the config and register it by role.
    let mut registry = ModelRegistry::new();
    for model_cfg in &config.models {
        let params = params_from_config(&model_cfg.parameters);

        // Key: explicit role field, or fall back to the model name.
        let role = if model_cfg.role.is_empty() {
            model_cfg.name.clone()
        } else {
            model_cfg.role.clone()
        };

        info!(role, name = %model_cfg.name, path = %model_cfg.model_path, "loading model");
        let runner = ModelRunner::load(&model_cfg.model_path, params)?;
        registry.register(role, runner);
    }

    let available_roles = registry.roles().join(", ");
    info!(roles = %available_roles, "all models loaded");

    let registry = Arc::new(registry);
    let app = api::create_router(registry);
    let addr = "0.0.0.0:3000";

    info!(addr, "LLML API server starting");
    info!("  POST /v1/chat/completions  — OpenAI-compatible chat (pass \"model\": \"<role>\" to select)");
    info!("  GET  /v1/models            — list registered model roles");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

