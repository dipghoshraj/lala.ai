# LLML — Local LLM Inference Server

> **Location:** `lala.ai/LLML/`  
> **Role:** Model layer — loads a GGUF model once at startup and serves inference via an OpenAI-compatible HTTP API.

---

## Overview

LLML is a standalone Rust HTTP server that wraps a local LLM (via `llama_cpp`) behind a clean REST API. It is intentionally thin: no user interaction, no session persistence — just load-once inference over HTTP.

```
ai-config.yaml  ──►  LLML server (port 3000)
                         │
                    LlamaModel (loaded once)
                         │
                  POST /v1/chat/completions
                         │
                    JSON response
```

---

## Source Layout

```
LLML/src/
  main.rs          # Startup — reads config, loads model, starts Axum server
  model/
    mod.rs         # Re-exports ModelParams and ModelRunner
    model.rs       # LlamaModel wrapper — load once, generate per request
  api/
    mod.rs         # OpenAI-compatible types, prompt builder, Axum router + handler
  loalYaml/
    mod.rs
    loadYaml.rs    # Deserializes ai-config.yaml into typed structs
```

---

## Configuration — `ai-config.yaml`

All model parameters are declared in the shared `ai-config.yaml` at the repo root. LLML reads this file on startup:

| Parameter      | Type    | Default | Description |
|---------------|---------|---------|-------------|
| `temperature`  | float   | 0.7     | Sampling temperature |
| `max_tokens`   | integer | 100     | Default token generation limit per request |
| `n_gpu_layers` | integer | 0       | Layers offloaded to GPU. `0` = CPU-only. `99` = all layers (requires CUDA/Metal build) |
| `n_threads`    | integer | 4       | CPU threads for generation. Set to your **physical** core count |
| `n_ctx`        | integer | 512     | Context window in tokens. `512` for short queries, `2048` for long conversations |
| `n_batch`      | integer | 512     | Prompt evaluation batch size. Larger = faster prompt processing |
| `modelPath`    | string  | —       | Absolute path to the `.gguf` model file |

---

## Model Layer — `model/model.rs`

### `ModelParams`
Typed struct holding all parameters extracted from config. Passed once to `ModelRunner::load()`.

### `ModelRunner`
```rust
ModelRunner::load(path, params) -> anyhow::Result<ModelRunner>
```
Calls `LlamaModel::load_from_file()` exactly once. The loaded model is held in an `Arc<ModelRunner>` shared across all HTTP requests. Marked `Send + Sync` — safe for concurrent access because each request creates its own `LlamaSession`.

```rust
runner.generate_from_prompt(prompt: &str, max_tokens: Option<usize>) -> anyhow::Result<String>
```
Creates a fresh session per call (no context bleed between requests), advances context with the pre-built prompt, and collects generated tokens until the max limit or an `[/INST]` marker is encountered.

---

## API Layer — `api/mod.rs`

### Endpoint

```
POST /v1/chat/completions
Content-Type: application/json
```

### Request Schema (OpenAI-compatible)

```json
{
  "model": "LLML",
  "messages": [
    { "role": "system",    "content": "You are a helpful assistant." },
    { "role": "user",      "content": "What is Rust?" },
    { "role": "assistant", "content": "Rust is a systems programming language..." },
    { "role": "user",      "content": "Give me an example." }
  ],
  "max_tokens": 200
}
```

| Field        | Required | Description |
|-------------|----------|-------------|
| `messages`   | Yes      | Ordered array of `role`/`content` pairs. Roles: `system`, `user`, `assistant` |
| `model`      | No       | Informational only — LLML serves one model at a time |
| `max_tokens` | No       | Overrides the config default for this request |
| `temperature`| No       | Accepted but currently informational (sampler uses defaults) |

### Response Schema

```json
{
  "id": "chatcmpl-<uuid>",
  "object": "chat.completion",
  "created": 1711000000,
  "model": "LLML",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "..." },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
}
```

> `usage` token counts are placeholder zeros — tracking is not yet implemented.

### Prompt Builder

`build_prompt(messages)` converts the OpenAI messages array into a Mistral/Llama `[INST]` string:

```
<s>[INST] <system>

<first user message> [/INST] <assistant reply> </s>
[INST] <next user message> [/INST]
```

- A leading `system` message is merged into the first `[INST]` block.  
- `user`/`assistant` alternation builds multi-turn history.  
- The final open `[/INST]` lets the model continue generation from there.

### Inference Dispatch

Inference is blocking (CPU-bound). The handler uses `tokio::task::spawn_blocking` to run `generate_from_prompt` on a dedicated thread pool, keeping the async Axum executor free during generation.

---

## Build & Run

```sh
# Debug build
cd LLML && cargo build

# Release build (significantly faster inference)
cd LLML && cargo build --release

# Run (reads ../ai-config.yaml relative to the LLML/ directory)
cargo run --release
# or
cargo run --release -- ../ai-config.yaml
```

Server starts on `0.0.0.0:3000` by default.

### Logging

Controlled via `RUST_LOG`:

```sh
RUST_LOG=info  cargo run --release        # default — key lifecycle events
RUST_LOG=LLML=debug  cargo run --release  # include per-request debug detail
RUST_LOG=debug cargo run --release        # full trace including deps
```

---

## Dependencies

| Crate              | Purpose |
|--------------------|---------|
| `llama_cpp`        | Local GGUF inference via llama.cpp C++ library |
| `axum`             | Async HTTP server framework |
| `tokio`            | Async runtime |
| `serde` / `serde_json` / `serde_yaml` | Serialization for HTTP and config |
| `anyhow`           | Error propagation |
| `tracing` / `tracing-subscriber` | Structured logging |
| `uuid`             | Response ID generation |
