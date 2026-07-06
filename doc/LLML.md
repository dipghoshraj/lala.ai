# LLML — Local LLM Inference Server

> **Location:** `lala.ai/LLML/`
> **Role:** Local inference service — loads GGUF models on demand, serves chat, embeddings, classification, and vector-store operations.

---

## Overview

LLML is a Python FastAPI service that exposes local GGUF models through an OpenAI-style REST API. It is stateless: no user sessions, no conversation persistence, just model inference and optional vector store operations.

```
ai-config.yaml  ──►  LLML server (port 3000)
                         │
                    ModelRegistry (lazy loads models)
                         │
                  POST /v1/chat/completions
                  POST /v1/classify
                  POST /v1/embeddings
                  GET  /v1/models
                  /v1/vector/* (optional)
                         │
                    JSON response / SSE streaming
```

---

## Source Layout

```
LLML/
  main.py               # Entry point — load config, init registry & vector store, start uvicorn
  config.py             # Deserializes ai-config.yaml into AiConfig / ModelConfig / ModelParams
  requirements.txt
  api/
    __init__.py
    routes.py           # Chat, classify, embeddings, and model listing endpoints
    classifier.py       # Heuristic + LLM fallback routing logic
    vector_routes.py    # Optional ChromaDB-backed vector store API
  model/
    __init__.py
    runner.py           # ModelRunner wrapping llama-cpp-python
    registry.py         # Lazy model registry for work and embedding models
  vector/
    store.py            # ChromaDB vector store integration
```

---

## Configuration — `ai-config.yaml`

LLML reads `ai-config.yaml` from the repo root by default. The current config format defines:

- `default_work_model`
- `work_models`
- optional `embedding_model`
- optional `chroma`
- optional `database` credentials for `lala serve` and CLI runtime fallback

Sample config:

```yaml
version: 1

default_work_model: "mistral-work"

work_models:
  - name: "mistral-work"
    model_path: "/models/qwen2.5-3b-instruct-q4_k_m.gguf"
    params:
      temperature: 0.7
      max_tokens: 2048
      n_gpu_layers: 0
      n_threads: 0
      n_threads_batch: 0
      n_ctx: 8000
      n_batch: 512
      use_mlock: 1

  - name: "deepseek-work"
    model_path: "/models/deepseek-coder-1.3b-instruct.Q4_K_M.gguf"
    params:
      temperature: 0.7
      max_tokens: 2048
      n_gpu_layers: 0
      n_threads: 0
      n_threads_batch: 0
      n_ctx: 4096
      n_batch: 512
      use_mlock: 1

embedding_model:
  name: "embedding"
  model_path: "/models/bge-small-en-v1.5-q4_k_m.gguf"
  params:
    n_gpu_layers: 0
    n_threads: 0
    n_threads_batch: 0
    n_ctx: 1024
    n_batch: 512
    use_mlock: 1
    embedding: true

chroma:
  mode: embedded
  path: ./chroma_db
  host: localhost
  port: 8000
  collection_name: lala_vectors

database:
  user: postgres
  password: mysecretpassword
  name: vector_db
```

Key config behavior:

- `default_work_model` is the fallback if `model` is omitted in chat/classify requests.
- `work_models` are loaded on demand and used for `/v1/chat/completions` and `/v1/classify`.
- `embedding_model` is loaded only for `/v1/embeddings`.
- `chroma` config enables ChromaDB vector store endpoints.

Legacy support:

- `config.py` can also parse older `Models` arrays and `role` fields, but the preferred schema is `work_models` + `embedding_model`.

---

## Model Layer — `model/runner.py`

`ModelRunner` wraps `llama_cpp.Llama` and owns a loaded GGUF model instance.

- Resolves `n_threads`/`n_threads_batch`: `0` means auto-detect CPU cores.
- Loads with `n_gpu_layers`, `n_ctx`, `n_batch`, `use_mlock`, and optional `embedding` support.
- Provides `generate()`, `stream()`, and `embed()` helpers.

### `generate()`

Performs non-streaming inference via `asyncio.to_thread()`.

- Uses `stop=["[/INST]"]` to prevent prompt echo.
- Returns stripped assistant text.

### `stream()`

Provides SSE-style token streaming.

- A daemon thread drives the synchronous llama-cpp-python stream.
- Token chunks are delivered over an `asyncio.Queue`.
- Ends with a final `data: [DONE]` event.

### `embed()`

Generates embeddings for a single text string.

- Uses `self._model.embed(text)` when available.
- Falls back with a warning if embedding is unsupported or fails.

---

## Model Registry — `model/registry.py`

The registry is lazy and memory-conscious.

- Loads only one work model at a time.
- Unloads the active model before switching to another.
- Serializes model access with an async lock.

Public API:

- `work_model_names()`
- `embedding_model_name()`
- `use_work(name)`
- `use_embedding(name)`

This replaces the older `reasoning`/`decision` role split with named work models such as `mistral-work` and `deepseek-work`.

---

## API Layer — `api/routes.py`

Current endpoints:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/classify`
- `POST /v1/embeddings`
- `POST /v1/vector/add`
- `POST /v1/vector/search`
- `DELETE /v1/vector/chunks`
- `DELETE /v1/vector/documents/{source}`
- `GET /v1/vector/count`

Vector endpoints are only available when ChromaDB initializes successfully.

### `GET /v1/models`

Returns configured model names in OpenAI list format.

### `POST /v1/chat/completions`

Request example:

```json
{
  "model": "mistral-work",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Explain Rust lifetimes." }
  ],
  "max_tokens": 200,
  "temperature": 0.7,
  "stream": false
}
```

Behavior:

1. Resolve the requested work model or fallback to `default_work_model`.
2. Validate non-empty `messages`.
3. Use the model's default `max_tokens` if omitted.
4. Slide history to fit `n_ctx`.
5. Build the prompt with `build_prompt()`.
6. Generate text or stream tokens.

Response is OpenAI-style chat completion JSON. Streaming responses emit `chat.completion.chunk` SSE records.

### `POST /v1/classify`

Request example:

```json
{
  "query": "explain transformers in ML",
  "context": [
    { "role": "user", "content": "hi" },
    { "role": "assistant", "content": "Hello! How can I help?" }
  ],
  "model": "mistral-work"
}
```

Classification flow:

- Uses `heuristic_route(query)` as a fast path.
- If the model exists, also runs the LLM classifier prompt.
- Returns `route` as `direct` or `reasoning`.
- `confidence` is `heuristic` when fallbacked or `llm` when the model is used.

### `POST /v1/embeddings`

Request example:

```json
{
  "model": "embedding",
  "input": ["hello world", "another sentence"]
}
```

Behavior:

- Resolves the embedding model.
- Rejects empty `input` arrays.
- Returns each vector with an `index`.

Example response:

```json
{
  "object": "list",
  "model": "embedding",
  "data": [
    { "object": "embedding", "index": 0, "embedding": [0.01, -0.42, ...] },
    { "object": "embedding", "index": 1, "embedding": [-0.13, 0.54, ...] }
  ]
}
```

### Vector Store API

Optional ChromaDB endpoints include:

- `POST /v1/vector/add`
- `POST /v1/vector/search`
- `DELETE /v1/vector/chunks`
- `DELETE /v1/vector/documents/{source}`
- `GET /v1/vector/count`

These only exist when `LLML/main.py` successfully creates a `VectorStore`.

---

## Classifier — `api/classifier.py`

The classifier module is used by `/v1/classify`.

- `heuristic_route(query)` returns `direct` or `reasoning` without an LLM call.
- `CLASSIFIER_SYSTEM` is the system prompt sent to the model when the LLM fallback runs.

Heuristic rules:

1. Greeting/social patterns → `direct`
2. ≤ 3 words without reasoning trigger → `direct`
3. Any reasoning trigger keyword → `reasoning`
4. ≤ 8 words without trigger → `direct`
5. Longer queries → `reasoning`

If LLM classification runs, it expects the model to respond with `REASON` or `DIRECT`.

---

## Prompt Format

`build_prompt()` converts OpenAI-style messages into Mistral/llama instruction format:

```text
<s>[INST] {system_prompt}

{first_user_message} [/INST]
{assistant_response} </s>
[INST] {next_user_message} [/INST]
```

- A leading `system` message is merged into the first `[INST]` block.
- Alternating `user` and `assistant` messages build multi-turn history.
- The final open `[/INST]` allows the model to continue generation.
- `stop=["[/INST]"]` prevents prompt echo.

---

## Context Window Management

`slide_messages()` ensures the prompt fits within the model's `n_ctx` budget:

- Budget = `n_ctx - max_tokens - 32`
- Uses a UTF-8 byte-based approximation for token count.
- Preserves the system prompt and recent messages.
- Drops oldest turn pairs until the prompt fits.

---

## Build & Run

### Local Python

```sh
cd LLML
pip install -r requirements.txt
python main.py
```

Default config path is `../ai-config.yaml`. Override with:

```sh
python main.py --config /path/to/ai-config.yaml --port 3000
```

### Docker

Use the repository `docker-compose.yml` or build the LLML image from `LLML/Dockerfile.llm-inference`.

---

## Current LLML Behavior vs. Old Documentation

Important updates from the legacy LLML design:

- The old `reasoning` / `decision` role split is gone.
- LLML now uses named work models via `work_models` and `default_work_model`.
- The embedding model is separate and optional.
- Model loading is lazy and single-model-at-a-time.
- `/v1/embeddings` exists now.
- Optional `/v1/vector/*` endpoints are added.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Async HTTP server and router |
| `uvicorn` | ASGI server |
| `llama-cpp-python` | GGUF model loading and inference |
| `pyyaml` | Config parsing |
| `pydantic` | Request validation |
