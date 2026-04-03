# LLML — Local LLM Inference Server

> **Location:** `lala.ai/LLML/`  
> **Role:** Model layer — loads GGUF models once at startup and serves inference via an OpenAI-compatible HTTP API.

---

## Overview

LLML is a standalone Python HTTP server (FastAPI + llama-cpp-python) that wraps local LLMs behind a clean REST API. It is intentionally thin: no user interaction, no session persistence — just load-once inference over HTTP.

```
ai-config.yaml  ──►  LLML server (port 3000)
                         │
                    ModelRegistry (loaded once per role)
                         │
                  POST /v1/chat/completions
                  POST /v1/classify
                  GET  /v1/models
                         │
                    JSON response
```

---

## Source Layout

```
LLML/
  main.py             # Entry — loads config, registers models, starts uvicorn on :3000
  config.py           # Deserializes ai-config.yaml → AiConfig / Model / ModelParams
  requirements.txt
  api/
    __init__.py
    routes.py         # Router: /v1/chat/completions, /v1/models, /v1/classify
                      # build_prompt() → Mistral [INST]...[/INST] format
                      # slide_messages() → context window management
    classifier.py     # Heuristic fast-path + LLM-based query classifier
  model/
    __init__.py
    runner.py         # ModelRunner — async generate() + stream() via asyncio.to_thread
    registry.py       # ModelRegistry: role (str) → ModelRunner
```

---

## Configuration — `ai-config.yaml`

All model parameters are declared in the shared `ai-config.yaml` at the repo root. LLML reads this file on startup:

| Parameter      | Type    | Default | Description |
|---------------|---------|---------|-------------|
| `temperature`  | float   | 0.7     | Sampling temperature |
| `max_tokens`   | integer | 100     | Default token generation limit per request |
| `n_gpu_layers` | integer | 0       | Layers offloaded to GPU. `0` = CPU-only. `99` = all layers (requires CUDA build) |
| `n_threads`    | integer | 4       | CPU threads for generation. `0` = auto-detect (`os.cpu_count()`) |
| `n_ctx`        | integer | 512     | Context window in tokens. `512` for short queries, `2048` for conversations |
| `n_batch`      | integer | 512     | Prompt evaluation batch size. Larger = faster prompt processing |
| `use_mlock`    | integer | 1       | Pin model weights in RAM (prevents swapping) |
| `modelPath`    | string  | —       | Absolute path to the `.gguf` model file |

---

## Model Layer — `model/runner.py`

### `ModelRunner`

Wraps `llama_cpp.Llama` (C FFI to llama.cpp). One instance per model role, loaded at startup.

```python
runner = ModelRunner(model_path, params)
```

Thread count resolution: if `n_threads=0` in config, auto-detects via `os.cpu_count()`.

### `async generate()`

Runs inference in a background thread via `asyncio.to_thread()` so the FastAPI event loop is never blocked:

```python
result = await runner.generate(prompt, max_tokens, temperature)
# → str (stripped completion text)
```

Calls `self._model(prompt, max_tokens, temperature, stop=["[/INST]"], echo=False)` internally.

### `async stream()`

Streaming variant — pushes token chunks to an `asyncio.Queue` from a daemon thread, yields them as an async iterator. Used for SSE streaming responses.

---

## Model Registry — `model/registry.py`

Simple dict-based registry mapping role strings to `ModelRunner` instances:

| Method | Description |
|--------|-------------|
| `register(role, runner)` | Maps a role key to its ModelRunner |
| `get(role)` | Lookup by role (returns `None` if not found) |
| `roles()` | Sorted list of all registered role names |
| `first()` | Returns `(role, runner)` tuple or `None` |

---

## API Layer — `api/routes.py`

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat inference (non-streaming + SSE streaming) |
| `POST` | `/v1/classify` | Query routing — classifies as `"direct"` or `"reasoning"` |
| `GET`  | `/v1/models` | Lists all registered role names |
| `POST` | `/v1/embeddings` | Generate text embeddings using the configured `embedding` model role |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat inference (non-streaming + SSE streaming) |

### POST `/v1/embeddings`

**Request:**
```json
{
  "input": ["text to embed", "another sentence"],
  "model": "embedding"  // optional; defaults to the first registered embedding role
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    { "object": "embedding", "index": 0, "embedding": [0.01, -0.42, ...] },
    { "object": "embedding", "index": 1, "embedding": [-0.13, 0.54, ...] }
  ]
}
```

**Behavior:**
- Uses `ModelRunner.embed(text)` when delivered by llama-cpp-python.
- If the model does not expose embed() (older or unsupported llama-cpp versions), falls back to a deterministic hash-based embedding generator.
- Accepts any length list in `input`; outputs matching indices.

### POST `/v1/chat/completions`

**Request:**
```json
{
  "model": "reasoning",
  "messages": [
    { "role": "system",    "content": "You are a helpful assistant." },
    { "role": "user",      "content": "What is Rust?" }
  ],
  "max_tokens": 200,
  "temperature": 0.7,
  "stream": false
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `messages` | yes | Non-empty array of `{role, content}` pairs |
| `model` | no | Role key from registry (defaults to first registered) |
| `max_tokens` | no | Overrides the config default for this request |
| `temperature` | no | Overrides the model config default (0.0–2.0) |
| `stream` | no | `true` for SSE streaming, `false` (default) for batch response |

**Flow:**
1. Resolve model role (from `req.model` or first registered)
2. Resolve `max_tokens` and `temperature` (request → config defaults)
3. **Slide context window** via `slide_messages()` — drops oldest turn-pairs if prompt exceeds budget (`n_ctx - max_tokens - 32`)
4. **Build prompt** via `build_prompt(messages)` → Mistral/Llama `[INST]...[/INST]` format
5. Call `runner.generate()` (non-streaming) or `runner.stream()` (SSE chunks)
6. Stop token `[/INST]` prevents echo/leakage
7. Return OpenAI-compatible response with usage stats

**Response:**
```json
{
  "id": "chatcmpl-<uuid>",
  "object": "chat.completion",
  "created": 1711000000,
  "model": "reasoning",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "..." },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 42, "completion_tokens": 128, "total_tokens": 170 }
}
```

### POST `/v1/classify`

Classifies a query as requiring reasoning or a direct answer. Uses a two-tier strategy:

1. **Heuristic fast-path** (no LLM call): greeting patterns, short queries, keyword triggers
2. **LLM fallback**: sends query + context to reasoning model with a classifier system prompt

**Request:**
```json
{
  "query": "explain transformers in ML",
  "context": [
    { "role": "user",      "content": "hi" },
    { "role": "assistant", "content": "Hello! How can I help?" }
  ]
}
```

**Response:**
```json
{ "route": "reasoning", "confidence": "heuristic" }
```

### GET `/v1/models`

Returns all registered roles in OpenAI list format:
```json
{
  "object": "list",
  "data": [
    { "id": "decision", "object": "model" },
    { "id": "reasoning", "object": "model" }
  ]
}
```

---

## Classifier — `api/classifier.py`

Five-step heuristic priority chain:

1. Exact-match or starts-with **greeting patterns** (`hello`, `hi`, `thanks`, `bye`, etc.) → `"direct"`
2. ≤3 words + no reasoning trigger → `"direct"`
3. Contains **reasoning triggers** (`why`, `how`, `explain`, `analyze`, `code`, `debug`, `implement`, etc.) → `"reasoning"`
4. ≤8 words + no trigger → `"direct"`
5. Default for longer queries → `"reasoning"`

The LLM classifier system prompt instructs the model to reply with exactly one word: `REASON` or `DIRECT`.

---

## Prompt Format

`build_prompt()` converts the OpenAI messages array into Mistral/Llama instruction format:

```
<s>[INST] {system_prompt}

{first_user_message} [/INST] {assistant_reply} </s>
[INST] {next_user_message} [/INST]
```

- A leading `system` message is merged into the first `[INST]` block.
- `user`/`assistant` alternation builds multi-turn history.
- The final open `[/INST]` lets the model continue generation.
- `[/INST]` in output tokens triggers an early stop (prevents prompt leakage).

---

## Context Window Management

`slide_messages()` ensures the prompt fits within the model's `n_ctx` budget:

- Budget = `n_ctx - max_tokens - 32` (reserves space for generation + safety margin)
- Estimates token count as `len(content) / 4` (rough char-to-token approximation)
- Always preserves the system prompt (index 0) and the last user message
- Drops oldest turn-pairs from the middle when budget is exceeded

---

## Build & Run

### Docker (recommended)

```sh
docker build -f LLML.Dockerfile -t lala-llml .
docker run -p 3000:3000 \
  -v /path/to/your/models:/models \
  -v ./ai-config.yaml:/app/ai-config.yaml \
  lala-llml
```

Update `modelPath` values in `ai-config.yaml` to use the container path:
```yaml
modelPath: "/models/your-model.Q4_K_M.gguf"
```

GPU (CUDA) support: uncomment the `CMAKE_ARGS` line in `LLML.Dockerfile` and switch to a `nvidia/cuda` base image.

### Local Python

```sh
cd LLML
pip install -r requirements.txt

# Reads ../ai-config.yaml by default; serves on :3000
python main.py

# Custom config path and port
python main.py --config /path/to/ai-config.yaml --port 3000
```

### Logging

Standard Python logging, `INFO` level by default:

```sh
# Streaming logs to stdout
PYTHONUNBUFFERED=1 python main.py
```

---

## Planned: Embedding Endpoint (Phase 1)

Phase 1 will add a `POST /v1/embed` endpoint to support Qdrant vector search. LLML will load a dedicated embedding model (e.g. `bge-small-en-v1.5`) registered under the role `"embedding"` in `ai-config.yaml`.

**Request:**
```json
{
  "input": ["chunk text 1", "chunk text 2"],
  "model": "embedding"
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    { "index": 0, "embedding": [0.021, -0.043, ...], "object": "embedding" },
    { "index": 1, "embedding": [0.007, 0.112, ...],  "object": "embedding" }
  ],
  "model": "embedding"
}
```

The `rag` crate (Rust) will call this endpoint during `store()` to embed each chunk and during `retrieve()` to embed the query before the Qdrant vector search. LLML remains stateless — no Qdrant client in LLML; Qdrant interaction stays in the `rag` crate.

Required `ai-config.yaml` addition:
```yaml
- name: "bge-small-embedding"
  description: "Sentence embedding model for semantic search"
  role: "embedding"
  parameters:
    - name: "n_ctx"
      default: 512
    - name: "n_batch"
      default: 512
  modelPath: "/models/bge-small-en-v1.5-q4_k_m.gguf"
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Async HTTP server and router |
| `uvicorn` | ASGI server (with standard extras for auto-reload, etc.) |
| `llama-cpp-python` | Local GGUF model loading and token generation via llama.cpp C FFI |
| `pyyaml` | `ai-config.yaml` parsing |
