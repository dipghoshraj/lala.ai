# LLML-py — Python Inference Server

> **Status:** Implemented  
> **Replaces:** `LLML/` Rust/Axum server (kept as archive fallback)  
> **Motivation:** `llama_cpp` 0.3.2 (Rust crate) is not mature enough for reliable local inference. `llama-cpp-python` provides the same `llama.cpp` C++ backend with a more stable, actively-maintained Python binding.

---

## Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Python 3.11+ | Mature ML ecosystem, stable llama.cpp bindings |
| Framework | FastAPI + uvicorn | Async, auto OpenAPI docs, Pydantic v2 validation |
| LLM binding | `llama-cpp-python` | Same backend as Rust crate, far more stable API surface |
| Streaming | Optional SSE (`"stream": true`) | Adds capability; existing callers unaffected (default `false`) |
| Config | Reuse `ai-config.yaml` unchanged | No migration cost |
| Port | `3000` (same as Rust server) | `lala` CLI and `telegram/` bot connect without changes |
| Rust code | Keep as archive | Low cost, safe fallback |

---

## Target Layout

```
LLML-py/
├── requirements.txt          # fastapi, uvicorn, llama-cpp-python, pyyaml
├── main.py                   # Entry: argparse → load config → build registry → uvicorn
├── config.py                 # Dataclasses mirroring ai-config.yaml; params_from_config()
├── model/
│   ├── __init__.py
│   ├── runner.py             # ModelRunner wrapping llama_cpp.Llama
│   └── registry.py           # ModelRegistry: role (str) → ModelRunner
└── api/
    ├── __init__.py
    └── routes.py             # FastAPI router, Pydantic models, build_prompt(), slide_messages()
```

---

## Module Responsibilities

### `config.py`

Deserializes `ai-config.yaml` into Python dataclasses. Direct port of `LLML/src/loalYaml/loadYaml.rs` and `LLML/src/model/registry.rs:params_from_config()`.

```python
@dataclass
class ModelParams:
    temperature: float       # default: 0.7
    max_tokens: int          # default: 100
    n_gpu_layers: int        # default: 0  (0=CPU, 99=all layers to GPU)
    n_threads: int           # default: 0  (0=auto-detect)
    n_threads_batch: int     # default: 0
    n_ctx: int               # default: 512
    n_batch: int             # default: 512
    use_mlock: bool          # default: False
```

`load_config(path)` reads YAML; `params_from_config(parameters)` extracts `ModelParams` from the `parameters` list with identical defaults to the Rust implementation.

---

### `model/runner.py`

Wraps `llama_cpp.Llama`. Model is loaded **once at startup** (same as Rust `ModelRunner::load()`).

```python
class ModelRunner:
    def __init__(self, model_path: str, params: ModelParams) -> None:
        # Calls llama_cpp.Llama(model_path, n_gpu_layers=..., n_ctx=..., ...)

    async def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        # asyncio.to_thread(self._model, prompt, max_tokens=..., stop=["[/INST]"])

    async def stream(self, prompt: str, max_tokens: int, temperature: float) -> AsyncIterator[str]:
        # background Thread runs model(stream=True); enqueues chunks via asyncio.Queue
        # async generator drains queue and yields token pieces

    @property
    def n_ctx(self) -> int: ...
    @property
    def max_tokens_default(self) -> int: ...
```

**Thread model:** `generate()` uses `asyncio.to_thread` to avoid blocking the event loop. `stream()` runs the synchronous llama generator in a `threading.Thread` and bridges to async via an `asyncio.Queue`.

#### `llama-cpp-python` Parameter Mapping

| `ModelParams` field | `llama_cpp.Llama()` kwarg |
|---------------------|---------------------------|
| `n_gpu_layers`      | `n_gpu_layers`            |
| `n_threads`         | `n_threads`               |
| `n_threads_batch`   | `n_threads_batch`         |
| `n_ctx`             | `n_ctx`                   |
| `n_batch`           | `n_batch`                 |
| `use_mlock`         | `use_mlock`               |
| `temperature`       | per-call kwarg            |
| `max_tokens`        | per-call kwarg            |

---

### `model/registry.py`

```python
class ModelRegistry:
    def register(self, role: str, runner: ModelRunner) -> None: ...
    def get(self, role: str) -> ModelRunner | None: ...
    def roles(self) -> list[str]: ...          # sorted
    def first(self) -> tuple[str, ModelRunner] | None: ...
```

Exact port of `LLML/src/model/registry.rs`.

---

### `api/routes.py`

#### Pydantic Models (OpenAI-compatible)

```python
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False          # NEW — not in Rust version

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage
```

#### `build_prompt(messages)`

Exact port of `LLML/src/api/mod.rs:build_prompt()`. Converts OpenAI message list to Mistral/Llama `[INST]` instruction format:

```
<s>[INST] {system}\n\n{first_user} [/INST] {assistant} </s>[INST] {next_user} [/INST]...
```

- System message is optional; if present it is prepended to the first `[INST]` block.
- Alternates user/assistant pairs.
- No trailing tokens after the final `[/INST]` (model fills in from here).

#### `slide_messages(messages, n_ctx, max_tokens)`

Context window management — port of `LLML/src/api/mod.rs:slide_messages()`:

- Token budget: `n_ctx - max_tokens - 32` (safety margin).
- Estimates tokens as `len(bytes) / 3` (conservative ~3 bytes per token).
- Drops oldest user+assistant pairs together to keep conversation coherent.
- **Never** drops the system message (index 0) or the final user message.
- Logs a warning when sliding occurs.

#### Routes

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Inference endpoint |
| `GET`  | `/v1/models` | Lists registered model roles |

**POST /v1/chat/completions** behaviour:

| `stream` | Response type | Format |
|----------|--------------|--------|
| `false` (default) | `JSONResponse` | Full `ChatResponse` JSON |
| `true` | `StreamingResponse(media_type="text/event-stream")` | OpenAI-compatible SSE chunks |

SSE frame format (streaming):
```
data: {"id":"...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"TOKEN"},"finish_reason":null}]}\n\n
```
Final frame:
```
data: [DONE]\n\n
```

---

### `main.py`

```
usage: main.py [--config PATH] [--port PORT]

  --config PATH   Path to ai-config.yaml  (default: ../ai-config.yaml)
  --port   PORT   Port to serve on         (default: 3000)
```

Startup sequence:
1. Parse args.
2. `load_config(config_path)` — deserialize YAML.
3. For each model in config: construct `ModelRunner(model_path, params)` → `registry.register(role, runner)`. Logs role name and path.
4. Log all registered roles.
5. Create `FastAPI` app; store registry in `app.state.registry`.
6. Mount `api.routes.router`.
7. `uvicorn.run(app, host="0.0.0.0", port=port)`.

---

## API Contract Compatibility

The Python server exposes the **identical** API contract as the Rust server. No changes required in:

- **`lala/`** Rust CLI — connects via `LLML_API_URL` env var or arg.
- **`telegram/agent/client.py`** — `LLMLClient` uses `POST /v1/chat/completions` with `{"model", "messages", "max_tokens?"}`.

The only addition is the optional `"stream": true` field, which is ignored by existing callers.

---

## Build & Run

```sh
# Install dependencies
cd LLML-py
pip install -r requirements.txt

# Start server (reads ../ai-config.yaml, serves on :3000)
python main.py

# Custom config / port
python main.py --config /path/to/ai-config.yaml --port 8080

# GPU acceleration (requires llama-cpp-python built with CUDA/Metal)
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

---

## Implementation Phases

### Phase 1 — Scaffold & Config
Files with no inter-dependencies, can be created in parallel.

- [x] `requirements.txt`
- [x] `config.py` — `AiConfig`, `Model`, `Parameter`, `ModelParams` dataclasses; `load_config()`; `params_from_config()`
- [x] `model/__init__.py`, `api/__init__.py`

### Phase 2 — Model Layer
Depends on: `config.py`

- [x] `model/runner.py` — `ModelRunner`
- [x] `model/registry.py` — `ModelRegistry`

### Phase 3 — API Layer
Depends on: `model/runner.py`, `model/registry.py`

- [x] `api/routes.py` — Pydantic models, `build_prompt()`, `slide_messages()`, route handlers

### Phase 4 — Entry Point
Depends on: Phase 2 + Phase 3

- [x] `main.py`

---

## Verification Checklist

```sh
# 1. Dependencies install cleanly
pip install -r requirements.txt

# 2. Server starts, logs both roles, binds :3000
python main.py

# 3. Model list
curl http://localhost:3000/v1/models
# → {"object":"list","data":[{"id":"reasoning",...},{"id":"decision",...}]}

# 4. Non-streaming inference
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
# → {"choices":[{"message":{"content":"..."}}],...}

# 5. Streaming inference
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}],"stream":true}'
# → data: {"choices":[{"delta":{"content":"..."}}]}
# → ...
# → data: [DONE]

# 6. lala CLI connects unchanged
cd lala && cargo run -- http://localhost:3000

# 7. Telegram bot connects unchanged  
cd telegram && python app.py
```

---

## Reference Files (Rust Originals)

| Python file | Ported from |
|-------------|------------|
| `config.py` | `LLML/src/loalYaml/loadYaml.rs` + `registry.rs:params_from_config()` |
| `model/runner.py` | `LLML/src/model/model.rs` |
| `model/registry.py` | `LLML/src/model/registry.rs` |
| `api/routes.py` | `LLML/src/api/mod.rs` |
| `main.py` | `LLML/src/main.rs` |
