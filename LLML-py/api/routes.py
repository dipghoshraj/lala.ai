"""
api/routes.py — FastAPI router with OpenAI-compatible endpoints.

Port of LLML/src/api/mod.rs.

New capability vs. Rust: optional streaming via ``"stream": true`` in the
request body (existing callers are unaffected; default is false).
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Pydantic models (OpenAI-compatible) ──────────────────────────────────────


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False  # Not in the Rust version — new capability


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ── Prompt builder ────────────────────────────────────────────────────────────


def build_prompt(messages: list[ChatMessage]) -> str:
    """Convert an OpenAI messages array to Mistral/Llama [INST] format.

    Port of LLML/src/api/mod.rs:build_prompt().

    Output format::

        <s>[INST] {system}\\n\\n{first_user} [/INST] {assistant} </s>[INST] {next_user} [/INST]...

    Rules:
    - System message is optional; prepended to the first [INST] block when present.
    - Alternates user/assistant pairs.
    - No trailing tokens after the final [/INST] — the model generates from there.
    """
    result: list[str] = []

    msgs = list(messages)
    system_content: str | None = None
    if msgs and msgs[0].role == "system":
        system_content = msgs[0].content
        msgs = msgs[1:]

    first_user = True
    for msg in msgs:
        if msg.role == "user":
            if first_user:
                if system_content is not None:
                    result.append(f"<s>[INST] {system_content}\n\n{msg.content} [/INST]")
                else:
                    result.append(f"<s>[INST] {msg.content} [/INST]")
                first_user = False
            else:
                result.append(f"[INST] {msg.content} [/INST]")
        elif msg.role == "assistant":
            result.append(f" {msg.content} </s>")

    return "".join(result)


# ── Context window management ─────────────────────────────────────────────────


def _estimate_tokens(text: str) -> int:
    """Conservative ~3-bytes-per-token estimate (same heuristic as the Rust impl)."""
    return (len(text.encode("utf-8")) + 2) // 3


def slide_messages(
    messages: list[ChatMessage],
    n_ctx: int,
    max_tokens: int,
) -> list[ChatMessage]:
    """Trim oldest turn-pairs until the prompt fits within the context budget.

    Port of LLML/src/api/mod.rs:slide_messages().

    Budget: ``n_ctx - max_tokens - 32`` (safety margin).

    Rules:
    - System message is never dropped.
    - The final user message is never dropped.
    - Oldest user+assistant pairs are dropped together to keep the
      conversation structure syntactically valid.
    """
    SAFETY_MARGIN = 32
    budget = max(0, n_ctx - max_tokens - SAFETY_MARGIN)

    system: list[ChatMessage] = [m for m in messages if m.role == "system"]
    turns: list[ChatMessage] = [m for m in messages if m.role != "system"]

    while True:
        candidate = system + turns
        estimated = _estimate_tokens(build_prompt(candidate))

        if estimated <= budget:
            return candidate

        if len(turns) <= 1:
            logger.warning(
                "context still over budget (estimated=%d budget=%d) after full slide; proceeding anyway",
                estimated,
                budget,
            )
            return candidate

        # Drop the oldest pair (user + assistant) together when possible,
        # otherwise just the oldest single message — mirrors the Rust logic.
        if (
            len(turns) > 1
            and turns[0].role == "user"
            and turns[1].role == "assistant"
        ):
            turns = turns[2:]
            logger.warning(
                "sliding context window: dropped turn pair (%d messages remaining)", len(turns)
            )
        else:
            turns = turns[1:]
            logger.warning(
                "sliding context window: dropped single message (%d messages remaining)", len(turns)
            )


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    registry = request.app.state.registry
    data = [{"id": role, "object": "model"} for role in registry.roles()]
    return JSONResponse({"object": "list", "data": data})


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    req: ChatRequest,
) -> JSONResponse | StreamingResponse:
    registry = request.app.state.registry

    if not req.messages:
        return JSONResponse({"error": "messages must not be empty"}, status_code=400)

    # ── Resolve model role ────────────────────────────────────────────────────
    if req.model is not None:
        runner = registry.get(req.model)
        if runner is None:
            available = ", ".join(registry.roles())
            return JSONResponse(
                {"error": f"unknown model role '{req.model}'. Available: {available}"},
                status_code=400,
            )
        resolved_role = req.model
    else:
        first = registry.first()
        if first is None:
            return JSONResponse({"error": "no models available"}, status_code=500)
        resolved_role, runner = first

    # ── Resolve generation budget ─────────────────────────────────────────────
    max_tokens = req.max_tokens if req.max_tokens is not None else runner.max_tokens_default
    temperature = req.temperature

    logger.info(
        "chat completion request  model=%s  messages=%d  max_tokens=%d  stream=%s",
        resolved_role,
        len(req.messages),
        max_tokens,
        req.stream,
    )

    # ── Slide context + build prompt ─────────────────────────────────────────
    slid = slide_messages(req.messages, runner.n_ctx, max_tokens)
    prompt = build_prompt(slid)

    logger.info(
        "prompt built  original_turns=%d  slid_turns=%d  prompt_len=%d",
        len(req.messages),
        len(slid),
        len(prompt),
    )

    response_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    # ── Streaming ─────────────────────────────────────────────────────────────
    if req.stream:
        return StreamingResponse(
            _stream_sse(runner, prompt, max_tokens, temperature, response_id, resolved_role, created),
            media_type="text/event-stream",
        )

    # ── Non-streaming ─────────────────────────────────────────────────────────
    content = await runner.generate(prompt, max_tokens, temperature)
    logger.info("inference complete  output_len=%d  model=%s", len(content), resolved_role)

    return JSONResponse({
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": resolved_role,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": _estimate_tokens(prompt),
            "completion_tokens": _estimate_tokens(content),
            "total_tokens": _estimate_tokens(prompt) + _estimate_tokens(content),
        },
    })


# ── SSE generator (streaming path) ────────────────────────────────────────────


async def _stream_sse(
    runner,
    prompt: str,
    max_tokens: int,
    temperature: float | None,
    response_id: str,
    model: str,
    created: int,
) -> AsyncIterator[str]:
    """Yield OpenAI-compatible SSE frames for each token, then a final [DONE] frame."""
    async for token in runner.stream(prompt, max_tokens, temperature):
        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Final stop chunk (empty delta, finish_reason = "stop")
    stop_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(stop_chunk)}\n\n"
    yield "data: [DONE]\n\n"
