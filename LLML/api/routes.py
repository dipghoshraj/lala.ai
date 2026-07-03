"""FastAPI router with OpenAI-compatible local inference endpoints."""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from api.classifier import CLASSIFIER_SYSTEM, heuristic_route

logger = logging.getLogger(__name__)

router = APIRouter(tags=["llm"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False


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


class EmbeddingItem(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    model: str
    data: list[EmbeddingItem]


class ClassifyRequest(BaseModel):
    query: str
    context: list[ChatMessage] = []
    model: str | None = None


class ClassifyResponse(BaseModel):
    route: str
    confidence: str


class EmbeddingsRequest(BaseModel):
    model: str | None = None
    input: list[str]


def build_prompt(messages: list[ChatMessage]) -> str:
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


def _estimate_tokens(text: str) -> int:
    return (len(text.encode("utf-8")) + 2) // 3


def slide_messages(
    messages: list[ChatMessage],
    n_ctx: int,
    max_tokens: int,
) -> list[ChatMessage]:
    safety_margin = 32
    budget = max(0, n_ctx - max_tokens - safety_margin)

    system = [m for m in messages if m.role == "system"]
    turns = [m for m in messages if m.role != "system"]

    while True:
        candidate = system + turns
        estimated = _estimate_tokens(build_prompt(candidate))
        if estimated <= budget:
            return candidate
        if len(turns) <= 1:
            logger.warning(
                "context still over budget (estimated=%d budget=%d); proceeding anyway",
                estimated,
                budget,
            )
            return candidate
        if len(turns) > 1 and turns[0].role == "user" and turns[1].role == "assistant":
            turns = turns[2:]
            logger.warning("sliding context window: dropped turn pair")
        else:
            turns = turns[1:]
            logger.warning("sliding context window: dropped single message")


@router.get("/v1/models", response_model=ModelListResponse, summary="List loaded model names")
async def list_models(request: Request) -> JSONResponse:
    registry = request.app.state.registry
    data = [{"id": role, "object": "model"} for role in registry.roles()]
    return JSONResponse({"object": "list", "data": data})


@router.post(
    "/v1/embeddings",
    response_model=EmbeddingsResponse,
    summary="Create embeddings",
)
async def embeddings(request: Request, req: EmbeddingsRequest) -> JSONResponse:
    if not req.input:
        return JSONResponse({"error": "input cannot be empty"}, status_code=400)

    registry = request.app.state.registry
    async with registry.use_embedding(req.model) as resolved:
        if resolved is None:
            available = ", ".join(registry.roles())
            return JSONResponse(
                {"error": f"unknown embedding model '{req.model}'. Available: {available}"},
                status_code=400,
            )
        model_name, runner = resolved
        data = []
        for idx, text in enumerate(req.input):
            embedding = await runner.embed(text)
            data.append({"object": "embedding", "index": idx, "embedding": embedding})

    return JSONResponse({"object": "list", "model": model_name, "data": data})


def _unknown_work_model_response(registry, model: str | None) -> JSONResponse:
    available = ", ".join(registry.work_model_names())
    return JSONResponse(
        {"error": f"unknown work model '{model}'. Available: {available}"},
        status_code=400,
    )


def _is_unknown_work_model(registry, model: str | None) -> bool:
    return model is not None and model not in registry.work_model_names()


@router.post(
    "/v1/chat/completions",
    response_model=ChatResponse,
    summary="Create a chat completion",
    responses={
        200: {
            "description": "JSON response when `stream` is false, SSE when `stream` is true.",
            "content": {
                "text/event-stream": {
                    "example": (
                        'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1710000000,'
                        '"model":"mistral-work","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
                        'data: [DONE]\n\n'
                    )
                }
            },
        }
    },
)
async def chat_completions(
    request: Request,
    req: ChatRequest,
) -> JSONResponse | StreamingResponse:
    registry = request.app.state.registry

    if not req.messages:
        return JSONResponse({"error": "messages must not be empty"}, status_code=400)

    response_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    if req.stream:
        if _is_unknown_work_model(registry, req.model):
            return _unknown_work_model_response(registry, req.model)
        return StreamingResponse(
            _stream_chat_response(
                registry,
                req.model,
                req.messages,
                req.max_tokens,
                req.temperature,
                response_id,
                created,
            ),
            media_type="text/event-stream",
        )

    async with registry.use_work(req.model) as resolved:
        if resolved is None:
            return _unknown_work_model_response(registry, req.model)
        resolved_model, runner = resolved
        max_tokens = req.max_tokens if req.max_tokens is not None else runner.max_tokens_default
        temperature = req.temperature
        slid = slide_messages(req.messages, runner.n_ctx, max_tokens)
        prompt = build_prompt(slid)
        logger.info(
            "chat completion request  model=%s  messages=%d  max_tokens=%d  stream=%s",
            resolved_model,
            len(req.messages),
            max_tokens,
            req.stream,
        )
        content = await runner.generate(prompt, max_tokens, temperature)

    return JSONResponse(
        {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": resolved_model,
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
        }
    )


async def _stream_chat_response(
    registry,
    requested_model: str | None,
    messages: list[ChatMessage],
    requested_max_tokens: int | None,
    temperature: float | None,
    response_id: str,
    created: int,
) -> AsyncIterator[str]:
    async with registry.use_work(requested_model) as resolved:
        if resolved is None:
            return
        resolved_model, runner = resolved
        max_tokens = (
            requested_max_tokens
            if requested_max_tokens is not None
            else runner.max_tokens_default
        )
        slid = slide_messages(messages, runner.n_ctx, max_tokens)
        prompt = build_prompt(slid)
        async for token in _stream_sse(
            runner,
            prompt,
            max_tokens,
            temperature,
            response_id,
            resolved_model,
            created,
        ):
            yield token


async def _stream_sse(
    runner,
    prompt: str,
    max_tokens: int,
    temperature: float | None,
    response_id: str,
    model: str,
    created: int,
) -> AsyncIterator[str]:
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

    stop_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(stop_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/v1/classify", response_model=ClassifyResponse, summary="Classify a query")
async def classify_query(
    request: Request,
    req: ClassifyRequest,
) -> JSONResponse:
    query = req.query.strip()
    if not query:
        return JSONResponse({"error": "query must not be empty"}, status_code=400)

    fast = heuristic_route(query)
    registry = request.app.state.registry

    async with registry.use_work(req.model) as resolved:
        if resolved is None:
            logger.warning("classify: unknown model %r, using heuristic", req.model)
            return JSONResponse(
                ClassifyResponse(route=fast, confidence="heuristic").model_dump()
            )
        role, runner = resolved

        context_tail = req.context[-2:] if req.context else []
        classify_messages = [
            ChatMessage(role="system", content=CLASSIFIER_SYSTEM),
            *context_tail,
            ChatMessage(role="user", content=query),
        ]
        prompt = build_prompt(classify_messages)

        try:
            raw = await runner.generate(prompt, max_tokens=5, temperature=0.0)
            route = "reasoning" if "REASON" in raw.strip().upper() else "direct"
            confidence = "llm"
            logger.info(
                "classify llm  model=%s  route=%s  raw=%r  query_len=%d",
                role,
                route,
                raw.strip(),
                len(query),
            )
        except Exception as exc:
            logger.warning("classify llm error, falling back to heuristic: %s", exc)
            route = fast
            confidence = "heuristic"

    return JSONResponse(ClassifyResponse(route=route, confidence=confidence).model_dump())
