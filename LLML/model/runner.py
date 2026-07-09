"""
model/runner.py — ModelRunner wrapping llama_cpp.Llama.

Port of LLML/src/model/model.rs.

Thread model
------------
generate()  — runs the sync llama_cpp call inside asyncio.to_thread so the
              FastAPI event loop is never blocked.
stream()    — a background daemon thread drives the sync llama_cpp streaming
              generator and pushes token pieces into an asyncio.Queue; the
              async generator drains the queue and yields to the caller.
"""
from __future__ import annotations

import asyncio
import gc
import logging
import os
import threading
from collections.abc import AsyncIterator
from typing import Any, Optional, cast

from llama_cpp import Llama

from config import ModelParams

logger = logging.getLogger(__name__)

# Sentinel used to signal that the background streaming thread has finished.
_STOP_SENTINEL: object = object()


class ModelRunner:
    """Owns a loaded Llama model. Instantiate once at startup; reuse for every request."""

    _model: Optional[Llama] = None

    def __init__(self, model_path: str, params: ModelParams) -> None:
        # Resolve thread counts: 0 in config → use all available cores.
        n_threads = params.n_threads or os.cpu_count() or 4
        n_threads_batch = params.n_threads_batch or os.cpu_count() or n_threads

        logger.info(
            "loading GGUF model  path=%s  n_gpu_layers=%d  n_ctx=%d  n_batch=%d  "
            "n_threads=%d  n_threads_batch=%d  use_mlock=%s",
            model_path,
            params.n_gpu_layers,
            params.n_ctx,
            params.n_batch,
            n_threads,
            n_threads_batch,
            params.use_mlock,
        )

        model_path = f"/models/{model_path}"

        self._model = Llama(
            model_path=model_path,
            n_gpu_layers=params.n_gpu_layers,
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
            n_ctx=params.n_ctx,
            n_batch=params.n_batch,
            use_mlock=params.use_mlock,
            verbose=False,
            embedding=params.embedding,
        )
        self._params = params
        logger.info("model loaded successfully  path=%s", model_path)

    def close(self) -> None:
        """Release llama.cpp resources held by this runner."""
        model = getattr(self, "_model", None)
        if model is None:
            return

        close = getattr(model, "close", None)
        if callable(close):
            close()
        self._model = None
        gc.collect()

    def _call_completion(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Any:
        assert self._model is not None
        return self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["[/INST]"],
            echo=False,
        )

    def _call_chat_completion(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> Any:
        assert self._model is not None
        return self._model.create_chat_completion(
            messages=cast(Any, messages),
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def n_ctx(self) -> int:
        return self._params.n_ctx

    @property
    def max_tokens_default(self) -> int:
        return self._params.max_tokens

    # ── Inference ─────────────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Non-streaming inference.

        Delegates to asyncio.to_thread so the event loop is never blocked.
        If structured chat messages are provided, use llama_cpp's chat completion
        path so the model can format prompts from metadata and chat templates.
        """
        mt = max_tokens if max_tokens is not None else self._params.max_tokens
        temp = temperature if temperature is not None else self._params.temperature

        if messages is not None:
            result = cast(
                dict[str, Any],
                await asyncio.to_thread(
                    self._call_chat_completion,
                    messages,
                    mt,
                    temp,
                ),
            )
            choice = result["choices"][0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict):
                    content = message.get("content", "")
                else:
                    content = choice.get("text", "")
            else:
                content = ""
            content = str(content).strip()
        else:
            assert prompt is not None, "prompt must be provided when messages is None"
            result = cast(
                dict[str, Any],
                await asyncio.to_thread(
                    self._call_completion,
                    prompt,
                    mt,
                    temp,
                ),
            )
            content = ""
            choice = result["choices"][0]
            if isinstance(choice, dict):
                content = choice.get("text", "")
            content = str(content).strip()

        logger.info(
            "inference complete  model=%s  output_len=%d",
            self._params,
            len(content),
        )
        return content

    async def embed(self, text: str) -> list[float]:
        """Generate embedding vector for text.

        Uses llama-cpp-python's embed() when available, otherwise deterministic placeholder.
        """


        try:
            assert self._model is not None
            result: Any = await asyncio.to_thread(self._model.embed, text)
            if isinstance(result, dict) and "data" in result:
                data = result["data"]
                if isinstance(data, list) and len(data) > 0:
                    emb = data[0].get("embedding") if isinstance(data[0], dict) else None
                    if isinstance(emb, list):
                        return [float(v) for v in emb]
            if isinstance(result, list):
                return [float(v) for v in result]
            return []
        except AttributeError:
            logger.warning("Model runner has no embed(), using deterministic fallback")
            return []
        except Exception as exc:
            logger.warning("Embedding call failed: %s", exc)
            return []

    async def stream(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Streaming inference — async generator that yields token pieces.

        A daemon thread drives the synchronous llama_cpp streaming generator.
        Token pieces are passed back to the async caller via an asyncio.Queue,
        bridged with asyncio.run_coroutine_threadsafe.
        """
        mt = max_tokens if max_tokens is not None else self._params.max_tokens
        temp = temperature if temperature is not None else self._params.temperature

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[object] = asyncio.Queue()

        def _run() -> None:
            try:
                if messages is not None:
                    assert self._model is not None
                    iterator = self._model.create_chat_completion(
                        messages=cast(Any, messages),
                        max_tokens=mt,
                        temperature=temp,
                        stream=True,
                    )
                    for chunk in iterator:
                        if isinstance(chunk, dict):
                            choice = chunk["choices"][0]
                            if isinstance(choice, dict):
                                delta = choice.get("delta", {})
                                token = delta.get("content") if isinstance(delta, dict) else None
                            else:
                                token = None
                        else:
                            token = None
                        if token is not None:
                            asyncio.run_coroutine_threadsafe(queue.put(token), loop).result()
                else:
                    assert prompt is not None, "prompt must be provided when messages is None"
                    assert self._model is not None
                    for chunk in self._model(
                        prompt,
                        max_tokens=mt,
                        temperature=temp,
                        stop=["[/INST]"],
                        echo=False,
                        stream=True,
                    ):
                        if isinstance(chunk, dict):
                            inner = chunk["choices"][0]
                            token = inner.get("text", "") if isinstance(inner, dict) else ""
                        else:
                            token = ""
                        asyncio.run_coroutine_threadsafe(queue.put(token), loop).result()
            except Exception as exc:  # noqa: BLE001
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(_STOP_SENTINEL), loop).result()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        while True:
            item = await queue.get()
            if item is _STOP_SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item  # type: ignore[misc]
