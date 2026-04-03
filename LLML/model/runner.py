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
import logging
import os
import threading
from collections.abc import AsyncIterator
from typing import Any

from llama_cpp import Llama

from config import ModelParams

logger = logging.getLogger(__name__)

# Sentinel used to signal that the background streaming thread has finished.
_STOP_SENTINEL: object = object()


class ModelRunner:
    """Owns a loaded Llama model. Instantiate once at startup; reuse for every request."""

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

        self._model = Llama(
            model_path=model_path,
            n_gpu_layers=params.n_gpu_layers,
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
            n_ctx=params.n_ctx,
            n_batch=params.n_batch,
            use_mlock=params.use_mlock,
            verbose=False,
            embedding= params.embedding,
        )
        self._params = params
        logger.info("model loaded successfully  path=%s", model_path)

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
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Non-streaming inference.

        Delegates to asyncio.to_thread so the event loop is never blocked.
        Stop token ``[/INST]`` prevents the model from echoing the prompt
        template — identical to the early-stop logic in the Rust implementation.
        """
        mt = max_tokens if max_tokens is not None else self._params.max_tokens
        temp = temperature if temperature is not None else self._params.temperature

        result: dict[str, Any] = await asyncio.to_thread(
            self._model,
            prompt,
            max_tokens=mt,
            temperature=temp,
            stop=["[/INST]"],
            echo=False,
        )
        content: str = result["choices"][0]["text"].strip()
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
            result = await asyncio.to_thread(self._model.embed, text)
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
            return ArithmeticError("Embedding not supported by this model")
        except Exception as exc:
            logger.warning("Embedding call failed: %s", exc)
            return ArithmeticError(f"Embedding failed: {exc}")

    async def stream(
        self,
        prompt: str,
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
                for chunk in self._model(
                    prompt,
                    max_tokens=mt,
                    temperature=temp,
                    stop=["[/INST]"],
                    echo=False,
                    stream=True,
                ):
                    token: str = chunk["choices"][0]["text"]
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
