"""Lazy model registry for memory-conscious local inference."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from config import ModelConfig

from .runner import ModelRunner

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Stores model configs and loads at most one runner at a time."""

    def __init__(
        self,
        work_models: list[ModelConfig],
        default_work_model: str,
        embedding_model: ModelConfig | None = None,
    ) -> None:
        self._work_models = {model.name: model for model in work_models}
        self._default_work_model = default_work_model
        self._embedding_model = embedding_model
        self._active_name: str | None = None
        self._active_runner: ModelRunner | None = None
        self._lock = asyncio.Lock()

    @property
    def default_work_model(self) -> str:
        return self._default_work_model

    def roles(self) -> list[str]:
        """Return all requestable model names, sorted for stable output."""
        names = set(self._work_models)
        if self._embedding_model is not None:
            names.add(self._embedding_model.name)
        return sorted(names)

    def work_model_names(self) -> list[str]:
        return sorted(self._work_models)

    def embedding_model_name(self) -> str | None:
        return self._embedding_model.name if self._embedding_model else None

    @asynccontextmanager
    async def use_work(
        self,
        name: str | None = None,
    ) -> AsyncIterator[tuple[str, ModelRunner] | None]:
        target = name or self._default_work_model
        model_cfg = self._work_models.get(target)
        if model_cfg is None:
            yield None
            return

        async with self._lock:
            yield target, await self._load(target, model_cfg)

    @asynccontextmanager
    async def use_embedding(
        self,
        name: str | None = None,
    ) -> AsyncIterator[tuple[str, ModelRunner] | None]:
        if self._embedding_model is None:
            yield None
            return
        target = name or self._embedding_model.name
        if target != self._embedding_model.name:
            yield None
            return

        async with self._lock:
            yield target, await self._load(target, self._embedding_model)

    async def _load(self, name: str, model_cfg: ModelConfig) -> ModelRunner:
        if self._active_name == name and self._active_runner is not None:
            return self._active_runner

        self._unload_active()
        logger.info("loading model on demand  name=%s  path=%s", name, model_cfg.model_path)
        self._active_runner = await asyncio.to_thread(
            ModelRunner,
            model_cfg.model_path,
            model_cfg.params,
        )
        self._active_name = name
        return self._active_runner

    def _unload_active(self) -> None:
        if self._active_runner is None:
            return
        logger.info("unloading active model  name=%s", self._active_name)
        self._active_runner.close()
        self._active_runner = None
        self._active_name = None
