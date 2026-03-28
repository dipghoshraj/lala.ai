"""
model/registry.py — ModelRegistry: maps role strings to ModelRunner instances.

Port of LLML/src/model/registry.rs.
"""
from __future__ import annotations

import logging

from .runner import ModelRunner

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Holds all loaded ModelRunners keyed by their logical role.

    Example roles: ``"reasoning"``, ``"decision"``.
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelRunner] = {}

    def register(self, role: str, runner: ModelRunner) -> None:
        logger.info("registering model runner  role=%s", role)
        self._models[role] = runner

    def get(self, role: str) -> ModelRunner | None:
        return self._models.get(role)

    def roles(self) -> list[str]:
        """Return all registered role names, sorted for stable output."""
        return sorted(self._models.keys())

    def first(self) -> tuple[str, ModelRunner] | None:
        """Return the first registered (role, runner) pair, or None."""
        try:
            role, runner = next(iter(self._models.items()))
            return role, runner
        except StopIteration:
            return None
