"""YAML config deserialisation for the local inference layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelParams:
    temperature: float = 0.7
    max_tokens: int = 100
    n_gpu_layers: int = 0
    n_threads: int = 0
    n_threads_batch: int = 0
    n_ctx: int = 512
    n_batch: int = 512
    use_mlock: bool = False
    embedding: bool = False


@dataclass
class ModelConfig:
    name: str
    model_path: str
    params: ModelParams = field(default_factory=ModelParams)


@dataclass
class AiConfig:
    version: int
    default_work_model: str
    work_models: list[ModelConfig]
    embedding_model: ModelConfig | None = None
    chroma: dict[str, Any] | None = None


def _coerce_float(raw: dict[str, Any], name: str, fallback: float) -> float:
    try:
        return float(raw.get(name, fallback))
    except (TypeError, ValueError):
        return fallback


def _coerce_bool(raw: dict[str, Any], name: str, fallback: bool) -> bool:
    value = raw.get(name, fallback)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _parse_params(
    raw: dict[str, Any] | list[dict[str, Any]] | None,
    *,
    embedding: bool,
) -> ModelParams:
    if isinstance(raw, list):
        raw = {p.get("name"): p.get("default") for p in raw if "name" in p}
    raw = raw or {}

    return ModelParams(
        temperature=_coerce_float(raw, "temperature", 0.7),
        max_tokens=int(_coerce_float(raw, "max_tokens", 100.0)),
        n_gpu_layers=int(_coerce_float(raw, "n_gpu_layers", 0.0)),
        n_threads=int(_coerce_float(raw, "n_threads", 0.0)),
        n_threads_batch=int(_coerce_float(raw, "n_threads_batch", 0.0)),
        n_ctx=int(_coerce_float(raw, "n_ctx", 512.0)),
        n_batch=int(_coerce_float(raw, "n_batch", 512.0)),
        use_mlock=_coerce_float(raw, "use_mlock", 0.0) != 0.0,
        embedding=_coerce_bool(raw, "embedding", embedding),
    )


def _model_path(raw: dict[str, Any]) -> str:
    return raw.get("model_path") or raw["modelPath"]


def _parse_model(raw: dict[str, Any], *, embedding: bool) -> ModelConfig:
    return ModelConfig(
        name=raw["name"],
        model_path=_model_path(raw),
        params=_parse_params(raw.get("params", raw.get("parameters")), embedding=embedding),
    )


def _legacy_models(raw: dict[str, Any]) -> tuple[list[ModelConfig], ModelConfig | None]:
    work_models: list[ModelConfig] = []
    embedding_model: ModelConfig | None = None

    for model in raw.get("Models", []):
        role = (model.get("role") or model.get("name") or "").lower()
        model_type = (model.get("type") or "").lower()
        is_embedding = role == "embedding" or model_type == "embedding"
        cfg = _parse_model(model, embedding=is_embedding)
        if is_embedding:
            embedding_model = cfg
        else:
            work_models.append(cfg)

    return work_models, embedding_model


def _validate_config(config: AiConfig) -> AiConfig:
    names = {model.name for model in config.work_models}
    if not config.work_models:
        raise ValueError("ai-config.yaml must define at least one work model")
    if config.default_work_model not in names:
        available = ", ".join(sorted(names))
        raise ValueError(
            f"default_work_model '{config.default_work_model}' is not a configured work model. "
            f"Available: {available}"
        )
    return config


def load_config(path: str | Path) -> AiConfig:
    """Read and validate *ai-config.yaml*."""
    content = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(content) or {}

    if "work_models" in raw:
        work_models = [_parse_model(m, embedding=False) for m in raw.get("work_models", [])]
        embedding_model = (
            _parse_model(raw["embedding_model"], embedding=True)
            if raw.get("embedding_model")
            else None
        )
    else:
        work_models, embedding_model = _legacy_models(raw)

    default_work_model = raw.get("default_work_model")
    if not default_work_model and work_models:
        default_work_model = work_models[0].name

    config = AiConfig(
        version=int(raw.get("version", 1)),
        default_work_model=default_work_model,
        work_models=work_models,
        embedding_model=embedding_model,
        chroma=raw.get("chroma"),
    )
    return _validate_config(config)
