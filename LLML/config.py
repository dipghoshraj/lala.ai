"""
config.py — YAML config deserialisation and ModelParams extraction.

Port of:
  LLML/src/loalYaml/loadYaml.rs      (AiConfig / Model / Parameter structs)
  LLML/src/model/registry.rs          (params_from_config())
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# ── Dataclasses mirroring ai-config.yaml ────────────────────────────────────


@dataclass
class ModelParams:
    temperature: float = 0.7
    max_tokens: int = 100
    n_gpu_layers: int = 0       # 0 = CPU-only, 99 = all layers to GPU
    n_threads: int = 0          # 0 = auto-detect at runtime
    n_threads_batch: int = 0    # 0 = auto-detect at runtime
    n_ctx: int = 512
    n_batch: int = 512
    use_mlock: bool = False


@dataclass
class Parameter:
    name: str
    description: str
    param_type: str
    default: Any


@dataclass
class Model:
    name: str
    description: str
    model_type: str
    role: str           # logical role key, e.g. "reasoning" / "decision"
    parameters: list[Parameter]
    model_path: str


@dataclass
class ModelTypes:
    types: list[str]


@dataclass
class AiConfig:
    version: int
    model_types: ModelTypes
    models: list[Model]


# ── Internal helpers ─────────────────────────────────────────────────────────


def _get_float(parameters: list[Parameter], name: str, fallback: float) -> float:
    """Return the numeric default of the named parameter, or *fallback*."""
    for p in parameters:
        if p.name == name:
            try:
                return float(p.default)
            except (TypeError, ValueError):
                break
    return fallback


def _parse_parameters(raw: list[dict]) -> list[Parameter]:
    return [
        Parameter(
            name=p["name"],
            description=p.get("description", ""),
            param_type=p.get("type", ""),
            default=p.get("default"),
        )
        for p in raw
    ]


def _parse_model(raw: dict) -> Model:
    return Model(
        name=raw["name"],
        description=raw.get("description", ""),
        model_type=raw.get("type", ""),
        # role falls back to model name when absent — same as Rust #[serde(default)]
        role=raw.get("role") or raw["name"],
        parameters=_parse_parameters(raw.get("parameters", [])),
        model_path=raw["modelPath"],
    )


# ── Public API ───────────────────────────────────────────────────────────────


def params_from_config(parameters: list[Parameter]) -> ModelParams:
    """Extract ModelParams from a model's parameter list.

    Identical defaults to the Rust params_from_config() in registry.rs.
    """
    def get(name: str, fallback: float) -> float:
        return _get_float(parameters, name, fallback)

    return ModelParams(
        temperature=get("temperature", 0.7),
        max_tokens=int(get("max_tokens", 100.0)),
        n_gpu_layers=int(get("n_gpu_layers", 0.0)),
        n_threads=int(get("n_threads", 0.0)),
        n_threads_batch=int(get("n_threads_batch", 0.0)),
        n_ctx=int(get("n_ctx", 512.0)),
        n_batch=int(get("n_batch", 512.0)),
        use_mlock=get("use_mlock", 0.0) != 0.0,
    )


def load_config(path: str | Path) -> AiConfig:
    """Read and deserialise *ai-config.yaml*.

    Raises FileNotFoundError / yaml.YAMLError on failure.
    """
    content = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(content)
    return AiConfig(
        version=int(raw.get("version", 1)),
        model_types=ModelTypes(types=raw.get("Modeltypes", {}).get("types", [])),
        models=[_parse_model(m) for m in raw.get("Models", [])],
    )
