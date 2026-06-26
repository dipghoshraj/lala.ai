"""
main.py — LLML-py entry point.

Port of LLML/src/main.rs.

Usage
-----
    python main.py [--config PATH] [--port PORT]

      --config PATH   Path to ai-config.yaml  (default: ../ai-config.yaml)
      --port   PORT   Port to listen on        (default: 3000)

Startup sequence
----------------
1. Parse CLI args.
2. load_config() — deserialise ai-config.yaml.
3. For each model entry: build ModelRunner → register in ModelRegistry.
4. Wire registry into FastAPI app state.
5. Mount API router.
6. uvicorn.run().
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from config import load_config, params_from_config
from model import ModelRegistry, ModelRunner
from api.routes import router
from api.vector_routes import vector_router
from vector.store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def build_app(config_path: str | Path) -> FastAPI:
    """Load config, instantiate models, wire FastAPI application."""
    config = load_config(config_path)
    registry = ModelRegistry()

    for model_cfg in config.models:
        role = model_cfg.role or model_cfg.name
        params = params_from_config(model_cfg.parameters)
        logger.info("loading model  role=%s  path=%s", role, model_cfg.model_path)
        runner = ModelRunner(model_cfg.model_path, params)
        registry.register(role, runner)

    logger.info("registered roles: %s", ", ".join(registry.roles()))

    # ── Vector store (ChromaDB) ───────────────────────────────────────────────
    chroma_cfg: dict = config.chroma if hasattr(config, "chroma") and config.chroma else {}
    try:
        vector_store = VectorStore.from_config(chroma_cfg)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ChromaDB init failed (%s) — vector search disabled", exc)
        vector_store = None

    app = FastAPI(
        title="LLML-py",
        description="Local LLM inference server — Python/FastAPI port of LLML (Rust/Axum)",
        version="0.1.0",
    )
    app.state.registry = registry
    app.state.vector_store = vector_store
    app.include_router(router)
    app.include_router(vector_router)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLML — Local LLM inference server (Python/FastAPI)"
    )
    parser.add_argument(
        "--config",
        default="../ai-config.yaml",
        metavar="PATH",
        help="Path to ai-config.yaml (default: ../ai-config.yaml)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3000,
        metavar="PORT",
        help="Port to serve on (default: 3000)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("config file not found: %s", config_path.resolve())
        sys.exit(1)

    app = build_app(config_path)
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
