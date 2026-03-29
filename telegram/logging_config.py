"""
Structured JSON logging for the telegram bot service.

Usage:
    from logging_config import setup_logging
    setup_logging()

Every log record emitted with extra fields (user_id, model, etc.) will be
serialised into the JSON "extra" block so log aggregators can filter on them.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone


class _JsonFormatter(logging.Formatter):
    """Format every log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        base = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach any extra fields supplied via logger.info("...", extra={...})
        reserved = logging.LogRecord.__dict__.keys() | {
            "message", "asctime", "args", "exc_info", "exc_text", "stack_info",
        }
        extra = {k: v for k, v in record.__dict__.items() if k not in reserved}
        if extra:
            base["extra"] = extra

        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(base, ensure_ascii=False, default=str)


def setup_logging(level: str | None = None) -> None:
    """
    Configure root logger.

    Level resolution order:
      1. `level` argument
      2. ``LOG_LEVEL`` environment variable
      3. ``INFO`` default
    """
    resolved = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())

    root = logging.getLogger()
    root.setLevel(resolved)
    root.handlers.clear()
    root.addHandler(handler)

    # Silence noisy third-party libraries
    for noisy in ("httpx", "httpcore", "telegram", "apscheduler"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
