"""
Central configuration — all values sourced from environment variables.
Load a .env file before importing (see app.py or use python-dotenv).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    # Telegram
    token: str
    authorized_user_id: int

    # LLML inference server
    llml_api_url: str

    # Inference tuning (optional overrides)
    reasoning_max_tokens: int
    decision_max_tokens: int

    # Conversation history cap (per user, rolling window)
    max_history_turns: int

    # Set SMART_ROUTER=1 to use the LLM-based /v1/classify endpoint.
    # When false (default) every message goes through the full reasoning pipeline.
    smart_router: bool

    @classmethod
    def from_env(cls) -> "Config":
        token = os.environ["TOKEN"]
        user_id = os.environ["USERID"]
        if not token:
            raise ValueError("TOKEN env var is required")
        if not user_id:
            raise ValueError("USERID env var is required")

        return cls(
            token=token,
            authorized_user_id=int(user_id),
            llml_api_url=os.getenv("LLML_API_URL", "http://localhost:3000"),
            reasoning_max_tokens=int(os.getenv("REASONING_MAX_TOKENS", "512")),
            decision_max_tokens=int(os.getenv("DECISION_MAX_TOKENS", "256")),
            max_history_turns=int(os.getenv("MAX_HISTORY_TURNS", "10")),
            smart_router=os.getenv("SMART_ROUTER", "0").strip() == "1",
        )
