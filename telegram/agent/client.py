"""
HTTP client for the LLML inference server.

Mirrors the role-based API from lala/src/agent/model.rs:
  POST /v1/chat/completions  { model, messages, max_tokens }
  → { choices: [{ message: { content } }] }

Two named helpers match the Rust counterparts:
  client.reason(messages)  → calls model "reasoning"
  client.decide(messages)  → calls model "decision"

Deliberately synchronous — inference is slow and single-user, so blocking
the event loop for one request at a time is the simplest correct approach.
"""
from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

# OpenAI-style message dict
Message = dict[str, str]  # {"role": str, "content": str}


class LLMLError(RuntimeError):
    """Raised when the LLML server returns an error or an unexpected response."""


class LLMLClient:
    """
    Blocking HTTP client that talks to the LLML inference server.

    A single ``requests.Session`` is reused for connection pooling.
    """

    def __init__(self, base_url: str, timeout: float | None = None) -> None:
        """
        Parameters
        ----------
        base_url:
            Root URL of the LLML server, e.g. ``http://localhost:3000``.
        timeout:
            Request timeout in seconds.  ``None`` disables the timeout
            (inference on CPU can be slow).
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers["Content-Type"] = "application/json"

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def reason(
        self,
        messages: list[Message],
        max_tokens: int | None = None,
    ) -> str:
        """Call the *reasoning* model.  Returns raw reasoning text."""
        return self._chat("reasoning", messages, max_tokens)

    def decide(
        self,
        messages: list[Message],
        max_tokens: int | None = None,
    ) -> str:
        """Call the *decision* model.  Returns the final answer text."""
        return self._chat("decision", messages, max_tokens)

    def classify(
        self,
        query: str,
        context: list[Message] | None = None,
    ) -> str:
        """
        Call ``POST /v1/classify`` and return ``"direct"`` or ``"reasoning"``.

        Parameters
        ----------
        query:
            The user's input text.
        context:
            Optional last few conversation turns.  Pass the tail of the
            conversation history so the server can handle follow-up queries
            correctly (e.g. "why?" after a complex answer).

        Returns
        -------
        ``"direct"`` or ``"reasoning"``.  On any error returns ``"reasoning"``
        (fail-closed — the full pipeline is always safe to run).
        """
        url = f"{self._base_url}/v1/classify"
        payload: dict[str, Any] = {"query": query}
        if context:
            payload["context"] = context[-2:]  # send at most 2 turns

        try:
            resp = self._session.post(url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            route = data.get("route", "reasoning")
            logger.debug(
                "classify",
                extra={"route": route, "confidence": data.get("confidence", "?")},
            )
            return route if route in ("direct", "reasoning") else "reasoning"
        except Exception as exc:  # noqa: BLE001
            # Any failure → safe default: run the full reasoning pipeline
            logger.warning("classify failed, defaulting to reasoning: %s", exc)
            return "reasoning"

    def close(self) -> None:
        self._session.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _chat(
        self,
        model: str,
        messages: list[Message],
        max_tokens: int | None,
    ) -> str:
        url = f"{self._base_url}/v1/chat/completions"
        payload: dict[str, Any] = {"model": model, "messages": messages}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        logger.debug(
            "llml request",
            extra={"model": model, "url": url, "n_messages": len(messages)},
        )

        try:
            resp = self._session.post(url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise LLMLError(
                f"LLML server returned {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except requests.ConnectionError as exc:
            raise LLMLError(f"Could not reach LLML server at {self._base_url}: {exc}") from exc
        except requests.Timeout as exc:
            raise LLMLError(f"Request to LLML server timed out: {exc}") from exc

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise LLMLError("LLML response contained no choices")

        content: str = choices[0]["message"]["content"]
        return content.strip()
