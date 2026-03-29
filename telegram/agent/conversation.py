"""
Per-user conversation history store (in-memory, thread-safe).

Each user gets a rolling window of the last N *turns* (one turn = one user
message + one assistant reply).  The system prompt is prepended at query time
and is never stored in the history.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Deque

from agent.client import Message


_SYSTEM_PROMPT = (
    "You are lala, a helpful AI assistant. "
    "Think carefully before answering. "
    "Respond in clear, natural sentences."
)


class ConversationStore:
    """
    Thread-safe, in-memory, per-user message history.

    History is stored as a flat deque of Message dicts (role/content pairs).
    The maximum length is ``2 * max_turns`` because each turn = user + assistant.
    """

    def __init__(self, max_turns: int = 10) -> None:
        self._max_len = max_turns * 2
        self._histories: dict[int, Deque[Message]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_messages(self, user_id: int, user_text: str) -> list[Message]:
        """
        Return the full message list to send to the LLML server.

        Structure:
          [system_prompt, ...history..., {role: user, content: user_text}]
        """
        with self._lock:
            history = list(self._get_history(user_id))

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": user_text},
        ]

    def commit(self, user_id: int, user_text: str, assistant_reply: str) -> None:
        """Append the latest user↔assistant exchange to the history."""
        with self._lock:
            hist = self._get_history(user_id)
            hist.append({"role": "user", "content": user_text})
            hist.append({"role": "assistant", "content": assistant_reply})

    def clear(self, user_id: int) -> None:
        """Wipe the conversation history for a user."""
        with self._lock:
            self._histories.pop(user_id, None)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_history(self, user_id: int) -> Deque[Message]:
        if user_id not in self._histories:
            self._histories[user_id] = deque(maxlen=self._max_len)
        return self._histories[user_id]
