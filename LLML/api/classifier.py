"""
api/classifier.py — Shared query-routing logic for the LLML server.

This module is the single source of truth for:
  - The heuristic fast-path (social/greeting patterns, word-count rules).
  - The LLM classifier system prompt.
  - The ``heuristic_route()`` helper used both as a fast-path shortcut and as
    the fallback when the LLM call fails.

Consumers
---------
- ``api/routes.py``          → ``POST /v1/classify`` handler
- Any future internal caller that needs routing without a network hop
"""
from __future__ import annotations

# ── Social / greeting patterns — always route DIRECT, no LLM call needed ──────

DIRECT_PATTERNS: tuple[str, ...] = (
    "hello",
    "hi",
    "hey",
    "thanks",
    "thank you",
    "bye",
    "goodbye",
    "good morning",
    "good evening",
    "good night",
    "good afternoon",
    "ok",
    "okay",
    "sure",
    "yes",
    "no",
    "great",
    "perfect",
    "nice",
    "cool",
    "awesome",
    "got it",
    "understood",
)

# ── Keywords that strongly signal multi-step reasoning is needed ──────────────

REASONING_TRIGGERS: tuple[str, ...] = (
    "why",
    "how",
    "explain",
    "analyze",
    "analyse",
    "compare",
    "difference",
    "what if",
    "implement",
    "write",
    "debug",
    "fix",
    "code",
    "algorithm",
    "calculate",
    "evaluate",
    "pros",
    "cons",
    "summarize",
    "summarise",
    "describe",
    "define",
    "plan",
    "design",
    "architecture",
    "step",
    "process",
    "derive",
    "prove",
    "optimise",
    "optimize",
    "refactor",
    "suggest",
    "recommend",
)

# ── LLM classifier prompt ─────────────────────────────────────────────────────

CLASSIFIER_SYSTEM: str = (
    "You are a routing classifier. "
    "Reply with exactly one word — nothing else. "
    "REASON: the query needs multi-step analysis, explanation, code, or comparison. "
    "DIRECT: the query is a greeting, simple factual question, or short conversational reply."
)


# ── Heuristic classifier ──────────────────────────────────────────────────────


def heuristic_route(query: str) -> str:
    """
    Fast, zero-LLM routing decision.

    Returns ``"direct"`` or ``"reasoning"``.

    Priority order
    --------------
    1. Matches a greeting/social pattern            → ``"direct"``
    2. ≤ 3 words and no reasoning trigger           → ``"direct"``
    3. Contains a reasoning trigger keyword         → ``"reasoning"``
    4. ≤ 8 words with no trigger                    → ``"direct"``
    5. Longer queries                               → ``"reasoning"``
    """
    lower = query.strip().lower()

    # 1 — social patterns: exact match or starts-with (e.g. "thanks a lot")
    for pat in DIRECT_PATTERNS:
        if lower == pat or lower.startswith(pat + " "):
            return "direct"

    words = lower.split()
    word_count = len(words)

    # 2 — very short, no trigger
    if word_count <= 3 and not any(t in lower for t in REASONING_TRIGGERS):
        return "direct"

    # 3 — explicit reasoning trigger present
    if any(t in lower for t in REASONING_TRIGGERS):
        return "reasoning"

    # 4 — medium length, no trigger
    if word_count <= 8:
        return "direct"

    # 5 — default for longer queries
    return "reasoning"
