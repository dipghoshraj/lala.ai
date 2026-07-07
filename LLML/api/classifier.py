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

CLASSIFIER_SYSTEM: str = """
You are a routing classifier.

Your task is to classify the user's query into exactly one category.

Reply with exactly one token:
DIRECT
REASON
METADATA

Definitions:

DIRECT
- Greetings.
- Small talk.
- Simple factual questions.
- Requests that can be answered directly without inspecting project or document data.

REASON
- Multi-step reasoning.
- Comparisons.
- Explanations.
- Code generation or debugging.
- Summarization.
- Planning.
- Analysis.

METADATA
- Requests about project or document information.
- Counts.
- Lists.
- Inventory.
- Names.
- IDs.
- Status.
- Dates.
- Owners.
- Any query asking to retrieve or count stored project/document information.

Priority:
If a query asks for project or document data (count, list, inventory, metadata), ALWAYS return METADATA, even if it is phrased as a simple factual question.

Examples:

User: how many projects do we have
METADATA

User: list all projects
METADATA

User: show project names
METADATA

User: how many documents are available
METADATA

User: who owns project A
METADATA

User: explain project architecture
REASON

User: compare project A and project B
REASON

User: write Python code to parse JSON
REASON

User: hello
DIRECT

User: what is Python
DIRECT
"""

def classifier_prompt(query: str) -> str:
    """
    Build the system + user prompt for the LLM classifier.

    The system prompt is a single, fixed string. The user prompt is the query
    itself, which is passed through without any modification.
    """
    return f"{CLASSIFIER_SYSTEM}\n\nUser query:\n{query}"

# ── Heuristic classifier ──────────────────────────────────────────────────────

METADATA_PATTERNS: tuple[str, ...] = (
    "how many projects",
    "how many documents",
    "what documents",
    "what projects",
    "list documents",
    "list projects",
    "documents in this project",
    "projects i have",
    "project inventory",
    "document inventory",
)

def heuristic_route(query: str) -> str:
    """
    Fast, zero-LLM routing decision.

    Returns ``"direct"``, ``"reasoning"``, or ``"metadata"``.

    Priority order
    --------------
    1. Matches a greeting/social pattern            → ``"direct"``
    2. Matches a metadata-specific pattern          → ``"metadata"``
    3. ≤ 3 words and no reasoning trigger           → ``"direct"``
    4. Contains a reasoning trigger keyword         → ``"reasoning"``
    5. ≤ 8 words with no trigger                    → ``"direct"``
    6. Longer queries                               → ``"reasoning"``
    """
    lower = query.strip().lower()
    reasoning_trigger = any(t in lower for t in REASONING_TRIGGERS)
    metadata_trigger = any(p in lower for p in METADATA_PATTERNS)

    # 1 — social patterns: exact match or starts-with (e.g. "thanks a lot")
    for pat in DIRECT_PATTERNS:
        if (lower == pat or lower.startswith(pat + " ")) and not reasoning_trigger:
            return "direct"

    # 2 — simple metadata patterns should bypass the direct fast-path.
    if metadata_trigger:
        return "metadata"

    words = lower.split()
    word_count = len(words)

    # 3 — very short, no trigger
    if word_count <= 3 and not reasoning_trigger:
        return "direct"

    # 4 — explicit reasoning trigger present
    if reasoning_trigger:
        return "reasoning"

    # 5 — medium length, no trigger
    if word_count <= 8 and not reasoning_trigger:
        return "direct"

    # 6 — default for longer queries
    return "reasoning"
