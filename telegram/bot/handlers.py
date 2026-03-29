"""
Telegram message + error handlers.

Message processing pipeline
───────────────────────────
1.  Auth guard (applied via @authorized_only decorator in app.py wiring)
2.  Send "typing…" action so the user knows the bot is working
3.  Build the full message list (system prompt + history + current query)
4.  Classify route (when SMART_ROUTER=1):
      └─ "direct"    → skip to decision, reply to user (steps 7–8)
      └─ "reasoning" → continue to full pipeline below
5.  Call LLML "reasoning" model  →  structured log (INFO, never sent to user)
6.  Augment messages with the reasoning context
7.  Call LLML "decision" model   →  reply sent back to the user
8.  Commit user message + decision to conversation history

/clear resets that user's conversation history.
"""
from __future__ import annotations

import html
import logging

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ContextTypes

from agent.client import LLMLClient, LLMLError
from agent.conversation import ConversationStore
from config import Config

logger = logging.getLogger(__name__)


def build_handlers(
    client: LLMLClient,
    store: ConversationStore,
    config: Config,
) -> "Handlers":
    return Handlers(client, store, config)


class Handlers:
    """
    Encapsulates all Telegram handler coroutines.

    Keeping them on a class lets us inject the client, store, and config
    without relying on ``context.bot_data`` or global state.
    """

    def __init__(
        self,
        client: LLMLClient,
        store: ConversationStore,
        config: Config,
    ) -> None:
        self._client = client
        self._store = store
        self._cfg = config

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_response(reasoning: str | None, answer: str) -> tuple[str, str | None]:
        """
        Build the reply text and parse_mode for a given (reasoning, answer) pair.

        When reasoning is present the message uses Telegram HTML:

            [tap to reveal]  ← tg-spoiler containing the reasoning
            ​
            💬 Answer
            [answer text]

        The spoiler is collapsed by default so users who don't need to see
        the internal thinking can ignore it entirely.

        Returns (text, parse_mode).  parse_mode is ``ParseMode.HTML`` when
        reasoning is present, ``None`` for plain-text direct replies.
        """
        if not reasoning:
            return answer, None

        r = html.escape(reasoning.strip())
        a = html.escape(answer.strip())

        text = (
            f'<tg-spoiler><b>\U0001f9e0 Reasoning</b>\n\n{r}</tg-spoiler>'
            f"\n\n"
            f"<b>\U0001f4ac Answer</b>\n\n{a}"
        )
        return text, ParseMode.HTML

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def handle_start(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Respond to /start with a greeting."""
        await update.message.reply_text(
            "Hello! I'm lala — your local AI assistant. "
            "Send me a message and I'll think it through and respond.\n\n"
            "Use /clear to reset the conversation history."
        )

    async def handle_clear(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Wipe the conversation history for the requesting user."""
        user_id = update.effective_user.id
        self._store.clear(user_id)
        logger.info("conversation cleared", extra={"user_id": user_id})
        await update.message.reply_text("Conversation history cleared.")

    # ------------------------------------------------------------------
    # Message handler
    # ------------------------------------------------------------------

    async def handle_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """
        Core handler: reason → log → decide → reply.
        """
        user_id = update.effective_user.id
        user_text = update.message.text.strip()

        if not user_text:
            return

        logger.info(
            "user message received",
            extra={"user_id": user_id, "text_length": len(user_text)},
        )

        # Show typing indicator while we wait for inference
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING,
        )

        # Build full message list for this turn
        messages = self._store.build_messages(user_id, user_text)

        # ── Router: ask LLML whether reasoning is needed ──────────────────
        # Only fires when SMART_ROUTER=1; otherwise every query goes through
        # the full reasoning pipeline unchanged.
        # Pass the last 2 history turns (excluding the system prompt and the
        # just-appended user message) so the server can handle follow-ups.
        if self._cfg.smart_router:
            history_context = [m for m in messages[1:-1]][-2:]
            route = self._client.classify(user_text, context=history_context)
            logger.info(
                "classify",
                extra={"user_id": user_id, "route": route},
            )
        else:
            route = "reasoning"

        if route == "direct":
            # Skip reasoning entirely — go straight to decision.
            try:
                decision = self._client.decide(
                    messages,
                    max_tokens=self._cfg.decision_max_tokens,
                )
            except LLMLError as exc:
                logger.error(
                    "decision model error (direct path)",
                    extra={"user_id": user_id, "error": str(exc)},
                )
                await update.message.reply_text(
                    "Sorry, I'm having trouble reaching the AI server. Please try again shortly."
                )
                return

            self._store.commit(user_id, user_text, decision)
            await update.message.reply_text(decision)
            return

        # ── Step 1: Reasoning (logged only) ──────────────────────────────
        reasoning: str | None = None
        try:
            reasoning = self._client.reason(
                messages,
                max_tokens=self._cfg.reasoning_max_tokens,
            )
            logger.info(
                "reasoning",
                extra={
                    "user_id": user_id,
                    "reasoning": reasoning,
                    "model": "reasoning",
                },
            )
        except LLMLError as exc:
            logger.warning(
                "reasoning model unavailable",
                extra={"user_id": user_id, "error": str(exc)},
            )
            # Proceed to decision without reasoning context

        # ── Step 2: Decision (sent to user) ──────────────────────────────
        # The decision model only needs the system prompt, the current user
        # question, and the reasoning output — not the full conversation
        # history.  This keeps the input well within the model's n_ctx limit.
        decision_messages: list = [
            {"role": "system", "content": messages[0]["content"]},
            {"role": "user", "content": user_text},
        ]
        if reasoning:
            decision_messages.append({"role": "assistant", "content": reasoning})
            decision_messages.append(
                {"role": "user", "content": "Based on the above reasoning, what is your final answer?"}
            )

        try:
            decision = self._client.decide(
                decision_messages,
                max_tokens=self._cfg.decision_max_tokens,
            )
        except LLMLError as exc:
            logger.error(
                "decision model error",
                extra={"user_id": user_id, "error": str(exc)},
            )
            await update.message.reply_text(
                "Sorry, I'm having trouble reaching the AI server. Please try again shortly."
            )
            return

        logger.info(
            "decision sent",
            extra={
                "user_id": user_id,
                "decision_length": len(decision),
                "model": "decision",
            },
        )

        # ── Step 3: Commit & reply ────────────────────────────────────────
        self._store.commit(user_id, user_text, decision)
        reply_text, parse_mode = self._format_response(reasoning, decision)
        await update.message.reply_text(reply_text, parse_mode=parse_mode)

    # ------------------------------------------------------------------
    # Error handler
    # ------------------------------------------------------------------

    async def handle_error(
        self,
        update: object,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """
        Global uncaught-exception handler registered with the Application.
        Logs the traceback; never exposes internals to the user.
        """
        logger.error(
            "unhandled exception in update handler",
            exc_info=context.error,
            extra={"update": str(update)},
        )
        if isinstance(update, Update) and update.message:
            await update.message.reply_text(
                "An unexpected error occurred. Please try again."
            )
