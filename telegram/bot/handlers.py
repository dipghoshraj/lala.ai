"""
Telegram message + error handlers.

Message processing pipeline
───────────────────────────
1.  Auth guard (applied via @authorized_only decorator in app.py wiring)
2.  Send "typing…" action so the user knows the bot is working
3.  Build the full message list (system prompt + history + current query)
4.  Call LLML "reasoning" model  →  structured log (INFO, never sent to user)
5.  Augment messages with the reasoning context
6.  Call LLML "decision" model   →  reply sent back to the user
7.  Commit user message + decision to conversation history

/clear resets that user's conversation history.
"""
from __future__ import annotations

import logging

from telegram import Update
from telegram.constants import ChatAction
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
        # Include reasoning as an intermediate assistant turn so the
        # decision model can build on it.
        decision_messages = list(messages)
        if reasoning:
            decision_messages.append({"role": "assistant", "content": reasoning})

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
        await update.message.reply_text(decision)

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
