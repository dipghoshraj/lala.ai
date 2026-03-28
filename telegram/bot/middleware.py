"""
Authorization middleware for the Telegram bot.

Only the user whose numeric ID matches ``Config.authorized_user_id`` is
permitted to interact with the bot.  All other senders receive a short
rejection message and the request is not forwarded to the agent.
"""
from __future__ import annotations

import logging
from functools import wraps
from typing import Callable, Awaitable, Any

from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


def authorized_only(
    authorized_user_id: int,
) -> Callable[[Callable[..., Awaitable[None]]], Callable[..., Awaitable[None]]]:
    """
    Decorator factory that gates a handler to a single Telegram user ID.

    Usage::

        @authorized_only(config.authorized_user_id)
        async def handle_message(update, context): ...
    """

    def decorator(
        handler: Callable[..., Awaitable[None]],
    ) -> Callable[..., Awaitable[None]]:
        @wraps(handler)
        async def wrapper(
            update: Update,
            context: ContextTypes.DEFAULT_TYPE,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            user = update.effective_user
            if user is None or user.id != authorized_user_id:
                uid = user.id if user else "unknown"
                logger.warning(
                    "unauthorized access attempt",
                    extra={"user_id": uid},
                )
                if update.message:
                    await update.message.reply_text("Unauthorized.")
                return
            await handler(update, context, *args, **kwargs)

        return wrapper

    return decorator
