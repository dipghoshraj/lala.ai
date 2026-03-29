
"""
app.py — Telegram bot entry point.

Responsibilities:
  - Load .env
  - Set up structured logging
  - Build Config, LLMLClient, ConversationStore
  - Wire auth middleware + handlers onto the Application
  - Start long-polling loop
"""
from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

from config import Config
from logging_config import setup_logging
from agent.client import LLMLClient
from agent.conversation import ConversationStore
from bot.handlers import build_handlers
from bot.middleware import authorized_only

load_dotenv()
setup_logging()

logger = logging.getLogger(__name__)


def main() -> None:
    cfg = Config.from_env()

    logger.info(
        "starting lala telegram bot",
        extra={
            "llml_api_url": cfg.llml_api_url,
            "authorized_user_id": cfg.authorized_user_id,
            "smart_router": cfg.smart_router,
        },
    )

    llml = LLMLClient(base_url=cfg.llml_api_url)
    store = ConversationStore(max_turns=cfg.max_history_turns)
    handlers = build_handlers(llml, store, cfg)

    app = Application.builder().token(cfg.token).build()

    # Commands — also gated by auth
    app.add_handler(
        CommandHandler(
            "start",
            authorized_only(cfg.authorized_user_id)(handlers.handle_start),
        )
    )
    app.add_handler(
        CommandHandler(
            "clear",
            authorized_only(cfg.authorized_user_id)(handlers.handle_clear),
        )
    )

    # All non-command text messages
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            authorized_only(cfg.authorized_user_id)(handlers.handle_message),
        )
    )

    # Global error handler
    app.add_error_handler(handlers.handle_error)

    # Python 3.10+ no longer auto-creates an event loop outside async context.
    # python-telegram-bot's run_polling() calls asyncio.get_event_loop() internally,
    # so we must set one explicitly before calling it.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    logger.info("bot polling started")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
