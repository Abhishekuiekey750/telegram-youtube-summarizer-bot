"""
Telegram Bot Client Wrapper
Production-safe version compatible with external asyncio loop.
"""

import asyncio
from typing import Optional, Dict, Any, Union, List
from datetime import datetime

from telegram import Bot, Update, Message, InlineKeyboardMarkup, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.error import (
    TimedOut,
    NetworkError,
    RetryAfter,
    BadRequest,
)

try:
    from telegram.error import Unauthorized
except ImportError:
    from telegram.error import InvalidToken as Unauthorized

from telegram.ext import (
    Application,
    ApplicationBuilder,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

import structlog

from internal.pkg.errors import (
    BotError,
    ErrorKind,
    RateLimitError,
    UnauthorizedError,
    ValidationError,
)
from internal.pkg.logger import StructuredLogger
from internal.pkg.metrics import MetricsCollector
from internal.pkg.retry import RetryConfig
from internal.domain.value_objects import Language
from internal.bot.keyboard import KeyboardBuilder


class TelegramBotClient:

    def __init__(
        self,
        token: str,
        dispatcher=None,
        logger: Optional[StructuredLogger] = None,
        metrics: Optional[MetricsCollector] = None,
        retry_config: Optional[RetryConfig] = None,
        allowed_updates: Optional[List[str]] = None,
    ):
        self.token = token
        self._dispatcher = dispatcher
        self.logger = logger or structlog.get_logger(__name__)
        self.metrics = metrics or MetricsCollector("telegram_client")

        self.retry_config = retry_config or RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0,
            exponential_base=2,
        )

        self.allowed_updates = allowed_updates or [
            "message",
            "callback_query",
            "edited_message",
        ]

        self._application: Optional[Application] = None
        self._bot: Optional[Bot] = None
        self._is_running = False

        self.MAX_MESSAGE_LENGTH = 4096
        self.MAX_CAPTION_LENGTH = 1024

        self.logger.info(
            "telegram_client.initialized",
            token_preview=f"{token[:8]}...",
        )

    # ============================================================
    # INITIALIZATION
    # ============================================================

    async def initialize(self) -> None:
        try:
            self._application = (
                ApplicationBuilder()
                .token(self.token)
                .concurrent_updates(True)
                .build()
            )

            if self._dispatcher:

                async def handle_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
                    await self._dispatcher.dispatch(update)

                self._application.add_handler(
                    MessageHandler(filters.ALL, handle_update)
                )
                self._application.add_handler(
                    CallbackQueryHandler(handle_update)
                )

            self._bot = self._application.bot

            await self._bot.get_me()

            self.logger.info(
                "telegram_client.initialized_successfully",
                bot_username=self._bot.username,
            )

        except Unauthorized as e:
            raise UnauthorizedError("Invalid Telegram bot token") from e
        except Exception as e:
            raise BotError(f"Initialization failed: {str(e)}") from e

    # ============================================================
    # POLLING (SAFE FOR EXISTING EVENT LOOP)
    # ============================================================

    async def start_polling(self, drop_pending_updates: bool = True) -> None:
        """
        Start polling safely inside existing asyncio loop.
        """
        if self._is_running:
            return

        if not self._application:
            await self.initialize()

        self._is_running = True

        try:
            await self._application.initialize()
            await self._application.start()

            await self._application.updater.start_polling(
                drop_pending_updates=drop_pending_updates,
                allowed_updates=self.allowed_updates,
            )

            self.logger.info("telegram_client.polling_started")

        except Exception:
            self.logger.exception("telegram_client.polling_error")
            raise

    async def stop(self) -> None:
        self.logger.info("telegram_client.stopping")
        self._is_running = False

        if self._application:
            try:
                await self._application.updater.stop()
                await self._application.stop()
                await self._application.shutdown()
                self.logger.info("telegram_client.stopped")
            except Exception as e:
                self.logger.exception("telegram_client.stop_error", error=str(e))

        await self.metrics.flush()

    # ============================================================
    # SEND MESSAGE
    # ============================================================
    
    async def send_message(
        self,
        chat_id: Union[int, str],
        text: str,
        parse_mode: str = ParseMode.HTML,
        disable_web_page_preview: bool = True,
        reply_to_message_id: Optional[int] = None,
        reply_markup: Optional[Union[InlineKeyboardMarkup, ReplyKeyboardMarkup]] = None,
        **kwargs,
    ) -> Message:

        if not text or not text.strip():
            raise ValidationError("Message text cannot be empty")

        if len(text) > self.MAX_MESSAGE_LENGTH:
            text = text[: self.MAX_MESSAGE_LENGTH - 3] + "..."

        if not self._bot:
            await self.initialize()

        message = await self._bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=parse_mode,
            disable_web_page_preview=disable_web_page_preview,
            reply_to_message_id=reply_to_message_id,
            reply_markup=reply_markup,
            **kwargs,
        )

        self.metrics.increment("messages_sent")
        return message
    
    async def send_structured_summary(
        self,
        chat_id: Union[int, str],
        summary: Dict[str, Any],
        language: Language,
    ) -> Message:
        """
        Send a structured summary as a nicely formatted message.
        
        The original code expected a rich helper; for this assignment we
        simply format the summary into Markdown and send it as a single
        message.
        """
        title = summary.get("title", "Video summary")
        key_points = summary.get("key_points", []) or []
        core_takeaway = summary.get("core_takeaway", "")
        video_url = summary.get("video_url", "")
        
        lines: List[str] = []
        lines.append(f"📹 *{title}*")
        if video_url:
            lines.append(f"{video_url}")
        lines.append("")
        lines.append("🔑 *Key Points:*")
        if not key_points:
            lines.append("• (no key points available)")
        else:
            for idx, point in enumerate(key_points, 1):
                ts = point.get("timestamp")
                text = point.get("point", "")
                if ts:
                    lines.append(f"{idx}. `{ts}` - {text}")
                else:
                    lines.append(f"{idx}. {text}")
        if core_takeaway:
            lines.append("")
            lines.append("💡 *Core Takeaway:*")
            lines.append(core_takeaway)
        
        text = "\n".join(lines)
        return await self.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=ParseMode.MARKDOWN,
        )
    
    async def send_typing_action(
        self,
        chat_id: Union[int, str],
        duration_seconds: Optional[float] = None,
    ) -> None:
        """
        Show typing indicator while processing.
        Safe fallback implementation.
        """
        try:
            if not self._bot:
                await self.initialize()
                
            await self._bot.send_chat_action(
              chat_id=chat_id,
              action="typing",
            )

        except Exception:
            # Typing indicator is non-critical
            pass
    
    async def send_with_keyboard(
        self,
        chat_id: Union[int, str],
        text: str,
        buttons: List[List[Dict[str, Any]]],
        keyboard_type: str = "inline",
    ) -> Message:
        """
        Send a message with an inline or reply keyboard.
        
        This is a lightweight implementation sufficient for this project.
        """
        if keyboard_type == "inline":
            keyboard = KeyboardBuilder.build_inline(buttons)
        else:
            keyboard = KeyboardBuilder.build_reply(buttons)
        
        return await self.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=keyboard,
        )
    # ============================================================
    # HEALTH CHECK
    # ============================================================

    async def health_check(self) -> Dict[str, Any]:
        try:
            me = await self._bot.get_me()
            return {
                "status": "healthy",
                "bot_username": me.username,
                "is_running": self._is_running,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }