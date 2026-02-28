"""
Base Handler Interface
Defines the contract for all Telegram bot handlers with shared utilities
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
from datetime import datetime
import traceback
from functools import wraps

import structlog
from telegram import Update, Message

from internal.bot.client import TelegramBotClient
from internal.bot.dispatcher import UpdateContext
from internal.pkg.logger import StructuredLogger
from internal.pkg.metrics import MetricsCollector
from internal.pkg.errors import BotError, ErrorKind, ValidationError
from internal.domain.value_objects import Language


def handler_error_boundary(func):
    """
    Decorator to provide error boundary for handler methods.
    Ensures consistent error handling across all handlers.
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except Exception as e:
            # Log the error with context
            self.logger.exception(
                "handler.error_boundary.caught",
                handler=self.handler_name,
                error=str(e),
                traceback=traceback.format_exc(),
            )
            
            # Track metrics
            self.metrics.increment(
                "handler.errors",
                tags={
                    "handler": self.handler_name,
                    "error_type": e.__class__.__name__,
                },
            )
            
            # Re-raise as BotError for consistent handling
            if isinstance(e, BotError):
                raise
            raise BotError(
                f"Handler error: {str(e)}",
                kind=ErrorKind.INTERNAL,
                original_error=e,
            ) from e
    return wrapper


class BaseHandler(ABC):
    """
    Abstract base class for all Telegram bot handlers.
    
    All concrete handlers must inherit from this class and implement:
    - handle(): Core business logic
    
    Provides shared utilities for:
    - Input validation
    - Typing indicators
    - Response formatting
    - Error handling
    - Metrics tracking
    - Language support
    """
    
    def __init__(
        self,
        telegram_client: TelegramBotClient,
        logger: Optional[StructuredLogger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        """
        Initialize base handler with shared dependencies.
        
        Args:
            telegram_client: Client for sending responses
            logger: Structured logger instance
            metrics: Metrics collector instance
        """
        self.telegram_client = telegram_client
        self.logger = logger or structlog.get_logger(__name__)
        self.metrics = metrics or MetricsCollector(f"handler.{self.handler_name}")
        
        # Configuration
        self.max_retries = 3
        self.typing_indicator_duration = 5.0  # seconds
        
        self.logger.debug(
            "handler.initialized",
            handler=self.handler_name,
        )
    
    @property
    @abstractmethod
    def handler_name(self) -> str:
        """Unique name of the handler for logging and metrics"""
        pass
    
    @abstractmethod
    @handler_error_boundary
    async def handle(self, context: UpdateContext) -> None:
        """
        Handle an incoming update.
        
        This is the main entry point for all handlers. Must be implemented
        by concrete handlers.
        
        Args:
            context: Update context with message and metadata
            
        Raises:
            ValidationError: If input validation fails
            BotError: For other handler-specific errors
        """
        pass
    
    # ------------------------------------------------------------------------
    # Shared Validation Utilities
    # ------------------------------------------------------------------------
    
    async def _validate_required_fields(
        self,
        context: UpdateContext,
        required_fields: list[str],
    ) -> None:
        """
        Validate that required fields exist in context.
        
        Args:
            context: Update context
            required_fields: List of field names to check
            
        Raises:
            ValidationError: If any required field is missing
        """
        missing = []
        
        for field in required_fields:
            if not hasattr(context, field) or getattr(context, field) is None:
                missing.append(field)
        
        if missing:
            raise ValidationError(
                f"Missing required fields: {', '.join(missing)}",
                kind=ErrorKind.VALIDATION,
                context={"missing_fields": missing},
            )
    
    def _validate_text_length(
        self,
        text: str,
        min_length: int = 1,
        max_length: int = 4096,
        field_name: str = "text",
    ) -> str:
        """
        Validate and optionally truncate text length.
        
        Args:
            text: Text to validate
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            field_name: Name of field for error messages
            
        Returns:
            Validated (and possibly truncated) text
            
        Raises:
            ValidationError: If text is too short
        """
        if not text or len(text.strip()) < min_length:
            raise ValidationError(
                f"{field_name} is too short (min {min_length} characters)",
                kind=ErrorKind.VALIDATION,
            )
        
        if len(text) > max_length:
            self.logger.warning(
                "handler.text_truncated",
                field=field_name,
                original_length=len(text),
                max_length=max_length,
            )
            return text[:max_length]
        
        return text
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract YouTube video ID from various URL formats.
        
        Args:
            url: YouTube URL
            
        Returns:
            Video ID or None if not found
        """
        import re
        
        patterns = [
            r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
            r"^([a-zA-Z0-9_-]{11})$",  # Just the ID itself
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    # ------------------------------------------------------------------------
    # Shared Response Utilities
    # ------------------------------------------------------------------------
    
    async def _send_typing_indicator(self, chat_id: Union[int, str]) -> None:
        """
        Show typing indicator while processing.
        
        Args:
            chat_id: Target chat ID
        """
        try:
            await self.telegram_client.send_typing_action(
                chat_id=chat_id,
                duration_seconds=self.typing_indicator_duration,
            )
        except Exception as e:
            # Non-critical, just log
            self.logger.debug(
                "handler.typing_indicator_failed",
                chat_id=chat_id,
                error=str(e),
            )
    
    async def _send_response(
        self,
        chat_id: Union[int, str],
        text: str,
        parse_mode: str = "HTML",
        reply_to_message_id: Optional[int] = None,
        **kwargs,
    ) -> Optional[Message]:
        """
        Send a response with consistent error handling.
        
        Args:
            chat_id: Target chat ID
            text: Response text
            parse_mode: Parse mode for formatting
            reply_to_message_id: Message to reply to
            **kwargs: Additional arguments for send_message
            
        Returns:
            Sent message or None if failed
        """
        try:
            # Validate text
            text = self._validate_text_length(text, max_length=4096)
            
            # Send message
            message = await self.telegram_client.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode,
                reply_to_message_id=reply_to_message_id,
                **kwargs,
            )
            
            # Track metrics
            self.metrics.increment(
                "responses.sent",
                tags={"parse_mode": parse_mode},
            )
            
            return message
            
        except Exception as e:
            self.logger.exception(
                "handler.send_response_failed",
                chat_id=chat_id,
                error=str(e),
            )
            return None
    
    async def _send_error_message(
        self,
        chat_id: Union[int, str],
        error: Union[str, Exception],
        user_friendly: bool = True,
    ) -> None:
        """
        Send a user-friendly error message.
        
        Args:
            chat_id: Target chat ID
            error: Error message or exception
            user_friendly: Whether to show user-friendly message or technical details
        """
        if isinstance(error, Exception):
            error_str = str(error)
        else:
            error_str = error
        
        # Determine message based on error type
        if user_friendly:
            if "rate limit" in error_str.lower():
                message = "⏳ Too many requests. Please wait a moment and try again."
            elif "not found" in error_str.lower():
                message = "🔍 The requested information could not be found."
            elif "transcript" in error_str.lower():
                message = "📝 This video doesn't have a transcript available."
            elif "invalid" in error_str.lower():
                message = "❌ Invalid input. Please check and try again."
            else:
                message = "😵 Something went wrong. Please try again later."
        else:
            message = f"❌ Error: {error_str}"
        
        await self._send_response(chat_id, message)
    
    # ------------------------------------------------------------------------
    # Shared Language Utilities
    # ------------------------------------------------------------------------
    
    def _detect_language_from_text(self, text: str) -> Language:
        """
        Simple language detection based on script.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected language
        """
        # Check for Devanagari (Hindi, Sanskrit, etc.)
        if any('\u0900' <= char <= '\u097F' for char in text):
            return Language.HINDI
        
        # Check for Tamil
        if any('\u0B80' <= char <= '\u0BFF' for char in text):
            return Language.TAMIL
        
        # Check for Telugu
        if any('\u0C00' <= char <= '\u0C7F' for char in text):
            return Language.TELUGU
        
        # Check for Kannada
        if any('\u0C80' <= char <= '\u0CFF' for char in text):
            return Language.KANNADA
        
        # Check for Malayalam
        if any('\u0D00' <= char <= '\u0D7F' for char in text):
            return Language.MALAYALAM
        
        # Check for Bengali
        if any('\u0980' <= char <= '\u09FF' for char in text):
            return Language.BENGALI
        
        # Default to English
        return Language.ENGLISH
    
    def _get_response_language(self, context: UpdateContext) -> Language:
        """
        Determine which language to respond in.
        
        Priority:
        1. Language explicitly set in session
        2. Language detected from user message
        3. Default (English)
        
        Args:
            context: Update context
            
        Returns:
            Language to use for response
        """
        # Check if language is in context metadata (set by session middleware)
        if context.metadata and "language" in context.metadata:
            return context.metadata["language"]
        
        # Detect from message text
        if context.message_text:
            return self._detect_language_from_text(context.message_text)
        
        return Language.ENGLISH
    
    # ------------------------------------------------------------------------
    # Shared Metrics Utilities
    # ------------------------------------------------------------------------
    
    def _track_processing_time(self, start_time: datetime) -> None:
        """
        Track handler processing time.
        
        Args:
            start_time: When processing started
        """
        duration = (datetime.now() - start_time).total_seconds()
        self.metrics.record_latency(
            "processing_time",
            duration,
            tags={"handler": self.handler_name},
        )
    
    def _track_user_activity(self, user_id: int, action: str) -> None:
        """
        Track user activity for analytics.
        
        Args:
            user_id: User ID
            action: Action being performed
        """
        self.metrics.increment(
            "user_activity",
            tags={
                "handler": self.handler_name,
                "action": action,
            },
        )
    
    # ------------------------------------------------------------------------
    # Shared Retry Logic
    # ------------------------------------------------------------------------
    
    async def _with_retry(
        self,
        func,
        *args,
        retry_count: int = 0,
        **kwargs,
    ):
        """
        Execute a function with retry logic.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for func
            retry_count: Current retry attempt
            **kwargs: Keyword arguments for func
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries fail
        """
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if retry_count >= self.max_retries:
                self.logger.error(
                    "handler.max_retries_exceeded",
                    func=func.__name__,
                    retries=retry_count,
                )
                raise
            
            # Exponential backoff
            wait_time = 2 ** retry_count
            self.logger.warning(
                "handler.retrying",
                func=func.__name__,
                attempt=retry_count + 1,
                max_retries=self.max_retries,
                wait_time=wait_time,
                error=str(e),
            )
            
            await asyncio.sleep(wait_time)
            return await self._with_retry(func, *args, retry_count=retry_count + 1, **kwargs)
    
    # ------------------------------------------------------------------------
    # Shared Formatting Utilities
    # ------------------------------------------------------------------------
    
    def _format_timestamp(self, seconds: float) -> str:
        """
        Format seconds as MM:SS timestamp.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp (e.g., "05:30")
        """
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes:02d}:{remaining_seconds:02d}"
    
    def _escape_markdown(self, text: str) -> str:
        """
        Escape Markdown special characters for Telegram.
        
        Args:
            text: Raw text
            
        Returns:
            Escaped text
        """
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        return text
    
    def _truncate_text(self, text: str, max_length: int, ellipsis: str = "...") -> str:
        """
        Truncate text to specified length with ellipsis.
        
        Args:
            text: Text to truncate
            max_length: Maximum length
            ellipsis: Ellipsis string
            
        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        
        return text[:max_length - len(ellipsis)] + ellipsis
    
    # ------------------------------------------------------------------------
    # Lifecycle Hooks
    # ------------------------------------------------------------------------
    
    async def before_handle(self, context: UpdateContext) -> None:
        """
        Hook called before handling.
        Can be overridden by concrete handlers.
        
        Args:
            context: Update context
        """
        self.metrics.increment("handles.started")
        
        # Send typing indicator
        if context.chat_id:
            await self._send_typing_indicator(context.chat_id)
    
    async def after_handle(self, context: UpdateContext, result: Any = None) -> None:
        """
        Hook called after successful handling.
        Can be overridden by concrete handlers.
        
        Args:
            context: Update context
            result: Result from handle method
        """
        self.metrics.increment("handles.completed")
    
    async def on_error(self, context: UpdateContext, error: Exception) -> None:
        """
        Hook called when handling fails.
        Can be overridden by concrete handlers.
        
        Args:
            context: Update context
            error: The exception that occurred
        """
        self.metrics.increment("handles.failed")
        
        # Send error message to user
        if context.chat_id:
            await self._send_error_message(context.chat_id, error)


# ------------------------------------------------------------------------
# Concrete Handler Examples (to illustrate usage)
# ------------------------------------------------------------------------

class CommandHandler(BaseHandler):
    """Example concrete handler for commands"""
    
    @property
    def handler_name(self) -> str:
        return "command"
    
    async def handle(self, context: UpdateContext) -> None:
        start_time = datetime.now()
        
        try:
            await self.before_handle(context)
            
            # Validate
            await self._validate_required_fields(context, ["chat_id", "message_text"])
            
            # Extract command
            text = context.message_text.strip()
            command = text.split()[0].lower()
            
            # Handle different commands
            if command == "/start":
                response = "Welcome! Send me a YouTube link to get started."
            elif command == "/help":
                response = self._get_help_text()
            elif command == "/language":
                response = self._get_language_options()
            else:
                response = f"Unknown command: {command}"
            
            # Send response
            await self._send_response(
                chat_id=context.chat_id,
                text=response,
                reply_to_message_id=context.update.effective_message.message_id,
            )
            
            # Track metrics
            self._track_user_activity(context.user_id, f"command_{command}")
            
            await self.after_handle(context)
            
        finally:
            self._track_processing_time(start_time)
    
    def _get_help_text(self) -> str:
        return """
🤖 *YouTube Summarizer Bot*

*Commands:*
/start - Start the bot
/help - Show this help
/language - Change language

*How to use:*
1. Send any YouTube link
2. Get instant summary
3. Ask questions about the video

*Examples:*
• "What are the key points?"
• "Summarize in Hindi"
• "Explain the pricing section"
        """
    
    def _get_language_options(self) -> str:
        return """
🌐 *Select Language:*

English • हिन्दी • தமிழ் • తెలుగు • ಕನ್ನಡ • മലയാളം • বাংলা

Send the language name to switch.
        """


class LinkHandler(BaseHandler):
    """Example concrete handler for YouTube links"""
    
    @property
    def handler_name(self) -> str:
        return "youtube_link"
    
    async def handle(self, context: UpdateContext) -> None:
        start_time = datetime.now()
        
        try:
            await self.before_handle(context)
            
            # Validate
            await self._validate_required_fields(context, ["chat_id", "message_text"])
            
            # Extract video ID
            video_id = self._extract_video_id(context.message_text)
            if not video_id:
                await self._send_error_message(
                    context.chat_id,
                    "Invalid YouTube link. Please check and try again.",
                )
                return
            
            # Send processing message
            processing_msg = await self._send_response(
                chat_id=context.chat_id,
                text="⏳ Fetching video transcript...",
                reply_to_message_id=context.update.effective_message.message_id,
            )
            
            # Here you would call your services to process the video
            # This is just an example
            await asyncio.sleep(2)  # Simulate processing
            
            # Send success response
            await self._send_response(
                chat_id=context.chat_id,
                text=f"✅ Video found! (ID: {video_id})\nSummary coming soon...",
            )
            
            # Track metrics
            self._track_user_activity(context.user_id, "link_processed")
            
            await self.after_handle(context)
            
        except Exception as e:
            await self.on_error(context, e)
            raise
            
        finally:
            self._track_processing_time(start_time)