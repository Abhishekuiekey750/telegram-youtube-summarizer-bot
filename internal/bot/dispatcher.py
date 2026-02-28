"""
Telegram Update Dispatcher
Routes incoming updates to appropriate handlers through middleware pipeline
"""

import asyncio
from typing import (
    Dict, 
    Any, 
    Callable, 
    Awaitable, 
    Optional, 
    List, 
    Union,
    Pattern,
)
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import re
from abc import ABC, abstractmethod

from telegram import Update, Message
import structlog

from internal.pkg.logger import StructuredLogger
from internal.pkg.metrics import MetricsCollector
from internal.pkg.errors import BotError, ErrorKind
from internal.bot.client import TelegramBotClient


class UpdateType(Enum):
    """Types of updates we can receive"""
    MESSAGE = "message"
    CALLBACK_QUERY = "callback_query"
    EDITED_MESSAGE = "edited_message"
    CHANNEL_POST = "channel_post"
    INLINE_QUERY = "inline_query"


class MessageType(Enum):
    """Types of messages we can handle"""
    COMMAND = "command"
    YOUTUBE_LINK = "youtube_link"
    QUESTION = "question"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass
class UpdateContext:
    """
    Context object passed through middleware pipeline to handlers.
    Immutable - middleware should create new contexts if they need to add data.
    """
    update: Update
    chat_id: Optional[int] = None
    user_id: Optional[int] = None
    message_text: Optional[str] = None
    message_type: Optional[MessageType] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_update(cls, update: Update) -> "UpdateContext":
        """Create context from Telegram update"""
        chat_id = None
        user_id = None
        message_text = None
        
        if update.effective_chat:
            chat_id = update.effective_chat.id
        
        if update.effective_user:
            user_id = update.effective_user.id
        
        if update.effective_message and update.effective_message.text:
            message_text = update.effective_message.text
        
        return cls(
            update=update,
            chat_id=chat_id,
            user_id=user_id,
            message_text=message_text,
        )


# Middleware Types
NextMiddleware = Callable[[UpdateContext], Awaitable[Optional[UpdateContext]]]
Middleware = Callable[[UpdateContext, NextMiddleware], Awaitable[Optional[UpdateContext]]]


class Handler(ABC):
    """Base interface for all handlers"""
    
    @abstractmethod
    async def handle(self, context: UpdateContext) -> None:
        """Handle an update"""
        pass
    
    @property
    @abstractmethod
    def handler_name(self) -> str:
        """Name of the handler for logging"""
        pass


class MessageClassifier(ABC):
    """Base interface for message classifiers"""
    
    @abstractmethod
    async def can_handle(self, context: UpdateContext) -> bool:
        """Determine if this classifier can handle the message"""
        pass
    
    @abstractmethod
    def get_message_type(self) -> MessageType:
        """Get the message type this classifier handles"""
        pass


class CommandClassifier(MessageClassifier):
    """Classifies command messages (/start, /help, etc.)"""
    
    def __init__(self, commands: List[str]):
        self.commands = commands
        self.pattern = re.compile(r"^/(\w+)(@\w+)?(\s.*)?$")
    
    async def can_handle(self, context: UpdateContext) -> bool:
        if not context.message_text:
            return False
        
        match = self.pattern.match(context.message_text.strip())
        if not match:
            return False
        
        command = match.group(1)
        return command in self.commands
    
    def get_message_type(self) -> MessageType:
        return MessageType.COMMAND


class YouTubeLinkClassifier(MessageClassifier):
    """Classifies YouTube links"""
    
    # YouTube URL patterns
    YOUTUBE_PATTERNS = [
        re.compile(r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/watch\?v=([^&\s]+)"),
        re.compile(r"(https?://)?(www\.)?youtu\.be/([^\?\s]+)"),
        re.compile(r"(https?://)?(www\.)?youtube\.com/shorts/([^\?\s]+)"),
        re.compile(r"(https?://)?(www\.)?youtube\.com/embed/([^\?\s]+)"),
    ]
    
    async def can_handle(self, context: UpdateContext) -> bool:
        if not context.message_text:
            return False
        
        text = context.message_text.strip()
        for pattern in self.YOUTUBE_PATTERNS:
            if pattern.search(text):
                return True
        return False
    
    def get_message_type(self) -> MessageType:
        return MessageType.YOUTUBE_LINK


class QuestionClassifier(MessageClassifier):
    """
    Classifies questions.
    This is intentionally broad - if it's not a command or link, treat as question.
    The QA service will handle "I don't know" responses if not in transcript.
    """
    
    async def can_handle(self, context: UpdateContext) -> bool:
        # If we have text and it's not a command or link, it's a question
        if not context.message_text:
            return False
        
        text = context.message_text.strip()
        
        # Empty message
        if not text:
            return False
        
        # Commands are handled separately
        if text.startswith('/'):
            return False
        
        # Links are handled separately (but check if it's just a link)
        # This is a simple heuristic - could be more sophisticated
        if 'youtube.com' in text or 'youtu.be' in text:
            return False
        
        # If we get here, treat as question
        return True
    
    def get_message_type(self) -> MessageType:
        return MessageType.QUESTION


class Dispatcher:
    """
    Main dispatcher that routes updates to handlers through middleware pipeline.
    
    Features:
    - Pluggable middleware pipeline
    - Extensible message classification
    - Handler registry
    - Metrics and logging
    - Error recovery
    """
    
    def __init__(
        self,
        telegram_client: Optional[TelegramBotClient] = None,
        logger: Optional[StructuredLogger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        """
        Initialize the dispatcher.
        
        Args:
            telegram_client: Client for sending responses (can be set later)
            logger: Structured logger instance
            metrics: Metrics collector instance
        """
        self.telegram_client = telegram_client
        self.logger = logger or structlog.get_logger(__name__)
        self.metrics = metrics or MetricsCollector("dispatcher")
        
        # Middleware pipeline
        self._middleware: List[Middleware] = []
        
        # Message classifiers
        self._classifiers: Dict[MessageType, MessageClassifier] = {}
        
        # Handler registry
        self._handlers: Dict[MessageType, Handler] = {}
        self._callback_handlers: Dict[str, Handler] = {}  # For callback queries
        
        # Command registry
        self._commands: List[str] = []
        
        self.logger.info("dispatcher.initialized")
    
    def use(self, middleware: Middleware) -> "Dispatcher":
        """
        Register middleware in the pipeline.
        Middleware executes in the order they are registered.
        
        Args:
            middleware: Middleware function
            
        Returns:
            Self for chaining
        """
        self._middleware.append(middleware)
        middleware_name = getattr(middleware, "__name__", getattr(middleware.__class__, "__name__", "middleware"))
        self.logger.debug(
            "dispatcher.middleware_registered",
            middleware=middleware_name,
            position=len(self._middleware),
        )
        return self
    
    def register_classifier(self, classifier: MessageClassifier) -> "Dispatcher":
        """
        Register a message classifier.
        
        Args:
            classifier: Classifier instance
            
        Returns:
            Self for chaining
        """
        message_type = classifier.get_message_type()
        self._classifiers[message_type] = classifier
        
        if message_type == MessageType.COMMAND:
            if isinstance(classifier, CommandClassifier):
                self._commands.extend(classifier.commands)
        
        self.logger.debug(
            "dispatcher.classifier_registered",
            message_type=message_type.value,
            classifier=classifier.__class__.__name__,
        )
        return self
    
    def register_handler(
        self,
        message_type: MessageType,
        handler: Handler,
    ) -> "Dispatcher":
        """
        Register a handler for a message type.
        
        Args:
            message_type: Type of message to handle
            handler: Handler instance
            
        Returns:
            Self for chaining
        """
        self._handlers[message_type] = handler
        self.logger.info(
            "dispatcher.handler_registered",
            message_type=message_type.value,
            handler=handler.handler_name,
        )
        return self
    
    def register_callback_handler(
        self,
        callback_data: str,
        handler: Handler,
    ) -> "Dispatcher":
        """
        Register a handler for callback queries.
        
        Args:
            callback_data: Callback data pattern
            handler: Handler instance
        """
        self._callback_handlers[callback_data] = handler
        self.logger.debug(
            "dispatcher.callback_handler_registered",
            callback_data=callback_data,
            handler=handler.handler_name,
        )
        return self
    
    async def dispatch(self, update: Update) -> None:
        """
        Main entry point for processing an update.
        
        Args:
            update: Telegram update object
        """
        start_time = datetime.now()
        
        try:
            # Create context
            context = UpdateContext.from_update(update)
            
            self.logger.debug(
                "dispatcher.received_update",
                update_id=update.update_id,
                chat_id=context.chat_id,
                user_id=context.user_id,
                has_text=bool(context.message_text),
            )
            
            # Run through middleware pipeline
            final_context = await self._run_middleware(context)
            
            if not final_context:
                self.logger.debug(
                    "dispatcher.update_consumed_by_middleware",
                    update_id=update.update_id,
                )
                return
            
            # Process based on update type
            if update.callback_query:
                await self._handle_callback(final_context)
            elif update.message or update.edited_message:
                await self._handle_message(final_context)
            else:
                self.logger.warning(
                    "dispatcher.unsupported_update",
                    update_id=update.update_id,
                    update_type=type(update).__name__,
                )
            
            # Record metrics
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics.record_latency("dispatch", duration)
            self.metrics.increment("updates_processed")
            
        except Exception as e:
            self.logger.exception(
                "dispatcher.dispatch_failed",
                update_id=update.update_id,
                error=str(e),
            )
            self.metrics.increment("dispatch_errors")
            
            # Try to notify user of error
            await self._send_error_message(update, e)
    
    async def _run_middleware(
        self,
        initial_context: UpdateContext,
    ) -> Optional[UpdateContext]:
        """
        Run the middleware pipeline.
        
        Args:
            initial_context: Starting context
            
        Returns:
            Final context or None if middleware stopped
        """
        if not self._middleware:
            return initial_context
        
        async def run_middleware(
            index: int,
            context: UpdateContext,
        ) -> Optional[UpdateContext]:
            if index >= len(self._middleware):
                return context
            
            middleware = self._middleware[index]
            
            async def next_middleware(ctx: UpdateContext) -> Optional[UpdateContext]:
                return await run_middleware(index + 1, ctx)
            
            try:
                return await middleware(context, next_middleware)
            except Exception as e:
                self.logger.exception(
                    "dispatcher.middleware_failed",
                    middleware=middleware.__class__.__name__,
                    error=str(e),
                )
                raise
        
        return await run_middleware(0, initial_context)
    
    async def _classify_message(self, context: UpdateContext) -> Optional[MessageType]:
        """
        Classify a message using registered classifiers.
        
        Args:
            context: Update context
            
        Returns:
            Message type or None if can't classify
        """
        for message_type, classifier in self._classifiers.items():
            try:
                if await classifier.can_handle(context):
                    self.logger.debug(
                        "dispatcher.message_classified",
                        message_type=message_type.value,
                        classifier=classifier.__class__.__name__,
                    )
                    return message_type
            except Exception as e:
                self.logger.warning(
                    "dispatcher.classifier_failed",
                    classifier=classifier.__class__.__name__,
                    error=str(e),
                )
                continue
        
        return MessageType.UNKNOWN
    
    async def _handle_message(self, context: UpdateContext) -> None:
        """
        Handle a message update.
        
        Args:
            context: Update context
        """
        # Classify the message
        message_type = await self._classify_message(context)
        context.message_type = message_type
        
        # Find handler
        handler = self._handlers.get(message_type)
        if not handler:
            self.logger.warning(
                "dispatcher.no_handler_for_type",
                message_type=message_type.value,
            )
            # Send fallback message
            if self.telegram_client and context.chat_id:
                await self.telegram_client.send_message(
                    chat_id=context.chat_id,
                    text="I don't understand that command. Send a YouTube link to get started!",
                )
            return
        
        # Execute handler
        try:
            self.logger.info(
                "dispatcher.executing_handler",
                handler=handler.handler_name,
                message_type=message_type.value,
                chat_id=context.chat_id,
            )
            
            self.metrics.increment(
                "handler_execution",
                tags={"handler": handler.handler_name},
            )
            
            await handler.handle(context)
            
        except Exception as e:
            self.logger.exception(
                "dispatcher.handler_failed",
                handler=handler.handler_name,
                error=str(e),
            )
            self.metrics.increment(
                "handler_errors",
                tags={"handler": handler.handler_name},
            )
            raise
    
    async def _handle_callback(self, context: UpdateContext) -> None:
        """
        Handle a callback query update.
        
        Args:
            context: Update context
        """
        callback_query = context.update.callback_query
        callback_data = callback_query.data
        
        # Find handler for this callback data
        handler = self._callback_handlers.get(callback_data)
        
        if not handler:
            self.logger.warning(
                "dispatcher.no_callback_handler",
                callback_data=callback_data,
            )
            await callback_query.answer(
                text="This button is no longer valid",
                show_alert=False,
            )
            return
        
        try:
            self.logger.info(
                "dispatcher.executing_callback_handler",
                handler=handler.handler_name,
                callback_data=callback_data,
            )
            
            await handler.handle(context)
            
        except Exception as e:
            self.logger.exception(
                "dispatcher.callback_handler_failed",
                handler=handler.handler_name,
                callback_data=callback_data,
                error=str(e),
            )
            await callback_query.answer(
                text="An error occurred",
                show_alert=True,
            )
    
    async def _send_error_message(self, update: Update, error: Exception) -> None:
        """
        Send a user-friendly error message.
        
        Args:
            update: Original update
            error: The exception that occurred
        """
        try:
            chat_id = update.effective_chat.id if update.effective_chat else None
            
            if not chat_id:
                return
            
            # Different messages for different error types
            if isinstance(error, BotError):
                if error.kind == ErrorKind.RATE_LIMIT:
                    text = "⏳ Too many requests. Please wait a moment and try again."
                elif error.kind == ErrorKind.UNAUTHORIZED:
                    text = "🔒 I don't have permission to do that."
                elif error.kind == ErrorKind.VALIDATION:
                    text = f"❌ Invalid input: {str(error)}"
                else:
                    text = "😵 Something went wrong. Please try again later."
            else:
                text = "😵 An unexpected error occurred. Please try again."
            
            if self.telegram_client:
                await self.telegram_client.send_message(
                    chat_id=chat_id,
                    text=text,
                )
            
        except Exception as e:
            self.logger.error(
                "dispatcher.failed_to_send_error",
                error=str(e),
            )


# Middleware Implementations

async def logging_middleware(
    context: UpdateContext,
    next_middleware: NextMiddleware,
) -> Optional[UpdateContext]:
    """
    Middleware for logging all updates.
    Should be one of the first in the pipeline.
    """
    logger = structlog.get_logger(__name__)
    
    logger.info(
        "middleware.logging.processing",
        chat_id=context.chat_id,
        user_id=context.user_id,
        text_preview=context.message_text[:50] if context.message_text else None,
    )
    
    result = await next_middleware(context)
    
    logger.debug("middleware.logging.completed")
    
    return result


async def auth_middleware(
    context: UpdateContext,
    next_middleware: NextMiddleware,
    allowed_users: Optional[List[int]] = None,
) -> Optional[UpdateContext]:
    """
    Middleware for authentication.
    If allowed_users is provided, only those users can access the bot.
    """
    if allowed_users and context.user_id not in allowed_users:
        logger = structlog.get_logger(__name__)
        logger.warning(
            "middleware.auth.unauthorized",
            user_id=context.user_id,
        )
        
        # You could send a message here, but we'll just stop propagation
        return None
    
    return await next_middleware(context)


async def rate_limit_middleware(
    context: UpdateContext,
    next_middleware: NextMiddleware,
    rate_limiter: Any,  # Your rate limiter implementation
) -> Optional[UpdateContext]:
    """
    Middleware for rate limiting.
    """
    if context.user_id:
        is_allowed = await rate_limiter.check_and_increment(
            key=f"user:{context.user_id}",
            max_requests=60,
            window_seconds=60,
        )
        
        if not is_allowed:
            logger = structlog.get_logger(__name__)
            logger.warning(
                "middleware.rate_limit.exceeded",
                user_id=context.user_id,
            )
            return None
    
    return await next_middleware(context)


async def recovery_middleware(
    context: UpdateContext,
    next_middleware: NextMiddleware,
) -> Optional[UpdateContext]:
    """
    Middleware for panic recovery.
    Should be the first middleware in the pipeline.
    """
    try:
        return await next_middleware(context)
    except Exception as e:
        logger = structlog.get_logger(__name__)
        logger.exception(
            "middleware.recovery.caught_panic",
            error=str(e),
            chat_id=context.chat_id,
        )
        # Don't propagate - we handled it
        return None