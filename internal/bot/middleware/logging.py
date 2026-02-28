"""
Structured Logging Middleware
Logs complete request lifecycle with context propagation

Features:
- Request/response lifecycle logging
- Context propagation with trace IDs
- Performance tracking
- Error capturing
- Structured JSON logs
- Sensitive data redaction
- Sampled logging for high-volume events
"""

import asyncio
import time
import uuid
import json
from typing import Optional, Dict, Any, Callable, Awaitable
from datetime import datetime
from functools import wraps
import traceback

import structlog
from structlog.processors import JSONRenderer

from internal.bot.dispatcher import UpdateContext, NextMiddleware
from internal.pkg.logger import StructuredLogger
from internal.pkg.metrics import MetricsCollector


class LoggingMiddleware:
    """
    Structured logging middleware that tracks complete request lifecycle.
    
    This middleware should be one of the first in the pipeline to ensure
    all subsequent operations are logged with proper context.
    
    Features:
    - Generates trace IDs for request correlation
    - Logs request start, processing, and completion
    - Captures handler and service execution times
    - Redacts sensitive information
    - Supports sampled logging for high-volume events
    - Integrates with metrics system
    """
    
    def __init__(
        self,
        logger: Optional[StructuredLogger] = None,
        metrics: Optional[MetricsCollector] = None,
        sample_rate: float = 1.0,  # Log all requests by default
        log_request_body: bool = True,
        log_response_body: bool = True,
        sensitive_fields: Optional[list] = None,
    ):
        """
        Initialize logging middleware
        
        Args:
            logger: Structured logger instance
            metrics: Metrics collector
            sample_rate: Fraction of requests to log (0.0-1.0)
            log_request_body: Whether to log message text
            log_response_body: Whether to log response text
            sensitive_fields: Fields to redact (e.g., ["token", "password"])
        """
        self.logger = logger or structlog.get_logger(__name__)
        self.metrics = metrics or MetricsCollector("logging_middleware")
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.sensitive_fields = sensitive_fields or [
            "token", "password", "secret", "key", "authorization"
        ]
        
        # Counter for sampled logging
        self._request_count = 0
        
        self.logger.info(
            "logging_middleware.initialized",
            sample_rate=self.sample_rate,
            log_request_body=self.log_request_body,
            log_response_body=self.log_response_body,
        )
    
    async def __call__(
        self,
        context: UpdateContext,
        next_middleware: NextMiddleware,
    ) -> Optional[UpdateContext]:
        """
        Execute logging middleware
        
        Args:
            context: Update context
            next_middleware: Next middleware
            
        Returns:
            Updated context
        """
        # Determine if we should log this request (sampling)
        should_log = self._should_log()
        
        # Generate or get trace ID
        trace_id = self._get_or_create_trace_id(context)
        
        # Start timing
        start_time = time.time()
        request_id = str(uuid.uuid4())[:8]
        
        # Log request start
        if should_log:
            await self._log_request_start(context, trace_id, request_id)
        
        try:
            # Add logger to context for downstream use
            if "logger" not in context.metadata:
                context.metadata["logger"] = self._create_child_logger(
                    trace_id=trace_id,
                    request_id=request_id,
                )
            
            # Add trace ID to context
            context.metadata["trace_id"] = trace_id
            context.metadata["request_id"] = request_id
            context.metadata["request_start_time"] = start_time
            
            # Execute next middleware
            result = await next_middleware(context)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log request completion
            if should_log:
                await self._log_request_complete(
                    context=context,
                    duration=duration,
                    error=None,
                )
            
            # Track metrics
            self.metrics.record_latency("request.duration", duration)
            self.metrics.increment("requests.total")
            
            return result
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log error
            if should_log:
                await self._log_request_complete(
                    context=context,
                    duration=duration,
                    error=e,
                )
            
            # Track error metrics
            self.metrics.increment(
                "requests.errors",
                tags={"error_type": e.__class__.__name__},
            )
            
            # Re-raise to let error middleware handle
            raise
        
        finally:
            # Always increment counter for sampling
            self._request_count += 1
    
    # ------------------------------------------------------------------------
    # Logging Methods
    # ------------------------------------------------------------------------
    
    async def _log_request_start(
        self,
        context: UpdateContext,
        trace_id: str,
        request_id: str,
    ) -> None:
        """
        Log the start of request processing
        
        Args:
            context: Update context
            trace_id: Correlation ID
            request_id: Unique request ID
        """
        # Build base log data
        log_data = {
            "trace_id": trace_id,
            "request_id": request_id,
            "update_id": context.update.update_id,
            "chat_id": context.chat_id,
            "user_id": context.user_id,
            "update_type": self._get_update_type(context),
        }
        
        # Add message info if available
        if context.message_text and self.log_request_body:
            log_data["message_text"] = self._redact_sensitive_data(
                context.message_text[:200]  # Limit length
            )
            log_data["message_length"] = len(context.message_text)
        
        # Add user info if available
        user = context.metadata.get("user")
        if user:
            log_data["username"] = getattr(user, "username", None)
            log_data["user_language"] = getattr(user, "language_code", None)
        
        # Log the event
        self.logger.info("request.started", **log_data)
    
    async def _log_request_complete(
        self,
        context: UpdateContext,
        duration: float,
        error: Optional[Exception] = None,
    ) -> None:
        """
        Log request completion
        
        Args:
            context: Update context
            duration: Request duration in seconds
            error: Exception if any
        """
        # Build base log data
        log_data = {
            "trace_id": context.metadata.get("trace_id"),
            "request_id": context.metadata.get("request_id"),
            "duration_ms": round(duration * 1000, 2),
            "chat_id": context.chat_id,
            "user_id": context.user_id,
        }
        
        # Add handler info
        if "handler" in context.metadata:
            log_data["handler"] = context.metadata["handler"]
        
        # Add session info
        session = context.metadata.get("session")
        if session:
            log_data["session_language"] = session.language.code
            if session.current_video_id:
                log_data["current_video_id"] = session.current_video_id
        
        # Add video info for link handlers
        if "video_id" in context.metadata:
            log_data["video_id"] = context.metadata["video_id"]
        
        if error:
            # Error case
            log_data["status"] = "error"
            log_data["error"] = error.__class__.__name__
            log_data["error_message"] = str(error)
            
            # Add stack trace for debugging
            if self.logger.isEnabledFor("DEBUG"):
                log_data["traceback"] = traceback.format_exc()
            
            self.logger.error("request.failed", **log_data)
            
        else:
            # Success case
            log_data["status"] = "success"
            
            # Add response info if available
            if "response_text" in context.metadata and self.log_response_body:
                response = context.metadata["response_text"]
                log_data["response_length"] = len(response)
                log_data["response_preview"] = self._redact_sensitive_data(
                    response[:100]  # Limit preview
                )
            
            self.logger.info("request.completed", **log_data)
    
    def _create_child_logger(
        self,
        trace_id: str,
        request_id: str,
    ) -> StructuredLogger:
        """
        Create a child logger with context bound
        
        Args:
            trace_id: Correlation ID
            request_id: Request ID
            
        Returns:
            Logger with bound context
        """
        return self.logger.bind(
            trace_id=trace_id,
            request_id=request_id,
        )
    
    # ------------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------------
    
    def _get_update_type(self, context: UpdateContext) -> str:
        """Determine the type of update"""
        update = context.update
        
        if update.message:
            if update.message.text and update.message.text.startswith("/"):
                return "command"
            elif update.message.text and ("youtube.com" in update.message.text or "youtu.be" in update.message.text):
                return "youtube_link"
            elif update.message.text:
                return "text_message"
            elif update.message.photo:
                return "photo"
            elif update.message.video:
                return "video"
            elif update.message.document:
                return "document"
            elif update.message.voice:
                return "voice"
            elif update.message.sticker:
                return "sticker"
            return "message"
        
        elif update.callback_query:
            return "callback_query"
        
        elif update.inline_query:
            return "inline_query"
        
        elif update.channel_post:
            return "channel_post"
        
        elif update.my_chat_member:
            return "chat_member_update"
        
        return "unknown"
    
    def _get_or_create_trace_id(self, context: UpdateContext) -> str:
        """Get existing trace ID or create new one"""
        # Check if trace ID already exists
        if "trace_id" in context.metadata:
            return context.metadata["trace_id"]
        
        # Generate new trace ID
        return str(uuid.uuid4())
    
    def _should_log(self) -> bool:
        """Determine if this request should be logged based on sampling rate"""
        if self.sample_rate >= 1.0:
            return True
        
        if self.sample_rate <= 0.0:
            return False
        
        # Simple counter-based sampling
        return (self._request_count % int(1.0 / self.sample_rate)) == 0
    
    def _redact_sensitive_data(self, text: str) -> str:
        """
        Redact sensitive information from log messages
        
        Args:
            text: Original text
            
        Returns:
            Redacted text
        """
        if not text:
            return text
        
        redacted = text
        
        # Redact common sensitive patterns
        for field in self.sensitive_fields:
            # Pattern: field=value or field: value
            import re
            pattern = rf'({field}[=:]\s*)([^\s,&]+)'
            redacted = re.sub(pattern, r'\1[REDACTED]', redacted, flags=re.IGNORECASE)
        
        # Redact potential tokens (long strings of letters/numbers)
        token_pattern = r'([a-zA-Z0-9_-]{20,})'
        redacted = re.sub(token_pattern, '[REDACTED_TOKEN]', redacted)
        
        return redacted


# ------------------------------------------------------------------------
# Decorators for Method-Level Logging
# ------------------------------------------------------------------------

def log_method(logger=None):
    """
    Decorator to log method entry/exit with timing
    
    Usage:
        @log_method()
        async def my_method(self, arg1, arg2):
            pass
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                # Try to get logger from instance
                instance = args[0] if args else None
                logger = getattr(instance, 'logger', structlog.get_logger(__name__))
            
            method_name = func.__qualname__
            logger.debug(f"method.entered", method=method_name)
            
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start
                logger.debug(
                    "method.exited",
                    method=method_name,
                    duration_ms=round(duration * 1000, 2),
                )
                return result
            except Exception as e:
                duration = time.time() - start
                logger.error(
                    "method.failed",
                    method=method_name,
                    duration_ms=round(duration * 1000, 2),
                    error=str(e),
                )
                raise
        
        return async_wrapper
    return decorator


# ------------------------------------------------------------------------
# Context Manager for Operation Logging
# ------------------------------------------------------------------------

class LogOperation:
    """
    Context manager for logging operations
    
    Usage:
        async with LogOperation(logger, "database.query", table="users"):
            await db.query()
    """
    
    def __init__(self, logger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
    
    async def __aenter__(self):
        self.start_time = time.time()
        self.logger.debug(
            f"operation.started",
            operation=self.operation,
            **self.context
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        duration_ms = round(duration * 1000, 2)
        
        if exc_type:
            self.logger.error(
                f"operation.failed",
                operation=self.operation,
                duration_ms=duration_ms,
                error=str(exc_val),
                **self.context
            )
        else:
            self.logger.debug(
                f"operation.completed",
                operation=self.operation,
                duration_ms=duration_ms,
                **self.context
            )


# ------------------------------------------------------------------------
# JSON Formatter for Structured Logging
# ------------------------------------------------------------------------

class JSONLogFormatter:
    """
    Custom JSON formatter for structured logging
    
    Produces logs in the format:
    {
        "timestamp": "2024-01-15T10:30:45.123Z",
        "level": "INFO",
        "service": "telegram-bot",
        "event": "request.completed",
        "data": { ... }
    }
    """
    
    def __init__(self, service_name: str = "telegram-bot"):
        self.service_name = service_name
    
    def __call__(self, logger, name, event_dict):
        # Add timestamp
        event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
        
        # Add service name
        event_dict["service"] = self.service_name
        
        # Move event name to top level
        if "event" in event_dict:
            event = event_dict.pop("event")
            event_dict["event"] = event
        
        # Ensure level is present
        if "level" not in event_dict:
            event_dict["level"] = name.upper()
        
        return json.dumps(event_dict, default=str)


# ------------------------------------------------------------------------
# Factory Functions
# ------------------------------------------------------------------------

def create_logging_middleware(
    service_name: str = "telegram-bot",
    sample_rate: float = 1.0,
    log_request_body: bool = True,
    log_response_body: bool = True,
    json_output: bool = True,
    metrics: Optional[MetricsCollector] = None,
) -> LoggingMiddleware:
    """
    Create logging middleware with standard configuration
    
    Args:
        service_name: Name of the service for logs
        sample_rate: Fraction of requests to log
        log_request_body: Whether to log message text
        log_response_body: Whether to log response text
        json_output: Whether to output JSON logs
        metrics: Metrics collector
        
    Returns:
        Configured LoggingMiddleware
    """
    # Configure structlog
    processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if json_output:
        processors.append(JSONLogFormatter(service_name))
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    logger = structlog.get_logger(service_name)
    
    return LoggingMiddleware(
        logger=logger,
        metrics=metrics,
        sample_rate=sample_rate,
        log_request_body=log_request_body,
        log_response_body=log_response_body,
    )


# ------------------------------------------------------------------------
# Example Usage in Dispatcher
# ------------------------------------------------------------------------

"""
# In main.py or dispatcher setup

from internal.bot.middleware.logging import create_logging_middleware

# Create logging middleware (should be first)
logging_middleware = create_logging_middleware(
    service_name="telegram-youtube-bot",
    sample_rate=1.0,  # Log all requests
    json_output=True,  # JSON format for production
)

# Add to dispatcher pipeline (FIRST!)
dispatcher.use(logging_middleware)
dispatcher.use(auth_middleware)
dispatcher.use(rate_limit_middleware)

# In handlers, use the bound logger
logger = context.metadata.get("logger")  # Already has trace_id bound
logger.info("custom.event", video_id="abc123", action="processed")
"""