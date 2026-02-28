"""
Panic/Error Recovery Middleware
Ensures bot never crashes by catching all panics and handling gracefully

Features:
- Catches all unhandled exceptions
- Prevents bot crashes
- Classifies errors by type
- Implements retry logic
- Sends user-friendly error messages
- Logs full stack traces
- Notifies admins of critical errors
- Circuit breaker pattern for failing services
"""

import asyncio
import sys
import traceback
from typing import Optional, Callable, Awaitable, Dict, Any, Type, Union
from datetime import datetime, timedelta
from enum import Enum
import functools

import structlog

from internal.bot.dispatcher import UpdateContext, NextMiddleware
from internal.pkg.logger import StructuredLogger
from internal.pkg.metrics import MetricsCollector
from internal.pkg.errors import (
    BotError,
    ErrorKind,
    RetryableError,
    ValidationError,
    NotFoundError,
    UnauthorizedError,
    RateLimitError,
)


class ErrorSeverity(Enum):
    """Severity levels for errors"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types"""
    RETRY = "retry"                 # Retry the operation
    CACHE = "cache"                 # Return cached response
    DEGRADE = "degrade"             # Return partial result
    NOTIFY = "notify"               # Notify admin and continue
    FAIL_GRACEFULLY = "fail_gracefully"  # Show error message
    FAIL_SILENTLY = "fail_silently"      # Log only, no response


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject requests
    HALF_OPEN = "half_open" # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern for failing services
    Prevents cascading failures by temporarily disabling failing services
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_limit: int = 3,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_limit = half_open_limit
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.half_open_successes = 0
        self.last_failure_time = None
        self.last_state_change = datetime.now()
    
    def record_success(self) -> None:
        """Record a successful operation"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_successes += 1
            if self.half_open_successes >= self.half_open_limit:
                self._close()
        elif self.state == CircuitBreakerState.OPEN:
            # Should not happen, but just in case
            pass
        else:
            # CLOSED state - reset failure count on success
            self.failure_count = 0
    
    def record_failure(self) -> None:
        """Record a failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            # Any failure in half-open state opens the circuit again
            self._open()
        elif self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self._open()
    
    def _open(self) -> None:
        """Open the circuit"""
        self.state = CircuitBreakerState.OPEN
        self.last_state_change = datetime.now()
        self.half_open_successes = 0
    
    def _close(self) -> None:
        """Close the circuit"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.half_open_successes = 0
        self.last_state_change = datetime.now()
    
    def _half_open(self) -> None:
        """Move to half-open state"""
        self.state = CircuitBreakerState.HALF_OPEN
        self.half_open_successes = 0
        self.last_state_change = datetime.now()
    
    def can_execute(self) -> bool:
        """Check if operation can be executed"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has elapsed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self._half_open()
                    return True
            return False
        
        # HALF_OPEN state - allow limited requests
        return self.half_open_successes < self.half_open_limit
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state for monitoring"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "half_open_successes": self.half_open_successes,
            "last_state_change": self.last_state_change.isoformat(),
        }


class RecoveryMiddleware:
    """
    Panic/Error recovery middleware that ensures bot never crashes.
    
    This middleware MUST be the first in the pipeline to catch all panics.
    
    Features:
    - Catches all unhandled exceptions
    - Classifies errors by type and severity
    - Implements retry logic with exponential backoff
    - Circuit breaker for failing services
    - User-friendly error messages in multiple languages
    - Admin notifications for critical errors
    - Metrics tracking for error rates
    """
    
    def __init__(
        self,
        logger: Optional[StructuredLogger] = None,
        metrics: Optional[MetricsCollector] = None,
        admin_chat_ids: Optional[list[int]] = None,
        telegram_client = None,  # For sending admin alerts
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_circuit_breakers: bool = True,
    ):
        """
        Initialize recovery middleware
        
        Args:
            logger: Structured logger
            metrics: Metrics collector
            admin_chat_ids: Chat IDs for admin notifications
            telegram_client: Client for sending admin alerts
            max_retries: Maximum retry attempts
            retry_delay: Base delay for exponential backoff
            enable_circuit_breakers: Enable circuit breaker pattern
        """
        self.logger = logger or structlog.get_logger(__name__)
        self.metrics = metrics or MetricsCollector("recovery_middleware")
        self.admin_chat_ids = admin_chat_ids or []
        self.telegram_client = telegram_client
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_circuit_breakers = enable_circuit_breakers
        
        # Circuit breakers for different services
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Error classification map
        self._init_error_classification()
        
        self.logger.info(
            "recovery_middleware.initialized",
            admin_chats=len(self.admin_chat_ids),
            max_retries=self.max_retries,
            circuit_breakers=self.enable_circuit_breakers,
        )
    
    def _init_error_classification(self) -> None:
        """Initialize error classification rules"""
        self.error_classification = {
            # Retryable errors
            asyncio.TimeoutError: {
                "severity": ErrorSeverity.WARNING,
                "strategy": RecoveryStrategy.RETRY,
                "user_message": "The operation timed out. Retrying...",
                "retryable": True,
            },
            ConnectionError: {
                "severity": ErrorSeverity.WARNING,
                "strategy": RecoveryStrategy.RETRY,
                "user_message": "Network connection issue. Retrying...",
                "retryable": True,
            },
            
            # Validation errors
            ValidationError: {
                "severity": ErrorSeverity.INFO,
                "strategy": RecoveryStrategy.FAIL_GRACEFULLY,
                "user_message": None,  # Use error's message
                "retryable": False,
            },
            
            # Not found errors
            NotFoundError: {
                "severity": ErrorSeverity.INFO,
                "strategy": RecoveryStrategy.FAIL_GRACEFULLY,
                "user_message": None,
                "retryable": False,
            },
            
            # Auth errors
            UnauthorizedError: {
                "severity": ErrorSeverity.WARNING,
                "strategy": RecoveryStrategy.FAIL_GRACEFULLY,
                "user_message": "You're not authorized to do that.",
                "retryable": False,
            },
            
            # Rate limit errors
            RateLimitError: {
                "severity": ErrorSeverity.WARNING,
                "strategy": RecoveryStrategy.FAIL_GRACEFULLY,
                "user_message": None,  # Rate limit message is specific
                "retryable": True,  # Can retry after delay
            },
            
            # Business logic errors
            BotError: {
                "severity": ErrorSeverity.ERROR,
                "strategy": RecoveryStrategy.FAIL_GRACEFULLY,
                "user_message": None,
                "retryable": False,
            },
        }
        
        # Default for unknown errors
        self.default_classification = {
            "severity": ErrorSeverity.ERROR,
            "strategy": RecoveryStrategy.FAIL_GRACEFULLY,
            "user_message": "An unexpected error occurred.",
            "retryable": False,
        }
    
    async def __call__(
        self,
        context: UpdateContext,
        next_middleware: NextMiddleware,
    ) -> Optional[UpdateContext]:
        """
        Execute recovery middleware
        
        This is a try/except block around the entire pipeline.
        Any exception that reaches here is caught and handled gracefully.
        
        Args:
            context: Update context
            next_middleware: Next middleware
            
        Returns:
            Updated context or None
        """
        # Get service name for circuit breaker
        service_name = self._get_service_name(context)
        
        # Check circuit breaker
        if self.enable_circuit_breakers:
            circuit_breaker = self._get_circuit_breaker(service_name)
            if not circuit_breaker.can_execute():
                await self._handle_circuit_open(context, service_name)
                return None
        
        # Execute with retry logic
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                # Execute next middleware
                result = await next_middleware(context)
                
                # Record success in circuit breaker
                if self.enable_circuit_breakers:
                    circuit_breaker.record_success()
                
                return result
                
            except Exception as e:
                last_error = e
                
                # Classify error
                classification = self._classify_error(e)
                
                # Log error
                await self._log_error(context, e, classification, retry_count)
                
                # Track metrics
                self.metrics.increment(
                    "errors.total",
                    tags={
                        "error_type": e.__class__.__name__,
                        "severity": classification["severity"].value,
                        "retry_count": retry_count,
                    },
                )
                
                # Check if we should retry
                if classification["retryable"] and retry_count < self.max_retries:
                    retry_count += 1
                    
                    # Calculate backoff delay
                    delay = self.retry_delay * (2 ** (retry_count - 1))
                    
                    self.logger.info(
                        "recovery_middleware.retrying",
                        retry_count=retry_count,
                        max_retries=self.max_retries,
                        delay_seconds=delay,
                        error=str(e),
                    )
                    
                    # Send retry message to user
                    if retry_count == 1 and context.chat_id:
                        await self._send_retry_message(context, delay)
                    
                    # Wait before retry
                    await asyncio.sleep(delay)
                    continue
                
                # Record failure in circuit breaker
                if self.enable_circuit_breakers:
                    circuit_breaker.record_failure()
                
                # Handle the error (no more retries)
                await self._handle_error(context, e, classification)
                
                # Check if admin notification needed
                if classification["severity"] in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]:
                    await self._notify_admin(context, e, classification)
                
                return None
        
        # Should never reach here
        return None
    
    # ------------------------------------------------------------------------
    # Error Handling
    # ------------------------------------------------------------------------
    
    def _classify_error(self, error: Exception) -> Dict[str, Any]:
        """
        Classify error to determine severity and recovery strategy
        
        Args:
            error: The exception that occurred
            
        Returns:
            Classification dictionary
        """
        # Check for exact match
        for error_type, classification in self.error_classification.items():
            if isinstance(error, error_type):
                return classification.copy()
        
        # Check for BotError subclasses
        if isinstance(error, BotError):
            # Use error's own severity if available
            return {
                "severity": getattr(error, "severity", ErrorSeverity.ERROR),
                "strategy": RecoveryStrategy.FAIL_GRACEFULLY,
                "user_message": str(error),
                "retryable": getattr(error, "retryable", False),
            }
        
        # Return default
        return self.default_classification.copy()
    
    async def _handle_error(
        self,
        context: UpdateContext,
        error: Exception,
        classification: Dict[str, Any],
    ) -> None:
        """
        Handle error based on classification
        
        Args:
            context: Update context
            error: The exception
            classification: Error classification
        """
        strategy = classification["strategy"]
        
        if strategy == RecoveryStrategy.FAIL_GRACEFULLY:
            await self._fail_gracefully(context, error, classification)
        elif strategy == RecoveryStrategy.FAIL_SILENTLY:
            await self._fail_silently(context, error)
        elif strategy == RecoveryStrategy.NOTIFY:
            await self._notify_only(context, error)
        elif strategy == RecoveryStrategy.DEGRADE:
            await self._degrade_gracefully(context, error)
        else:
            # Default to graceful failure
            await self._fail_gracefully(context, error, classification)
    
    async def _fail_gracefully(
        self,
        context: UpdateContext,
        error: Exception,
        classification: Dict[str, Any],
    ) -> None:
        """
        Fail gracefully by sending user-friendly error message
        
        Args:
            context: Update context
            error: The exception
            classification: Error classification
        """
        # Get user's language
        language = context.metadata.get("language", "en")
        
        # Determine error message
        if classification["user_message"]:
            message = classification["user_message"]
        else:
            message = str(error)
        
        # If no message, use generic one
        if not message:
            message = self._get_generic_error_message(language)
        
        # Add friendly note
        if language == "hi":
            message += "\n\nकृपया पुनः प्रयास करें या बाद में आएं।"
        elif language == "ta":
            message += "\n\nதயவுசெய்து மீண்டும் முயற்சிக்கவும் அல்லது பின்னர் வரவும்."
        else:
            message += "\n\nPlease try again or come back later."
        
        # Send message if we have a client
        telegram_client = context.metadata.get("telegram_client") or self.telegram_client
        if telegram_client and context.chat_id:
            await telegram_client.send_message(
                chat_id=context.chat_id,
                text=message,
                parse_mode="Markdown",
            )
    
    async def _fail_silently(self, context: UpdateContext, error: Exception) -> None:
        """
        Fail silently - just log, don't notify user
        Used for non-critical background errors
        """
        self.logger.debug(
            "recovery_middleware.silent_failure",
            error=str(error),
            update_id=context.update.update_id,
        )
    
    async def _notify_only(self, context: UpdateContext, error: Exception) -> None:
        """
        Only notify admin, don't send user message
        """
        # Just log - notification handled elsewhere
        self.logger.info(
            "recovery_middleware.notify_only",
            error=str(error),
        )
    
    async def _degrade_gracefully(self, context: UpdateContext, error: Exception) -> None:
        """
        Degrade gracefully - return partial/cached result
        """
        # For now, just fail gracefully
        # In a more advanced implementation, this would try to return cached data
        await self._fail_gracefully(
            context,
            error,
            self.default_classification,
        )
    
    async def _handle_circuit_open(self, context: UpdateContext, service: str) -> None:
        """
        Handle open circuit breaker
        
        Args:
            context: Update context
            service: Service name
        """
        self.logger.warning(
            "recovery_middleware.circuit_open",
            service=service,
            chat_id=context.chat_id,
        )
        
        self.metrics.increment("circuit_breaker.open", tags={"service": service})
        
        # Send message to user
        telegram_client = context.metadata.get("telegram_client") or self.telegram_client
        if telegram_client and context.chat_id:
            language = context.metadata.get("language", "en")
            
            if language == "hi":
                message = (
                    "🔧 *सेवा अस्थायी रूप से अनुपलब्ध*\n\n"
                    "यह सुविधा अभी काम नहीं कर रही है। कृपया कुछ मिनटों में पुनः प्रयास करें।"
                )
            elif language == "ta":
                message = (
                    "🔧 *சேவை தற்காலிகமாக கிடைக்கவில்லை*\n\n"
                    "இந்த அம்சம் இப்போது வேலை செய்யவில்லை. சில நிமிடங்களில் மீண்டும் முயற்சிக்கவும்."
                )
            else:
                message = (
                    "🔧 *Service Temporarily Unavailable*\n\n"
                    "This feature is currently experiencing issues. Please try again in a few minutes."
                )
            
            await telegram_client.send_message(
                chat_id=context.chat_id,
                text=message,
                parse_mode="Markdown",
            )
    
    async def _send_retry_message(self, context: UpdateContext, delay: float) -> None:
        """
        Send retry notification to user
        
        Args:
            context: Update context
            delay: Retry delay in seconds
        """
        telegram_client = context.metadata.get("telegram_client") or self.telegram_client
        if not telegram_client or not context.chat_id:
            return
        
        language = context.metadata.get("language", "en")
        
        if language == "hi":
            message = f"⏳ *क्षणिक समस्या*\n\nपुनः प्रयास किया जा रहा है... {int(delay)} सेकंड प्रतीक्षा करें।"
        elif language == "ta":
            message = f"⏳ *தற்காலிக சிக்கல்*\n\nமீண்டும் முயற்சிக்கிறது... {int(delay)} விநாடிகள் காத்திருக்கவும்."
        else:
            message = f"⏳ *Temporary Issue*\n\nRetrying... please wait {int(delay)} seconds."
        
        await telegram_client.send_message(
            chat_id=context.chat_id,
            text=message,
            parse_mode="Markdown",
        )
    
    # ------------------------------------------------------------------------
    # Logging and Monitoring
    # ------------------------------------------------------------------------
    
    async def _log_error(
        self,
        context: UpdateContext,
        error: Exception,
        classification: Dict[str, Any],
        retry_count: int,
    ) -> None:
        """
        Log error with full context
        
        Args:
            context: Update context
            error: The exception
            classification: Error classification
            retry_count: Current retry attempt
        """
        log_data = {
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            "severity": classification["severity"].value,
            "strategy": classification["strategy"].value,
            "retry_count": retry_count,
            "max_retries": self.max_retries,
            "chat_id": context.chat_id,
            "user_id": context.user_id,
            "update_id": context.update.update_id,
            "traceback": traceback.format_exc(),
        }
        
        # Add trace ID if available
        if "trace_id" in context.metadata:
            log_data["trace_id"] = context.metadata["trace_id"]
        
        # Log at appropriate level
        severity = classification["severity"]
        if severity == ErrorSeverity.DEBUG:
            self.logger.debug("recovery_middleware.error", **log_data)
        elif severity == ErrorSeverity.INFO:
            self.logger.info("recovery_middleware.error", **log_data)
        elif severity == ErrorSeverity.WARNING:
            self.logger.warning("recovery_middleware.error", **log_data)
        elif severity == ErrorSeverity.ERROR:
            self.logger.error("recovery_middleware.error", **log_data)
        elif severity == ErrorSeverity.CRITICAL:
            self.logger.critical("recovery_middleware.error", **log_data)
    
    async def _notify_admin(
        self,
        context: UpdateContext,
        error: Exception,
        classification: Dict[str, Any],
    ) -> None:
        """
        Send critical error notification to admins
        
        Args:
            context: Update context
            error: The exception
            classification: Error classification
        """
        if not self.admin_chat_ids or not self.telegram_client:
            return
        
        # Format error message for admin
        message = (
            f"🚨 *Critical Error*\n\n"
            f"*Error:* {error.__class__.__name__}\n"
            f"*Message:* {str(error)}\n"
            f"*Severity:* {classification['severity'].value}\n"
            f"*User:* {context.user_id}\n"
            f"*Chat:* {context.chat_id}\n"
            f"*Update:* {context.update.update_id}\n"
        )
        
        if "trace_id" in context.metadata:
            message += f"*Trace:* `{context.metadata['trace_id']}`\n"
        
        # Send to all admin chats
        for admin_chat_id in self.admin_chat_ids:
            try:
                await self.telegram_client.send_message(
                    chat_id=admin_chat_id,
                    text=message,
                    parse_mode="Markdown",
                )
            except Exception as e:
                self.logger.error(
                    "recovery_middleware.admin_notification_failed",
                    admin_chat_id=admin_chat_id,
                    error=str(e),
                )
    
    # ------------------------------------------------------------------------
    # Circuit Breaker Management
    # ------------------------------------------------------------------------
    
    def _get_circuit_breaker(self, service: str) -> CircuitBreaker:
        """
        Get or create circuit breaker for service
        
        Args:
            service: Service name
            
        Returns:
            Circuit breaker instance
        """
        if service not in self.circuit_breakers:
            self.circuit_breakers[service] = CircuitBreaker(
                name=service,
                failure_threshold=5,
                recovery_timeout=60.0,
                half_open_limit=3,
            )
        
        return self.circuit_breakers[service]
    
    def _get_service_name(self, context: UpdateContext) -> str:
        """
        Determine which service is being called
        
        Args:
            context: Update context
            
        Returns:
            Service name for circuit breaker
        """
        # Try to determine from update type
        if context.update.callback_query:
            return "callback_handler"
        
        if context.message_text:
            if "youtube.com" in context.message_text or "youtu.be" in context.message_text:
                return "youtube_service"
            elif context.message_text.startswith("/"):
                return "command_handler"
            else:
                return "question_handler"
        
        return "unknown"
    
    # ------------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------------
    
    def _get_generic_error_message(self, language: str) -> str:
        """
        Get generic error message in user's language
        
        Args:
            language: User's language code
            
        Returns:
            Error message
        """
        messages = {
            "hi": "❌ *एक त्रुटि हुई*",
            "ta": "❌ *ஒரு பிழை ஏற்பட்டது*",
            "te": "❌ *ఒక లోపం సంభవించింది*",
            "kn": "❌ *ದೋಷ ಸಂಭವಿಸಿದೆ*",
            "ml": "❌ *ഒരു പിശക് സംഭവിച്ചു*",
            "bn": "❌ *একটি ত্রুটি ঘটেছে*",
        }
        
        return messages.get(language, "❌ *An error occurred*")
    
    def get_circuit_breaker_states(self) -> Dict[str, Any]:
        """
        Get states of all circuit breakers for monitoring
        
        Returns:
            Dictionary of circuit breaker states
        """
        return {
            name: cb.get_state()
            for name, cb in self.circuit_breakers.items()
        }
    
    async def reset_circuit_breaker(self, service: str) -> bool:
        """
        Reset circuit breaker for a service
        
        Args:
            service: Service name
            
        Returns:
            True if reset, False if not found
        """
        if service in self.circuit_breakers:
            # Remove and recreate
            del self.circuit_breakers[service]
            self._get_circuit_breaker(service)  # Recreate
            return True
        return False


# ------------------------------------------------------------------------
# Factory Functions
# ------------------------------------------------------------------------

def create_recovery_middleware(
    logger: Optional[StructuredLogger] = None,
    metrics: Optional[MetricsCollector] = None,
    admin_chat_ids: Optional[list[int]] = None,
    telegram_client=None,
    max_retries: int = 3,
    enable_circuit_breakers: bool = True,
) -> RecoveryMiddleware:
    """
    Create recovery middleware with standard configuration
    
    Args:
        logger: Logger instance
        metrics: Metrics collector
        admin_chat_ids: Admin chat IDs for alerts
        telegram_client: Client for admin alerts
        max_retries: Maximum retry attempts
        enable_circuit_breakers: Enable circuit breaker pattern
        
    Returns:
        Configured RecoveryMiddleware
    """
    return RecoveryMiddleware(
        logger=logger,
        metrics=metrics,
        admin_chat_ids=admin_chat_ids,
        telegram_client=telegram_client,
        max_retries=max_retries,
        enable_circuit_breakers=enable_circuit_breakers,
    )


# ------------------------------------------------------------------------
# Decorator for Function-Level Recovery
# ------------------------------------------------------------------------

def with_recovery(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    fallback_return=None,
    log_errors: bool = True,
):
    """
    Decorator to add recovery to individual functions
    
    Usage:
        @with_recovery(max_retries=2, fallback_return=None)
        async def risky_function():
            # This will be retried on failure
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            retry_count = 0
            
            while retry_count <= max_retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    retry_count += 1
                    
                    if log_errors:
                        logger = structlog.get_logger(__name__)
                        logger.warning(
                            "function.retry",
                            func=func.__name__,
                            retry_count=retry_count,
                            max_retries=max_retries,
                            error=str(e),
                        )
                    
                    if retry_count <= max_retries:
                        # Exponential backoff
                        delay = retry_delay * (2 ** (retry_count - 1))
                        await asyncio.sleep(delay)
            
            # All retries failed
            if log_errors:
                logger = structlog.get_logger(__name__)
                logger.error(
                    "function.failed",
                    func=func.__name__,
                    error=str(last_error),
                )
            
            if fallback_return is not None:
                return fallback_return
            
            raise last_error
        
        return wrapper
    return decorator


# ------------------------------------------------------------------------
# Example Usage in Dispatcher
# ------------------------------------------------------------------------

"""
# In main.py - MUST BE FIRST MIDDLEWARE

from internal.bot.middleware.recovery import create_recovery_middleware

# Create recovery middleware
recovery_middleware = create_recovery_middleware(
    admin_chat_ids=[12345],  # Admin IDs for alerts
    telegram_client=bot_client,
    max_retries=3,
    enable_circuit_breakers=True,
)

# Add to dispatcher - THIS MUST BE FIRST!
dispatcher.use(recovery_middleware)  # First!
dispatcher.use(logging_middleware)   # Second
dispatcher.use(auth_middleware)       # Third
dispatcher.use(rate_limit_middleware) # Fourth

# Now any panic in any middleware or handler will be caught
# and the bot will never crash
"""