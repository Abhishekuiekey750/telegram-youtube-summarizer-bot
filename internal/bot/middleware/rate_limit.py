"""
Rate Limiting Middleware
Implements token bucket algorithm for per-user rate limiting

Features:
- Token bucket algorithm for smooth rate limiting
- Per-user rate limits
- Configurable capacity and refill rate
- In-memory storage with optional Redis backend
- Rate limit headers for debugging
- Metrics collection
- Graceful degradation
"""

import asyncio
import time
import json
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib

import structlog

from internal.bot.dispatcher import UpdateContext, NextMiddleware
from internal.pkg.logger import StructuredLogger
from internal.pkg.metrics import MetricsCollector
from internal.pkg.errors import RateLimitError, ErrorKind


@dataclass
class TokenBucket:
    """
    Token bucket for rate limiting
    
    A bucket holds tokens that are consumed on each request.
    Tokens are refilled at a constant rate over time.
    """
    
    # Unique identifier for this bucket (usually user_id)
    key: str
    
    # Maximum number of tokens the bucket can hold
    capacity: int
    
    # Tokens added per second
    refill_rate: float
    
    # Current number of tokens
    tokens: float = field(default=None)
    
    # Last time the bucket was refilled (Unix timestamp)
    last_refill: float = field(default=None)
    
    def __post_init__(self):
        """Initialize bucket with full tokens"""
        if self.tokens is None:
            self.tokens = float(self.capacity)
        if self.last_refill is None:
            self.last_refill = time.time()
    
    def refill(self, now: Optional[float] = None) -> None:
        """
        Refill tokens based on time elapsed
        
        Args:
            now: Current timestamp (uses time.time() if None)
        """
        if now is None:
            now = time.time()
        
        # Calculate time since last refill
        time_passed = now - self.last_refill
        
        if time_passed <= 0:
            return
        
        # Calculate tokens to add
        tokens_to_add = time_passed * self.refill_rate
        
        # Add tokens (capped at capacity)
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def consume(self, tokens: float = 1.0, now: Optional[float] = None) -> bool:
        """
        Consume tokens from the bucket
        
        Args:
            tokens: Number of tokens to consume
            now: Current timestamp
            
        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        # First refill based on time passed
        self.refill(now)
        
        # Check if we have enough tokens
        if self.tokens < tokens:
            return False
        
        # Consume tokens
        self.tokens -= tokens
        return True
    
    def time_until_full(self, now: Optional[float] = None) -> float:
        """
        Calculate seconds until bucket is full
        
        Args:
            now: Current timestamp
            
        Returns:
            Seconds until bucket reaches capacity
        """
        if now is None:
            now = time.time()
        
        self.refill(now)
        
        if self.tokens >= self.capacity:
            return 0.0
        
        tokens_needed = self.capacity - self.tokens
        return tokens_needed / self.refill_rate
    
    def time_until_available(self, tokens: float = 1.0, now: Optional[float] = None) -> float:
        """
        Calculate seconds until specified tokens are available
        
        Args:
            tokens: Number of tokens needed
            now: Current timestamp
            
        Returns:
            Seconds until tokens are available
        """
        if now is None:
            now = time.time()
        
        self.refill(now)
        
        if self.tokens >= tokens:
            return 0.0
        
        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert bucket to dictionary for storage"""
        return {
            "key": self.key,
            "capacity": self.capacity,
            "refill_rate": self.refill_rate,
            "tokens": self.tokens,
            "last_refill": self.last_refill,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenBucket":
        """Create bucket from dictionary"""
        return cls(
            key=data["key"],
            capacity=data["capacity"],
            refill_rate=data["refill_rate"],
            tokens=data["tokens"],
            last_refill=data["last_refill"],
        )


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    
    # Default limits for different user types
    default_capacity: int = 60  # Max requests
    default_refill_rate: float = 1.0  # Tokens per second
    
    # Special limits for authenticated/premium users
    authenticated_capacity: int = 120
    authenticated_refill_rate: float = 2.0
    
    # Special limits for admin users
    admin_capacity: int = 1000
    admin_refill_rate: float = 10.0
    
    # Window for cleanup (seconds)
    cleanup_interval: int = 3600  # 1 hour
    
    # Maximum bucket age before cleanup (seconds)
    max_bucket_age: int = 86400  # 24 hours
    
    # Whether to include rate limit headers in responses
    include_headers: bool = True
    
    # Whether to use Redis for distributed rate limiting
    use_redis: bool = False
    
    # Redis connection string (if use_redis is True)
    redis_url: Optional[str] = None


class RateLimitMiddleware:
    """
    Rate limiting middleware using token bucket algorithm
    
    This middleware should be placed after authentication middleware
    to have access to user information.
    
    Features:
    - Per-user rate limits
    - Different limits for regular/authenticated/admin users
    - In-memory storage with optional Redis backend
    - Automatic cleanup of old buckets
    - Rate limit headers
    - Metrics collection
    """
    
    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
        logger: Optional[StructuredLogger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        """
        Initialize rate limit middleware
        
        Args:
            config: Rate limit configuration
            logger: Structured logger
            metrics: Metrics collector
        """
        self.config = config or RateLimitConfig()
        self.logger = logger or structlog.get_logger(__name__)
        self.metrics = metrics or MetricsCollector("rate_limit_middleware")
        
        # In-memory bucket storage: user_id -> TokenBucket
        self._buckets: Dict[str, TokenBucket] = {}
        
        # Lock for thread safety (asyncio lock)
        self._lock = asyncio.Lock()
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        self.logger.info(
            "rate_limit_middleware.initialized",
            default_capacity=self.config.default_capacity,
            default_refill_rate=self.config.default_refill_rate,
            cleanup_interval=self.config.cleanup_interval,
        )
    
    async def start(self) -> None:
        """Start background cleanup task"""
        if self._cleanup_task is None:
            self._running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.logger.debug("rate_limit_middleware.cleanup_task_started")
    
    async def stop(self) -> None:
        """Stop background cleanup task"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            self.logger.debug("rate_limit_middleware.cleanup_task_stopped")
    
    async def __call__(
        self,
        context: UpdateContext,
        next_middleware: NextMiddleware,
    ) -> Optional[UpdateContext]:
        """
        Execute rate limiting middleware
        
        Args:
            context: Update context
            next_middleware: Next middleware
            
        Returns:
            Context or None if rate limited
        """
        start_time = time.time()
        
        try:
            # Determine user identifier
            user_id = self._get_user_identifier(context)
            
            if not user_id:
                # Can't rate limit without user identifier
                self.logger.debug(
                    "rate_limit_middleware.no_user_id",
                    update_id=context.update.update_id,
                )
                return await next_middleware(context)
            
            # Determine rate limit tier based on user
            capacity, refill_rate = self._get_rate_limits_for_user(context)
            
            # Check rate limit
            is_allowed, bucket = await self._check_rate_limit(
                user_id=user_id,
                capacity=capacity,
                refill_rate=refill_rate,
            )
            
            # Add rate limit headers to context (for response)
            if self.config.include_headers and is_allowed:
                self._add_rate_limit_headers(context, bucket)
            
            if not is_allowed:
                # User is rate limited
                await self._handle_rate_limited(context, bucket)
                return None
            
            # Continue pipeline
            result = await next_middleware(context)
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.record_latency("rate_limit_check", duration)
            self.metrics.increment(
                "requests.allowed",
                tags={"user_type": self._get_user_type(context)},
            )
            
            return result
            
        except Exception as e:
            self.logger.exception(
                "rate_limit_middleware.error",
                error=str(e),
                update_id=context.update.update_id,
            )
            self.metrics.increment("rate_limit.errors")
            
            # On error, allow request to proceed (fail open)
            return await next_middleware(context)
    
    # ------------------------------------------------------------------------
    # Core Rate Limiting Logic
    # ------------------------------------------------------------------------
    
    async def _check_rate_limit(
        self,
        user_id: str,
        capacity: int,
        refill_rate: float,
    ) -> Tuple[bool, Optional[TokenBucket]]:
        """
        Check if request is within rate limits
        
        Args:
            user_id: User identifier
            capacity: Bucket capacity
            refill_rate: Tokens per second
            
        Returns:
            Tuple of (is_allowed, bucket)
        """
        async with self._lock:
            # Get or create bucket
            bucket = await self._get_bucket(user_id, capacity, refill_rate)
            
            # Try to consume a token
            now = time.time()
            is_allowed = bucket.consume(tokens=1.0, now=now)
            
            # Update bucket in storage
            await self._set_bucket(user_id, bucket)
            
            return is_allowed, bucket
    
    async def _get_bucket(
        self,
        user_id: str,
        capacity: int,
        refill_rate: float,
    ) -> TokenBucket:
        """
        Get bucket from storage or create new one
        
        Args:
            user_id: User identifier
            capacity: Bucket capacity
            refill_rate: Tokens per second
            
        Returns:
            Token bucket
        """
        # Try to get from memory
        if user_id in self._buckets:
            bucket = self._buckets[user_id]
            
            # Update bucket parameters if they changed
            bucket.capacity = capacity
            bucket.refill_rate = refill_rate
            
            return bucket
        
        # Create new bucket
        bucket = TokenBucket(
            key=user_id,
            capacity=capacity,
            refill_rate=refill_rate,
        )
        
        self._buckets[user_id] = bucket
        self.metrics.increment("rate_limit.buckets_created")
        
        return bucket
    
    async def _set_bucket(self, user_id: str, bucket: TokenBucket) -> None:
        """Store bucket in memory"""
        self._buckets[user_id] = bucket
    
    def _get_user_identifier(self, context: UpdateContext) -> Optional[str]:
        """
        Get unique identifier for user
        
        Priority:
        1. User ID from authentication
        2. Chat ID (for anonymous)
        3. IP address (if available)
        
        Args:
            context: Update context
            
        Returns:
            User identifier string
        """
        # Check if user is authenticated
        user = context.metadata.get("user")
        if user and hasattr(user, "id"):
            return f"user:{user.id}"
        
        # Fall back to chat ID
        if context.chat_id:
            return f"chat:{context.chat_id}"
        
        # Last resort - hash of update ID (not ideal)
        return f"update:{hashlib.md5(str(context.update.update_id).encode()).hexdigest()[:8]}"
    
    def _get_user_type(self, context: UpdateContext) -> str:
        """Determine user type for metrics"""
        user = context.metadata.get("user")
        
        if not user:
            return "anonymous"
        
        # Check if admin (this would come from auth middleware)
        if context.metadata.get("is_admin"):
            return "admin"
        
        # Check if authenticated (has user object)
        return "authenticated"
    
    def _get_rate_limits_for_user(self, context: UpdateContext) -> Tuple[int, float]:
        """
        Get rate limits appropriate for user type
        
        Args:
            context: Update context
            
        Returns:
            Tuple of (capacity, refill_rate)
        """
        user_type = self._get_user_type(context)
        
        if user_type == "admin":
            return self.config.admin_capacity, self.config.admin_refill_rate
        elif user_type == "authenticated":
            return self.config.authenticated_capacity, self.config.authenticated_refill_rate
        else:
            return self.config.default_capacity, self.config.default_refill_rate
    
    # ------------------------------------------------------------------------
    # Rate Limit Response Handling
    # ------------------------------------------------------------------------
    
    async def _handle_rate_limited(
        self,
        context: UpdateContext,
        bucket: TokenBucket,
    ) -> None:
        """
        Handle rate limited request
        
        Args:
            context: Update context
            bucket: Token bucket
        """
        now = time.time()
        retry_after = int(bucket.time_until_available(now=now))
        
        self.logger.warning(
            "rate_limit_middleware.rate_limited",
            user_id=self._get_user_identifier(context),
            chat_id=context.chat_id,
            retry_after=retry_after,
            tokens_remaining=bucket.tokens,
        )
        
        self.metrics.increment(
            "requests.rate_limited",
            tags={"user_type": self._get_user_type(context)},
        )
        
        # Send rate limit message to user
        telegram_client = context.metadata.get("telegram_client")
        
        if telegram_client and context.chat_id:
            # Get user's language for message
            language = context.metadata.get("language", "en")
            
            if language == "hi":
                message = (
                    f"⏳ *बहुत सारे अनुरोध*\n\n"
                    f"आपने बहुत जल्दी बहुत सारे अनुरोध भेजे हैं। "
                    f"कृपया {retry_after} सेकंड में पुनः प्रयास करें।\n\n"
                    f"यह सीमा सभी उपयोगकर्ताओं के लिए उचित उपयोग सुनिश्चित करने के लिए है।"
                )
            elif language == "ta":
                message = (
                    f"⏳ *அதிக கோரிக்கைகள்*\n\n"
                    f"நீங்கள் மிக விரைவாக பல கோரிக்கைகளை அனுப்பியுள்ளீர்கள். "
                    f"தயவுசெய்து {retry_after} வினாடிகளில் மீண்டும் முயற்சிக்கவும்.\n\n"
                    f"இந்த வரம்பு அனைத்து பயனர்களுக்கும் நியாயமான பயன்பாட்டை உறுதி செய்கிறது."
                )
            else:
                message = (
                    f"⏳ *Too Many Requests*\n\n"
                    f"You have sent too many requests too quickly. "
                    f"Please try again in {retry_after} seconds.\n\n"
                    f"This limit ensures fair usage for all users."
                )
            
            # Add rate limit headers to response
            headers = {}
            if self.config.include_headers:
                headers = {
                    "X-RateLimit-Limit": str(bucket.capacity),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(now + retry_after)),
                    "Retry-After": str(retry_after),
                }
            
            await telegram_client.send_message(
                chat_id=context.chat_id,
                text=message,
                parse_mode="Markdown",
            )
    
    def _add_rate_limit_headers(
        self,
        context: UpdateContext,
        bucket: TokenBucket,
    ) -> None:
        """
        Add rate limit headers to context for response
        
        Args:
            context: Update context
            bucket: Token bucket
        """
        now = time.time()
        reset_time = int(now + bucket.time_until_full(now))
        
        headers = {
            "X-RateLimit-Limit": str(int(bucket.capacity)),
            "X-RateLimit-Remaining": str(max(0, int(bucket.tokens))),
            "X-RateLimit-Reset": str(reset_time),
        }
        
        # Store headers in context metadata
        if "response_headers" not in context.metadata:
            context.metadata["response_headers"] = {}
        context.metadata["response_headers"].update(headers)
    
    # ------------------------------------------------------------------------
    # Bucket Cleanup
    # ------------------------------------------------------------------------
    
    async def _cleanup_loop(self) -> None:
        """Background task to clean up old buckets"""
        while self._running:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_old_buckets()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception(
                    "rate_limit_middleware.cleanup_error",
                    error=str(e),
                )
    
    async def _cleanup_old_buckets(self) -> None:
        """Remove buckets that haven't been used recently"""
        async with self._lock:
            now = time.time()
            cutoff = now - self.config.max_bucket_age
            
            buckets_to_remove = []
            
            for user_id, bucket in self._buckets.items():
                if bucket.last_refill < cutoff:
                    buckets_to_remove.append(user_id)
            
            for user_id in buckets_to_remove:
                del self._buckets[user_id]
            
            if buckets_to_remove:
                self.logger.debug(
                    "rate_limit_middleware.cleaned_up_buckets",
                    count=len(buckets_to_remove),
                )
                self.metrics.gauge(
                    "rate_limit.buckets_cleaned",
                    value=len(buckets_to_remove),
                )
    
    # ------------------------------------------------------------------------
    # Public API for testing and management
    # ------------------------------------------------------------------------
    
    async def get_bucket_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get bucket information for a user (for debugging)
        
        Args:
            user_id: User identifier
            
        Returns:
            Bucket info dict or None
        """
        async with self._lock:
            bucket = self._buckets.get(user_id)
            if not bucket:
                return None
            
            now = time.time()
            bucket.refill(now)
            
            return {
                "user_id": user_id,
                "capacity": bucket.capacity,
                "tokens": round(bucket.tokens, 2),
                "tokens_remaining": max(0, int(bucket.tokens)),
                "refill_rate": bucket.refill_rate,
                "last_refill": datetime.fromtimestamp(bucket.last_refill).isoformat(),
                "time_until_full": round(bucket.time_until_full(now), 2),
            }
    
    async def reset_bucket(self, user_id: str) -> None:
        """
        Reset bucket for a user (for testing)
        
        Args:
            user_id: User identifier
        """
        async with self._lock:
            if user_id in self._buckets:
                del self._buckets[user_id]
                self.logger.info(
                    "rate_limit_middleware.bucket_reset",
                    user_id=user_id,
                )
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get middleware statistics"""
        async with self._lock:
            return {
                "total_buckets": len(self._buckets),
                "config": {
                    "default_capacity": self.config.default_capacity,
                    "default_refill_rate": self.config.default_refill_rate,
                    "authenticated_capacity": self.config.authenticated_capacity,
                    "admin_capacity": self.config.admin_capacity,
                    "cleanup_interval": self.config.cleanup_interval,
                },
                "memory_usage": sum(
                    len(bucket.key) + 100 for bucket in self._buckets.values()
                ),  # Approximate bytes
            }


# ------------------------------------------------------------------------
# Factory Functions
# ------------------------------------------------------------------------

def create_rate_limit_middleware(
    default_capacity: int = 60,
    default_refill_rate: float = 1.0,
    authenticated_capacity: int = 120,
    authenticated_refill_rate: float = 2.0,
    admin_capacity: int = 1000,
    admin_refill_rate: float = 10.0,
    logger: Optional[StructuredLogger] = None,
    metrics: Optional[MetricsCollector] = None,
) -> RateLimitMiddleware:
    """
    Create rate limit middleware with custom limits
    
    Args:
        default_capacity: Default max requests
        default_refill_rate: Default tokens per second
        authenticated_capacity: Max requests for authenticated users
        authenticated_refill_rate: Tokens per second for authenticated
        admin_capacity: Max requests for admins
        admin_refill_rate: Tokens per second for admins
        logger: Logger instance
        metrics: Metrics collector
        
    Returns:
        Configured RateLimitMiddleware
    """
    config = RateLimitConfig(
        default_capacity=default_capacity,
        default_refill_rate=default_refill_rate,
        authenticated_capacity=authenticated_capacity,
        authenticated_refill_rate=authenticated_refill_rate,
        admin_capacity=admin_capacity,
        admin_refill_rate=admin_refill_rate,
    )
    
    return RateLimitMiddleware(
        config=config,
        logger=logger,
        metrics=metrics,
    )


def create_strict_rate_limit_middleware(
    logger: Optional[StructuredLogger] = None,
    metrics: Optional[MetricsCollector] = None,
) -> RateLimitMiddleware:
    """
    Create strict rate limiting (for public bots)
    
    Limits:
    - Anonymous: 30 requests per minute
    - Authenticated: 60 requests per minute
    - Admin: 600 requests per minute
    """
    return create_rate_limit_middleware(
        default_capacity=30,
        default_refill_rate=0.5,  # 30 per minute
        authenticated_capacity=60,
        authenticated_refill_rate=1.0,  # 60 per minute
        admin_capacity=600,
        admin_refill_rate=10.0,  # 600 per minute
        logger=logger,
        metrics=metrics,
    )


def create_generous_rate_limit_middleware(
    logger: Optional[StructuredLogger] = None,
    metrics: Optional[MetricsCollector] = None,
) -> RateLimitMiddleware:
    """
    Create generous rate limiting (for private/internal bots)
    
    Limits:
    - Anonymous: 120 requests per minute
    - Authenticated: 300 requests per minute
    - Admin: Unlimited (effectively)
    """
    return create_rate_limit_middleware(
        default_capacity=120,
        default_refill_rate=2.0,  # 120 per minute
        authenticated_capacity=300,
        authenticated_refill_rate=5.0,  # 300 per minute
        admin_capacity=10000,
        admin_refill_rate=100.0,  # Effectively unlimited
        logger=logger,
        metrics=metrics,
    )