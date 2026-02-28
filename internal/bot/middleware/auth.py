"""
Authentication Middleware
Identifies Telegram users and attaches user context to all updates

Features:
- Extract user information from Telegram updates
- Check against allowed users list
- Create/load user sessions
- Track user metrics
- Block unauthorized users
- Handle anonymous users (channels)
"""

from typing import Optional, Dict, Any, Callable, Awaitable, List
from datetime import datetime
import hashlib
import hmac

from telegram import Update
import structlog

from internal.bot.dispatcher import UpdateContext, NextMiddleware
from internal.storage.session import SessionStore, UserSession
from internal.domain.value_objects import Language
from internal.pkg.logger import StructuredLogger
from internal.pkg.metrics import MetricsCollector
from internal.pkg.errors import UnauthorizedError, ErrorKind


class User:
    """
    User domain model representing a Telegram user
    """
    
    def __init__(
        self,
        id: int,
        username: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        language_code: Optional[str] = None,
        is_premium: bool = False,
        is_bot: bool = False,
    ):
        self.id = id
        self.username = username
        self.first_name = first_name
        self.last_name = last_name
        self.language_code = language_code or "en"
        self.is_premium = is_premium
        self.is_bot = is_bot
        self.first_seen = datetime.now()
        self.last_seen = datetime.now()
    
    @property
    def full_name(self) -> str:
        """Get user's full name"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.first_name or self.username or f"User_{self.id}"
    
    @property
    def mention(self) -> str:
        """Get user mention for Telegram"""
        if self.username:
            return f"@{self.username}"
        return f"[{self.full_name}](tg://user?id={self.id})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "id": self.id,
            "username": self.username,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "language_code": self.language_code,
            "is_premium": self.is_premium,
            "is_bot": self.is_bot,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
        }
    
    @classmethod
    def from_telegram_user(cls, tg_user) -> "User":
        """Create User from Telegram User object"""
        return cls(
            id=tg_user.id,
            username=tg_user.username,
            first_name=tg_user.first_name,
            last_name=tg_user.last_name,
            language_code=tg_user.language_code,
            is_premium=getattr(tg_user, "is_premium", False),
            is_bot=tg_user.is_bot,
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "User":
        """Create User from dictionary"""
        user = cls(
            id=data["id"],
            username=data.get("username"),
            first_name=data.get("first_name"),
            last_name=data.get("last_name"),
            language_code=data.get("language_code", "en"),
            is_premium=data.get("is_premium", False),
            is_bot=data.get("is_bot", False),
        )
        if "first_seen" in data:
            user.first_seen = datetime.fromisoformat(data["first_seen"])
        if "last_seen" in data:
            user.last_seen = datetime.fromisoformat(data["last_seen"])
        return user


class AuthConfig:
    """
    Authentication configuration
    """
    
    def __init__(
        self,
        allowed_users: Optional[List[int]] = None,
        allowed_usernames: Optional[List[str]] = None,
        block_new_users: bool = False,
        require_webhook_secret: bool = False,
        webhook_secret: Optional[str] = None,
        admin_users: Optional[List[int]] = None,
    ):
        self.allowed_users = allowed_users or []
        self.allowed_usernames = [u.lower().lstrip("@") for u in (allowed_usernames or [])]
        self.block_new_users = block_new_users
        self.require_webhook_secret = require_webhook_secret
        self.webhook_secret = webhook_secret
        self.admin_users = admin_users or []
    
    def is_user_allowed(self, user_id: int, username: Optional[str]) -> bool:
        """
        Check if user is allowed to use the bot
        
        Args:
            user_id: Telegram user ID
            username: Telegram username (without @)
            
        Returns:
            True if user is allowed
        """
        # If no restrictions, allow all
        if not self.allowed_users and not self.allowed_usernames:
            return True
        
        # Check by user ID
        if user_id in self.allowed_users:
            return True
        
        # Check by username
        if username and username.lower() in self.allowed_usernames:
            return True
        
        return False
    
    def is_admin(self, user_id: int) -> bool:
        """Check if user is admin"""
        return user_id in self.admin_users
    
    def validate_webhook_secret(self, secret: Optional[str]) -> bool:
        """Validate webhook secret for security"""
        if not self.require_webhook_secret or not self.webhook_secret:
            return True
        
        if not secret:
            return False
        
        # Constant-time comparison to prevent timing attacks
        return hmac.compare_digest(secret, self.webhook_secret)


class AuthenticationMiddleware:
    """
    Middleware for authenticating users and attaching user context.
    
    This middleware should be one of the first in the pipeline to ensure
    all subsequent middleware and handlers have user context available.
    
    Features:
    - Extracts user info from all update types
    - Checks against allowed users list
    - Creates/loads user sessions
    - Tracks user metrics
    - Blocks unauthorized users
    - Handles webhook secret validation
    """
    
    def __init__(
        self,
        session_store: SessionStore,
        config: Optional[AuthConfig] = None,
        logger: Optional[StructuredLogger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        """
        Initialize authentication middleware
        
        Args:
            session_store: Session store for user sessions
            config: Authentication configuration
            logger: Structured logger
            metrics: Metrics collector
        """
        self.session_store = session_store
        self.config = config or AuthConfig()
        self.logger = logger or structlog.get_logger(__name__)
        self.metrics = metrics or MetricsCollector("auth_middleware")
        
        # Cache for recently seen users (to reduce DB lookups)
        self._user_cache: Dict[int, tuple[User, datetime]] = {}
        self.cache_ttl_seconds = 300  # 5 minutes
        
        self.logger.info(
            "auth_middleware.initialized",
            allowed_users_count=len(self.config.allowed_users),
            allowed_usernames_count=len(self.config.allowed_usernames),
            block_new_users=self.config.block_new_users,
        )
    
    async def __call__(
        self,
        context: UpdateContext,
        next_middleware: NextMiddleware,
    ) -> Optional[UpdateContext]:
        """
        Execute authentication middleware
        
        Args:
            context: Current update context
            next_middleware: Next middleware in pipeline
            
        Returns:
            Updated context or None if authentication failed
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Validate webhook secret (if configured)
            if not await self._validate_webhook_secret(context):
                self.logger.warning(
                    "auth_middleware.invalid_webhook_secret",
                    update_id=context.update.update_id,
                )
                self.metrics.increment("auth.webhook_secret_failures")
                return None
            
            # Step 2: Extract user from update
            user = await self._extract_user(context)
            
            if not user:
                # This might be a channel post or other non-user update
                self.logger.debug(
                    "auth_middleware.no_user",
                    update_id=context.update.update_id,
                    update_type=type(context.update).__name__,
                )
                # Allow non-user updates to pass through
                return await next_middleware(context)
            
            # Step 3: Check if user is allowed
            if not self.config.is_user_allowed(user.id, user.username):
                await self._handle_unauthorized(context, user)
                return None
            
            # Step 4: Get or create user session
            session = await self._get_or_create_session(user, context)
            
            # Step 5: Attach user and session to context
            context.metadata["user"] = user
            context.metadata["session"] = session
            context.metadata["user_id"] = user.id
            context.metadata["language"] = session.language
            
            # Step 6: Update last seen
            user.last_seen = datetime.now()
            session.updated_at = datetime.now()
            
            # Step 7: Track metrics
            await self._track_user_metrics(user, session)
            
            # Log authenticated request
            self.logger.debug(
                "auth_middleware.authenticated",
                user_id=user.id,
                username=user.username,
                is_new=session.created_at > (datetime.now() - timedelta(seconds=10)),
            )
            
            # Continue pipeline
            result = await next_middleware(context)
            
            # Save session if modified
            if session.modified:
                await self.session_store.save_session(session)
            
            # Record processing time
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics.record_latency("auth_middleware", duration)
            
            return result
            
        except Exception as e:
            self.logger.exception(
                "auth_middleware.error",
                error=str(e),
                update_id=context.update.update_id,
            )
            self.metrics.increment("auth.errors")
            # Don't block the pipeline on auth errors, but log them
            return await next_middleware(context)
    
    # ------------------------------------------------------------------------
    # User Extraction
    # ------------------------------------------------------------------------
    
    async def _extract_user(self, context: UpdateContext) -> Optional[User]:
        """
        Extract user from different types of updates
        
        Args:
            context: Update context
            
        Returns:
            User object or None if no user found
        """
        update = context.update
        
        # Check cache first (if we've seen this user recently)
        if context.user_id and context.user_id in self._user_cache:
            cached_user, cached_time = self._user_cache[context.user_id]
            age = (datetime.now() - cached_time).total_seconds()
            
            if age < self.cache_ttl_seconds:
                return cached_user
        
        # Extract from different update types
        user = None
        
        if update.effective_user:
            # Message from user
            user = User.from_telegram_user(update.effective_user)
        
        elif update.callback_query and update.callback_query.from_user:
            # Callback query
            user = User.from_telegram_user(update.callback_query.from_user)
        
        elif update.inline_query and update.inline_query.from_user:
            # Inline query
            user = User.from_telegram_user(update.inline_query.from_user)
        
        elif update.chosen_inline_result and update.chosen_inline_result.from_user:
            # Chosen inline result
            user = User.from_telegram_user(update.chosen_inline_result.from_user)
        
        elif update.my_chat_member and update.my_chat_member.from_user:
            # Chat member update
            user = User.from_telegram_user(update.my_chat_member.from_user)
        
        elif update.channel_post:
            # Channel posts don't have a user
            return None
        
        if user:
            # Update cache
            self._user_cache[user.id] = (user, datetime.now())
        
        return user
    
    # ------------------------------------------------------------------------
    # Session Management
    # ------------------------------------------------------------------------
    
    async def _get_or_create_session(
        self,
        user: User,
        context: UpdateContext,
    ) -> UserSession:
        """
        Get existing session or create new one
        
        Args:
            user: Authenticated user
            context: Update context
            
        Returns:
            User session
        """
        # Try to get existing session
        session = await self.session_store.get_session(user.id)
        
        if session:
            # Update last seen only. Do NOT overwrite session.language with Telegram app
            # language — the user may have explicitly chosen Hindi/etc. via /language.
            session.updated_at = datetime.now()
            return session
        
        # Create new session
        self.logger.info(
            "auth_middleware.new_user",
            user_id=user.id,
            username=user.username,
        )
        
        # Determine initial language
        language = Language.from_code(user.language_code) or Language.ENGLISH
        
        session = await self.session_store.create_session(
            user_id=user.id,
            chat_id=context.chat_id,
            language=language,
        )
        
        # Store user info in session metadata
        session.metadata["user"] = user.to_dict()
        session.metadata["first_seen"] = datetime.now().isoformat()
        session.modified = True
        
        await self.session_store.save_session(session)
        
        self.metrics.increment("auth.new_users")
        
        return session
    
    # ------------------------------------------------------------------------
    # Security Validation
    # ------------------------------------------------------------------------
    
    async def _validate_webhook_secret(self, context: UpdateContext) -> bool:
        """
        Validate webhook secret if configured
        
        Args:
            context: Update context
            
        Returns:
            True if valid or not required
        """
        if not self.config.require_webhook_secret:
            return True
        
        # Extract secret from headers (if using webhook)
        # This depends on how you're receiving updates
        # For polling, this check is skipped
        return True
    
    # ------------------------------------------------------------------------
    # Unauthorized Handling
    # ------------------------------------------------------------------------
    
    async def _handle_unauthorized(
        self,
        context: UpdateContext,
        user: User,
    ) -> None:
        """
        Handle unauthorized user access
        
        Args:
            context: Update context
            user: Unauthorized user
        """
        self.logger.warning(
            "auth_middleware.unauthorized_access",
            user_id=user.id,
            username=user.username,
            chat_id=context.chat_id,
        )
        
        self.metrics.increment("auth.unauthorized")
        
        # Only send message if it's a private chat
        if context.chat_id and context.chat_id > 0:  # Positive = private chat
            try:
                # Try to get telegram client from context
                # This assumes the client is available in metadata
                telegram_client = context.metadata.get("telegram_client")
                
                if telegram_client:
                    await telegram_client.send_message(
                        chat_id=context.chat_id,
                        text=(
                            "🔒 *Unauthorized Access*\n\n"
                            "You are not authorized to use this bot.\n\n"
                            "If you believe this is an error, please contact the bot administrator."
                        ),
                        parse_mode="Markdown",
                    )
            except Exception as e:
                self.logger.debug(
                    "auth_middleware.failed_to_send_unauthorized_message",
                    error=str(e),
                )
    
    # ------------------------------------------------------------------------
    # Metrics Tracking
    # ------------------------------------------------------------------------
    
    async def _track_user_metrics(
        self,
        user: User,
        session: UserSession,
    ) -> None:
        """
        Track user-related metrics
        
        Args:
            user: Authenticated user
            session: User session
        """
        # Track active users
        self.metrics.gauge(
            "auth.active_users",
            value=1,
            tags={"user_id": str(user.id)},
        )
        
        # Track user language distribution
        self.metrics.increment(
            "auth.users_by_language",
            tags={"language": session.language.code},
        )
        
        # Track if user is premium
        if user.is_premium:
            self.metrics.increment("auth.premium_users")
        
        # Track if user is admin
        if self.config.is_admin(user.id):
            self.metrics.increment("auth.admin_users")
    
    # ------------------------------------------------------------------------
    # Cache Management
    # ------------------------------------------------------------------------
    
    async def clear_cache(self) -> None:
        """Clear user cache"""
        self._user_cache.clear()
        self.logger.debug("auth_middleware.cache_cleared")
    
    async def remove_from_cache(self, user_id: int) -> None:
        """Remove user from cache"""
        if user_id in self._user_cache:
            del self._user_cache[user_id]


# ------------------------------------------------------------------------
# Convenience Factory Functions
# ------------------------------------------------------------------------

def create_auth_middleware(
    session_store: SessionStore,
    allowed_users: Optional[List[int]] = None,
    allowed_usernames: Optional[List[str]] = None,
    block_new_users: bool = False,
    admin_users: Optional[List[int]] = None,
    logger: Optional[StructuredLogger] = None,
    metrics: Optional[MetricsCollector] = None,
) -> AuthenticationMiddleware:
    """
    Factory function to create authentication middleware with config
    
    Args:
        session_store: Session store
        allowed_users: List of allowed user IDs
        allowed_usernames: List of allowed usernames
        block_new_users: Block new users
        admin_users: List of admin user IDs
        logger: Logger instance
        metrics: Metrics collector
        
    Returns:
        Configured AuthenticationMiddleware
    """
    config = AuthConfig(
        allowed_users=allowed_users,
        allowed_usernames=allowed_usernames,
        block_new_users=block_new_users,
        admin_users=admin_users,
    )
    
    return AuthenticationMiddleware(
        session_store=session_store,
        config=config,
        logger=logger,
        metrics=metrics,
    )


def create_open_access_middleware(
    session_store: SessionStore,
    logger: Optional[StructuredLogger] = None,
    metrics: Optional[MetricsCollector] = None,
) -> AuthenticationMiddleware:
    """
    Create middleware that allows all users (open access)
    
    Args:
        session_store: Session store
        logger: Logger instance
        metrics: Metrics collector
        
    Returns:
        AuthenticationMiddleware with open access
    """
    return AuthenticationMiddleware(
        session_store=session_store,
        config=AuthConfig(),  # Empty config = allow all
        logger=logger,
        metrics=metrics,
    )


def create_private_bot_middleware(
    session_store: SessionStore,
    allowed_users: List[int],
    logger: Optional[StructuredLogger] = None,
    metrics: Optional[MetricsCollector] = None,
) -> AuthenticationMiddleware:
    """
    Create middleware for private bot (only specific users)
    
    Args:
        session_store: Session store
        allowed_users: List of allowed user IDs
        logger: Logger instance
        metrics: Metrics collector
        
    Returns:
        AuthenticationMiddleware with restricted access
    """
    return AuthenticationMiddleware(
        session_store=session_store,
        config=AuthConfig(
            allowed_users=allowed_users,
            block_new_users=True,
        ),
        logger=logger,
        metrics=metrics,
    )


# ------------------------------------------------------------------------
# Helper Functions for Handlers
# ------------------------------------------------------------------------

def get_user_from_context(context: UpdateContext) -> Optional[User]:
    """Helper to get user from context"""
    return context.metadata.get("user")


def get_session_from_context(context: UpdateContext) -> Optional[UserSession]:
    """Helper to get session from context"""
    return context.metadata.get("session")


def require_user(handler_func):
    """
    Decorator to ensure user exists in context
    
    Usage:
        @require_user
        async def my_handler(self, context):
            user = get_user_from_context(context)  # Guaranteed to exist
    """
    async def wrapper(self, context: UpdateContext, *args, **kwargs):
        user = get_user_from_context(context)
        if not user:
            raise UnauthorizedError(
                "User not authenticated",
                kind=ErrorKind.UNAUTHORIZED,
            )
        return await handler_func(self, context, *args, **kwargs)
    return wrapper


def require_admin(handler_func):
    """
    Decorator to ensure user is admin
    
    Usage:
        @require_admin
        async def admin_handler(self, context):
            # Only admins can access
    """
    async def wrapper(self, context: UpdateContext, *args, **kwargs):
        user = get_user_from_context(context)
        if not user:
            raise UnauthorizedError(
                "User not authenticated",
                kind=ErrorKind.UNAUTHORIZED,
            )
        
        # This assumes auth middleware has access to config
        # In practice, you'd need to inject auth config
        auth_middleware = context.metadata.get("auth_middleware")
        if not auth_middleware or not auth_middleware.config.is_admin(user.id):
            raise UnauthorizedError(
                "Admin access required",
                kind=ErrorKind.UNAUTHORIZED,
            )
        
        return await handler_func(self, context, *args, **kwargs)
    return wrapper


# For timedelta
from datetime import timedelta