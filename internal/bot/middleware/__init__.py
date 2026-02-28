"""Bot middleware pipeline."""

from internal.bot.middleware.logging import LoggingMiddleware
from internal.bot.middleware.rate_limit import RateLimitMiddleware
from internal.bot.middleware.recovery import RecoveryMiddleware
from internal.bot.middleware.auth import AuthenticationMiddleware

# AuthMiddleware is alias for AuthenticationMiddleware
AuthMiddleware = AuthenticationMiddleware

__all__ = [
    "LoggingMiddleware",
    "RateLimitMiddleware",
    "RecoveryMiddleware",
    "AuthMiddleware",
]
