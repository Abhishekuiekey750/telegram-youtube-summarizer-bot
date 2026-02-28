"""
Shared error types and exception hierarchy
"""

from enum import Enum
from typing import Optional, Dict, Any, Type


class ErrorKind(Enum):
    """Error classification for handling"""
    VALIDATION = "validation"
    NOT_FOUND = "not_found"
    INTERNAL = "internal"
    EXTERNAL_API = "external_api"
    UNAUTHORIZED = "unauthorized"
    RATE_LIMIT = "rate_limit"
    RETRYABLE = "retryable"


class BotError(Exception):
    """Base exception for bot errors"""
    
    def __init__(
        self,
        message: str,
        kind: ErrorKind = ErrorKind.INTERNAL,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.kind = kind
        self.context = context or {}
        self.original_error = original_error


class ValidationError(BotError):
    """Invalid input or state"""
    
    def __init__(
        self,
        message: str,
        kind: ErrorKind = ErrorKind.VALIDATION,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message, kind, context, original_error)


class NotFoundError(BotError):
    """Resource not found"""
    
    def __init__(
        self,
        message: str,
        kind: ErrorKind = ErrorKind.NOT_FOUND,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message, kind, context, original_error)


class SessionError(BotError):
    """Session storage/retrieval error"""
    
    def __init__(
        self,
        message: str,
        kind: ErrorKind = ErrorKind.INTERNAL,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message, kind, context, original_error)


class RetryableError(BotError):
    """Transient error that can be retried"""
    
    def __init__(
        self,
        message: str,
        kind: ErrorKind = ErrorKind.RETRYABLE,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message, kind, context, original_error)


class TranslationError(BotError):
    """Translation service error"""
    
    def __init__(
        self,
        message: str,
        kind: ErrorKind = ErrorKind.EXTERNAL_API,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message, kind, context, original_error)


class RateLimitError(BotError):
    """Rate limit exceeded"""
    
    def __init__(
        self,
        message: str,
        kind: ErrorKind = ErrorKind.RATE_LIMIT,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message, kind, context, original_error)


class TranscriptError(BotError):
    """Transcript fetch/parse error"""
    
    def __init__(
        self,
        message: str,
        kind: ErrorKind = ErrorKind.NOT_FOUND,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message, kind, context, original_error)


class UnauthorizedError(BotError):
    """Unauthorized access"""
    
    def __init__(
        self,
        message: str,
        kind: ErrorKind = ErrorKind.UNAUTHORIZED,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message, kind, context, original_error)
