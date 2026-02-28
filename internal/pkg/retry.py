"""
Retry utilities with exponential backoff
"""

import asyncio
from typing import TypeVar, Callable, Awaitable, Optional, Type
from dataclasses import dataclass


T = TypeVar("T")


@dataclass
class RetryConfig:
    """Retry configuration"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0


async def retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    *args,
    config: Optional[RetryConfig] = None,
    retryable_exceptions: Optional[tuple] = None,
    **kwargs,
) -> T:
    """
    Execute async function with exponential backoff retry.
    
    Args:
        func: Async function to execute
        *args: Positional arguments for func
        config: Retry configuration
        retryable_exceptions: Tuple of exception types to retry
        **kwargs: Keyword arguments for func
        
    Returns:
        Result of func
        
    Raises:
        Last exception if all retries fail
    """
    cfg = config or RetryConfig()
    retryable = retryable_exceptions or (Exception,)
    last_error = None
    
    for attempt in range(cfg.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except retryable as e:
            last_error = e
            if attempt == cfg.max_retries:
                raise
            delay = min(
                cfg.base_delay * (cfg.exponential_base ** attempt),
                cfg.max_delay
            )
            await asyncio.sleep(delay)
    
    if last_error:
        raise last_error
    raise RuntimeError("Retry loop exited without result or exception")
