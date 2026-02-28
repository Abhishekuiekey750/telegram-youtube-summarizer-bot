"""
Domain events (event bus)
"""

from typing import Any, Callable, Awaitable, Dict, List
import asyncio


class EventBus:
    """
    Minimal event bus for domain events.
    Allows subscribe/publish pattern.
    """
    
    def __init__(self, logger=None, **kwargs):
        self._logger = logger
        self._handlers: Dict[str, List[Callable[..., Awaitable[None]]]] = {}
    
    async def publish(self, event_type: str, **kwargs: Any) -> None:
        """Publish event to all subscribers"""
        handlers = self._handlers.get(event_type, [])
        for handler in handlers:
            try:
                await handler(**kwargs)
            except Exception as e:
                if self._logger:
                    self._logger.exception("event_bus.handler_failed", event=event_type, error=str(e))
    
    def subscribe(self, event_type: str, handler: Callable[..., Awaitable[None]]) -> None:
        """Subscribe to event type"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
