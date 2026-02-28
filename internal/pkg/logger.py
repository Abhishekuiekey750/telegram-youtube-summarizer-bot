"""
Structured logging with structlog
"""

from typing import Optional, Any, Dict
from dataclasses import dataclass, field
import structlog


@dataclass
class LogConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "json"
    include_timestamp: bool = True
    include_level: bool = True


class StructuredLogger:
    def __init__(self, *args, **kwargs):
        self._logger = structlog.get_logger()

    def bind(self, **kwargs):
        return self._logger.bind(**kwargs)

    def isEnabledFor(self, level):
        return True  # simple fallback

    def debug(self, *args, **kwargs):
        self._logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        self._logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        self._logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        self._logger.error(*args, **kwargs)

    def exception(self, *args, **kwargs):
        self._logger.exception(*args, **kwargs)

    async def close(self):
        pass
   
   
    
 
