"""
Metrics collection (stub implementation)
"""

from typing import Optional, Dict, Any

import structlog


class MetricsCollector:
    """
    Collects application metrics.
    Minimal stub - keeps counters/latencies in memory and logs gauges.
    """
    
    def __init__(self, service_name: str = "app", config: Any = None, **kwargs: Any):
        self._service_name = service_name
        self._counters: Dict[str, int] = {}
        self._latencies: Dict[str, list] = {}
        self.logger = structlog.get_logger("metrics").bind(service_name=service_name)
    
    def increment(
        self,
        name: str,
        value: int = 1,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        key = name
        if tags:
            key = f"{name}:{hash(frozenset(tags.items()))}"
        self._counters[key] = self._counters.get(key, 0) + value
    
    def record_latency(
        self,
        name: str,
        duration: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        key = name
        if tags:
            key = f"{name}:{hash(frozenset(tags.items()))}"
        if key not in self._latencies:
            self._latencies[key] = []
        self._latencies[key].append(duration)
        # Keep last 1000
        if len(self._latencies[key]) > 1000:
            self._latencies[key] = self._latencies[key][-1000:]
    
    async def flush(self) -> None:
        """Flush metrics to backend (no-op in stub)"""
        # In a real implementation, this would push metrics to a backend.
        return None
    
    def gauge(
        self,
        name: str,
        value: float = 1,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a gauge-style metric (logged only in this stub)."""
        self.logger.debug(
            "metrics.gauge",
            metric=name,
            value=value,
            tags=tags or {},
        )
    
    def record_distribution(
        self,
        name: str,
        values: list[float] | list[int],
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record a distribution of values (stub).
        
        The real implementation might send histograms to a metrics backend.
        Here we just log the basic stats so callers like the chunker can
        call this safely.
        """
        if not values:
            return
        try:
            count = len(values)
            minimum = min(values)
            maximum = max(values)
            avg = sum(values) / count
        except Exception:
            # If something goes wrong, avoid crashing the app.
            count = len(values)
            minimum = maximum = avg = 0
        
        self.logger.debug(
            "metrics.distribution",
            metric=name,
            count=count,
            min=minimum,
            max=maximum,
            avg=avg,
            tags=tags or {},
        )