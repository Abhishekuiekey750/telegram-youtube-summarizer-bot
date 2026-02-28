"""Summary generation."""

from internal.services.summarizer.orchestrator import (
    SummarizationOrchestrator as SummarizerService,
    VideoSummary as Summary,
    SummaryType,
)
from internal.pkg.errors import BotError
SummaryError = BotError  # Alias

__all__ = ["SummarizerService", "Summary", "SummaryType", "SummaryError"]
