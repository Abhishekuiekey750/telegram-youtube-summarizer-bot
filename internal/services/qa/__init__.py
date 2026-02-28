"""Q&A orchestration and retrieval."""

from internal.services.qa.orchestrator import (
    QAOrchestrator,
    QAContext,
    QAAnswer,
)
from internal.services.qa.chunker import SemanticChunker
from internal.services.qa.retriever import ContextRetriever
from internal.services.qa.chunker import Chunk
from datetime import datetime
from typing import List, Dict, Any


class Answer:
    """Answer model for Q&A responses"""
    
    def __init__(self, text: str, found_in_context: bool = True, confidence: float = 1.0,
                 chunks_used: List[str] = None, metadata: Dict[str, Any] = None):
        self.text = text
        self.found_in_context = found_in_context
        self.confidence = confidence
        self.chunks_used = chunks_used or []
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "found_in_context": self.found_in_context,
            "confidence": self.confidence,
            "chunks_used": self.chunks_used,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class QAService(QAOrchestrator):
    """QAService alias - also provides chunk_transcript"""
    
    def __init__(self, vector_db=None, embedding_generator=None, model_factory=None,
                 session_store=None, logger=None, metrics=None, **kwargs):
        from internal.ai.prompts.manager import PromptManager
        from internal.services.language import LanguageService
        
        chunker = SemanticChunker(logger=logger, metrics=metrics)
        retriever = ContextRetriever(
            vector_db=vector_db,
            embedding_generator=embedding_generator,
            logger=logger,
            metrics=metrics,
        )
        prompt_manager = PromptManager(logger=logger, metrics=metrics)
        language_service = LanguageService(model_factory=model_factory, logger=logger)
        
        super().__init__(
            retriever=retriever,
            model_factory=model_factory,
            prompt_manager=prompt_manager,
            language_service=language_service,
            session_store=session_store,
            logger=logger,
            metrics=metrics,
            **kwargs,
        )
        self._chunker = chunker
    
    async def chunk_transcript(self, transcript, chunk_size=500, overlap=50):
        """Chunk transcript for vector storage"""
        return await self._chunker.chunk_transcript(transcript)

__all__ = ["QAService", "QAContext", "QAAnswer", "Answer"]
