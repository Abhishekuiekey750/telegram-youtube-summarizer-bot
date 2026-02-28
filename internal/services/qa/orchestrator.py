"""
QA Orchestrator
Coordinates the entire Q&A process from question to validated answer

Flow:
1. Validate input and session
2. Retrieve relevant context
3. Generate answer using LLM
4. Validate grounding
5. Format and return response
6. Update conversation history
"""

import time
import asyncio
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field

import structlog

from internal.services.qa.retriever import ContextRetriever, RetrievedChunk
from internal.services.qa.chunker import Chunk
from internal.ai.models.factory import ModelFactory
from internal.ai.prompts.manager import PromptManager
from internal.services.language import LanguageService
from internal.storage.session import SessionStore, UserSession
from internal.domain.value_objects import Language
from internal.pkg.errors import ValidationError, NotFoundError, BotError
from internal.pkg.metrics import MetricsCollector


@dataclass
class QAContext:
    """Context for a Q&A interaction"""
    question: str
    video_id: str
    user_id: int
    language: Language
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    
    @property
    def has_context(self) -> bool:
        return len(self.retrieved_chunks) > 0
    
    @property
    def retrieval_confidence(self) -> float:
        if not self.retrieved_chunks:
            return 0.0
        return sum(r.relevance_score for r in self.retrieved_chunks) / len(self.retrieved_chunks)


@dataclass
class QAAnswer:
    """Structured answer from QA system"""
    text: str
    confidence: float
    grounded: bool
    retrieved_chunks: List[RetrievedChunk]
    processing_time: float
    language: Language
    needs_translation: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_reliable(self) -> bool:
        """Check if answer is reliable enough to show"""
        return self.grounded and self.confidence >= 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for response"""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "grounded": self.grounded,
            "language": self.language.code,
            "processing_time_ms": round(self.processing_time * 1000),
        }


class QAOrchestrator:
    """
    Orchestrates the entire Q&A process
    
    Responsibilities:
    - Coordinate retrieval, generation, and validation
    - Manage conversation context
    - Track metrics and performance
    - Handle errors gracefully
    - Ensure answers are grounded
    """
    
    def __init__(
        self,
        retriever: ContextRetriever,
        model_factory: ModelFactory,
        prompt_manager: PromptManager,
        language_service: LanguageService,
        session_store: SessionStore,
        logger=None,
        metrics=None,
        model_name: str = "gpt-3.5-turbo",
        min_confidence: float = 0.5,
        enable_history: bool = True,
        max_history_turns: int = 3,
    ):
        """
        Initialize QA orchestrator
        
        Args:
            retriever: Context retriever
            model_factory: Factory for AI models
            prompt_manager: Prompt template manager
            language_service: Language service
            session_store: Session store
            logger: Structured logger
            metrics: Metrics collector
            model_name: Default model to use
            min_confidence: Minimum confidence for answers
            enable_history: Whether to use conversation history
            max_history_turns: Maximum history turns to include
        """
        self.retriever = retriever
        self.model_factory = model_factory
        self.prompt_manager = prompt_manager
        self.language_service = language_service
        self.session_store = session_store
        self.logger = logger or structlog.get_logger(__name__)
        self.metrics = metrics or MetricsCollector("qa_orchestrator")
        
        self.model_name = model_name
        self.min_confidence = min_confidence
        self.enable_history = enable_history
        self.max_history_turns = max_history_turns
        
        self.logger.info(
            "qa_orchestrator.initialized",
            model=model_name,
            min_confidence=min_confidence,
            enable_history=enable_history,
        )
    
    async def answer_question(
        self,
        question: str,
        user_id: int,
        video_id: Optional[str] = None,
        language: Optional[Language] = None,
        session: Optional[UserSession] = None,
    ) -> QAAnswer:
        """
        Main entry point for answering questions
        
        Args:
            question: User's question
            user_id: User ID
            video_id: Optional video ID (uses session if not provided)
            language: Optional language (uses session if not provided)
            session: Optional pre-loaded session
            
        Returns:
            Structured answer with metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Load or validate session
            if not session:
                session = await self._load_session(user_id, video_id)
            
            # Step 2: Validate we have a video to ask about
            if not session.current_video_id and not video_id:
                raise ValidationError(
                    "No active video. Please send a YouTube link first."
                )
            
            # Use provided video_id or from session
            target_video_id = video_id or session.current_video_id
            
            # Determine language
            target_language = language or session.language
            
            self.logger.info(
                "qa_orchestrator.starting",
                user_id=user_id,
                video_id=target_video_id,
                language=target_language.code,
                question_preview=question[:50],
            )
            
            # Step 3: Build QA context
            context = QAContext(
                question=question,
                video_id=target_video_id,
                user_id=user_id,
                language=target_language,
            )
            
            # Step 4: Add conversation history if enabled
            if self.enable_history:
                context.conversation_history = await self._get_conversation_history(
                    session, target_video_id
                )
            
            # Step 5: Retrieve relevant chunks
            context.retrieved_chunks = await self.retriever.retrieve(
                question=question,
                video_id=target_video_id,
            )
            
            # Step 6: Check if we have any context
            if not context.has_context:
                return await self._handle_no_context(context, start_time)
            
            # Step 7: Generate answer
            answer_text = await self._generate_answer(context)
            
            # Step 8: Validate grounding
            is_grounded, confidence = await self.retriever.verify_answer(
                answer=answer_text,
                retrieved_chunks=context.retrieved_chunks,
            )
            
            # Step 9: Create answer object
            answer = QAAnswer(
                text=answer_text,
                confidence=confidence,
                grounded=is_grounded,
                retrieved_chunks=context.retrieved_chunks,
                processing_time=time.time() - start_time,
                language=target_language,
                metadata={
                    "retrieval_confidence": context.retrieval_confidence,
                    "chunks_used": len(context.retrieved_chunks),
                    "history_used": len(context.conversation_history) > 0,
                },
            )
            
            # Step 10: Translate if needed
            if target_language != Language.ENGLISH and not self._is_model_multilingual():
                answer = await self._translate_answer(answer, target_language)
            
            # Step 11: Update session with conversation history
            await self._update_conversation_history(session, question, answer)
            
            # Step 12: Track metrics
            await self._track_metrics(answer, context)
            
            self.logger.info(
                "qa_orchestrator.completed",
                user_id=user_id,
                grounded=answer.grounded,
                confidence=round(answer.confidence, 2),
                duration_ms=round(answer.processing_time * 1000),
                chunks=len(answer.retrieved_chunks),
            )
            
            return answer
            
        except Exception as e:
            self.logger.exception(
                "qa_orchestrator.failed",
                user_id=user_id,
                error=str(e),
            )
            self.metrics.increment("qa.errors", tags={"error": e.__class__.__name__})
            raise
    
    # ------------------------------------------------------------------------
    # Answer Generation
    # ------------------------------------------------------------------------
    
    async def _generate_answer(self, context: QAContext) -> str:
        """
        Generate answer using LLM with context
        
        Args:
            context: QA context with question and retrieved chunks
            
        Returns:
            Generated answer text
        """
        # Format context with timestamps
        formatted_context = await self.retriever.format_context(
            context.retrieved_chunks,
            include_timestamps=True,
        )
        
        # Format conversation history
        history_text = ""
        if context.conversation_history:
            history_text = self._format_conversation_history(
                context.conversation_history
            )
        
        # Get prompt (with fallback when no templates are configured)
        try:
            prompt = await self.prompt_manager.get_qa_prompt(
                question=context.question,
                context=formatted_context,
                history=history_text if history_text else None,
                language=context.language,
            )
        except NotFoundError:
            prompt = self._build_fallback_qa_prompt(
                question=context.question,
                context=formatted_context,
                history=history_text,
            )
        
        # Get model and generate
        model = await self.model_factory.get_generation_model(self.model_name)
        response = await model.generate(
            prompt=prompt,
            language=context.language,
            temperature=0.3,  # Lower temperature for factual answers
            max_tokens=500,
        )
        
        return response.strip()
    
    def _format_conversation_history(
        self,
        history: List[Dict[str, Any]],
    ) -> str:
        """Format conversation history for prompt"""
        lines = ["Previous conversation:"]
        
        for turn in history[-self.max_history_turns:]:
            lines.append(f"User: {turn['question']}")
            lines.append(f"Assistant: {turn['answer']}")
        
        return "\n".join(lines)
    
    def _build_fallback_qa_prompt(
        self,
        question: str,
        context: str,
        history: str = "",
    ) -> str:
        """Build a minimal Q&A prompt when no template is configured."""
        parts = [
            "Answer the user's question using ONLY the following transcript excerpts. Do not invent or add information.",
            "",
            "Transcript excerpts:",
            context,
            "",
        ]
        if history:
            parts.extend([history, ""])
        parts.extend([
            "Question: " + question,
            "",
            "Answer (based only on the transcript above):",
        ])
        return "\n".join(parts)
    
    # ------------------------------------------------------------------------
    # No Context Handling
    # ------------------------------------------------------------------------
    
    async def _handle_no_context(
        self,
        context: QAContext,
        start_time: float,
    ) -> QAAnswer:
        """
        Handle case where no relevant context found
        
        Returns a "not covered" response
        """
        self.logger.debug(
            "qa_orchestrator.no_context",
            user_id=context.user_id,
            question=context.question[:50],
        )
        
        # Get not covered message in user's language
        message = await self._get_not_covered_message(context.language)
        
        return QAAnswer(
            text=message,
            confidence=1.0,  # High confidence for this meta-response
            grounded=True,    # Always grounded (it's a system response)
            retrieved_chunks=[],
            processing_time=time.time() - start_time,
            language=context.language,
            metadata={"reason": "no_relevant_context"},
        )
    
    async def _get_not_covered_message(self, language: Language) -> str:
        """Get "not covered" message in user's language"""
        messages = {
            Language.ENGLISH: "I couldn't find information about that in the video transcript. The video might not cover this topic.",
            Language.HINDI: "मुझे वीडियो ट्रांसक्रिप्ट में इस बारे में जानकारी नहीं मिली। हो सकता है वीडियो में यह विषय कवर न किया गया हो।",
            Language.TAMIL: "வீடியோ டிரான்ஸ்கிரிப்டில் இந்த தகவலை என்னால் கண்டுபிடிக்க முடியவில்லை. வீடியோ இந்த தலைப்பை உள்ளடக்கியிருக்காமல் இருக்கலாம்.",
            Language.TELUGU: "వీడియో ట్రాన్స్క్రిప్ట్‌లో ఈ సమాచారం నాకు కనుగొనబడలేదు. వీడియో ఈ అంశాన్ని కవర్ చేయకపోవచ్చు.",
            Language.KANNADA: "ವೀಡಿಯೊ ಟ್ರಾನ್ಸ್ಕ್ರಿಪ್ಟ್‌ನಲ್ಲಿ ಈ ಮಾಹಿತಿಯನ್ನು ನನಗೆ ಕಂಡುಹಿಡಿಯಲಾಗಲಿಲ್ಲ. ವೀಡಿಯೊ ಈ ವಿಷಯವನ್ನು ಒಳಗೊಂಡಿರದಿರಬಹುದು.",
        }
        
        return messages.get(language, messages[Language.ENGLISH])
    
    # ------------------------------------------------------------------------
    # Translation Support
    # ------------------------------------------------------------------------
    
    def _is_model_multilingual(self) -> bool:
        """Check if current model supports multiple languages"""
        multilingual_models = ["gpt-4", "claude", "gemini"]
        return any(m in self.model_name.lower() for m in multilingual_models)
    
    async def _translate_answer(
        self,
        answer: QAAnswer,
        target_language: Language,
    ) -> QAAnswer:
        """Translate answer to target language"""
        try:
            translated = await self.language_service.translate(
                text=answer.text,
                source_language=Language.ENGLISH,
                target_language=target_language,
            )
            
            answer.text = translated
            answer.needs_translation = True
            
        except Exception as e:
            self.logger.warning(
                "qa_orchestrator.translation_failed",
                error=str(e),
                target_language=target_language.code,
            )
            # Keep original English
        
        return answer
    
    # ------------------------------------------------------------------------
    # Session Management
    # ------------------------------------------------------------------------
    
    async def _load_session(
        self,
        user_id: int,
        video_id: Optional[str] = None,
    ) -> UserSession:
        """Load user session"""
        session = await self.session_store.get_session(user_id)
        
        if not session:
            session = await self.session_store.create_session(
                user_id=user_id,
                chat_id=0,  # Will be updated by handler
            )
        
        # Update current video if provided
        if video_id:
            session.current_video_id = video_id
            session.modified = True
            await self.session_store.save_session(session)
        
        return session
    
    async def _get_conversation_history(
        self,
        session: UserSession,
        video_id: str,
    ) -> List[Dict[str, Any]]:
        """Get conversation history for this video"""
        if not session.metadata:
            return []
        
        history = session.metadata.get("conversation_history", [])
        
        # Filter by video_id
        video_history = [
            turn for turn in history
            if turn.get("video_id") == video_id
        ]
        
        return video_history[-self.max_history_turns:]
    
    async def _update_conversation_history(
        self,
        session: UserSession,
        question: str,
        answer: QAAnswer,
    ) -> None:
        """Update conversation history"""
        if not self.enable_history:
            return
        
        if not session.metadata:
            session.metadata = {}
        
        history = session.metadata.get("conversation_history", [])
        
        # Add new turn
        history.append({
            "question": question,
            "answer": answer.text,
            "video_id": session.current_video_id,
            "confidence": answer.confidence,
            "grounded": answer.grounded,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Keep only last N turns
        if len(history) > self.max_history_turns * 2:  # Allow some buffer
            history = history[-self.max_history_turns:]
        
        session.metadata["conversation_history"] = history
        session.metadata["last_question_time"] = datetime.now().isoformat()
        session.metadata["questions_asked"] = session.metadata.get("questions_asked", 0) + 1
        
        session.modified = True
        await self.session_store.save_session(session)
    
    # ------------------------------------------------------------------------
    # Metrics Tracking
    # ------------------------------------------------------------------------
    
    async def _track_metrics(
        self,
        answer: QAAnswer,
        context: QAContext,
    ) -> None:
        """Track Q&A metrics"""
        self.metrics.record_latency(
            "qa.total_duration",
            answer.processing_time,
            tags={"grounded": str(answer.grounded)},
        )
        
        self.metrics.gauge(
            "qa.confidence",
            value=answer.confidence,
            tags={"grounded": str(answer.grounded)},
        )
        
        self.metrics.gauge(
            "qa.chunks_used",
            value=len(answer.retrieved_chunks),
        )
        
        self.metrics.gauge(
            "qa.retrieval_confidence",
            value=context.retrieval_confidence,
        )
        
        self.metrics.increment(
            "qa.total",
            tags={
                "grounded": str(answer.grounded),
                "reliable": str(answer.is_reliable),
            },
        )
        
        if not answer.grounded:
            self.metrics.increment("qa.ungrounded_answers")
    
    # ------------------------------------------------------------------------
    # Public API for Testing
    # ------------------------------------------------------------------------
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            "model": self.model_name,
            "min_confidence": self.min_confidence,
            "enable_history": self.enable_history,
            "max_history_turns": self.max_history_turns,
        }
    
    async def clear_history(self, user_id: int) -> None:
        """Clear conversation history for a user"""
        session = await self.session_store.get_session(user_id)
        if session and session.metadata:
            session.metadata.pop("conversation_history", None)
            session.modified = True
            await self.session_store.save_session(session)
            
        self.logger.info("qa_orchestrator.history_cleared", user_id=user_id)


# ------------------------------------------------------------------------
# Factory Function
# ------------------------------------------------------------------------

async def create_qa_orchestrator(
    retriever: ContextRetriever,
    model_factory: ModelFactory,
    prompt_manager: PromptManager,
    language_service: LanguageService,
    session_store: SessionStore,
    model_name: str = "gpt-3.5-turbo",
    min_confidence: float = 0.5,
    logger=None,
    metrics=None,
) -> QAOrchestrator:
    """
    Create QA orchestrator with configuration
    
    Args:
        retriever: Context retriever
        model_factory: Model factory
        prompt_manager: Prompt manager
        language_service: Language service
        session_store: Session store
        model_name: Default model
        min_confidence: Minimum confidence
        logger: Logger instance
        metrics: Metrics collector
        
    Returns:
        Configured QAOrchestrator
    """
    return QAOrchestrator(
        retriever=retriever,
        model_factory=model_factory,
        prompt_manager=prompt_manager,
        language_service=language_service,
        session_store=session_store,
        logger=logger,
        metrics=metrics,
        model_name=model_name,
        min_confidence=min_confidence,
    )


# ------------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------------

"""
# Usage example:

orchestrator = await create_qa_orchestrator(
    retriever=retriever,
    model_factory=model_factory,
    prompt_manager=prompt_manager,
    language_service=language_service,
    session_store=session_store,
)

# Answer a question
answer = await orchestrator.answer_question(
    question="What are the pricing options?",
    user_id=12345,
    video_id="abc123",
    language=Language.ENGLISH,
)

# Use the answer
if answer.is_reliable:
    print(f"Answer: {answer.text}")
    print(f"Confidence: {answer.confidence:.2f}")
    print(f"Chunks used: {len(answer.retrieved_chunks)}")
else:
    print("Answer may not be reliable")
    print(f"Fallback: {answer.text}")

# With conversation history
answer2 = await orchestrator.answer_question(
    question="What about enterprise?",
    user_id=12345,  # Same user, remembers context
)

# Clear history
await orchestrator.clear_history(12345)
"""

# ------------------------------------------------------------------------
# Example Answer Output
# ------------------------------------------------------------------------

"""
Reliable answer:
{
    "text": "The video mentions three pricing tiers: Basic at $49/month, Pro at $99/month, and Enterprise with custom pricing starting at $999/month.",
    "confidence": 0.92,
    "grounded": true,
    "language": "en",
    "processing_time_ms": 1234,
    "chunks_used": 3
}

Not covered:
{
    "text": "I couldn't find information about that in the video transcript.",
    "confidence": 1.0,
    "grounded": true,
    "language": "en",
    "processing_time_ms": 45,
    "chunks_used": 0
}

Low confidence:
{
    "text": "The video briefly mentions pricing but doesn't give specific numbers...",
    "confidence": 0.35,
    "grounded": false,
    "language": "en",
    "processing_time_ms": 890,
    "chunks_used": 2
}
"""