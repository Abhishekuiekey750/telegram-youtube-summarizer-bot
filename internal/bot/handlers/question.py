"""
Question Handler
Processes follow-up questions about videos using RAG architecture.

Meets Q&A requirements:
- Multiple follow-up questions: conversation history per user session.
- Answers grounded in transcript: retrieval from vector DB + grounding validation.
- No hallucinations: ungrounded answers are replaced with "not covered" response.
- Multiple users simultaneously: session and current_video_id are per user_id.
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import asyncio
import json

from telegram import Message
import structlog
import numpy as np

from internal.bot.handlers.base import BaseHandler
from internal.bot.dispatcher import UpdateContext
from internal.services.qa import QAService, QAContext, Answer  # noqa: F401
from internal.services.language import LanguageService
from internal.services.youtube import YouTubeService
from internal.services.youtube.transcript import Transcript
from internal.storage.session import SessionStore, UserSession
from internal.storage.vector import VectorDBClient
from internal.ai.embedding import EmbeddingGenerator
from internal.domain.value_objects import Language, VideoId
from internal.pkg.errors import ValidationError, ErrorKind, NotFoundError


class QuestionHandler(BaseHandler):
    """
    Handles follow-up questions about videos using RAG architecture.
    
    Workflow:
    1. Verify user has an active video session
    2. Retrieve relevant transcript chunks via semantic search
    3. Re-rank chunks for optimal context
    4. Generate grounded answer using LLM
    5. Validate answer is in context (no hallucinations)
    6. Translate answer if needed
    7. Send response with confidence indicator
    
    Features:
    - Multi-turn conversations with context
    - Semantic search for relevant chunks
    - Hallucination prevention through validation
    - Confidence scoring
    - Multilingual Q&A
    - Conversation history tracking
    """
    
    def __init__(
        self,
        telegram_client,
        qa_service: QAService,
        language_service: LanguageService,
        session_store: SessionStore,
        vector_db: VectorDBClient,
        embedding_generator: EmbeddingGenerator,
        youtube_service: Optional[YouTubeService] = None,
        logger=None,
        metrics=None,
    ):
        """
        Initialize question handler with all required services.

        Args:
            telegram_client: Client for sending messages
            qa_service: Service for answering questions
            language_service: Service for multilingual support
            session_store: Session manager for user data
            vector_db: Vector database for transcript retrieval
            embedding_generator: Generator for question embeddings
            youtube_service: For on-demand transcript fetch when vector DB is empty (e.g. after restart)
            logger: Structured logger
            metrics: Metrics collector
        """
        super().__init__(telegram_client, logger, metrics)

        self.qa_service = qa_service
        self.language_service = language_service
        self.session_store = session_store
        self.vector_db = vector_db
        self.embedding_generator = embedding_generator
        self.youtube_service = youtube_service
        
        # Configuration
        self.max_context_chunks = 5
        self.min_relevance_score = 0.7
        self.max_question_length = 500
        self.enable_conversation_history = True
        self.max_history_turns = 3
        
        # Response templates
        self.not_found_responses = {
            Language.ENGLISH: "This topic is not covered in the video.",
            Language.HINDI: "यह विषय वीडियो में कवर नहीं किया गया है।",
            Language.TAMIL: "இந்த தலைப்பு வீடியோவில் உள்ளடக்கப்படவில்லை.",
            Language.TELUGU: "ఈ అంశం వీడియోలో కవర్ చేయబడలేదు.",
            Language.KANNADA: "ಈ ವಿಷಯವನ್ನು ವೀಡಿಯೊದಲ್ಲಿ ಒಳಗೊಂಡಿಲ್ಲ.",
        }
        
        self.logger.info(
            "question_handler.initialized",
            max_chunks=self.max_context_chunks,
            min_score=self.min_relevance_score,
        )
    
    @property
    def handler_name(self) -> str:
        return "question"
    
    async def handle(self, context: UpdateContext) -> None:
        """
        Main entry point for processing questions.
        
        Args:
            context: Update context with user's question
        """
        start_time = datetime.now()
        thinking_message: Optional[Message] = None
        
        try:
            await self.before_handle(context)
            
            # Validate required fields
            await self._validate_required_fields(
                context,
                ["chat_id", "user_id", "message_text"],
            )
            
            # Validate question length
            if len(context.message_text) > self.max_question_length:
                raise ValidationError(
                    f"Question too long (max {self.max_question_length} characters)",
                    kind=ErrorKind.VALIDATION,
                )
            
            # Load user session
            session = await self.session_store.get_session(context.user_id)
            if not session or not session.current_video_id:
                await self._send_no_video_response(context)
                return
            
            # Add language to context
            context.metadata["language"] = session.language
            
            # Send typing indicator
            await self._send_typing_indicator(context.chat_id)
            
            # Send "thinking" message for better UX
            thinking_message = await self._send_thinking_message(
                context.chat_id,
                session.language,
            )
            
            # Process the question
            answer = await self._process_question(
                question=context.message_text,
                video_id=session.current_video_id,
                user_id=context.user_id,
                language=session.language,
                session=session,
            )
            
            # Delete thinking message
            if thinking_message:
                await thinking_message.delete()
            
            # Send answer
            await self._send_answer(
                chat_id=context.chat_id,
                answer=answer,
                language=session.language,
                reply_to_message_id=context.update.effective_message.message_id,
            )
            
            # Update conversation history
            if self.enable_conversation_history:
                await self._update_conversation_history(
                    session=session,
                    question=context.message_text,
                    answer=answer,
                )
            
            # Track metrics
            self._track_user_activity(context.user_id, "question_asked")
            self.metrics.increment(
                "questions.answered",
                tags={
                    "language": session.language.code,
                    "found": answer.found_in_context,
                    "confidence": answer.confidence,
                },
            )
            
            self.logger.info(
                "question_handler.processed",
                user_id=context.user_id,
                video_id=session.current_video_id,
                found=answer.found_in_context,
                confidence=answer.confidence,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )
            
            await self.after_handle(context)
            
        except ValidationError as e:
            await self._handle_validation_error(context, e)
        except NotFoundError as e:
            await self._handle_not_found_error(context, e)
        except Exception as e:
            await self.on_error(context, e)
            raise
        finally:
            self._track_processing_time(start_time)
    
    # ------------------------------------------------------------------------
    # Core Q&A Processing
    # ------------------------------------------------------------------------
    
    async def _process_question(
        self,
        question: str,
        video_id: str,
        user_id: int,
        language: Language,
        session: UserSession,
    ) -> Answer:
        """
        Process a question using RAG architecture.
        
        Args:
            question: User's question
            video_id: Current video ID
            user_id: User ID for history
            language: Target language
            session: User session for history
            
        Returns:
            Answer object with response and metadata
        """
        # Ensure vector DB has chunks for this video (e.g. after bot restart)
        if self.youtube_service:
            dummy_embedding = await self.embedding_generator.embed("summary")
            existing = await self.vector_db.search_similar(
                video_id=video_id, query_embedding=dummy_embedding, limit=1
            )
            if not existing:
                await self._prepare_qa_for_video(video_id)

        # Use QA orchestrator: it does retrieval, answer generation, and grounding
        qa_answer = await self.qa_service.answer_question(
            question=question,
            user_id=user_id,
            video_id=video_id,
            language=language,
            session=session,
        )

        # Map QAAnswer to handler's Answer type
        chunks_used = []
        for r in getattr(qa_answer, "retrieved_chunks", []) or []:
            text = getattr(getattr(r, "chunk", None), "text", None) or getattr(r, "text", str(r))
            if text:
                chunks_used.append(text if isinstance(text, str) else str(text))

        return Answer(
            text=qa_answer.text,
            found_in_context=qa_answer.grounded,
            confidence=qa_answer.confidence,
            chunks_used=chunks_used,
        )

    async def _prepare_qa_for_video(self, video_id: str) -> None:
        """
        Prepare Q&A for a video when vector DB has no chunks (e.g. after bot restart).
        Fetches transcript, chunks it, and stores in vector DB.
        """
        if not self.youtube_service:
            return
        try:
            transcript = await self.youtube_service.get_transcript(video_id)
            metadata: Dict[str, Any] = {}
            try:
                meta = await self.youtube_service.get_video_metadata(video_id)
                metadata = {"title": meta.get("title"), "language": getattr(transcript, "language", None)}
            except Exception:
                metadata = {"title": None, "language": getattr(transcript, "language", None)}
            chunks = await self.qa_service.chunk_transcript(
                transcript=transcript,
                chunk_size=500,
                overlap=50,
            )
            await self.vector_db.store_transcript_chunks(
                video_id=video_id,
                chunks=chunks,
                metadata={
                    **metadata,
                    "processed_at": datetime.now().isoformat(),
                },
            )
            if self.logger:
                self.logger.info(
                    "question_handler.qa_prepared_on_demand",
                    video_id=video_id,
                    chunks=len(chunks),
                )
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    "question_handler.qa_prepare_failed",
                    video_id=video_id,
                    error=str(e),
                )

    async def _rerank_chunks(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Re-rank chunks using cross-encoder for better relevance.
        
        Args:
            question: User's question
            chunks: Retrieved chunks with embeddings
            
        Returns:
            Re-ranked chunks with relevance scores
        """
        # This would use a cross-encoder model for better relevance
        # For now, use a simple scoring mechanism
        
        # Simple keyword matching as fallback
        question_words = set(question.lower().split())
        
        for chunk in chunks:
            chunk_text = chunk.get("text", "").lower()
            
            # Calculate keyword overlap
            chunk_words = set(chunk_text.split())
            overlap = len(question_words & chunk_words)
            
            # Normalize score
            max_possible = max(len(question_words), 1)
            keyword_score = overlap / max_possible
            
            # Combine with vector similarity (if available)
            vector_score = chunk.get("similarity", 0.5)
            
            # Weighted combination
            chunk["relevance_score"] = (0.3 * keyword_score) + (0.7 * vector_score)
        
        # Sort by relevance score
        return sorted(chunks, key=lambda x: x["relevance_score"], reverse=True)
    
    async def _validate_grounding(
        self,
        answer: str,
        context_chunks: List[Dict[str, Any]],
    ) -> bool:
        """
        Validate that answer is grounded in context chunks.
        
        Args:
            answer: Generated answer
            context_chunks: Chunks used as context
            
        Returns:
            True if answer appears to be grounded in context
        """
        # Combine all context text
        context_text = " ".join([c["text"] for c in context_chunks]).lower()
        answer_lower = answer.lower()
        
        # Extract key phrases from answer (simple version)
        words = answer_lower.split()
        key_phrases = []
        
        # Look for 3+ word sequences
        for i in range(len(words) - 2):
            phrase = " ".join(words[i:i+3])
            if len(phrase) > 10:  # Minimum length to check
                key_phrases.append(phrase)
        
        # Check if key phrases appear in context
        if key_phrases:
            matches = sum(1 for phrase in key_phrases if phrase in context_text)
            match_ratio = matches / len(key_phrases)
            
            # Require at least 30% of key phrases to match
            return match_ratio >= 0.3
        
        # If no key phrases, check if answer is very short
        if len(answer) < 50:
            # Short answers might be like "Yes" or "No"
            # Check if answer words appear in context
            answer_words = set(answer_lower.split())
            context_words = set(context_text.split())
            
            if answer_words:
                overlap = len(answer_words & context_words)
                return overlap >= len(answer_words) * 0.5
        
        return True
    
    # ------------------------------------------------------------------------
    # Conversation History
    # ------------------------------------------------------------------------
    
    async def _get_conversation_history(
        self,
        session: UserSession,
    ) -> List[Dict[str, str]]:
        """
        Get recent conversation history for context.
        
        Args:
            session: User session
            
        Returns:
            List of conversation turns
        """
        if not session.metadata:
            return []
        
        history = session.metadata.get("conversation_history", [])
        return history[-self.max_history_turns:]
    
    async def _update_conversation_history(
        self,
        session: UserSession,
        question: str,
        answer: Answer,
    ) -> None:
        """
        Update conversation history with new turn.
        
        Args:
            session: User session
            question: User's question
            answer: Bot's answer
        """
        if not session.metadata:
            session.metadata = {}
        
        history = session.metadata.get("conversation_history", [])
        
        # Add new turn
        history.append({
            "question": question,
            "answer": answer.text,
            "found": answer.found_in_context,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Keep only last N turns
        if len(history) > self.max_history_turns:
            history = history[-self.max_history_turns:]
        
        session.metadata["conversation_history"] = history
        session.modified = True
        await self.session_store.save_session(session)
    
    # ------------------------------------------------------------------------
    # Response Handling
    # ------------------------------------------------------------------------
    
    async def _send_no_video_response(self, context: UpdateContext) -> None:
        """
        Send response when user has no active video.
        
        Args:
            context: Update context
        """
        language = context.metadata.get("language", Language.ENGLISH)
        
        if language == Language.ENGLISH:
            text = (
                "❓ You haven't watched any video yet!\n\n"
                "Send me a YouTube link to get started."
            )
        elif language == Language.HINDI:
            text = (
                "❓ आपने अभी तक कोई वीडियो नहीं देखा है!\n\n"
                "शुरू करने के लिए मुझे एक YouTube लिंक भेजें।"
            )
        elif language == Language.TAMIL:
            text = (
                "❓ நீங்கள் இன்னும் எந்த வீடியோவையும் பார்க்கவில்லை!\n\n"
                "தொடங்க எனக்கு YouTube இணைப்பை அனுப்புங்கள்."
            )
        else:
            text = "Send me a YouTube link first!"
        
        await self._send_response(
            chat_id=context.chat_id,
            text=text,
            reply_to_message_id=context.update.effective_message.message_id,
        )
    
    async def _send_thinking_message(
        self,
        chat_id: int,
        language: Language,
    ) -> Message:
        """
        Send thinking indicator message.
        
        Args:
            chat_id: Target chat ID
            language: User's language
            
        Returns:
            Sent message
        """
        if language == Language.ENGLISH:
            text = "🤔 Thinking..."
        elif language == Language.HINDI:
            text = "🤔 सोच रहा हूँ..."
        elif language == Language.TAMIL:
            text = "🤔 யோசிக்கிறேன்..."
        else:
            text = "🤔 Thinking..."
        
        return await self.telegram_client.send_message(
            chat_id=chat_id,
            text=text,
        )
    
    async def _send_answer(
        self,
        chat_id: int,
        answer: Answer,
        language: Language,
        reply_to_message_id: Optional[int] = None,
    ) -> None:
        """
        Send answer with confidence indicator.
        
        Args:
            chat_id: Target chat ID
            answer: Answer object
            language: User's language
            reply_to_message_id: Message to reply to
        """
        # Add confidence indicator for low confidence answers
        if answer.found_in_context and answer.confidence < 0.7:
            if language == Language.ENGLISH:
                confidence_text = "\n\n_⚠️ Low confidence answer_"
            elif language == Language.HINDI:
                confidence_text = "\n\n_⚠️ कम विश्वास वाला उत्तर_"
            else:
                confidence_text = "\n\n_⚠️ Low confidence answer_"
            
            answer.text += confidence_text
        
        # Add source indicator
        if answer.found_in_context and answer.chunks_used:
            if language == Language.ENGLISH:
                source_text = f"\n\n📚 Based on {len(answer.chunks_used)} sections of the video"
            elif language == Language.HINDI:
                source_text = f"\n\n📚 वीडियो के {len(answer.chunks_used)} भागों पर आधारित"
            else:
                source_text = f"\n\n📚 Based on {len(answer.chunks_used)} sections"
            
            answer.text += source_text
        
        await self._send_response(
            chat_id=chat_id,
            text=answer.text,
            reply_to_message_id=reply_to_message_id,
        )
    
    def _get_not_found_message(self, language: Language) -> str:
        """
        Get "not found" message in appropriate language.
        
        Args:
            language: User's language
            
        Returns:
            Not found message
        """
        return self.not_found_responses.get(
            language,
            self.not_found_responses[Language.ENGLISH],
        )
    
    # ------------------------------------------------------------------------
    # Error Handling
    # ------------------------------------------------------------------------
    
    async def _handle_validation_error(
        self,
        context: UpdateContext,
        error: ValidationError,
    ) -> None:
        """Handle validation errors"""
        await self._send_error_message(
            context.chat_id,
            f"❌ {str(error)}",
        )
        self.metrics.increment("errors.validation")
    
    async def _handle_not_found_error(
        self,
        context: UpdateContext,
        error: NotFoundError,
    ) -> None:
        """Handle not found errors"""
        await self._send_error_message(
            context.chat_id,
            "❌ Video information not found. Please send the YouTube link again.",
        )
        self.metrics.increment("errors.not_found")
    
    # ------------------------------------------------------------------------
    # Callback Handlers
    # ------------------------------------------------------------------------
    
    async def handle_callback(self, context: UpdateContext) -> None:
        """
        Handle callback queries from follow-up keyboard.
        
        Args:
            context: Update context with callback query
        """
        callback_query = context.update.callback_query
        data = callback_query.data
        
        if data == "ask":
            await callback_query.answer()
            await callback_query.edit_message_text(
                text="💬 Send me your question about this video!",
            )
        
        elif data == "deepdive":
            # Trigger deep dive mode (more detailed analysis)
            await callback_query.answer(
                text="Deep dive mode activated! Ask detailed questions.",
                show_alert=False,
            )


# ------------------------------------------------------------------------
# Supporting Models
# ------------------------------------------------------------------------

class Answer:
    """Answer model for Q&A responses"""
    
    def __init__(
        self,
        text: str,
        found_in_context: bool = True,
        confidence: float = 1.0,
        chunks_used: List[str] = None,
        metadata: Dict[str, Any] = None,
    ):
        self.text = text
        self.found_in_context = found_in_context
        self.confidence = confidence
        self.chunks_used = chunks_used or []
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "found_in_context": self.found_in_context,
            "confidence": self.confidence,
            "chunks_used": self.chunks_used,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }