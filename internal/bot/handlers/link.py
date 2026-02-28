"""
YouTube Link Handler
Processes YouTube links and orchestrates the summarization workflow
"""

import re
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import asyncio

from telegram import Message
import structlog

from internal.bot.handlers.base import BaseHandler
from internal.bot.dispatcher import UpdateContext
from internal.bot.keyboard import KeyboardBuilder
from internal.services.youtube import YouTubeService, Transcript, TranscriptError
from internal.services.summarizer import SummarizerService, Summary, SummaryError
from internal.services.qa import QAService
from internal.services.language import LanguageService
from internal.storage.session import SessionStore, UserSession
from internal.storage.vector import VectorDBClient
from internal.domain.value_objects import Language, VideoId
from internal.pkg.errors import ValidationError, ErrorKind, NotFoundError


def _translate_segment_sync(text: str, target_code: str) -> str:
    """Translate one segment with free Google (sync). Returns original text on failure."""
    if not text or not text.strip() or target_code == "en":
        return text or ""
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="en", target=target_code).translate(text[:5000]) or text
    except Exception:
        return text


class ProcessingState(Enum):
    """Processing states for progress updates"""
    VALIDATING = "validating"
    FETCHING = "fetching"
    PROCESSING = "processing"
    SUMMARIZING = "summarizing"
    COMPLETE = "complete"
    ERROR = "error"


class LinkHandler(BaseHandler):
    """
    Handles YouTube links with full orchestration of the summarization pipeline.
    
    Workflow:
    1. Extract and validate video ID
    2. Send progress updates to user
    3. Fetch transcript (with retry logic)
    4. Store chunks in vector DB for Q&A
    5. Generate structured summary
    6. Send formatted response
    7. Update session with current video
    8. Show follow-up action keyboard
    
    Features:
    - Progress updates during long operations
    - Automatic retry on transient failures
    - Caching to avoid duplicate processing
    - Session management for context
    - Multilingual summaries
    """
    
    def __init__(
        self,
        telegram_client,
        youtube_service: YouTubeService,
        summarizer_service: SummarizerService,
        qa_service: QAService,
        language_service: LanguageService,
        session_store: SessionStore,
        vector_db: VectorDBClient,
        logger=None,
        metrics=None,
    ):
        """
        Initialize link handler with all required services.
        
        Args:
            telegram_client: Client for sending messages
            youtube_service: Service for fetching transcripts
            summarizer_service: Service for generating summaries
            qa_service: Service for Q&A preparation
            language_service: Service for multilingual support
            session_store: Session manager for user data
            vector_db: Vector database for transcript storage
            logger: Structured logger
            metrics: Metrics collector
        """
        super().__init__(telegram_client, logger, metrics)
        
        self.youtube_service = youtube_service
        self.summarizer_service = summarizer_service
        self.qa_service = qa_service
        self.language_service = language_service
        self.session_store = session_store
        self.vector_db = vector_db
        
        # Configuration
        self.max_video_duration_minutes = 180  # 3 hours max
        self.enable_caching = True
        self.progress_updates = True
        
        # YouTube URL patterns
        self.url_patterns = [
            re.compile(r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})"),
            re.compile(r"^([a-zA-Z0-9_-]{11})$"),  # Just the ID
        ]
        
        self.logger.info(
            "link_handler.initialized",
            max_duration=self.max_video_duration_minutes,
            caching=self.enable_caching,
        )

    @property
    def handler_name(self) -> str:
        return "youtube_link"
    
    async def handle(self, context: UpdateContext) -> None:
        """
        Main entry point for YouTube link processing.
        
        Args:
            context: Update context with message containing YouTube link
        """
        start_time = datetime.now()
        progress_message: Optional[Message] = None
        
        try:
            await self.before_handle(context)

            # If this is a callback (e.g. Ask Question, New Summary from follow-up keyboard)
            if getattr(context.update, "callback_query", None):
                await self.handle_callback(context)
                return

            # Validate required fields
            await self._validate_required_fields(
                context,
                ["chat_id", "user_id", "message_text"],
            )

            # Load user session
            session = await self.session_store.get_session(context.user_id)
            if not session:
                session = await self.session_store.create_session(
                    user_id=context.user_id,
                    chat_id=context.chat_id,
                )
            
            # Add language to context
            context.metadata["language"] = session.language
            
            # Extract video ID from message
            video_id = await self._extract_video_id(context.message_text)
            if not video_id:
                await self._send_error_message(
                    context.chat_id,
                    "❌ Invalid YouTube link. Please check and try again.\n\n"
                    "Supported formats:\n"
                    "• youtube.com/watch?v=XXXXX\n"
                    "• youtu.be/XXXXX\n"
                    "• youtube.com/shorts/XXXXX",
                )
                return
            
            # Check if video was already processed (caching)
            if self.enable_caching:
                cached_summary = await self._get_cached_summary(video_id, session.language)
                if cached_summary:
                    await self._send_cached_response(context, session, video_id, cached_summary)
                    return
            
            # Send initial progress message
            if self.progress_updates:
                progress_message = await self._send_progress(
                    context.chat_id,
                    ProcessingState.VALIDATING,
                    "🔍 Checking YouTube link...",
                )
            
            # Step 1: Validate video and get metadata
            video_metadata = await self._validate_video(video_id)
            
            # Step 2: Update progress - fetching transcript
            await self._update_progress(
                progress_message,
                ProcessingState.FETCHING,
                f"📥 Downloading transcript for: {video_metadata['title'][:50]}...",
            )
            
            # Step 3: Fetch transcript
            transcript = await self._fetch_transcript_with_retry(video_id, session.language)
            
            # Step 4: Update progress - processing
            await self._update_progress(
                progress_message,
                ProcessingState.PROCESSING,
                "🤖 Analyzing video content...",
            )
            
            # Step 5: Store in vector DB for Q&A (background task)
            asyncio.create_task(
                self._prepare_for_qa(video_id, transcript, video_metadata)
            )
            
            # Step 6: Update progress - summarizing
            await self._update_progress(
                progress_message,
                ProcessingState.SUMMARIZING,
                "📝 Generating structured summary...",
            )
            
            # Step 7: Generate summary (in English, then translate if needed)
            summary = await self._generate_summary(
                transcript=transcript,
                video_metadata=video_metadata,
                language=session.language,
            )
            # Guarantee translation when user chose another language (direct free Google)
            if session.language != Language.ENGLISH:
                summary = await self._translate_summary_direct(summary, session.language)

            # Step 8: Update progress - complete
            await self._update_progress(
                progress_message,
                ProcessingState.COMPLETE,
                "✅ Summary ready!",
                delete_after=True,
            )
            
            # Step 9: Send formatted summary
            await self._send_summary_response(
                context=context,
                session=session,
                video_id=video_id,
                video_metadata=video_metadata,
                summary=summary,
            )
            
            # Step 10: Update session with current video and last summary (for /summary, /actionpoints)
            session.set_current_video(video_id, video_metadata.get("title"))
            if not session.metadata:
                session.metadata = {}
            session.metadata["last_summary"] = {
                "key_points": list(summary.key_points) if summary.key_points else [],
                "core_takeaway": (summary.core_takeaway or "").strip(),
                "action_items": list(getattr(summary, "action_items", None) or []),
                "title": (video_metadata.get("title") or "Video summary").strip(),
            }
            session.modified = True
            await self.session_store.save_session(session)
            
            # Track metrics
            self._track_user_activity(context.user_id, "video_processed")
            self.metrics.increment(
                "videos.processed",
                tags={
                    "language": session.language.code,
                    "duration_minutes": video_metadata.get("duration", 0) // 60,
                },
            )
            
            self.logger.info(
                "link_handler.video_processed",
                video_id=video_id,
                user_id=context.user_id,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                transcript_length=len(transcript.text),
            )
            
            await self.after_handle(context)
            
        except ValidationError as e:
            await self._handle_validation_error(context, e)
        except NotFoundError as e:
            await self._handle_not_found_error(context, e)
        except TranscriptError as e:
            await self._handle_transcript_error(context, e)
        except Exception as e:
            await self.on_error(context, e)
            raise
        finally:
            self._track_processing_time(start_time)
    
    # ------------------------------------------------------------------------
    # YouTube Link Processing
    # ------------------------------------------------------------------------
    
    async def _extract_video_id(self, text: str) -> Optional[str]:
        """
        Extract YouTube video ID from message text.
        
        Args:
            text: Message text containing URL
            
        Returns:
            Video ID or None if not found
        """
        text = text.strip()
        
        for pattern in self.url_patterns:
            match = pattern.search(text)
            if match:
                return match.group(1)
        
        return None
    
    async def _validate_video(self, video_id: str) -> Dict[str, Any]:
        """
        Validate video and get metadata.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Video metadata dictionary
            
        Raises:
            ValidationError: If video is invalid or too long
            NotFoundError: If video doesn't exist
        """
        try:
            metadata = await self.youtube_service.get_video_metadata(video_id)
            
            # Check duration
            duration_minutes = metadata.get("duration", 0) / 60
            if duration_minutes > self.max_video_duration_minutes:
                raise ValidationError(
                    f"Video too long ({duration_minutes:.0f} minutes). "
                    f"Maximum supported: {self.max_video_duration_minutes} minutes.",
                    kind=ErrorKind.VALIDATION,
                )
            
            self.logger.debug(
                "link_handler.video_validated",
                video_id=video_id,
                title=metadata.get("title"),
                duration_minutes=duration_minutes,
            )
            
            return metadata
            
        except Exception as e:
            self.logger.warning(
                "link_handler.video_validation_failed",
                video_id=video_id,
                error=str(e),
            )
            raise NotFoundError(
                f"Video not found or inaccessible: {video_id}",
                kind=ErrorKind.NOT_FOUND,
                original_error=e,
            ) from e
    
    async def _fetch_transcript_with_retry(
        self,
        video_id: str,
        language: Language,
    ) -> Transcript:
        """
        Fetch transcript with automatic retry.
        
        Args:
            video_id: YouTube video ID
            language: Preferred language
            
        Returns:
            Transcript object
            
        Raises:
            TranscriptError: If transcript unavailable
        """
        try:
            # Try with retry (handles transient failures)
            transcript = await self._with_retry(
                self.youtube_service.get_transcript,
                video_id,
                language_codes=[language.code, "en"],  # Fallback to English
            )
            
            self.logger.debug(
                "link_handler.transcript_fetched",
                video_id=video_id,
                language=transcript.language,
                segments=len(transcript.segments),
            )
            
            return transcript
            
        except Exception as e:
            self.logger.warning(
                "link_handler.transcript_failed",
                video_id=video_id,
                error=str(e),
            )
            raise TranscriptError(
                f"No transcript available for this video. The video might have:\n"
                f"• Auto-generated captions disabled\n"
                f"• No captions at all\n"
                f"• Non-English audio without captions",
                kind=ErrorKind.NOT_FOUND,
                original_error=e,
            ) from e
    
    async def _prepare_for_qa(
        self,
        video_id: str,
        transcript: Transcript,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Prepare video for Q&A by chunking and storing in vector DB.
        Runs in background to avoid blocking the response.
        
        Args:
            video_id: YouTube video ID
            transcript: Full transcript
            metadata: Video metadata
        """
        try:
            # Chunk the transcript (async call)
            chunks = await self.qa_service.chunk_transcript(
                transcript=transcript,
                chunk_size=500,  # characters
                overlap=50,
            )
            
            # Generate embeddings and store
            await self.vector_db.store_transcript_chunks(
                video_id=video_id,
                chunks=chunks,
                metadata={
                    "title": metadata.get("title"),
                    "language": transcript.language,
                    "processed_at": datetime.now().isoformat(),
                },
            )
            
            self.logger.info(
                "link_handler.qa_prepared",
                video_id=video_id,
                chunks=len(chunks),
            )
            
        except Exception as e:
            # Non-critical error - log but don't fail
            self.logger.warning(
                "link_handler.qa_preparation_failed",
                video_id=video_id,
                error=str(e),
            )
    
    async def _translate_summary_direct(self, summary: Summary, target_language: Language) -> Summary:
        """Translate summary using free Google Translate only. Always runs when language != EN."""
        if target_language == Language.ENGLISH:
            return summary
        tgt = target_language.code
        loop = asyncio.get_event_loop()

        def tr(s: str) -> str:
            return _translate_segment_sync(s, tgt)

        # Translate key points
        kp = list(summary.key_points) if summary.key_points else []
        for i, pt in enumerate(kp):
            if isinstance(pt, dict) and pt.get("point"):
                kp[i] = {**pt, "point": await loop.run_in_executor(None, lambda p=pt["point"]: tr(p))}
        # Translate core takeaway
        core = (summary.core_takeaway or "").strip()
        if core:
            core = await loop.run_in_executor(None, lambda: tr(core))
        # Translate action items
        actions = list(getattr(summary, "action_items", None) or [])
        for i, item in enumerate(actions):
            if isinstance(item, str):
                actions[i] = await loop.run_in_executor(None, lambda a=item: tr(a))

        return Summary(
            video_id=getattr(summary, "video_id", ""),
            title=getattr(summary, "title", ""),
            key_points=kp,
            core_takeaway=core,
            action_items=actions,
            summary_type=getattr(summary, "summary_type", None),
            language=target_language,
            chunk_count=getattr(summary, "chunk_count", 1),
            processing_time=getattr(summary, "processing_time", 0.0),
            metadata=getattr(summary, "metadata", {}),
        )

    async def _generate_summary(
        self,
        transcript: Transcript,
        video_metadata: Dict[str, Any],
        language: Language,
    ) -> Summary:
        """
        Generate structured summary from transcript.
        
        Args:
            transcript: Video transcript
            video_metadata: Video metadata
            language: Target language for summary
            
        Returns:
            Structured summary
        """
        # Generate summary in English first (better quality)
        summary = await self.summarizer_service.generate_summary(
            transcript=transcript,
            metadata=video_metadata,
        )
        
        # Translate if needed
        if language != Language.ENGLISH:
            summary = await self.language_service.translate_summary(
                summary=summary,
                target_language=language,
            )
        
        return summary
    
    async def _get_cached_summary(
        self,
        video_id: str,
        language: Language,
    ) -> Optional[Summary]:
        """
        Check if video was already processed and cached.
        
        Args:
            video_id: YouTube video ID
            language: Language preference
            
        Returns:
            Cached summary or None
        """
        # This would check Redis or similar cache
        # For now, return None (no caching)
        return None
    
    # ------------------------------------------------------------------------
    # Response Handling
    # ------------------------------------------------------------------------
    
    async def _send_progress(
        self,
        chat_id: int,
        state: ProcessingState,
        text: str,
    ) -> Message:
        """
        Send initial progress message.
        
        Args:
            chat_id: Target chat ID
            state: Processing state
            text: Progress text
            
        Returns:
            Sent message for later updates
        """
        return await self.telegram_client.send_message(
            chat_id=chat_id,
            text=text,
        )
    
    async def _update_progress(
        self,
        message: Optional[Message],
        state: ProcessingState,
        text: str,
        delete_after: bool = False,
    ) -> None:
        """
        Update progress message.
        
        Args:
            message: Message to update
            state: New processing state
            text: New progress text
            delete_after: Delete message after update
        """
        if not message or not self.progress_updates:
            return
        
        try:
            if delete_after:
                await message.delete()
            else:
                await message.edit_text(text=text)
        except Exception as e:
            # Non-critical error
            self.logger.debug(
                "link_handler.progress_update_failed",
                error=str(e),
            )
    
    async def _send_cached_response(
        self,
        context: UpdateContext,
        session: UserSession,
        video_id: str,
        summary: Summary,
    ) -> None:
        """
        Send cached summary response.
        
        Args:
            context: Update context
            session: User session
            video_id: Video ID
            summary: Cached summary
        """
        # Send cached summary
        await self._send_summary_response(
            context=context,
            session=session,
            video_id=video_id,
            video_metadata=summary.metadata,
            summary=summary,
            cached=True,
        )
        
        # Update session
        session.current_video_id = video_id
        session.modified = True
        await self.session_store.save_session(session)
        
        self.metrics.increment("videos.cached_hit")
    
    async def _send_summary_response(
        self,
        context: UpdateContext,
        session: UserSession,
        video_id: str,
        video_metadata: Dict[str, Any],
        summary: Summary,
        cached: bool = False,
    ) -> None:
        """
        Send formatted summary with follow-up options.
        
        Args:
            context: Update context
            session: User session
            video_id: Video ID
            video_metadata: Video metadata
            summary: Generated summary
            cached: Whether this is from cache
        """
        # Build summary text
        summary_text = self._format_summary_text(
            metadata=video_metadata,
            summary=summary,
            language=session.language,
            cached=cached,
        )
        
        # Send summary with follow-up keyboard (Q&A, New Summary, etc.)
        reply_markup = KeyboardBuilder.build(
            self._build_followup_keyboard(session.language),
            "inline",
        )
        await self.telegram_client.send_message(
            chat_id=context.chat_id,
            text=summary_text,
            parse_mode="Markdown",
            reply_markup=reply_markup,
        )
    
    def _format_summary_text(
        self,
        metadata: Dict[str, Any],
        summary: Summary,
        language: Language,
        cached: bool,
    ) -> str:
        """
        Format summary per spec: structured (not paragraph dump), key points,
        timestamps when available, core insight; concise but meaningful.
        Ensures title, key points, and core insight are never empty in the message.
        """
        lines = []
        
        # Header: never show "Unknown"
        title = (metadata.get("title") or "").strip()
        if not title or title.lower() == "unknown":
            title = "Video summary"
        lines.append(f"📹 *{title}*")
        
        if cached:
            if language == Language.ENGLISH:
                lines.append("_⚡ Retrieved from cache (processed earlier)_\n")
            elif language == Language.HINDI:
                lines.append("_⚡ कैश से प्राप्त (पहले प्रोसेस किया गया)_\n")
        
        # Duration
        duration_min = metadata.get("duration", 0) // 60
        if duration_min > 0:
            lines.append(f"⏱️ Duration: {duration_min} minutes\n")
        
        # Key points (structured, with timestamps when available)
        lines.append("🔑 *Key Points:*")
        key_points = list(summary.key_points) if summary.key_points else []
        if not key_points and summary.core_takeaway:
            key_points = [{"point": summary.core_takeaway[:400] + ("..." if len(summary.core_takeaway) > 400 else ""), "timestamp": ""}]
        if not key_points:
            key_points = [{"point": "Summary could not be extracted for this video.", "timestamp": ""}]
        for i, point in enumerate(key_points, 1):
            pt = point.get("point", "").strip() or "(No description)"
            timestamp = point.get("timestamp", "") or ""
            if timestamp:
                lines.append(f"{i}. `{timestamp}` - {pt}")
            else:
                lines.append(f"{i}. {pt}")
        
        # Core insight (concise but meaningful)
        lines.append("\n💡 *Core insight:*")
        core = (summary.core_takeaway or "").strip()
        if not core:
            core = "See key points above."
        lines.append(core)
        
        return "\n".join(lines)
    
    def _build_followup_keyboard(self, language: Language) -> List[List[Dict[str, str]]]:
        """
        Build follow-up action keyboard.
        
        Args:
            language: User's language
            
        Returns:
            Keyboard configuration
        """
        if language == Language.ENGLISH:
            return [
                [
                    {"text": "❓ Ask Question", "callback_data": "ask"},
                    {"text": "📝 New Summary", "callback_data": "new"},
                ],
                [
                    {"text": "🔍 Deep Dive", "callback_data": "deepdive"},
                    {"text": "🌐 Change Language", "callback_data": "language"},
                ],
            ]
        elif language == Language.HINDI:
            return [
                [
                    {"text": "❓ सवाल पूछें", "callback_data": "ask"},
                    {"text": "📝 नया सारांश", "callback_data": "new"},
                ],
                [
                    {"text": "🔍 गहराई से", "callback_data": "deepdive"},
                    {"text": "🌐 भाषा बदलें", "callback_data": "language"},
                ],
            ]
        else:
            # Default English
            return [
                [
                    {"text": "❓ Ask Question", "callback_data": "ask"},
                    {"text": "📝 New Summary", "callback_data": "new"},
                ],
            ]
    
    # ------------------------------------------------------------------------
    # Error Handling
    # ------------------------------------------------------------------------
    
    async def _handle_validation_error(
        self,
        context: UpdateContext,
        error: ValidationError,
    ) -> None:
        """Handle validation errors with user-friendly messages"""
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
            "❌ Video not found. Please check the link and try again.",
        )
        self.metrics.increment("errors.not_found")
    
    async def _handle_transcript_error(
        self,
        context: UpdateContext,
        error: Exception,
    ) -> None:
        """Handle transcript errors with helpful suggestions"""
        await self._send_error_message(
            context.chat_id,
            "📝 This video doesn't have a transcript available.\n\n"
            "Reasons:\n"
            "• Auto-generated captions are disabled\n"
            "• Video is in a language without captions\n"
            "• Video is too new (captions not generated yet)\n\n"
            "Try another video!",
        )
        self.metrics.increment("errors.transcript")
    
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
        
        elif data == "new":
            await callback_query.answer()
            await callback_query.edit_message_text(
                text="📹 Send another YouTube link to summarize!",
            )
        
        elif data == "deepdive":
            await callback_query.answer(
                text="Deep dive mode coming soon!",
                show_alert=True,
            )
        
        elif data == "language":
            await callback_query.answer()
            await self.telegram_client.send_message(
                chat_id=callback_query.message.chat_id,
                text="🌐 Use /language to choose your language for summaries and answers.",
            )