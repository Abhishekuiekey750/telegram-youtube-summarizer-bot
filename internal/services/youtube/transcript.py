"""
YouTube Transcript Fetcher
Uses youtube-transcript-api with robust error handling and processing

Features:
- Multi-language support with fallback
- Automatic retry with exponential backoff
- Transcript cleaning and normalization
- Segment merging/splitting for optimal chunking
- Caching support
- Detailed error messages
- Long transcript handling
"""

import asyncio
import re
from typing import List, Optional, Dict, Any, Tuple
from datetime import timedelta
from dataclasses import dataclass, field
import html
import json

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    NotTranslatable,
    TranslationLanguageNotAvailable,
)
try:
    from youtube_transcript_api import TooManyRequests
except ImportError:
    from youtube_transcript_api._errors import RequestBlocked as TooManyRequests
import structlog

from internal.pkg.errors import (
    NotFoundError,
    ValidationError,
    RetryableError,
    ErrorKind,
)
from internal.pkg.retry import RetryConfig
from internal.services.youtube.validator import YouTubeURLValidator
from internal.domain.value_objects import Language


@dataclass
class TranscriptSegment:
    """Single transcript segment with timestamp"""
    text: str
    start: float  # Start time in seconds
    duration: float  # Duration in seconds
    language: str = "en"
    
    def __post_init__(self):
        """Clean text after initialization"""
        self.text = self.text.strip()
    
    @property
    def end(self) -> float:
        """Get end time"""
        return self.start + self.duration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "start": self.start,
            "duration": self.duration,
            "language": self.language,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptSegment":
        """Create from dictionary"""
        return cls(
            text=data["text"],
            start=data["start"],
            duration=data["duration"],
            language=data.get("language", "en"),
        )


@dataclass
class Transcript:
    """Complete transcript for a video"""
    video_id: str
    segments: List[TranscriptSegment]
    language: str
    is_generated: bool = False  # Auto-generated captions
    video_title: Optional[str] = None
    video_duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def text(self) -> str:
        """Get full text by joining all segments"""
        return " ".join(segment.text for segment in self.segments)
    
    @property
    def length(self) -> int:
        """Get total character count"""
        return len(self.text)
    
    @property
    def segment_count(self) -> int:
        """Get number of segments"""
        return len(self.segments)
    
    @property
    def duration(self) -> float:
        """Get total duration covered by transcript"""
        if not self.segments:
            return 0.0
        return self.segments[-1].end - self.segments[0].start
    
    def get_segments_in_range(
        self,
        start_time: float,
        end_time: float
    ) -> List[TranscriptSegment]:
        """Get segments within time range"""
        return [
            segment for segment in self.segments
            if segment.start >= start_time and segment.end <= end_time
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "video_id": self.video_id,
            "language": self.language,
            "is_generated": self.is_generated,
            "video_title": self.video_title,
            "video_duration": self.video_duration,
            "segments": [s.to_dict() for s in self.segments],
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transcript":
        """Create from dictionary"""
        segments = [
            TranscriptSegment.from_dict(s) for s in data["segments"]
        ]
        return cls(
            video_id=data["video_id"],
            segments=segments,
            language=data["language"],
            is_generated=data.get("is_generated", False),
            video_title=data.get("video_title"),
            video_duration=data.get("video_duration"),
            metadata=data.get("metadata", {}),
        )


class TranscriptFetcher:
    """
    Fetches and processes YouTube transcripts with robust error handling
    
    Features:
    - Multiple language fallback
    - Automatic retry with backoff
    - Transcript cleaning and normalization
    - Segment optimization
    - Caching support
    - Detailed error messages
    """
    
    # Default languages to try in order
    DEFAULT_LANGUAGE_PRIORITY = [
        "hi", "ta", "te", "kn", "ml", "bn",  # Indian languages
        "en",  # English
        "es", "fr", "de",  # Other common languages
    ]
    
    def __init__(
        self,
        validator: Optional[YouTubeURLValidator] = None,
        cache=None,  # Redis cache client
        retry_config: Optional[RetryConfig] = None,
        logger=None,
        metrics=None,
        max_segment_duration: float = 30.0,  # Max seconds per segment
        min_segment_duration: float = 1.0,   # Min seconds to keep
        max_transcript_length: int = 100000,  # Max characters
    ):
        """
        Initialize transcript fetcher
        
        Args:
            validator: YouTube URL validator
            cache: Cache client (Redis or memory)
            retry_config: Retry configuration
            logger: Structured logger
            metrics: Metrics collector
            max_segment_duration: Maximum segment duration before splitting
            min_segment_duration: Minimum segment duration before merging
            max_transcript_length: Maximum transcript length to process
        """
        self.validator = validator or YouTubeURLValidator()
        self.cache = cache
        self.retry_config = retry_config or RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=10.0,
            exponential_base=2,
        )
        self.logger = logger or structlog.get_logger(__name__)
        self.metrics = metrics
        self.max_segment_duration = max_segment_duration
        self.min_segment_duration = min_segment_duration
        self.max_transcript_length = max_transcript_length
        
        # API instance
        self.api = YouTubeTranscriptApi()
        
        self.logger.info(
            "transcript_fetcher.initialized",
            max_retries=self.retry_config.max_retries,
            max_segment_duration=self.max_segment_duration,
        )
    
    async def fetch(
        self,
        video_id: str,
        preferred_languages: Optional[List[str]] = None,
        use_cache: bool = True,
        process: bool = True,
    ) -> Transcript:
        """
        Fetch transcript for a video
        
        Args:
            video_id: YouTube video ID
            preferred_languages: List of language codes in priority order
            use_cache: Whether to check cache first
            process: Whether to process/clean the transcript
            
        Returns:
            Transcript object
            
        Raises:
            NotFoundError: If no transcript available
            ValidationError: If video ID invalid
            RetryableError: For temporary failures
        """
        # Validate video ID
        if not self.validator._validate_video_id(video_id):
            raise ValidationError(
                f"Invalid video ID: {video_id}",
                kind=ErrorKind.VALIDATION,
                context={"video_id": video_id},
            )
        
        # Check cache
        if use_cache and self.cache:
            cached = await self._get_cached(video_id, preferred_languages)
            if cached:
                self.logger.debug(
                    "transcript_fetcher.cache_hit",
                    video_id=video_id,
                )
                if self.metrics:
                    self.metrics.increment("transcript.cache_hit")
                return cached
        
        # Fetch transcript with retry
        try:
            transcript = await self._fetch_with_retry(
                video_id=video_id,
                preferred_languages=preferred_languages or self.DEFAULT_LANGUAGE_PRIORITY,
            )
            
            # Process transcript if requested
            if process:
                transcript = self._process_transcript(transcript)
            
            # Cache result
            if use_cache and self.cache:
                await self._cache_transcript(transcript)
            
            if self.metrics:
                self.metrics.increment(
                    "transcript.fetched",
                    tags={
                        "language": transcript.language,
                        "generated": str(transcript.is_generated),
                        "segment_count": transcript.segment_count,
                    },
                )
            
            return transcript
            
        except Exception as e:
            self.logger.error(
                "transcript_fetcher.fetch_failed",
                video_id=video_id,
                error=str(e),
            )
            if self.metrics:
                self.metrics.increment("transcript.fetch_failed")
            raise
    
    async def _fetch_with_retry(
        self,
        video_id: str,
        preferred_languages: List[str],
    ) -> Transcript:
        """
        Fetch transcript with retry logic
        
        This runs in a thread pool since youtube-transcript-api is synchronous
        """
        loop = asyncio.get_event_loop()
        
        try:
            # Run API call in thread pool (youtube-transcript-api uses requests)
            transcript_list = await loop.run_in_executor(
                None,
                self.api.list,  # returns TranscriptList for this video
                video_id,
            )
            
            # Try to get transcript in preferred languages
            transcript, language, is_generated = await self._find_transcript(
                transcript_list,
                preferred_languages,
            )
            
            if not transcript:
                raise NotFoundError(
                    f"No transcript found for video {video_id}",
                    kind=ErrorKind.NOT_FOUND,
                    context={"video_id": video_id},
                )
            
            # Fetch the actual transcript data
            segments = await loop.run_in_executor(
                None,
                self._fetch_transcript_data,
                transcript,
            )
            
            # Convert to our segment format
            transcript_segments = []
            for segment in segments:
                transcript_segments.append(TranscriptSegment(
                    text=segment["text"],
                    start=segment["start"],
                    duration=segment["duration"],
                    language=language,
                ))
            
            return Transcript(
                video_id=video_id,
                segments=transcript_segments,
                language=language,
                is_generated=is_generated,
            )
            
        except (TranscriptsDisabled, VideoUnavailable) as e:
            raise NotFoundError(
                str(e),
                kind=ErrorKind.NOT_FOUND,
                original_error=e,
            ) from e
            
        except TooManyRequests as e:
            raise RetryableError(
                "Rate limited by YouTube",
                kind=ErrorKind.RATE_LIMIT,
                retry_after=60,
                original_error=e,
            ) from e
            
        except Exception as e:
            # Wrap other exceptions as retryable
            raise RetryableError(
                f"Transcript fetch failed: {str(e)}",
                kind=ErrorKind.EXTERNAL_API,
                original_error=e,
            ) from e
    
    async def _find_transcript(
        self,
        transcript_list,
        preferred_languages: List[str],
    ) -> Tuple[Optional[Any], str, bool]:
        """
        Find best matching transcript
        
        Strategy:
        1. Try manually created transcripts in preferred languages
        2. Try auto-generated transcripts in preferred languages
        3. Try any available transcript
        
        Args:
            transcript_list: YouTube transcript list object
            preferred_languages: Language priority list
            
        Returns:
            Tuple of (transcript, language, is_generated)
        """
        # Try manually created transcripts first
        for lang in preferred_languages:
            try:
                transcript = transcript_list.find_transcript([lang])
                return transcript, lang, False
            except NoTranscriptFound:
                continue
        
        # Try auto-generated transcripts
        for lang in preferred_languages:
            try:
                transcript = transcript_list.find_generated_transcript([lang])
                return transcript, lang, True
            except NoTranscriptFound:
                continue
        
        # Try any available transcript
        try:
            # Get first available transcript
            available = transcript_list._manually_created_transcripts or transcript_list._generated_transcripts
            if available:
                first_key = next(iter(available))
                transcript = available[first_key]
                return transcript, first_key, first_key in transcript_list._generated_transcripts
        except Exception:
            pass
        
        return None, "", False
    
    def _fetch_transcript_data(self, transcript) -> List[Dict[str, Any]]:
        """
        Fetch actual transcript data.
        
        In newer versions of youtube-transcript-api, ``transcript.fetch()`` returns a
        ``FetchedTranscript`` object (an iterable of ``FetchedTranscriptSnippet``),
        not a plain list of dicts. We normalize it here to a list of dicts so the
        rest of the pipeline can stay the same.
        """
        fetched = transcript.fetch()
        # FetchedTranscript has to_raw_data(); older versions may already return list[dict]
        if hasattr(fetched, "to_raw_data"):
            return fetched.to_raw_data()
        return fetched
    
    def _process_transcript(self, transcript: Transcript) -> Transcript:
        """
        Clean and optimize transcript
        
        Steps:
        1. Clean text (remove HTML, fix encoding)
        2. Merge very short segments
        3. Split very long segments
        4. Remove empty segments
        
        Args:
            transcript: Raw transcript
            
        Returns:
            Processed transcript
        """
        if not transcript.segments:
            return transcript
        
        # Clean each segment
        for segment in transcript.segments:
            segment.text = self._clean_text(segment.text)
        
        # Remove empty segments
        transcript.segments = [
            s for s in transcript.segments if s.text.strip()
        ]
        
        if not transcript.segments:
            return transcript
        
        # Merge short segments
        transcript.segments = self._merge_short_segments(
            transcript.segments,
            min_duration=self.min_segment_duration,
        )
        
        # Split long segments
        transcript.segments = self._split_long_segments(
            transcript.segments,
            max_duration=self.max_segment_duration,
        )
        
        # Check total length
        if transcript.length > self.max_transcript_length:
            self.logger.warning(
                "transcript_fetcher.transcript_truncated",
                video_id=transcript.video_id,
                length=transcript.length,
                max_length=self.max_transcript_length,
            )
            # Truncate segments to fit
            transcript = self._truncate_transcript(
                transcript,
                self.max_transcript_length,
            )
        
        return transcript
    
    def _clean_text(self, text: str) -> str:
        """
        Clean transcript text
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Unescape HTML
        text = html.unescape(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove music/ applause markers
        text = re.sub(r'\[[^\]]*\]', '', text)  # [Music], [Applause]
        text = re.sub(r'\([^)]*\)', '', text)   # (applause)
        
        # Fix common encoding issues
        text = text.replace('&#39;', "'")
        text = text.replace('&amp;', '&')
        text = text.replace('&quot;', '"')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _merge_short_segments(
        self,
        segments: List[TranscriptSegment],
        min_duration: float,
    ) -> List[TranscriptSegment]:
        """
        Merge very short segments with neighbors
        
        Args:
            segments: List of segments
            min_duration: Minimum duration to keep separate
            
        Returns:
            Merged segments
        """
        if len(segments) <= 1:
            return segments
        
        merged = []
        current = segments[0]
        
        for next_seg in segments[1:]:
            # If current segment is too short, merge with next
            if current.duration < min_duration:
                # Merge texts
                current.text = f"{current.text} {next_seg.text}"
                # Extend duration
                current.duration = next_seg.end - current.start
            else:
                # Current is fine, add to result
                merged.append(current)
                current = next_seg
        
        # Add last segment
        merged.append(current)
        
        return merged
    
    def _split_long_segments(
        self,
        segments: List[TranscriptSegment],
        max_duration: float,
    ) -> List[TranscriptSegment]:
        """
        Split very long segments into smaller ones
        
        Args:
            segments: List of segments
            max_duration: Maximum duration per segment
            
        Returns:
            Split segments
        """
        result = []
        
        for segment in segments:
            if segment.duration <= max_duration:
                result.append(segment)
                continue
            
            # Split long segment by sentences
            sentences = re.split(r'(?<=[.!?])\s+', segment.text)
            if len(sentences) <= 1:
                # Can't split by sentences, just keep as is
                result.append(segment)
                continue
            
            # Distribute text across time
            time_per_char = segment.duration / len(segment.text)
            current_time = segment.start
            current_text = []
            current_duration = 0
            
            for sentence in sentences:
                sentence_duration = len(sentence) * time_per_char
                
                if current_duration + sentence_duration > max_duration and current_text:
                    # Create new segment
                    result.append(TranscriptSegment(
                        text=" ".join(current_text),
                        start=current_time,
                        duration=current_duration,
                        language=segment.language,
                    ))
                    current_time += current_duration
                    current_text = [sentence]
                    current_duration = sentence_duration
                else:
                    current_text.append(sentence)
                    current_duration += sentence_duration
            
            # Add last segment
            if current_text:
                result.append(TranscriptSegment(
                    text=" ".join(current_text),
                    start=current_time,
                    duration=current_duration,
                    language=segment.language,
                ))
        
        return result
    
    def _truncate_transcript(
        self,
        transcript: Transcript,
        max_length: int,
    ) -> Transcript:
        """
        Truncate transcript to max length
        
        Args:
            transcript: Original transcript
            max_length: Maximum characters
            
        Returns:
            Truncated transcript
        """
        total_length = 0
        truncated_segments = []
        
        for segment in transcript.segments:
            segment_length = len(segment.text)
            if total_length + segment_length <= max_length:
                truncated_segments.append(segment)
                total_length += segment_length
            else:
                # Truncate this segment
                remaining = max_length - total_length
                if remaining > 0:
                    segment.text = segment.text[:remaining] + "..."
                    truncated_segments.append(segment)
                break
        
        transcript.segments = truncated_segments
        return transcript
    
    async def _get_cached(
        self,
        video_id: str,
        languages: Optional[List[str]],
    ) -> Optional[Transcript]:
        """Get transcript from cache"""
        if not self.cache:
            return None
        
        cache_key = f"transcript:{video_id}:{','.join(languages or [])}"
        cached = await self.cache.get(cache_key)
        
        if cached:
            try:
                data = json.loads(cached)
                return Transcript.from_dict(data)
            except Exception as e:
                self.logger.warning(
                    "transcript_fetcher.cache_corrupted",
                    video_id=video_id,
                    error=str(e),
                )
        
        return None
    
    async def _cache_transcript(self, transcript: Transcript) -> None:
        """Cache transcript for future use"""
        if not self.cache:
            return
        
        try:
            cache_key = f"transcript:{transcript.video_id}:{transcript.language}"
            data = json.dumps(transcript.to_dict())
            # Cache for 24 hours
            await self.cache.setex(cache_key, 86400, data)
        except Exception as e:
            self.logger.warning(
                "transcript_fetcher.cache_failed",
                video_id=transcript.video_id,
                error=str(e),
            )
    
    async def get_available_languages(self, video_id: str) -> List[str]:
        """
        Get list of available languages for a video
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            List of language codes
        """
        loop = asyncio.get_event_loop()
        
        try:
            transcript_list = await loop.run_in_executor(
                None,
                self.api.list_transcripts,
                video_id,
            )
            
            languages = set()
            
            # Add manually created transcripts
            for lang in transcript_list._manually_created_transcripts:
                languages.add(lang)
            
            # Add generated transcripts
            for lang in transcript_list._generated_transcripts:
                languages.add(lang)
            
            return sorted(list(languages))
            
        except Exception as e:
            self.logger.error(
                "transcript_fetcher.languages_failed",
                video_id=video_id,
                error=str(e),
            )
            return []


# ------------------------------------------------------------------------
# Factory Functions
# ------------------------------------------------------------------------

def create_transcript_fetcher(
    cache=None,
    retry_config: Optional[RetryConfig] = None,
    logger=None,
    metrics=None,
) -> TranscriptFetcher:
    """
    Create transcript fetcher with default configuration
    
    Args:
        cache: Cache client
        retry_config: Retry configuration
        logger: Logger instance
        metrics: Metrics collector
        
    Returns:
        Configured TranscriptFetcher
    """
    validator = YouTubeURLValidator(strict_mode=False)
    
    return TranscriptFetcher(
        validator=validator,
        cache=cache,
        retry_config=retry_config,
        logger=logger,
        metrics=metrics,
    )


# ------------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------------

"""
# Usage examples:

fetcher = create_transcript_fetcher()

# Fetch transcript in preferred language
try:
    transcript = await fetcher.fetch(
        video_id="dQw4w9WgXcQ",
        preferred_languages=["hi", "en"],
    )
    
    print(f"Language: {transcript.language}")
    print(f"Segments: {transcript.segment_count}")
    print(f"Total text: {transcript.length} chars")
    
    # Access segments
    for segment in transcript.segments[:5]:
        print(f"[{segment.start:.1f}s] {segment.text}")
        
except NotFoundError:
    print("No transcript available")
except RetryableError:
    print("Temporary failure, try again")

# Get available languages
languages = await fetcher.get_available_languages("dQw4w9WgXcQ")
print(f"Available: {languages}")

# Processed vs raw
raw_transcript = await fetcher.fetch(video_id, process=False)
clean_transcript = await fetcher.fetch(video_id, process=True)
"""

# ------------------------------------------------------------------------
# Error Handling Examples
# ------------------------------------------------------------------------

"""
Error scenarios handled:

1. No transcript at all:
   raise NotFoundError("No transcript found")

2. Video unavailable/deleted:
   raise NotFoundError("Video unavailable")

3. Transcripts disabled:
   raise NotFoundError("Transcripts disabled")

4. Rate limited:
   raise RetryableError("Rate limited", retry_after=60)

5. Network timeout:
   raise RetryableError("Connection timeout")

6. Invalid video ID:
   raise ValidationError("Invalid video ID")
"""

# ------------------------------------------------------------------------
# Performance Considerations
# ------------------------------------------------------------------------

"""
Performance Optimizations:

1. Caching: Store transcripts to avoid repeated API calls
2. Async execution: Run sync API in thread pool
3. Segment optimization: Merge/split for better chunking
4. Lazy processing: Only process when requested
5. Memory efficient: Stream processing for large transcripts

Memory Usage:
- Full transcript stored in memory
- ~1MB for 1-hour video
- Can be truncated if needed
"""