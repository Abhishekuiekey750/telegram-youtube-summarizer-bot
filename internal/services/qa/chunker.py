"""
Semantic Chunking Service
Splits transcripts into meaningful chunks while preserving timestamps

Features:
- Sentence boundary detection
- Semantic grouping
- Timestamp preservation
- Configurable chunk sizes
- Overlap support
- Metadata extraction
- Multi-language support
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import timedelta
import hashlib

import structlog

from internal.services.youtube.transcript import Transcript, TranscriptSegment
from internal.pkg.errors import ValidationError
from internal.pkg.metrics import MetricsCollector


@dataclass
class Sentence:
    """A single sentence with metadata"""
    text: str
    start_time: float
    end_time: float
    segment_indices: List[int]  # Original segment indices
    token_count: int = 0
    
    @property
    def timestamp_range(self) -> str:
        """Get formatted timestamp range"""
        start = self._format_time(self.start_time)
        end = self._format_time(self.end_time)
        return f"{start} - {end}"
    
    def _format_time(self, seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class Chunk:
    """A semantic chunk of transcript"""
    id: int
    text: str
    sentences: List[Sentence]
    start_time: float
    end_time: float
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def timestamp_range(self) -> str:
        """Get formatted timestamp range"""
        start = self._format_time(self.start_time)
        end = self._format_time(self.end_time)
        return f"{start} - {end}"
    
    def _format_time(self, seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def sentence_count(self) -> int:
        return len(self.sentences)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "chunk_id": self.id,
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "timestamp_range": self.timestamp_range,
            "sentence_count": self.sentence_count,
            "token_count": self.token_count,
            "duration": self.duration,
            "metadata": self.metadata,
        }


class SemanticChunker:
    """
    Splits transcripts into semantic chunks for optimal Q&A retrieval
    
    Features:
    - Sentence-aware splitting
    - Timestamp preservation
    - Configurable chunk sizes
    - Overlap support
    - Metadata extraction for better retrieval
    - Multi-language support (handles different sentence boundaries)
    """
    
    # Sentence boundary patterns for different languages
    SENTENCE_BOUNDARIES = {
        "en": r'(?<=[.!?])\s+(?=[A-Z])',  # English: .!? followed by capital
        "hi": r'(?<=[.!?।])\s+',          # Hindi: includes । (danda)
        "ta": r'(?<=[.!?।])\s+',          # Tamil: similar
        "te": r'(?<=[.!?।])\s+',          # Telugu
        "kn": r'(?<=[.!?।])\s+',          # Kannada
        "ml": r'(?<=[.!?।])\s+',          # Malayalam
        "bn": r'(?<=[.!?।])\s+',          # Bengali
    }
    
    # Default fallback pattern
    DEFAULT_BOUNDARY = r'(?<=[.!?])\s+'
    
    def __init__(
        self,
        target_chunk_size: int = 1000,  # characters
        min_chunk_size: int = 300,
        max_chunk_size: int = 2000,
        overlap_sentences: int = 1,  # Number of sentences to overlap
        enable_metadata: bool = True,
        logger=None,
        metrics=None,
    ):
        """
        Initialize semantic chunker
        
        Args:
            target_chunk_size: Target chunk size in characters
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
            overlap_sentences: Number of sentences to overlap between chunks
            enable_metadata: Whether to extract metadata from chunks
            logger: Structured logger
            metrics: Metrics collector
        """
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences
        self.enable_metadata = enable_metadata
        self.logger = logger or structlog.get_logger(__name__)
        self.metrics = metrics or MetricsCollector("chunker")
        
        self.logger.info(
            "semantic_chunker.initialized",
            target_size=target_chunk_size,
            min_size=min_chunk_size,
            max_size=max_chunk_size,
            overlap=overlap_sentences,
        )
    
    async def chunk_transcript(
        self,
        transcript: Transcript,
        language: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Split transcript into semantic chunks
        
        Args:
            transcript: Transcript object with segments
            language: Language code (auto-detected if not provided)
            
        Returns:
            List of semantic chunks
        """
        if not transcript.segments:
            self.logger.warning(
                "chunker.empty_transcript",
                video_id=transcript.video_id,
            )
            return []
        
        # Determine language
        lang = language or transcript.language or "en"
        
        # Step 1: Convert segments to sentences
        sentences = await self._segments_to_sentences(transcript.segments, lang)
        
        if not sentences:
            self.logger.warning(
                "chunker.no_sentences",
                video_id=transcript.video_id,
                segments=len(transcript.segments),
            )
            return []
        
        self.logger.debug(
            "chunker.sentences_created",
            video_id=transcript.video_id,
            sentence_count=len(sentences),
        )
        
        # Step 2: Merge very short sentences
        sentences = await self._merge_short_sentences(sentences)
        
        # Step 3: Create chunks
        chunks = await self._create_chunks(sentences, lang)
        
        # Step 4: Add metadata if enabled
        if self.enable_metadata:
            chunks = await self._add_metadata(chunks, transcript)
        
        self.logger.info(
            "chunker.completed",
            video_id=transcript.video_id,
            sentences=len(sentences),
            chunks=len(chunks),
            avg_chunk_size=sum(len(c.text) for c in chunks) // len(chunks) if chunks else 0,
        )
        
        if self.metrics:
            self.metrics.record_distribution(
                "chunker.chunk_sizes",
                [len(c.text) for c in chunks],
            )
            self.metrics.gauge(
                "chunker.chunk_count",
                value=len(chunks),
                tags={"video_id": transcript.video_id},
            )
        
        return chunks
    
    # ------------------------------------------------------------------------
    # Sentence Processing
    # ------------------------------------------------------------------------
    
    async def _segments_to_sentences(
        self,
        segments: List[TranscriptSegment],
        language: str,
    ) -> List[Sentence]:
        """
        Convert transcript segments to sentences with timestamps
        
        This is the core algorithm that preserves timing information
        while grouping into natural language sentences.
        """
        if not segments:
            return []
        
        # Get sentence boundary pattern for language
        pattern = self.SENTENCE_BOUNDARIES.get(language, self.DEFAULT_BOUNDARY)
        
        sentences = []
        current_sentence = []
        current_start = segments[0].start
        current_segment_indices = []
        
        for i, segment in enumerate(segments):
            text = segment.text.strip()
            if not text:
                continue
            
            # Split segment into potential sentences
            # But keep in mind that a segment might contain multiple sentences
            parts = re.split(pattern, text)
            
            for j, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                
                # Add to current sentence
                current_sentence.append(part)
                current_segment_indices.append(i)
                
                # Check if this part ends with sentence boundary
                if j < len(parts) - 1 or self._ends_with_boundary(part, language):
                    # Complete sentence
                    sentence_text = " ".join(current_sentence)
                    end_time = segment.start + segment.duration
                    
                    sentences.append(Sentence(
                        text=sentence_text,
                        start_time=current_start,
                        end_time=end_time,
                        segment_indices=current_segment_indices.copy(),
                        token_count=self._count_tokens(sentence_text),
                    ))
                    
                    # Reset for next sentence
                    current_sentence = []
                    current_start = segment.start + segment.duration
                    current_segment_indices = []
        
        # Handle any remaining text
        if current_sentence:
            sentence_text = " ".join(current_sentence)
            sentences.append(Sentence(
                text=sentence_text,
                start_time=current_start,
                end_time=segments[-1].start + segments[-1].duration,
                segment_indices=current_segment_indices,
                token_count=self._count_tokens(sentence_text),
            ))
        
        return sentences
    
    def _ends_with_boundary(self, text: str, language: str) -> bool:
        """Check if text ends with a sentence boundary"""
        if not text:
            return False
        
        last_char = text[-1]
        
        # Common sentence endings
        if last_char in {'.', '!', '?'}:
            return True
        
        # Indian language specific (danda)
        if language in {'hi', 'ta', 'te', 'kn', 'ml', 'bn'} and last_char == '।':
            return True
        
        return False
    
    async def _merge_short_sentences(
        self,
        sentences: List[Sentence],
        min_chars: int = 50,
    ) -> List[Sentence]:
        """
        Merge very short sentences with next sentence
        
        This prevents tiny chunks that don't contain meaningful content.
        """
        if len(sentences) <= 1:
            return sentences
        
        merged = []
        i = 0
        
        while i < len(sentences):
            current = sentences[i]
            
            # If current sentence is too short and there's a next sentence
            if len(current.text) < min_chars and i + 1 < len(sentences):
                next_sent = sentences[i + 1]
                
                # Merge with next
                merged_text = f"{current.text} {next_sent.text}"
                merged_sentence = Sentence(
                    text=merged_text,
                    start_time=current.start_time,
                    end_time=next_sent.end_time,
                    segment_indices=current.segment_indices + next_sent.segment_indices,
                    token_count=self._count_tokens(merged_text),
                )
                merged.append(merged_sentence)
                i += 2  # Skip next sentence
            else:
                merged.append(current)
                i += 1
        
        return merged
    
    # ------------------------------------------------------------------------
    # Chunk Creation
    # ------------------------------------------------------------------------
    
    async def _create_chunks(
        self,
        sentences: List[Sentence],
        language: str,
    ) -> List[Chunk]:
        """
        Create semantic chunks from sentences
        
        Strategy:
        1. Start with first sentence
        2. Add sentences until target size reached
        3. Don't split sentences
        4. Add overlap if configured
        """
        if not sentences:
            return []
        
        chunks = []
        chunk_id = 0
        i = 0
        
        while i < len(sentences):
            chunk_sentences = []
            chunk_text = []
            chunk_size = 0
            start_time = sentences[i].start_time
            
            # Build chunk
            j = i
            while j < len(sentences):
                sent = sentences[j]
                sent_size = len(sent.text)
                
                # If adding this sentence exceeds max size and we have at least min size
                if chunk_size + sent_size > self.max_chunk_size and chunk_size >= self.min_chunk_size:
                    break
                
                # Add sentence to chunk
                chunk_sentences.append(sent)
                chunk_text.append(sent.text)
                chunk_size += sent_size
                j += 1
            
            if not chunk_sentences:
                # Should never happen
                i += 1
                continue
            
            # Calculate end time
            end_time = chunk_sentences[-1].end_time
            
            # Create chunk
            chunk = Chunk(
                id=chunk_id,
                text=" ".join(chunk_text),
                sentences=chunk_sentences,
                start_time=start_time,
                end_time=end_time,
                token_count=self._count_tokens(" ".join(chunk_text)),
            )
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            if self.overlap_sentences > 0:
                # Start from previous sentence for overlap
                i = max(j - self.overlap_sentences, i + 1)
            else:
                i = j
            
            chunk_id += 1
        
        return chunks
    
    # ------------------------------------------------------------------------
    # Metadata Extraction
    # ------------------------------------------------------------------------
    
    async def _add_metadata(
        self,
        chunks: List[Chunk],
        transcript: Transcript,
    ) -> List[Chunk]:
        """
        Add metadata to chunks for better retrieval
        
        Metadata includes:
        - Contains numbers (pricing, dates)
        - Contains questions
        - Topic indicators
        - Named entities (basic)
        """
        for chunk in chunks:
            metadata = {}
            text = chunk.text.lower()
            
            # Check for numbers (potential pricing, statistics)
            metadata["has_numbers"] = bool(re.search(r'\d+', text))
            
            # Check for currency symbols
            metadata["has_currency"] = bool(re.search(r'[$€£₹]', text))
            
            # Check for questions
            metadata["has_questions"] = "?" in text
            
            # Check for list indicators
            metadata["has_list"] = bool(re.search(r'(\n|^)[\*\-\•]', chunk.text))
            
            # Topic indicators (simple keyword matching)
            topics = []
            if re.search(r'price|cost|pricing|paid|free|subscription', text):
                topics.append("pricing")
            if re.search(r'feature|capability|can do|ability', text):
                topics.append("features")
            if re.search(r'install|setup|configure|how to', text):
                topics.append("tutorial")
            if re.search(r'problem|issue|bug|error|fix', text):
                topics.append("troubleshooting")
            
            metadata["topics"] = topics
            
            # Add to chunk
            chunk.metadata = metadata
        
        return chunks
    
    # ------------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------------
    
    def _count_tokens(self, text: str) -> int:
        """Rough token count (words * 1.3)"""
        words = len(text.split())
        return int(words * 1.3)
    
    # ------------------------------------------------------------------------
    # Chunk Retrieval and Search
    # ------------------------------------------------------------------------
    
    async def get_chunks_by_time_range(
        self,
        chunks: List[Chunk],
        start_time: float,
        end_time: float,
    ) -> List[Chunk]:
        """Get chunks that overlap with time range"""
        return [
            chunk for chunk in chunks
            if chunk.start_time <= end_time and chunk.end_time >= start_time
        ]
    
    async def get_chunks_by_topic(
        self,
        chunks: List[Chunk],
        topic: str,
    ) -> List[Chunk]:
        """Get chunks related to a specific topic"""
        if not self.enable_metadata:
            return []
        
        return [
            chunk for chunk in chunks
            if topic in chunk.metadata.get("topics", [])
        ]
    
    async def get_chunk_preview(
        self,
        chunk: Chunk,
        max_length: int = 200,
    ) -> str:
        """Get preview text with timestamp"""
        preview = chunk.text[:max_length]
        if len(chunk.text) > max_length:
            preview += "..."
        
        return f"[{chunk.timestamp_range}] {preview}"
    
    # ------------------------------------------------------------------------
    # Chunk Statistics
    # ------------------------------------------------------------------------
    
    async def get_chunk_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about chunks"""
        if not chunks:
            return {}
        
        sizes = [len(c.text) for c in chunks]
        durations = [c.duration for c in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_size": sum(sizes) // len(chunks),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "avg_duration": sum(durations) / len(durations),
            "total_duration": sum(durations),
            "total_tokens": sum(c.token_count for c in chunks),
            "chunks_with_numbers": sum(1 for c in chunks if c.metadata.get("has_numbers")),
            "chunks_with_questions": sum(1 for c in chunks if c.metadata.get("has_questions")),
            "top_topics": self._get_top_topics(chunks),
        }
    
    def _get_top_topics(self, chunks: List[Chunk]) -> Dict[str, int]:
        """Get topic frequency"""
        topic_counts = {}
        for chunk in chunks:
            for topic in chunk.metadata.get("topics", []):
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        return dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True))


# ------------------------------------------------------------------------
# Factory Function
# ------------------------------------------------------------------------

def create_chunker(
    target_chunk_size: int = 1000,
    min_chunk_size: int = 300,
    max_chunk_size: int = 2000,
    overlap_sentences: int = 1,
    enable_metadata: bool = True,
    logger=None,
    metrics=None,
) -> SemanticChunker:
    """
    Create semantic chunker with configuration
    
    Args:
        target_chunk_size: Target chunk size in characters
        min_chunk_size: Minimum chunk size
        max_chunk_size: Maximum chunk size
        overlap_sentences: Number of sentences to overlap
        enable_metadata: Whether to extract metadata
        logger: Logger instance
        metrics: Metrics collector
        
    Returns:
        Configured SemanticChunker
    """
    return SemanticChunker(
        target_chunk_size=target_chunk_size,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        overlap_sentences=overlap_sentences,
        enable_metadata=enable_metadata,
        logger=logger,
        metrics=metrics,
    )


# ------------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------------

"""
# Usage example:

chunker = create_chunker()

# Chunk a transcript
chunks = await chunker.chunk_transcript(transcript)

# Work with chunks
for chunk in chunks:
    print(f"Chunk {chunk.id} [{chunk.timestamp_range}]")
    print(f"Text: {chunk.text[:100]}...")
    print(f"Topics: {chunk.metadata.get('topics', [])}")
    print("---")

# Get stats
stats = await chunker.get_chunk_stats(chunks)
print(f"Total chunks: {stats['total_chunks']}")
print(f"Average size: {stats['avg_size']} chars")

# Search by topic
pricing_chunks = await chunker.get_chunks_by_topic(chunks, "pricing")
print(f"Found {len(pricing_chunks)} pricing-related chunks")
"""

# ------------------------------------------------------------------------
# Example Output
# ------------------------------------------------------------------------

"""
Sample chunk output:

Chunk 0 [00:00 - 01:23]
Text: Welcome to our pricing overview. We offer three tiers: 
      Basic at $49 per month, Pro at $99 per month, and Enterprise 
      with custom pricing...
Topics: ['pricing']
Has numbers: True

Chunk 1 [01:23 - 02:45]
Text: Now let's talk about features. The Basic tier includes 
      100 API calls per day, while Pro includes 1000 calls...
Topics: ['features']
Has numbers: True
"""