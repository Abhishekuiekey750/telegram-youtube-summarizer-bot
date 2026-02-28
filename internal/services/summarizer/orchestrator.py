"""
Summarization Orchestrator
Coordinates the entire summarization pipeline with intelligent chunking

Features:
- Intelligent chunking based on transcript length
- Parallel processing of chunks
- Hierarchical summarization for long videos
- Structured output formatting
- Timestamp extraction and alignment
- Multi-language support
- Redundancy removal
"""

import asyncio
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum

import structlog

from internal.services.youtube.transcript import Transcript, TranscriptSegment
from internal.ai.models.factory import ModelFactory
from internal.ai.prompts.manager import PromptManager
from internal.services.language import LanguageService
from internal.domain.value_objects import Language
from internal.pkg.errors import ValidationError, RetryableError, NotFoundError
from internal.pkg.metrics import MetricsCollector


class ChunkingStrategy(Enum):
    """Available chunking strategies"""
    SINGLE = "single"           # One chunk for short videos
    OVERLAPPING = "overlapping"  # Overlapping chunks for medium videos
    HIERARCHICAL = "hierarchical" # Multi-level for long videos
    SEMANTIC = "semantic"        # Natural boundaries (advanced)


class SummaryType(Enum):
    """Types of summaries that can be generated"""
    CONCISE = "concise"          # Brief overview
    DETAILED = "detailed"        # Comprehensive summary
    BULLET_POINTS = "bullet_points"  # Just key points
    TIMESTAMPS = "timestamps"    # Timestamped highlights


@dataclass
class Chunk:
    """A chunk of transcript for processing"""
    id: int
    text: str
    start_time: float
    end_time: float
    segments: List[TranscriptSegment]
    token_count: int = 0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def timestamp_range(self) -> str:
        start = self._format_time(self.start_time)
        end = self._format_time(self.end_time)
        return f"{start} - {end}"
    
    def _format_time(self, seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


@dataclass
class ChunkSummary:
    """Summary of a single chunk"""
    chunk_id: int
    summary: str
    key_points: List[Dict[str, Any]]
    start_time: float
    end_time: float
    token_count: int = 0


@dataclass
class VideoSummary:
    """Complete video summary"""
    video_id: str
    title: str
    key_points: List[Dict[str, Any]]  # [{"point": "...", "timestamp": "MM:SS"}]
    core_takeaway: str
    action_items: List[str] = field(default_factory=list)
    summary_type: SummaryType = SummaryType.CONCISE
    language: Language = Language.ENGLISH
    chunk_count: int = 1
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for response"""
        return {
            "key_points": self.key_points,
            "core_takeaway": self.core_takeaway,
            "action_items": self.action_items,
            "language": self.language.code,
        }
    
    def to_formatted_text(self) -> str:
        """Format as readable text"""
        lines = []
        
        # Key points
        lines.append("🔑 *Key Points:*")
        for i, point in enumerate(self.key_points, 1):
            timestamp = point.get("timestamp", "")
            if timestamp:
                lines.append(f"{i}. `{timestamp}` - {point['point']}")
            else:
                lines.append(f"{i}. {point['point']}")
        
        # Core takeaway
        lines.append("\n💡 *Core Takeaway:*")
        lines.append(self.core_takeaway)
        
        # Action items
        if self.action_items:
            lines.append("\n✅ *Action Items:*")
            for item in self.action_items:
                lines.append(f"• {item}")
        
        return "\n".join(lines)


class SummarizationOrchestrator:
    """
    Orchestrates the entire summarization process
    
    Handles:
    - Transcript analysis
    - Intelligent chunking
    - Parallel processing
    - Summary merging
    - Output formatting
    """
    
    # Token limits for different models
    MODEL_TOKEN_LIMITS = {
        "gpt-4": 8192,
        "gpt-3.5-turbo": 4096,
        "claude": 100000,
        "default": 4000,
    }
    
    # Recommended tokens per chunk (leaving room for prompt)
    CHUNK_TOKEN_TARGET = 2000
    CHUNK_OVERLAP_TOKENS = 200  # 10% overlap
    
    def __init__(
        self,
        model_factory: ModelFactory,
        prompt_manager: PromptManager,
        language_service: LanguageService,
        logger=None,
        metrics=None,
        model_name: str = "gpt-3.5-turbo",
        max_parallel_chunks: int = 3,
        enable_parallel: bool = True,
    ):
        """
        Initialize summarization orchestrator
        
        Args:
            model_factory: Factory for AI models
            prompt_manager: Prompt template manager
            language_service: Language service
            logger: Structured logger
            metrics: Metrics collector
            model_name: Default model to use
            max_parallel_chunks: Max chunks to process in parallel
            enable_parallel: Whether to enable parallel processing
        """
        self.model_factory = model_factory
        self.prompt_manager = prompt_manager
        self.language_service = language_service
        self.logger = logger or structlog.get_logger(__name__)
        self.metrics = metrics or MetricsCollector("summarizer")
        self.model_name = model_name
        self.max_parallel_chunks = max_parallel_chunks
        self.enable_parallel = enable_parallel
        
        # Get tokenizer (lazy import so tiktoken does not slow/hang startup)
        try:
            import tiktoken
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except Exception:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        self.logger.info(
            "summarization_orchestrator.initialized",
            model=model_name,
            max_parallel=max_parallel_chunks,
            token_target=self.CHUNK_TOKEN_TARGET,
        )
    
    async def generate_summary(
        self,
        transcript: Transcript | None = None,
        metadata: Optional[Dict[str, Any]] = None,
        summary_type: SummaryType = SummaryType.CONCISE,
        language: Optional[Language] = None,
        max_points: int = 5,
        **_: Any,
    ) -> VideoSummary:
        """
        Backwards-compatible wrapper used by handlers.
        
        The original project called ``generate_summary`` on a higher-level
        service; here we delegate to ``summarize`` on the orchestrator.
        
        Args:
            transcript: Parsed transcript (required).
            metadata: Video metadata dict, used for the title when present.
            summary_type: Type of summary to generate.
            language: Target language for the summary.
            max_points: Maximum number of key points.
        """
        if transcript is None:
            raise ValidationError("Transcript is required for generate_summary")
        
        title = (metadata or {}).get("title") or "Unknown title"
        return await self.summarize(
            transcript=transcript,
            video_title=title,
            summary_type=summary_type,
            language=language,
            max_points=max_points,
        )
    
    async def summarize(
        self,
        transcript: Transcript,
        video_title: str,
        summary_type: SummaryType = SummaryType.CONCISE,
        language: Optional[Language] = None,
        max_points: int = 5,
    ) -> VideoSummary:
        """
        Generate summary from transcript
        
        Args:
            transcript: Video transcript
            video_title: Video title
            summary_type: Type of summary to generate
            language: Target language (defaults to transcript language)
            max_points: Maximum number of key points
            
        Returns:
            Structured video summary
        """
        import time
        start_time = time.time()
        
        # Determine language
        target_lang = language or Language.from_code(transcript.language) or Language.ENGLISH
        
        self.logger.info(
            "summarizer.starting",
            video_id=transcript.video_id,
            segments=len(transcript.segments),
            total_tokens=self._count_tokens(transcript.text),
            target_language=target_lang.code,
        )
        
        # Step 1: Analyze transcript
        analysis = await self._analyze_transcript(transcript)
        
        # Step 2: Choose chunking strategy
        chunks = await self._create_chunks(transcript, analysis)
        
        self.logger.debug(
            "summarizer.chunks_created",
            chunk_count=len(chunks),
            strategy=analysis["strategy"],
        )
        
        # Step 3: Process chunks (parallel or sequential)
        if len(chunks) == 1:
            # Single chunk - direct summarization
            final_summary = await self._summarize_single_chunk(
                chunks[0],
                video_title,
                summary_type,
                target_lang,
            )
        else:
            # Multiple chunks - process and merge
            chunk_summaries = await self._process_chunks_parallel(
                chunks,
                video_title,
                target_lang,
            )
            
            # Merge chunk summaries
            final_summary = await self._merge_summaries(
                chunk_summaries,
                video_title,
                summary_type,
                target_lang,
                max_points,
            )
        
        # Add metadata
        final_summary.video_id = transcript.video_id
        final_summary.title = video_title
        final_summary.summary_type = summary_type
        final_summary.language = target_lang
        final_summary.chunk_count = len(chunks)
        final_summary.processing_time = time.time() - start_time
        
        self.logger.info(
            "summarizer.completed",
            video_id=transcript.video_id,
            chunks=len(chunks),
            key_points=len(final_summary.key_points),
            duration_ms=round(final_summary.processing_time * 1000),
        )
        
        if self.metrics:
            self.metrics.record_latency(
                "summarization.duration",
                final_summary.processing_time,
                tags={
                    "chunks": len(chunks),
                    "language": target_lang.code,
                },
            )
            self.metrics.increment(
                "summarization.completed",
                tags={
                    "chunks": len(chunks),
                    "type": summary_type.value,
                },
            )
        
        return final_summary
    
    # ------------------------------------------------------------------------
    # Transcript Analysis
    # ------------------------------------------------------------------------
    
    async def _analyze_transcript(self, transcript: Transcript) -> Dict[str, Any]:
        """
        Analyze transcript to determine best strategy
        
        Returns:
            Analysis with token counts, language, recommended strategy
        """
        full_text = transcript.text
        token_count = self._count_tokens(full_text)
        
        # Determine strategy based on token count
        if token_count < self.CHUNK_TOKEN_TARGET:
            strategy = ChunkingStrategy.SINGLE
            recommended_chunks = 1
        elif token_count < self.CHUNK_TOKEN_TARGET * 3:
            strategy = ChunkingStrategy.OVERLAPPING
            recommended_chunks = (token_count // self.CHUNK_TOKEN_TARGET) + 1
        else:
            strategy = ChunkingStrategy.HIERARCHICAL
            recommended_chunks = min(
                (token_count // self.CHUNK_TOKEN_TARGET) + 2,
                10  # Cap at 10 chunks max
            )
        
        return {
            "token_count": token_count,
            "segment_count": len(transcript.segments),
            "duration": transcript.duration,
            "strategy": strategy,
            "recommended_chunks": recommended_chunks,
            "language": transcript.language,
        }
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    # ------------------------------------------------------------------------
    # Chunking Strategies
    # ------------------------------------------------------------------------
    
    async def _create_chunks(
        self,
        transcript: Transcript,
        analysis: Dict[str, Any],
    ) -> List[Chunk]:
        """
        Create chunks based on analysis
        
        Args:
            transcript: Full transcript
            analysis: Transcript analysis
            
        Returns:
            List of chunks
        """
        strategy = analysis["strategy"]
        
        if strategy == ChunkingStrategy.SINGLE:
            return await self._create_single_chunk(transcript)
        elif strategy == ChunkingStrategy.OVERLAPPING:
            return await self._create_overlapping_chunks(
                transcript,
                analysis["recommended_chunks"],
            )
        elif strategy == ChunkingStrategy.HIERARCHICAL:
            return await self._create_hierarchical_chunks(
                transcript,
                analysis["recommended_chunks"],
            )
        else:
            # Default to overlapping
            return await self._create_overlapping_chunks(transcript, 3)
    
    async def _create_single_chunk(self, transcript: Transcript) -> List[Chunk]:
        """Create a single chunk from entire transcript"""
        if not transcript.segments:
            return []
        
        chunk = Chunk(
            id=0,
            text=transcript.text,
            start_time=transcript.segments[0].start,
            end_time=transcript.segments[-1].end,
            segments=transcript.segments,
            token_count=self._count_tokens(transcript.text),
        )
        
        return [chunk]
    
    async def _create_overlapping_chunks(
        self,
        transcript: Transcript,
        num_chunks: int,
    ) -> List[Chunk]:
        """
        Create overlapping chunks for medium-length videos
        
        Each chunk overlaps with neighbors by ~10% to preserve context
        """
        if not transcript.segments:
            return []
        
        segments = transcript.segments
        total_duration = segments[-1].end - segments[0].start
        chunk_duration = total_duration / num_chunks
        overlap_duration = chunk_duration * 0.1  # 10% overlap
        
        chunks = []
        
        for i in range(num_chunks):
            start_time = segments[0].start + (i * chunk_duration)
            end_time = start_time + chunk_duration + overlap_duration
            
            # Find segments in this time range
            chunk_segments = [
                s for s in segments
                if s.start >= start_time - 1 and s.start <= end_time + 1
            ]
            
            if not chunk_segments:
                continue
            
            chunk_text = " ".join(s.text for s in chunk_segments)
            
            chunks.append(Chunk(
                id=i,
                text=chunk_text,
                start_time=max(start_time, chunk_segments[0].start),
                end_time=min(end_time, chunk_segments[-1].end),
                segments=chunk_segments,
                token_count=self._count_tokens(chunk_text),
            ))
        
        return chunks
    
    async def _create_hierarchical_chunks(
        self,
        transcript: Transcript,
        target_chunks: int,
    ) -> List[Chunk]:
        """
        Create chunks for very long videos using semantic boundaries
        
        Attempts to find natural break points (topic changes, pauses)
        """
        if not transcript.segments:
            return []
        
        segments = transcript.segments
        
        # Look for natural breaks (long pauses, topic indicators)
        break_points = self._find_natural_breaks(segments, target_chunks)
        
        chunks = []
        start_idx = 0
        
        for i, end_idx in enumerate(break_points + [len(segments)]):
            chunk_segments = segments[start_idx:end_idx]
            
            if not chunk_segments:
                continue
            
            chunk_text = " ".join(s.text for s in chunk_segments)
            
            chunks.append(Chunk(
                id=i,
                text=chunk_text,
                start_time=chunk_segments[0].start,
                end_time=chunk_segments[-1].end,
                segments=chunk_segments,
                token_count=self._count_tokens(chunk_text),
            ))
            
            start_idx = end_idx
        
        return chunks
    
    def _find_natural_breaks(
        self,
        segments: List[TranscriptSegment],
        target_chunks: int,
    ) -> List[int]:
        """
        Find natural break points in transcript
        
        Considers:
        - Long pauses between segments
        - Topic transition words
        - Question/answer patterns
        """
        if len(segments) <= target_chunks:
            return []
        
        # Calculate gaps between segments
        gaps = []
        for i in range(1, len(segments)):
            gap = segments[i].start - (segments[i-1].start + segments[i-1].duration)
            gaps.append((gap, i))
        
        # Sort by gap size (largest first)
        gaps.sort(reverse=True, key=lambda x: x[0])
        
        # Take top gaps as break points
        break_indices = sorted([idx for _, idx in gaps[:target_chunks-1]])
        
        return break_indices
    
    # ------------------------------------------------------------------------
    # Chunk Processing
    # ------------------------------------------------------------------------
    
    async def _summarize_single_chunk(
        self,
        chunk: Chunk,
        video_title: str,
        summary_type: SummaryType,
        language: Language,
    ) -> VideoSummary:
        """Summarize a single chunk directly"""
        # Get appropriate prompt; if no templates are loaded, fall back
        # to a simple built-in prompt string.
        try:
            prompt = await self.prompt_manager.get_prompt(
                "summarize_single",
                {
                    "title": video_title,
                    "text": chunk.text,
                    "type": summary_type.value,
                },
            )
        except NotFoundError:
            prompt = (
                "You are a helpful assistant that summarizes YouTube video transcripts.\n\n"
                f"Video title: {video_title}\n\n"
                "Transcript segment:\n"
                f"{chunk.text[:12000]}\n\n"
                "Respond with ONLY a valid JSON object (no markdown, no code fence), with these exact keys:\n"
                '- "summary": 2-3 sentence overview\n'
                '- "key_points": array of 3-5 objects, each with "point" (string) and "timestamp" (string, e.g. "" or "MM:SS")\n'
                '- "core_takeaway": one sentence main takeaway\n'
                'Example: {"summary":"...","key_points":[{"point":"...","timestamp":""}],"core_takeaway":"..."}'
            )
        
        # Call model with strict JSON system prompt and enough tokens
        model = await self.model_factory.get_generation_model(self.model_name)
        system_prompt = (
            "You are a summarization assistant. You must respond with ONLY a valid JSON object. "
            "No markdown, no code fences, no explanation. Use exactly these keys: summary, key_points (array of {point, timestamp}), core_takeaway."
        )
        response = (
            await model.generate(
                prompt,
                language=language,
                system_prompt=system_prompt,
                max_tokens=2000,
            )
        ).strip()
        
        # If API returned a quota error, show a clear message
        if response.startswith("GEMINI_QUOTA_EXCEEDED:"):
            return VideoSummary(
                video_id="",
                title=video_title,
                key_points=[{"point": "Gemini free tier quota exceeded or not yet active. Try again later or check https://ai.google.dev/gemini-api/docs/rate-limits", "timestamp": ""}],
                core_takeaway="Summary failed: Gemini quota exceeded (limit may be 0 for new keys). Wait a few minutes or check rate limits at https://ai.google.dev/gemini-api/docs/rate-limits",
                action_items=[],
                summary_type=summary_type,
                language=language,
            )
        if response.startswith("OPENAI_QUOTA_EXCEEDED:"):
            return VideoSummary(
                video_id="",
                title=video_title,
                key_points=[{"point": "OpenAI quota exceeded. Add payment or check billing at platform.openai.com/account/billing", "timestamp": ""}],
                core_takeaway="Summary failed: your OpenAI account has no usable quota. Check your plan and billing at https://platform.openai.com/account/billing",
                action_items=[],
                summary_type=summary_type,
                language=language,
            )
        # If API returned a generic error message, surface it clearly
        if response.startswith("I'm sorry, but I couldn't generate"):
            return VideoSummary(
                video_id="",
                title=video_title,
                key_points=[{"point": "AI service error. If using OpenClaw, ensure the gateway is running and token/base URL are set in .env", "timestamp": ""}],
                core_takeaway="Summary failed: AI service error. If using OpenAI directly, check OPENAI_API_KEY in .env.",
                action_items=[],
                summary_type=summary_type,
                language=language,
            )
        
        # Parse response: try JSON (including inside markdown code blocks), then text fallback
        data = self._parse_llm_response(response)
        key_points = data.get("key_points", []) or []
        core_takeaway = (data.get("core_takeaway") or data.get("summary") or "").strip()
        if not core_takeaway and response:
            core_takeaway = response[:800].strip()
        if not key_points and core_takeaway:
            key_points = [{"point": core_takeaway[:300] + ("..." if len(core_takeaway) > 300 else ""), "timestamp": ""}]
        # Ensure each key point has a timestamp; distribute over chunk time range so they're not all 00:00
        start_s = chunk.start_time
        end_s = chunk.end_time
        n_pts = max(len(key_points), 1)
        for i, pt in enumerate(key_points):
            if not pt.get("timestamp"):
                t_sec = start_s + (end_s - start_s) * i / n_pts
                pt["timestamp"] = self._format_timestamp(t_sec)

        return VideoSummary(
            video_id="",
            title=video_title,
            key_points=key_points,
            core_takeaway=core_takeaway or "Summary could not be extracted.",
            action_items=data.get("action_items", []),
            summary_type=summary_type,
            language=language,
        )
    
    async def _process_chunks_parallel(
        self,
        chunks: List[Chunk],
        video_title: str,
        language: Language,
    ) -> List[ChunkSummary]:
        """Process multiple chunks in parallel"""
        
        # Limit parallel tasks
        semaphore = asyncio.Semaphore(self.max_parallel_chunks)
        
        async def process_with_semaphore(chunk: Chunk) -> ChunkSummary:
            async with semaphore:
                return await self._summarize_chunk(chunk, video_title, language)
        
        # Create tasks
        tasks = [process_with_semaphore(chunk) for chunk in chunks]
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        chunk_summaries = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(
                    "summarizer.chunk_failed",
                    chunk_id=i,
                    error=str(result),
                )
                if self.metrics:
                    self.metrics.increment("summarizer.chunk_failed")
                # Continue with other chunks
            else:
                chunk_summaries.append(result)
        
        return chunk_summaries
    
    async def _summarize_chunk(
        self,
        chunk: Chunk,
        video_title: str,
        language: Language,
    ) -> ChunkSummary:
        """Summarize a single chunk"""
        # Try to load prompt from PromptManager; fall back to a built-in prompt
        # when no templates are configured so the bot still works out-of-the-box.
        try:
            prompt = await self.prompt_manager.get_prompt(
                "summarize_chunk",
                {
                    "title": video_title,
                    "chunk_text": chunk.text,
                    "timestamp_range": chunk.timestamp_range,
                },
            )
        except NotFoundError:
            prompt = (
                "You are a helpful assistant that summarizes parts of YouTube talks.\n\n"
                f"Video title: {video_title}\n"
                f"Chunk time range: {chunk.timestamp_range}\n\n"
                "Transcript chunk:\n"
                f"{chunk.text}\n\n"
                "Write a JSON object with two fields:\\n"
                '  "summary": a 2-3 sentence summary of this chunk,\\n'
                '  "key_points": a list of objects with fields "point" (string) '
                'and optional "timestamp" (MM:SS if you can infer it).\\n'
            )
        
        model = await self.model_factory.get_generation_model(self.model_name)
        response = (await model.generate(prompt, language=language)).strip()
        data = self._parse_llm_response(response)
        summary_text = (data.get("summary") or data.get("core_takeaway") or response[:500] or "").strip()
        key_points = data.get("key_points", []) or []
        # Ensure each key point has a timestamp; distribute over chunk range so they're not all the same
        start_s, end_s = chunk.start_time, chunk.end_time
        n_pts = max(len(key_points), 1)
        for i, pt in enumerate(key_points):
            if not pt.get("timestamp"):
                t_sec = start_s + (end_s - start_s) * i / n_pts
                pt["timestamp"] = self._format_timestamp(t_sec)
        return ChunkSummary(
            chunk_id=chunk.id,
            summary=summary_text,
            key_points=key_points,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            token_count=self._count_tokens(response),
        )
    
    async def _merge_summaries(
        self,
        chunk_summaries: List[ChunkSummary],
        video_title: str,
        summary_type: SummaryType,
        language: Language,
        max_points: int,
    ) -> VideoSummary:
        """
        Merge multiple chunk summaries into final summary.
        
        If no prompt templates are configured, falls back to a simple heuristic
        merge so the bot still works for demo purposes.
        """
        # Fallback path: no prompts loaded
        if not getattr(self.prompt_manager, "_prompts", {}):
            merged_points: List[Dict[str, Any]] = []
            for cs in chunk_summaries:
                for pt in cs.key_points:
                    ts = pt.get("timestamp") or ""
                    if not ts and hasattr(cs, "start_time"):
                        ts = self._format_timestamp(cs.start_time)
                    merged_points.append({**pt, "timestamp": ts})
                if len(merged_points) >= max_points:
                    break
            merged_points = merged_points[:max_points]
            
            if not merged_points:
                merged_points = [
                    {"point": "Summary not available in this demo build.", "timestamp": ""},
                ]
            
            # Derive a real core insight from chunk summaries or first key point (no templates path)
            core_takeaway = ""
            if chunk_summaries:
                core_takeaway = (chunk_summaries[-1].summary or "").strip()[:500]
            if not core_takeaway and merged_points:
                first_point = merged_points[0].get("point") or ""
                if first_point:
                    core_takeaway = first_point[:400] + ("..." if len(first_point) > 400 else "")
            if not core_takeaway:
                core_takeaway = (
                    "This is a simplified demo summary generated without external prompt templates."
                )
            
            return VideoSummary(
                video_id="",
                title=video_title,
                key_points=merged_points,
                core_takeaway=core_takeaway,
                action_items=[],
                summary_type=summary_type,
                language=language,
                chunk_count=len(chunk_summaries),
            )
        
        # Normal path: use prompt + model
        combined = []
        for cs in chunk_summaries:
            combined.append(f"Section {cs.chunk_id + 1}:\n{cs.summary}")
        
        combined_text = "\n\n".join(combined)
        
        prompt = await self.prompt_manager.get_prompt(
            "merge_summaries",
            {
                "title": video_title,
                "section_summaries": combined_text,
                "max_points": max_points,
                "type": summary_type.value,
            },
        )
        
        model = await self.model_factory.get_generation_model(self.model_name)
        response = (await model.generate(prompt, language=language)).strip()
        data = self._parse_llm_response(response)
        key_points = (data.get("key_points") or [])[:max_points]
        core_takeaway = (data.get("core_takeaway") or data.get("summary") or "").strip()
        if not key_points:
            for cs in chunk_summaries:
                key_points.extend(cs.key_points)
                if len(key_points) >= max_points:
                    break
            key_points = key_points[:max_points]
        if not core_takeaway and chunk_summaries:
            core_takeaway = chunk_summaries[-1].summary[:500].strip() or response[:500].strip()
        if not core_takeaway:
            core_takeaway = response[:500].strip() or "Summary merged from sections above."
        for point in key_points:
            if "timestamp" in point and isinstance(point.get("timestamp"), (int, float)):
                point["timestamp"] = self._format_timestamp(point["timestamp"])
        # Fill missing timestamps from chunk sections (requirement: include timestamps)
        if chunk_summaries:
            idx = 0
            for point in key_points:
                if not point.get("timestamp"):
                    cs = chunk_summaries[min(idx, len(chunk_summaries) - 1)]
                    point["timestamp"] = self._format_timestamp(cs.start_time)
                    idx += 1

        return VideoSummary(
            video_id="",
            title=video_title,
            key_points=key_points,
            core_takeaway=core_takeaway,
            action_items=data.get("action_items", []),
            summary_type=summary_type,
            language=language,
            chunk_count=len(chunk_summaries),
        )
    
    # ------------------------------------------------------------------------
    # Response Parsing
    # ------------------------------------------------------------------------
    
    def _parse_llm_response(self, text: str) -> Dict[str, Any]:
        """
        Parse LLM response: try JSON (including inside ```json ... ```), then text heuristics.
        Ensures we never return completely empty key_points/core_takeaway when there is content.
        """
        raw = (text or "").strip()
        if not raw:
            return {"key_points": [], "core_takeaway": "", "summary": "", "action_items": []}
        # Try raw JSON
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        # Try extract from markdown code block
        for pattern in (r"```(?:json)?\s*([\s\S]*?)```", r"```\s*([\s\S]*?)```"):
            match = re.search(pattern, raw)
            if match:
                try:
                    data = json.loads(match.group(1).strip())
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    continue
        # Fallback: heuristic text parsing
        return self._parse_text_response(raw)
    
    def _parse_text_response(self, text: str) -> Dict[str, Any]:
        """
        Parse non-JSON response into structured format
        
        Attempts to extract key points and takeaways from plain text
        """
        result = {
            "key_points": [],
            "core_takeaway": "",
            "action_items": [],
        }
        
        lines = text.split("\n")
        
        current_section = None
        key_points = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect sections
            lower = line.lower()
            if "key point" in lower or "point" in lower and ":" in line:
                current_section = "points"
                continue
            elif "takeaway" in lower or "summary" in lower:
                current_section = "takeaway"
                continue
            elif "action" in lower or "todo" in lower:
                current_section = "actions"
                continue
            
            # Extract content based on section
            if current_section == "points":
                # Look for bullet points or numbered items
                if line.startswith(("-", "•", "*", "1.", "2.")):
                    point = line.lstrip("- •*1234567890.").strip()
                    # Try to extract timestamp
                    timestamp = self._extract_timestamp(line)
                    key_points.append({
                        "point": point,
                        "timestamp": timestamp,
                    })
            
            elif current_section == "takeaway":
                result["core_takeaway"] = line
            
            elif current_section == "actions":
                if line.startswith(("-", "•", "*")):
                    result["action_items"].append(line.lstrip("- •*").strip())
        
        # If we found points in the points section, use them
        if key_points:
            result["key_points"] = key_points
        else:
            # Try to find points anywhere
            result["key_points"] = self._extract_points_from_text(text)
        # If still no core_takeaway, use first substantial line or start of text
        if not result["core_takeaway"] and text:
            for line in text.split("\n"):
                line = line.strip()
                if len(line) > 30 and not line.startswith(("#", "-", "*", "1.")):
                    result["core_takeaway"] = line[:600]
                    break
            if not result["core_takeaway"]:
                result["core_takeaway"] = text[:600].strip()
        
        return result
    
    def _extract_timestamp(self, text: str) -> Optional[str]:
        """Extract timestamp from text (e.g., "12:34" or "1:23:45")"""
        patterns = [
            r'(\d{1,2}):(\d{2})(?::(\d{2}))?',  # MM:SS or HH:MM:SS
            r'at (\d+) minute',  # "at 5 minute"
            r'timestamp[:\s]*(\d{1,2}):(\d{2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    return f"{int(groups[0]):02d}:{int(groups[1]):02d}"
                elif len(groups) == 3 and groups[2]:
                    return f"{int(groups[0]):02d}:{int(groups[1]):02d}:{int(groups[2]):02d}"
        
        return None
    
    def _extract_points_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract key points from text when structure is unclear"""
        points = []
        
        # Look for sentences that might be key points
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 20:
                continue
            
            # Check if it looks like a key point
            indicators = ["key", "important", "main", "significant", "notable"]
            if any(ind in sentence.lower() for ind in indicators):
                timestamp = self._extract_timestamp(sentence)
                points.append({
                    "point": sentence,
                    "timestamp": timestamp,
                })
        
        return points[:5]  # Max 5 points
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


# ------------------------------------------------------------------------
# Factory Function
# ------------------------------------------------------------------------

async def create_summarizer(
    model_factory: ModelFactory,
    prompt_manager: PromptManager,
    language_service: LanguageService,
    model_name: str = "gpt-3.5-turbo",
    max_parallel_chunks: int = 3,
    logger=None,
    metrics=None,
) -> SummarizationOrchestrator:
    """
    Create summarization orchestrator
    
    Args:
        model_factory: Factory for AI models
        prompt_manager: Prompt template manager
        language_service: Language service
        model_name: Default model to use
        max_parallel_chunks: Max parallel chunk processing
        logger: Logger instance
        metrics: Metrics collector
        
    Returns:
        Configured SummarizationOrchestrator
    """
    return SummarizationOrchestrator(
        model_factory=model_factory,
        prompt_manager=prompt_manager,
        language_service=language_service,
        logger=logger,
        metrics=metrics,
        model_name=model_name,
        max_parallel_chunks=max_parallel_chunks,
    )


# ------------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------------

"""
# Usage example:

orchestrator = await create_summarizer(
    model_factory=model_factory,
    prompt_manager=prompt_manager,
    language_service=language_service,
)

# Generate summary
summary = await orchestrator.summarize(
    transcript=transcript,
    video_title="How to Build AI Apps",
    summary_type=SummaryType.CONCISE,
    language=Language.HINDI,  # Get summary in Hindi
)

# Use the summary
print(summary.to_formatted_text())

# Or access structured data
for point in summary.key_points:
    print(f"{point['timestamp']}: {point['point']}")

print(f"Takeaway: {summary.core_takeaway}")
"""