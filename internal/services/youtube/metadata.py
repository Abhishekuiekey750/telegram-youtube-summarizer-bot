"""
YouTube Metadata Service
Fetches video metadata with clean interface and multiple provider support

Features:
- Clean abstract interface
- Multiple provider implementations
- Automatic fallback
- Caching support
- Rate limit handling
- Comprehensive metadata model
- Error handling
"""

import abc
import asyncio
import re
from typing import Optional, Dict, Any, List, Protocol
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import json
from enum import Enum

import aiohttp
import structlog

from internal.pkg.errors import (
    NotFoundError,
    ValidationError,
    RetryableError,
    ErrorKind,
)
from internal.services.youtube.validator import YouTubeURLValidator
from internal.pkg.retry import retry_with_backoff, RetryConfig


class VideoType(Enum):
    """Type of YouTube video"""
    REGULAR = "regular"
    SHORTS = "shorts"
    LIVE = "live"
    PREMIERE = "premiere"


@dataclass
class VideoMetadata:
    """
    Comprehensive video metadata model
    """
    # Core fields
    video_id: str
    title: str
    description: str = ""
    duration: int = 0  # seconds
    
    # Channel info
    channel_id: Optional[str] = None
    channel_name: Optional[str] = None
    channel_url: Optional[str] = None
    
    # Statistics
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    comment_count: Optional[int] = None
    
    # Dates
    upload_date: Optional[str] = None  # YYYY-MM-DD
    published_at: Optional[datetime] = None
    
    # Media
    thumbnail_url: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    category: Optional[str] = None
    
    # Status
    video_type: VideoType = VideoType.REGULAR
    is_live: bool = False
    was_live: bool = False
    is_unlisted: bool = False
    is_family_safe: bool = True
    
    # Additional
    language: Optional[str] = None
    region: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    fetched_at: datetime = field(default_factory=datetime.now)
    
    @property
    def duration_formatted(self) -> str:
        """Get formatted duration (MM:SS or HH:MM:SS)"""
        if self.duration < 3600:
            minutes = self.duration // 60
            seconds = self.duration % 60
            return f"{minutes}:{seconds:02d}"
        else:
            hours = self.duration // 3600
            minutes = (self.duration % 3600) // 60
            seconds = self.duration % 60
            return f"{hours}:{minutes:02d}:{seconds:02d}"
    
    @property
    def short_title(self) -> str:
        """Get truncated title for display"""
        if len(self.title) <= 50:
            return self.title
        return self.title[:47] + "..."
    
    @property
    def is_short(self) -> bool:
        """Check if this is a YouTube Short"""
        return self.video_type == VideoType.SHORTS or self.duration < 60
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert datetime to string
        if self.published_at:
            data["published_at"] = self.published_at.isoformat()
        data["fetched_at"] = self.fetched_at.isoformat()
        data["video_type"] = self.video_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoMetadata":
        """Create from dictionary"""
        # Convert string enums back
        if "video_type" in data and isinstance(data["video_type"], str):
            try:
                data["video_type"] = VideoType(data["video_type"])
            except ValueError:
                data["video_type"] = VideoType.REGULAR
        
        # Parse datetime strings
        if "published_at" in data and isinstance(data["published_at"], str):
            try:
                data["published_at"] = datetime.fromisoformat(data["published_at"])
            except ValueError:
                data["published_at"] = None
        
        if "fetched_at" in data and isinstance(data["fetched_at"], str):
            try:
                data["fetched_at"] = datetime.fromisoformat(data["fetched_at"])
            except ValueError:
                data["fetched_at"] = datetime.now()
        
        return cls(**data)


# ------------------------------------------------------------------------
# Abstract Interface
# ------------------------------------------------------------------------

class YouTubeMetadataProvider(abc.ABC):
    """
    Abstract interface for YouTube metadata providers
    
    All concrete implementations must implement this interface,
    allowing easy swapping of providers.
    """
    
    @abc.abstractmethod
    async def get_metadata(self, video_id: str) -> VideoMetadata:
        """
        Get comprehensive metadata for a video
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            VideoMetadata object
            
        Raises:
            NotFoundError: If video not found
            ValidationError: If video ID invalid
            RetryableError: For temporary failures
        """
        pass
    
    @abc.abstractmethod
    async def get_title(self, video_id: str) -> str:
        """
        Get just the video title (minimal)
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Video title
        """
        pass
    
    @abc.abstractmethod
    async def get_duration(self, video_id: str) -> int:
        """
        Get video duration in seconds
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Duration in seconds
        """
        pass
    
    @abc.abstractmethod
    async def health_check(self) -> bool:
        """
        Check if provider is operational
        
        Returns:
            True if healthy
        """
        pass


# ------------------------------------------------------------------------
# YouTube Data API Provider (Official)
# ------------------------------------------------------------------------

class YouTubeDataAPIProvider(YouTubeMetadataProvider):
    """
    Official YouTube Data API v3 provider
    
    Requires API key but provides reliable, official metadata.
    """
    
    BASE_URL = "https://www.googleapis.com/youtube/v3"
    
    def __init__(
        self,
        api_key: str,
        session: Optional[aiohttp.ClientSession] = None,
        logger=None,
        metrics=None,
    ):
        """
        Initialize YouTube Data API provider
        
        Args:
            api_key: YouTube Data API key
            session: aiohttp session (creates one if not provided)
            logger: Structured logger
            metrics: Metrics collector
        """
        self.api_key = api_key
        self.session = session or aiohttp.ClientSession()
        self._own_session = session is None
        self.logger = logger or structlog.get_logger(__name__)
        self.metrics = metrics
        
        self.logger.info(
            "youtube_data_api.initialized",
            api_key_provided=bool(api_key),
        )
    
    async def get_metadata(self, video_id: str) -> VideoMetadata:
        """Get metadata using YouTube Data API"""
        params = {
            "part": "snippet,contentDetails,statistics",
            "id": video_id,
            "key": self.api_key,
        }
        
        try:
            async with self.session.get(
                f"{self.BASE_URL}/videos",
                params=params,
                timeout=10,
            ) as response:
                if response.status == 403:
                    # API key issue
                    raise RetryableError(
                        "YouTube API key invalid or quota exceeded",
                        kind=ErrorKind.EXTERNAL_API,
                        retry_after=3600,  # Wait an hour
                    )
                
                if response.status == 404:
                    raise NotFoundError(
                        f"Video {video_id} not found",
                        kind=ErrorKind.NOT_FOUND,
                    )
                
                if response.status != 200:
                    raise RetryableError(
                        f"YouTube API returned {response.status}",
                        kind=ErrorKind.EXTERNAL_API,
                    )
                
                data = await response.json()
                
                if not data.get("items"):
                    raise NotFoundError(
                        f"Video {video_id} not found",
                        kind=ErrorKind.NOT_FOUND,
                    )
                
                item = data["items"][0]
                return self._parse_api_response(video_id, item)
                
        except asyncio.TimeoutError as e:
            raise RetryableError(
                "YouTube API timeout",
                kind=ErrorKind.EXTERNAL_API,
                original_error=e,
            ) from e
        
        except aiohttp.ClientError as e:
            raise RetryableError(
                f"YouTube API connection error: {str(e)}",
                kind=ErrorKind.EXTERNAL_API,
                original_error=e,
            ) from e
    
    async def get_title(self, video_id: str) -> str:
        """Get just the title (minimal API call)"""
        params = {
            "part": "snippet",
            "id": video_id,
            "key": self.api_key,
            "fields": "items/snippet/title",
        }
        
        try:
            async with self.session.get(
                f"{self.BASE_URL}/videos",
                params=params,
                timeout=5,
            ) as response:
                if response.status != 200:
                    raise RetryableError(f"API returned {response.status}")
                
                data = await response.json()
                if not data.get("items"):
                    raise NotFoundError(f"Video {video_id} not found")
                
                return data["items"][0]["snippet"]["title"]
                
        except Exception as e:
            # Fall back to full metadata
            metadata = await self.get_metadata(video_id)
            return metadata.title
    
    async def get_duration(self, video_id: str) -> int:
        """Get duration in seconds"""
        params = {
            "part": "contentDetails",
            "id": video_id,
            "key": self.api_key,
            "fields": "items/contentDetails/duration",
        }
        
        try:
            async with self.session.get(
                f"{self.BASE_URL}/videos",
                params=params,
                timeout=5,
            ) as response:
                if response.status != 200:
                    raise RetryableError(f"API returned {response.status}")
                
                data = await response.json()
                if not data.get("items"):
                    raise NotFoundError(f"Video {video_id} not found")
                
                duration_str = data["items"][0]["contentDetails"]["duration"]
                return self._parse_iso_duration(duration_str)
                
        except Exception as e:
            # Fall back to full metadata
            metadata = await self.get_metadata(video_id)
            return metadata.duration
    
    async def health_check(self) -> bool:
        """Check if API is accessible"""
        try:
            # Try to fetch a known video
            await self.get_title("dQw4w9WgXcQ")  # Rick Astley - never gonna give you up
            return True
        except Exception:
            return False
    
    def _parse_api_response(self, video_id: str, item: Dict) -> VideoMetadata:
        """Parse YouTube API response into VideoMetadata"""
        snippet = item.get("snippet", {})
        content = item.get("contentDetails", {})
        statistics = item.get("statistics", {})
        
        # Parse duration
        duration = self._parse_iso_duration(content.get("duration", "PT0S"))
        
        # Determine video type
        video_type = VideoType.REGULAR
        if content.get("definition") == "shorts" or duration < 60:
            video_type = VideoType.SHORTS
        if content.get("liveContent") == "live":
            video_type = VideoType.LIVE
        
        # Parse upload date
        published_at = None
        if snippet.get("publishedAt"):
            try:
                published_at = datetime.fromisoformat(
                    snippet["publishedAt"].replace("Z", "+00:00")
                )
            except ValueError:
                pass
        
        return VideoMetadata(
            video_id=video_id,
            title=snippet.get("title", "Unknown Title"),
            description=snippet.get("description", ""),
            duration=duration,
            channel_id=snippet.get("channelId"),
            channel_name=snippet.get("channelTitle"),
            channel_url=f"https://youtube.com/channel/{snippet.get('channelId')}",
            view_count=int(statistics.get("viewCount", 0)),
            like_count=int(statistics.get("likeCount", 0)),
            comment_count=int(statistics.get("commentCount", 0)),
            published_at=published_at,
            thumbnail_url=snippet.get("thumbnails", {}).get("high", {}).get("url"),
            tags=snippet.get("tags", []),
            category=snippet.get("categoryId"),
            video_type=video_type,
            is_live=video_type == VideoType.LIVE,
            is_unlisted=snippet.get("privacyStatus") == "unlisted",
        )
    
    def _parse_iso_duration(self, duration: str) -> int:
        """
        Parse ISO 8601 duration to seconds
        PT1H30M15S -> 5415 seconds
        """
        import re
        
        match = re.match(
            r'^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$',
            duration
        )
        
        if not match:
            return 0
        
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        
        return hours * 3600 + minutes * 60 + seconds
    
    async def close(self):
        """Close session if we created it"""
        if self._own_session and self.session:
            await self.session.close()


# ------------------------------------------------------------------------
# youtube-dl Provider (Fallback)
# ------------------------------------------------------------------------

class YouTubeDLProvider(YouTubeMetadataProvider):
    """
    youtube-dl/yt-dlp based provider
    
    Works without API key but may be slower and can break with YouTube changes.
    Used as fallback when API is unavailable.
    """
    
    def __init__(
        self,
        yt_dlp_path: str = "yt-dlp",
        timeout: int = 30,
        logger=None,
        metrics=None,
    ):
        """
        Initialize youtube-dl provider
        
        Args:
            yt_dlp_path: Path to yt-dlp executable
            timeout: Timeout in seconds
            logger: Structured logger
            metrics: Metrics collector
        """
        self.yt_dlp_path = yt_dlp_path
        self.timeout = timeout
        self.logger = logger or structlog.get_logger(__name__)
        self.metrics = metrics
        
        self.logger.info(
            "youtube_dl_provider.initialized",
            path=yt_dlp_path,
            timeout=timeout,
        )
    
    async def get_metadata(self, video_id: str) -> VideoMetadata:
        """Get metadata using yt-dlp"""
        import asyncio.subprocess as asp
        
        url = f"https://youtu.be/{video_id}"
        
        cmd = [
            self.yt_dlp_path,
            "--dump-json",
            "--no-playlist",
            "--skip-download",
            url,
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asp.PIPE,
                stderr=asp.PIPE,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                raise RetryableError(
                    "yt-dlp timeout",
                    kind=ErrorKind.EXTERNAL_API,
                )
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                if "Video unavailable" in error_msg:
                    raise NotFoundError(
                        f"Video {video_id} not found",
                        kind=ErrorKind.NOT_FOUND,
                    )
                raise RetryableError(
                    f"yt-dlp failed: {error_msg}",
                    kind=ErrorKind.EXTERNAL_API,
                )
            
            data = json.loads(stdout.decode())
            return self._parse_yt_dlp_output(video_id, data)
            
        except FileNotFoundError as e:
            raise RuntimeError(
                f"yt-dlp not found at {self.yt_dlp_path}. Please install.",
            ) from e
        
        except Exception as e:
            if not isinstance(e, (NotFoundError, RetryableError)):
                raise RetryableError(
                    f"yt-dlp error: {str(e)}",
                    kind=ErrorKind.EXTERNAL_API,
                    original_error=e,
                ) from e
            raise
    
    async def get_title(self, video_id: str) -> str:
        """Get title using minimal yt-dlp call"""
        metadata = await self.get_metadata(video_id)
        return metadata.title
    
    async def get_duration(self, video_id: str) -> int:
        """Get duration using minimal yt-dlp call"""
        metadata = await self.get_metadata(video_id)
        return metadata.duration
    
    async def health_check(self) -> bool:
        """Check if yt-dlp is installed"""
        import shutil
        return shutil.which(self.yt_dlp_path) is not None
    
    def _parse_yt_dlp_output(self, video_id: str, data: Dict) -> VideoMetadata:
        """Parse yt-dlp JSON output"""
        # Determine video type
        video_type = VideoType.REGULAR
        if data.get("duration", 0) < 60:
            video_type = VideoType.SHORTS
        if data.get("is_live", False):
            video_type = VideoType.LIVE
        
        # Parse upload date
        upload_date = None
        if data.get("upload_date"):
            upload_str = data["upload_date"]
            try:
                upload_date = datetime.strptime(upload_str, "%Y%m%d").date()
            except ValueError:
                pass
        
        return VideoMetadata(
            video_id=video_id,
            title=data.get("title", "Unknown Title"),
            description=data.get("description", ""),
            duration=data.get("duration", 0),
            channel_id=data.get("channel_id"),
            channel_name=data.get("channel", data.get("uploader")),
            channel_url=data.get("channel_url"),
            view_count=data.get("view_count"),
            like_count=data.get("like_count"),
            comment_count=data.get("comment_count"),
            upload_date=upload_date.isoformat() if upload_date else None,
            thumbnail_url=data.get("thumbnail"),
            tags=data.get("tags", []),
            category=data.get("categories", [None])[0],
            video_type=video_type,
            is_live=data.get("is_live", False),
            was_live=data.get("was_live", False),
            language=data.get("language"),
        )


# ------------------------------------------------------------------------
# Main Service with Fallback
# ------------------------------------------------------------------------

class YouTubeMetadataService:
    """
    Main YouTube metadata service with multiple providers and fallback
    
    Uses primary provider (YouTube Data API) with fallback to secondary (yt-dlp)
    """
    
    def __init__(
        self,
        primary_provider: YouTubeMetadataProvider,
        fallback_provider: Optional[YouTubeMetadataProvider] = None,
        cache=None,
        validator: Optional[YouTubeURLValidator] = None,
        logger=None,
        metrics=None,
        cache_ttl: int = 3600,  # 1 hour
    ):
        """
        Initialize metadata service
        
        Args:
            primary_provider: Primary metadata provider
            fallback_provider: Fallback provider (if primary fails)
            cache: Cache client (Redis or memory)
            validator: URL validator
            logger: Structured logger
            metrics: Metrics collector
            cache_ttl: Cache TTL in seconds
        """
        self.primary = primary_provider
        self.fallback = fallback_provider
        self.cache = cache
        self.validator = validator or YouTubeURLValidator()
        self.logger = logger or structlog.get_logger(__name__)
        self.metrics = metrics
        self.cache_ttl = cache_ttl
        
        self.logger.info(
            "youtube_metadata_service.initialized",
            has_primary=primary_provider is not None,
            has_fallback=fallback_provider is not None,
            cache_ttl=cache_ttl,
        )
    
    async def get_metadata(
        self,
        video_id: str,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> VideoMetadata:
        """
        Get video metadata with caching and fallback
        
        Args:
            video_id: YouTube video ID
            use_cache: Whether to check cache first
            force_refresh: Skip cache and force fetch
            
        Returns:
            VideoMetadata object
            
        Raises:
            NotFoundError: If video not found
            ValidationError: If video ID invalid
        """
        # Validate video ID
        if not self.validator._validate_video_id(video_id):
            raise ValidationError(
                f"Invalid video ID: {video_id}",
                kind=ErrorKind.VALIDATION,
            )
        
        # Check cache
        if use_cache and not force_refresh and self.cache:
            cached = await self._get_cached(video_id)
            if cached:
                self.logger.debug(
                    "metadata.cache_hit",
                    video_id=video_id,
                )
                if self.metrics:
                    self.metrics.increment("metadata.cache_hit")
                return cached
        
        # Try primary provider
        try:
            metadata = await self.primary.get_metadata(video_id)
            
            if self.metrics:
                self.metrics.increment(
                    "metadata.fetched",
                    tags={"provider": "primary"},
                )
            
        except Exception as e:
            self.logger.warning(
                "metadata.primary_failed",
                video_id=video_id,
                error=str(e),
            )
            
            if self.metrics:
                self.metrics.increment(
                    "metadata.primary_failed",
                    tags={"error": e.__class__.__name__},
                )
            
            # Try fallback if available
            if self.fallback:
                try:
                    metadata = await self.fallback.get_metadata(video_id)
                    
                    if self.metrics:
                        self.metrics.increment(
                            "metadata.fetched",
                            tags={"provider": "fallback"},
                        )
                    
                except Exception as fallback_error:
                    self.logger.error(
                        "metadata.both_providers_failed",
                        video_id=video_id,
                        primary_error=str(e),
                        fallback_error=str(fallback_error),
                    )
                    
                    if self.metrics:
                        self.metrics.increment("metadata.all_failed")
                    
                    raise NotFoundError(
                        f"No metadata found for video {video_id}",
                        kind=ErrorKind.NOT_FOUND,
                    ) from fallback_error
            else:
                # No fallback, re-raise
                raise
        
        # Cache result
        if use_cache and self.cache:
            await self._cache_metadata(metadata)
        
        return metadata
    
    async def get_title(self, video_id: str) -> str:
        """Get just the video title"""
        metadata = await self.get_metadata(video_id)
        return metadata.title
    
    async def get_duration(self, video_id: str) -> int:
        """Get video duration in seconds"""
        metadata = await self.get_metadata(video_id)
        return metadata.duration
    
    async def get_basic_info(self, video_id: str) -> Dict[str, Any]:
        """
        Get basic info (minimal) for quick display
        
        Returns:
            Dict with title, duration, channel
        """
        metadata = await self.get_metadata(video_id)
        return {
            "video_id": metadata.video_id,
            "title": metadata.title,
            "duration": metadata.duration,
            "duration_formatted": metadata.duration_formatted,
            "channel_name": metadata.channel_name,
            "thumbnail": metadata.thumbnail_url,
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all providers"""
        primary_healthy = await self.primary.health_check() if self.primary else False
        fallback_healthy = await self.fallback.health_check() if self.fallback else False
        
        return {
            "primary_healthy": primary_healthy,
            "fallback_healthy": fallback_healthy,
            "overall": primary_healthy or fallback_healthy,
        }
    
    async def _get_cached(self, video_id: str) -> Optional[VideoMetadata]:
        """Get metadata from cache"""
        if not self.cache:
            return None
        
        cache_key = f"metadata:{video_id}"
        cached = await self.cache.get(cache_key)
        
        if cached:
            try:
                data = json.loads(cached)
                return VideoMetadata.from_dict(data)
            except Exception as e:
                self.logger.warning(
                    "metadata.cache_corrupted",
                    video_id=video_id,
                    error=str(e),
                )
        
        return None
    
    async def _cache_metadata(self, metadata: VideoMetadata) -> None:
        """Cache metadata"""
        if not self.cache:
            return
        
        try:
            cache_key = f"metadata:{metadata.video_id}"
            data = json.dumps(metadata.to_dict())
            await self.cache.setex(cache_key, self.cache_ttl, data)
        except Exception as e:
            self.logger.warning(
                "metadata.cache_failed",
                video_id=metadata.video_id,
                error=str(e),
            )
    
    async def close(self):
        """Close provider sessions"""
        if hasattr(self.primary, 'close'):
            await self.primary.close()
        if self.fallback and hasattr(self.fallback, 'close'):
            await self.fallback.close()


# ------------------------------------------------------------------------
# Factory Functions
# ------------------------------------------------------------------------

async def create_metadata_service(
    api_key: Optional[str] = None,
    use_yt_dlp: bool = True,
    cache=None,
    logger=None,
    metrics=None,
) -> YouTubeMetadataService:
    """
    Create metadata service with appropriate providers
    
    Args:
        api_key: YouTube Data API key (optional)
        use_yt_dlp: Whether to use yt-dlp as fallback
        cache: Cache client
        logger: Logger instance
        metrics: Metrics collector
        
    Returns:
        Configured YouTubeMetadataService
    """
    # Create providers
    primary = None
    fallback = None
    
    if api_key:
        primary = YouTubeDataAPIProvider(
            api_key=api_key,
            logger=logger,
            metrics=metrics,
        )
    
    if use_yt_dlp:
        fallback = YouTubeDLProvider(
            logger=logger,
            metrics=metrics,
        )
    
    # If no primary, use fallback as primary
    if not primary and fallback:
        primary = fallback
        fallback = None
    
    return YouTubeMetadataService(
        primary_provider=primary,
        fallback_provider=fallback,
        cache=cache,
        logger=logger,
        metrics=metrics,
    )


# ------------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------------

"""
# Usage examples:

# Create service (auto-configure)
service = await create_metadata_service(
    api_key="YOUR_API_KEY",  # Optional
    use_yt_dlp=True,  # Fallback
)

# Get full metadata
metadata = await service.get_metadata("dQw4w9WgXcQ")
print(f"Title: {metadata.title}")
print(f"Channel: {metadata.channel_name}")
print(f"Duration: {metadata.duration_formatted}")

# Get just title (cached)
title = await service.get_title("dQw4w9WgXcQ")

# Get basic info
info = await service.get_basic_info("dQw4w9WgXcQ")

# Health check
health = await service.health_check()
print(f"Service healthy: {health['overall']}")

# Clean up
await service.close()
"""

# ------------------------------------------------------------------------
# Error Handling Examples
# ------------------------------------------------------------------------

"""
try:
    metadata = await service.get_metadata("invalid_id")
except ValidationError:
    print("Invalid video ID")

try:
    metadata = await service.get_metadata("nonexistent_video")
except NotFoundError:
    print("Video not found")

# Temporary failures are wrapped in RetryableError
# for the recovery middleware to handle
"""