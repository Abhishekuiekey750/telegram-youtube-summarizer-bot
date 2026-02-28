"""
YouTube Service - facade combining transcript and metadata
"""

from typing import Optional, Dict, Any, List, TYPE_CHECKING

from internal.services.youtube.transcript import Transcript, TranscriptFetcher
from internal.services.youtube.validator import YouTubeURLValidator
from internal.pkg.errors import TranscriptError, NotFoundError

if TYPE_CHECKING:
    pass


class YouTubeService:
    """
    Facade for YouTube transcript and metadata.
    """
    
    def __init__(
        self,
        config: Any = None,
        cache_manager=None,
        logger=None,
        metrics=None,
        http_session=None,
        **kwargs,
    ):
        self._config = config or type('Config', (), {'api_key': ''})()
        self._cache = cache_manager
        self._logger = logger
        self._metrics = metrics
        self._http_session = http_session
        
        api_key = getattr(self._config, 'api_key', '') or ''
        validator = YouTubeURLValidator()
        self._transcript_fetcher = TranscriptFetcher(
            validator=validator,
            cache=self._cache,
            logger=self._logger,
            metrics=self._metrics,
        )
        self._metadata_service = None
        if api_key:
            try:
                from internal.services.youtube.metadata import (
                    YouTubeMetadataService,
                    YouTubeDataAPIProvider,
                )
                metadata_provider = YouTubeDataAPIProvider(
                    api_key=api_key,
                    session=http_session,
                    logger=logger,
                    metrics=metrics,
                )
                self._metadata_service = YouTubeMetadataService(primary=metadata_provider, cache=self._cache)
            except Exception:
                pass
    
    async def get_transcript(
        self,
        video_id: str,
        language_codes: Optional[List[str]] = None,
    ) -> Transcript:
        """Fetch transcript for video"""
        try:
            return await self._transcript_fetcher.fetch(
                video_id=video_id,
                preferred_languages=language_codes or ["en"],
            )
        except Exception as e:
            raise TranscriptError(str(e), original_error=e) from e
    
    async def get_video_metadata(self, video_id: str) -> Dict[str, Any]:
        """Get video metadata"""
        try:
            if self._metadata_service:
                metadata = await self._metadata_service.get_metadata(video_id)
                return {
                    "title": metadata.title,
                    "duration": metadata.duration or 0,
                    "channel_name": metadata.channel_name,
                    "thumbnail_url": metadata.thumbnail_url,
                }
            return {"title": "Unknown", "duration": 0, "channel_name": None, "thumbnail_url": None}
        except Exception as e:
            raise NotFoundError(f"Video not found: {video_id}", original_error=e) from e
    
    async def health_check(self) -> bool:
        """Check service health"""
        try:
            if self._metadata_service:
                result = await self._metadata_service.health_check()
                return result.get("overall", False)
            return True
        except Exception:
            return False
