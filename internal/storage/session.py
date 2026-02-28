"""
Session Manager
Manages user sessions with Redis backend and memory fallback

Features:
- Redis backend for production
- In-memory fallback for development
- Automatic session creation
- TTL management
- Conversation history tracking
- Language preferences
- Video context storage
- Metrics collection
- Batch operations
"""

import json
import pickle
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import hashlib
import asyncio
from enum import Enum

import structlog
import redis.asyncio as redis
from redis.exceptions import RedisError

from internal.domain.value_objects import Language
from internal.pkg.errors import SessionError, NotFoundError
from internal.pkg.metrics import MetricsCollector


class SessionBackend(Enum):
    """Supported session backends"""
    REDIS = "redis"
    MEMORY = "memory"
    HYBRID = "hybrid"  # Redis with memory fallback


@dataclass
class ConversationTurn:
    """Single conversation turn"""
    question: str
    answer: str
    video_id: str
    timestamp: datetime
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "video_id": self.video_id,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTurn":
        return cls(
            question=data["question"],
            answer=data["answer"],
            video_id=data["video_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class UserSession:
    """User session data"""
    user_id: int
    chat_id: int
    language: Language = Language.ENGLISH
    current_video_id: Optional[str] = None
    current_video_title: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Internal flags
    _modified: bool = False
    _new: bool = False
    
    @property
    def modified(self) -> bool:
        """Whether session has been modified since last save"""
        return self._modified

    @modified.setter
    def modified(self, value: bool) -> None:
        self._modified = value

    @property
    def is_expired(self) -> bool:
        """Check if session is expired"""
        if not self.expires_at:
            return False
        return datetime.now() > self.expires_at
    
    @property
    def question_count(self) -> int:
        """Get number of questions asked"""
        return len(self.conversation_history)
    
    @property
    def videos_processed(self) -> List[str]:
        """Get list of videos processed"""
        videos = set()
        for turn in self.conversation_history:
            if turn.video_id:
                videos.add(turn.video_id)
        if self.current_video_id:
            videos.add(self.current_video_id)
        return list(videos)
    
    def add_conversation_turn(self, turn: ConversationTurn):
        """Add a conversation turn"""
        self.conversation_history.append(turn)
        self._modified = True
        
        # Limit history size (keep last 50)
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
    
    def set_language(self, language: Language):
        """Set user language"""
        if self.language != language:
            self.language = language
            self._modified = True
    
    def set_current_video(self, video_id: str, title: Optional[str] = None):
        """Set current video"""
        if self.current_video_id != video_id:
            self.current_video_id = video_id
            self.current_video_title = title
            self._modified = True
    
    def clear_current_video(self):
        """Clear current video"""
        if self.current_video_id:
            self.current_video_id = None
            self.current_video_title = None
            self._modified = True
    
    def touch(self):
        """Update timestamp"""
        self.updated_at = datetime.now()
        self._modified = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "user_id": self.user_id,
            "chat_id": self.chat_id,
            "language": self.language.code,
            "current_video_id": self.current_video_id,
            "current_video_title": self.current_video_title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "conversation_history": [t.to_dict() for t in self.conversation_history],
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserSession":
        """Create from dictionary"""
        session = cls(
            user_id=data["user_id"],
            chat_id=data["chat_id"],
            language=Language.from_code(data.get("language", "en")),
            current_video_id=data.get("current_video_id"),
            current_video_title=data.get("current_video_title"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            conversation_history=[
                ConversationTurn.from_dict(t) for t in data.get("conversation_history", [])
            ],
            metadata=data.get("metadata", {}),
        )
        session._modified = False
        return session
    
    def to_redis_hash(self) -> Dict[str, Union[str, bytes]]:
        """Convert to Redis hash format"""
        return {
            "data": pickle.dumps(self),
            "user_id": str(self.user_id),
            "chat_id": str(self.chat_id),
            "language": self.language.code,
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_redis_hash(cls, data: Dict[str, bytes]) -> "UserSession":
        """Create from Redis hash"""
        return pickle.loads(data[b"data"])


class MemorySessionStore:
    """In-memory session store for development/testing"""
    
    def __init__(self, default_ttl: int = 86400):  # 24 hours
        self._sessions: Dict[int, UserSession] = {}
        self.default_ttl = default_ttl
    
    async def get(self, user_id: int) -> Optional[UserSession]:
        """Get session by user ID"""
        session = self._sessions.get(user_id)
        if session and session.is_expired:
            await self.delete(user_id)
            return None
        return session
    
    async def set(self, user_id: int, session: UserSession, ttl: Optional[int] = None):
        """Store session"""
        if ttl:
            session.expires_at = datetime.now() + timedelta(seconds=ttl)
        self._sessions[user_id] = session
    
    async def delete(self, user_id: int):
        """Delete session"""
        self._sessions.pop(user_id, None)
    
    async def exists(self, user_id: int) -> bool:
        """Check if session exists"""
        return user_id in self._sessions
    
    async def clear(self):
        """Clear all sessions"""
        self._sessions.clear()
    
    async def size(self) -> int:
        """Get number of sessions"""
        return len(self._sessions)


class SessionManager:
    """
    Manages user sessions with Redis backend and memory fallback
    
    Features:
    - Redis for production (fast, distributed)
    - Memory fallback for development
    - Automatic TTL management
    - Conversation history
    - Language preferences
    - Video context
    - Metrics tracking
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: int = 86400,  # 24 hours
        backend: SessionBackend = SessionBackend.HYBRID,
        key_prefix: str = "session",
        logger=None,
        metrics=None,
    ):
        """
        Initialize session manager
        
        Args:
            redis_url: Redis connection URL (None for memory-only)
            default_ttl: Default session TTL in seconds
            backend: Backend type
            key_prefix: Prefix for Redis keys
            logger: Structured logger
            metrics: Metrics collector
        """
        self.default_ttl = default_ttl
        self.backend = backend
        self.key_prefix = key_prefix
        self.logger = logger or structlog.get_logger(__name__)
        self.metrics = metrics or MetricsCollector("session_manager")
        
        # Initialize backends
        self.redis: Optional[redis.Redis] = None
        self.memory = MemorySessionStore(default_ttl)
        
        # Connect to Redis if URL provided
        if redis_url and backend in [SessionBackend.REDIS, SessionBackend.HYBRID]:
            self._init_redis(redis_url)
        
        self.logger.info(
            "session_manager.initialized",
            backend=backend.value,
            default_ttl=default_ttl,
            redis_connected=self.redis is not None,
        )
    
    def _init_redis(self, redis_url: str):
        """Initialize Redis connection"""
        try:
            self.redis = redis.from_url(
                redis_url,
                decode_responses=False,  # We'll handle decoding
                socket_timeout=2,
                socket_connect_timeout=2,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            self.logger.info("session_manager.redis_connected")
        except Exception as e:
            self.logger.error("session_manager.redis_connection_failed", error=str(e))
            self.redis = None
    
    async def get_session(
        self,
        user_id: int,
        create_if_missing: bool = False,
        chat_id: Optional[int] = None,
    ) -> Optional[UserSession]:
        """
        Get session for user
        
        Args:
            user_id: Telegram user ID
            create_if_missing: Create new session if not found
            chat_id: Required if creating new session
            
        Returns:
            User session or None
        """
        # Try Redis first
        session = await self._get_from_redis(user_id)
        
        # Fallback to memory
        if not session and self.backend in [SessionBackend.MEMORY, SessionBackend.HYBRID]:
            session = await self.memory.get(user_id)
            if session:
                self.metrics.increment("sources.memory")
        
        # Create if needed
        if not session and create_if_missing:
            if not chat_id:
                raise ValueError("chat_id required when creating new session")
            
            session = await self.create_session(user_id, chat_id)
            self.metrics.increment("sessions.created")
        else:
            self.metrics.increment("sources.cache" if session else "sources.miss")
        
        return session
    
    async def _get_from_redis(self, user_id: int) -> Optional[UserSession]:
        """Get session from Redis"""
        if not self.redis:
            return None
        
        try:
            key = f"{self.key_prefix}:{user_id}"
            data = await self.redis.hgetall(key)
            
            if data:
                session = UserSession.from_redis_hash(data)
                
                # Check expiry
                if session.is_expired:
                    await self.delete_session(user_id)
                    return None
                
                self.metrics.increment("sources.redis")
                return session
                
        except RedisError as e:
            self.logger.warning(
                "session_manager.redis_get_failed",
                user_id=user_id,
                error=str(e),
            )
            self.metrics.increment("errors.redis")
        
        return None
    
    async def save_session(self, session: UserSession) -> bool:
        """
        Save session to storage
        
        Args:
            session: User session to save
            
        Returns:
            True if saved successfully
        """
        session.touch()
        
        success = False
        
        # Save to Redis
        if self.redis:
            success = await self._save_to_redis(session) or success
        
        # Save to memory
        if self.backend in [SessionBackend.MEMORY, SessionBackend.HYBRID]:
            await self.memory.set(session.user_id, session, self.default_ttl)
            success = True
        
        if success:
            session._modified = False
            self.metrics.increment("sessions.saved")
        
        return success
    
    async def _save_to_redis(self, session: UserSession) -> bool:
        """Save session to Redis"""
        if not self.redis:
            return False
        
        try:
            key = f"{self.key_prefix}:{session.user_id}"
            
            # Set expiry
            if not session.expires_at:
                session.expires_at = datetime.now() + timedelta(seconds=self.default_ttl)
            
            # Save as hash
            await self.redis.hset(key, mapping=session.to_redis_hash())
            await self.redis.expireat(key, session.expires_at)
            
            return True
            
        except RedisError as e:
            self.logger.warning(
                "session_manager.redis_save_failed",
                user_id=session.user_id,
                error=str(e),
            )
            self.metrics.increment("errors.redis")
            return False
    
    async def create_session(
        self,
        user_id: int,
        chat_id: int,
        language: Language = Language.ENGLISH,
    ) -> UserSession:
        """
        Create a new user session
        
        Args:
            user_id: Telegram user ID
            chat_id: Telegram chat ID
            language: Preferred language
            
        Returns:
            New user session
        """
        session = UserSession(
            user_id=user_id,
            chat_id=chat_id,
            language=language,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=self.default_ttl),
        )
        session._new = True
        session._modified = True
        
        await self.save_session(session)
        
        self.logger.info(
            "session_manager.created",
            user_id=user_id,
            chat_id=chat_id,
            language=language.code,
        )
        
        return session
    
    async def delete_session(self, user_id: int) -> bool:
        """
        Delete user session
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            True if deleted
        """
        deleted = False
        
        # Delete from Redis
        if self.redis:
            try:
                key = f"{self.key_prefix}:{user_id}"
                result = await self.redis.delete(key)
                deleted = result > 0 or deleted
            except RedisError as e:
                self.logger.warning(
                    "session_manager.redis_delete_failed",
                    user_id=user_id,
                    error=str(e),
                )
        
        # Delete from memory
        await self.memory.delete(user_id)
        deleted = True
        
        if deleted:
            self.logger.debug("session_manager.deleted", user_id=user_id)
            self.metrics.increment("sessions.deleted")
        
        return deleted
    
    async def update_session(
        self,
        user_id: int,
        **updates,
    ) -> Optional[UserSession]:
        """
        Update session fields
        
        Args:
            user_id: Telegram user ID
            **updates: Fields to update
            
        Returns:
            Updated session or None
        """
        session = await self.get_session(user_id)
        if not session:
            return None
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(session, key):
                setattr(session, key, value)
                session._modified = True
        
        await self.save_session(session)
        
        return session
    
    async def add_conversation_turn(
        self,
        user_id: int,
        question: str,
        answer: str,
        video_id: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[UserSession]:
        """
        Add a conversation turn to history
        
        Args:
            user_id: Telegram user ID
            question: User's question
            answer: Bot's answer
            video_id: Current video ID
            confidence: Answer confidence
            metadata: Additional metadata
            
        Returns:
            Updated session
        """
        session = await self.get_session(user_id)
        if not session:
            return None
        
        turn = ConversationTurn(
            question=question,
            answer=answer,
            video_id=video_id,
            timestamp=datetime.now(),
            confidence=confidence,
            metadata=metadata or {},
        )
        
        session.add_conversation_turn(turn)
        await self.save_session(session)
        
        self.metrics.increment("conversation.turns_added")
        
        return session
    
    async def get_conversation_history(
        self,
        user_id: int,
        limit: int = 10,
        video_id: Optional[str] = None,
    ) -> List[ConversationTurn]:
        """
        Get conversation history
        
        Args:
            user_id: Telegram user ID
            limit: Maximum number of turns
            video_id: Filter by video ID
            
        Returns:
            List of conversation turns
        """
        session = await self.get_session(user_id)
        if not session:
            return []
        
        history = session.conversation_history
        
        if video_id:
            history = [t for t in history if t.video_id == video_id]
        
        return history[-limit:]
    
    async def set_language(self, user_id: int, language: Language) -> Optional[UserSession]:
        """Set user's language preference"""
        return await self.update_session(user_id, language=language)
    
    async def set_current_video(
        self,
        user_id: int,
        video_id: str,
        title: Optional[str] = None,
    ) -> Optional[UserSession]:
        """Set current video for user"""
        session = await self.get_session(user_id)
        if not session:
            return None
        
        session.set_current_video(video_id, title)
        await self.save_session(session)
        
        return session
    
    async def clear_current_video(self, user_id: int) -> Optional[UserSession]:
        """Clear current video"""
        session = await self.get_session(user_id)
        if not session:
            return None
        
        session.clear_current_video()
        await self.save_session(session)
        
        return session
    
    async def exists(self, user_id: int) -> bool:
        """Check if session exists"""
        return await self.get_session(user_id) is not None
    
    async def touch(self, user_id: int) -> bool:
        """Update session timestamp"""
        session = await self.get_session(user_id)
        if not session:
            return False
        
        session.touch()
        await self.save_session(session)
        return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        stats = {
            "memory_sessions": await self.memory.size(),
            "default_ttl": self.default_ttl,
            "backend": self.backend.value,
        }
        
        # Get Redis info
        if self.redis:
            try:
                info = await self.redis.info("stats")
                keys = await self.redis.keys(f"{self.key_prefix}:*")
                stats.update({
                    "redis_connected": True,
                    "redis_sessions": len(keys),
                    "redis_memory_used": info.get("used_memory_human", "unknown"),
                })
            except Exception as e:
                stats["redis_error"] = str(e)
        else:
            stats["redis_connected"] = False
        
        return stats
    
    async def cleanup_expired(self) -> int:
        """Clean up expired sessions"""
        cleaned = 0
        
        # Redis handles TTL automatically
        # Clean memory store
        if self.backend in [SessionBackend.MEMORY, SessionBackend.HYBRID]:
            # Memory store cleanup is done on access
            pass
        
        self.logger.info("session_manager.cleanup_completed", cleaned=cleaned)
        return cleaned
    
    async def initialize(self) -> None:
        """Initialize session store (no-op for sync backends)"""
        pass
    
    async def health_check(self) -> bool:
        """Check session store connectivity"""
        try:
            if self.redis:
                await self.redis.ping()
            return True
        except Exception:
            return self.backend == SessionBackend.MEMORY
    
    async def close(self):
        """Close connections"""
        if self.redis:
            await self.redis.close()
            self.logger.debug("session_manager.redis_closed")


# ------------------------------------------------------------------------
# Factory Functions
# ------------------------------------------------------------------------

def create_session_manager(
    redis_url: Optional[str] = None,
    default_ttl: int = 86400,
    backend: str = "hybrid",
    key_prefix: str = "session",
    logger=None,
    metrics=None,
) -> SessionManager:
    """
    Create session manager with configuration
    
    Args:
        redis_url: Redis connection URL
        default_ttl: Default session TTL in seconds
        backend: Backend type (redis, memory, hybrid)
        key_prefix: Prefix for Redis keys
        logger: Logger instance
        metrics: Metrics collector
        
    Returns:
        Configured SessionManager
    """
    backend_enum = SessionBackend(backend)
    
    return SessionManager(
        redis_url=redis_url,
        default_ttl=default_ttl,
        backend=backend_enum,
        key_prefix=key_prefix,
        logger=logger,
        metrics=metrics,
    )


def create_memory_session_manager(
    default_ttl: int = 86400,
    logger=None,
    metrics=None,
) -> SessionManager:
    """Create memory-only session manager (for development)"""
    return SessionManager(
        redis_url=None,
        default_ttl=default_ttl,
        backend=SessionBackend.MEMORY,
        logger=logger,
        metrics=metrics,
    )


# ------------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------------

"""
# Usage examples:

# Production with Redis
manager = create_session_manager(
    redis_url="redis://localhost:6379/0",
    default_ttl=86400,  # 24 hours
)

# Development with memory
manager = create_memory_session_manager()

# Get or create session
session = await manager.get_session(
    user_id=12345,
    create_if_missing=True,
    chat_id=-100123456,
)

# Update session
await manager.set_language(12345, Language.HINDI)
await manager.set_current_video(12345, "abc123", "My Video")

# Add conversation turn
await manager.add_conversation_turn(
    user_id=12345,
    question="What is pricing?",
    answer="It's $49/month",
    video_id="abc123",
    confidence=0.95,
)

# Get conversation history
history = await manager.get_conversation_history(
    user_id=12345,
    limit=5,
    video_id="abc123",
)

# Get stats
stats = await manager.get_stats()
print(f"Active sessions: {stats['memory_sessions']}")

# Clean up
await manager.close()
"""

# Alias for compatibility
SessionStore = SessionManager


"""
docker-compose.yml:
version: '3'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

volumes:
  redis_data:

Environment:
REDIS_URL=redis://localhost:6379/0
SESSION_TTL=86400
"""