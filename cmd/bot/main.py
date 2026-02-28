#!/usr/bin/env python3
"""
Telegram YouTube Summarizer Bot - Main Entry Point
Production-ready async bot with clean architecture
"""
import sys
# Unbuffered stdout so progress shows immediately (e.g. in Windows terminal)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

print("Starting Telegram YouTube Summarizer Bot...", flush=True)

# Workaround for Windows: platform.system() can hang or raise when WMI is slow/broken.
# aiohttp calls it on import. Patch before any aiohttp import so we never call the real one.
import platform as _platform
if sys.platform == "win32":
    _platform.system = lambda: "Windows"
print("  platform ok", flush=True)

# Load .env from project root
from pathlib import Path
try:
    from dotenv import load_dotenv
    _root = Path(__file__).resolve().parents[2]
    load_dotenv(_root / ".env")
except Exception:
    pass
print("  env loaded", flush=True)

print("Loading core libs...", flush=True)
import asyncio
import signal
from typing import Optional, Dict, Any
import aiohttp
import structlog

try:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
except Exception:
    pass
print("Core imports done.", flush=True)

print("Loading dependencies...", flush=True)
from dependency_injector import containers, providers
from dependency_injector.wiring import inject, Provide
from config.loader import ConfigLoader
from config.base import BaseConfig
print("Loading internal modules...", flush=True)
from internal.pkg.logger import StructuredLogger, LogConfig
from internal.pkg.metrics import MetricsCollector
from internal.bot.client import TelegramBotClient
from internal.bot.dispatcher import Dispatcher, MessageType
from internal.bot.middleware import (
    LoggingMiddleware,
    RateLimitMiddleware,
    RecoveryMiddleware,
    AuthMiddleware,
)
from internal.bot.handlers import (
    CommandHandler,
    LinkHandler,
    QuestionHandler,
    CallbackHandler,
)
from internal.services.youtube import YouTubeService
from internal.services.summarizer import SummarizerService
from internal.services.qa import QAService
from internal.services.language import LanguageService
from internal.services.cache import CacheManager
from internal.storage.vector import VectorDBClient
from internal.storage.session import SessionStore, SessionBackend
from internal.ai.models.factory import ModelFactory
from internal.ai.embedding import EmbeddingGenerator
from internal.ai.prompts.manager import PromptManager
from internal.domain.events import EventBus
print("All imports done.", flush=True)


class ApplicationContainer(containers.DeclarativeContainer):
    """
    Dependency Injection Container
    Manages all application dependencies and their lifetimes
    """
    
    # Configuration (loaded at runtime)
    config = providers.Dependency(instance_of=BaseConfig)
    
    # Core infrastructure
    logger = providers.Singleton(
        StructuredLogger,
        config=config,
    )
    
    metrics = providers.Singleton(
        MetricsCollector,
        service_name="telegram-youtube-bot",
    )
    
    event_bus = providers.Singleton(
        EventBus,
        logger=logger,
    )
    
    # Storage Layer
    vector_db = providers.Singleton(
        VectorDBClient,
        config=config.provided.vector_db,
        logger=logger,
        metrics=metrics,
    )
    
    session_store = providers.Singleton(
        SessionStore,
        redis_url=None,  # Memory-only for default
        default_ttl=86400,
        backend=SessionBackend.MEMORY,
        logger=logger,
    )
    
    cache_manager = providers.Singleton(
        CacheManager,
        config=config.provided.cache,
        logger=logger,
    )
    
    # AI Layer
    model_factory = providers.Singleton(
        ModelFactory,
        config=config.provided.models,
        logger=logger,
        metrics=metrics,
    )
    
    embedding_generator = providers.Singleton(
        EmbeddingGenerator,
        model_factory=model_factory,
        cache_manager=cache_manager,
        logger=logger,
    )
    
    prompt_manager = providers.Singleton(
        PromptManager,
        logger=logger,
    )
    
    # Services Layer
    language_service = providers.Factory(
        LanguageService,
        model_factory=model_factory,
        config=config.provided.language,
        logger=logger,
    )
    
    http_session = providers.Dependency()
    youtube_service = providers.Factory(
        YouTubeService,
        config=config.provided.youtube,
        cache_manager=cache_manager,
        logger=logger,
        metrics=metrics,
        http_session=http_session,
    )
    
    summarizer_service = providers.Factory(
        SummarizerService,
        model_factory=model_factory,
        prompt_manager=prompt_manager,
        language_service=language_service,
        logger=logger,
        metrics=metrics,
    )
    
    qa_service = providers.Factory(
        QAService,
        vector_db=vector_db,
        embedding_generator=embedding_generator,
        model_factory=model_factory,
        session_store=session_store,
        logger=logger,
        metrics=metrics,
    )
    
    # Bot Layer
    dispatcher = providers.Singleton(
        Dispatcher,
        logger=logger,
        metrics=metrics,
    )
    
    telegram_client = providers.Singleton(
        TelegramBotClient,
        token=config.provided.telegram.token,
        dispatcher=dispatcher,
        logger=logger,
        metrics=metrics,
    )


class Application:
    """
    Main Application Class
    Manages the complete application lifecycle
    """
    
    def __init__(self):
        self.container: Optional[ApplicationContainer] = None
        self.logger: Optional[StructuredLogger] = None
        self._shutdown_event = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._http_session: Optional[aiohttp.ClientSession] = None
        
    async def startup(self) -> None:
        """
        Initialize all application components
        Follows the startup lifecycle documented above
        """
        try:
            print("Startup: loading config...", flush=True)
            config_loader = ConfigLoader()
            config = config_loader.load()

            print("Startup: creating container...", flush=True)
            self.container = ApplicationContainer()
            self.container.config.override(config)

            self._http_session = aiohttp.ClientSession()
            self.container.http_session.override(
                providers.Object(self._http_session)
            )

            self.logger = self.container.logger()
            self.logger.info("application.starting", version="1.0.0")

            print("Startup: validating dependencies (may take a few seconds)...", flush=True)
            await self._validate_dependencies_with_timeout()
            
            # Step 5-6: Initialize core components
            # These are lazy-initialized, but we trigger initialization
            # to fail fast if any issues
            await self._initialize_components()
            
            # Step 7: Register handlers and middleware
            self._register_bot_components()
            
            # Step 8: Start the bot
            await self._start_bot()
            
            self.logger.info("application.started", 
                           environment=config.environment,
                           polling_mode=config.telegram.mode)
            
        except Exception as e:
            if self.logger:
                self.logger.exception("application.startup_failed", error=str(e))
            else:
                print(f"FATAL: Startup failed - {e}", file=sys.stderr)
            sys.exit(1)
    
    async def _validate_dependencies_with_timeout(self) -> None:
        """Validate dependencies with a short timeout so startup never hangs forever."""
        self.logger.info("application.validating_dependencies")
        config = self.container.config()
        required_configs = [
            "telegram.token",
            "youtube.api_key",
            "models.embedding.name",
            "models.generation.name",
        ]
        missing = []
        for cfg in required_configs:
            try:
                parts = cfg.split(".")
                value = config
                for part in parts:
                    value = getattr(value, part)
                if value is None:
                    missing.append(cfg)
            except AttributeError:
                missing.append(cfg)
        if missing:
            raise ValueError(f"Missing required configurations: {missing}")

        timeout_sec = 8.0
        try:
            await asyncio.wait_for(
                self.container.youtube_service().health_check(),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            self.logger.warning("youtube.health_check_timeout", timeout=timeout_sec)
        except Exception as e:
            self.logger.warning("youtube.health_check_failed", error=str(e))

        try:
            await asyncio.wait_for(
                self.container.model_factory().health_check(),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            self.logger.warning("model_factory.health_check_timeout", timeout=timeout_sec)
        except Exception as e:
            self.logger.warning("model_factory.health_check_failed", error=str(e))

        try:
            await asyncio.wait_for(
                self.container.vector_db().health_check(),
                timeout=timeout_sec,
            )
        except (asyncio.TimeoutError, Exception) as e:
            self.logger.warning("vector_db.health_check_skipped", error=str(e))

        try:
            await asyncio.wait_for(
                self.container.session_store().health_check(),
                timeout=timeout_sec,
            )
        except (asyncio.TimeoutError, Exception) as e:
            self.logger.warning("session_store.health_check_skipped", error=str(e))

        self.logger.info("application.dependencies_validated")
    
    async def _initialize_components(self) -> None:
        """
        Initialize all lazy components to fail fast
        """
        self.logger.info("application.initializing_components")
        
        # Initialize storage
        await self.container.vector_db().initialize()
        await self.container.session_store().initialize()
        
        # Initialize AI models
        await self.container.embedding_generator().initialize()
        
        # Initialize cache
        await self.container.cache_manager().initialize()
        
        self.logger.info("application.components_initialized")
    
    def _register_bot_components(self) -> None:
        """
        Register all middleware and handlers with the dispatcher
        """
        self.logger.info("application.registering_bot_components")
        
        dispatcher = self.container.dispatcher()
        telegram_client = self.container.telegram_client()
        
        # Register middleware (executed in order)
        from internal.bot.middleware.auth import AuthConfig
        from internal.bot.middleware.rate_limit import RateLimitConfig
        
        dispatcher.use(RecoveryMiddleware())
        dispatcher.use(LoggingMiddleware(logger=self.container.logger(), metrics=self.container.metrics()))
        dispatcher.use(AuthMiddleware(
            session_store=self.container.session_store(),
            config=AuthConfig(allowed_users=self.container.config().telegram.allowed_users or []),
            logger=self.container.logger(),
            metrics=self.container.metrics(),
        ))
        dispatcher.use(RateLimitMiddleware(
            config=RateLimitConfig(default_capacity=60, default_refill_rate=1.0),
            logger=self.container.logger(),
            metrics=self.container.metrics(),
        ))
        
        # Register handlers
        command_handler = CommandHandler(
            telegram_client=telegram_client,
            session_store=self.container.session_store(),
            language_service=self.container.language_service(),
            summarizer_service=self.container.summarizer_service(),
            youtube_service=self.container.youtube_service(),
            logger=self.logger,
            metrics=self.container.metrics(),
        )
        dispatcher.register_handler(MessageType.COMMAND, command_handler)
        # Language selection buttons (from /language) — must be registered so clicks are handled
        for _cb in ("lang_en", "lang_hi", "lang_ta", "lang_te", "lang_kn", "close_language_menu"):
            dispatcher.register_callback_handler(_cb, command_handler)
        
        link_handler = LinkHandler(
            telegram_client=telegram_client,
            youtube_service=self.container.youtube_service(),
            summarizer_service=self.container.summarizer_service(),
            qa_service=self.container.qa_service(),
            language_service=self.container.language_service(),
            session_store=self.container.session_store(),
            vector_db=self.container.vector_db(),
            logger=self.logger,
            metrics=self.container.metrics(),
        )
        dispatcher.register_handler(MessageType.YOUTUBE_LINK, link_handler)
        # Follow-up buttons after summary (Ask Question, New Summary, etc.)
        for _cb in ("ask", "new", "deepdive", "language"):
            dispatcher.register_callback_handler(_cb, link_handler)
        
        dispatcher.register_handler(
            MessageType.QUESTION,
            QuestionHandler(
                qa_service=self.container.qa_service(),
                language_service=self.container.language_service(),
                session_store=self.container.session_store(),
                vector_db=self.container.vector_db(),
                embedding_generator=self.container.embedding_generator(),
                youtube_service=self.container.youtube_service(),
                telegram_client=telegram_client,
                logger=self.logger,
                metrics=self.container.metrics(),
            )
        )
        
        # Wire dispatcher with telegram_client (for sending responses)
        dispatcher.telegram_client = telegram_client
        
        # Register classifiers
        from internal.bot.dispatcher import CommandClassifier, YouTubeLinkClassifier, QuestionClassifier
        dispatcher.register_classifier(CommandClassifier([
            "start", "help", "language", "lang",
            "summary", "deepdive", "actionpoints", "deep", "detailed", "actions", "ap",
        ]))
        dispatcher.register_classifier(YouTubeLinkClassifier())
        dispatcher.register_classifier(QuestionClassifier())
        
        self.logger.info("application.components_registered")
    
    async def _start_bot(self) -> None:
        """
        Start the bot in either polling or webhook mode
        """
        config = self.container.config()
        client = self.container.telegram_client()
        
        if config.telegram.mode == "webhook":
            # Webhook mode (production)
            await client.set_webhook(
                url=config.telegram.webhook_url,
                secret_token=config.telegram.webhook_secret,
                max_connections=config.telegram.max_connections,
            )
            self.logger.info("bot.webhook_set", 
                           url=config.telegram.webhook_url)
            
            # Start webhook server
            server_task = asyncio.create_task(
                client.start_webhook_server(
                    host=config.server.host,
                    port=config.server.port,
                )
            )
            self._tasks.append(server_task)
            
        else:
            # Polling mode (development)
            polling_task = asyncio.create_task(
                client.start_polling(drop_pending_updates=True)
            )
            self._tasks.append(polling_task)
            self.logger.info("bot.polling_started")
    
    async def shutdown(self) -> None:
        """
        Graceful shutdown of all components
        """
        self.logger.info("application.shutting_down")
        
        # Signal all tasks to stop
        self._shutdown_event.set()
        
        # Stop receiving new updates
        client = self.container.telegram_client()
        await client.stop()
        
        # Cancel all background tasks with timeout
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._tasks, return_exceptions=True),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            self.logger.warning("application.shutdown_timeout")
        
        # Close HTTP session (fixes "Unclosed client session" warnings)
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
        
        # Close database connections
        await self.container.vector_db().close()
        await self.container.session_store().close()
        
        # Flush metrics
        await self.container.metrics().flush()
        
        self.logger.info("application.shutdown_complete")
        
        # Force close logger
        await self.logger.close()
    
    async def run(self) -> None:
        """
        Main application loop
        Handles signals and coordinates startup/shutdown
        """
        # Set up signal handlers (not supported on Windows)
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig,
                    lambda: asyncio.create_task(self.shutdown())
                )
        except NotImplementedError:
            pass  # add_signal_handler not available on Windows
        
        try:
            # Start the application
            await self.startup()
            
            # Wait for shutdown signal
            await self._shutdown_event.wait()
            
        except Exception as e:
            self.logger.exception("application.crashed", error=str(e))
            await self.shutdown()
            raise


def main():
    """
    Main entry point with proper async handling
    """
    print("Creating application...", flush=True)
    app = Application()
    print("Starting event loop...", flush=True)
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nReceived shutdown signal", file=sys.stderr)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()