"""
AI model factory.

- If `OPENROUTER_API_KEY` or `OPEN_ROUTER_KEY` is set, uses OpenRouter (OpenAI-compatible).
- Else if `GEMINI_API_KEY` is set, uses Google Gemini.
- Else if `OPENAI_API_KEY` is set, uses OpenAI (or OpenClaw when OPENAI_BASE_URL is set).
- Otherwise returns a stub model so the bot still runs.
"""

import asyncio
import time
import warnings
from typing import Optional, Any, List
import os

import structlog

# Gemini free tier: 5 RPM, 250K TPM. Throttle to 4 RPM to stay under limit.
_GEMINI_RPM_LIMIT = 4
_GEMINI_WINDOW_SEC = 60.0
_gemini_request_times: List[float] = []
_gemini_rate_lock = asyncio.Lock()


async def _gemini_wait_for_rate_limit(logger) -> None:
    """Wait if needed so we stay under Gemini free tier RPM."""
    async with _gemini_rate_lock:
        now = time.monotonic()
        # drop times outside the window
        while _gemini_request_times and _gemini_request_times[0] < now - _GEMINI_WINDOW_SEC:
            _gemini_request_times.pop(0)
        while len(_gemini_request_times) >= _GEMINI_RPM_LIMIT:
            wait_sec = _GEMINI_WINDOW_SEC - (now - _gemini_request_times[0])
            if wait_sec > 0:
                logger.info("gemini.rate_limit_wait", wait_sec=round(wait_sec, 1))
                await asyncio.sleep(wait_sec)
            now = time.monotonic()
            while _gemini_request_times and _gemini_request_times[0] < now - _GEMINI_WINDOW_SEC:
                _gemini_request_times.pop(0)
        _gemini_request_times.append(now)

try:
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None  # type: ignore


class ModelFactory:
    """
    Factory for AI models (generation, embedding).
    Prefers Gemini when GEMINI_API_KEY is set; otherwise OpenAI or stub.
    """

    def __init__(self, config: Any = None, logger=None, metrics=None, **kwargs):
        self._config = config
        self._logger = logger or structlog.get_logger(__name__)
        self._metrics = metrics

        try:
            from dotenv import load_dotenv
            from pathlib import Path
            load_dotenv(Path(__file__).resolve().parents[3] / ".env")
        except Exception:
            pass

        self._default_model_name = "gpt-4o-mini"
        self._client: Optional[AsyncOpenAI] = None
        self._base_url = os.environ.get("OPENAI_BASE_URL")
        self._openrouter_client: Optional[AsyncOpenAI] = None
        self._openrouter_model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")
        self._gemini_key: Optional[str] = os.environ.get("GEMINI_API_KEY") or None
        self._gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

        openrouter_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPEN_ROUTER_KEY")
        if AsyncOpenAI is not None and openrouter_key:
            try:
                self._openrouter_client = AsyncOpenAI(
                    api_key=openrouter_key,
                    base_url="https://openrouter.ai/api/v1",
                )
                self._logger.info(
                    "model_factory.openrouter_enabled",
                    model=self._openrouter_model,
                )
            except Exception as exc:  # pragma: no cover
                self._logger.error("model_factory.openrouter_init_failed", error=str(exc))
                self._openrouter_client = None

        if genai is not None and self._gemini_key:
            try:
                genai.configure(api_key=self._gemini_key)
                self._logger.info(
                    "model_factory.gemini_enabled",
                    model=self._gemini_model,
                )
            except Exception as exc:  # pragma: no cover
                self._logger.error("model_factory.gemini_init_failed", error=str(exc))
                self._gemini_key = None
        elif not self._gemini_key:
            self._logger.info("model_factory.gemini_disabled", reason="missing_sdk_or_api_key")

        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = self._base_url
        if AsyncOpenAI is not None and api_key and self._openrouter_client is None:
            try:
                if base_url:
                    self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
                    self._logger.info(
                        "model_factory.openai_enabled",
                        model=self._default_model_name,
                        base_url=base_url,
                    )
                else:
                    self._client = AsyncOpenAI(api_key=api_key)
                    self._logger.info("model_factory.openai_enabled", model=self._default_model_name)
            except Exception as exc:  # pragma: no cover
                self._logger.error("model_factory.openai_init_failed", error=str(exc))
                self._client = None
        else:
            self._logger.info("model_factory.openai_disabled", reason="missing_sdk_or_api_key")

    async def get_generation_model(self, model_name: str = "gpt-3.5-turbo") -> Any:
        """
        Get a generation model. Prefers OpenRouter, then Gemini, then OpenAI (or OpenClaw), else stub.
        """
        if self._openrouter_client is not None:
            return _OpenAIChatModel(
                client=self._openrouter_client,
                model_name=self._openrouter_model,
                logger=self._logger,
                metrics=self._metrics,
            )
        if self._gemini_key and genai is not None:
            return _GeminiChatModel(
                model_name=self._gemini_model,
                logger=self._logger,
                metrics=self._metrics,
            )
        if self._client is not None:
            effective_name = (
                os.environ.get("OPENCLAW_AGENT_ID", "main")
                if self._base_url
                else (model_name or self._default_model_name)
            )
            if self._base_url:
                effective_name = f"openclaw:{effective_name}"
            return _OpenAIChatModel(
                client=self._client,
                model_name=effective_name,
                logger=self._logger,
                metrics=self._metrics,
            )
        return _StubGenerationModel()

    async def health_check(self) -> bool:
        """Return True for OpenRouter, Gemini, or stub; for OpenAI, ping models.list() when not using gateway."""
        if self._openrouter_client is not None:
            return True
        if self._gemini_key and genai is not None:
            return True
        if self._client is None:
            return True
        if self._base_url:
            return True
        try:
            await self._client.models.list()
            return True
        except Exception as exc:  # pragma: no cover
            self._logger.warning("model_factory.health_check_failed", error=str(exc))
            return False


def _gemini_generate_sync(
    model_name: str,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Sync Gemini call for use inside asyncio.to_thread."""
    model = genai.GenerativeModel(
        model_name,
        system_instruction=system_prompt or "You are a helpful assistant.",
    )
    config = {"temperature": temperature, "max_output_tokens": max_tokens}
    response = model.generate_content(prompt, generation_config=config)
    if not response or not response.text:
        return ""
    return response.text.strip()


class _GeminiChatModel:
    """Wrapper over Google Gemini. Exposes generate() compatible with the rest of the codebase."""

    def __init__(self, model_name: str, logger=None, metrics=None):
        self._model_name = model_name
        self._logger = logger or structlog.get_logger(__name__)
        self._metrics = metrics

    async def generate(self, prompt: str = None, **kwargs) -> str:
        if prompt is None:
            raise ValueError("prompt is required")
        temperature = float(kwargs.get("temperature", 0.7))
        max_tokens = int(kwargs.get("max_tokens", 2000))
        system_prompt = kwargs.get("system_prompt") or "You are a helpful assistant."
        try:
            await _gemini_wait_for_rate_limit(self._logger)
            return await asyncio.to_thread(
                _gemini_generate_sync,
                self._model_name,
                prompt,
                system_prompt,
                temperature,
                max_tokens,
            )
        except Exception as exc:  # pragma: no cover
            err_str = str(exc)
            self._logger.error(
                "gemini.generate_failed",
                error=err_str,
                model=self._model_name,
            )
            if "429" in err_str or "quota" in err_str.lower() or "resource_exhausted" in err_str.lower():
                return (
                    "GEMINI_QUOTA_EXCEEDED: Gemini free tier quota exceeded or limit is 0. "
                    "Check https://ai.google.dev/gemini-api/docs/rate-limits or try again later."
                )
            return "I'm sorry, but I couldn't generate a summary right now due to an AI service error."

    async def generate_with_system(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        return await self.generate(user_prompt, system_prompt=system_prompt, **kwargs)


class _OpenAIChatModel:
    """
    Thin wrapper over OpenAI Chat Completions.
    Exposes a `generate` method compatible with the rest of the codebase.
    """
    
    def __init__(self, client: AsyncOpenAI, model_name: str, logger=None, metrics=None):
        self._client = client
        self._model_name = model_name
        self._logger = logger or structlog.get_logger(__name__)
        self._metrics = metrics
    
    async def generate(self, prompt: str = None, **kwargs) -> str:
        """
        Generate text for a given prompt.
        
        Supported kwargs (all optional):
        - temperature: float
        - max_tokens: int
        - system_prompt: str (or `language`/other extras are ignored)
        """
        if prompt is None:
            raise ValueError("prompt is required")
        
        temperature = float(kwargs.get("temperature", 0.7))
        max_tokens = int(kwargs.get("max_tokens", 2000))
        
        # Allow callers to pass an explicit system prompt; otherwise use a generic one.
        system_prompt = kwargs.get("system_prompt") or "You are a helpful assistant."
        
        try:
            response = await self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # pragma: no cover
            # Fall back to a safe message instead of breaking the bot.
            err_str = str(exc)
            self._logger.error(
                "openai.generate_failed",
                error=err_str,
                model=self._model_name,
            )
            if "429" in err_str or "insufficient_quota" in err_str.lower():
                return (
                    "OPENAI_QUOTA_EXCEEDED: Your OpenAI account has no usable quota. "
                    "Check your plan and billing at https://platform.openai.com/account/billing"
                )
            return (
                "I'm sorry, but I couldn't generate a summary right now due to an AI service error."
            )
        
        message = response.choices[0].message
        content = (message.content or "").strip()
        return content
    
    async def generate_with_system(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Compatibility helper used by some callers."""
        return await self.generate(user_prompt, system_prompt=system_prompt, **kwargs)


class _StubGenerationModel:
    """
    Stub used when no API key is set. Returns JSON so the pipeline works;
    message tells user to set OPENAI_BASE_URL and OPENAI_API_KEY for real summaries.
    """
    
    async def generate(self, prompt: str = None, **kwargs) -> str:
        base = (prompt or "").strip()
        if not base:
            topic_preview = "this video"
        else:
            preview = base[-400:]
            preview = " ".join(preview.split())
            topic_preview = preview[:120] + ("..." if len(preview) > 120 else "")
        summary_text = f"This part of the video discusses: {topic_preview}. (Stub: set OPENAI_BASE_URL and OPENAI_API_KEY in .env for real summaries.)"
        return (
            '{"summary": "'
            + summary_text.replace('"', "'")
            + '", "key_points": ['
            '{"point": "Using stub model. Add OPENAI_BASE_URL and OPENAI_API_KEY to .env and restart the bot.", "timestamp": ""},'
            '{"point": "Then run: openclaw gateway (and keep it running).", "timestamp": ""}'
            '], "core_takeaway": "Real summaries need OpenClaw or OpenAI. See INSTALL_OPENCLAW.md."}'
        )
    
    async def generate_with_system(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        return await self.generate(user_prompt, **kwargs)
