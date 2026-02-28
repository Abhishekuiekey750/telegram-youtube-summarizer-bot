"""Language detection and translation."""

import asyncio
from typing import Optional, Dict, Any, List

from internal.services.language.detector import LanguageDetector
from internal.services.language.translator import TranslationService
from internal.domain.value_objects import Language


def _translate_text_free_google_sync(text: str, target_code: str, source_code: str = "en") -> Optional[str]:
    """Sync translation using free Google (deep_translator). Returns None on failure."""
    if not text or not text.strip() or target_code == source_code:
        return text if (text and target_code == source_code) else None
    try:
        from deep_translator import GoogleTranslator
        t = GoogleTranslator(source=source_code, target=target_code)
        return t.translate(text[:5000])
    except Exception:
        return None


class LanguageService:
    """Facade combining detection and translation."""

    def __init__(self, model_factory=None, config=None, logger=None, prompt_manager=None, **kwargs):
        self._logger = logger
        self._detector = LanguageDetector(logger=logger, **kwargs)
        prompt_mgr = prompt_manager
        if prompt_mgr is None:
            from internal.ai.prompts.manager import PromptManager
            prompt_mgr = PromptManager(logger=logger)
        self._translator = TranslationService(
            model_factory=model_factory,
            prompt_manager=prompt_mgr,
            logger=logger,
            **kwargs,
        )

    def detect_language(self, text: str):
        return self._detector.detect(text)

    async def translate(
        self,
        text: str,
        target_language: Language,
        source_language: Optional[Language] = None,
        **kwargs,
    ) -> str:
        """Translate text to target language. Used by QA orchestrator."""
        if target_language == Language.ENGLISH:
            return text
        try:
            result = await self._translator.translate(
                text=text,
                target_language=target_language,
                source_language=source_language,
                **kwargs,
            )
            return result.text if hasattr(result, "text") else str(result)
        except Exception as e:
            if getattr(self, "_logger", None):
                logger = self._logger
                logger.warning("language_service.translate_failed", error=str(e))
            return text

    async def _translate_summary_free_google(
        self, raw: Dict[str, Any], target_language: Language
    ) -> Optional[Dict[str, Any]]:
        """Translate summary dict using only free Google Translate. No API key needed."""
        tgt = target_language.code
        if tgt == "en":
            return raw
        loop = asyncio.get_event_loop()

        async def translate_one(text: str) -> str:
            out = await loop.run_in_executor(
                None, lambda t=text: _translate_text_free_google_sync(t, tgt, "en")
            )
            return out if out is not None else text

        try:
            key_points = list(raw.get("key_points") or [])
            for i, pt in enumerate(key_points):
                if isinstance(pt, dict) and pt.get("point"):
                    key_points[i] = {**pt, "point": await translate_one(pt["point"])}
                elif isinstance(pt, str):
                    key_points[i] = await translate_one(pt)
            core = (raw.get("core_takeaway") or "").strip()
            if core:
                core = await translate_one(core)
            action_items = list(raw.get("action_items") or [])
            for i, item in enumerate(action_items):
                if isinstance(item, str):
                    action_items[i] = await translate_one(item)
            return {
                **raw,
                "key_points": key_points,
                "core_takeaway": core,
                "action_items": action_items,
            }
        except Exception:
            return None

    async def translate_summary(self, summary, target_language: Language):
        """Translate summary (key_points, core_takeaway, action_items) to target language."""
        if target_language == Language.ENGLISH:
            return summary
        raw = getattr(summary, "to_dict", lambda: None)()
        if raw is None:
            raw = {
                "key_points": getattr(summary, "key_points", []),
                "core_takeaway": getattr(summary, "core_takeaway", ""),
                "action_items": getattr(summary, "action_items", []),
            }
        # Prefer free Google first so Hindi/other languages always work without API
        translated = await self._translate_summary_free_google(raw, target_language)
        if translated is None:
            try:
                translated = await self._translator.translate_summary(raw, target_language)
            except Exception as e:
                if getattr(self, "_logger", None):
                    self._logger.warning("language_service.translate_summary_failed", error=str(e))
                return summary
        from internal.services.summarizer import Summary
        return Summary(
            video_id=getattr(summary, "video_id", ""),
            title=getattr(summary, "title", ""),
            key_points=translated.get("key_points", getattr(summary, "key_points", [])),
            core_takeaway=translated.get("core_takeaway", getattr(summary, "core_takeaway", "")),
            action_items=translated.get("action_items", getattr(summary, "action_items", [])),
            summary_type=getattr(summary, "summary_type", None),
            language=target_language,
            chunk_count=getattr(summary, "chunk_count", 1),
            processing_time=getattr(summary, "processing_time", 0.0),
            metadata=getattr(summary, "metadata", {}),
        )

    async def translate_answer(self, answer: str, target_language: Language, confidence: float = 1.0) -> str:
        """Translate a Q&A answer to target language."""
        if target_language == Language.ENGLISH:
            return answer
        try:
            result = await self._translator.translate(answer, target_language)
            return result.text if hasattr(result, "text") else str(result)
        except Exception as e:
            if getattr(self, "_logger", None):
                logger = self._logger
                logger.warning("language_service.translate_answer_failed", error=str(e))
            return answer

    def get_language_from_code(self, code: str):
        return Language.from_code(code)


__all__ = ["LanguageService"]
