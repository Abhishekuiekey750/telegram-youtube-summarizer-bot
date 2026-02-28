"""
Translation Service
Translates responses while preserving formatting and structure

Features:
- Preserves markdown/HTML formatting
- Maintains timestamps (MM:SS)
- Handles structured JSON data
- Technical term preservation
- Batch translation
- Multiple provider support
- Quality checks
- Fallback strategies
"""

import re
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio

import structlog

from internal.domain.value_objects import Language
from internal.ai.models.factory import ModelFactory
from internal.ai.prompts.manager import PromptManager
from internal.pkg.errors import TranslationError, RetryableError
from internal.pkg.metrics import MetricsCollector


class TranslationProvider(Enum):
    """Supported translation providers"""
    OPENAI = "openai"
    GOOGLE = "google"
    DEEPL = "deepl"
    LOCAL = "local"


@dataclass
class TranslationRequest:
    """Request for translation"""
    text: str
    source_language: Language
    target_language: Language
    preserve_formatting: bool = True
    preserve_placeholders: bool = True
    technical_terms: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_preview": self.text[:50],
            "source": self.source_language.code,
            "target": self.target_language.code,
            "preserve_formatting": self.preserve_formatting,
        }


@dataclass
class TranslationResult:
    """Result of translation"""
    text: str
    source_language: Language
    target_language: Language
    confidence: float
    processing_time: float
    characters: int
    segments_translated: int
    fallback_used: bool = False
    issues: List[str] = field(default_factory=list)
    
    @property
    def quality_acceptable(self) -> bool:
        return self.confidence >= 0.7 and not self.issues
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_preview": self.text[:50],
            "source": self.source_language.code,
            "target": self.target_language.code,
            "confidence": self.confidence,
            "processing_time_ms": round(self.processing_time * 1000),
            "characters": self.characters,
            "segments": self.segments_translated,
            "fallback_used": self.fallback_used,
        }


class FormatPreserver:
    """
    Preserves formatting during translation
    Handles markdown, HTML, timestamps, and structured data
    """
    
    # Patterns to preserve
    TIMESTAMP_PATTERN = re.compile(r'`(\d{1,2}:\d{2}(?::\d{2})?)`')
    MARKDOWN_BOLD = re.compile(r'\*\*(.*?)\*\*')
    MARKDOWN_ITALIC = re.compile(r'_(.*?)_')
    MARKDOWN_CODE = re.compile(r'`(.*?)`')
    EMOJI_PATTERN = re.compile(r'([🔑💡📹❓✅⏳🌐📝🔍📋])')
    URL_PATTERN = re.compile(r'https?://[^\s]+')
    PLACEHOLDER_PATTERN = re.compile(r'{{(.*?)}}')
    
    def __init__(self):
        self.placeholders = {}
        self.placeholder_counter = 0
    
    def extract_formatting(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Extract formatting and replace with placeholders
        
        Returns:
            Tuple of (text_with_placeholders, placeholder_map)
        """
        self.placeholders = {}
        self.placeholder_counter = 0
        
        processed = text
        
        # Protect timestamps
        processed = self._protect_pattern(processed, self.TIMESTAMP_PATTERN, "TS")
        
        # Protect markdown formatting
        processed = self._protect_pattern(processed, self.MARKDOWN_BOLD, "BOLD")
        processed = self._protect_pattern(processed, self.MARKDOWN_ITALIC, "ITALIC")
        processed = self._protect_pattern(processed, self.MARKDOWN_CODE, "CODE")
        
        # Protect emojis
        processed = self._protect_pattern(processed, self.EMOJI_PATTERN, "EMOJI")
        
        # Protect URLs
        processed = self._protect_pattern(processed, self.URL_PATTERN, "URL")
        
        # Protect placeholders
        processed = self._protect_pattern(processed, self.PLACEHOLDER_PATTERN, "PH")
        
        return processed, self.placeholders
    
    def _protect_pattern(
        self,
        text: str,
        pattern: re.Pattern,
        prefix: str,
    ) -> str:
        """Protect pattern matches with placeholders"""
        
        def replacer(match):
            self.placeholder_counter += 1
            placeholder = f"__{prefix}_{self.placeholder_counter}__"
            self.placeholders[placeholder] = match.group(0)
            return placeholder
        
        return pattern.sub(replacer, text)
    
    def restore_formatting(
        self,
        text: str,
        placeholders: Dict[str, str],
    ) -> str:
        """Restore placeholders to original formatting"""
        result = text
        for placeholder, original in placeholders.items():
            result = result.replace(placeholder, original)
        return result
    
    def extract_structured_content(
        self,
        data: Union[Dict, List, str],
    ) -> Tuple[Union[Dict, List, str], Dict[str, str]]:
        """
        Extract translatable text from structured data
        
        Returns:
            Tuple of (structure_with_placeholders, text_map)
        """
        text_map = {}
        text_counter = 0
        
        def extract_recursive(obj, path=""):
            nonlocal text_counter
            
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    # Skip non-translatable fields
                    if key in ["timestamp", "video_id", "url"]:
                        result[key] = value
                    else:
                        result[key] = extract_recursive(value, f"{path}.{key}")
                return result
                
            elif isinstance(obj, list):
                return [extract_recursive(item, f"{path}[{i}]") for i, item in enumerate(obj)]
                
            elif isinstance(obj, str):
                # Check if this is translatable
                if len(obj.strip()) > 0 and not obj.startswith(('http://', 'https://')):
                    text_counter += 1
                    placeholder = f"__TEXT_{text_counter}__"
                    text_map[placeholder] = obj
                    return placeholder
                return obj
                
            else:
                return obj
        
        structure = extract_recursive(data)
        return structure, text_map
    
    def restore_structured_content(
        self,
        structure: Union[Dict, List, str],
        text_map: Dict[str, str],
    ) -> Union[Dict, List, str]:
        """Restore translated text into structure"""
        
        def restore_recursive(obj):
            if isinstance(obj, dict):
                return {k: restore_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [restore_recursive(item) for item in obj]
            elif isinstance(obj, str) and obj in text_map:
                return text_map[obj]
            else:
                return obj
        
        return restore_recursive(structure)


class TranslationService:
    """
    Translates content while preserving formatting and structure
    
    Features:
    - Multiple provider support (OpenAI, Google, DeepL)
    - Format preservation (markdown, HTML, timestamps)
    - Structured data translation (JSON)
    - Technical term preservation
    - Batch translation
    - Quality checks
    - Fallback strategies
    """
    
    def __init__(
        self,
        model_factory: ModelFactory,
        prompt_manager: PromptManager,
        logger=None,
        metrics=None,
        primary_provider: TranslationProvider = TranslationProvider.OPENAI,
        fallback_provider: Optional[TranslationProvider] = None,
        enable_format_preservation: bool = True,
        max_batch_size: int = 10,
    ):
        """
        Initialize translation service
        
        Args:
            model_factory: Factory for AI models
            prompt_manager: Prompt manager for translation prompts
            logger: Structured logger
            metrics: Metrics collector
            primary_provider: Primary translation provider
            fallback_provider: Fallback provider if primary fails
            enable_format_preservation: Whether to preserve formatting
            max_batch_size: Maximum items in batch translation
        """
        self.model_factory = model_factory
        self.prompt_manager = prompt_manager
        self.logger = logger or structlog.get_logger(__name__)
        self.metrics = metrics or MetricsCollector("translation_service")
        
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
        self.enable_format_preservation = enable_format_preservation
        self.max_batch_size = max_batch_size
        
        self.format_preserver = FormatPreserver()
        
        self.logger.info(
            "translation_service.initialized",
            primary_provider=primary_provider.value,
            fallback_provider=fallback_provider.value if fallback_provider else None,
            enable_format_preservation=enable_format_preservation,
        )
    
    async def translate(
        self,
        text: str,
        target_language: Language,
        source_language: Optional[Language] = None,
        preserve_formatting: bool = True,
        technical_terms: Optional[List[str]] = None,
    ) -> TranslationResult:
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            target_language: Target language
            source_language: Source language (auto-detect if None)
            preserve_formatting: Whether to preserve formatting
            technical_terms: List of technical terms to preserve
            
        Returns:
            Translation result
        """
        import time
        start_time = time.time()
        
        # Detect source language if not provided
        if not source_language:
            source_language = await self._detect_language(text)
        
        # Create request
        request = TranslationRequest(
            text=text,
            source_language=source_language,
            target_language=target_language,
            preserve_formatting=preserve_formatting and self.enable_format_preservation,
            technical_terms=technical_terms or [],
        )
        
        self.logger.debug(
            "translation.starting",
            **request.to_dict(),
        )
        
        try:
            # Translate with primary provider
            result = await self._translate_with_provider(
                request,
                self.primary_provider,
            )
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            self.metrics.record_latency(
                "translation.duration",
                processing_time,
                tags={
                    "provider": self.primary_provider.value,
                    "target": target_language.code,
                },
            )
            
            self.metrics.increment(
                "translation.completed",
                tags={
                    "provider": self.primary_provider.value,
                    "success": "true",
                },
            )
            
            return result
            
        except Exception as e:
            self.logger.warning(
                "translation.primary_failed",
                provider=self.primary_provider.value,
                error=str(e),
            )
            
            self.metrics.increment(
                "translation.failed",
                tags={"provider": self.primary_provider.value},
            )
            
            # Try fallback if available
            if self.fallback_provider:
                try:
                    self.logger.info(
                        "translation.trying_fallback",
                        provider=self.fallback_provider.value,
                    )
                    
                    result = await self._translate_with_provider(
                        request,
                        self.fallback_provider,
                    )
                    
                    result.fallback_used = True
                    processing_time = time.time() - start_time
                    result.processing_time = processing_time
                    
                    self.metrics.increment(
                        "translation.fallback_success",
                        tags={"provider": self.fallback_provider.value},
                    )
                    
                    return result
                    
                except Exception as fallback_error:
                    self.logger.error(
                        "translation.fallback_failed",
                        provider=self.fallback_provider.value,
                        error=str(fallback_error),
                    )
            
            # All providers failed - try free Google Translate (no API key)
            try:
                self.logger.info("translation.trying_free_google")
                free_text = await self._translate_with_free_google(
                    request.text,
                    source_language,
                    target_language,
                )
                if free_text:
                    processing_time = time.time() - start_time
                    self.metrics.increment("translation.free_google_success")
                    return TranslationResult(
                        text=free_text,
                        source_language=source_language,
                        target_language=target_language,
                        confidence=0.85,
                        processing_time=processing_time,
                        characters=len(request.text),
                        segments_translated=1,
                        fallback_used=True,
                    )
            except Exception as free_err:
                self.logger.warning(
                    "translation.free_google_failed",
                    error=str(free_err),
                )

            self.metrics.increment("translation.all_failed")
            raise TranslationError(
                f"Translation failed: {str(e)}",
                source_language=source_language,
                target_language=target_language,
            ) from e
    
    async def _translate_with_provider(
        self,
        request: TranslationRequest,
        provider: TranslationProvider,
    ) -> TranslationResult:
        """Translate using specific provider"""
        
        if request.preserve_formatting:
            # Extract formatting
            text_with_placeholders, placeholders = self.format_preserver.extract_formatting(
                request.text
            )
            
            # Translate the clean text
            translated_text = await self._call_translation_api(
                text_with_placeholders,
                request.source_language,
                request.target_language,
                provider,
                request.technical_terms,
            )
            
            # Restore formatting
            final_text = self.format_preserver.restore_formatting(
                translated_text,
                placeholders,
            )
        else:
            # Simple translation without formatting
            final_text = await self._call_translation_api(
                request.text,
                request.source_language,
                request.target_language,
                provider,
                request.technical_terms,
            )
        
        # Calculate confidence
        confidence = await self._calculate_confidence(
            request.text,
            final_text,
            request.source_language,
            request.target_language,
        )
        
        # Count segments (rough estimate)
        segments = len(re.findall(r'[.!?]+', request.text)) or 1
        
        return TranslationResult(
            text=final_text,
            source_language=request.source_language,
            target_language=request.target_language,
            confidence=confidence,
            processing_time=0,  # Will be set by caller
            characters=len(request.text),
            segments_translated=segments,
        )
    
    async def _call_translation_api(
        self,
        text: str,
        source: Language,
        target: Language,
        provider: TranslationProvider,
        technical_terms: List[str],
    ) -> str:
        """Call actual translation API"""
        
        if provider == TranslationProvider.OPENAI:
            return await self._translate_with_openai(text, source, target, technical_terms)
        elif provider == TranslationProvider.GOOGLE:
            return await self._translate_with_google(text, source, target)
        elif provider == TranslationProvider.DEEPL:
            return await self._translate_with_deepl(text, source, target)
        elif provider == TranslationProvider.LOCAL:
            return await self._translate_with_local(text, source, target)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def _translate_with_free_google(
        self,
        text: str,
        source_language: Language,
        target_language: Language,
    ) -> Optional[str]:
        """Translate using free Google Translate (deep_translator). No API key required."""
        if not text or not text.strip():
            return text
        try:
            from deep_translator import GoogleTranslator
        except ImportError:
            return None
        src = source_language.code if source_language else "en"
        tgt = target_language.code
        if src == tgt:
            return text
        try:
            # GoogleTranslator is sync; run in thread to avoid blocking
            def _sync_translate() -> str:
                translator = GoogleTranslator(source=src, target=tgt)
                return translator.translate(text[:5000])  # API limit ~5000 chars
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _sync_translate)
        except Exception as e:
            self.logger.debug("translation.free_google_error", error=str(e))
            return None

    async def _translate_with_openai(
        self,
        text: str,
        source: Language,
        target: Language,
        technical_terms: List[str],
    ) -> str:
        """Translate using OpenAI"""
        
        # Get translation prompt
        prompt = await self.prompt_manager.get_translation_prompt(
            text=text,
            source_lang=source,
            target_lang=target,
            technical_terms=technical_terms,
        )
        
        # Get model
        model = await self.model_factory.get_generation_model("gpt-3.5-turbo")
        
        # Generate translation
        response = await model.generate(
            prompt=prompt,
            temperature=0.3,  # Lower temperature for translation
            max_tokens=len(text) * 2,  # Rough estimate
        )
        
        return response.strip()
    
    async def _translate_with_google(
        self,
        text: str,
        source: Language,
        target: Language,
    ) -> str:
        """Translate using Google Translate API"""
        # Implementation would use google-cloud-translate
        # For now, fallback to OpenAI
        return await self._translate_with_openai(text, source, target, [])
    
    async def _translate_with_deepl(
        self,
        text: str,
        source: Language,
        target: Language,
    ) -> str:
        """Translate using DeepL API"""
        # Implementation would use deepl library
        # For now, fallback to OpenAI
        return await self._translate_with_openai(text, source, target, [])
    
    async def _translate_with_local(
        self,
        text: str,
        source: Language,
        target: Language,
    ) -> str:
        """Translate using local model (IndicTrans, etc.)"""
        # Implementation would use local model
        # For now, fallback to OpenAI
        return await self._translate_with_openai(text, source, target, [])
    
    async def _detect_language(self, text: str) -> Language:
        """Detect language of text"""
        # Simple detection based on script
        if re.search(r'[\u0900-\u097F]', text):
            return Language.HINDI
        elif re.search(r'[\u0B80-\u0BFF]', text):
            return Language.TAMIL
        elif re.search(r'[\u0C00-\u0C7F]', text):
            return Language.TELUGU
        # ... other scripts
        else:
            return Language.ENGLISH
    
    async def _calculate_confidence(
        self,
        original: str,
        translated: str,
        source: Language,
        target: Language,
    ) -> float:
        """Calculate translation confidence"""
        # Simple heuristic based on length ratio
        len_ratio = len(translated) / len(original) if original else 1.0
        
        if 0.7 <= len_ratio <= 1.3:
            base_confidence = 0.9
        elif 0.5 <= len_ratio <= 1.5:
            base_confidence = 0.7
        else:
            base_confidence = 0.5
        
        # Check for placeholder preservation
        original_placeholders = set(re.findall(r'__\w+_\d+__', original))
        translated_placeholders = set(re.findall(r'__\w+_\d+__', translated))
        
        if original_placeholders:
            preserved = original_placeholders.issubset(translated_placeholders)
            placeholder_score = 1.0 if preserved else 0.5
        else:
            placeholder_score = 1.0
        
        return base_confidence * placeholder_score
    
    # ------------------------------------------------------------------------
    # Batch Translation
    # ------------------------------------------------------------------------
    
    async def translate_batch(
        self,
        texts: List[str],
        target_language: Language,
        source_language: Optional[Language] = None,
    ) -> List[TranslationResult]:
        """Translate multiple texts"""
        # Process in batches
        results = []
        
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            
            # Translate batch
            batch_results = await asyncio.gather(*[
                self.translate(text, target_language, source_language)
                for text in batch
            ], return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    self.logger.error("batch_translation.failed", error=str(result))
                    results.append(None)
                else:
                    results.append(result)
        
        return results
    
    # ------------------------------------------------------------------------
    # Structured Data Translation
    # ------------------------------------------------------------------------
    
    async def translate_structured(
        self,
        data: Union[Dict, List],
        target_language: Language,
        source_language: Optional[Language] = None,
    ) -> Union[Dict, List]:
        """
        Translate structured data (JSON) while preserving structure
        
        Example:
        {
            "key_points": [
                {"point": "Price is $49", "timestamp": "02:30"}
            ]
        }
        → Translate "point" field only
        """
        # Extract translatable text
        structure, text_map = self.format_preserver.extract_structured_content(data)
        
        if not text_map:
            return data
        
        # Translate all extracted texts
        texts_to_translate = list(text_map.values())
        translated_texts = await self.translate_batch(
            texts_to_translate,
            target_language,
            source_language,
        )
        
        # Map back
        translated_map = {}
        for (placeholder, original), translated in zip(text_map.items(), translated_texts):
            if translated and translated.text:
                translated_map[placeholder] = translated.text
            else:
                translated_map[placeholder] = original  # Fallback to original
        
        # Restore structure
        return self.format_preserver.restore_structured_content(structure, translated_map)
    
    # ------------------------------------------------------------------------
    # Specialized Translations
    # ------------------------------------------------------------------------
    
    async def translate_summary(
        self,
        summary: Dict[str, Any],
        target_language: Language,
    ) -> Dict[str, Any]:
        """Translate a video summary"""
        return await self.translate_structured(summary, target_language)
    
    async def translate_answer(
        self,
        answer: str,
        target_language: Language,
        confidence: float = 1.0,
    ) -> str:
        """Translate a Q&A answer"""
        result = await self.translate(answer, target_language)
        
        # Add confidence note if low
        if confidence < 0.7 and target_language != Language.ENGLISH:
            disclaimer = {
                Language.HINDI: "\n\n(अनुवादित उत्तर - मूल अंग्रेजी से)",
                Language.TAMIL: "\n\n(மொழிபெயர்க்கப்பட்ட பதில் - அசல் ஆங்கிலத்திலிருந்து)",
                Language.TELUGU: "\n\n(అనువదించబడిన సమాధానం - అసలు ఆంగ్లం నుండి)",
            }.get(target_language, "\n\n(Translated answer)")
            
            return result.text + disclaimer
        
        return result.text
    
    # ------------------------------------------------------------------------
    # Quality Checks
    # ------------------------------------------------------------------------
    
    async def verify_translation(
        self,
        original: str,
        translated: str,
        source: Language,
        target: Language,
    ) -> Tuple[bool, List[str]]:
        """Verify translation quality"""
        issues = []
        
        # Check for placeholder preservation
        original_placeholders = set(re.findall(r'__\w+_\d+__', original))
        translated_placeholders = set(re.findall(r'__\w+_\d+__', translated))
        
        missing_placeholders = original_placeholders - translated_placeholders
        if missing_placeholders:
            issues.append(f"Missing placeholders: {missing_placeholders}")
        
        # Check for truncation
        if len(translated) < len(original) * 0.5:
            issues.append("Translation may be truncated")
        
        # Check for repeated text
        if translated.count(translated[:50]) > 1:
            issues.append("Possible repetition in translation")
        
        return len(issues) == 0, issues


# ------------------------------------------------------------------------
# Factory Function
# ------------------------------------------------------------------------

def create_translation_service(
    model_factory: ModelFactory,
    prompt_manager: PromptManager,
    primary_provider: str = "openai",
    fallback_provider: Optional[str] = None,
    enable_format_preservation: bool = True,
    logger=None,
    metrics=None,
) -> TranslationService:
    """
    Create translation service with configuration
    
    Args:
        model_factory: Model factory
        prompt_manager: Prompt manager
        primary_provider: Primary provider name
        fallback_provider: Fallback provider name
        enable_format_preservation: Whether to preserve formatting
        logger: Logger instance
        metrics: Metrics collector
        
    Returns:
        Configured TranslationService
    """
    primary = TranslationProvider(primary_provider)
    fallback = TranslationProvider(fallback_provider) if fallback_provider else None
    
    return TranslationService(
        model_factory=model_factory,
        prompt_manager=prompt_manager,
        logger=logger,
        metrics=metrics,
        primary_provider=primary,
        fallback_provider=fallback,
        enable_format_preservation=enable_format_preservation,
    )


# ------------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------------

"""
# Usage examples:

translator = create_translation_service(model_factory, prompt_manager)

# Simple text translation
result = await translator.translate(
    text="🔑 *Key Points:*\n1. `02:30` - Pricing starts at **$49/month**",
    target_language=Language.HINDI,
)

print(result.text)
# Output: 🔑 *मुख्य बिंदु:*\n1. `02:30` - कीमत शुरू होती है **$49/month**

# Structured data translation
summary = {
    "key_points": [
        {"point": "Price is $49", "timestamp": "02:30"},
        {"point": "Unlimited API calls", "timestamp": "05:45"},
    ],
    "core_takeaway": "Great value for money"
}

translated = await translator.translate_structured(
    summary,
    target_language=Language.TAMIL,
)

# Batch translation
texts = ["Hello", "How are you?", "Thank you"]
results = await translator.translate_batch(texts, Language.HINDI)

# Answer translation with confidence
answer = await translator.translate_answer(
    answer="The price is $49 per month",
    target_language=Language.TELUGU,
    confidence=0.65,  # Low confidence adds disclaimer
)
"""

# ------------------------------------------------------------------------
# Example Output
# ------------------------------------------------------------------------

"""
Original:
🔑 *Key Points:*
1. `02:30` - Pricing starts at **$49/month**
2. `05:45` - Features include unlimited API calls

Translated (Hindi):
🔑 *मुख्य बिंदु:*
1. `02:30` - कीमत शुरू होती है **$49/month**
2. `05:45` - सुविधाओं में शामिल है unlimited API calls

Translated (Tamil) with structure:
{
    "key_points": [
        {"point": "விலை $49", "timestamp": "02:30"},
        {"point": "வரம்பற்ற API அழைப்புகள்", "timestamp": "05:45"}
    ],
    "core_takeaway": "பணத்திற்கு சிறந்த மதிப்பு"
}
"""