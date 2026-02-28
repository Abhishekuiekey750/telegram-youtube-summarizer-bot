"""
Centralized Prompt Manager
Manages all AI prompts with versioning, templating, and i18n support

Features:
- Template-based prompts with variable substitution
- Version control for prompts
- Language-specific prompts
- Prompt composition (reusable components)
- Metrics and monitoring
- A/B testing support
- Prompt validation
"""

import os
import json
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import asyncio
import hashlib

import structlog
import aiofiles

from internal.domain.value_objects import Language
from internal.pkg.errors import ValidationError, NotFoundError
from internal.pkg.metrics import MetricsCollector


class PromptTemplate:
    """
    A single prompt template with metadata
    """
    
    def __init__(
        self,
        name: str,
        template: str,
        version: str,
        description: str = "",
        variables: List[str] = None,
        language: str = "en",
        tags: List[str] = None,
        created_at: Optional[datetime] = None,
    ):
        self.name = name
        self.template = template
        self.version = version
        self.description = description
        self.variables = variables or self._extract_variables(template)
        self.language = language
        self.tags = tags or []
        self.created_at = created_at or datetime.now()
        self.usage_count = 0
        self._compiled = None  # For caching compiled template
    
    def _extract_variables(self, template: str) -> List[str]:
        """Extract variable names from template {{variable}}"""
        pattern = r'\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}'
        return list(set(re.findall(pattern, template)))
    
    def render(self, **kwargs) -> str:
        """
        Render template with variables
        
        Args:
            **kwargs: Variable values
            
        Returns:
            Rendered prompt
            
        Raises:
            ValidationError: If missing required variables
        """
        # Check for missing variables
        missing = [v for v in self.variables if v not in kwargs]
        if missing:
            raise ValidationError(
                f"Missing required variables: {missing}",
                context={"template": self.name, "missing": missing},
            )
        
        # Simple string replacement
        result = self.template
        for key, value in kwargs.items():
            placeholder = f'{{{{{key}}}}}'
            result = result.replace(placeholder, str(value))
        
        self.usage_count += 1
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "variables": self.variables,
            "language": self.language,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "usage_count": self.usage_count,
            "template_preview": self.template[:100] + "...",
        }


class PromptManager:
    """
    Centralized manager for all AI prompts
    
    Responsibilities:
    - Load prompts from files
    - Version management
    - Variable substitution
    - Language-specific prompts
    - Prompt composition
    - Metrics tracking
    - A/B testing
    """
    
    # Default paths
    DEFAULT_PROMPT_PATH = Path(__file__).parent / "templates"
    
    def __init__(
        self,
        prompt_path: Optional[Path] = None,
        logger=None,
        metrics=None,
        enable_cache: bool = True,
        cache_ttl: int = 3600,  # 1 hour
    ):
        """
        Initialize prompt manager
        
        Args:
            prompt_path: Path to prompt template files
            logger: Structured logger
            metrics: Metrics collector
            enable_cache: Whether to cache compiled prompts
            cache_ttl: Cache TTL in seconds
        """
        self.prompt_path = prompt_path or self.DEFAULT_PROMPT_PATH
        self.logger = logger or structlog.get_logger(__name__)
        self.metrics = metrics or MetricsCollector("prompt_manager")
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        
        # Prompt storage
        self._prompts: Dict[str, PromptTemplate] = {}
        self._versions: Dict[str, str] = {}  # name -> active version
        self._cache: Dict[str, tuple[PromptTemplate, datetime]] = {}
        
        # Load prompts
        self._load_prompts()
        
        self.logger.info(
            "prompt_manager.initialized",
            prompt_path=str(self.prompt_path),
            prompt_count=len(self._prompts),
            versions=len(self._versions),
        )
    
    def _load_prompts(self) -> None:
        """Load all prompt templates from files"""
        if not self.prompt_path.exists():
            self.logger.warning(
                "prompt_manager.path_not_found",
                path=str(self.prompt_path),
            )
            return
        
        # Walk through all .txt and .prompt files
        for file_path in self.prompt_path.glob("**/*.txt"):
            self._load_prompt_file(file_path)
        
        for file_path in self.prompt_path.glob("**/*.prompt"):
            self._load_prompt_file(file_path)
        
        # Load version configuration if exists
        version_file = self.prompt_path / "active_versions.json"
        if version_file.exists():
            self._load_versions(version_file)
    
    def _load_prompt_file(self, file_path: Path) -> None:
        """
        Load a single prompt file
        
        File naming convention:
        - category/name_v1.txt
        - category/name_v1.en.txt (language specific)
        """
        try:
            # Parse file path
            relative = file_path.relative_to(self.prompt_path)
            parts = str(relative).split('/')
            
            # Extract metadata from filename
            filename = parts[-1]
            name_parts = filename.replace('.txt', '').replace('.prompt', '').split('_')
            
            if len(name_parts) >= 2 and name_parts[-1].startswith('v'):
                # Has version: name_v1
                name = '_'.join(name_parts[:-1])
                version = name_parts[-1]
            else:
                # No version: name
                name = '_'.join(name_parts)
                version = "v1"
            
            # Check for language suffix
            language = "en"
            if '.' in name:
                name, lang = name.split('.')
                if len(lang) == 2:  # en, hi, ta
                    language = lang
            
            # Read template
            with open(file_path, 'r', encoding='utf-8') as f:
                template = f.read()
            
            # Create prompt template
            prompt = PromptTemplate(
                name=name,
                template=template,
                version=version,
                language=language,
                tags=parts[:-1],  # Use directories as tags
            )
            
            # Store with composite key
            key = f"{name}:{version}:{language}"
            self._prompts[key] = prompt
            
            # Set default version if not set
            if name not in self._versions:
                self._versions[name] = version
            
            self.logger.debug(
                "prompt_manager.loaded",
                name=name,
                version=version,
                language=language,
                variables=prompt.variables,
            )
            
        except Exception as e:
            self.logger.error(
                "prompt_manager.load_failed",
                file=str(file_path),
                error=str(e),
            )
    
    def _load_versions(self, version_file: Path) -> None:
        """Load active versions configuration"""
        try:
            with open(version_file, 'r') as f:
                versions = json.load(f)
            
            for name, version in versions.items():
                if name in self._versions:
                    self._versions[name] = version
                    self.logger.info(
                        "prompt_manager.version_set",
                        name=name,
                        version=version,
                    )
                    
        except Exception as e:
            self.logger.error(
                "prompt_manager.version_load_failed",
                error=str(e),
            )
    
    # ------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------
    
    async def get_prompt(
        self,
        name: str,
        variables: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
        language: Language = Language.ENGLISH,
        **kwargs,
    ) -> str:
        """
        Get rendered prompt by name
        
        Args:
            name: Prompt name (e.g., "summary/concise")
            variables: Template variables
            version: Specific version (uses active if not specified)
            language: Language for prompt
            **kwargs: Additional variables (merged with variables)
            
        Returns:
            Rendered prompt string
            
        Raises:
            NotFoundError: If prompt not found
            ValidationError: If variables missing
        """
        # Merge variables
        all_vars = {**(variables or {}), **kwargs}
        
        # Check cache first
        cache_key = f"{name}:{version}:{language.code}"
        if self.enable_cache and cache_key in self._cache:
            prompt, timestamp = self._cache[cache_key]
            age = (datetime.now() - timestamp).total_seconds()
            if age < self.cache_ttl:
                self.metrics.increment("prompt.cache_hit")
                return prompt.render(**all_vars)
        
        # Get prompt template
        prompt = await self._get_template(name, version, language)
        
        # Cache if enabled
        if self.enable_cache:
            self._cache[cache_key] = (prompt, datetime.now())
        
        # Track metrics
        self.metrics.increment(
            "prompt.rendered",
            tags={
                "name": name,
                "version": prompt.version,
                "language": language.code,
            },
        )
        
        # Render and return
        return prompt.render(**all_vars)
    
    async def _get_template(
        self,
        name: str,
        version: Optional[str] = None,
        language: Language = Language.ENGLISH,
    ) -> PromptTemplate:
        """
        Get prompt template with fallback
        
        Tries:
        1. Exact match: name:version:language
        2. Language-specific default: name:active:language
        3. Version-specific English: name:version:en
        4. Default English: name:active:en
        """
        active_version = version or self._versions.get(name, "v1")
        
        # Try exact match
        key = f"{name}:{active_version}:{language.code}"
        if key in self._prompts:
            return self._prompts[key]
        
        # Try language-specific with active version
        key = f"{name}:{active_version}:{language.code}"
        if key in self._prompts:
            return self._prompts[key]
        
        # Try version-specific English
        key = f"{name}:{active_version}:en"
        if key in self._prompts:
            self.logger.debug(
                "prompt_manager.falling_back_to_english",
                name=name,
                version=active_version,
                language=language.code,
            )
            return self._prompts[key]
        
        # Try any English version
        for key, prompt in self._prompts.items():
            if prompt.name == name and prompt.language == "en":
                self.logger.warning(
                    "prompt_manager.using_any_english",
                    name=name,
                    found_version=prompt.version,
                )
                return prompt
        
        # Not found
        raise NotFoundError(
            f"Prompt not found: {name}",
            context={
                "name": name,
                "version": active_version,
                "language": language.code,
            },
        )
    
    async def get_prompt_info(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get information about a prompt"""
        # Try to find any version of this prompt
        for key, prompt in self._prompts.items():
            if prompt.name == name:
                if not version or prompt.version == version:
                    return prompt.to_dict()
        
        raise NotFoundError(f"Prompt not found: {name}")
    
    def list_prompts(
        self,
        category: Optional[str] = None,
        version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all available prompts"""
        prompts = []
        
        for prompt in self._prompts.values():
            if category and category not in prompt.tags:
                continue
            if version and prompt.version != version:
                continue
            
            prompts.append(prompt.to_dict())
        
        return prompts
    
    def set_active_version(self, name: str, version: str) -> None:
        """Set active version for a prompt"""
        if name not in self._versions:
            raise NotFoundError(f"Prompt not found: {name}")
        
        # Verify version exists
        found = False
        for key, prompt in self._prompts.items():
            if prompt.name == name and prompt.version == version:
                found = True
                break
        
        if not found:
            raise NotFoundError(f"Version {version} not found for {name}")
        
        self._versions[name] = version
        self.logger.info(
            "prompt_manager.version_updated",
            name=name,
            version=version,
        )
    
    # ------------------------------------------------------------------------
    # Specific Prompt Helpers
    # ------------------------------------------------------------------------
    
    async def get_summary_prompt(
        self,
        title: str,
        transcript: str,
        summary_type: str = "concise",
        language: Language = Language.ENGLISH,
    ) -> str:
        """
        Get summary prompt with common variables
        
        Args:
            title: Video title
            transcript: Video transcript
            summary_type: Type of summary (concise/detailed/bullet)
            language: Target language
        """
        return await self.get_prompt(
            f"summary/{summary_type}",
            title=title,
            transcript=transcript,
            language=language,
        )
    
    async def get_qa_prompt(
        self,
        question: str,
        context: str,
        language: Language = Language.ENGLISH,
        include_history: bool = False,
        history: Optional[str] = None,
    ) -> str:
        """
        Get Q&A prompt
        
        Args:
            question: User question
            context: Relevant transcript chunks
            language: Target language
            include_history: Whether to include conversation history
            history: Optional formatted conversation history text for the template
        """
        prompt_name = "qa/grounded"
        if include_history or history:
            prompt_name = "qa/with_history"
        
        kwargs = dict(question=question, context=context, language=language)
        if history is not None:
            kwargs["history"] = history
        return await self.get_prompt(prompt_name, **kwargs)
    
    async def get_translation_prompt(
        self,
        text: str,
        source_lang: Language,
        target_lang: Language,
    ) -> str:
        """
        Get translation prompt
        
        Args:
            text: Text to translate
            source_lang: Source language
            target_lang: Target language
        """
        return await self.get_prompt(
            "language/translate",
            text=text,
            source_language=source_lang.name,
            target_language=target_lang.name,
            language=target_lang,  # Prompt in target language
        )
    
    async def get_language_detection_prompt(self, text: str) -> str:
        """Get language detection prompt"""
        return await self.get_prompt(
            "language/detect",
            text=text,
        )
    
    # ------------------------------------------------------------------------
    # Prompt Testing and Validation
    # ------------------------------------------------------------------------
    
    async def validate_prompt(
        self,
        name: str,
        test_vars: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validate a prompt by rendering it
        
        Args:
            name: Prompt name
            test_vars: Test variables
            
        Returns:
            Validation results
        """
        try:
            # Get prompt info
            info = await self.get_prompt_info(name)
            
            # Generate test variables if not provided
            if not test_vars:
                test_vars = {}
                for var in info["variables"]:
                    test_vars[var] = f"test_{var}"
            
            # Try rendering
            rendered = await self.get_prompt(name, **test_vars)
            
            return {
                "valid": True,
                "name": name,
                "version": info["version"],
                "variables": info["variables"],
                "rendered_length": len(rendered),
                "rendered_preview": rendered[:200] + "..." if len(rendered) > 200 else rendered,
            }
            
        except Exception as e:
            return {
                "valid": False,
                "name": name,
                "error": str(e),
            }
    
    async def test_all_prompts(self) -> Dict[str, Any]:
        """Test all prompts with dummy data"""
        results = {}
        
        for key, prompt in self._prompts.items():
            # Generate test variables
            test_vars = {}
            for var in prompt.variables:
                if "title" in var:
                    test_vars[var] = "Test Video Title"
                elif "transcript" in var:
                    test_vars[var] = "This is a test transcript with some content."
                elif "question" in var:
                    test_vars[var] = "What is the main point?"
                elif "context" in var:
                    test_vars[var] = "Test context for Q&A."
                elif "language" in var:
                    test_vars[var] = "English"
                else:
                    test_vars[var] = f"test_{var}"
            
            try:
                rendered = prompt.render(**test_vars)
                results[key] = {
                    "success": True,
                    "length": len(rendered),
                }
            except Exception as e:
                results[key] = {
                    "success": False,
                    "error": str(e),
                }
        
        return results
    
    # ------------------------------------------------------------------------
    # Cache Management
    # ------------------------------------------------------------------------
    
    def clear_cache(self) -> None:
        """Clear prompt cache"""
        self._cache.clear()
        self.logger.debug("prompt_manager.cache_cleared")
    
    def reload_prompts(self) -> None:
        """Reload all prompts from disk"""
        self._prompts.clear()
        self._versions.clear()
        self.clear_cache()
        self._load_prompts()
        self.logger.info("prompt_manager.reloaded")


# ------------------------------------------------------------------------
# Example Prompt Templates
# ------------------------------------------------------------------------

"""
Example template files to create:

prompts/templates/
├── summary/
│   ├── concise_v1.txt
│   ├── detailed_v1.txt
│   ├── bullet_points_v1.txt
│   └── concise_v1.hi.txt  # Hindi version
├── qa/
│   ├── grounded_v1.txt
│   ├── with_history_v1.txt
│   └── grounded_v1.ta.txt  # Tamil version
├── language/
│   ├── translate_v1.txt
│   └── detect_v1.txt
└── active_versions.json

Example: prompts/templates/summary/concise_v1.txt
----------------------------------------
You are a YouTube video summarizer. Create a concise summary.

Video Title: {{title}}
Language: {{language}}

Transcript:
{{transcript}}

Instructions:
1. Extract the 5 most important key points
2. Include timestamps for each point in MM:SS format
3. Provide one core takeaway sentence
4. Keep it concise and clear

Format your response as JSON:
{
    "key_points": [
        {"point": "...", "timestamp": "MM:SS"}
    ],
    "core_takeaway": "..."
}
"""


# ------------------------------------------------------------------------
# Factory Function
# ------------------------------------------------------------------------

def create_prompt_manager(
    prompt_path: Optional[str] = None,
    logger=None,
    metrics=None,
) -> PromptManager:
    """
    Create prompt manager with default configuration
    
    Args:
        prompt_path: Path to prompt templates
        logger: Logger instance
        metrics: Metrics collector
        
    Returns:
        Configured PromptManager
    """
    path = Path(prompt_path) if prompt_path else None
    
    return PromptManager(
        prompt_path=path,
        logger=logger,
        metrics=metrics,
    )


# ------------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------------

"""
# Usage examples:

# Initialize
prompt_manager = create_prompt_manager()

# Get summary prompt
prompt = await prompt_manager.get_summary_prompt(
    title="How to Build AI Apps",
    transcript="This is the video transcript...",
    summary_type="concise",
    language=Language.ENGLISH,
)

# Get Q&A prompt
qa_prompt = await prompt_manager.get_qa_prompt(
    question="What are the pricing options?",
    context="Relevant transcript chunks...",
    language=Language.HINDI,
)

# Get translation prompt
trans_prompt = await prompt_manager.get_translation_prompt(
    text="Hello, how are you?",
    source_lang=Language.ENGLISH,
    target_lang=Language.TAMIL,
)

# List available prompts
prompts = prompt_manager.list_prompts(category="summary")
for p in prompts:
    print(f"{p['name']} v{p['version']}")

# Test a prompt
result = await prompt_manager.validate_prompt(
    "summary/concise",
    {"title": "Test", "transcript": "Test content"}
)
print(f"Valid: {result['valid']}")

# Change active version
prompt_manager.set_active_version("summary/concise", "v2")
"""

# ------------------------------------------------------------------------
# Benefits for Maintainability
# ------------------------------------------------------------------------

"""
Maintainability Benefits:

1. Single Source of Truth
   - All prompts in one place
   - No duplication across handlers
   - Consistent formatting

2. Version Control
   - Track prompt changes over time
   - Rollback to previous versions
   - A/B test different versions

3. Language Separation
   - Prompts in different languages
   - Easy to add new languages
   - Consistent translations

4. Testing
   - Validate all prompts
   - Catch missing variables
   - Preview rendered prompts

5. Metrics
   - Track prompt usage
   - Monitor performance
   - Identify popular prompts

6. Easy Updates
   - Change prompts without code deploy
   - Update multiple handlers at once
   - Experiment with different phrasings

7. Reusable Components
   - Common instructions in base prompts
   - Compose complex prompts from simple ones
   - DRY principle applied to prompts
"""