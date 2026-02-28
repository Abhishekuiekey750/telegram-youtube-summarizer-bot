"""
Prompt Template Loader
Loads prompt templates from files with versioning support

Features:
- File-based template storage
- Version management
- Automatic variable extraction
- Template caching
- Encoding detection
- Frontmatter support (YAML metadata)
- Fallback chains
- Hot reload capability
"""

import os
import re
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import asyncio
from functools import lru_cache

import structlog
import aiofiles

from internal.pkg.errors import NotFoundError, ValidationError
from internal.pkg.metrics import MetricsCollector


@dataclass
class TemplateMetadata:
    """Metadata for a template"""
    name: str
    version: str
    path: Path
    description: str = ""
    author: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    variables: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "variables": self.variables,
            "tags": self.tags,
            "hash": self.hash,
        }


@dataclass
class Template:
    """A loaded prompt template"""
    name: str
    version: str
    content: str
    metadata: TemplateMetadata
    variables: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Extract variables from content"""
        if not self.variables:
            self.variables = self._extract_variables()
    
    def _extract_variables(self) -> Set[str]:
        """Extract variable names from template {{var}}"""
        pattern = r'\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}'
        return set(re.findall(pattern, self.content))
    
    def render(self, **kwargs) -> str:
        """
        Render template with variables
        
        Args:
            **kwargs: Variable values
            
        Returns:
            Rendered template
            
        Raises:
            ValidationError: If missing required variables
        """
        # Check for missing variables
        missing = self.variables - set(kwargs.keys())
        if missing:
            raise ValidationError(
                f"Missing required variables: {missing}",
                context={"template": self.name, "missing": list(missing)},
            )
        
        # Render
        result = self.content
        for key, value in kwargs.items():
            placeholder = f'{{{{{key}}}}}'
            result = result.replace(placeholder, str(value))
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "metadata": self.metadata.to_dict(),
            "variables": list(self.variables),
            "content_preview": self.content[:100] + "..." if len(self.content) > 100 else self.content,
        }


class PromptLoader:
    """
    Loads prompt templates from files with versioning
    
    Directory structure:
    prompts/
    ├── v1/
    │   ├── category/
    │   │   └── name.txt
    │   └── ...
    ├── v2/
    │   └── ...
    ├── active_versions.json
    └── metadata.json
    """
    
    def __init__(
        self,
        prompts_dir: str = "prompts",
        encoding: str = "utf-8",
        enable_cache: bool = True,
        cache_ttl: int = 3600,  # 1 hour
        watch_for_changes: bool = False,
        logger=None,
        metrics=None,
    ):
        """
        Initialize prompt loader
        
        Args:
            prompts_dir: Root directory for prompt templates
            encoding: File encoding
            enable_cache: Whether to cache loaded templates
            cache_ttl: Cache TTL in seconds
            watch_for_changes: Whether to watch for file changes
            logger: Structured logger
            metrics: Metrics collector
        """
        self.prompts_dir = Path(prompts_dir)
        self.encoding = encoding
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        self.watch_for_changes = watch_for_changes
        self.logger = logger or structlog.get_logger(__name__)
        self.metrics = metrics or MetricsCollector("prompt_loader")
        
        # Cache: template_key -> (template, timestamp)
        self._cache: Dict[str, Tuple[Template, datetime]] = {}
        
        # Version mappings
        self._active_versions: Dict[str, str] = {}
        self._metadata: Dict[str, Any] = {}
        
        # File watcher task
        self._watcher_task = None
        self._running = False
        
        # Load initial data
        self._load_version_mappings()
        self._load_metadata()
        
        self.logger.info(
            "prompt_loader.initialized",
            prompts_dir=str(self.prompts_dir),
            encoding=encoding,
            enable_cache=enable_cache,
            watch_for_changes=watch_for_changes,
        )
        
        # Start file watcher if enabled
        if watch_for_changes:
            self._start_watcher()
    
    def _load_version_mappings(self):
        """Load active version mappings from file"""
        version_file = self.prompts_dir / "active_versions.json"
        
        if version_file.exists():
            try:
                with open(version_file, 'r', encoding=self.encoding) as f:
                    self._active_versions = json.load(f)
                self.logger.info(
                    "prompt_loader.versions_loaded",
                    count=len(self._active_versions),
                )
            except Exception as e:
                self.logger.error(
                    "prompt_loader.versions_load_failed",
                    error=str(e),
                )
    
    def _load_metadata(self):
        """Load template metadata"""
        metadata_file = self.prompts_dir / "metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding=self.encoding) as f:
                    self._metadata = json.load(f)
                self.logger.info(
                    "prompt_loader.metadata_loaded",
                    count=len(self._metadata),
                )
            except Exception as e:
                self.logger.error(
                    "prompt_loader.metadata_load_failed",
                    error=str(e),
                )
    
    def _start_watcher(self):
        """Start file watcher for hot reload"""
        self._running = True
        self._watcher_task = asyncio.create_task(self._watch_files())
        self.logger.debug("prompt_loader.watcher_started")
    
    async def _watch_files(self):
        """Watch for file changes and clear cache"""
        import watchfiles
        
        async for changes in watchfiles.awatch(str(self.prompts_dir)):
            if not self._running:
                break
            
            for change_type, file_path in changes:
                self.logger.info(
                    "prompt_loader.file_changed",
                    change_type=change_type,
                    file=file_path,
                )
                # Clear cache on any change
                self.clear_cache()
                
                # Reload version mappings if needed
                if file_path.endswith("active_versions.json"):
                    self._load_version_mappings()
                
                self.metrics.increment("file_changes")
    
    async def load_template(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> Template:
        """
        Load a template by name
        
        Args:
            name: Template name (e.g., "summary/concise")
            version: Specific version (uses active version if not specified)
            
        Returns:
            Loaded template
            
        Raises:
            NotFoundError: If template not found
            ValidationError: If template is invalid
        """
        # Determine version
        if not version:
            version = self._active_versions.get(name, "v1")
        
        # Check cache
        cache_key = f"{name}:{version}"
        if self.enable_cache and cache_key in self._cache:
            template, timestamp = self._cache[cache_key]
            age = (datetime.now() - timestamp).total_seconds()
            
            if age < self.cache_ttl:
                self.metrics.increment("cache.hit")
                return template
            else:
                # Expired
                del self._cache[cache_key]
        
        self.metrics.increment("cache.miss")
        
        # Construct file path
        file_path = self.prompts_dir / version / f"{name}.txt"
        
        if not file_path.exists():
            # Try without .txt extension
            file_path = self.prompts_dir / version / name
            if not file_path.exists():
                # Try fallback to v1
                if version != "v1":
                    self.logger.warning(
                        "prompt_loader.version_not_found",
                        name=name,
                        version=version,
                        falling_back="v1",
                    )
                    return await self.load_template(name, "v1")
                
                raise NotFoundError(
                    f"Template not found: {name} (version: {version})",
                    context={"name": name, "version": version},
                )
        
        try:
            # Read file
            async with aiofiles.open(file_path, 'r', encoding=self.encoding) as f:
                content = await f.read()
            
            # Parse frontmatter if present
            metadata, content = self._parse_frontmatter(content)
            
            # Create template metadata
            template_metadata = TemplateMetadata(
                name=name,
                version=version,
                path=file_path,
                description=metadata.get("description", ""),
                author=metadata.get("author", ""),
                tags=metadata.get("tags", []),
                hash=hashlib.md5(content.encode()).hexdigest(),
            )
            
            # Create template
            template = Template(
                name=name,
                version=version,
                content=content,
                metadata=template_metadata,
            )
            
            # Cache
            if self.enable_cache:
                self._cache[cache_key] = (template, datetime.now())
            
            self.logger.debug(
                "prompt_loader.template_loaded",
                name=name,
                version=version,
                variables=len(template.variables),
                size=len(content),
            )
            
            self.metrics.increment(
                "templates.loaded",
                tags={"version": version},
            )
            
            return template
            
        except Exception as e:
            self.logger.error(
                "prompt_loader.load_failed",
                name=name,
                version=version,
                error=str(e),
            )
            self.metrics.increment("load_errors")
            raise ValidationError(f"Failed to load template: {str(e)}") from e
    
    def _parse_frontmatter(self, content: str) -> Tuple[Dict[str, Any], str]:
        """
        Parse YAML frontmatter from template
        
        Format:
        ---
        description: Template description
        author: John Doe
        tags: [summary, concise]
        ---
        Template content...
        """
        if not content.startswith('---'):
            return {}, content
        
        parts = content.split('---', 2)
        if len(parts) < 3:
            return {}, content
        
        try:
            metadata = yaml.safe_load(parts[1])
            content = parts[2].lstrip()
            return metadata or {}, content
        except Exception as e:
            self.logger.warning("prompt_loader.frontmatter_parse_failed", error=str(e))
            return {}, content
    
    async def load_batch(
        self,
        templates: List[Tuple[str, Optional[str]]],
    ) -> Dict[str, Template]:
        """
        Load multiple templates in parallel
        
        Args:
            templates: List of (name, version) tuples
            
        Returns:
            Dictionary mapping template key to Template
        """
        tasks = []
        for name, version in templates:
            tasks.append(self.load_template(name, version))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        loaded = {}
        for (name, version), result in zip(templates, results):
            key = f"{name}:{version or 'default'}"
            if isinstance(result, Exception):
                self.logger.error(
                    "prompt_loader.batch_failed",
                    name=name,
                    error=str(result),
                )
            else:
                loaded[key] = result
        
        return loaded
    
    def set_active_version(self, name: str, version: str):
        """Set active version for a template"""
        self._active_versions[name] = version
        self._save_active_versions()
        
        # Clear cache for this template
        keys_to_remove = [
            k for k in self._cache.keys()
            if k.startswith(f"{name}:")
        ]
        for key in keys_to_remove:
            del self._cache[key]
        
        self.logger.info(
            "prompt_loader.version_set",
            name=name,
            version=version,
        )
    
    def _save_active_versions(self):
        """Save active versions to file"""
        version_file = self.prompts_dir / "active_versions.json"
        try:
            with open(version_file, 'w', encoding=self.encoding) as f:
                json.dump(self._active_versions, f, indent=2)
        except Exception as e:
            self.logger.error(
                "prompt_loader.versions_save_failed",
                error=str(e),
            )
    
    def get_active_version(self, name: str) -> str:
        """Get active version for a template"""
        return self._active_versions.get(name, "v1")
    
    def list_templates(
        self,
        version: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List available templates"""
        templates = []
        
        version_dirs = [d for d in self.prompts_dir.iterdir() if d.is_dir()]
        
        for v_dir in version_dirs:
            v_name = v_dir.name
            if version and v_name != version:
                continue
            
            for txt_file in v_dir.rglob("*.txt"):
                rel_path = txt_file.relative_to(v_dir)
                name = str(rel_path.with_suffix('')).replace('\\', '/')
                
                # Get metadata from file
                try:
                    with open(txt_file, 'r', encoding=self.encoding) as f:
                        content = f.read()
                    metadata, _ = self._parse_frontmatter(content)
                    
                    # Filter by tag
                    if tag and tag not in metadata.get("tags", []):
                        continue
                    
                    templates.append({
                        "name": name,
                        "version": v_name,
                        "path": str(txt_file),
                        "active": self.get_active_version(name) == v_name,
                        "description": metadata.get("description", ""),
                        "tags": metadata.get("tags", []),
                    })
                except Exception as e:
                    self.logger.warning(
                        "prompt_loader.list_scan_failed",
                        file=str(txt_file),
                        error=str(e),
                    )
        
        return templates
    
    def clear_cache(self):
        """Clear template cache"""
        self._cache.clear()
        self.logger.debug("prompt_loader.cache_cleared")
    
    async def reload_all(self):
        """Reload all templates"""
        self.clear_cache()
        self._load_version_mappings()
        self._load_metadata()
        self.logger.info("prompt_loader.reloaded")
    
    async def close(self):
        """Clean up resources"""
        self._running = False
        if self._watcher_task:
            self._watcher_task.cancel()
            try:
                await self._watcher_task
            except asyncio.CancelledError:
                pass
        self.clear_cache()
        self.logger.info("prompt_loader.closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics"""
        return {
            "cached_templates": len(self._cache),
            "active_versions": len(self._active_versions),
            "prompts_dir": str(self.prompts_dir),
            "encoding": self.encoding,
            "cache_enabled": self.enable_cache,
            "watching": self.watch_for_changes,
        }


# ------------------------------------------------------------------------
# Factory Function
# ------------------------------------------------------------------------

def create_prompt_loader(
    prompts_dir: str = "prompts",
    encoding: str = "utf-8",
    enable_cache: bool = True,
    watch_for_changes: bool = False,
    logger=None,
    metrics=None,
) -> PromptLoader:
    """
    Create prompt loader with configuration
    
    Args:
        prompts_dir: Directory containing prompt templates
        encoding: File encoding
        enable_cache: Whether to cache templates
        watch_for_changes: Whether to watch for file changes
        logger: Logger instance
        metrics: Metrics collector
        
    Returns:
        Configured PromptLoader
    """
    return PromptLoader(
        prompts_dir=prompts_dir,
        encoding=encoding,
        enable_cache=enable_cache,
        watch_for_changes=watch_for_changes,
        logger=logger,
        metrics=metrics,
    )


# ------------------------------------------------------------------------
# Example Template Files
# ------------------------------------------------------------------------

"""
Example template files to create:

prompts/
├── v1/
│   ├── summary/
│   │   ├── concise.txt
│   │   └── detailed.txt
│   └── qa/
│       └── grounded.txt
├── v2/
│   └── summary/
│       └── concise.txt
├── active_versions.json
└── metadata.json

prompts/v1/summary/concise.txt:
---
description: Concise video summary with 5 key points
author: System
tags: [summary, concise]
---
You are a YouTube video summarizer. Create a concise summary.

Video Title: {{title}}
Language: {{language}}

Transcript:
{{transcript}}

Instructions:
1. Extract the 5 most important key points
2. Include timestamps for each point in MM:SS format
3. Provide one core takeaway sentence

Format your response as JSON:
{
    "key_points": [
        {"point": "...", "timestamp": "MM:SS"}
    ],
    "core_takeaway": "..."
}

prompts/v2/summary/concise.txt:
---
description: Improved concise summary with examples
author: System
tags: [summary, concise, improved]
version: 2.0
---
You are a YouTube video summarizer. Create a concise summary.

Video Title: {{title}}
Language: {{language}}

Transcript:
{{transcript}}

Example good summary:
{
    "key_points": [
        {"point": "The product costs $49/month", "timestamp": "02:30"},
        {"point": "Includes unlimited API calls", "timestamp": "05:45"}
    ],
    "core_takeaway": "Great value for developers"
}

Instructions:
1. Extract the 5 most important key points
2. Include timestamps for each point in MM:SS format
3. Provide one core takeaway sentence
4. Be specific and factual

Format your response as JSON.

active_versions.json:
{
    "summary/concise": "v2",
    "summary/detailed": "v1",
    "qa/grounded": "v1"
}
"""

# ------------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------------

"""
# Usage examples:

loader = create_prompt_loader(
    prompts_dir="prompts",
    watch_for_changes=True,  # Hot reload in development
)

# Load a template
template = await loader.load_template("summary/concise")
print(f"Template variables: {template.variables}")

# Render with variables
rendered = template.render(
    title="My Video",
    language="English",
    transcript="Video transcript here...",
)

# Load specific version
v1_template = await loader.load_template("summary/concise", version="v1")

# Set active version
loader.set_active_version("summary/concise", "v2")

# List available templates
templates = loader.list_templates(version="v1")
for t in templates:
    print(f"{t['name']} (v{t['version']}) - {t['description']}")

# Batch load
templates = await loader.load_batch([
    ("summary/concise", None),
    ("qa/grounded", "v1"),
])

# Get stats
stats = loader.get_stats()
print(f"Cached: {stats['cached_templates']}")

# Clean up
await loader.close()
"""