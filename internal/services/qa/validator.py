"""
Answer Validator
Validates that generated answers are grounded in the source context

Features:
- Fact extraction from answers
- Context verification
- Confidence scoring
- Hallucination detection
- Fallback responses
- Multi-language support
- Detailed validation reports
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import string

import structlog
import numpy as np

from internal.services.qa.retriever import RetrievedChunk
from internal.domain.value_objects import Language
from internal.pkg.metrics import MetricsCollector


class FactType(Enum):
    """Types of facts that can be extracted"""
    PRICING = "pricing"
    FEATURE = "feature"
    DATE = "date"
    NUMBER = "number"
    NAME = "name"
    LOCATION = "location"
    ACTION = "action"
    COMPARISON = "comparison"
    EXISTENCE = "existence"
    OTHER = "other"


@dataclass
class Fact:
    """An atomic fact extracted from an answer"""
    text: str
    type: FactType
    entities: List[str] = field(default_factory=list)
    importance: float = 1.0  # 0.0 to 1.0
    verified: bool = False
    confidence: float = 0.0
    source_chunks: List[int] = field(default_factory=list)  # Chunk IDs
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "type": self.type.value,
            "entities": self.entities,
            "importance": self.importance,
            "verified": self.verified,
            "confidence": self.confidence,
        }


@dataclass
class ValidationResult:
    """Result of answer validation"""
    answer: str
    facts: List[Fact]
    confidence: float
    is_hallucination: bool
    verified_facts: int
    total_facts: int
    coverage_ratio: float
    fallback_triggered: bool
    fallback_reason: Optional[str] = None
    fallback_answer: Optional[str] = None
    
    @property
    def summary(self) -> str:
        """Get human-readable summary"""
        if self.is_hallucination:
            return f"HALLUCINATION DETECTED: {self.verified_facts}/{self.total_facts} facts verified"
        return f"VALID: {self.verified_facts}/{self.total_facts} facts verified, confidence={self.confidence:.2f}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": self.confidence,
            "is_hallucination": self.is_hallucination,
            "verified_facts": self.verified_facts,
            "total_facts": self.total_facts,
            "coverage_ratio": self.coverage_ratio,
            "fallback_triggered": self.fallback_triggered,
            "facts": [f.to_dict() for f in self.facts],
        }


class FactExtractor:
    """
    Extracts atomic facts from answers for verification
    """
    
    # Patterns for different fact types
    PATTERNS = {
        FactType.PRICING: [
            r'\$\s*\d+(?:\.\d+)?',  # $49, $99.99
            r'(?:cost|price|paid|subscription|fee)s?\s+(\d+)',
            r'(?:free|paid|premium|basic|pro|enterprise)',
        ],
        FactType.NUMBER: [
            r'\d+(?:\.\d+)?',
            r'(?:hundred|thousand|million|billion)',
        ],
        FactType.DATE: [
            r'\d{4}-\d{2}-\d{2}',
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}',
            r'(?:today|yesterday|tomorrow|next week|last month)',
        ],
        FactType.NAME: [
            r'[A-Z][a-z]+ [A-Z][a-z]+',  # Person names
            r'(?:Mr\.|Ms\.|Dr\.) [A-Z][a-z]+',
            r'[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd)',
        ],
    }
    
    # Words that indicate importance
    IMPORTANCE_INDICATORS = {
        'important': 0.8,
        'key': 0.8,
        'critical': 0.9,
        'essential': 0.9,
        'main': 0.7,
        'primary': 0.7,
        'significant': 0.8,
        'major': 0.7,
    }
    
    def __init__(self, language: Language = Language.ENGLISH):
        self.language = language
    
    def extract_facts(self, answer: str) -> List[Fact]:
        """
        Extract atomic facts from answer
        
        Args:
            answer: Generated answer text
            
        Returns:
            List of extracted facts
        """
        facts = []
        
        # Split into sentences
        sentences = self._split_sentences(answer)
        
        for sentence in sentences:
            # Skip very short sentences
            if len(sentence.split()) < 3:
                continue
            
            # Detect fact type
            fact_type = self._detect_fact_type(sentence)
            
            # Extract entities
            entities = self._extract_entities(sentence)
            
            # Calculate importance
            importance = self._calculate_importance(sentence)
            
            facts.append(Fact(
                text=sentence,
                type=fact_type,
                entities=entities,
                importance=importance,
            ))
        
        return facts
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _detect_fact_type(self, sentence: str) -> FactType:
        """Detect the type of fact"""
        sentence_lower = sentence.lower()
        
        for fact_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, sentence_lower):
                    return fact_type
        
        return FactType.OTHER
    
    def _extract_entities(self, sentence: str) -> List[str]:
        """Extract named entities from sentence"""
        entities = []
        
        # Extract numbers
        numbers = re.findall(r'\d+(?:\.\d+)?', sentence)
        entities.extend(numbers)
        
        # Extract currency
        currencies = re.findall(r'[\$\€\£\₹]\s*\d+(?:\.\d+)?', sentence)
        entities.extend(currencies)
        
        # Extract quoted text
        quotes = re.findall(r'"([^"]*)"', sentence)
        entities.extend(quotes)
        
        return entities
    
    def _calculate_importance(self, sentence: str) -> float:
        """Calculate fact importance (0-1)"""
        sentence_lower = sentence.lower()
        
        # Default importance
        importance = 0.5
        
        # Adjust based on indicators
        for word, boost in self.IMPORTANCE_INDICATORS.items():
            if word in sentence_lower:
                importance = max(importance, boost)
        
        # Facts with numbers are more important
        if re.search(r'\d+', sentence):
            importance += 0.2
        
        # Facts with pricing terms are very important
        if re.search(r'price|cost|\$|€|£|₹', sentence_lower):
            importance += 0.3
        
        return min(1.0, importance)


class ContextVerifier:
    """
    Verifies facts against source context chunks
    """
    
    def __init__(self, similarity_threshold: float = 0.6):
        self.similarity_threshold = similarity_threshold
    
    async def verify_fact(
        self,
        fact: Fact,
        chunks: List[RetrievedChunk],
    ) -> Tuple[bool, float, List[int]]:
        """
        Verify a single fact against context chunks
        
        Returns:
            Tuple of (verified, confidence, chunk_ids)
        """
        best_score = 0.0
        supporting_chunks = []
        
        for chunk in chunks:
            # Check each chunk
            score = self._check_fact_in_chunk(fact, chunk)
            
            if score > best_score:
                best_score = score
            
            if score >= self.similarity_threshold:
                supporting_chunks.append(chunk.chunk.id)
        
        # Determine verification status
        verified = best_score >= self.similarity_threshold
        
        return verified, best_score, supporting_chunks
    
    def _check_fact_in_chunk(
        self,
        fact: Fact,
        chunk: RetrievedChunk,
    ) -> float:
        """
        Check if fact appears in chunk
        
        Uses multiple strategies:
        1. Exact match
        2. Entity presence
        3. Semantic similarity (simplified)
        """
        chunk_text = chunk.chunk.text.lower()
        fact_text = fact.text.lower()
        
        scores = []
        
        # Strategy 1: Exact match of key phrases
        if len(fact_text) > 20:
            # Extract key phrases (3-4 word sequences)
            fact_words = fact_text.split()
            for i in range(len(fact_words) - 2):
                phrase = " ".join(fact_words[i:i+3])
                if phrase in chunk_text:
                    scores.append(1.0)
                    break
        
        # Strategy 2: Entity presence
        if fact.entities:
            entity_matches = sum(1 for e in fact.entities if e.lower() in chunk_text)
            if entity_matches > 0:
                scores.append(entity_matches / len(fact.entities))
        
        # Strategy 3: Keyword overlap
        fact_keywords = set(fact_text.split())
        chunk_keywords = set(chunk_text.split())
        
        # Remove common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        fact_keywords -= stopwords
        chunk_keywords -= stopwords
        
        if fact_keywords:
            overlap = len(fact_keywords & chunk_keywords)
            keyword_score = overlap / len(fact_keywords)
            scores.append(keyword_score * 0.8)  # Weight keyword score
        
        # Return max score
        return max(scores) if scores else 0.0


class AnswerValidator:
    """
    Validates that answers are grounded in source context
    
    Features:
    - Fact extraction
    - Context verification
    - Confidence scoring
    - Hallucination detection
    - Fallback responses
    - Detailed validation reports
    """
    
    def __init__(
        self,
        fact_extractor: Optional[FactExtractor] = None,
        context_verifier: Optional[ContextVerifier] = None,
        logger=None,
        metrics=None,
        confidence_threshold: float = 0.7,
        hallucination_threshold: float = 0.3,
        enable_fallback: bool = True,
    ):
        """
        Initialize answer validator
        
        Args:
            fact_extractor: Fact extractor
            context_verifier: Context verifier
            logger: Structured logger
            metrics: Metrics collector
            confidence_threshold: Threshold for acceptable confidence
            hallucination_threshold: Threshold for hallucination detection
            enable_fallback: Whether to provide fallback answers
        """
        self.fact_extractor = fact_extractor or FactExtractor()
        self.context_verifier = context_verifier or ContextVerifier()
        self.logger = logger or structlog.get_logger(__name__)
        self.metrics = metrics or MetricsCollector("answer_validator")
        
        self.confidence_threshold = confidence_threshold
        self.hallucination_threshold = hallucination_threshold
        self.enable_fallback = enable_fallback
        
        self.logger.info(
            "answer_validator.initialized",
            confidence_threshold=confidence_threshold,
            hallucination_threshold=hallucination_threshold,
            enable_fallback=enable_fallback,
        )
    
    async def validate(
        self,
        answer: str,
        retrieved_chunks: List[RetrievedChunk],
        question: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate answer against source context
        
        Args:
            answer: Generated answer
            retrieved_chunks: Retrieved context chunks
            question: Original question (for context)
            
        Returns:
            Validation result
        """
        # Step 1: Extract facts
        facts = self.fact_extractor.extract_facts(answer)
        
        if not facts:
            # No facts to verify - treat as low confidence
            return ValidationResult(
                answer=answer,
                facts=[],
                confidence=0.0,
                is_hallucination=True,
                verified_facts=0,
                total_facts=0,
                coverage_ratio=0.0,
                fallback_triggered=False,
                fallback_reason="No facts to verify",
            )
        
        # Step 2: Verify each fact
        verified_count = 0
        total_importance = 0.0
        weighted_confidence = 0.0
        
        for fact in facts:
            verified, confidence, chunk_ids = await self.context_verifier.verify_fact(
                fact, retrieved_chunks
            )
            
            fact.verified = verified
            fact.confidence = confidence
            fact.source_chunks = chunk_ids
            
            if verified:
                verified_count += 1
            
            # Weighted confidence by importance
            total_importance += fact.importance
            weighted_confidence += confidence * fact.importance
        
        # Step 3: Calculate overall metrics
        coverage_ratio = verified_count / len(facts) if facts else 0
        avg_confidence = weighted_confidence / total_importance if total_importance > 0 else 0
        
        # Step 4: Determine if hallucination
        is_hallucination = (
            avg_confidence < self.hallucination_threshold or
            coverage_ratio < self.hallucination_threshold
        )
        
        # Step 5: Generate fallback if needed
        fallback_triggered = False
        fallback_reason = None
        fallback_answer = None
        
        if is_hallucination and self.enable_fallback:
            fallback_triggered = True
            fallback_reason = self._get_fallback_reason(avg_confidence, coverage_ratio)
            fallback_answer = await self._generate_fallback(
                question, retrieved_chunks, avg_confidence
            )
        
        # Step 6: Track metrics
        await self._track_validation_metrics(
            is_hallucination,
            avg_confidence,
            coverage_ratio,
            len(facts),
        )
        
        self.logger.debug(
            "answer_validator.completed",
            facts=len(facts),
            verified=verified_count,
            confidence=round(avg_confidence, 2),
            hallucination=is_hallucination,
        )
        
        return ValidationResult(
            answer=answer,
            facts=facts,
            confidence=avg_confidence,
            is_hallucination=is_hallucination,
            verified_facts=verified_count,
            total_facts=len(facts),
            coverage_ratio=coverage_ratio,
            fallback_triggered=fallback_triggered,
            fallback_reason=fallback_reason,
            fallback_answer=fallback_answer,
        )
    
    # ------------------------------------------------------------------------
    # Fallback Generation
    # ------------------------------------------------------------------------
    
    def _get_fallback_reason(
        self,
        confidence: float,
        coverage: float,
    ) -> str:
        """Get reason for fallback"""
        if confidence < 0.1:
            return "No supporting evidence found in video"
        elif confidence < 0.3:
            return "Very limited supporting evidence"
        elif coverage < 0.2:
            return "Most claims not found in video"
        else:
            return "Low confidence in answer accuracy"
    
    async def _generate_fallback(
        self,
        question: Optional[str],
        chunks: List[RetrievedChunk],
        confidence: float,
    ) -> str:
        """
        Generate fallback response when hallucination detected
        
        Provides partial information or suggests what is actually in the video
        """
        if not chunks:
            return "I couldn't find information about that in the video transcript."
        
        # Build response based on available chunks
        response_parts = []
        
        if confidence < 0.1:
            response_parts.append("I couldn't verify that information in the video.")
        else:
            response_parts.append("I found some related information in the video:")
        
        # Add relevant snippets from top chunks
        seen_texts = set()
        snippet_count = 0
        
        for chunk in chunks[:2]:  # Top 2 chunks
            if chunk.relevance_score > 0.6:  # Only relevant chunks
                # Extract a short snippet
                snippet = chunk.chunk.text[:150].strip()
                if snippet and snippet not in seen_texts:
                    if chunk.chunk.timestamp_range:
                        response_parts.append(f"\n• [{chunk.chunk.timestamp_range}] {snippet}...")
                    else:
                        response_parts.append(f"\n• {snippet}...")
                    seen_texts.add(snippet)
                    snippet_count += 1
        
        if snippet_count == 0:
            # No good snippets found
            return "The video mentions related topics, but I couldn't find specific information about your question."
        
        response_parts.append("\n\nWould you like to ask about something else from the video?")
        
        return "\n".join(response_parts)
    
    # ------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------
    
    async def _track_validation_metrics(
        self,
        is_hallucination: bool,
        confidence: float,
        coverage: float,
        fact_count: int,
    ) -> None:
        """Track validation metrics"""
        self.metrics.gauge(
            "validation.confidence",
            value=confidence,
            tags={"hallucination": str(is_hallucination)},
        )
        
        self.metrics.gauge(
            "validation.coverage",
            value=coverage,
            tags={"hallucination": str(is_hallucination)},
        )
        
        self.metrics.histogram(
            "validation.fact_count",
            value=fact_count,
        )
        
        if is_hallucination:
            self.metrics.increment("validation.hallucinations")
        else:
            self.metrics.increment("validation.valid")
    
    # ------------------------------------------------------------------------
    # Batch Validation
    # ------------------------------------------------------------------------
    
    async def validate_batch(
        self,
        answers: List[str],
        chunks_list: List[List[RetrievedChunk]],
        questions: Optional[List[str]] = None,
    ) -> List[ValidationResult]:
        """Validate multiple answers"""
        results = []
        for i, answer in enumerate(answers):
            chunks = chunks_list[i] if i < len(chunks_list) else []
            question = questions[i] if questions and i < len(questions) else None
            
            result = await self.validate(
                answer=answer,
                retrieved_chunks=chunks,
                question=question,
            )
            results.append(result)
        
        return results
    
    # ------------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------------
    
    def get_validation_report(self, result: ValidationResult) -> str:
        """Generate human-readable validation report"""
        lines = [
            f"Validation Report:",
            f"Confidence: {result.confidence:.2f}",
            f"Hallucination: {'YES' if result.is_hallucination else 'NO'}",
            f"Facts: {result.verified_facts}/{result.total_facts} verified",
            f"Coverage: {result.coverage_ratio:.2%}",
        ]
        
        if result.facts:
            lines.append("\nFact Details:")
            for i, fact in enumerate(result.facts[:5], 1):
                status = "✓" if fact.verified else "✗"
                lines.append(f"  {status} [{fact.type.value}] {fact.text[:50]}...")
        
        return "\n".join(lines)


# ------------------------------------------------------------------------
# Factory Function
# ------------------------------------------------------------------------

def create_validator(
    confidence_threshold: float = 0.7,
    hallucination_threshold: float = 0.3,
    enable_fallback: bool = True,
    logger=None,
    metrics=None,
) -> AnswerValidator:
    """
    Create answer validator with configuration
    
    Args:
        confidence_threshold: Threshold for acceptable confidence
        hallucination_threshold: Threshold for hallucination detection
        enable_fallback: Whether to provide fallback answers
        logger: Logger instance
        metrics: Metrics collector
        
    Returns:
        Configured AnswerValidator
    """
    fact_extractor = FactExtractor()
    context_verifier = ContextVerifier()
    
    return AnswerValidator(
        fact_extractor=fact_extractor,
        context_verifier=context_verifier,
        logger=logger,
        metrics=metrics,
        confidence_threshold=confidence_threshold,
        hallucination_threshold=hallucination_threshold,
        enable_fallback=enable_fallback,
    )


# ------------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------------

"""
# Usage example:

validator = create_validator()

# Validate an answer
result = await validator.validate(
    answer="The Basic plan costs $49 per month and includes 100 API calls.",
    retrieved_chunks=chunks,
    question="What are the pricing options?",
)

# Check if answer is valid
if result.is_hallucination:
    print("⚠️ Possible hallucination detected!")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Fallback: {result.fallback_answer}")
else:
    print(f"✅ Valid answer (confidence: {result.confidence:.2f})")
    print(result.answer)

# Get detailed report
report = validator.get_validation_report(result)
print(report)

# Access fact-level details
for fact in result.facts:
    print(f"{'✓' if fact.verified else '✗'} {fact.text}")
    print(f"  Confidence: {fact.confidence:.2f}")
    print(f"  Source chunks: {fact.source_chunks}")
"""

# ------------------------------------------------------------------------
# Example Output
# ------------------------------------------------------------------------

"""
Valid answer output:
{
    "confidence": 0.92,
    "is_hallucination": false,
    "verified_facts": 3,
    "total_facts": 3,
    "coverage_ratio": 1.0,
    "facts": [
        {
            "text": "The Basic plan costs $49 per month",
            "type": "pricing",
            "entities": ["Basic", "$49"],
            "importance": 0.8,
            "verified": true,
            "confidence": 0.95
        }
    ]
}

Hallucination detected:
{
    "confidence": 0.15,
    "is_hallucination": true,
    "verified_facts": 1,
    "total_facts": 4,
    "coverage_ratio": 0.25,
    "fallback_triggered": true,
    "fallback_reason": "No supporting evidence found in video",
    "fallback_answer": "I couldn't verify that information in the video.\n\n• [02:15] The video discusses pricing tiers including Basic, Pro, and Enterprise...\n\nWould you like to ask about something else from the video?"
}
"""