"""
Context Retriever for Q&A
Retrieves relevant transcript chunks using hybrid search

Features:
- Hybrid search (semantic + keyword)
- Configurable relevance thresholds
- Query expansion
- Metadata filtering
- Hallucination prevention
- Confidence scoring
- Source tracking
"""

import asyncio
import re
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
import numpy as np

import structlog

from internal.services.qa.chunker import Chunk
from internal.ai.embedding import EmbeddingGenerator
from internal.storage.vector import VectorDBClient
from internal.pkg.errors import ValidationError, NotFoundError
from internal.pkg.metrics import MetricsCollector


@dataclass
class RetrievedChunk:
    """A chunk with relevance information"""
    chunk: Chunk
    relevance_score: float
    vector_score: float = 0.0
    keyword_score: float = 0.0
    matched_terms: List[str] = field(default_factory=list)
    
    @property
    def is_relevant(self) -> bool:
        """Check if chunk meets relevance threshold"""
        return self.relevance_score >= 0.5  # Configurable threshold


class BM25:
    """
    BM25 ranking function for keyword search
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.avg_doc_len = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.vocab = set()
    
    def fit(self, corpus: List[str]):
        """Fit BM25 on corpus"""
        self.corpus = corpus
        self.doc_len = [len(doc.split()) for doc in corpus]
        self.avg_doc_len = sum(self.doc_len) / len(corpus)
        
        # Calculate document frequencies
        doc_freqs = []
        for doc in corpus:
            words = set(doc.lower().split())
            doc_freqs.append(words)
            self.vocab.update(words)
        
        # Calculate IDF
        N = len(corpus)
        for term in self.vocab:
            df = sum(1 for doc_words in doc_freqs if term in doc_words)
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)
    
    def score(self, query: str, doc_index: int) -> float:
        """Score document against query"""
        score = 0.0
        doc_words = self.corpus[doc_index].lower().split()
        doc_len = len(doc_words)
        
        for term in query.lower().split():
            if term not in self.idf:
                continue
            
            # Term frequency in document
            tf = doc_words.count(term)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += self.idf[term] * (numerator / denominator)
        
        return score


class QueryAnalyzer:
    """
    Analyzes and expands queries for better retrieval
    """
    
    # Common question patterns and expansions
    QUESTION_PATTERNS = {
        "pricing": ["price", "cost", "paid", "subscription", "fee", "plan", "tier"],
        "features": ["feature", "capability", "function", "can do", "ability"],
        "how_to": ["how", "steps", "process", "way", "method", "guide"],
        "why": ["reason", "purpose", "explanation", "cause"],
        "when": ["time", "date", "schedule", "release"],
        "where": ["location", "place", "website", "link"],
        "who": ["person", "people", "team", "founder", "ceo"],
        "comparison": ["vs", "versus", "compare", "difference", "better", "worse"],
    }
    
    # Stopwords to ignore
    STOPWORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
        'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
        'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
        'to', 'from', 'in', 'on', 'at', 'by', 'with', 'without', 'after', 'before'
    }
    
    def __init__(self, enable_expansion: bool = True):
        self.enable_expansion = enable_expansion
    
    def analyze(self, question: str) -> Dict[str, Any]:
        """
        Analyze question for better retrieval
        
        Returns:
            Dictionary with:
            - tokens: Cleaned tokens
            - question_type: Detected type (pricing, features, etc.)
            - expanded_terms: Related terms for expansion
            - keywords: Important keywords
        """
        # Clean and tokenize
        tokens = self._tokenize(question)
        
        # Detect question type
        q_type = self._detect_type(question, tokens)
        
        # Extract keywords (remove stopwords)
        keywords = [t for t in tokens if t not in self.STOPWORDS]
        
        # Generate expanded terms
        expanded_terms = []
        if self.enable_expansion and q_type:
            expanded_terms = self.QUESTION_PATTERNS.get(q_type, [])
        
        return {
            "original": question,
            "tokens": tokens,
            "question_type": q_type,
            "keywords": keywords,
            "expanded_terms": expanded_terms,
            "all_terms": list(set(keywords + expanded_terms)),
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and clean text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split and filter
        tokens = text.split()
        
        return tokens
    
    def _detect_type(self, question: str, tokens: List[str]) -> Optional[str]:
        """Detect question type from patterns"""
        question_lower = question.lower()
        
        for q_type, patterns in self.QUESTION_PATTERNS.items():
            for pattern in patterns:
                if pattern in question_lower:
                    return q_type
        
        return None


class ContextRetriever:
    """
    Retrieves relevant context chunks for questions using hybrid search
    
    Features:
    - Hybrid search (vector + keyword)
    - Query expansion
    - Relevance scoring
    - Hallucination prevention through strict thresholds
    - Source tracking with timestamps
    - Confidence estimation
    """
    
    def __init__(
        self,
        vector_db: VectorDBClient,
        embedding_generator: EmbeddingGenerator,
        query_analyzer: Optional[QueryAnalyzer] = None,
        logger=None,
        metrics=None,
        vector_weight: float = 0.7,  # Weight for vector search
        keyword_weight: float = 0.3,  # Weight for keyword search
        min_relevance_score: float = 0.5,
        max_chunks: int = 5,
        enable_query_expansion: bool = True,
    ):
        """
        Initialize context retriever
        
        Args:
            vector_db: Vector database client
            embedding_generator: Embedding generator
            query_analyzer: Query analyzer
            logger: Structured logger
            metrics: Metrics collector
            vector_weight: Weight for vector search results
            keyword_weight: Weight for keyword search
            min_relevance_score: Minimum score to consider relevant
            max_chunks: Maximum chunks to retrieve
            enable_query_expansion: Whether to expand queries
        """
        self.vector_db = vector_db
        self.embedding_generator = embedding_generator
        self.query_analyzer = query_analyzer or QueryAnalyzer(enable_expansion=enable_query_expansion)
        self.logger = logger or structlog.get_logger(__name__)
        self.metrics = metrics or MetricsCollector("retriever")
        
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.min_relevance_score = min_relevance_score
        self.max_chunks = max_chunks
        
        # BM25 index cache (video_id -> BM25)
        self._bm25_cache: Dict[str, BM25] = {}
        
        self.logger.info(
            "context_retriever.initialized",
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            min_score=min_relevance_score,
            max_chunks=max_chunks,
        )
    
    async def retrieve(
        self,
        question: str,
        video_id: str,
        chunks: Optional[List[Chunk]] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for question
        
        Args:
            question: User's question
            video_id: Video ID
            chunks: Optional pre-loaded chunks (if not provided, fetch from DB)
            filter_metadata: Optional metadata filters
            
        Returns:
            List of retrieved chunks with relevance scores
        """
        # Step 1: Analyze question
        analysis = self.query_analyzer.analyze(question)
        
        self.logger.debug(
            "retriever.analyzed_question",
            video_id=video_id,
            question_type=analysis["question_type"],
            keywords=analysis["keywords"],
            expanded_terms=analysis["expanded_terms"],
        )
        
        # Step 2: Get chunks if not provided
        if not chunks:
            chunks = await self._fetch_chunks(video_id)
            if not chunks:
                self.logger.warning(
                    "retriever.no_chunks",
                    video_id=video_id,
                )
                return []
        
        # Step 3: Apply metadata filters
        if filter_metadata:
            chunks = self._filter_chunks(chunks, filter_metadata)
        
        # Step 4: Perform hybrid search
        retrieved = await self._hybrid_search(question, analysis, chunks, video_id)
        
        # Step 5: Filter by relevance threshold
        relevant = [r for r in retrieved if r.relevance_score >= self.min_relevance_score]
        
        # Step 6: Limit to max chunks
        relevant = relevant[:self.max_chunks]
        
        self.logger.info(
            "retriever.completed",
            video_id=video_id,
            candidates=len(retrieved),
            relevant=len(relevant),
            top_score=relevant[0].relevance_score if relevant else 0,
        )
        
        if self.metrics:
            self.metrics.record_distribution(
                "retriever.relevance_scores",
                [r.relevance_score for r in retrieved],
            )
            self.metrics.gauge(
                "retriever.chunks_retrieved",
                value=len(relevant),
            )
        
        return relevant
    
    # ------------------------------------------------------------------------
    # Hybrid Search
    # ------------------------------------------------------------------------
    
    async def _hybrid_search(
        self,
        question: str,
        analysis: Dict[str, Any],
        chunks: List[Chunk],
        video_id: str,
    ) -> List[RetrievedChunk]:
        """
        Perform hybrid search combining vector and keyword approaches
        """
        # Vector search (semantic)
        vector_scores = await self._vector_search(question, chunks, video_id)
        
        # Keyword search (BM25)
        keyword_scores = await self._keyword_search(analysis, chunks, video_id)
        
        # Combine scores
        retrieved = []
        for i, chunk in enumerate(chunks):
            vector_score = vector_scores.get(i, 0.0)
            keyword_score = keyword_scores.get(i, 0.0)
            
            # Weighted combination
            combined_score = (
                self.vector_weight * vector_score +
                self.keyword_weight * keyword_score
            )
            
            # Find matched terms (for explainability)
            matched_terms = self._find_matched_terms(
                analysis["all_terms"],
                chunk.text,
            )
            
            retrieved.append(RetrievedChunk(
                chunk=chunk,
                relevance_score=combined_score,
                vector_score=vector_score,
                keyword_score=keyword_score,
                matched_terms=matched_terms,
            ))
        
        # Sort by relevance
        retrieved.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return retrieved
    
    async def _vector_search(
        self,
        question: str,
        chunks: List[Chunk],
        video_id: str,
    ) -> Dict[int, float]:
        """
        Perform semantic search using embeddings
        
        Returns:
            Dictionary mapping chunk index to similarity score
        """
        try:
            # Generate embedding for question
            question_embedding = await self.embedding_generator.embed(question)
            
            # Search in vector DB
            results = await self.vector_db.search_similar(
                video_id=video_id,
                query_embedding=question_embedding,
                limit=len(chunks),  # Get all for scoring
            )
            
            # Map results to chunk indices
            scores = {}
            for result in results:
                chunk_id = result.get("chunk_id")
                if chunk_id is not None:
                    # Find chunk index
                    for i, chunk in enumerate(chunks):
                        if chunk.id == chunk_id:
                            scores[i] = result.get("similarity", 0.0)
                            break
            
            return scores
            
        except Exception as e:
            self.logger.warning(
                "retriever.vector_search_failed",
                video_id=video_id,
                error=str(e),
            )
            return {}
    
    async def _keyword_search(
        self,
        analysis: Dict[str, Any],
        chunks: List[Chunk],
        video_id: str,
    ) -> Dict[int, float]:
        """
        Perform keyword search using BM25
        
        Returns:
            Dictionary mapping chunk index to BM25 score
        """
        try:
            # Get or create BM25 index for this video
            bm25 = await self._get_bm25(video_id, chunks)
            
            # Build query from all terms
            query = " ".join(analysis["all_terms"])
            
            # Score all chunks
            scores = {}
            for i, chunk in enumerate(chunks):
                # Find chunk in BM25 corpus
                for j, doc in enumerate(bm25.corpus):
                    if chunk.text in doc:  # Simple matching
                        score = bm25.score(query, j)
                        scores[i] = score
                        break
            
            # Normalize scores to 0-1 range
            if scores:
                max_score = max(scores.values())
                if max_score > 0:
                    scores = {i: s / max_score for i, s in scores.items()}
            
            return scores
            
        except Exception as e:
            self.logger.warning(
                "retriever.keyword_search_failed",
                video_id=video_id,
                error=str(e),
            )
            return {}
    
    def _find_matched_terms(
        self,
        terms: List[str],
        text: str,
    ) -> List[str]:
        """Find which terms matched in the text"""
        text_lower = text.lower()
        matched = []
        
        for term in terms:
            if term in text_lower:
                matched.append(term)
        
        return matched
    
    # ------------------------------------------------------------------------
    # BM25 Management
    # ------------------------------------------------------------------------
    
    async def _get_bm25(
        self,
        video_id: str,
        chunks: List[Chunk],
    ) -> BM25:
        """Get or create BM25 index for video"""
        if video_id in self._bm25_cache:
            return self._bm25_cache[video_id]
        
        # Create new BM25 index
        corpus = [chunk.text for chunk in chunks]
        bm25 = BM25()
        bm25.fit(corpus)
        
        self._bm25_cache[video_id] = bm25
        
        return bm25
    
    # ------------------------------------------------------------------------
    # Chunk Management
    # ------------------------------------------------------------------------
    
    async def _fetch_chunks(self, video_id: str) -> List[Chunk]:
        """Fetch chunks from vector DB and convert to Chunk objects for retrieval."""
        try:
            raw = await self.vector_db.get_chunks(video_id)
            out: List[Chunk] = []
            for i, c in enumerate(raw):
                cid = c.get("chunk_id", i)
                text = c.get("text", "")
                start = c.get("start_time", 0.0)
                end = c.get("end_time", 0.0)
                out.append(
                    Chunk(
                        id=cid,
                        text=text,
                        sentences=[],
                        start_time=start,
                        end_time=end,
                        metadata=c.get("metadata", {}),
                    )
                )
            return out
        except Exception as e:
            self.logger.error(
                "retriever.fetch_chunks_failed",
                video_id=video_id,
                error=str(e),
            )
            return []
    
    def _filter_chunks(
        self,
        chunks: List[Chunk],
        filters: Dict[str, Any],
    ) -> List[Chunk]:
        """Apply metadata filters to chunks"""
        filtered = []
        
        for chunk in chunks:
            match = True
            for key, value in filters.items():
                if key not in chunk.metadata:
                    match = False
                    break
                if chunk.metadata[key] != value:
                    match = False
                    break
            
            if match:
                filtered.append(chunk)
        
        return filtered
    
    # ------------------------------------------------------------------------
    # Hallucination Prevention
    # ------------------------------------------------------------------------
    
    async def verify_answer(
        self,
        answer: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> Tuple[bool, float]:
        """
        Verify that answer is grounded in retrieved chunks
        
        Returns:
            Tuple of (is_grounded, confidence_score)
        """
        if not retrieved_chunks:
            return False, 0.0
        
        # Combine all chunk text
        context = " ".join([r.chunk.text for r in retrieved_chunks]).lower()
        answer_lower = answer.lower()
        
        # Check if key terms from answer appear in context
        answer_tokens = set(answer_lower.split())
        context_tokens = set(context.split())
        
        # Remove common words
        stopwords = QueryAnalyzer.STOPWORDS
        answer_tokens = {t for t in answer_tokens if t not in stopwords and len(t) > 3}
        
        if not answer_tokens:
            return True, 1.0  # No substantive tokens to check
        
        # Calculate overlap
        overlap = answer_tokens & context_tokens
        overlap_ratio = len(overlap) / len(answer_tokens)
        
        # Check for key phrases (3+ word sequences)
        answer_words = answer_lower.split()
        key_phrases = []
        for i in range(len(answer_words) - 2):
            phrase = " ".join(answer_words[i:i+3])
            if len(phrase) > 10:  # Only check meaningful phrases
                key_phrases.append(phrase)
        
        if key_phrases:
            phrase_matches = sum(1 for p in key_phrases if p in context)
            phrase_ratio = phrase_matches / len(key_phrases)
            
            # Combine scores
            confidence = (overlap_ratio * 0.4) + (phrase_ratio * 0.6)
        else:
            confidence = overlap_ratio
        
        # Determine if grounded
        is_grounded = confidence >= 0.3  # Threshold for grounding
        
        return is_grounded, confidence
    
    async def estimate_confidence(
        self,
        retrieved_chunks: List[RetrievedChunk],
    ) -> float:
        """
        Estimate confidence in retrieved context
        
        Based on:
        - Relevance scores
        - Number of chunks
        - Consistency between chunks
        """
        if not retrieved_chunks:
            return 0.0
        
        # Average relevance score
        avg_score = sum(r.relevance_score for r in retrieved_chunks) / len(retrieved_chunks)
        
        # Score consistency (low variance is good)
        if len(retrieved_chunks) > 1:
            scores = [r.relevance_score for r in retrieved_chunks]
            variance = np.var(scores)
            consistency = 1.0 / (1.0 + variance)  # Normalize
        else:
            consistency = 1.0
        
        # Number of chunks factor (more chunks = more evidence)
        chunk_factor = min(1.0, len(retrieved_chunks) / 3.0)
        
        # Combined confidence
        confidence = (
            avg_score * 0.5 +
            consistency * 0.3 +
            chunk_factor * 0.2
        )
        
        return min(1.0, confidence)
    
    # ------------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------------
    
    async def format_context(
        self,
        retrieved_chunks: List[RetrievedChunk],
        include_timestamps: bool = True,
        max_length: Optional[int] = None,
    ) -> str:
        """
        Format retrieved chunks for LLM prompt
        
        Args:
            retrieved_chunks: Retrieved chunks
            include_timestamps: Whether to include timestamps
            max_length: Maximum context length (truncate if needed)
            
        Returns:
            Formatted context string
        """
        if not retrieved_chunks:
            return ""
        
        parts = []
        total_length = 0
        
        for i, r in enumerate(retrieved_chunks, 1):
            chunk = r.chunk
            
            if include_timestamps:
                part = f"[Section {i} - {chunk.timestamp_range}]\n{chunk.text}\n"
            else:
                part = f"{chunk.text}\n"
            
            part_length = len(part)
            
            if max_length and total_length + part_length > max_length:
                # Truncate this chunk
                remaining = max_length - total_length
                if remaining > 50:  # Only include if meaningful
                    parts.append(part[:remaining] + "...\n")
                break
            
            parts.append(part)
            total_length += part_length
        
        return "\n".join(parts)
    
    def clear_cache(self) -> None:
        """Clear BM25 cache"""
        self._bm25_cache.clear()
        self.logger.debug("retriever.cache_cleared")


# ------------------------------------------------------------------------
# Factory Function
# ------------------------------------------------------------------------

def create_retriever(
    vector_db: VectorDBClient,
    embedding_generator: EmbeddingGenerator,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
    min_relevance_score: float = 0.5,
    max_chunks: int = 5,
    logger=None,
    metrics=None,
) -> ContextRetriever:
    """
    Create context retriever with configuration
    
    Args:
        vector_db: Vector database client
        embedding_generator: Embedding generator
        vector_weight: Weight for vector search
        keyword_weight: Weight for keyword search
        min_relevance_score: Minimum relevance score
        max_chunks: Maximum chunks to retrieve
        logger: Logger instance
        metrics: Metrics collector
        
    Returns:
        Configured ContextRetriever
    """
    return ContextRetriever(
        vector_db=vector_db,
        embedding_generator=embedding_generator,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
        min_relevance_score=min_relevance_score,
        max_chunks=max_chunks,
        logger=logger,
        metrics=metrics,
    )


# ------------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------------

"""
# Usage example:

retriever = create_retriever(vector_db, embedding_generator)

# Retrieve context for question
chunks = await retriever.retrieve(
    question="What are the pricing options?",
    video_id="abc123",
)

# Format for LLM
context = await retriever.format_context(chunks)

# Verify answer grounding
is_grounded, confidence = await retriever.verify_answer(
    answer="Pricing starts at $49/month",
    retrieved_chunks=chunks,
)

if not is_grounded:
    print("Warning: Answer may not be grounded in context")

# Estimate confidence in retrieval
confidence = await retriever.estimate_confidence(chunks)
print(f"Retrieval confidence: {confidence:.2f}")
"""