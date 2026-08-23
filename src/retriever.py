"""Hybrid Retrieval and Context Assembly Engine for Knowledge Assistant.

This module provides the hybrid retriever, scoring fusion (dense similarity,
keyword frequency, prefix matching), confidence calibration, and structured context formatting.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from src.config import IngestionConfig, default_config
from src.embed_store import BaseEmbeddingModel, VectorStore, get_default_embedding_model
from src.ingest import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievalHit:
    """Represents a scored and ranked chunk retrieved for a search query."""

    chunk: DocumentChunk
    dense_score: float = 0.0
    keyword_score: float = 0.0
    prefix_score: float = 0.0
    hybrid_score: float = 0.0
    confidence_score: float = 0.0
    rank: int = 0

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def source(self) -> str:
        return self.chunk.source

    @property
    def section_header(self) -> Optional[str]:
        return self.chunk.section_header

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.chunk.metadata

    def to_dict(self) -> Dict[str, Any]:
        """Serialize hit into a clean dictionary."""
        return {
            "chunk": self.chunk.to_dict(),
            "dense_score": round(self.dense_score, 4),
            "keyword_score": round(self.keyword_score, 4),
            "prefix_score": round(self.prefix_score, 4),
            "hybrid_score": round(self.hybrid_score, 4),
            "confidence_score": round(self.confidence_score, 4),
            "rank": self.rank,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalHit":
        """Deserialize hit from dictionary."""
        return cls(
            chunk=DocumentChunk.from_dict(data["chunk"]),
            dense_score=data.get("dense_score", 0.0),
            keyword_score=data.get("keyword_score", 0.0),
            prefix_score=data.get("prefix_score", 0.0),
            hybrid_score=data.get("hybrid_score", 0.0),
            confidence_score=data.get("confidence_score", 0.0),
            rank=data.get("rank", 0),
        )


@dataclass
class RetrievalResult:
    """Represents the complete output of a retrieval operation."""

    query: str
    hits: List[RetrievalHit] = field(default_factory=list)
    total_hits: int = 0
    execution_time_ms: float = 0.0
    is_confident: bool = True
    top_confidence: float = 0.0
    formatted_context: str = ""
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize retrieval result into dictionary."""
        return {
            "query": self.query,
            "hits": [hit.to_dict() for hit in self.hits],
            "total_hits": self.total_hits,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "is_confident": self.is_confident,
            "top_confidence": round(self.top_confidence, 4),
            "formatted_context": self.formatted_context,
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalResult":
        """Deserialize retrieval result from dictionary."""
        hits = [RetrievalHit.from_dict(h) for h in data.get("hits", [])]
        return cls(
            query=data["query"],
            hits=hits,
            total_hits=data.get("total_hits", len(hits)),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            is_confident=data.get("is_confident", True),
            top_confidence=data.get("top_confidence", 0.0),
            formatted_context=data.get("formatted_context", ""),
            rejection_reason=data.get("rejection_reason"),
        )


def tokenize_query(query: str) -> List[str]:
    """Tokenize and filter search terms into alphanumeric tokens."""
    tokens = re.findall(r"\b\w+\b", query.lower())
    stop_words = {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or", "is", "are", "it",
        "what", "how", "who", "where", "why", "when", "which", "do", "does", "did", "can",
        "could", "would", "should", "will", "shall", "about"
    }
    filtered = [t for t in tokens if t not in stop_words and len(t) > 1]
    return filtered or tokens


def format_context(
    hits: Sequence[RetrievalHit],
    max_characters: int = 4000,
    include_provenance: bool = True,
) -> str:
    """Format a list of retrieval hits into a clean structured Markdown context block."""
    if not hits:
        return ""

    context_blocks: List[str] = []
    current_char_count = 0

    for idx, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        meta = chunk.metadata
        file_name = meta.get("file_name", Path(chunk.source).name if chunk.source else "Document")
        section = chunk.section_header or "General"
        page = meta.get("page_number")
        page_str = f" | Page: {page}" if page is not None else ""

        if include_provenance:
            header = f"### [Source {idx}: {file_name} | Section: {section}{page_str} | Score: {hit.confidence_score:.2f}]"
            block = f"{header}\n{chunk.text.strip()}"
        else:
            block = f"--- Document {idx} ---\n{chunk.text.strip()}"

        block_len = len(block) + 2
        if current_char_count + block_len > max_characters and context_blocks:
            break

        context_blocks.append(block)
        current_char_count += block_len

    return "\n\n".join(context_blocks)


class HybridRetriever:
    """Hybrid retrieval engine combining vector similarity, sparse keywords, and prefix matching."""

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        dense_weight: float = 0.6,
        keyword_weight: float = 0.3,
        prefix_weight: float = 0.1,
        min_confidence: float = 0.15,
        max_context_chars: int = 4000,
    ) -> None:
        self.vector_store = vector_store
        self.default_top_k = top_k
        self.dense_weight = dense_weight
        self.keyword_weight = keyword_weight
        self.prefix_weight = prefix_weight
        self.min_confidence = min_confidence
        self.max_context_chars = max_context_chars

    def compute_keyword_score(self, query_tokens: Sequence[str], chunk_text: str, raw_query: str) -> float:
        """Compute keyword frequency and phrase match score for a chunk."""
        if not query_tokens:
            return 0.0

        lower_text = chunk_text.lower()
        chunk_words = set(re.findall(r"\b\w+\b", lower_text))

        matched_tokens = sum(1 for t in query_tokens if t in chunk_words)
        coverage_score = matched_tokens / len(query_tokens)

        clean_raw = raw_query.lower().strip()
        phrase_bonus = 0.2 if len(clean_raw) > 3 and clean_raw in lower_text else 0.0

        score = min(1.0, coverage_score * 0.8 + phrase_bonus)
        return float(score)

    def compute_prefix_score(self, query_tokens: Sequence[str], chunk_text: str) -> float:
        """Compute prefix and stem overlap matching for morphological variations."""
        if not query_tokens:
            return 0.0

        lower_text = chunk_text.lower()
        chunk_words = re.findall(r"\b\w+\b", lower_text)
        if not chunk_words:
            return 0.0

        prefix_matches = 0
        for token in query_tokens:
            stem = token[:min(4, len(token))]
            if len(stem) >= 3:
                for word in chunk_words:
                    if word.startswith(stem) or (len(word) >= 4 and stem in word):
                        prefix_matches += 1
                        break

        return float(min(1.0, prefix_matches / len(query_tokens)))

    def calibrate_confidence(
        self,
        dense_score: float,
        kw_score: float,
        prefix_score: float,
        hybrid_score: float,
    ) -> float:
        """Calibrate raw similarity into a probabilistic confidence score in [0.0, 1.0]."""
        if hybrid_score <= 0.01:
            return 0.0

        base_confidence = 1.0 - math.exp(-2.2 * hybrid_score)

        if dense_score > 0.4 and kw_score > 0.4:
            base_confidence = min(1.0, base_confidence * 1.15)
        elif dense_score < 0.1 and kw_score == 0.0 and prefix_score == 0.0:
            base_confidence = max(0.0, base_confidence * 0.2)

        return float(np.clip(base_confidence, 0.0, 1.0))

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_confidence: Optional[float] = None,
        filter_below_threshold: bool = False,
        format_context_block: bool = True,
    ) -> RetrievalResult:
        """Retrieve the top-k most relevant document chunks for a query with threshold gating.

        Args:
            query: User question or search text.
            top_k: Optional override for number of hits.
            min_confidence: Optional confidence threshold override.
            filter_below_threshold: If True, hits below threshold are excluded from output.
            format_context_block: If True, builds formatted_context markdown.

        Returns:
            RetrievalResult containing ranked RetrievalHit objects and confidence metrics.
        """
        start_time = time.perf_counter()
        k = top_k or self.default_top_k
        thresh = min_confidence if min_confidence is not None else self.min_confidence
        clean_q = query.strip()

        if not clean_q or self.vector_store.total_chunks == 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return RetrievalResult(
                query=query,
                hits=[],
                total_hits=0,
                execution_time_ms=elapsed_ms,
                is_confident=False,
                top_confidence=0.0,
                formatted_context="",
                rejection_reason="Empty query or unpopulated vector store",
            )

        query_tokens = tokenize_query(clean_q)

        # Fetch dense candidates
        fetch_k = min(self.vector_store.total_chunks, max(k * 3, 20))
        dense_results = self.vector_store.similarity_search(clean_q, top_k=fetch_k, min_score=0.0)

        scored_candidates: List[RetrievalHit] = []
        for chunk, dense_score in dense_results:
            kw_score = self.compute_keyword_score(query_tokens, chunk.text, clean_q)
            prefix_score = self.compute_prefix_score(query_tokens, chunk.text)

            combined_score = (
                self.dense_weight * dense_score
                + self.keyword_weight * kw_score
                + self.prefix_weight * prefix_score
            )
            confidence = self.calibrate_confidence(dense_score, kw_score, prefix_score, combined_score)

            hit = RetrievalHit(
                chunk=chunk,
                dense_score=dense_score,
                keyword_score=kw_score,
                prefix_score=prefix_score,
                hybrid_score=combined_score,
                confidence_score=confidence,
            )
            scored_candidates.append(hit)

        scored_candidates.sort(key=lambda h: h.hybrid_score, reverse=True)

        top_hits = scored_candidates[:k]
        top_conf = top_hits[0].confidence_score if top_hits else 0.0
        is_confident = top_conf >= thresh

        if filter_below_threshold:
            top_hits = [h for h in top_hits if h.confidence_score >= thresh]

        for rank_idx, hit in enumerate(top_hits, start=1):
            hit.rank = rank_idx

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        rejection = None if is_confident else f"Top retrieval confidence ({top_conf:.3f}) below threshold ({thresh:.3f})"
        ctx = format_context(top_hits, max_characters=self.max_context_chars) if format_context_block and is_confident else ""

        return RetrievalResult(
            query=query,
            hits=top_hits,
            total_hits=len(top_hits),
            execution_time_ms=elapsed_ms,
            is_confident=is_confident,
            top_confidence=top_conf,
            formatted_context=ctx,
            rejection_reason=rejection,
        )
