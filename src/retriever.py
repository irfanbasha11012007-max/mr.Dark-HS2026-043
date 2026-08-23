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
        )
