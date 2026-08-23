"""Embeddings and Vector Storage Module for Knowledge Assistant.

This module defines the abstract embedding model interface, concrete embedding implementations
(TF-IDF, dense sentence embeddings), and the persistent VectorStore for similarity retrieval.
"""

from __future__ import annotations

import abc
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.config import IngestionConfig, default_config
from src.ingest import DocumentChunk

logger = logging.getLogger(__name__)


class BaseEmbeddingModel(abc.ABC):
    """Abstract base interface for document and query embedding models."""

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Return human-readable identifier of the embedding model."""
        pass

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality produced by this model."""
        pass

    @abc.abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Generate 2D embedding matrix for a sequence of text strings.

        Args:
            texts: Sequence of document or chunk strings.

        Returns:
            2D numpy array of shape (len(texts), dimension) with float32 dtype.
        """
        pass

    @abc.abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Generate 1D embedding vector for a search query.

        Args:
            text: Query string.

        Returns:
            1D numpy array of shape (dimension,) with float32 dtype.
        """
        pass
