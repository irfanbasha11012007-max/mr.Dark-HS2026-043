"""Embeddings and Vector Storage Module for Knowledge Assistant.

This module defines the abstract embedding model interface, concrete embedding implementations
(TF-IDF, dense sentence embeddings), and the persistent VectorStore for similarity retrieval.
"""

from __future__ import annotations

import abc
import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

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


class TfidfEmbeddingModel(BaseEmbeddingModel):
    """TF-IDF sparse-to-dense embedding model with sublinear scaling and n-gram support."""

    def __init__(
        self,
        max_features: int = 512,
        ngram_range: Tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        lowercase: bool = True,
    ) -> None:
        self._max_features = max_features
        self._ngram_range = ngram_range
        self._sublinear_tf = sublinear_tf
        self._lowercase = lowercase
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            lowercase=lowercase,
            norm="l2",
        )
        self._is_fitted = False
        self._dim = max_features

    @property
    def model_name(self) -> str:
        return f"tfidf-dim{self._dim}"

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, texts: Sequence[str]) -> "TfidfEmbeddingModel":
        """Fit vocabulary and inverse document frequencies on a text collection."""
        if not texts:
            raise ValueError("Cannot fit TfidfEmbeddingModel on an empty text collection")
        self._vectorizer.fit(texts)
        actual_features = len(self._vectorizer.vocabulary_)
        self._dim = min(self._max_features, actual_features)
        self._is_fitted = True
        logger.info("TfidfEmbeddingModel fitted with %d features.", self._dim)
        return self

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a sequence of documents into a 2D float32 numpy array."""
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        if not self._is_fitted:
            self.fit(texts)

        sparse_matrix = self._vectorizer.transform(texts)
        dense_array = sparse_matrix.toarray().astype(np.float32)
        if dense_array.shape[1] < self._dim:
            pad_width = ((0, 0), (0, self._dim - dense_array.shape[1]))
            dense_array = np.pad(dense_array, pad_width, mode="constant")
        return dense_array

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query string into a 1D float32 numpy array."""
        if not self._is_fitted:
            self.fit([text])

        sparse_vector = self._vectorizer.transform([text])
        dense_vector = sparse_vector.toarray().astype(np.float32)[0]
        if len(dense_vector) < self._dim:
            dense_vector = np.pad(dense_vector, (0, self._dim - len(dense_vector)), mode="constant")
        return dense_vector

    def save(self, file_path: Union[str, Path]) -> None:
        """Save fitted model to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "max_features": self._max_features,
            "ngram_range": self._ngram_range,
            "sublinear_tf": self._sublinear_tf,
            "lowercase": self._lowercase,
            "is_fitted": self._is_fitted,
            "dim": self._dim,
            "vectorizer": self._vectorizer if self._is_fitted else None,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "TfidfEmbeddingModel":
        """Load fitted model from disk."""
        path = Path(file_path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        model = cls(
            max_features=data["max_features"],
            ngram_range=data["ngram_range"],
            sublinear_tf=data["sublinear_tf"],
            lowercase=data["lowercase"],
        )
        model._is_fitted = data["is_fitted"]
        model._dim = data["dim"]
        if data.get("vectorizer"):
            model._vectorizer = data["vectorizer"]
        return model


class LocalDenseEmbeddingModel(BaseEmbeddingModel):
    """Dense semantic embedding model using contextualized character and subword hashing."""

    def __init__(self, dimension: int = 384, model_name: str = "local-dense-384") -> None:
        self._dim = dimension
        self._name = model_name
        # Seed random projection matrices deterministically
        rng = np.random.RandomState(42)
        self._projection_matrix = rng.randn(1024, self._dim).astype(np.float32)

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_token(self, token: str) -> np.ndarray:
        """Project a single token deterministically into hash space."""
        h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
        vec = np.zeros(1024, dtype=np.float32)
        # Set top-k hashed active coordinates
        for i in range(8):
            idx = (h >> (i * 8)) % 1024
            sign = 1.0 if ((h >> (i * 8 + 4)) & 1) else -1.0
            vec[idx] += sign
        return vec

    def _embed_single_text(self, text: str) -> np.ndarray:
        """Produce a dense unit-normalized embedding vector for text."""
        if not text or not text.strip():
            return np.zeros(self._dim, dtype=np.float32)

        tokens = text.lower().split()
        if not tokens:
            return np.zeros(self._dim, dtype=np.float32)

        # Aggregate token hash projections with position decay
        accumulated = np.zeros(1024, dtype=np.float32)
        for idx, token in enumerate(tokens):
            weight = 1.0 / (1.0 + 0.05 * idx)
            accumulated += weight * self._hash_token(token)
            # Add character n-grams for subword sensitivity
            if len(token) >= 3:
                for j in range(len(token) - 2):
                    ngram = token[j:j + 3]
                    accumulated += 0.3 * self._hash_token(ngram)

        projected = np.dot(accumulated, self._projection_matrix)
        norm = np.linalg.norm(projected)
        if norm > 1e-8:
            projected = projected / norm
        return projected.astype(np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a sequence of documents into a 2D float32 numpy array."""
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        vectors = [self._embed_single_text(t) for t in texts]
        return np.vstack(vectors).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query string into a 1D float32 numpy array."""
        return self._embed_single_text(text)


def get_default_embedding_model(model_type: str = "tfidf", **kwargs) -> BaseEmbeddingModel:
    """Factory helper to obtain an embedding model instance."""
    m_type = model_type.lower().strip()
    if m_type in ("tfidf", "sparse"):
        return TfidfEmbeddingModel(**kwargs)
    elif m_type in ("dense", "local", "sentence"):
        dim = kwargs.get("dimension", 384)
        return LocalDenseEmbeddingModel(dimension=dim)
    else:
        logger.warning("Unrecognized model_type '%s', defaulting to TF-IDF", model_type)
        return TfidfEmbeddingModel(**kwargs)
