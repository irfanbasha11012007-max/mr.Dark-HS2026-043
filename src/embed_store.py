"""Embeddings and Vector Storage Module for Knowledge Assistant.

This module defines the abstract embedding model interface, concrete embedding implementations
(TF-IDF, dense sentence embeddings), and the persistent VectorStore for similarity retrieval.
"""

from __future__ import annotations

import abc
import argparse
import hashlib
import json
import logging
import pickle
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

        accumulated = np.zeros(1024, dtype=np.float32)
        for idx, token in enumerate(tokens):
            weight = 1.0 / (1.0 + 0.05 * idx)
            accumulated += weight * self._hash_token(token)
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


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize a 2D float32 array along rows to unit L2 norm."""
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D array for vector normalization, got {vectors.ndim}D")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return (vectors / norms).astype(np.float32)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a 1D float32 array to unit L2 norm."""
    if vector.ndim != 1:
        raise ValueError(f"Expected 1D array for vector normalization, got {vector.ndim}D")
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


class VectorStore:
    """In-memory and disk-persistent vector storage for document chunks and embeddings."""

    def __init__(
        self,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        normalize_embeddings: bool = True,
    ) -> None:
        self.embedding_model = embedding_model or get_default_embedding_model("tfidf")
        self.normalize_embeddings = normalize_embeddings
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self._chunk_id_to_idx: Dict[str, int] = {}

    @property
    def total_chunks(self) -> int:
        """Return the number of stored document chunks."""
        return len(self.chunks)

    @property
    def dimension(self) -> int:
        """Return the dimensionality of stored embeddings."""
        if self.embeddings is not None and len(self.embeddings) > 0:
            return self.embeddings.shape[1]
        return self.embedding_model.dimension

    def get_chunk(self, index: int) -> DocumentChunk:
        """Retrieve chunk by numeric index."""
        if index < 0 or index >= len(self.chunks):
            raise IndexError(f"Chunk index {index} out of bounds (total chunks: {len(self.chunks)})")
        return self.chunks[index]

    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve chunk by unique chunk_id string."""
        idx = self._chunk_id_to_idx.get(chunk_id)
        if idx is not None:
            return self.chunks[idx]
        return None

    def add_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Optional[np.ndarray] = None,
    ) -> None:
        """Add chunks and index their embeddings into the store."""
        if not chunks:
            return

        chunk_list = list(chunks)
        if embeddings is None:
            texts = [c.text for c in chunk_list]
            new_embeddings = self.embedding_model.embed_documents(texts)
        else:
            new_embeddings = np.asarray(embeddings, dtype=np.float32)
            if len(new_embeddings) != len(chunk_list):
                raise ValueError(
                    f"Embeddings count ({len(new_embeddings)}) must match chunks count ({len(chunk_list)})"
                )

        if self.normalize_embeddings:
            new_embeddings = normalize_vectors(new_embeddings)

        start_idx = len(self.chunks)
        for i, chunk in enumerate(chunk_list):
            self.chunks.append(chunk)
            self._chunk_id_to_idx[chunk.chunk_id] = start_idx + i

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        logger.info(
            "Added %d chunks to VectorStore (Total: %d, Dimension: %d)",
            len(chunk_list),
            self.total_chunks,
            self.dimension,
        )

    def similarity_search(
        self,
        query: Union[str, np.ndarray],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Perform accelerated cosine similarity search against stored document vectors."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        if isinstance(query, str):
            query_vec = self.embedding_model.embed_query(query)
        else:
            query_vec = np.asarray(query, dtype=np.float32)

        if self.normalize_embeddings:
            query_vec = normalize_vector(query_vec)

        # Vectorized BLAS matrix-vector dot product
        scores = np.dot(self.embeddings, query_vec)
        scores = np.nan_to_num(scores, nan=0.0)

        n = len(scores)
        k = min(top_k, n)
        if k <= 0:
            return []

        # Optimization: use argpartition for O(N) top-k selection when N is large
        if n > 100 and k < n:
            top_k_partition = np.argpartition(scores, -k)[-k:]
            sorted_indices = top_k_partition[np.argsort(scores[top_k_partition])[::-1]]
        else:
            sorted_indices = np.argsort(scores)[::-1][:k]

        results: List[Tuple[DocumentChunk, float]] = []
        for idx in sorted_indices:
            score = float(scores[idx])
            if score >= min_score:
                results.append((self.chunks[idx], score))

        return results

    def batch_similarity_search(
        self,
        queries: Sequence[Union[str, np.ndarray]],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[List[Tuple[DocumentChunk, float]]]:
        """Perform vectorized batch similarity search across multiple queries simultaneously."""
        if self.embeddings is None or len(self.chunks) == 0 or not queries:
            return [[] for _ in queries]

        query_vectors: List[np.ndarray] = []
        for q in queries:
            if isinstance(q, str):
                vec = self.embedding_model.embed_query(q)
            else:
                vec = np.asarray(q, dtype=np.float32)
            if self.normalize_embeddings:
                vec = normalize_vector(vec)
            query_vectors.append(vec)

        # Shape (num_queries, dimension)
        q_matrix = np.vstack(query_vectors)

        # Shape (num_queries, num_documents) = q_matrix @ embeddings.T
        all_scores = np.dot(q_matrix, self.embeddings.T)
        all_scores = np.nan_to_num(all_scores, nan=0.0)

        batch_results: List[List[Tuple[DocumentChunk, float]]] = []
        n = self.total_chunks
        k = min(top_k, n)

        for row_idx in range(len(queries)):
            scores = all_scores[row_idx]
            if n > 100 and k < n:
                top_partition = np.argpartition(scores, -k)[-k:]
                sorted_idx = top_partition[np.argsort(scores[top_partition])[::-1]]
            else:
                sorted_idx = np.argsort(scores)[::-1][:k]

            res: List[Tuple[DocumentChunk, float]] = []
            for idx in sorted_idx:
                s = float(scores[idx])
                if s >= min_score:
                    res.append((self.chunks[idx], s))
            batch_results.append(res)

        return batch_results

    def save(self, directory: Union[str, Path]) -> None:
        """Persist vector store index, chunks, and embeddings to disk."""
        dir_path = Path(directory).resolve()
        dir_path.mkdir(parents=True, exist_ok=True)

        if self.embeddings is not None:
            np.savez_compressed(dir_path / "embeddings.npz", embeddings=self.embeddings)

        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        with open(dir_path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        config_data = {
            "total_chunks": self.total_chunks,
            "dimension": self.dimension,
            "normalize_embeddings": self.normalize_embeddings,
            "model_name": self.embedding_model.model_name,
            "model_class": self.embedding_model.__class__.__name__,
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        with open(dir_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        if hasattr(self.embedding_model, "save"):
            self.embedding_model.save(dir_path / "model.pkl")

        logger.info("Successfully persisted VectorStore to %s", dir_path)

    @classmethod
    def load(
        cls,
        directory: Union[str, Path],
        embedding_model: Optional[BaseEmbeddingModel] = None,
    ) -> "VectorStore":
        """Load a persisted VectorStore from disk."""
        dir_path = Path(directory).resolve()
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"VectorStore directory not found: {directory}")

        chunks_file = dir_path / "chunks.json"
        config_file = dir_path / "config.json"
        embeddings_file = dir_path / "embeddings.npz"
        model_file = dir_path / "model.pkl"

        if not chunks_file.exists() or not config_file.exists():
            raise FileNotFoundError(f"Missing required index files in {directory}")

        with open(config_file, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        normalize_embeddings = config_data.get("normalize_embeddings", True)

        model = embedding_model
        if model is None:
            if model_file.exists():
                try:
                    model = TfidfEmbeddingModel.load(model_file)
                except Exception as e:
                    logger.warning("Could not load model.pkl (%s), using default model", e)
            if model is None:
                model_class = config_data.get("model_class", "TfidfEmbeddingModel")
                if "Dense" in model_class:
                    model = get_default_embedding_model("dense", dimension=config_data.get("dimension", 384))
                else:
                    model = get_default_embedding_model("tfidf", max_features=config_data.get("dimension", 512))

        store = cls(
            embedding_model=model,
            normalize_embeddings=normalize_embeddings,
        )

        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks_raw = json.load(f)

        loaded_chunks = [DocumentChunk.from_dict(c) for c in chunks_raw]
        for idx, chunk in enumerate(loaded_chunks):
            store.chunks.append(chunk)
            store._chunk_id_to_idx[chunk.chunk_id] = idx

        if embeddings_file.exists():
            data = np.load(embeddings_file)
            store.embeddings = data["embeddings"].astype(np.float32)

        logger.info(
            "Loaded VectorStore from %s (%d chunks, dim=%d)",
            dir_path,
            store.total_chunks,
            store.dimension,
        )
        return store


def build_vector_store_from_jsonl(
    jsonl_path: Union[str, Path],
    output_dir: Union[str, Path],
    model_type: str = "tfidf",
    dimension: Optional[int] = None,
) -> VectorStore:
    """Read DocumentChunk JSONL file, embed all chunks, and save VectorStore index."""
    start_time = time.time()
    in_file = Path(jsonl_path).resolve()
    out_dir = Path(output_dir).resolve()

    if not in_file.exists():
        raise FileNotFoundError(f"Chunks JSONL file not found: {jsonl_path}")

    chunks: List[DocumentChunk] = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                chunks.append(DocumentChunk.from_dict(json.loads(line_str)))

    if not chunks:
        logger.warning("No chunks found in %s", jsonl_path)

    kwargs: Dict[str, Any] = {}
    if dimension:
        if model_type == "tfidf":
            kwargs["max_features"] = dimension
        else:
            kwargs["dimension"] = dimension

    embedding_model = get_default_embedding_model(model_type, **kwargs)

    if isinstance(embedding_model, TfidfEmbeddingModel) and chunks:
        texts = [c.text for c in chunks]
        embedding_model.fit(texts)

    store = VectorStore(embedding_model=embedding_model, normalize_embeddings=True)
    store.add_chunks(chunks)
    store.save(out_dir)

    elapsed = time.time() - start_time
    logger.info(
        "Built and persisted VectorStore: %d chunks -> %s in %.2fs",
        len(chunks),
        out_dir,
        elapsed,
    )
    return store


def build_cli_parser() -> argparse.ArgumentParser:
    """Build argument parser for the embed_store CLI."""
    parser = argparse.ArgumentParser(
        description="Knowledge Assistant - Vector Store & Index Builder CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        dest="input_path",
        default="data/processed/chunks.jsonl",
        help="Path to input chunks JSONL file",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output_dir",
        default="data/index",
        help="Target directory to save vector index artifacts",
    )
    parser.add_argument(
        "--model",
        "-m",
        choices=["tfidf", "dense"],
        default="tfidf",
        help="Embedding model type",
    )
    parser.add_argument(
        "--dimension",
        "-d",
        type=int,
        default=None,
        help="Vector dimensionality override",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entrypoint for building/rebuilding the VectorStore."""
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        store = build_vector_store_from_jsonl(
            jsonl_path=args.input_path,
            output_dir=args.output_dir,
            model_type=args.model,
            dimension=args.dimension,
        )

        print("\n" + "=" * 55)
        print("  VECTOR STORE INDEXING COMPLETED")
        print("=" * 55)
        print(f"  Total Chunks Indexed : {store.total_chunks}")
        print(f"  Embedding Model      : {store.embedding_model.model_name}")
        print(f"  Vector Dimensionality: {store.dimension}")
        print(f"  Index Output Path    : {Path(args.output_dir).resolve()}")
        print("=" * 55 + "\n")
        return 0

    except Exception as e:
        logger.error("Failed to build vector store: %s", e, exc_info=True)
        print(f"\n[ERROR] Failed to build vector store: {e}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
