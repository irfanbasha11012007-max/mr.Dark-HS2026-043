"""Unit and Integration Tests for Phase 2: Embeddings, Vector Store, and Hybrid Retrieval."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.embed_store import (
    BaseEmbeddingModel,
    LocalDenseEmbeddingModel,
    TfidfEmbeddingModel,
    VectorStore,
    build_vector_store_from_jsonl,
    get_default_embedding_model,
    normalize_vector,
    normalize_vectors,
)
from src.ingest import DocumentChunk
from src.retriever import (
    HybridRetriever,
    RetrievalHit,
    RetrievalResult,
    format_context,
    tokenize_query,
)


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    """Sample document chunks for testing."""
    return [
        DocumentChunk(
            chunk_id="chunk_001",
            doc_id="doc_rag_intro",
            text="Retrieval-Augmented Generation (RAG) combines semantic information retrieval with large language models.",
            start_char=0,
            end_char=105,
            chunk_index=0,
            source="docs/rag_overview.md",
            section_header="Introduction to RAG",
            metadata={"file_name": "rag_overview.md", "page_number": 1},
        ),
        DocumentChunk(
            chunk_id="chunk_002",
            doc_id="doc_rag_intro",
            text="Vector databases index dense vector embeddings to perform sub-millisecond approximate nearest neighbor search.",
            start_char=106,
            end_char=217,
            chunk_index=1,
            source="docs/rag_overview.md",
            section_header="Vector Indexing",
            metadata={"file_name": "rag_overview.md", "page_number": 1},
        ),
        DocumentChunk(
            chunk_id="chunk_003",
            doc_id="doc_pizza_recipe",
            text="To make authentic Neapolitan pizza, ferment the dough for 24 hours at room temperature with high hydration.",
            start_char=0,
            end_char=110,
            chunk_index=0,
            source="recipes/pizza.txt",
            section_header="Dough Preparation",
            metadata={"file_name": "pizza.txt"},
        ),
        DocumentChunk(
            chunk_id="chunk_004",
            doc_id="doc_eval_guide",
            text="Evaluation of retrieval systems uses Precision@K, Recall@K, Mean Reciprocal Rank (MRR), and NDCG metrics.",
            start_char=0,
            end_char=106,
            chunk_index=0,
            source="docs/evaluation.md",
            section_header="Retrieval Benchmarks",
            metadata={"file_name": "evaluation.md", "page_number": 3},
        ),
    ]


@pytest.fixture
def vector_store(sample_chunks: list[DocumentChunk]) -> VectorStore:
    """Pre-populated VectorStore fixture."""
    model = TfidfEmbeddingModel(max_features=128)
    store = VectorStore(embedding_model=model, normalize_embeddings=True)
    store.add_chunks(sample_chunks)
    return store


class TestEmbeddingModels:
    """Tests for embedding model implementations."""

    def test_tfidf_embedding_model_fit_and_embed(self) -> None:
        texts = [
            "Information retrieval with embeddings",
            "Natural language processing pipelines",
        ]
        model = TfidfEmbeddingModel(max_features=32)
        model.fit(texts)
        assert model.is_fitted
        assert model.dimension <= 32

        embeddings = model.embed_documents(texts)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2
        assert embeddings.dtype == np.float32

        query_vec = model.embed_query("information embeddings")
        assert isinstance(query_vec, np.ndarray)
        assert len(query_vec) == model.dimension

    def test_local_dense_embedding_model(self) -> None:
        model = LocalDenseEmbeddingModel(dimension=64)
        assert model.dimension == 64
        assert "local-dense" in model.model_name

        docs = ["Alpha beta gamma", "Delta epsilon zeta"]
        vecs = model.embed_documents(docs)
        assert vecs.shape == (2, 64)
        assert vecs.dtype == np.float32

        q_vec = model.embed_query("alpha gamma")
        assert q_vec.shape == (64,)
        # Check unit normalization
        norm = np.linalg.norm(q_vec)
        assert pytest.approx(norm, rel=1e-3) == 1.0

    def test_factory_get_default_embedding_model(self) -> None:
        tfidf = get_default_embedding_model("tfidf", max_features=64)
        assert isinstance(tfidf, TfidfEmbeddingModel)

        dense = get_default_embedding_model("dense", dimension=128)
        assert isinstance(dense, LocalDenseEmbeddingModel)


class TestVectorStore:
    """Tests for VectorStore indexing, similarity search, and serialization."""

    def test_vector_store_add_and_get_chunks(self, vector_store: VectorStore, sample_chunks: list[DocumentChunk]) -> None:
        assert vector_store.total_chunks == len(sample_chunks)
        assert vector_store.get_chunk(0).chunk_id == "chunk_001"
        assert vector_store.get_chunk_by_id("chunk_003") is not None
        assert vector_store.get_chunk_by_id("non_existent") is None

    def test_vector_normalization(self) -> None:
        matrix = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
        normalized = normalize_vectors(matrix)
        norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(norms, [1.0, 1.0])

        single = normalize_vector(np.array([1.0, 2.0, 2.0], dtype=np.float32))
        assert pytest.approx(np.linalg.norm(single), rel=1e-5) == 1.0

    def test_similarity_search_ranks_relevant_document_first(self, vector_store: VectorStore) -> None:
        results = vector_store.similarity_search("pizza dough fermentation", top_k=2)
        assert len(results) > 0
        top_chunk, score = results[0]
        assert top_chunk.chunk_id == "chunk_003"
        assert score > 0.1

    def test_save_and_load_persistence_roundtrip(self, vector_store: VectorStore) -> None:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            vector_store.save(tmp_dir)
            assert (tmp_dir / "embeddings.npz").exists()
            assert (tmp_dir / "chunks.json").exists()
            assert (tmp_dir / "config.json").exists()

            loaded_store = VectorStore.load(tmp_dir)
            assert loaded_store.total_chunks == vector_store.total_chunks
            assert loaded_store.dimension == vector_store.dimension

            search_res = loaded_store.similarity_search("Vector databases", top_k=1)
            assert len(search_res) == 1
            assert search_res[0][0].chunk_id == "chunk_002"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestHybridRetrieverRanking:
    """Tests for hybrid scoring, ranking, and context assembly."""

    def test_tokenize_query(self) -> None:
        tokens = tokenize_query("What is the Precision@K for RAG systems?")
        assert "precision" in tokens
        assert "rag" in tokens
        assert "systems" in tokens
        assert "is" not in tokens  # Stop word filtered

    def test_hybrid_ranking_top_hit_matches_query_intent(self, vector_store: VectorStore) -> None:
        retriever = HybridRetriever(vector_store, top_k=3)
        res = retriever.retrieve("NDCG and Mean Reciprocal Rank MRR metrics")
        assert res.total_hits > 0
        top_hit = res.hits[0]
        assert top_hit.chunk.chunk_id == "chunk_004"
        assert top_hit.rank == 1
        assert top_hit.keyword_score > 0.0
        assert top_hit.confidence_score > 0.3
        assert res.is_confident

    def test_format_context_includes_provenance_and_respects_max_chars(self, vector_store: VectorStore) -> None:
        retriever = HybridRetriever(vector_store, top_k=2, max_context_chars=500)
        res = retriever.retrieve("Retrieval-Augmented Generation")
        assert res.formatted_context != ""
        assert "### [Source 1:" in res.formatted_context
        assert "rag_overview.md" in res.formatted_context
        assert len(res.formatted_context) <= 500
