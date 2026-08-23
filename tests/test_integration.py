"""End-to-End integration tests for Knowledge Assistant.

Tests full document processing pipeline: ingestion -> embedding -> retrieval -> generation.
"""

from __future__ import annotations

import json
import tempfile
import shutil
from pathlib import Path

import pytest

from src.config import default_config
from src.ingest import IngestionPipeline
from src.embed_store import build_vector_store_from_jsonl, VectorStore
from src.retriever import HybridRetriever
from src.answer_engine import AnswerEngine


@pytest.fixture
def temp_workspace():
    """Setup temporary directory workspace for end-to-end ingestion and indexing."""
    tmp_path = Path(tempfile.mkdtemp())
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    index_dir = tmp_path / "index"

    raw_dir.mkdir()
    processed_dir.mkdir()
    index_dir.mkdir()

    # Create dummy documentation files
    doc1 = raw_dir / "doc1.md"
    doc1.write_text(
        "# Knowledge Assistant Ingestion\n\n"
        "The ingestion pipeline supports .txt, .md, and .pdf formats.\n"
        "It uses a recursive character chunker to divide documents into logical parts.\n"
        "Metadata attributes are preserved for each generated chunk.\n",
        encoding="utf-8",
    )

    doc2 = raw_dir / "doc2.txt"
    doc2.write_text(
        "Knowledge Assistant Retrieval System\n\n"
        "The retrieval component uses hybrid scoring matching TF-IDF and dense embeddings.\n"
        "A threshold score is applied to reject out-of-scope queries.\n",
        encoding="utf-8",
    )

    yield {
        "root": tmp_path,
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "index_dir": index_dir,
    }

    shutil.rmtree(tmp_path, ignore_errors=True)


def test_end_to_end_pipeline(temp_workspace):
    """Verify complete flow from raw files to grounded/abstained answers."""
    raw_path = str(temp_workspace["raw_dir"])
    chunks_jsonl = str(temp_workspace["processed_dir"] / "chunks.jsonl")
    index_path = str(temp_workspace["index_dir"])

    # Phase 1: Ingest
    pipeline = IngestionPipeline(default_config)
    stats = pipeline.run(raw_path, chunks_jsonl)
    assert stats["total_documents"] == 2
    assert stats["total_chunks"] >= 2

    # Phase 2: Embed / Build VectorStore Index
    vstore = build_vector_store_from_jsonl(
        jsonl_path=chunks_jsonl,
        output_dir=index_path,
        model_type="tfidf",
    )
    assert vstore.total_chunks == stats["total_chunks"]

    # Phase 3 & 4: Retrieval and Grounded/Refusal Generation
    retriever = HybridRetriever(vector_store=vstore, min_confidence=0.15)
    engine = AnswerEngine(
        retriever=retriever,
        model_name="openai/gpt-4o-mini",
        min_confidence_threshold=0.15,
        offline_mode=True,  # offline mode guarantees no API dependencies
    )

    # 1. In-Scope query
    res_in = engine.generate_answer("What document formats are supported by the document ingestion pipeline?")
    assert not res_in.abstained
    assert "doc1.md" in [Path(c.source).name for c in res_in.citations]
    assert any(ext in res_in.answer for ext in [".txt", ".md", ".pdf"])

    # 2. Out-of-Scope query (should trigger refusal/abstention)
    res_out = engine.generate_answer("Who is the prime minister of Canada?")
    assert res_out.abstained
    assert "I don't have that information" in res_out.answer or "not contained" in res_out.answer or "provided material" in res_out.answer
