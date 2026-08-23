"""Unit and Integration Tests for Phase 3: Answer Engine, Grounding, and Abstention."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.answer_engine import (
    STANDARD_ABSTENTION_MESSAGE,
    STRICT_SYSTEM_PROMPT,
    AnswerEngine,
    AnswerResponse,
    Citation,
    GenerationConfig,
    LLMClient,
    build_user_prompt,
    evaluate_confidence_abstention,
    generate_offline_grounded_answer,
    normalize_abstention_response,
    parse_and_bind_citations,
    verify_context_sufficiency,
)
from src.embed_store import VectorStore
from src.ingest import DocumentChunk
from src.retriever import HybridRetriever, RetrievalHit, RetrievalResult


@pytest.fixture
def sample_retrieval_hit() -> RetrievalHit:
    """Sample retrieval hit fixture."""
    chunk = DocumentChunk(
        chunk_id="chunk_101",
        doc_id="doc_guide",
        text="The Knowledge Assistant uses a 4-phase architecture for enterprise document question answering.",
        start_char=0,
        end_char=96,
        chunk_index=0,
        source="docs/architecture.md",
        section_header="System Overview",
        metadata={"file_name": "architecture.md", "page_number": 1},
    )
    return RetrievalHit(
        chunk=chunk,
        dense_score=0.92,
        keyword_score=0.85,
        prefix_score=0.70,
        hybrid_score=0.88,
        confidence_score=0.88,
        rank=1,
    )


@pytest.fixture
def sample_retrieval_result(sample_retrieval_hit: RetrievalHit) -> RetrievalResult:
    """Sample retrieval result fixture."""
    return RetrievalResult(
        query="What architecture does Knowledge Assistant use?",
        hits=[sample_retrieval_hit],
        total_hits=1,
        execution_time_ms=12.5,
        is_confident=True,
        top_confidence=0.88,
        formatted_context="### [Source 1: architecture.md | Section: System Overview | Page: 1 | Score: 0.88]\nThe Knowledge Assistant uses a 4-phase architecture for enterprise document question answering.",
    )


class TestGroundedAnswerGeneration:
    """Tests for grounded answer synthesis, formatting, and parameters."""

    def test_build_user_prompt_format(self) -> None:
        prompt = build_user_prompt("How does indexing work?", "Context text block")
        assert "=== RETRIEVED CONTEXT DOCUMENTS ===" in prompt
        assert "Context text block" in prompt
        assert "=== USER QUESTION ===" in prompt
        assert "How does indexing work?" in prompt
        assert "=== GROUNDED ANSWER ===" in prompt

    def test_offline_grounded_answer_extractor(self, sample_retrieval_hit: RetrievalHit) -> None:
        answer, citations = generate_offline_grounded_answer(
            "What architecture does Knowledge Assistant use?",
            [sample_retrieval_hit],
        )
        assert "4-phase architecture" in answer
        assert "[Source 1: architecture.md" in answer
        assert len(citations) == 1
        assert citations[0].source == "docs/architecture.md"
        assert citations[0].section == "System Overview"

    def test_answer_engine_offline_execution_end_to_end(
        self,
        sample_retrieval_result: RetrievalResult,
    ) -> None:
        engine = AnswerEngine(offline_mode=True)
        response = engine.generate_answer(
            "What architecture does Knowledge Assistant use?",
            retrieval_result=sample_retrieval_result,
        )
        assert isinstance(response, AnswerResponse)
        assert not response.abstained
        assert "4-phase architecture" in response.answer
        assert len(response.citations) >= 1
        assert response.citations[0].source == "docs/architecture.md"
        assert response.retrieval_confidence == 0.88
        assert response.latency_ms > 0

    def test_generation_config_parameters(self) -> None:
        cfg = GenerationConfig(temperature=0.0, max_tokens=256, seed=42)
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 256
        assert cfg.seed == 42
        d = cfg.to_dict()
        assert d["temperature"] == 0.0
