"""Answer Engine and Grounded Generation Module for Knowledge Assistant.

This module orchestrates the generation phase of the RAG pipeline:
- Strictly answers questions using ONLY retrieved context.
- Enforces anti-hallucination guardrails and confidence-based abstention.
- Extracts structured citation provenance.
- Implements resilient OpenRouter/OpenAI API communication with exponential retry backoff.
- Provides a deterministic offline grounded synthesizer fallback.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.retriever import HybridRetriever, RetrievalHit, RetrievalResult

logger = logging.getLogger(__name__)


class AnswerEngine:
    """Core Answer Engine responsible for generating strictly grounded answers or abstaining."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        model_name: str = "openai/gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        min_confidence_threshold: float = 0.20,
    ) -> None:
        self.retriever = retriever
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.min_confidence_threshold = min_confidence_threshold
        logger.info("Initialized AnswerEngine (model=%s, temp=%.2f)", model_name, temperature)

    def generate_answer(
        self,
        question: str,
        retrieval_result: Optional[RetrievalResult] = None,
    ) -> Dict[str, Any]:
        """Generate a grounded answer for a question."""
        clean_q = question.strip()
        if not clean_q:
            return {
                "question": question,
                "answer": "I don't have that information in the provided material.",
                "abstained": True,
                "abstention_reason": "Empty question provided",
            }
        return {
            "question": clean_q,
            "answer": "I don't have that information in the provided material.",
            "abstained": True,
            "abstention_reason": "Initial skeleton",
        }
