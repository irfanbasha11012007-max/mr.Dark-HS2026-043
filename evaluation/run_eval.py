"""Evaluation Runner for Knowledge Assistant.

Loads evaluation dataset, queries the AnswerEngine, and logs performance.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from src.answer_engine import AnswerEngine
from src.embed_store import VectorStore
from src.retriever import HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Knowledge Assistant performance")
    parser.add_argument(
        "--dataset",
        default="evaluation/eval_questions.jsonl",
        help="Path to evaluation dataset JSONL file",
    )
    parser.add_argument(
        "--vector-store",
        default="data/index",
        help="Path to VectorStore index directory",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="Model name for generation",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force offline grounded answer generation fallback",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help="Minimum retrieval confidence score threshold",
    )
    parser.add_argument(
        "--output",
        default="evaluation/report.json",
        help="Path to save evaluation report JSON",
    )
    return parser.parse_args()


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL dataset."""
    records = []
    if not dataset_path.exists():
        logger.error("Dataset not found at %s", dataset_path)
        return []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute evaluation queries and collect raw outputs."""
    dataset_path = Path(args.dataset)
    records = load_dataset(dataset_path)
    if not records:
        return {"error": "Dataset empty or not found"}

    # Initialize Engine
    vs_path = Path(args.vector_store)
    retriever = None
    if vs_path.exists() and vs_path.is_dir():
        vstore = VectorStore.load(vs_path)
        retriever = HybridRetriever(vector_store=vstore)
    
    engine = AnswerEngine(
        retriever=retriever,
        model_name=args.model,
        min_confidence_threshold=args.threshold,
        offline_mode=args.offline,
    )

    results = []
    logger.info("Running evaluation on %d questions...", len(records))

    for idx, record in enumerate(records, 1):
        q = record["question"]
        is_in_scope = record["is_in_scope"]
        logger.info("[%d/%d] Query: %s (In-Scope: %s)", idx, len(records), q, is_in_scope)

        start_time = time.perf_counter()
        response = engine.generate_answer(q)
        latency = (time.perf_counter() - start_time) * 1000

        results.append({
            "id": record.get("id"),
            "question": q,
            "category": record.get("category"),
            "is_in_scope": is_in_scope,
            "expected_answer": record.get("expected_answer"),
            "key_entities": record.get("key_entities", []),
            "generated_answer": response.answer,
            "abstained": response.abstained,
            "retrieval_confidence": response.retrieval_confidence,
            "latency_ms": latency,
        })

    report = {
        "metadata": {
            "eval_time": time.asctime(),
            "model": args.model,
            "threshold": args.threshold,
            "offline": args.offline,
            "total_questions": len(records),
        },
        "results": results,
    }
    return report


def main() -> None:
    """Main CLI entrypoint."""
    args = parse_args()
    report = run_evaluation(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Evaluation report saved to %s", output_path)


if __name__ == "__main__":
    main()
