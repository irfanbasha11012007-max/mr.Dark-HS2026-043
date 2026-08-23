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


def calculate_factual_accuracy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate accuracy and entity coverage metrics for in-scope answerable questions."""
    in_scope_results = [r for r in results if r["is_in_scope"]]
    if not in_scope_results:
        return {"accuracy": 0.0, "total_in_scope": 0}

    total_correct = 0
    total_in_scope = len(in_scope_results)
    entity_match_percentages = []

    for r in in_scope_results:
        gen_ans = r["generated_answer"]
        key_entities = r.get("key_entities", [])
        
        # If model abstained on an in-scope question, it's incorrect (false negative/rejection)
        if r["abstained"]:
            entity_match_percentages.append(0.0)
            continue

        if not key_entities:
            # If no key entities are defined and model didn't abstain, count as correct
            total_correct += 1
            entity_match_percentages.append(1.0)
            continue

        matched = sum(1 for e in key_entities if e.lower() in gen_ans.lower())
        match_ratio = matched / len(key_entities)
        entity_match_percentages.append(match_ratio)

        # Factual coverage rule: correct if at least 50% of the key entities are present in generated text
        if match_ratio >= 0.5:
            total_correct += 1

    avg_entity_coverage = sum(entity_match_percentages) / len(entity_match_percentages) if entity_match_percentages else 0.0

    return {
        "accuracy": total_correct / total_in_scope if total_in_scope > 0 else 0.0,
        "total_in_scope": total_in_scope,
        "correct_in_scope": total_correct,
        "average_entity_coverage": avg_entity_coverage,
    }


def calculate_abstention_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate precision, recall, and F1 metrics for abstentions/refusals."""
    tp = 0  # Out-of-scope query correctly abstained
    fp = 0  # In-scope query incorrectly abstained
    fn = 0  # Out-of-scope query incorrectly answered (failure to abstain)
    tn = 0  # In-scope query correctly answered

    for r in results:
        is_in_scope = r["is_in_scope"]
        abstained = r["abstained"]

        if not is_in_scope:
            if abstained:
                tp += 1
            else:
                fn += 1
        else:
            if abstained:
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    rejection_rate = sum(1 for r in results if r["abstained"]) / len(results) if results else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "abstention_precision": precision,
        "abstention_recall": recall,
        "abstention_f1": f1,
        "rejection_rate": rejection_rate,
    }


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

    accuracy_metrics = calculate_factual_accuracy(results)
    abstention_metrics = calculate_abstention_metrics(results)

    report = {
        "metadata": {
            "eval_time": time.asctime(),
            "model": args.model,
            "threshold": args.threshold,
            "offline": args.offline,
            "total_questions": len(records),
        },
        "accuracy_metrics": accuracy_metrics,
        "abstention_metrics": abstention_metrics,
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
    logger.info("Factual Accuracy (Entity Coverage Score): %.2f%%", report["accuracy_metrics"]["accuracy"] * 100)
    logger.info("Abstention Refusal Precision: %.2f%%", report["abstention_metrics"]["abstention_precision"] * 100)
    logger.info("Abstention Refusal Recall: %.2f%%", report["abstention_metrics"]["abstention_recall"] * 100)
    logger.info("Abstention Refusal F1 Score: %.2f%%", report["abstention_metrics"]["abstention_f1"] * 100)


if __name__ == "__main__":
    main()
