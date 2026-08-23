"""Knowledge Assistant Chat Interface.

Supports both terminal CLI chat and interactive Streamlit web application.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Knowledge Assistant Chat Interface")
    parser.add_argument(
        "--mode",
        choices=["cli", "streamlit"],
        default="streamlit" if "streamlit" in sys.argv[0] or any("streamlit" in arg for arg in sys.argv) else "cli",
        help="Run mode: terminal CLI or Streamlit web UI",
    )
    parser.add_argument(
        "--vector-store",
        default="data/vector_store.json",
        help="Path to persistent vector store JSON file",
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
        "--query",
        type=str,
        help="Single query to execute in CLI mode without terminal loop",
    )
    return parser.parse_args()


def run_terminal_chat(args: argparse.Namespace) -> None:
    """Run interactive terminal chat interface."""
    print(f"Welcome to Knowledge Assistant CLI (Model: {args.model}, Threshold: {args.threshold})")


def run_streamlit_app(args: argparse.Namespace) -> None:
    """Run interactive Streamlit web application."""
    print(f"Running Streamlit UI (Model: {args.model})")


def main() -> None:
    """Main entrypoint parsing CLI arguments and routing execution."""
    args = parse_args()
    if args.mode == "streamlit":
        run_streamlit_app(args)
    else:
        run_terminal_chat(args)


if __name__ == "__main__":
    main()
