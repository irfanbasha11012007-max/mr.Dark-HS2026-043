"""Knowledge Assistant Chat Interface.

Supports both terminal CLI chat and interactive Streamlit web application.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src.answer_engine import AnswerEngine, AnswerResponse
from src.embed_store import VectorStore
from src.retriever import HybridRetriever

logger = logging.getLogger(__name__)
console = Console()


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


def display_answer(response: AnswerResponse) -> None:
    """Print the formatted answer using Rich Console components."""
    console.print()
    if response.abstained:
        console.print(
            Panel(
                f"[bold red]ABSTENTION / INFORMATION NOT FOUND[/bold red]\n\n{response.answer}",
                title="[bold yellow]Refusal Gating[/bold yellow]",
                border_style="red",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold green]GROUNDED ANSWER[/bold green]\n\n{response.answer}",
                title="[bold blue]Grounded Generation[/bold blue]",
                border_style="green",
            )
        )

    # Print metadata
    meta_table = Table(title="Retrieval & Generation Metadata", show_header=False, box=None)
    meta_table.add_row("Model Name:", response.model_name)
    meta_table.add_row("Retrieval Confidence:", f"{response.retrieval_confidence:.3f}")
    meta_table.add_row("Latency:", f"{response.latency_ms:.1f} ms")
    if response.abstention_reason:
        meta_table.add_row("Abstention Reason:", response.abstention_reason)
    console.print(meta_table)

    # Print citations
    if response.citations:
        console.print("\n[bold cyan]Citations & Provenance:[/bold cyan]")
        cit_table = Table(show_header=True, header_style="bold cyan")
        cit_table.add_column("Index", style="dim", width=6)
        cit_table.add_column("Source Document", style="magenta")
        cit_table.add_column("Section Header", style="green")
        cit_table.add_column("Page", justify="right")
        cit_table.add_column("Confidence", justify="right")

        for idx, cit in enumerate(response.citations, 1):
            cit_table.add_row(
                str(idx),
                Path(cit.source).name,
                cit.section or "N/A",
                str(cit.page) if cit.page is not None else "N/A",
                f"{cit.confidence:.3f}",
            )
        console.print(cit_table)
    console.print("-" * 80)


def init_engine(args: argparse.Namespace) -> AnswerEngine:
    """Initialize Retriever and AnswerEngine."""
    vs_path = Path(args.vector_store)
    retriever = None
    if vs_path.exists():
        try:
            vstore = VectorStore.load(vs_path)
            retriever = HybridRetriever(vector_store=vstore)
            console.print(f"[green]Loaded Vector Store with {len(vstore.chunks)} chunks from {vs_path}[/green]")
        except Exception as e:
            console.print(f"[red]Error loading Vector Store ({e}). Running without retriever context.[/red]")
    else:
        console.print(f"[yellow]Vector Store not found at {vs_path}. Please build index first.[/yellow]")

    return AnswerEngine(
        retriever=retriever,
        model_name=args.model,
        min_confidence_threshold=args.threshold,
        offline_mode=args.offline,
    )


def run_terminal_chat(args: argparse.Namespace) -> None:
    """Run interactive terminal chat interface."""
    engine = init_engine(args)

    if args.query:
        console.print(f"\n[bold yellow]Query:[/bold yellow] {args.query}")
        response = engine.generate_answer(args.query)
        display_answer(response)
        return

    console.print(
        Panel(
            "Welcome to the Knowledge Assistant Terminal Chat!\n"
            "Ask questions grounded in your document repository.\n"
            "Type [bold cyan]exit[/bold cyan] or [bold cyan]quit[/bold cyan] to terminate.",
            title="[bold green]Knowledge Assistant CLI[/bold green]",
            border_style="green",
        )
    )

    while True:
        try:
            query = console.input("\n[bold yellow]User Ask > [/bold yellow]").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                console.print("[cyan]Goodbye![/cyan]")
                break

            response = engine.generate_answer(query)
            display_answer(response)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[cyan]Goodbye![/cyan]")
            break


def run_streamlit_app(args: argparse.Namespace) -> None:
    """Run interactive Streamlit web application."""
    console.print(f"Running Streamlit UI (Model: {args.model})")
    # Streamlit execution will be added in subsequent commits


def main() -> None:
    """Main entrypoint parsing CLI arguments and routing execution."""
    args = parse_args()
    if args.mode == "streamlit":
        run_streamlit_app(args)
    else:
        run_terminal_chat(args)


if __name__ == "__main__":
    main()
