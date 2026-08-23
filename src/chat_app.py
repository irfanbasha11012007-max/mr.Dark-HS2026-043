"""Knowledge Assistant Chat Interface.

Supports both terminal CLI chat and interactive Streamlit web application.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import streamlit as st
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
            logger.info("Loaded Vector Store from %s", vs_path)
        except Exception as e:
            logger.error("Error loading Vector Store: %s", e)
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


def setup_streamlit_styling() -> None:
    """Inject custom CSS styling for premium glassmorphism dark theme and typography."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        h1, h2, h3 {
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            background: linear-gradient(135deg, #a5f3fc 0%, #38bdf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Glassmorphic main panel styling */
        .stChatMessage {
            background: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1) !important;
            backdrop-filter: blur(5px) !important;
            -webkit-backdrop-filter: blur(5px) !important;
            margin-bottom: 12px !important;
            padding: 16px !important;
            transition: transform 0.2s ease, border 0.2s ease;
        }

        .stChatMessage:hover {
            transform: translateY(-2px);
            border: 1px solid rgba(56, 189, 248, 0.2) !important;
        }

        /* Premium Alert/Card panels */
        .grounded-card {
            background-color: rgba(34, 197, 94, 0.1);
            border-left: 5px solid #22c55e;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            font-size: 1rem;
        }

        .abstention-card {
            background-color: rgba(239, 68, 68, 0.1);
            border-left: 5px solid #ef4444;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            font-size: 1rem;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #0f172a !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_streamlit_app(args: argparse.Namespace) -> None:
    """Run interactive Streamlit web application."""
    st.set_page_config(
        page_title="Knowledge Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    setup_streamlit_styling()

    # Session State Initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "threshold" not in st.session_state:
        st.session_state.threshold = args.threshold
    if "offline" not in st.session_state:
        st.session_state.offline = args.offline
    if "model" not in st.session_state:
        st.session_state.model = args.model

    st.title("🤖 AI Knowledge Assistant")
    st.caption("Phase 4 Grounded Generation & Verification System")

    # Sidebar configuration panel
    st.sidebar.header("Configuration")
    st.session_state.model = st.sidebar.text_input("LLM Model Name", value=st.session_state.model)
    st.session_state.threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, value=st.session_state.threshold, step=0.05)
    st.session_state.offline = st.sidebar.checkbox("Force Offline Mode", value=st.session_state.offline)

    # Render current chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input chat box
    if user_input := st.chat_input("Ask a grounded question..."):
        # Append and display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Initialize engine with updated session states
        engine_args = argparse.Namespace(
            vector_store=args.vector_store,
            model=st.session_state.model,
            offline=st.session_state.offline,
            threshold=st.session_state.threshold,
        )
        engine = init_engine(engine_args)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving knowledge and generating response..."):
                response = engine.generate_answer(user_input)

            # For now, display answer text. In subsequent commits, we will style citations and metadata.
            st.markdown(response.answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.answer,
                "response_data": response.to_dict(),
            })
            st.rerun()


def main() -> None:
    """Main entrypoint parsing CLI arguments and routing execution."""
    args = parse_args()
    if args.mode == "streamlit":
        run_streamlit_app(args)
    else:
        run_terminal_chat(args)


if __name__ == "__main__":
    main()
