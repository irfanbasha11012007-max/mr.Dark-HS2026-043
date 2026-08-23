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
from src.embed_store import VectorStore, build_vector_store_from_jsonl
from src.ingest import IngestionPipeline
from src.config import default_config
from src.retriever import HybridRetriever

logger = logging.getLogger(__name__)
console = Console()

# Clean sample queries for the ChatGPT-like welcome cards (no emojis)
SAMPLE_CARDS = [
    {"label": "Attendance Policy", "query": "What is the minimum attendance requirement at MCET?"},
    {"label": "Library Rules", "query": "How many books can a student borrow from the library?"},
    {"label": "Leave Policy", "query": "How many working days does a student have to report emergency leave?"},
    {"label": "Admissions Info", "query": "How are B.Tech admissions conducted at MCET?"},
]


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
        default="data/index",
        help="Path to persistent vector store index directory",
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
    if vs_path.exists() and vs_path.is_dir():
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


def rebuild_index_pipeline(vector_store_dir: str) -> str:
    """End-to-end rebuild index: loader -> ingestion -> embedding vector store."""
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Write a default technical guide if raw directory is completely empty
    guide_path = raw_dir / "guide.md"
    if not any(raw_dir.iterdir()):
        guide_content = """# Knowledge Assistant Guide

The Knowledge Assistant is an enterprise-grade AI question answering system.
It utilizes a multi-channel hybrid retriever fusing dense embeddings, keywords (TF-IDF), and prefix matching.

## Architecture

The system consists of 4 distinct phases:
1. Document Ingestion: processes TXT, MD, and PDF documents.
2. Vector Indexing: builds dense and sparse representations.
3. Hybrid Retrieval: searches for relevant context.
4. Grounded Generation: generates answers strictly from context.
"""
        guide_path.write_text(guide_content, encoding="utf-8")

    chunks_jsonl = processed_dir / "chunks.jsonl"
    
    # 1. Ingest
    pipeline = IngestionPipeline(default_config)
    stats = pipeline.run(str(raw_dir), str(chunks_jsonl))

    # 2. Vector Store Index
    build_vector_store_from_jsonl(
        jsonl_path=str(chunks_jsonl),
        output_dir=vector_store_dir,
        model_type="tfidf",
    )

    return f"Indexed {stats.get('total_chunks', 0)} chunks successfully!"


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
            background-color: #0d0f12 !important;
        }

        h1, h2, h3 {
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
        }

        /* ChatGPT-like Chat styling */
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px 0;
        }

        .stChatMessage {
            background: rgba(255, 255, 255, 0.02) !important;
            border: 1px solid rgba(255, 255, 255, 0.04) !important;
            border-radius: 12px !important;
            margin-bottom: 16px !important;
            padding: 16px 20px !important;
        }

        /* Subtle indicator cards for grounding status */
        .grounded-card {
            border-left: 4px solid #10b981;
            padding-left: 14px;
            margin-bottom: 12px;
        }

        .abstention-card {
            border-left: 4px solid #ef4444;
            padding-left: 14px;
            margin-bottom: 12px;
        }
        
        .header-grounded {
            color: #10b981;
            font-size: 0.8rem;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }

        .header-abstention {
            color: #ef4444;
            font-size: 0.8rem;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }

        /* Clean suggest cards in grid layout */
        .card-suggestion {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 10px;
            padding: 16px;
            cursor: pointer;
            transition: all 0.2s ease;
            text-align: left;
            height: 100%;
        }

        .card-suggestion:hover {
            background: rgba(255, 255, 255, 0.06);
            border-color: rgba(56, 189, 248, 0.3);
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #090b0e !important;
            border-right: 1px solid rgba(255, 255, 255, 0.03) !important;
        }

        /* Sidebar buttons style */
        .stButton>button {
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_streamlit_assistant_msg(content: str, response_data: dict | None) -> None:
    """Helper to render assistant text and details like citations in Streamlit."""
    is_abstained = response_data.get("abstained", False) if response_data else False

    if is_abstained:
        st.markdown(
            f"""
            <div class="abstention-card">
                <div class="header-abstention">Abstention / Information Not Found</div>
                {content}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="grounded-card">
                <div class="header-grounded">Grounded Answer</div>
                {content}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if response_data:
        conf = response_data.get("retrieval_confidence", 0.0)
        latency = response_data.get("latency_ms", 0.0)
        model = response_data.get("model_name", "openai/gpt-4o-mini")

        if conf >= 0.70:
            badge_color = "#10b981"
        elif conf >= 0.30:
            badge_color = "#f97316"
        else:
            badge_color = "#ef4444"

        # Metadata Row
        st.markdown(
            f"<div style='font-size: 0.75rem; opacity: 0.6; margin-top: 8px; font-family: monospace;'>"
            f"model: {model} | "
            f"confidence: <span style='color: {badge_color}; font-weight: bold;'>{conf:.3f}</span> | "
            f"latency: {latency:.1f} ms"
            f"</div>",
            unsafe_allow_html=True,
        )

        if response_data.get("citations"):
            citations = response_data["citations"]
            with st.expander("Grounding Sources & Citations", expanded=False):
                for idx, cit in enumerate(citations, 1):
                    src_name = Path(cit["source"]).name
                    sec = cit.get("section") or "N/A"
                    page = f"Page {cit['page']}" if cit.get("page") is not None else "Page N/A"
                    st.markdown(
                        f"**[{idx}] `{src_name}`** | Section: *{sec}* | {page} | Confidence: `{cit.get('confidence', 0.0):.3f}`"
                    )
                    if cit.get("snippet"):
                        st.caption(f'Snippet: "{cit["snippet"]}"')


def run_streamlit_app(args: argparse.Namespace) -> None:
    """Run interactive Streamlit web application."""
    st.set_page_config(
        page_title="AI Knowledge Assistant",
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

    # --- SIDEBAR: CLEAN & PROFESSIONAL ---
    st.sidebar.markdown(
        "<div style='padding: 10px 0; text-align: center;'>"
        "<h3 style='margin: 0; font-family: \"Outfit\"; color: #f8fafc;'>Knowledge Assistant</h3>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Top level actions
    if st.sidebar.button("New Chat", key="btn_new_chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.sidebar.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # Collapsible Settings
    with st.sidebar.expander("Settings", expanded=False):
        st.session_state.model = st.text_input("LLM Model Name", value=st.session_state.model)
        st.session_state.threshold = st.slider("Confidence Threshold", 0.0, 1.0, value=st.session_state.threshold, step=0.05)
        st.caption("Higher values reject more out-of-scope queries.")
        st.session_state.offline = st.checkbox("Force Offline Mode", value=st.session_state.offline)

        if st.button("Rebuild Index", use_container_width=True):
            with st.spinner("Rebuilding..."):
                msg = rebuild_index_pipeline(args.vector_store)
                st.success(msg)

    # Collapsible Inspector
    with st.sidebar.expander("Database Inspector", expanded=False):
        st.subheader("Document Index Chunks")
        vs_path = Path(args.vector_store)
        if vs_path.exists():
            try:
                vstore = VectorStore.load(vs_path)
                st.caption(f"Loaded {len(vstore.chunks)} chunks from `{vs_path}`")
                
                search_q = st.text_input("Simulate Hybrid Search", key="inspector_search_box")
                if search_q:
                    retriever = HybridRetriever(vector_store=vstore)
                    res = retriever.retrieve(search_q)
                    st.write(f"Hits ({len(res.hits)}):")
                    for h_idx, hit in enumerate(res.hits, 1):
                        with st.expander(f"Hit {h_idx}: {Path(hit.chunk.source).name} ({hit.confidence_score:.3f})"):
                            st.write(f"Section: `{hit.chunk.section_header or 'N/A'}`")
                            st.text_area("Content", value=hit.chunk.text, height=100)
                else:
                    chunk_data = []
                    for idx, c in enumerate(vstore.chunks):
                        chunk_data.append({
                            "Index": idx + 1,
                            "Source": Path(c.source).name,
                            "Section": c.section_header or "N/A",
                            "Snippet": c.text[:80] + "..." if len(c.text) > 80 else c.text,
                        })
                    st.dataframe(chunk_data, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading: {e}")
        else:
            st.warning("No persistent index found.")

    # --- MAIN CHAT LAYOUT ---
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    # Welcome screen when there are no messages
    selected_sample = None
    if len(st.session_state.messages) == 0:
        st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align: center; max-width: 800px; margin: 0 auto;'>"
            "<h1 style='font-family: \"Outfit\"; font-size: 2.8rem; background: linear-gradient(135deg, #f8fafc 30%, #38bdf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>How can I help you today?</h1>"
            "<p style='color: #64748b; font-size: 1.1rem; margin-top: 10px; margin-bottom: 40px;'>Grounded in the MCET Student Handbook</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        # 2x2 Suggestion grid
        cols = st.columns(2)
        for idx, card in enumerate(SAMPLE_CARDS):
            col_idx = idx % 2
            with cols[col_idx]:
                if st.button(f"{card['label']}\n\n→ {card['query']}", key=f"card_{idx}", use_container_width=True):
                    selected_sample = card['query']

        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    else:
        # Render message history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    render_streamlit_assistant_msg(msg["content"], msg.get("response_data"))
                else:
                    st.markdown(msg["content"])

    st.markdown("</div>", unsafe_allow_html=True)

    # Chat Input Box at the bottom
    user_input = st.chat_input("Ask a grounded question...")

    # Choose query input source
    final_input = selected_sample or user_input
    if final_input:
        st.session_state.messages.append({"role": "user", "content": final_input})
        st.rerun()

    # If new query, invoke engine and generate response
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        user_query = st.session_state.messages[-1]["content"]

        engine_args = argparse.Namespace(
            vector_store=args.vector_store,
            model=st.session_state.model,
            offline=st.session_state.offline,
            threshold=st.session_state.threshold,
        )
        engine = init_engine(engine_args)

        with st.chat_message("assistant"):
            with st.spinner(""):
                response = engine.generate_answer(user_query)

            render_streamlit_assistant_msg(response.answer, response.to_dict())
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
