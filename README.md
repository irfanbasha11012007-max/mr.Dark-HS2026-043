# Knowledge Assistant — Problem Statement 4

Knowledge Assistant is an enterprise-grade AI knowledge retrieval and question-answering system powered by a modular Retrieval-Augmented Generation (RAG) architecture.

---

## Team Responsibilities & Ownership

| Member | Branch | Assigned Phase | Ownership |
|---|---|---|---|
| **Member 1 (@harivarman-007)** | `feature/ingest-eval` | **Phase 1: Document Ingestion + Evaluation** | `src/config.py`, `src/ingest.py`, `tests/test_ingest.py`, `evaluation/eval_questions.jsonl`, `docs/ingestion.md` |
| **Member 2** | `feature/embeddings-retrieval` | **Phase 2: Embeddings & Vector Search** | Embedding generation, vector database indexing, hybrid search |
| **Member 3** | `feature/generation-rag` | **Phase 3: LLM Generation & Citations** | Context assembly, prompt engineering, grounded answer generation |
| **Member 4** | `feature/ui-deployment` | **Phase 4: Streamlit UI & Orchestration** | Interactive user interface, end-to-end evaluation runner |

---

## Phase 1 Implementation Summary

Member 1 (`@harivarman-007`) has completed Phase 1:
- Multi-format document loaders: **Plain Text (`.txt`)**, **Markdown (`.md`)**, **PDF (`.pdf`)**.
- Comprehensive text cleaning: Unicode NFKC normalization, zero-width stripping, line break standardization.
- Recursive character chunker with hierarchical separators and natural boundary overlap.
- Section/header extraction with ancestor hierarchy tracking.
- Precise character offsets (`start_char`, `end_char`) and PDF page number mappings.
- Ingestion CLI with rich statistics export (`python -m src.ingest`).
- Unit and integration test suite with 100% pass rate (`pytest`).
- Benchmark evaluation dataset with in-scope and out-of-scope/adversarial question suites (`evaluation/eval_questions.jsonl`).

---

## Quick Start

### Installation
```bash
python -m pip install pytest pypdf
```

### Running Ingestion CLI
```bash
# Ingest raw documents into structured JSONL chunks
python -m src.ingest --input data/raw --output data/processed/chunks.jsonl --stats
```

### Running Tests
```bash
python -m pytest -v tests/test_ingest.py
```
