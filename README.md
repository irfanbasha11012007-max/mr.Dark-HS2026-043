# Knowledge Assistant — Problem Statement 4

Knowledge Assistant is an enterprise-grade AI knowledge retrieval and question-answering system powered by a modular Retrieval-Augmented Generation (RAG) architecture.

---

## Team Responsibilities & Ownership

| Member | Branch | Assigned Phase | Ownership |
|---|---|---|---|
| **Member 1 (@harivarman-007)** | `feature/ingest-eval` | **Phase 1: Document Ingestion + Evaluation** | `src/config.py`, `src/ingest.py`, `tests/test_ingest.py`, `evaluation/eval_questions.jsonl`, `docs/ingestion.md` |
| **Member 2 (@mrdark5133)** | `feature/retriever` | **Phase 2: Embeddings & Hybrid Retrieval** | `src/embed_store.py`, `src/retriever.py`, `tests/test_retriever.py`, `docs/retrieval.md` |
| **Member 3** | `feature/generation-rag` | **Phase 3: LLM Generation & Citations** | Context assembly, prompt engineering, grounded answer generation |
| **Member 4** | `feature/ui-deployment` | **Phase 4: Streamlit UI & Orchestration** | Interactive user interface, end-to-end evaluation runner |

---

## Phase 1 Implementation Summary (Member 1: @harivarman-007)
- Multi-format document loaders: **Plain Text (`.txt`)**, **Markdown (`.md`)**, **PDF (`.pdf`)**.
- Comprehensive text cleaning: Unicode NFKC normalization, zero-width stripping, line break standardization.
- Recursive character chunker with hierarchical separators and natural boundary overlap.
- Section/header extraction with ancestor hierarchy tracking.
- Precise character offsets (`start_char`, `end_char`) and PDF page number mappings.
- Ingestion CLI with rich statistics export (`python -m src.ingest`).
- Benchmark evaluation dataset with in-scope and out-of-scope/adversarial question suites (`evaluation/eval_questions.jsonl`).

---

## Phase 2 Implementation Summary (Member 2: @mrdark5133)
- **Embedding Models**: Abstract `BaseEmbeddingModel`, `TfidfEmbeddingModel` with sublinear TF and n-grams, and deterministic `LocalDenseEmbeddingModel`.
- **VectorStore**: In-memory and persistent vector storage with $L_2$ unit normalization and $O(N)$ top-$k$ similarity search.
- **Index Rebuilder CLI**: `python -m src.embed_store` for end-to-end JSONL chunk vectorization and disk serialization.
- **Hybrid Retrieval Engine**: `HybridRetriever` fusing dense cosine similarity ($W=0.6$), sparse keyword frequency ($W=0.3$), and prefix matching ($W=0.1$).
- **Confidence Calibration & Threshold Gating**: Probabilistic confidence scaling in $[0.0, 1.0]$ with automatic rejection of out-of-scope queries.
- **Context Formatter**: Structured Markdown prompt assembly with document, section, and page citation provenance.
- **Test Suite**: 14 tests in `tests/test_retriever.py` covering embedding generation, persistence roundtrip, ranking, and threshold gating.

---

## Quick Start

### Installation
```bash
python -m pip install pytest pypdf scikit-learn numpy
```

### Running Ingestion & Indexing
```bash
# 1. Ingest raw documents into structured JSONL chunks
python -m src.ingest --input data/raw --output data/processed/chunks.jsonl --stats

# 2. Build persistent vector store index
python -m src.embed_store --input data/processed/chunks.jsonl --output data/index --model tfidf
```

### Running Tests
```bash
python -m pytest -v
```
