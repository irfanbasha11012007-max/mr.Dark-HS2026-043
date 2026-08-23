# Project Plan: Knowledge Assistant (Problem Statement 4)

## Architecture Overview

```mermaid
graph TD
    subgraph Phase 1 [Phase 1: Ingestion & Evaluation - Member 1: @harivarman-007]
        RawDocs[Raw Documents: TXT, MD, PDF] --> Loaders[Document Loaders]
        Loaders --> Clean[Text Cleaning & Normalizer]
        Clean --> Chunker[Recursive Chunker with Overlap]
        Chunker --> Meta[Metadata & Section Offsets]
        Meta --> ChunksJSONL[Structured Chunks JSONL]
        EvalSet[Evaluation Benchmark Dataset]
    end

    subgraph Phase 2 [Phase 2: Embeddings & Retrieval - Member 2: @mrdark5133]
        ChunksJSONL --> Embedder[Embedding Pipeline]
        Embedder --> VectorDB[(Vector Store)]
        VectorDB --> Retriever[Hybrid Retriever]
        Retriever --> ConfidenceGate[Confidence & Threshold Gate]
    end

    subgraph Phase 3 [Phase 3: Generation & Grounding - Member 3]
        ConfidenceGate --> ContextBuilder[Context Assembler]
        ContextBuilder --> LLM[LLM Answer Generator]
        LLM --> Citations[Citations & Source Grounding]
    end

    subgraph Phase 4 [Phase 4: Interface & Evaluation Runner - Member 4]
        Citations --> StreamlitUI[Streamlit Interactive UI]
        EvalSet --> EvalRunner[RAG Benchmark Runner]
        EvalRunner --> MetricsReport[Evaluation Metrics Dashboard]
    end
```

## Phase Breakdown

1. **Phase 1: Document Ingestion + Evaluation (Completed - Member 1: @harivarman-007)**
   - Configuration management (`src/config.py`)
   - Document loaders for TXT, MD, and PDF (`src/ingest.py`)
   - Text cleaning, normalization, and zero-width filtering
   - Recursive character chunker with sliding window overlap
   - Section hierarchy and character offset computation
   - Ingestion CLI with statistics (`python -m src.ingest`)
   - Comprehensive test suite (`tests/test_ingest.py`)
   - In-scope and out-of-scope evaluation questions (`evaluation/eval_questions.jsonl`)
   - Ingestion workflow documentation (`docs/ingestion.md`)

2. **Phase 2: Embeddings & Vector Search (Completed - Member 2: @mrdark5133)**
   - Abstract embedding interface and concrete models (`BaseEmbeddingModel`, `TfidfEmbeddingModel`, `LocalDenseEmbeddingModel`)
   - Persistent VectorStore with L2 normalization and cosine similarity (`VectorStore`)
   - Index building and rebuilder CLI (`python -m src.embed_store`)
   - Hybrid retriever with 3-channel scoring: dense, sparse keywords, prefix matching (`HybridRetriever`)
   - Non-linear confidence score calibration and threshold gating
   - Structured context assembly with provenance citations (`format_context`)
   - Unit & integration test suite (`tests/test_retriever.py`)
   - Retrieval architecture documentation (`docs/retrieval.md`)

3. **Phase 3: LLM Generation & Grounding (Member 3)**
   - Prompt templates and context assembly
   - LLM generation with citation grounding
   - Hallucination prevention and boundary guardrails

4. **Phase 4: UI & End-to-End Evaluation (Member 4)**
   - Streamlit user interface
   - Automated evaluation suite runner
   - Precision, recall, and groundedness metrics reporting
