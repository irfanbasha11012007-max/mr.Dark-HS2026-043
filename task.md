# Task Tracking: Knowledge Assistant

## Member 1 Tasks — Phase 1: Document Ingestion + Evaluation
- [x] Project configuration (`src/config.py`, `.env.example`, `.gitignore`)
- [x] TXT document loader (`load_text_document`)
- [x] Markdown document loader with frontmatter extraction (`load_markdown_document`)
- [x] PDF document loader with page mapping (`load_pdf_document`)
- [x] Text cleaning & Unicode normalization (`clean_text`)
- [x] Recursive text chunking (`RecursiveCharacterChunker`)
- [x] Chunk overlap with natural boundary detection (`_extract_overlap_prefix`)
- [x] Section/header extraction & hierarchy tracking (`extract_section_spans`, `find_active_section`)
- [x] Chunk metadata & provenance mapping (`DocumentChunk`)
- [x] Character offset computation (`compute_character_offsets`)
- [x] Ingestion CLI & pipeline runner (`IngestionPipeline`, `main`)
- [x] Ingestion unit and integration tests (`tests/test_ingest.py`)
- [x] Evaluation questions dataset - in-scope (`evaluation/eval_questions.jsonl`)
- [x] Evaluation questions dataset - out-of-scope & adversarial (`evaluation/eval_questions.jsonl`)
- [x] Comprehensive documentation (`docs/ingestion.md`, `README.md`, `plan.md`, `task.md`)

## Next Phases (Other Members)
- [ ] Phase 2: Embeddings & Vector Indexing (Member 2)
- [ ] Phase 3: LLM Generation & Grounding (Member 3)
- [ ] Phase 4: Streamlit UI & Evaluation Execution (Member 4)
