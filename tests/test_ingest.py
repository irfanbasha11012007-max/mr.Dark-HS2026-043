"""Unit and integration tests for document ingestion and evaluation."""

import os
import tempfile
from pathlib import Path

import pytest
from pypdf import PdfWriter

from src.config import IngestionConfig
from src.ingest import (
    Document,
    DocumentChunk,
    RecursiveCharacterChunker,
    chunk_document,
    clean_text,
    compute_character_offsets,
    extract_section_spans,
    find_active_section,
    generate_doc_id,
    load_directory,
    load_document,
    load_markdown_document,
    load_pdf_document,
    load_text_document,
    parse_frontmatter,
)


class TestTextDocumentLoader:
    """Test suite for plain text document loading and validation."""

    def test_load_text_document_valid(self, tmp_path: Path):
        sample_file = tmp_path / "sample.txt"
        content = "Knowledge Assistant is an AI-powered system designed for RAG workflows.\nIt processes documents efficiently."
        sample_file.write_text(content, encoding="utf-8")

        doc = load_text_document(sample_file)

        assert isinstance(doc, Document)
        assert doc.doc_type == "text"
        assert doc.content == content
        assert doc.source == str(sample_file.as_posix())
        assert doc.char_count == len(content)
        assert doc.word_count == len(content.split())
        assert doc.metadata["file_name"] == "sample.txt"
        assert doc.metadata["file_extension"] == ".txt"
        assert doc.metadata["encoding"] == "utf-8"
        assert doc.metadata["file_size_bytes"] > 0
        assert "modified_time" in doc.metadata

    def test_load_text_document_fallback_encoding(self, tmp_path: Path):
        sample_file = tmp_path / "latin1_sample.txt"
        latin_text = "Accented characters: café, naïve, façade."
        sample_file.write_bytes(latin_text.encode("latin-1"))

        doc = load_text_document(sample_file)
        assert doc.doc_type == "text"
        assert "café" in doc.content
        assert doc.metadata["encoding"] in ("latin-1", "cp1252", "utf-8")

    def test_load_text_document_empty_file(self, tmp_path: Path):
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("", encoding="utf-8")

        doc = load_text_document(empty_file)
        assert doc.content == ""
        assert doc.char_count == 0
        assert doc.word_count == 0

    def test_load_text_document_not_found(self, tmp_path: Path):
        non_existent = tmp_path / "missing.txt"
        with pytest.raises(FileNotFoundError):
            load_text_document(non_existent)

    def test_load_text_document_directory_raises_error(self, tmp_path: Path):
        dir_path = tmp_path / "subdir"
        dir_path.mkdir()
        with pytest.raises(ValueError, match="not a regular file"):
            load_text_document(dir_path)

    def test_generate_doc_id_deterministic(self, tmp_path: Path):
        file_path = tmp_path / "doc1.txt"
        content = "Deterministic content test"
        id1 = generate_doc_id(file_path, content)
        id2 = generate_doc_id(file_path, content)
        assert id1 == id2
        assert "doc1" in id1


class TestMarkdownDocumentLoader:
    """Test suite for Markdown document loading and frontmatter parsing."""

    def test_load_markdown_basic(self, tmp_path: Path):
        md_file = tmp_path / "guide.md"
        content = "# Getting Started\nWelcome to the knowledge base.\n\n## Installation\nRun `pip install -r requirements.txt`."
        md_file.write_text(content, encoding="utf-8")

        doc = load_markdown_document(md_file)
        assert doc.doc_type == "markdown"
        assert doc.metadata["title"] == "Getting Started"
        assert doc.metadata["header_count"] == 2
        assert doc.metadata["headers"][0] == {"level": 1, "title": "Getting Started"}
        assert doc.metadata["headers"][1] == {"level": 2, "title": "Installation"}

    def test_load_markdown_with_frontmatter(self, tmp_path: Path):
        md_file = tmp_path / "article.md"
        content = """---
title: System Architecture
author: Harivarman
draft: false
version: 2
---
# Overview
This document describes the core ingestion subsystem.
"""
        md_file.write_text(content, encoding="utf-8")

        doc = load_markdown_document(md_file)
        assert doc.metadata["frontmatter"]["title"] == "System Architecture"
        assert doc.metadata["frontmatter"]["author"] == "Harivarman"
        assert doc.metadata["frontmatter"]["draft"] is False
        assert doc.metadata["frontmatter"]["version"] == 2
        assert doc.metadata["title"] == "System Architecture"

    def test_load_markdown_strip_frontmatter(self, tmp_path: Path):
        md_file = tmp_path / "clean_article.md"
        content = """---
title: API Spec
version: 1
---
# Endpoint Reference
GET /api/v1/health
"""
        md_file.write_text(content, encoding="utf-8")

        doc = load_markdown_document(md_file, strip_frontmatter=True)
        assert not doc.content.startswith("---")
        assert doc.content.startswith("# Endpoint Reference")
        assert doc.metadata["frontmatter"]["title"] == "API Spec"

    def test_parse_frontmatter_edge_cases(self):
        fm, body = parse_frontmatter("---\ntitle: Incomplete\nno closing marker")
        assert fm == {}
        assert "Incomplete" in body

        fm, body = parse_frontmatter("Just a regular string without frontmatter.")
        assert fm == {}
        assert body == "Just a regular string without frontmatter."


class TestPdfDocumentLoader:
    """Test suite for PDF document loading and metadata extraction."""

    def test_load_pdf_multi_page(self, tmp_path: Path):
        pdf_path = tmp_path / "sample_doc.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=300, height=300)
        writer.add_blank_page(width=300, height=300)
        with open(pdf_path, "wb") as f:
            writer.write(f)
        writer.close()

        doc = load_pdf_document(pdf_path)
        assert doc.doc_type == "pdf"
        assert doc.metadata["page_count"] == 2
        assert len(doc.metadata["pages"]) == 2
        assert doc.metadata["pages"][0]["page_number"] == 1
        assert doc.metadata["pages"][1]["page_number"] == 2

    def test_load_pdf_not_found(self, tmp_path: Path):
        non_existent = tmp_path / "missing.pdf"
        with pytest.raises(FileNotFoundError):
            load_pdf_document(non_existent)

    def test_load_document_dispatcher(self, tmp_path: Path):
        t_file = tmp_path / "test.txt"
        t_file.write_text("Plain text content", encoding="utf-8")
        m_file = tmp_path / "test.md"
        m_file.write_text("# Markdown Title\nContent", encoding="utf-8")

        doc_t = load_document(t_file)
        assert doc_t.doc_type == "text"

        doc_m = load_document(m_file)
        assert doc_m.doc_type == "markdown"

        bad_file = tmp_path / "test.exe"
        bad_file.write_bytes(b"bad binary")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_document(bad_file)

    def test_load_directory_mixed_files(self, tmp_path: Path):
        (tmp_path / "doc1.txt").write_text("First file", encoding="utf-8")
        (tmp_path / "doc2.md").write_text("# Second file", encoding="utf-8")
        sub = tmp_path / "nested"
        sub.mkdir()
        (sub / "doc3.txt").write_text("Nested third file", encoding="utf-8")

        docs = load_directory(tmp_path, recursive=True)
        assert len(docs) == 3
        types = {d.doc_type for d in docs}
        assert "text" in types
        assert "markdown" in types


class TestTextCleaning:
    """Test suite for text cleaning and normalization."""

    def test_clean_text_normalizes_whitespace(self):
        raw = "Line 1   \n\n\n\n\nLine 2     with   extra    spaces."
        cleaned = clean_text(raw)
        assert cleaned == "Line 1\n\nLine 2 with extra spaces."

    def test_clean_text_removes_zero_width_and_unifies_linebreaks(self):
        raw = "Hello\r\n\r\n\r\nWorld\u200b with \u00a0space\r\nand\ufeff BOM."
        cleaned = clean_text(raw)
        assert "\r" not in cleaned
        assert "\u200b" not in cleaned
        assert "\ufeff" not in cleaned
        assert cleaned == "Hello\n\nWorld with space\nand BOM."

    def test_clean_empty_text(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""


class TestRecursiveChunkerAndOffsets:
    """Test suite for recursive chunking, overlap, header mapping, and offsets."""

    def test_chunker_basic_splitting(self):
        text = "Short introductory paragraph.\n\n" + "A" * 300 + "\n\n" + "B" * 300
        chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=0, min_chunk_length=10)
        chunks = chunker.split_text(text)
        assert len(chunks) >= 3
        for c in chunks:
            assert len(c) <= 250

    def test_chunker_with_overlap(self):
        text = "Sentence one is clear. Sentence two is descriptive. Sentence three has details. Sentence four concludes."
        chunker = RecursiveCharacterChunker(chunk_size=60, chunk_overlap=25, min_chunk_length=10)
        chunks = chunker.split_text(text)
        assert len(chunks) > 1

    def test_extract_sections_and_active_header(self):
        doc_text = "# Overview\nSystem overview details.\n\n## Component A\nDetails about component A.\n\n### Sub-feature A1\nDeep details."
        spans = extract_section_spans(doc_text)
        assert len(spans) == 3
        assert spans[0]["title"] == "Overview"
        assert spans[1]["title"] == "Component A"
        assert spans[2]["title"] == "Sub-feature A1"

        # Check active section at offset inside Sub-feature A1
        sub_offset = doc_text.find("Deep details")
        header, hierarchy = find_active_section(sub_offset, spans)
        assert header == "Sub-feature A1"
        assert hierarchy == ["Overview", "Component A", "Sub-feature A1"]

    def test_chunk_document_end_to_end(self):
        content = "# Knowledge Assistant\n\n## Ingestion Module\nThe ingestion pipeline is responsible for parsing TXT, Markdown, and PDF documents.\n\n## Evaluation Module\nThe evaluation module benchmarks retrieval and response accuracy."
        doc = Document(
            doc_id="test_doc_01",
            content=content,
            source="docs/test.md",
            doc_type="markdown",
            metadata={"file_name": "test.md"},
        )
        config = IngestionConfig(chunk_size=100, chunk_overlap=20, min_chunk_length=15)
        chunks = chunk_document(doc, config)

        assert len(chunks) > 0
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"test_doc_01#chunk_{i:04d}"
            assert chunk.doc_id == "test_doc_01"
            assert chunk.start_char >= 0
            assert chunk.end_char >= chunk.start_char
            assert chunk.metadata["doc_type"] == "markdown"
            assert chunk.metadata["file_name"] == "test.md"
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["total_chunks"] == len(chunks)

    def test_document_chunk_serialization_roundtrip(self):
        chunk = DocumentChunk(
            chunk_id="doc1#chunk_0000",
            doc_id="doc1",
            text="Testing serialization roundtrip.",
            start_char=0,
            end_char=32,
            chunk_index=0,
            source="doc1.txt",
            section_header="Intro",
            metadata={"author": "Harivarman", "priority": 1},
        )
        data = chunk.to_dict()
        restored = DocumentChunk.from_dict(data)

        assert restored.chunk_id == chunk.chunk_id
        assert restored.doc_id == chunk.doc_id
        assert restored.text == chunk.text
        assert restored.start_char == chunk.start_char
        assert restored.end_char == chunk.end_char
        assert restored.chunk_index == chunk.chunk_index
        assert restored.section_header == chunk.section_header
        assert restored.metadata == chunk.metadata
