"""Unit and integration tests for document ingestion and evaluation."""

import os
import tempfile
from pathlib import Path

import pytest

from src.config import IngestionConfig
from src.ingest import (
    Document,
    DocumentChunk,
    clean_text,
    generate_doc_id,
    load_markdown_document,
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
        # Missing closing marker
        fm, body = parse_frontmatter("---\ntitle: Incomplete\nno closing marker")
        assert fm == {}
        assert "Incomplete" in body

        # Plain text without frontmatter
        fm, body = parse_frontmatter("Just a regular string without frontmatter.")
        assert fm == {}
        assert body == "Just a regular string without frontmatter."
