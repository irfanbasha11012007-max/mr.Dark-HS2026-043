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
    load_text_document,
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
