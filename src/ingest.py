"""Document Ingestion and Processing Pipeline.

This module provides data models, document loaders, text cleaning utilities,
recursive text chunkers with overlap and metadata tracking, and a command-line interface.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pypdf import PdfReader

from src.config import IngestionConfig, default_config

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a raw ingested document before chunking."""

    doc_id: str
    content: str
    source: str
    doc_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.content)

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize document to dictionary."""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "source": self.source,
            "doc_type": self.doc_type,
            "metadata": self.metadata,
            "char_count": self.char_count,
            "word_count": self.word_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Deserialize document from dictionary."""
        return cls(
            doc_id=data["doc_id"],
            content=data["content"],
            source=data.get("source", ""),
            doc_type=data.get("doc_type", "unknown"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DocumentChunk:
    """Represents a processed and chunked segment of a document."""

    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    chunk_index: int
    source: str
    section_header: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize document chunk to a clean dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "chunk_index": self.chunk_index,
            "section_header": self.section_header,
            "source": self.source,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        """Deserialize document chunk from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            text=data["text"],
            start_char=data.get("start_char", 0),
            end_char=data.get("end_char", len(data.get("text", ""))),
            chunk_index=data.get("chunk_index", 0),
            source=data.get("source", ""),
            section_header=data.get("section_header"),
            metadata=data.get("metadata", {}),
        )


def generate_doc_id(file_path: Union[str, Path], content: Optional[str] = None) -> str:
    """Generate a deterministic document identifier from path and optional content hash."""
    path_obj = Path(file_path)
    clean_path = str(path_obj.as_posix())
    if content:
        content_hash = hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()[:8]
        return f"{path_obj.stem}_{content_hash}"
    path_hash = hashlib.sha256(clean_path.encode("utf-8")).hexdigest()[:8]
    return f"{path_obj.stem}_{path_hash}"


def load_text_document(file_path: Union[str, Path]) -> Document:
    """Load a plain text document with multi-encoding fallback support."""
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not path.is_file():
        raise ValueError(f"Path is not a regular file: {file_path}")

    encodings = ("utf-8", "utf-8-sig", "latin-1", "cp1252")
    content: Optional[str] = None
    used_encoding: Optional[str] = None

    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                content = f.read()
                used_encoding = enc
                break
        except UnicodeDecodeError:
            continue

    if content is None:
        raise ValueError(f"Failed to decode text file with supported encodings: {file_path}")

    file_stat = path.stat()
    doc_id = generate_doc_id(path, content)

    metadata: Dict[str, Any] = {
        "file_name": path.name,
        "file_path": str(path.as_posix()),
        "file_extension": path.suffix.lower(),
        "file_size_bytes": file_stat.st_size,
        "encoding": used_encoding,
        "modified_time": datetime.fromtimestamp(file_stat.st_mtime, tz=timezone.utc).isoformat(),
    }

    return Document(
        doc_id=doc_id,
        content=content,
        source=str(path.as_posix()),
        doc_type="text",
        metadata=metadata,
    )


def parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """Parse optional YAML/metadata frontmatter block at the top of markdown files.

    Returns:
        A tuple of (frontmatter_dict, remaining_body_text).
    """
    frontmatter: Dict[str, Any] = {}
    if not text.startswith("---"):
        return frontmatter, text

    end_idx = text.find("\n---", 3)
    if end_idx == -1:
        return frontmatter, text

    fm_raw = text[3:end_idx].strip()
    body = text[end_idx + 4:].lstrip("\r\n")

    for line in fm_raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip().strip("\"'")
        # Convert simple types
        if val.lower() == "true":
            frontmatter[key] = True
        elif val.lower() == "false":
            frontmatter[key] = False
        elif val.isdigit():
            frontmatter[key] = int(val)
        else:
            frontmatter[key] = val

    return frontmatter, body


def load_markdown_document(file_path: Union[str, Path], strip_frontmatter: bool = False) -> Document:
    """Load a markdown document with frontmatter parsing and header extraction."""
    raw_doc = load_text_document(file_path)
    frontmatter, body = parse_frontmatter(raw_doc.content)

    # Extract all markdown headers
    headers = re.findall(r"^(#{1,6})\s+(.+)$", raw_doc.content, flags=re.MULTILINE)
    detected_headers = [{"level": len(h[0]), "title": h[1].strip()} for h in headers]

    # Extract title if present (first H1 or frontmatter title)
    title = frontmatter.get("title")
    if not title:
        for h in detected_headers:
            if h["level"] == 1:
                title = h["title"]
                break

    metadata = dict(raw_doc.metadata)
    metadata.update({
        "frontmatter": frontmatter,
        "headers": detected_headers,
        "header_count": len(detected_headers),
        "title": title or Path(file_path).stem,
    })

    content_to_use = body if strip_frontmatter and frontmatter else raw_doc.content

    return Document(
        doc_id=raw_doc.doc_id,
        content=content_to_use,
        source=raw_doc.source,
        doc_type="markdown",
        metadata=metadata,
    )


def load_pdf_document(file_path: Union[str, Path]) -> Document:
    """Load a PDF document page by page using pypdf.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Document instance with per-page text, page offsets, and PDF metadata.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If file is not a valid or readable PDF.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    if not path.is_file():
        raise ValueError(f"Path is not a regular file: {file_path}")

    try:
        reader = PdfReader(str(path))
    except Exception as e:
        raise ValueError(f"Could not open or parse PDF '{file_path}': {e}") from e

    num_pages = len(reader.pages)
    page_texts: List[str] = []
    page_spans: List[Dict[str, Any]] = []
    current_char_offset = 0

    for page_idx, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
        except Exception as e:
            logger.warning("Error extracting text from page %d of %s: %s", page_idx + 1, path.name, e)
            page_text = ""

        page_texts.append(page_text)
        page_len = len(page_text)
        page_spans.append({
            "page_number": page_idx + 1,
            "start_char": current_char_offset,
            "end_char": current_char_offset + page_len,
            "char_count": page_len,
        })
        # Account for page separator newline
        current_char_offset += page_len + 1

    full_content = "\n".join(page_texts)
    file_stat = path.stat()
    doc_id = generate_doc_id(path, full_content)

    # Extract standard PDF metadata if present
    pdf_info: Dict[str, Any] = {}
    if reader.metadata:
        for k, v in reader.metadata.items():
            if v:
                clean_key = k.lstrip("/").lower()
                pdf_info[clean_key] = str(v)

    metadata: Dict[str, Any] = {
        "file_name": path.name,
        "file_path": str(path.as_posix()),
        "file_extension": ".pdf",
        "file_size_bytes": file_stat.st_size,
        "page_count": num_pages,
        "pages": page_spans,
        "pdf_info": pdf_info,
        "title": pdf_info.get("title") or path.stem,
        "author": pdf_info.get("author"),
        "modified_time": datetime.fromtimestamp(file_stat.st_mtime, tz=timezone.utc).isoformat(),
    }

    return Document(
        doc_id=doc_id,
        content=full_content,
        source=str(path.as_posix()),
        doc_type="pdf",
        metadata=metadata,
    )


def load_document(file_path: Union[str, Path], config: Optional[IngestionConfig] = None) -> Document:
    """Dispatch and load a document based on its file extension."""
    path = Path(file_path).resolve()
    ext = path.suffix.lower()
    cfg = config or default_config

    if ext not in cfg.supported_extensions:
        raise ValueError(
            f"Unsupported file extension '{ext}' for {file_path}. Supported extensions: {cfg.supported_extensions}"
        )

    if ext in (".txt", ".text"):
        return load_text_document(path)
    elif ext in (".md", ".markdown"):
        return load_markdown_document(path)
    elif ext == ".pdf":
        return load_pdf_document(path)
    else:
        # Fallback to plain text for generic supported extensions
        return load_text_document(path)


def load_directory(
    dir_path: Union[str, Path],
    recursive: bool = True,
    config: Optional[IngestionConfig] = None,
) -> List[Document]:
    """Scan and load all supported documents from a directory.

    Args:
        dir_path: Path to directory.
        recursive: Whether to scan subdirectories recursively.
        config: Ingestion configuration.

    Returns:
        List of loaded Document objects.
    """
    path = Path(dir_path).resolve()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    cfg = config or default_config
    docs: List[Document] = []

    pattern = "**/*" if recursive else "*"
    for item in sorted(path.glob(pattern)):
        if item.is_file() and item.suffix.lower() in cfg.supported_extensions:
            try:
                docs.append(load_document(item, config=cfg))
            except Exception as e:
                logger.error("Failed to load document '%s': %s", item, e)

    return docs
