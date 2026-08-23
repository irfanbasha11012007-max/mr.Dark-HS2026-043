"""Document Ingestion and Processing Pipeline.

This module provides data models, document loaders, text cleaning utilities,
recursive text chunkers with overlap and metadata tracking, and a command-line interface.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
    """Load a plain text document with multi-encoding fallback support.

    Args:
        file_path: Path to the plain text file.

    Returns:
        Document instance containing content and file metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a file or cannot be decoded.
    """
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
