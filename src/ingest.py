"""Document Ingestion and Processing Pipeline.

This module provides data models, document loaders, text cleaning utilities,
recursive text chunkers with overlap and metadata tracking, and a command-line interface.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
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
