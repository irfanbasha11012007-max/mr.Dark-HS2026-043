"""Document Ingestion and Processing Pipeline.

This module provides data models, document loaders, text cleaning utilities,
recursive text chunkers with overlap and metadata tracking, and a command-line interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import unicodedata
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


def clean_text(
    text: str,
    clean_whitespace: bool = True,
    normalize_unicode: bool = True,
    fix_linebreaks: bool = True,
    remove_control_chars: bool = True,
) -> str:
    """Clean and normalize raw document text."""
    if not text:
        return ""

    cleaned = text

    if fix_linebreaks:
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    if normalize_unicode:
        cleaned = unicodedata.normalize("NFKC", cleaned)
        cleaned = cleaned.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
        cleaned = cleaned.replace("\u200c", "").replace("\u200d", "")

    if remove_control_chars:
        cleaned = "".join(
            ch for ch in cleaned
            if ch in ("\n", "\t") or (unicodedata.category(ch) != "Cc" and unicodedata.category(ch) != "Cf")
        )

    if clean_whitespace:
        lines = [re.sub(r"[ \t]+$", "", line) for line in cleaned.split("\n")]
        cleaned = "\n".join(lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[^\S\n]+", " ", cleaned)

    return cleaned.strip()


def generate_doc_id(file_path: Union[str, Path], content: Optional[str] = None) -> str:
    """Generate a deterministic document identifier from path and optional content hash."""
    path_obj = Path(file_path)
    clean_path = str(path_obj.as_posix())
    if content:
        content_hash = hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()[:8]
        return f"{path_obj.stem}_{content_hash}"
    path_hash = hashlib.sha256(clean_path.encode("utf-8")).hexdigest()[:8]
    return f"{path_obj.stem}_{path_hash}"


def extract_section_spans(text: str) -> List[Dict[str, Any]]:
    """Detect section headings and their character ranges in document text."""
    section_spans: List[Dict[str, Any]] = []

    md_header_regex = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    for match in md_header_regex.finditer(text):
        level = len(match.group(1))
        title = match.group(2).strip()
        section_spans.append({
            "start": match.start(),
            "end": match.end(),
            "level": level,
            "title": title,
            "raw": match.group(0),
        })

    if not section_spans:
        num_header_regex = re.compile(r"^(?:Section\s+)?(\d+(?:\.\d+)*)\s+([A-Z][^\n\r]+)$", re.MULTILINE)
        for match in num_header_regex.finditer(text):
            sec_num = match.group(1)
            title = match.group(2).strip()
            level = len(sec_num.split("."))
            section_spans.append({
                "start": match.start(),
                "end": match.end(),
                "level": level,
                "title": f"{sec_num} {title}",
                "raw": match.group(0),
            })

    section_spans.sort(key=lambda s: s["start"])
    return section_spans


def find_active_section(char_offset: int, section_spans: Sequence[Dict[str, Any]]) -> Tuple[Optional[str], List[str]]:
    """Determine the active section header and hierarchical trail for a given character offset."""
    if not section_spans:
        return None, []

    active_span: Optional[Dict[str, Any]] = None
    hierarchy: List[str] = []

    for span in section_spans:
        if span["start"] <= char_offset:
            active_span = span
            level = span["level"]
            while hierarchy and len(hierarchy) >= level:
                hierarchy.pop()
            hierarchy.append(span["title"])
        else:
            break

    if active_span is None:
        return None, []

    return active_span["title"], list(hierarchy)


def map_offset_to_page(char_offset: int, page_spans: Sequence[Dict[str, Any]]) -> Optional[int]:
    """Map a character offset in PDF content to a 1-indexed page number."""
    for page in page_spans:
        if page["start_char"] <= char_offset <= page["end_char"]:
            return page["page_number"]
    return page_spans[0]["page_number"] if page_spans else None


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
    """Parse optional YAML/metadata frontmatter block at the top of markdown files."""
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

    headers = re.findall(r"^(#{1,6})\s+(.+)$", raw_doc.content, flags=re.MULTILINE)
    detected_headers = [{"level": len(h[0]), "title": h[1].strip()} for h in headers]

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
    """Load a PDF document page by page using pypdf."""
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
        current_char_offset += page_len + 1

    full_content = "\n".join(page_texts)
    file_stat = path.stat()
    doc_id = generate_doc_id(path, full_content)

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
        return load_text_document(path)


def load_directory(
    dir_path: Union[str, Path],
    recursive: bool = True,
    config: Optional[IngestionConfig] = None,
) -> List[Document]:
    """Scan and load all supported documents from a directory."""
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


class RecursiveCharacterChunker:
    """Splits text recursively based on a prioritized hierarchy of natural boundary separators."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_length: int = 20,
        separators: Optional[Sequence[str]] = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.separators = tuple(separators or (
            "\n\n# ",
            "\n\n## ",
            "\n\n### ",
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            "",
        ))

    def _split_text_recursive(self, text: str, separators: Sequence[str]) -> List[str]:
        """Recursively split text by separators until pieces fit within chunk_size."""
        final_chunks: List[str] = []
        if not text:
            return final_chunks

        chosen_separator = ""
        new_separators: Sequence[str] = ()

        for i, sep in enumerate(separators):
            if sep == "":
                chosen_separator = ""
                new_separators = ()
                break
            if sep in text:
                chosen_separator = sep
                new_separators = separators[i + 1:]
                break

        if chosen_separator:
            splits = text.split(chosen_separator)
        else:
            splits = list(text)

        good_splits: List[str] = []
        for s in splits:
            if chosen_separator and chosen_separator.strip():
                segment = s if not good_splits else (chosen_separator + s if not s.startswith(chosen_separator) else s)
            else:
                segment = s

            if len(segment) <= self.chunk_size:
                good_splits.append(segment)
            else:
                if new_separators:
                    other_splits = self._split_text_recursive(segment, new_separators)
                    good_splits.extend(other_splits)
                else:
                    for j in range(0, len(segment), self.chunk_size):
                        good_splits.append(segment[j:j + self.chunk_size])

        return good_splits

    def _extract_overlap_prefix(self, previous_chunk: str) -> str:
        """Extract a natural trailing slice from previous chunk to use as overlap."""
        if not previous_chunk or self.chunk_overlap <= 0:
            return ""

        overlap_slice = previous_chunk[-self.chunk_overlap:]
        for delimiter in ("\n\n", "\n", ". ", "? ", "! ", " "):
            idx = overlap_slice.find(delimiter)
            if idx != -1 and idx + len(delimiter) < len(overlap_slice):
                return overlap_slice[idx + len(delimiter):]

        return overlap_slice

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks with recursive boundary splitting and chunk overlap."""
        if not text or not text.strip():
            return []

        if len(text) <= self.chunk_size:
            return [text.strip()] if len(text.strip()) >= self.min_chunk_length else [text.strip()]

        raw_splits = self._split_text_recursive(text, self.separators)

        merged_chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for piece in raw_splits:
            if not piece:
                continue
            piece_len = len(piece)
            if current_length + piece_len <= self.chunk_size:
                current_chunk.append(piece)
                current_length += piece_len
            else:
                if current_chunk:
                    merged = "".join(current_chunk).strip()
                    if len(merged) >= self.min_chunk_length:
                        merged_chunks.append(merged)
                current_chunk = [piece]
                current_length = piece_len

        if current_chunk:
            merged = "".join(current_chunk).strip()
            if len(merged) >= self.min_chunk_length or not merged_chunks:
                merged_chunks.append(merged)

        if self.chunk_overlap <= 0 or len(merged_chunks) <= 1:
            return merged_chunks

        overlapped_chunks: List[str] = [merged_chunks[0]]
        for i in range(1, len(merged_chunks)):
            prev_chunk = merged_chunks[i - 1]
            curr_chunk = merged_chunks[i]
            prefix = self._extract_overlap_prefix(prev_chunk)
            if prefix and not curr_chunk.startswith(prefix):
                combined = (prefix + " " + curr_chunk).strip()
                if len(combined) <= self.chunk_size + self.chunk_overlap:
                    overlapped_chunks.append(combined)
                else:
                    overlapped_chunks.append(curr_chunk)
            else:
                overlapped_chunks.append(curr_chunk)

        return overlapped_chunks


def compute_character_offsets(
    chunks: Sequence[str],
    original_text: str,
) -> List[Tuple[int, int]]:
    """Accurately compute (start_char, end_char) offsets for each chunk in the source text."""
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    doc_len = len(original_text)

    for chunk in chunks:
        clean_target = chunk.strip()
        if not clean_target:
            offsets.append((cursor, cursor))
            continue

        found_idx = original_text.find(clean_target, cursor)

        if found_idx == -1:
            found_idx = original_text.find(clean_target)

        if found_idx == -1:
            prefix_sample = clean_target[:min(35, len(clean_target))]
            found_idx = original_text.find(prefix_sample, cursor)
            if found_idx == -1:
                found_idx = original_text.find(prefix_sample)

        if found_idx != -1:
            start_pos = found_idx
            end_pos = min(start_pos + len(clean_target), doc_len)
            cursor = max(cursor, start_pos + 1)
        else:
            start_pos = min(cursor, doc_len)
            end_pos = min(start_pos + len(clean_target), doc_len)
            cursor = end_pos

        offsets.append((start_pos, end_pos))

    return offsets


def chunk_document(
    doc: Document,
    config: Optional[IngestionConfig] = None,
) -> List[DocumentChunk]:
    """Process and chunk a Document into a list of enriched DocumentChunk objects."""
    cfg = config or default_config
    text_to_process = doc.content

    if cfg.clean_whitespace:
        text_to_process = clean_text(
            text_to_process,
            clean_whitespace=cfg.clean_whitespace,
            normalize_unicode=cfg.normalize_unicode,
        )

    section_spans = extract_section_spans(text_to_process) if cfg.extract_headers else []

    chunker = RecursiveCharacterChunker(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        min_chunk_length=cfg.min_chunk_length,
        separators=cfg.separators,
    )

    raw_chunk_texts = chunker.split_text(text_to_process)
    if not raw_chunk_texts:
        return []

    offsets = compute_character_offsets(raw_chunk_texts, text_to_process)
    total_chunks = len(raw_chunk_texts)
    chunks: List[DocumentChunk] = []

    for idx, (chunk_text, (start_char, end_char)) in enumerate(zip(raw_chunk_texts, offsets)):
        chunk_id = f"{doc.doc_id}#chunk_{idx:04d}"
        
        # Check if any section headers start inside the boundaries of this chunk
        headers_in_chunk = [span for span in section_spans if start_char <= span["start"] < end_char]
        if headers_in_chunk:
            # Use the last header starting inside the chunk as the active one
            active_span = headers_in_chunk[-1]
            hierarchy = []
            for span in section_spans:
                if span["start"] <= active_span["start"]:
                    level = span["level"]
                    while hierarchy and len(hierarchy) >= level:
                        hierarchy.pop()
                    hierarchy.append(span["title"])
                else:
                    break
            active_header = active_span["title"]
        else:
            active_header, hierarchy = find_active_section(start_char, section_spans)

        chunk_meta: Dict[str, Any] = {
            **doc.metadata,
            "chunk_index": idx,
            "total_chunks": total_chunks,
            "section_hierarchy": hierarchy,
            "doc_id": doc.doc_id,
            "doc_type": doc.doc_type,
            "file_name": doc.metadata.get("file_name", Path(doc.source).name),
            "char_count": len(chunk_text),
            "word_count": len(chunk_text.split()),
        }

        if doc.doc_type == "pdf" and "pages" in doc.metadata:
            chunk_meta["page_number"] = map_offset_to_page(start_char, doc.metadata["pages"])

        chunk = DocumentChunk(
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            text=chunk_text,
            start_char=start_char,
            end_char=end_char,
            chunk_index=idx,
            source=doc.source,
            section_header=active_header,
            metadata=chunk_meta,
        )
        chunks.append(chunk)

    return chunks


class IngestionPipeline:
    """Orchestrates document loading, text cleaning, chunking, and JSONL export."""

    def __init__(self, config: Optional[IngestionConfig] = None) -> None:
        self.config = config or default_config

    def process_document(self, doc: Document) -> List[DocumentChunk]:
        """Process a single document."""
        return chunk_document(doc, self.config)

    def run(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        recursive: bool = True,
    ) -> Dict[str, Any]:
        """Run the ingestion pipeline on a file or directory and write JSONL output.

        Args:
            input_path: Path to file or directory.
            output_path: Destination JSONL file path (optional).
            recursive: Whether to scan directory recursively.

        Returns:
            Dictionary containing pipeline execution metrics and summary.
        """
        start_time = time.time()
        in_p = Path(input_path).resolve()
        out_p = Path(output_path or self.config.output_file).resolve()

        if not in_p.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")

        # Load documents
        docs: List[Document] = []
        if in_p.is_file():
            docs.append(load_document(in_p, config=self.config))
        else:
            docs = load_directory(in_p, recursive=recursive, config=self.config)

        # Chunk all loaded documents
        all_chunks: List[DocumentChunk] = []
        for doc in docs:
            doc_chunks = self.process_document(doc)
            all_chunks.extend(doc_chunks)

        # Ensure destination folder exists
        out_p.parent.mkdir(parents=True, exist_ok=True)

        # Export chunks to JSONL
        with open(out_p, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        elapsed = time.time() - start_time
        total_chars = sum(c.char_count for c in all_chunks)
        total_words = sum(c.word_count for c in all_chunks)
        avg_chunk_size = total_chars / len(all_chunks) if all_chunks else 0

        summary = {
            "total_documents": len(docs),
            "total_chunks": len(all_chunks),
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_chunk_size": round(avg_chunk_size, 2),
            "output_file": str(out_p.as_posix()),
            "elapsed_seconds": round(elapsed, 4),
        }

        logger.info(
            "Ingestion completed: %d documents -> %d chunks in %.2fs. Saved to: %s",
            len(docs),
            len(all_chunks),
            elapsed,
            out_p,
        )
        return summary


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the command-line interface argument parser."""
    parser = argparse.ArgumentParser(
        description="Knowledge Assistant - Document Ingestion CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        dest="input_path",
        default="data/raw",
        help="Path to input document file or directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output_path",
        default="data/processed/chunks.jsonl",
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--chunk-size",
        "-c",
        type=int,
        default=500,
        help="Maximum characters per chunk",
    )
    parser.add_argument(
        "--chunk-overlap",
        "-v",
        type=int,
        default=50,
        help="Character overlap between consecutive chunks",
    )
    parser.add_argument(
        "--min-chunk-length",
        type=int,
        default=20,
        help="Minimum character threshold for a valid chunk",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively scan subdirectories",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Do not scan subdirectories",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print summary statistics table",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entrypoint for document ingestion."""
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = IngestionConfig.from_env(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_length=args.min_chunk_length,
        output_file=args.output_path,
    )

    pipeline = IngestionPipeline(config=config)

    try:
        summary = pipeline.run(
            input_path=args.input_path,
            output_path=args.output_path,
            recursive=args.recursive,
        )

        print("\n" + "=" * 55)
        print("  DOCUMENT INGESTION COMPLETED")
        print("=" * 55)
        print(f"  Total Documents Ingested : {summary['total_documents']}")
        print(f"  Total Chunks Generated   : {summary['total_chunks']}")
        print(f"  Total Characters         : {summary['total_characters']}")
        print(f"  Total Words              : {summary['total_words']}")
        print(f"  Average Chunk Size       : {summary['avg_chunk_size']} chars")
        print(f"  Output JSONL Path        : {summary['output_file']}")
        print(f"  Elapsed Processing Time  : {summary['elapsed_seconds']} s")
        print("=" * 55 + "\n")
        return 0

    except Exception as e:
        logger.error("Ingestion failed: %s", e, exc_info=True)
        print(f"\n[ERROR] Ingestion failed: {e}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
