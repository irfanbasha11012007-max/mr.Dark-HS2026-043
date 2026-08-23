"""Configuration module for the Document Ingestion Pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional


@dataclass(frozen=True)
class IngestionConfig:
    """Immutable configuration container for document ingestion and evaluation."""

    chunk_size: int = 500
    chunk_overlap: int = 50
    min_chunk_length: int = 20
    clean_whitespace: bool = True
    normalize_unicode: bool = True
    extract_headers: bool = True
    supported_extensions: Tuple[str, ...] = field(
        default=(".txt", ".md", ".markdown", ".pdf")
    )
    separators: Tuple[str, ...] = field(
        default=(
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
        )
    )
    input_dir: str = "data/raw"
    output_file: str = "data/processed/chunks.jsonl"
    eval_questions_file: str = "evaluation/eval_questions.jsonl"
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})"
            )
        if self.min_chunk_length < 0:
            raise ValueError(
                f"min_chunk_length cannot be negative, got {self.min_chunk_length}"
            )

    @classmethod
    def from_env(cls, **overrides) -> "IngestionConfig":
        """Load configuration from environment variables with optional explicit overrides."""
        def _get_bool(key: str, default: bool) -> bool:
            val = os.getenv(key)
            if val is None:
                return default
            return val.strip().lower() in ("true", "1", "yes", "y", "t")

        def _get_int(key: str, default: int) -> int:
            val = os.getenv(key)
            if val is None:
                return default
            try:
                return int(val.strip())
            except ValueError:
                return default

        def _get_tuple_str(key: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
            val = os.getenv(key)
            if not val:
                return default
            items = [item.strip() for item in val.split(",") if item.strip()]
            return tuple(items) if items else default

        config_kwargs = {
            "chunk_size": _get_int("INGEST_CHUNK_SIZE", 500),
            "chunk_overlap": _get_int("INGEST_CHUNK_OVERLAP", 50),
            "min_chunk_length": _get_int("INGEST_MIN_CHUNK_LENGTH", 20),
            "clean_whitespace": _get_bool("INGEST_CLEAN_WHITESPACE", True),
            "normalize_unicode": _get_bool("INGEST_NORMALIZE_UNICODE", True),
            "extract_headers": _get_bool("INGEST_EXTRACT_HEADERS", True),
            "supported_extensions": _get_tuple_str(
                "INGEST_SUPPORTED_EXTENSIONS", (".txt", ".md", ".markdown", ".pdf")
            ),
            "input_dir": os.getenv("INGEST_INPUT_DIR", "data/raw"),
            "output_file": os.getenv("INGEST_OUTPUT_FILE", "data/processed/chunks.jsonl"),
            "eval_questions_file": os.getenv(
                "EVAL_QUESTIONS_FILE", "evaluation/eval_questions.jsonl"
            ),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
        }

        # Apply runtime overrides
        config_kwargs.update(overrides)
        return cls(**config_kwargs)


# Default module-level configuration instance
default_config = IngestionConfig()
