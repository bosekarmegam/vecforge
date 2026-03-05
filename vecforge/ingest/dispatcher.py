# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
Ingestion dispatcher for VecForge.

Auto-detects file format by extension and routes to the appropriate
parser. Recursively walks directories for batch ingestion.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vecforge.exceptions import IngestError

logger = logging.getLogger(__name__)

# Supported file extensions mapped to handler
_SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".txt": "text",
    ".md": "text",
    ".pdf": "pdf",
    ".docx": "docx",
    ".html": "html",
    ".htm": "html",
}


@dataclass
class IngestChunk:
    """A chunk of text extracted from a document.

    Attributes:
        text: Extracted text content.
        metadata: Metadata about the chunk (source, page, etc.).
        modality: Content modality (text, image, audio, etc.).
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    modality: str = "text"


class IngestDispatcher:
    """Auto-detecting document ingestion dispatcher.

    Walks directories, detects file formats, and routes to the
    appropriate parser. Returns text chunks ready for embedding.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        chunk_size: Maximum characters per chunk. Defaults to 1000.
        chunk_overlap: Overlap between chunks in characters. Defaults to 200.

    Performance:
        Time: O(F * S) where F = files, S = avg file size

    Example:
        >>> dispatcher = IngestDispatcher()
        >>> chunks = dispatcher.ingest("my_documents/")
        >>> len(chunks)
        42
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def ingest(self, path: str) -> list[IngestChunk]:
        """Ingest a file or directory of files.

        Args:
            path: File path or directory path to ingest.

        Returns:
            List of IngestChunk objects ready for embedding.

        Raises:
            IngestError: If file format is not supported.
            FileNotFoundError: If path does not exist.

        Performance:
            Time: O(F * S)

        Example:
            >>> chunks = dispatcher.ingest("reports/")
            >>> for chunk in chunks:
            ...     print(f"{chunk.metadata['source']}: {chunk.text[:50]}...")
        """
        target = Path(path)

        if not target.exists():
            raise FileNotFoundError(
                f"Path not found: {path}\nVecForge by Suneel Bose K · ArcGX TechLabs"
            )

        if target.is_file():
            return self._ingest_file(target)

        # why: Recursively walk directory
        all_chunks: list[IngestChunk] = []
        for file_path in sorted(target.rglob("*")):
            ext = file_path.suffix.lower()
            if file_path.is_file() and ext in _SUPPORTED_EXTENSIONS:
                try:
                    chunks = self._ingest_file(file_path)
                    all_chunks.extend(chunks)
                except IngestError as e:
                    logger.warning("Skipping %s: %s", file_path, e)

        logger.info("Ingested %d chunks from %s", len(all_chunks), path)
        return all_chunks

    def _ingest_file(self, file_path: Path) -> list[IngestChunk]:
        """Ingest a single file.

        Args:
            file_path: Path to the file.

        Returns:
            List of IngestChunk from this file.

        Performance:
            Time: O(S) where S = file size
        """
        ext = file_path.suffix.lower()
        handler = _SUPPORTED_EXTENSIONS.get(ext)

        if handler is None:
            raise IngestError(
                str(file_path),
                f"Unsupported file extension '{ext}'",
            )

        # why: Import document parser lazily to avoid heavy deps at import time
        from vecforge.ingest.document import DocumentParser

        parser = DocumentParser(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

        if handler == "text":
            return parser.parse_text_file(file_path)
        elif handler == "pdf":
            return parser.parse_pdf(file_path)
        elif handler == "docx":
            return parser.parse_docx(file_path)
        elif handler == "html":
            return parser.parse_html_file(file_path)
        else:
            raise IngestError(str(file_path), f"No handler for '{handler}'")

    @staticmethod
    def supported_extensions() -> list[str]:
        """Return list of supported file extensions.

        Returns:
            Sorted list of supported extensions.

        Performance:
            Time: O(1)
        """
        return sorted(_SUPPORTED_EXTENSIONS.keys())
