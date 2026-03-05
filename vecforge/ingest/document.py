# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
Document parser for VecForge.

Handles text extraction and chunking for common document formats:
PDF (via PyMuPDF), DOCX (via python-docx), HTML (via BeautifulSoup),
and plain text/markdown.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from vecforge.ingest.dispatcher import IngestChunk

logger = logging.getLogger(__name__)


class DocumentParser:
    """Multi-format document text extractor with chunking.

    Extracts raw text from supported formats and splits into
    overlapping chunks suitable for embedding.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        chunk_size: Maximum characters per chunk. Defaults to 1000.
        chunk_overlap: Overlap between consecutive chunks. Defaults to 200.

    Performance:
        Time: O(S) where S = total text size
        Chunking: O(S / chunk_size) chunks produced

    Example:
        >>> parser = DocumentParser(chunk_size=500, chunk_overlap=100)
        >>> chunks = parser.parse_text_file(Path("report.txt"))
        >>> len(chunks)
        15
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def _chunk_text(
        self,
        text: str,
        source: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[IngestChunk]:
        """Split text into overlapping chunks.

        Args:
            text: Raw text to chunk.
            source: Source file path for metadata.
            extra_metadata: Additional metadata to add to each chunk.

        Returns:
            List of IngestChunk with text and metadata.

        Performance:
            Time: O(S / chunk_size)
        """
        if not text.strip():
            return []

        chunks: list[IngestChunk] = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self._chunk_size

            # why: Try to break at sentence/paragraph boundary
            if end < len(text):
                # Look for paragraph break first, then sentence end
                for sep in ["\n\n", "\n", ". ", "! ", "? "]:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start:
                        end = last_sep + len(sep)
                        break

            chunk_text = text[start:end].strip()

            if chunk_text:
                meta = {
                    "source": source,
                    "chunk_index": chunk_index,
                    "char_start": start,
                    "char_end": end,
                }
                if extra_metadata:
                    meta.update(extra_metadata)

                chunks.append(IngestChunk(text=chunk_text, metadata=meta))
                chunk_index += 1

            # why: Move forward by chunk_size - overlap for continuity
            start = end - self._chunk_overlap
            if start <= chunks[-1].metadata["char_start"] if chunks else True:
                # safety: Prevent infinite loop
                start = end

        logger.debug("Chunked %s into %d chunks", source, len(chunks))
        return chunks

    def parse_text_file(self, path: Path) -> list[IngestChunk]:
        """Parse a plain text or markdown file.

        Args:
            path: Path to .txt or .md file.

        Returns:
            List of IngestChunk from the file.

        Performance:
            Time: O(S) where S = file size
        """
        text = path.read_text(encoding="utf-8", errors="replace")
        return self._chunk_text(text, source=str(path))

    def parse_pdf(self, path: Path) -> list[IngestChunk]:
        """Parse a PDF file using PyMuPDF (fitz).

        Args:
            path: Path to .pdf file.

        Returns:
            List of IngestChunk with page metadata.

        Performance:
            Time: O(P * S) where P = pages, S = avg page text size
        """
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF ingestion.\n"
                "Install with: pip install pymupdf\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            ) from e

        all_chunks: list[IngestChunk] = []

        with fitz.open(str(path)) as doc:
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    chunks = self._chunk_text(
                        text,
                        source=str(path),
                        extra_metadata={"page": page_num + 1},
                    )
                    all_chunks.extend(chunks)

        logger.info("Parsed PDF %s: %d chunks", path.name, len(all_chunks))
        return all_chunks

    def parse_docx(self, path: Path) -> list[IngestChunk]:
        """Parse a DOCX file using python-docx.

        Args:
            path: Path to .docx file.

        Returns:
            List of IngestChunk from the document.

        Performance:
            Time: O(P) where P = number of paragraphs
        """
        try:
            import docx
        except ImportError as e:
            raise ImportError(
                "python-docx is required for DOCX ingestion.\n"
                "Install with: pip install python-docx\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            ) from e

        doc = docx.Document(str(path))
        full_text = "\n\n".join(
            para.text for para in doc.paragraphs if para.text.strip()
        )

        chunks = self._chunk_text(full_text, source=str(path))
        logger.info("Parsed DOCX %s: %d chunks", path.name, len(chunks))
        return chunks

    def parse_html_file(self, path: Path) -> list[IngestChunk]:
        """Parse an HTML file using BeautifulSoup.

        Args:
            path: Path to .html or .htm file.

        Returns:
            List of IngestChunk from the HTML content.

        Performance:
            Time: O(S) where S = file size
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError as e:
            raise ImportError(
                "beautifulsoup4 is required for HTML ingestion.\n"
                "Install with: pip install beautifulsoup4\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            ) from e

        html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")

        # why: Remove script and style elements before extracting text
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        chunks = self._chunk_text(text, source=str(path))
        logger.info("Parsed HTML %s: %d chunks", path.name, len(chunks))
        return chunks
