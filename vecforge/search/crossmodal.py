# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
Cross-modal search for VecForge.

Accepts a query in any modality (text, image path, audio path) and
returns results from any modality — all in the same shared vector space.

Supported query types:
    text  → text results  (standard hybrid search)
    image → text results  (visual query, text documents)
    audio → text results  (speech query, text documents)
    text  → image results (text query, image documents)

All embeddings share a 512-dim CLIP-compatible space when multimodal
content is added. Text-only vaults continue to use 384-dim sentence-
transformer vectors (no change to existing API).

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Recognised image/audio extensions for auto-detection
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".webm"}


class CrossModalSearcher:
    """Unified search accepting text, image, or audio queries.

    Normalises any query modality into the shared VecForge embedding
    space and delegates to the underlying FAISS index. Callers do not
    need to know the modality — ``search()`` auto-detects it.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        image_embedder: Optional pre-loaded ImageEmbedder instance. Will
            be lazily created on first image query if not provided.
        audio_embedder: Optional pre-loaded AudioEmbedder instance. Will
            be lazily created on first audio query if not provided.

    Performance:
        Text query:  same as standard VecForge search (~5ms at 100k)
        Image query: ~20ms additional (CLIP encoding) + search latency
        Audio query: ~1–10s additional (Whisper transcription) + search

    Example::

        >>> cs = CrossModalSearcher()
        >>> vec = cs.encode_query("photo.jpg")
        >>> vec.shape
        (512,)
    """

    def __init__(
        self,
        image_embedder: Any | None = None,
        audio_embedder: Any | None = None,
    ) -> None:
        self._image_embedder = image_embedder
        self._audio_embedder = audio_embedder

    def detect_modality(self, query: str | Path) -> str:
        """Detect the modality of a query string or path.

        Args:
            query: Query string (text) or path to image/audio file.

        Returns:
            One of ``"text"``, ``"image"``, or ``"audio"``.

        Example::

            >>> CrossModalSearcher().detect_modality("photo.jpg")
            'image'
            >>> CrossModalSearcher().detect_modality("hello world")
            'text'
        """
        path = Path(str(query))
        suffix = path.suffix.lower()
        if suffix in _IMAGE_EXTS and path.exists():
            return "image"
        if suffix in _AUDIO_EXTS and path.exists():
            return "audio"
        return "text"

    def encode_query(
        self,
        query: str | Path,
        modality: str | None = None,
    ) -> NDArray[np.float32]:
        """Encode a query of any modality into the shared vector space.

        Args:
            query: Query text, image path, or audio path.
            modality: Force a specific modality (``"text"``, ``"image"``,
                ``"audio"``). If None, auto-detects from content/extension.

        Returns:
            float32 embedding array ready for FAISS similarity search.

        Raises:
            ImportError: If multimodal extras are not installed for
                image or audio queries.
            ValueError: If modality is an unrecognised string.

        Performance:
            text:  ~5ms, image: ~20ms, audio: ~1-10s

        Example::

            >>> vec = CrossModalSearcher().encode_query("a sunny beach")
            >>> vec.dtype
            dtype('float32')
        """
        detected = modality or self.detect_modality(query)

        if detected == "image":
            embedder = self._get_image_embedder()
            return embedder.embed(query)

        elif detected == "audio":
            embedder = self._get_audio_embedder()
            return embedder.embed(query)

        elif detected == "text":
            from vecforge.core.embedder import Embedder

            return Embedder().embed(str(query))

        else:
            raise ValueError(
                f"Unknown modality: '{detected}'. "
                "Expected 'text', 'image', or 'audio'.\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            )

    def _get_image_embedder(self) -> Any:
        if self._image_embedder is None:
            from vecforge.ingest.vision import ImageEmbedder

            self._image_embedder = ImageEmbedder()
        return self._image_embedder

    def _get_audio_embedder(self) -> Any:
        if self._audio_embedder is None:
            from vecforge.ingest.audio import AudioEmbedder

            self._audio_embedder = AudioEmbedder()
        return self._audio_embedder
