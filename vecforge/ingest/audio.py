# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
Audio embedding via OpenAI Whisper (local model, no API key needed).

Provides AudioEmbedder which:
  1. Transcribes audio → text using Whisper (local inference)
  2. Returns both the transcript and a text embedding for semantic search

This enables cross-modal search: query text → find relevant audio documents.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Supported audio formats
SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".webm"}

_whisper: object | None = None


def _ensure_whisper() -> None:
    global _whisper
    if _whisper is None:
        try:
            import whisper

            _whisper = whisper
        except ImportError as exc:
            raise ImportError(
                "AudioEmbedder requires the [multimodal] extra.\n"
                "Install with: pip install vecforge[multimodal]\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            ) from exc


class AudioEmbedder:
    """Transcribe audio and generate text embeddings for VecForge search.

    Uses OpenAI Whisper (local inference) to transcribe speech → text, then
    embeds the transcript using the same sentence-transformer model used for
    standard text, enabling seamless cross-modal search.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        whisper_model: Whisper model size. One of: ``tiny``, ``base``,
            ``small``, ``medium``, ``large``. Default ``base`` (~75MB).
        text_embedder: Optional external text embedder instance. If None,
            a new instance is created from `vecforge.core.embedder`.

    Performance:
        Transcription: ~1-10s per minute of audio (CPU, base model)
        Embedding:     ~20ms (same as text search)
        Memory:        ~150MB (Whisper base) + ~90MB (sentence-transformer)

    Example::

        >>> ae = AudioEmbedder()
        >>> transcript, vec = ae.transcribe_and_embed("speech.mp3")
        >>> print(transcript[:50])
        'The patient was admitted with severe chest pain...'
        >>> vec.shape
        (384,)
    """

    def __init__(
        self,
        whisper_model: str = "base",
        text_embedder: object | None = None,
    ) -> None:
        self._whisper_model_name = whisper_model
        self._whisper_model: object | None = None
        self._text_embedder = text_embedder

    def _load_whisper(self) -> None:
        """Lazy-load Whisper on first use."""
        if self._whisper_model is not None:
            return
        _ensure_whisper()
        import whisper

        self._whisper_model = whisper.load_model(self._whisper_model_name)
        logger.info("AudioEmbedder: loaded Whisper '%s'", self._whisper_model_name)

    def _get_text_embedder(self) -> object:
        if self._text_embedder is None:
            from vecforge.core.embedder import Embedder

            self._text_embedder = Embedder()
        return self._text_embedder

    def transcribe(self, audio_path: str | Path) -> str:
        """Transcribe an audio file to text using Whisper.

        Args:
            audio_path: Path to audio file. Supported: mp3, wav, flac,
                m4a, ogg, webm.

        Returns:
            Full transcript as a string.

        Raises:
            ImportError: If ``vecforge[multimodal]`` is not installed.
            FileNotFoundError: If the audio file does not exist.
            ValueError: If the file format is not supported.

        Performance:
            ~1–10x realtime on CPU (Whisper ``base`` model).

        Example::

            >>> text = AudioEmbedder().transcribe("lecture.mp3")
            >>> print(text[:60])
            'Welcome to this introduction to machine learning...'
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {path}\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            )
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format: '{path.suffix}'. "
                f"Supported: {sorted(SUPPORTED_FORMATS)}\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            )

        self._load_whisper()
        result = self._whisper_model.transcribe(str(path))  # type: ignore[union-attr]
        transcript: str = result["text"].strip()
        logger.info(
            "AudioEmbedder: transcribed '%s' → %d chars", path.name, len(transcript)
        )
        return transcript

    def embed(self, audio_path: str | Path) -> NDArray[np.float32]:
        """Transcribe audio and return a text embedding for the transcript.

        Args:
            audio_path: Path to audio file.

        Returns:
            float32 embedding array compatible with the VecForge text space.

        Performance:
            Transcription + embedding combined. See transcribe() for timing.

        Example::

            >>> vec = AudioEmbedder().embed("interview.wav")
            >>> vec.shape
            (384,)
        """
        transcript = self.transcribe(audio_path)
        embedder = self._get_text_embedder()
        vec: NDArray[np.float32] = embedder.embed(transcript)  # type: ignore[union-attr]
        return vec

    def transcribe_and_embed(
        self,
        audio_path: str | Path,
    ) -> tuple[str, NDArray[np.float32]]:
        """Transcribe audio and return both transcript and its embedding.

        Args:
            audio_path: Path to audio file.

        Returns:
            Tuple of (transcript: str, embedding: NDArray[float32]).

        Example::

            >>> transcript, vec = AudioEmbedder().transcribe_and_embed("x.mp3")
            >>> type(transcript)
            <class 'str'>
            >>> vec.dtype
            dtype('float32')
        """
        transcript = self.transcribe(audio_path)
        embedder = self._get_text_embedder()
        vec: NDArray[np.float32] = embedder.embed(transcript)  # type: ignore[union-attr]
        return transcript, vec
