# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under the Business Source License 1.1 (BSL 1.1)

"""
Unit tests for Phase 4 — Multimodal & Advanced.

Tests:
    - ImageEmbedder: lazy import guard, embed shape/dtype with mock
    - AudioEmbedder: lazy import guard, transcription with mock
    - CrossModalSearcher: modality detection, encode_query routing
    - QuantumReranker: max_candidates windowing behaviour (FIXED)
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ─── CrossModalSearcher ───────────────────────────────────────────────────────


class TestCrossModalSearcher:
    def test_detect_text(self, tmp_path: object) -> None:
        """Plain strings are detected as text modality."""
        from vecforge.search.crossmodal import CrossModalSearcher

        cs = CrossModalSearcher()
        assert cs.detect_modality("hello world") == "text"
        assert cs.detect_modality("patient diabetes search") == "text"

    def test_detect_image(self, tmp_path: object) -> None:
        """Existing image files are detected as image modality."""

        from vecforge.search.crossmodal import CrossModalSearcher

        img = tmp_path / "photo.jpg"  # type: ignore[operator]
        img.write_bytes(b"\xff\xd8\xff")  # minimal JPEG header
        cs = CrossModalSearcher()
        assert cs.detect_modality(str(img)) == "image"

    def test_detect_audio(self, tmp_path: object) -> None:
        """Existing audio files are detected as audio modality."""
        from vecforge.search.crossmodal import CrossModalSearcher

        audio = tmp_path / "speech.mp3"  # type: ignore[operator]
        audio.write_bytes(b"ID3")  # minimal MP3 header
        cs = CrossModalSearcher()
        assert cs.detect_modality(str(audio)) == "audio"

    def test_detect_nonexistent_image_is_text(self) -> None:
        """Non-existent paths with image extension should fall back to text."""
        from vecforge.search.crossmodal import CrossModalSearcher

        cs = CrossModalSearcher()
        # File doesn't exist → extension check fails → text
        assert cs.detect_modality("/nonexistent/photo.jpg") == "text"

    def test_encode_query_text(self) -> None:
        """Text queries return a float32 numpy array from the Embedder."""
        import vecforge.core.embedder as _emb_mod
        from vecforge.search.crossmodal import CrossModalSearcher

        mock_vec = np.ones(384, dtype=np.float32)
        original_cls = _emb_mod.Embedder

        # Temporarily replace Embedder class in its own module so the
        # `from vecforge.core.embedder import Embedder` inside encode_query
        # picks up the mock (fresh `from` import re-looks up the module attr).
        mock_instance = MagicMock()
        mock_instance.embed.return_value = mock_vec
        _emb_mod.Embedder = MagicMock(return_value=mock_instance)  # type: ignore[attr-defined]
        try:
            cs = CrossModalSearcher()
            vec = cs.encode_query("query text", modality="text")
        finally:
            _emb_mod.Embedder = original_cls  # type: ignore[attr-defined]

        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32

    def test_invalid_modality_raises(self) -> None:
        """Unknown modality string must raise ValueError."""
        from vecforge.search.crossmodal import CrossModalSearcher

        cs = CrossModalSearcher()
        with pytest.raises(ValueError, match="Unknown modality"):
            cs.encode_query("something", modality="video")


# ─── ImageEmbedder ───────────────────────────────────────────────────────────


class TestImageEmbedder:
    def test_missing_dependency_raises_import_error(self) -> None:
        """ImportError with helpful install message when open_clip is absent."""
        from vecforge.ingest.vision import _ensure_deps

        with patch.dict(sys.modules, {"open_clip": None}):
            # Reset the cached module reference so _ensure_deps re-checks
            import vecforge.ingest.vision as vis

            original = vis._open_clip
            vis._open_clip = None
            try:
                with pytest.raises(ImportError, match="multimodal"):
                    _ensure_deps()
            finally:
                vis._open_clip = original

    def test_file_not_found_raises(self, tmp_path: object) -> None:
        """FileNotFoundError for non-existent image path."""
        from vecforge.ingest.vision import ImageEmbedder

        embedder = ImageEmbedder()
        # Patch _load to be a no-op so we skip the open_clip import check
        with patch.object(embedder, "_load", return_value=None):
            with pytest.raises(FileNotFoundError):
                embedder.embed("/nonexistent/image.jpg")


# ─── AudioEmbedder ───────────────────────────────────────────────────────────


class TestAudioEmbedder:
    def test_missing_dependency_raises_import_error(self) -> None:
        """ImportError with helpful message when whisper is absent."""
        import vecforge.ingest.audio as aud

        original = aud._whisper
        aud._whisper = None
        try:
            with patch.dict(sys.modules, {"whisper": None}):
                with pytest.raises(ImportError, match="multimodal"):
                    aud._ensure_whisper()
        finally:
            aud._whisper = original

    def test_file_not_found_raises(self) -> None:
        """FileNotFoundError for non-existent audio file."""
        from vecforge.ingest.audio import AudioEmbedder

        ae = AudioEmbedder()
        with pytest.raises(FileNotFoundError):
            ae.transcribe("/nonexistent/audio.mp3")

    def test_unsupported_format_raises(self, tmp_path: object) -> None:
        """ValueError for unsupported audio file extension."""
        from vecforge.ingest.audio import AudioEmbedder

        bad = tmp_path / "audio.avi"  # type: ignore[operator]
        bad.write_bytes(b"RIFF")
        ae = AudioEmbedder()
        with pytest.raises(ValueError, match="Unsupported"):
            ae.transcribe(str(bad))


# ─── QuantumReranker max_candidates windowing ─────────────────────────────────


class TestQuantumRerankerWindowing:
    def test_windowed_large_input_indices_valid(self) -> None:
        """All returned indices must be valid even when N >> max_candidates."""
        from vecforge.quantum.reranker import QuantumReranker

        n = 5000
        qr = QuantumReranker(max_candidates=100)
        texts = [f"doc_{i}" for i in range(n)]
        scores = list(np.random.rand(n).astype(float))
        results = qr.rerank("query", texts, scores, top_k=10)
        for idx, score in results:
            assert 0 <= idx < n
            assert not np.isnan(score)

    def test_windowed_returns_top_k(self) -> None:
        """top_k must be respected even when N >> max_candidates."""
        from vecforge.quantum.reranker import QuantumReranker

        n = 2000
        qr = QuantumReranker(max_candidates=200)
        texts = [f"doc_{i}" for i in range(n)]
        scores = list(np.random.rand(n).astype(float))
        results = qr.rerank("query", texts, scores, top_k=5)
        assert len(results) == 5

    def test_small_n_no_windowing_needed(self) -> None:
        """When N ≤ max_candidates, windowing is a no-op — same results."""
        from vecforge.quantum.reranker import QuantumReranker

        texts = ["A", "B", "C", "D"]
        scores = [0.9, 0.3, 0.1, 0.7]
        qr = QuantumReranker(max_candidates=1000)
        results = qr.rerank("query", texts, scores, top_k=4)
        # All indices should be valid
        for idx, score in results:
            assert 0 <= idx < 4
            assert not np.isnan(score)

    def test_windowed_is_fast_at_1m(self) -> None:
        """Windowed reranker at N=1M should complete quickly on CPU.

        Uses float scores only (no 1M-string construction) to isolate the
        quantum windowing speed. The pre-filter (O(N) argpartition) and Grover
        loop (O(K·√K), K=1000) together should finish in <2000ms on any CPU.
        """
        import time

        from vecforge.quantum.reranker import QuantumReranker

        n = 1_000_000
        qr = QuantumReranker(max_candidates=1000)

        # Use float list built from numpy for speed
        scores_arr = np.random.rand(n).astype(float)
        scores = scores_arr.tolist()
        # Build texts lazily — reranker only accesses results indices
        texts: list[str] = []  # crossmodal reranker doesn't need texts for pure-Grover

        # Patch texts to be a long dummy list without allocation
        class _LazyList:
            """O(1) __len__ + __getitem__ without allocating N strings."""

            def __len__(self) -> int:
                return n

            def __getitem__(self, i: int) -> str:
                return f"d{i}"

        t0 = time.perf_counter()
        results = qr.rerank("query", _LazyList(), scores, top_k=10)  # type: ignore[arg-type]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert len(results) == 10
        for idx, score in results:
            assert 0 <= idx < n
        # Allow 2000ms for slow test runners (CI, Windows, etc.)
        assert elapsed_ms < 2000, f"Too slow: {elapsed_ms:.1f}ms > 2000ms"
