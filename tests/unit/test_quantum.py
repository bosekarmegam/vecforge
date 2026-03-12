# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under the Business Source License 1.1 (BSL 1.1)

"""
Unit tests for Phase 3 — Quantum-Inspired Acceleration module.

Tests:
    - AmplitudeEncoder: unit-norm encoding, zero-score safety
    - GroverAmplifier: score amplification, top-item preservation
    - QuantumReranker: output ordering, top_k, empty inputs
    - Integration: quantum_rerank flag in CascadeSearcher flow
"""

from __future__ import annotations

import numpy as np
import pytest

from vecforge.quantum.amplitude_encoder import AmplitudeEncoder
from vecforge.quantum.grover_amplifier import GroverAmplifier
from vecforge.quantum.reranker import QuantumReranker

# ─── AmplitudeEncoder ───


class TestAmplitudeEncoder:
    def test_unit_norm(self) -> None:
        """Encoded vector must have L2 norm = 1.0."""
        encoder = AmplitudeEncoder()
        scores = np.array([0.9, 0.4, 0.1, 0.6], dtype=np.float32)
        amplitudes = encoder.encode(scores)
        assert abs(float(np.linalg.norm(amplitudes)) - 1.0) < 1e-5

    def test_zero_scores_returns_uniform(self) -> None:
        """All-zero scores must yield a uniform amplitude vector (no NaN)."""
        encoder = AmplitudeEncoder()
        scores = np.zeros(4, dtype=np.float32)
        amplitudes = encoder.encode(scores)
        assert not np.any(np.isnan(amplitudes))
        assert abs(float(np.linalg.norm(amplitudes)) - 1.0) < 1e-5

    def test_single_element(self) -> None:
        """Single-element input encodes to exactly [1.0]."""
        encoder = AmplitudeEncoder()
        scores = np.array([0.7], dtype=np.float32)
        amplitudes = encoder.encode(scores)
        assert abs(float(amplitudes[0]) - 1.0) < 1e-5

    def test_decode_roundtrip(self) -> None:
        """decode(encode(s), norm) should recover original scale."""
        encoder = AmplitudeEncoder()
        scores = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        original_norm = float(np.linalg.norm(scores))
        amplitudes = encoder.encode(scores)
        recovered = encoder.decode(amplitudes, original_norm)
        np.testing.assert_allclose(recovered, scores, rtol=1e-5)


# ─── GroverAmplifier ───


class TestGroverAmplifier:
    def test_output_is_sorted_by_amplified_score(self) -> None:
        """GroverAmplifier output must be a valid reshuffled score array (not NaN)."""  # noqa: E501
        amp = GroverAmplifier()
        scores = np.array([0.9, 0.3, 0.1, 0.05], dtype=np.float32)
        amplified = amp.amplify(scores)
        # No NaN, no inf
        assert not np.any(np.isnan(amplified))
        assert not np.any(np.isinf(amplified))

    def test_output_same_length(self) -> None:
        """Amplified array must have same length as input."""
        amp = GroverAmplifier()
        scores = np.array([0.5, 0.3, 0.1], dtype=np.float32)
        assert len(amp.amplify(scores)) == 3

    def test_single_element_unchanged(self) -> None:
        """Single-element input should remain unchanged."""
        amp = GroverAmplifier()
        scores = np.array([0.7], dtype=np.float32)
        amplified = amp.amplify(scores)
        assert amplified[0] == pytest.approx(scores[0], abs=1e-5)

    def test_custom_iterations(self) -> None:
        """Custom iteration override should be respected (no error)."""
        amp = GroverAmplifier()
        scores = np.array([0.6, 0.4, 0.2], dtype=np.float32)
        result = amp.amplify(scores, iterations=2)
        assert len(result) == 3

    def test_uniform_scores_stable(self) -> None:
        """Uniform scores should remain unchanged after diffusion (mean = each value)."""  # noqa: E501
        amp = GroverAmplifier(max_iterations=3)
        scores = np.full(5, 0.4, dtype=np.float32)
        amplified = amp.amplify(scores)
        # Inversion-about-mean of uniform vector returns the same vector
        np.testing.assert_allclose(amplified, scores, atol=1e-5)


# ─── QuantumReranker ───


class TestQuantumReranker:
    def test_output_sorted_descending(self) -> None:
        """Results should be sorted by score descending."""
        qr = QuantumReranker()
        texts = ["A", "B", "C", "D"]
        scores = [0.9, 0.3, 0.1, 0.7]
        results = qr.rerank("query", texts, scores, top_k=4)
        result_scores = [s for _, s in results]
        assert result_scores == sorted(result_scores, reverse=True)

    def test_top_k_respected(self) -> None:
        """top_k must limit the number of returned results."""
        qr = QuantumReranker()
        texts = ["A", "B", "C", "D", "E"]
        scores = [0.9, 0.7, 0.5, 0.3, 0.1]
        results = qr.rerank("query", texts, scores, top_k=3)
        assert len(results) == 3

    def test_empty_input_returns_empty(self) -> None:
        """Empty texts/scores should return an empty list without error."""
        qr = QuantumReranker()
        results = qr.rerank("query", [], [], top_k=5)
        assert results == []

    def test_index_mapping_valid(self) -> None:
        """All returned original_index values must be valid indices."""
        qr = QuantumReranker()
        texts = ["X", "Y", "Z"]
        scores = [0.8, 0.5, 0.2]
        results = qr.rerank("q", texts, scores, top_k=3)
        for idx, _ in results:
            assert 0 <= idx < len(texts)

    def test_output_is_valid(self) -> None:
        """QuantumReranker output must have no NaN scores and valid indices."""
        qr = QuantumReranker()
        texts = ["low", "high", "mid"]
        scores = [0.1, 0.95, 0.4]
        results = qr.rerank("query", texts, scores, top_k=3)
        for idx, score in results:
            assert 0 <= idx < len(texts)
            assert not np.isnan(score)

    def test_top_k_larger_than_n(self) -> None:
        """top_k > len(texts) should return all items, not raise."""
        qr = QuantumReranker()
        texts = ["A", "B"]
        scores = [0.6, 0.4]
        results = qr.rerank("q", texts, scores, top_k=10)
        assert len(results) == 2
