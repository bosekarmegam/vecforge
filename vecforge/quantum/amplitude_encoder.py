# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under the Business Source License 1.1 (BSL 1.1)

"""
Quantum Amplitude Encoding for VecForge.

Maps classical relevance scores into a unit-norm "amplitude vector"
analogous to a quantum state |ψ⟩. Squaring the amplitudes yields a
probability distribution over candidates, which is the quantity Grover
amplification manipulates.

Mathematical basis:
    Given non-negative scores s ∈ ℝ^N, the amplitude encoding is:

        |ψ⟩ = s / ‖s‖₂   (L2 normalisation)

    so that Σ|ψ_i|² = 1 and p_i = ψ_i² is the probability of measuring
    candidate i.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class AmplitudeEncoder:
    """Encode classical score vectors into quantum amplitude space.

    Treats a vector of relevance scores as quantum probability amplitudes
    by L2-normalising it onto the unit hypersphere. Negative scores (for
    example raw cross-encoder logits) are shifted so the minimum is zero
    before normalising, because amplitudes with mixed signs would not map
    monotonically onto probabilities.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Example::

        >>> encoder = AmplitudeEncoder()
        >>> scores = np.array([0.9, 0.4, 0.1], dtype=np.float32)
        >>> amplitudes = encoder.encode(scores)
        >>> round(float(np.linalg.norm(amplitudes)), 5)
        1.0
    """

    def encode(self, scores: NDArray[np.float32]) -> NDArray[np.float32]:
        """Map scores into unit-norm amplitude space.

        Args:
            scores: 1-D array of relevance scores.

        Returns:
            Unit-norm float32 array of the same shape. If every score is
            identical (including all-zero), returns a uniform amplitude
            vector so that all candidates remain equally ranked rather
            than producing NaN values.

        Performance:
            Time: O(N) — single pass L2 normalisation.
        """
        arr = np.asarray(scores, dtype=np.float64).ravel()
        n = len(arr)
        if n == 0:
            return np.zeros(0, dtype=np.float32)

        finite = np.isfinite(arr)
        if not finite.all():
            arr = np.where(finite, arr, 0.0)

        min_val = float(arr.min())
        if min_val < 0.0:
            arr = arr - min_val

        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            logger.debug("AmplitudeEncoder: degenerate scores, returning uniform")
            return np.full(n, 1.0 / np.sqrt(n), dtype=np.float32)

        amplitudes: NDArray[np.float32] = (arr / norm).astype(np.float32)
        return amplitudes

    def decode(
        self, amplitudes: NDArray[np.float32], original_norm: float
    ) -> NDArray[np.float32]:
        """Recover the original scale from amplitude space.

        Args:
            amplitudes: Unit-norm amplitude vector from :meth:`encode`.
            original_norm: The L2 norm of the original score array.

        Returns:
            Scores rescaled to the original magnitude.
        """
        return (np.asarray(amplitudes, dtype=np.float32) * original_norm).astype(
            np.float32
        )

    @staticmethod
    def probabilities(amplitudes: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert amplitudes to measurement probabilities (``a_i²``).

        Args:
            amplitudes: Amplitude vector (any norm).

        Returns:
            float32 probability array that sums to 1 (or all zeros if the
            input is all zeros).

        Performance:
            Time: O(N)
        """
        a = np.asarray(amplitudes, dtype=np.float64)
        p = a * a
        total = float(p.sum())
        if total > 0.0:
            p = p / total
        return p.astype(np.float32)
