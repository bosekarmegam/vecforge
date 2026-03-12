# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under the Business Source License 1.1 (BSL 1.1)

"""
Quantum Amplitude Encoding for VecForge.

Maps classical relevance scores into a unit-norm "amplitude vector"
analogous to a quantum state |ψ⟩. This encoding provides a more
numerically stable distance metric and is the first stage of the
quantum-inspired reranking pipeline.

Mathematical basis:
    Given scores s ∈ ℝ^N, the amplitude encoding is:

        |ψ⟩ = s / ‖s‖₂   (L2 normalisation)

    Inner products between amplitude vectors correspond to quantum
    fidelity (state overlap), making cosine distances meaningful even
    for sparse, heavily skewed score distributions.

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
    by L2-normalising it onto the unit hypersphere. This is mathematically
    equivalent to preparing a quantum state |ψ⟩ via amplitude encoding.

    The encoding preserves the *relative* ordering of scores while
    compressing the dynamic range, making subsequent Grover amplification
    more numerically stable.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Example::

        >>> encoder = AmplitudeEncoder()
        >>> scores = np.array([0.9, 0.4, 0.1], dtype=np.float32)
        >>> amplitudes = encoder.encode(scores)
        >>> float(np.linalg.norm(amplitudes))
        1.0
    """

    def encode(self, scores: NDArray[np.float32]) -> NDArray[np.float32]:
        """Map scores into unit-norm amplitude space.

        Args:
            scores: 1-D array of relevance scores (any non-negative values).

        Returns:
            Unit-norm array of the same shape. If the input norm is zero
            (all scores are zero), returns a uniform amplitude vector so
            that all candidates remain equally ranked rather than producing
            NaN values.

        Performance:
            Time: O(N) — single pass L2 normalisation.
        """
        scores = np.asarray(scores, dtype=np.float32)

        # Edge: if all scores are zero, return uniform distribution
        norm = float(np.linalg.norm(scores))
        if norm == 0.0:
            n = len(scores)
            logger.debug(
                "AmplitudeEncoder: all-zero scores, returning uniform amplitudes"
            )
            return np.full(n, 1.0 / np.sqrt(max(n, 1)), dtype=np.float32)

        amplitudes: NDArray[np.float32] = (scores / norm).astype(np.float32)
        logger.debug(
            "AmplitudeEncoder: encoded %d scores, norm=%.6f", len(scores), norm
        )
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
        return (amplitudes * original_norm).astype(np.float32)
