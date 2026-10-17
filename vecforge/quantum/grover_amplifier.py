# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under the Business Source License 1.1 (BSL 1.1)

"""
Grover-Inspired Amplitude Amplifier for VecForge.

Implements a faithful classical simulation of Grover's amplitude
amplification on a real-valued amplitude vector.

Grover's iterate is the product of two reflections:

    1. **Oracle**  ``O = I − 2·P_M``  — flips the sign of every amplitude
       in the *marked* set ``M`` (the candidates we want to boost).
    2. **Diffusion** ``D = 2|u⟩⟨u| − I`` — inversion about the mean,
       ``a'_i = 2·mean(a) − a_i``.

Applied together, ``G = D·O`` rotates the state toward the marked
subspace by ``2θ`` per step, where ``sin θ = √(M/N)``. After
``k ≈ (π/4)·√(N/M)`` steps the marked candidates hold almost all of the
probability mass.

VecForge 1.x applied the diffusion step **without** an oracle. Because
``D`` is a reflection that preserves the mean, two applications are the
identity and one application reverses the ranking — so the 1.x reranker
either did nothing (even ``k``) or returned results worst-first (odd ``k``;
for example any candidate set of size 2–6, 15–26 or 41–58). VecForge 2.0
fixes the algorithm and, because the Grover rotation acts on a known
two-dimensional subspace, evaluates it in **closed form in O(N)** instead
of iterating ``O(√N)`` times over ``N`` elements.

Within the marked set the relative order of amplitudes is preserved
exactly (each marked amplitude is shifted by the same constant), so the
amplifier never scrambles a ranking — it only moves probability mass from
unmarked to marked candidates.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

Method = Literal["closed_form", "iterative"]


class GroverAmplifier:
    """Grover amplitude amplification on classical hardware.

    Boosts the amplitudes of a *marked* subset of candidates and
    suppresses the rest, exactly as Grover's algorithm would on a quantum
    register prepared in the given amplitude state.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        max_iterations: Fixed number of Grover iterations. Defaults to
            None, meaning the optimal count is computed from the initial
            state (``k = round((π/2 − φ₀) / 2θ)``).
        method: ``"closed_form"`` (default, O(N)) evaluates the rotation
            analytically; ``"iterative"`` applies oracle + diffusion
            ``k`` times (O(k·N)) and exists for validation and teaching.

    Performance:
        closed_form: O(N) — one pass to decompose, one pass to rebuild
        iterative:   O(k·N) with k = O(√(N/M))
        Practical:   <1ms for N ≤ 100,000 (closed form) on any modern CPU

    Example::

        >>> amp = GroverAmplifier()
        >>> a = np.array([0.3, 0.1, 0.9, 0.05], dtype=np.float32)
        >>> a = a / np.linalg.norm(a)
        >>> out = amp.amplify(a, marked=np.array([False, False, True, False]))
        >>> int(np.argmax(out ** 2))
        2
        >>> float(out[2] ** 2) > float(a[2] ** 2)
        True
    """

    def __init__(
        self,
        max_iterations: int | None = None,
        method: Method = "closed_form",
    ) -> None:
        if method not in ("closed_form", "iterative"):
            raise ValueError(
                f"Unknown method '{method}'. Expected 'closed_form' or 'iterative'.\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            )
        self._max_iterations = max_iterations
        self._method: Method = method

    # ─── Iteration count ───

    @staticmethod
    def _optimal_iterations(
        n: int,
        m: int = 1,
        initial_angle: float | None = None,
    ) -> int:
        """Compute the Grover-optimal number of iterations.

        For a uniform start the classic result is
        ``k = floor(π / (4·asin(√(M/N))))``. For a general start at angle
        ``φ₀`` from the unmarked axis the state reaches the marked axis
        after ``(π/2 − φ₀) / (2θ)`` rotations, which is rounded to the
        nearest integer.

        Args:
            n: Number of candidates.
            m: Number of marked candidates (1 ≤ m < n).
            initial_angle: ``φ₀ = atan2(α₀, β₀)`` of the initial state;
                None assumes a uniform superposition.

        Returns:
            Non-negative iteration count (0 means the state is already
            concentrated on the marked set).
        """
        if n <= 1 or m <= 0 or m >= n:
            return 0
        theta = math.asin(math.sqrt(m / n))
        phi0 = theta if initial_angle is None else float(initial_angle)
        k = int(round((math.pi / 2.0 - phi0) / (2.0 * theta)))
        return max(0, k)

    # ─── Core ───

    @staticmethod
    def default_marked(
        amplitudes: NDArray[np.float32], m: int | None = None
    ) -> NDArray[np.bool_]:
        """Mark the top-``m`` candidates by amplitude (default oracle).

        Args:
            amplitudes: 1-D amplitude array.
            m: Number of candidates to mark. Defaults to ``ceil(√N)``.

        Returns:
            Boolean mask of marked positions.

        Performance:
            Time: O(N) via ``np.argpartition``.
        """
        n = len(amplitudes)
        mask = np.zeros(n, dtype=np.bool_)
        if n <= 1:
            return mask
        if m is None:
            m = int(math.ceil(math.sqrt(n)))
        m = max(1, min(int(m), n - 1))
        top = np.argpartition(amplitudes, -m)[-m:]
        mask[top] = True
        return mask

    def amplify(
        self,
        scores: NDArray[np.float32],
        marked: NDArray[np.bool_] | None = None,
        iterations: int | None = None,
    ) -> NDArray[np.float32]:
        """Amplify the marked candidates' amplitudes.

        Args:
            scores: 1-D array of amplitudes. Any non-negative scores are
                accepted and L2-normalised internally; use
                :class:`~vecforge.quantum.AmplitudeEncoder` for an explicit
                encoding step.
            marked: Boolean mask selecting the oracle's marked set. None
                marks the top ``ceil(√N)`` amplitudes.
            iterations: Override the number of Grover iterations.

        Returns:
            Unit-norm float32 amplitude array of the same length. Square it
            to obtain probabilities. Marked amplitudes are all shifted by the
            same non-negative constant, so their relative order is
            preserved; unmarked mass is reduced.

        Performance:
            Time: O(N) closed form, O(k·N) iterative

        Example::

            >>> amp = GroverAmplifier()
            >>> a = np.array([0.5, 0.3, 0.1], dtype=np.float32)
            >>> out = amp.amplify(a, marked=np.array([True, False, False]))
            >>> int(np.argmax(out))
            0
        """
        a = np.asarray(scores, dtype=np.float64).copy()
        n = len(a)
        if n <= 1:
            return a.astype(np.float32)

        norm = float(np.linalg.norm(a))
        a = np.full(n, 1.0 / math.sqrt(n)) if norm == 0.0 else a / norm

        if marked is None:
            mask = self.default_marked(a.astype(np.float32))
        else:
            mask = np.asarray(marked, dtype=np.bool_)
            if mask.shape != (n,):
                raise ValueError(
                    f"marked mask shape {mask.shape} does not match {n} scores.\n"
                    "VecForge by Suneel Bose K · ArcGX TechLabs"
                )

        m = int(mask.sum())
        if m == 0 or m == n:
            logger.debug("GroverAmplifier: nothing to amplify (m=%d of %d)", m, n)
            return a.astype(np.float32)

        a_m = a[mask]
        a_u = a[~mask]
        alpha0 = float(a_m.sum()) / math.sqrt(m)
        beta0 = float(a_u.sum()) / math.sqrt(n - m)
        theta = math.asin(math.sqrt(m / n))
        phi0 = math.atan2(alpha0, beta0)

        if iterations is not None:
            k = max(0, int(iterations))
        elif self._max_iterations is not None:
            k = max(0, int(self._max_iterations))
        else:
            k = self._optimal_iterations(n, m, initial_angle=phi0)

        logger.debug(
            "GroverAmplifier: N=%d, M=%d, theta=%.4f, phi0=%.4f, k=%d, method=%s",
            n,
            m,
            theta,
            phi0,
            k,
            self._method,
        )

        if self._method == "iterative":
            out = self._amplify_iterative(a, mask, k)
        else:
            out = self._amplify_closed_form(a, mask, alpha0, beta0, theta, phi0, k)
        return out.astype(np.float32)

    @staticmethod
    def _amplify_iterative(
        a: NDArray[np.float64], mask: NDArray[np.bool_], k: int
    ) -> NDArray[np.float64]:
        """Reference implementation: apply oracle + diffusion ``k`` times."""
        out = a.copy()
        for _ in range(k):
            out[mask] *= -1.0  # oracle: phase flip on marked
            out = 2.0 * out.mean() - out  # diffusion: inversion about mean
        return out

    @staticmethod
    def _amplify_closed_form(
        a: NDArray[np.float64],
        mask: NDArray[np.bool_],
        alpha0: float,
        beta0: float,
        theta: float,
        phi0: float,
        k: int,
    ) -> NDArray[np.float64]:
        """Evaluate ``G^k`` analytically.

        The state decomposes as ``α|u_M⟩ + β|u_U⟩ + r_M + r_U`` where
        ``u_M``/``u_U`` are the uniform vectors over the marked/unmarked
        sets and ``r_M``/``r_U`` are the zero-mean residuals within each
        set. ``G`` rotates ``(β, α)`` by ``2θ``, leaves ``r_M`` untouched
        and negates ``r_U``.
        """
        n = len(a)
        m = int(mask.sum())
        radius = math.hypot(alpha0, beta0)
        phi_k = phi0 + 2.0 * k * theta
        alpha_k = radius * math.sin(phi_k)
        beta_k = radius * math.cos(phi_k)

        out = np.empty(n, dtype=np.float64)
        a_m = a[mask]
        a_u = a[~mask]
        out[mask] = alpha_k / math.sqrt(m) + (a_m - a_m.mean())
        sign = -1.0 if k % 2 else 1.0
        out[~mask] = beta_k / math.sqrt(n - m) + sign * (a_u - a_u.mean())
        return out

    def marked_probability(
        self,
        amplitudes: NDArray[np.float32],
        marked: NDArray[np.bool_],
    ) -> float:
        """Return the probability mass currently on the marked set.

        Args:
            amplitudes: Unit-norm amplitude array.
            marked: Boolean mask of the marked set.

        Returns:
            ``Σ_{i∈M} a_i²`` in ``[0, 1]``.

        Performance:
            Time: O(N)
        """
        a = np.asarray(amplitudes, dtype=np.float64)
        total = float(np.dot(a, a)) or 1.0
        return float(np.dot(a[marked], a[marked]) / total)
