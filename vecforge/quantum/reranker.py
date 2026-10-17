# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
Quantum-Inspired Reranker for VecForge.

Reranks search candidates with Grover amplitude amplification. An
*oracle* marks the candidates believed to be relevant (by default the
top ``√K`` by score; VecForge's search pipeline passes a *consensus*
oracle — candidates that both the dense and the sparse retriever agreed
on). Amplification then concentrates probability mass on the marked set
while preserving the relative order inside it, which lifts oracle-marked
candidates above unmarked ones and sharpens the score gap.

Pipeline:
    1. Pre-filter to the top ``max_candidates`` by raw score (O(N)
       ``np.argpartition``), so the rest of the pipeline is bounded.
    2. Amplitude-encode the window's scores (L2 normalise).
    3. Oracle: build the marked mask (consensus mask or top-√K).
    4. Grover amplification in closed form — O(K), no iteration loop.
    5. Probabilities ``p_i = a_i²`` become the new scores. Marked
       candidates keep their internal order; unmarked candidates keep
       theirs and share the residual probability mass.
    6. Optionally run a classical cross-encoder on the marked survivors
       only (O(M) model calls instead of O(N)).

Complexity:
    Pre-filter: O(N)     — one argpartition over all candidates
    Grover:     O(K)     — closed form, K = min(N, max_candidates)
    Total:      O(N + K) ≈ O(N)

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vecforge.quantum.amplitude_encoder import AmplitudeEncoder
from vecforge.quantum.grover_amplifier import GroverAmplifier

logger = logging.getLogger(__name__)

# Default candidate window — Grover never processes more than this many
# elements, bounding the amplification stage regardless of corpus size.
_DEFAULT_MAX_CANDIDATES = 1_000


class QuantumReranker:
    """Quantum-inspired reranker using Grover amplitude amplification.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        classical_reranker: Optional classical ``Reranker`` instance. When
            provided, it is applied only to the oracle-marked survivors
            after amplification rather than to every candidate.
        grover_iterations: Override for the number of Grover iterations.
            None uses the optimal count computed from the initial state.
        max_candidates: Maximum candidate window. Larger inputs are
            pre-filtered to the top ``max_candidates`` by raw score before
            amplification. Default: 1000.
        marked_fraction: When no oracle mask is supplied, mark the top
            ``max(top_k, ceil(K ** marked_fraction))`` candidates. The
            default 0.5 marks ``√K``.

    Performance:
        Pre-filter (O(N)):  ~3ms at 1M candidates (numpy argpartition)
        Grover (O(K)):      <0.1ms at K=1000
        Total at 1M docs:   <5ms

    Example::

        >>> qr = QuantumReranker()
        >>> scores = [0.9, 0.3, 0.1, 0.7]
        >>> texts = ["A", "B", "C", "D"]
        >>> results = qr.rerank("query", texts, scores, top_k=2)
        >>> [idx for idx, _ in results]
        [0, 3]
    """

    def __init__(
        self,
        classical_reranker: Any | None = None,
        grover_iterations: int | None = None,
        max_candidates: int = _DEFAULT_MAX_CANDIDATES,
        marked_fraction: float = 0.5,
    ) -> None:
        self._classical_reranker = classical_reranker
        self._encoder = AmplitudeEncoder()
        self._amplifier = GroverAmplifier(max_iterations=grover_iterations)
        self._max_candidates = max(1, int(max_candidates))
        self._marked_fraction = float(min(max(marked_fraction, 0.0), 1.0))
        self.last_explanation: dict[str, Any] = {}

    def amplify_scores(
        self,
        scores: Sequence[float] | NDArray[np.float32],
        marked: Iterable[int] | NDArray[np.bool_] | None = None,
        top_k: int = 10,
    ) -> tuple[NDArray[np.intp], NDArray[np.float32], NDArray[np.bool_]]:
        """Run the windowed amplification and return per-candidate results.

        Args:
            scores: Raw relevance scores, one per candidate.
            marked: Oracle — either a boolean mask over all candidates or
                an iterable of candidate indices to mark. None uses the
                default top-√K oracle.
            top_k: Used to size the default oracle (never fewer than
                ``top_k`` marked candidates).

        Returns:
            Tuple ``(window_indices, window_scores, window_marked)`` where
            ``window_indices`` are original candidate indices sorted by
            the new score descending, ``window_scores`` the amplified
            scores (max-normalised to ``[0, 1]``) in that order, and
            ``window_marked`` whether each was oracle-marked.

        Performance:
            Time: O(N + K)
        """
        score_array = np.asarray(scores, dtype=np.float32).ravel()
        n = len(score_array)
        empty = (
            np.zeros(0, dtype=np.intp),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.bool_),
        )
        if n == 0:
            return empty

        # ─── Stage 1: window ───
        k_window = min(n, self._max_candidates)
        if k_window < n:
            window_indices: NDArray[np.intp] = np.argpartition(score_array, -k_window)[
                -k_window:
            ].astype(np.intp)
        else:
            window_indices = np.arange(n, dtype=np.intp)
        window_scores = score_array[window_indices]

        # ─── Stage 2: amplitudes ───
        amplitudes = self._encoder.encode(window_scores)

        # ─── Stage 3: oracle ───
        if marked is None:
            m_default = max(top_k, int(math.ceil(k_window**self._marked_fraction)))
            mask = GroverAmplifier.default_marked(amplitudes, m=m_default)
        else:
            full_mask = self._as_mask(marked, n)
            mask = full_mask[window_indices]
            if mask.sum() == 0 or mask.sum() == k_window:
                # Oracle gives no signal inside the window — fall back
                m_default = max(top_k, int(math.ceil(k_window**self._marked_fraction)))
                mask = GroverAmplifier.default_marked(amplitudes, m=m_default)

        # ─── Stage 4: amplification (closed form) ───
        amplified = self._amplifier.amplify(amplitudes, marked=mask)
        probs = AmplitudeEncoder.probabilities(amplified)

        # ─── Stage 5: scores ───
        # Marked candidates: p_i = (a_i + c)² keeps their original order.
        # Unmarked candidates: distribute the residual mass in proportion to
        # the original probabilities so their order is preserved too.
        new_scores = np.zeros(k_window, dtype=np.float64)
        new_scores[mask] = probs[mask]
        unmarked = ~mask
        if unmarked.any():
            residual = float(probs[unmarked].sum())
            orig_p = amplitudes[unmarked].astype(np.float64) ** 2
            total = float(orig_p.sum())
            if total > 0.0:
                new_scores[unmarked] = residual * orig_p / total
            else:
                new_scores[unmarked] = residual / float(unmarked.sum())

        max_score = float(new_scores.max())
        if max_score > 0.0:
            new_scores = new_scores / max_score

        # why: ties keep the caller's input order (window_indices may come from
        # argpartition, whose order is arbitrary)
        order = np.lexsort(
            (window_indices, -window_scores.astype(np.float64), -new_scores)
        )
        self.last_explanation = {
            "candidates": int(n),
            "window": int(k_window),
            "marked": int(mask.sum()),
            "marked_probability_before": float(
                self._amplifier.marked_probability(amplitudes, mask)
            ),
            "marked_probability_after": float(
                self._amplifier.marked_probability(amplified, mask)
            ),
        }
        return (
            window_indices[order],
            new_scores[order].astype(np.float32),
            mask[order],
        )

    def rerank(
        self,
        query: str,
        texts: Sequence[str],
        scores: Sequence[float] | NDArray[np.float32],
        top_k: int = 10,
        marked: Iterable[int] | NDArray[np.bool_] | None = None,
    ) -> list[tuple[int, float]]:
        """Rerank candidates using Grover amplitude amplification.

        Args:
            query: Original search query string (used only by the optional
                cross-encoder pass).
            texts: Candidate document texts matching ``scores``.
            scores: Current relevance scores from the hybrid fusion cascade.
            top_k: Number of final results to return.
            marked: Optional oracle (boolean mask or iterable of indices)
                selecting the candidates to amplify. None uses the top-√K
                by score.

        Returns:
            List of ``(original_index, score)`` tuples sorted by descending
            relevance. Scores are in ``[0, 1]`` with the best candidate at
            1.0 (unless a cross-encoder is attached, in which case they are
            cross-encoder scores).

        Performance:
            Time: O(N + K) plus O(M) cross-encoder calls when attached

        Example::

            >>> qr = QuantumReranker()
            >>> res = qr.rerank(
            ...     "space", ["NASA", "pizza", "fox"], [0.9, 0.2, 0.05], top_k=2
            ... )
            >>> res[0][0]
            0
        """
        n = len(scores)
        if n == 0:
            return []
        top_k = max(0, min(top_k, n))
        if top_k == 0:
            return []

        ordered_idx, ordered_scores, ordered_marked = self.amplify_scores(
            scores, marked=marked, top_k=top_k
        )

        if self._classical_reranker is not None and len(ordered_idx) > 0:
            # Cross-encoder only on the marked survivors (at least top_k of them)
            survivors = [
                int(i)
                for i, flag in zip(ordered_idx, ordered_marked, strict=True)
                if flag
            ]
            if len(survivors) < top_k:
                extra = [int(i) for i in ordered_idx if int(i) not in set(survivors)]
                survivors.extend(extra[: top_k - len(survivors)])
            survivor_texts = [texts[i] for i in survivors]
            cross_ranked = self._classical_reranker.rerank(
                query, survivor_texts, top_k=top_k
            )
            results = [
                (survivors[local_idx], float(score))
                for local_idx, score in cross_ranked
            ]
            logger.debug(
                "QuantumReranker: cross-encoder ran on %d/%d candidates",
                len(survivor_texts),
                n,
            )
            return results[:top_k]

        return [
            (int(i), float(s))
            for i, s in zip(ordered_idx[:top_k], ordered_scores[:top_k], strict=True)
        ]

    @staticmethod
    def _as_mask(
        marked: Iterable[int] | NDArray[np.bool_], n: int
    ) -> NDArray[np.bool_]:
        arr = np.asarray(list(marked) if not isinstance(marked, np.ndarray) else marked)
        if arr.dtype == np.bool_:
            if arr.shape != (n,):
                raise ValueError(
                    f"marked mask shape {arr.shape} does not match {n} candidates.\n"
                    "VecForge by Suneel Bose K · ArcGX TechLabs"
                )
            return arr
        mask = np.zeros(n, dtype=np.bool_)
        if arr.size:
            idx = arr.astype(np.int64)
            idx = idx[(idx >= 0) & (idx < n)]
            mask[idx] = True
        return mask
