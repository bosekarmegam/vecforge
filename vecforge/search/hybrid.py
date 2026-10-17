# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
Hybrid dense + sparse search fusion for VecForge.

Combines FAISS dense retrieval scores with BM25 sparse keyword scores
using configurable weighted fusion. Alpha controls the balance between
semantic understanding and keyword matching.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class FusedHit:
    """One fused candidate with its per-retriever components.

    Attributes:
        doc_index: Index position of the document.
        score: Fused score in ``[0, 1]``.
        dense_score: Min-max normalised dense score, or None if the dense
            retriever did not return this document.
        sparse_score: Min-max normalised BM25 score, or None if the sparse
            retriever did not return this document.
        raw_dense: Raw inner-product (cosine) similarity, if available.
        raw_sparse: Raw BM25 score, if available.
    """

    doc_index: int
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    raw_dense: float | None = None
    raw_sparse: float | None = None

    @property
    def signals(self) -> list[str]:
        """Names of the retrievers that matched this document."""
        out = []
        if self.dense_score is not None:
            out.append("dense")
        if self.sparse_score is not None:
            out.append("sparse")
        return out


def reciprocal_rank_fusion(
    dense_ids: NDArray[np.int64],
    dense_scores: NDArray[np.float32],
    sparse_ids: list[int],
    sparse_scores: list[float],
    alpha: float = 0.5,
    k: int = 60,
) -> list[tuple[int, float]]:
    """Fuse dense and sparse search results using weighted RRF.

    Reciprocal Rank Fusion (RRF) combines rankings from multiple
    retrieval systems. Each document's score is computed as:

        score = alpha * 1/(k + dense_rank) + (1 - alpha) * 1/(k + sparse_rank)

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        dense_ids: Document indices from FAISS dense retrieval.
        dense_scores: Corresponding FAISS inner-product scores.
        sparse_ids: Document indices from BM25 sparse retrieval.
        sparse_scores: Corresponding BM25 scores.
        alpha: Weight for dense vs sparse (0.0 = keyword only,
            1.0 = semantic only). Defaults to 0.5 (balanced).
        k: RRF constant. Higher values reduce the impact of ranking
            position. Defaults to 60 (standard RRF constant).

    Returns:
        List of (doc_index, fused_score) tuples sorted by descending
        fused score.

    Performance:
        Time: O(D + S) where D = dense results, S = sparse results
        Typical: <1ms for top-100 results

    Example:
        >>> fused = reciprocal_rank_fusion(
        ...     dense_ids=np.array([0, 2, 5]),
        ...     dense_scores=np.array([0.9, 0.7, 0.5]),
        ...     sparse_ids=[2, 0, 3],
        ...     sparse_scores=[5.0, 3.0, 1.0],
        ...     alpha=0.5,
        ... )
        >>> fused[0]  # (doc_id, combined_score)
    """
    del dense_scores, sparse_scores  # RRF uses ranks only
    fused_scores: dict[int, float] = {}

    # perf: Process dense results — already sorted by score descending
    for rank, doc_id in enumerate(dense_ids):
        doc_id_int = int(doc_id)
        if doc_id_int < 0:  # why: FAISS returns -1 for padded slots
            continue
        rrf_score = alpha * (1.0 / (k + rank + 1))
        fused_scores[doc_id_int] = fused_scores.get(doc_id_int, 0.0) + rrf_score

    # perf: Process sparse results — already sorted by score descending
    for rank, doc_id in enumerate(sparse_ids):
        rrf_score = (1.0 - alpha) * (1.0 / (k + rank + 1))
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + rrf_score

    # why: Sort by fused score descending
    results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return results


def weighted_linear_fusion_detailed(
    dense_ids: NDArray[np.int64],
    dense_scores: NDArray[np.float32],
    sparse_ids: list[int],
    sparse_scores: list[float],
    alpha: float = 0.5,
) -> list[FusedHit]:
    """Fuse dense and sparse scores, keeping per-retriever components.

    Both score distributions are min-max normalised to ``[0, 1]`` over the
    candidate set and combined as ``alpha * dense + (1 - alpha) * sparse``.
    A document returned by only one retriever receives only that
    retriever's weighted contribution. When one retriever returns nothing
    (for example BM25 with no keyword overlap) the other receives the full
    weight so scores stay on the ``[0, 1]`` scale.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        dense_ids: Document indices from FAISS.
        dense_scores: FAISS inner-product scores.
        sparse_ids: Document indices from BM25.
        sparse_scores: BM25 scores.
        alpha: Semantic weight. Defaults to 0.5.

    Returns:
        List of :class:`FusedHit` sorted by descending fused score, ties
        broken by raw dense similarity.

    Performance:
        Time: O(D + S + U log U) where U = unique docs
    """
    hits: dict[int, FusedHit] = {}

    has_dense = len(dense_scores) > 0
    has_sparse = len(sparse_scores) > 0
    dense_weight = alpha if has_sparse else 1.0
    sparse_weight = (1.0 - alpha) if has_dense else 1.0

    if has_dense:
        d_min, d_max = float(dense_scores.min()), float(dense_scores.max())
        d_range = d_max - d_min if d_max > d_min else 1.0
        for doc_id, score in zip(dense_ids, dense_scores, strict=False):
            doc_id_int = int(doc_id)
            if doc_id_int < 0:
                continue
            raw = float(score)
            norm_score = (raw - d_min) / d_range if d_max > d_min else 1.0
            hits[doc_id_int] = FusedHit(
                doc_index=doc_id_int,
                score=dense_weight * norm_score,
                dense_score=norm_score,
                raw_dense=raw,
            )

    if has_sparse:
        s_min = min(sparse_scores)
        s_max = max(sparse_scores)
        s_range = s_max - s_min if s_max > s_min else 1.0
        for doc_id, score in zip(sparse_ids, sparse_scores, strict=False):
            norm_score = (score - s_min) / s_range if s_max > s_min else 1.0
            hit = hits.get(doc_id)
            if hit is None:
                hits[doc_id] = FusedHit(
                    doc_index=doc_id,
                    score=sparse_weight * norm_score,
                    sparse_score=norm_score,
                    raw_sparse=float(score),
                )
            else:
                hit.score += sparse_weight * norm_score
                hit.sparse_score = norm_score
                hit.raw_sparse = float(score)

    # why: on equal fused scores prefer documents both retrievers agreed on,
    # then the higher raw semantic similarity
    return sorted(
        hits.values(),
        key=lambda h: (
            round(h.score, 9),
            len(h.signals),
            h.raw_dense if h.raw_dense is not None else -1.0,
        ),
        reverse=True,
    )


def weighted_linear_fusion(
    dense_ids: NDArray[np.int64],
    dense_scores: NDArray[np.float32],
    sparse_ids: list[int],
    sparse_scores: list[float],
    alpha: float = 0.5,
) -> list[tuple[int, float]]:
    """Fuse dense and sparse scores using weighted linear combination.

    Thin wrapper over :func:`weighted_linear_fusion_detailed` returning
    plain ``(doc_index, fused_score)`` tuples.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        dense_ids: Document indices from FAISS.
        dense_scores: FAISS scores.
        sparse_ids: Document indices from BM25.
        sparse_scores: BM25 scores.
        alpha: Semantic weight. Defaults to 0.5.

    Returns:
        Sorted list of (doc_index, fused_score) tuples.

    Performance:
        Time: O(D + S + U*log(U)) where U = unique docs
    """
    return [
        (h.doc_index, h.score)
        for h in weighted_linear_fusion_detailed(
            dense_ids, dense_scores, sparse_ids, sparse_scores, alpha=alpha
        )
    ]
