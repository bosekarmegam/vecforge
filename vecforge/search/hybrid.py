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

import numpy as np
from numpy.typing import NDArray


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


def weighted_linear_fusion(
    dense_ids: NDArray[np.int64],
    dense_scores: NDArray[np.float32],
    sparse_ids: list[int],
    sparse_scores: list[float],
    alpha: float = 0.5,
) -> list[tuple[int, float]]:
    """Fuse dense and sparse scores using weighted linear combination.

    Normalizes both score distributions to [0, 1] and combines:

        score = alpha * norm_dense + (1 - alpha) * norm_sparse

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
    fused_scores: dict[int, float] = {}

    # perf: Normalize dense scores to [0, 1]
    if len(dense_scores) > 0:
        d_min, d_max = float(dense_scores.min()), float(dense_scores.max())
        d_range = d_max - d_min if d_max > d_min else 1.0

        for doc_id, score in zip(dense_ids, dense_scores, strict=False):
            doc_id_int = int(doc_id)
            if doc_id_int < 0:
                continue
            norm_score = (float(score) - d_min) / d_range
            fused_scores[doc_id_int] = alpha * norm_score

    # perf: Normalize sparse scores to [0, 1]
    if len(sparse_scores) > 0:
        s_min = min(sparse_scores)
        s_max = max(sparse_scores)
        s_range = s_max - s_min if s_max > s_min else 1.0

        for doc_id, score in zip(sparse_ids, sparse_scores, strict=False):
            norm_score = (score - s_min) / s_range
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (
                (1.0 - alpha) * norm_score
            )

    results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return results
