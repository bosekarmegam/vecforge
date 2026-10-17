# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
4-stage cascading retrieval pipeline for VecForge.

Pipeline stages:
    1. FAISS dense retrieval (broad recall, scoped by allowed/excluded ids)
    2. BM25 keyword merge via hybrid fusion (precision boost)
    3. Metadata + namespace filtering
    4. Optional cross-encoder reranking

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vecforge.core.bm25 import BM25Engine
from vecforge.core.indexer import FaissIndexer
from vecforge.core.reranker import Reranker
from vecforge.search.filters import MetadataFilter
from vecforge.search.hybrid import weighted_linear_fusion_detailed

logger = logging.getLogger(__name__)


@dataclass
class CascadeCandidate:
    """Intermediate candidate during cascade search.

    Attributes:
        doc_index: Index position in the vault.
        score: Current relevance score.
        text: Document text (loaded during hydration).
        metadata: Document metadata (None until hydrated).
        namespace: Document namespace.
        doc_id: Unique document identifier.
        modality: Content modality.
        created_at: Document creation timestamp.
        dense_score: Normalised dense score (None if not a dense hit).
        sparse_score: Normalised BM25 score (None if not a sparse hit).
        signals: Retrievers that matched (``dense``, ``sparse``).
    """

    doc_index: int
    score: float
    text: str = ""
    metadata: dict[str, Any] | None = None
    namespace: str = "default"
    doc_id: str = ""
    modality: str = "text"
    created_at: float = 0.0
    dense_score: float | None = None
    sparse_score: float | None = None
    signals: list[str] = field(default_factory=list)


class CascadeSearcher:
    """4-stage cascading search pipeline.

    Processes search through increasingly precise stages:
    1. Dense: FAISS retrieves broad candidate set
    2. Sparse: BM25 scores merged via linear fusion
    3. Filter: Metadata filtering applied (when metadata is loaded)
    4. Rerank: Optional cross-encoder reranking for precision

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        indexer: FAISS index for dense retrieval.
        bm25: BM25 engine for keyword search.
        reranker: Optional cross-encoder reranker.

    Performance:
        Time: O(log N) FAISS + O(k) rerank where k << N
        Typical: <15ms at 100k docs, <50ms at 1M docs

    Example:
        >>> searcher = CascadeSearcher(indexer, bm25_engine)
        >>> results = searcher.search(query_vec, "diabetes", top_k=10)
    """

    def __init__(
        self,
        indexer: FaissIndexer,
        bm25: BM25Engine,
        reranker: Reranker | None = None,
    ) -> None:
        self._indexer = indexer
        self._bm25 = bm25
        self._reranker = reranker

    def search(
        self,
        query_vector: NDArray[np.float32],
        query_text: str,
        top_k: int = 10,
        alpha: float = 0.5,
        rerank: bool = False,
        filters: dict[str, Any] | None = None,
        recency_weight: float = 0.0,
        allowed_ids: NDArray[np.int64] | None = None,
        excluded_ids: NDArray[np.int64] | None = None,
        over_fetch: int = 4,
    ) -> list[CascadeCandidate]:
        """Execute the cascading search.

        Args:
            query_vector: Dense query embedding from embedder.
            query_text: Original query string for BM25 and reranking.
            top_k: Number of final results to return.
            alpha: Semantic vs keyword weight (0.0-1.0).
            rerank: Enable cross-encoder reranking (Stage 4). Only has an
                effect when candidates carry text.
            filters: Metadata filter conditions, applied to candidates
                whose metadata has been loaded.
            recency_weight: Reserved for the vault layer (ignored here).
            allowed_ids: Restrict retrieval to these index positions
                (namespace scoping).
            excluded_ids: Never return these positions (tombstones).
            over_fetch: Retrieve ``top_k * over_fetch`` candidates from
                each retriever before fusion.

        Returns:
            List of CascadeCandidate sorted by descending relevance.

        Performance:
            Time: O(log N) + O(k) + O(k*F) + O(k*d_rerank)
            Typical: <15ms without rerank, <50ms with rerank
        """
        del recency_weight
        # why: an allowed set is already free of tombstones; otherwise the
        # excluded positions shrink the pool
        if allowed_ids is not None:
            pool = min(self._indexer.count, len(allowed_ids))
        elif excluded_ids is not None:
            pool = max(0, self._indexer.count - len(excluded_ids))
        else:
            pool = self._indexer.count
        retrieval_k = min(max(top_k * max(1, over_fetch), top_k), pool)
        if retrieval_k <= 0:
            return []

        # ─── Stage 1: Dense retrieval via FAISS ───
        dense_scores, dense_ids = self._indexer.search(
            query_vector,
            top_k=retrieval_k,
            allowed_ids=allowed_ids,
            excluded_ids=excluded_ids,
        )
        logger.debug("Stage 1 (Dense): retrieved %d candidates", len(dense_ids))

        # ─── Stage 2: Sparse keyword merge via weighted linear fusion ───
        if alpha >= 1.0 or not query_text:
            bm25_results = []
        else:
            bm25_results = self._bm25.search(
                query_text, top_k=retrieval_k, allowed=allowed_ids
            )
        if excluded_ids is not None and len(excluded_ids) > 0 and bm25_results:
            excluded_set = {int(e) for e in excluded_ids}
            bm25_results = [r for r in bm25_results if r.doc_index not in excluded_set]
        sparse_ids = [r.doc_index for r in bm25_results]
        sparse_scores = [r.score for r in bm25_results]

        if alpha <= 0.0 and sparse_ids:
            dense_ids = np.zeros(0, dtype=np.int64)
            dense_scores = np.zeros(0, dtype=np.float32)

        fused = weighted_linear_fusion_detailed(
            dense_ids=dense_ids,
            dense_scores=dense_scores,
            sparse_ids=sparse_ids,
            sparse_scores=sparse_scores,
            alpha=alpha,
        )
        logger.debug("Stage 2 (Hybrid): fused %d candidates", len(fused))

        candidates = [
            CascadeCandidate(
                doc_index=h.doc_index,
                score=h.score,
                dense_score=h.dense_score,
                sparse_score=h.sparse_score,
                signals=h.signals,
            )
            for h in fused
        ]

        # ─── Stage 3: Metadata filtering (only for hydrated candidates) ───
        if filters:
            meta_filter = MetadataFilter(filters)
            candidates = [
                c
                for c in candidates
                if c.metadata is None or meta_filter.matches(c.metadata)
            ]
            logger.debug(
                "Stage 3 (Filter): %d candidates after filtering",
                len(candidates),
            )

        # ─── Stage 4: Cross-encoder reranking (optional) ───
        if rerank and self._reranker is not None and candidates:
            with_text = [c for c in candidates if c.text]
            if with_text:
                reranked = self._reranker.rerank(
                    query_text, [c.text for c in with_text], top_k=top_k
                )
                reranked_candidates = []
                for orig_idx, rerank_score in reranked:
                    if orig_idx < len(with_text):
                        c = with_text[orig_idx]
                        c.score = rerank_score
                        reranked_candidates.append(c)
                candidates = reranked_candidates
                logger.debug(
                    "Stage 4 (Rerank): reranked to %d results",
                    len(candidates),
                )

        return candidates[:top_k]
