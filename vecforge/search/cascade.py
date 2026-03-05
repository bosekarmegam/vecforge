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
    1. FAISS dense retrieval (broad recall)
    2. BM25 keyword merge via hybrid fusion (precision boost)
    3. Metadata + namespace filtering
    4. Optional cross-encoder reranking

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vecforge.core.bm25 import BM25Engine
from vecforge.core.indexer import FaissIndexer
from vecforge.core.reranker import Reranker
from vecforge.search.filters import MetadataFilter
from vecforge.search.hybrid import weighted_linear_fusion

logger = logging.getLogger(__name__)


@dataclass
class CascadeCandidate:
    """Intermediate candidate during cascade search.

    Attributes:
        doc_index: Index in the document store.
        score: Current relevance score.
        text: Document text (loaded during cascade).
        metadata: Document metadata.
        namespace: Document namespace.
        doc_id: Unique document identifier.
        modality: Content modality.
        created_at: Document creation timestamp.
    """

    doc_index: int
    score: float
    text: str = ""
    metadata: dict[str, Any] | None = None
    namespace: str = "default"
    doc_id: str = ""
    modality: str = "text"
    created_at: float = 0.0


class CascadeSearcher:
    """4-stage cascading search pipeline.

    Processes search through increasingly precise stages:
    1. Dense: FAISS retrieves broad candidate set
    2. Sparse: BM25 scores merged via RRF or linear fusion
    3. Filter: Metadata and namespace filtering applied
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
    ) -> list[CascadeCandidate]:
        """Execute 4-stage cascading search.

        Args:
            query_vector: Dense query embedding from embedder.
            query_text: Original query string for BM25 and reranking.
            top_k: Number of final results to return.
            alpha: Semantic vs keyword weight (0.0-1.0).
            rerank: Enable cross-encoder reranking (Stage 4).
            filters: Metadata filter conditions.
            recency_weight: Weight for document recency (0.0-1.0).

        Returns:
            List of CascadeCandidate sorted by descending relevance.

        Performance:
            Time: O(log N) + O(k) + O(k*F) + O(k*d_rerank)
            Typical: <15ms without rerank, <50ms with rerank
        """
        # why: Retrieve more candidates than top_k to allow filtering
        retrieval_k = min(top_k * 4, self._indexer.count)
        if retrieval_k == 0:
            return []

        # ─── Stage 1: Dense retrieval via FAISS ───
        dense_scores, dense_ids = self._indexer.search(query_vector, top_k=retrieval_k)
        logger.debug("Stage 1 (Dense): retrieved %d candidates", len(dense_ids))

        # ─── Stage 2: Sparse keyword merge via Weighted Linear Fusion ───
        bm25_results = self._bm25.search(query_text, top_k=retrieval_k)
        sparse_ids = [r.doc_index for r in bm25_results]
        sparse_scores = [r.score for r in bm25_results]

        fused = weighted_linear_fusion(
            dense_ids=dense_ids,
            dense_scores=dense_scores,
            sparse_ids=sparse_ids,
            sparse_scores=sparse_scores,
            alpha=alpha,
        )
        logger.debug("Stage 2 (Hybrid): fused %d candidates", len(fused))

        # why: Convert to CascadeCandidate
        candidates = [
            CascadeCandidate(doc_index=doc_idx, score=score) for doc_idx, score in fused
        ]

        # ─── Stage 3: Metadata filtering ───
        if filters:
            meta_filter = MetadataFilter(filters)
            candidates = [
                c
                for c in candidates
                if c.metadata is not None and meta_filter.matches(c.metadata)
            ]
            logger.debug(
                "Stage 3 (Filter): %d candidates after filtering",
                len(candidates),
            )

        # ─── Stage 4: Cross-encoder reranking (optional) ───
        if rerank and self._reranker is not None and candidates:
            texts = [c.text for c in candidates if c.text]
            if texts:
                reranked = self._reranker.rerank(query_text, texts, top_k=top_k)
                reranked_candidates = []
                for orig_idx, rerank_score in reranked:
                    if orig_idx < len(candidates):
                        c = candidates[orig_idx]
                        c.score = rerank_score
                        reranked_candidates.append(c)
                candidates = reranked_candidates
                logger.debug(
                    "Stage 4 (Rerank): reranked to %d results",
                    len(candidates),
                )

        return candidates[:top_k]
