# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
Cross-encoder reranker for VecForge.

Provides high-precision reranking of search candidates using a
cross-encoder model. Applied as the final stage of the cascade
search pipeline for improved accuracy.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Cross-encoder reranker for precision search refinement.

    Lazily loads the cross-encoder model on first use. Reranks candidate
    results by computing query-document relevance scores using a
    cross-attention model — more accurate but slower than bi-encoder.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        model_name: Cross-encoder model name.
            Defaults to 'cross-encoder/ms-marco-MiniLM-L-6-v2'.

    Performance:
        Time: O(k * d) where k = candidates, d = model complexity
        Typical: ~20-50ms for top-20 candidates

    Example:
        >>> reranker = Reranker()
        >>> scored = reranker.rerank("diabetes treatment", candidates)
        >>> print(scored[0])  # highest relevance
    """

    def __init__(self, model_name: str = _DEFAULT_RERANK_MODEL) -> None:
        self._model_name = model_name
        self._model: Any = None

    def _load_model(self) -> None:
        """Lazily load the cross-encoder model.

        Performance:
            Time: O(1) — one-time cost of ~1-2 seconds
        """
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for reranking.\n"
                "Install with: pip install sentence-transformers\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            ) from e

        logger.info("Loading reranker model: %s", self._model_name)
        self._model = CrossEncoder(self._model_name)
        logger.info("Reranker model loaded: %s", self._model_name)

    def rerank(
        self,
        query: str,
        texts: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Rerank candidate texts by relevance to query.

        Args:
            query: Search query string.
            texts: List of candidate document texts to rerank.
            top_k: Number of top results to return. If None, returns all.

        Returns:
            List of (original_index, score) tuples sorted by descending
            relevance score. original_index maps back to the input texts.

        Performance:
            Time: O(k * d) where k = len(texts)
            Typical: ~20-50ms for 20 candidates

        Example:
            >>> ranked = reranker.rerank("hip fracture", ["broken hip", "diabetes"])
            >>> ranked[0]  # (0, 0.95) — "broken hip" most relevant
        """
        if not texts:
            return []

        self._load_model()

        # why: Cross-encoder expects (query, doc) pairs
        pairs = [(query, text) for text in texts]
        scores = self._model.predict(pairs)

        # why: Create (index, score) pairs and sort by score descending
        indexed_scores = [(i, float(score)) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            indexed_scores = indexed_scores[:top_k]

        return indexed_scores
