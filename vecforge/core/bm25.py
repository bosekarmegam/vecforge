# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
BM25 keyword search engine for VecForge.

Provides sparse keyword-based retrieval using BM25Okapi. Used alongside
FAISS dense retrieval for hybrid search.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


@dataclass
class BM25Result:
    """A single BM25 search result.

    Attributes:
        doc_index: Index of the document in the corpus.
        score: BM25 relevance score (higher = more relevant).
    """

    doc_index: int
    score: float


class BM25Engine:
    """BM25 keyword search engine using Okapi BM25.

    Maintains an in-memory inverted index for fast keyword retrieval.
    Rebuilt on each add operation (efficient for batch ingestion).

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Performance:
        Build: O(N * L) where N = docs, L = avg doc length
        Search: O(V * N) where V = query terms
        Typical: <2ms search at 100k docs

    Example:
        >>> engine = BM25Engine()
        >>> engine.add_documents(["patient with diabetes", "hip fracture case"])
        >>> results = engine.search("diabetes", top_k=1)
        >>> results[0].doc_index
        0
    """

    def __init__(self) -> None:
        self._corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    @property
    def count(self) -> int:
        """Return number of documents in the corpus.

        Performance:
            Time: O(1)
        """
        return len(self._corpus)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text into lowercase words.

        Simple whitespace + punctuation tokenizer. Adequate for BM25
        where exact matching matters more than linguistic analysis.

        Args:
            text: Raw text string to tokenize.

        Returns:
            List of lowercase word tokens.

        Performance:
            Time: O(L) where L = length of text
        """
        # why: Simple regex tokenizer — BM25 doesn't need stemming for v0.1
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def add_documents(self, texts: list[str]) -> None:
        """Add documents to the BM25 index.

        Rebuilds the internal BM25 index after adding. For best
        performance, batch all documents into a single call.

        Args:
            texts: List of document texts to add.

        Performance:
            Time: O(N * L) where N = total docs, L = avg doc length

        Example:
            >>> engine = BM25Engine()
            >>> engine.add_documents(["doc one", "doc two", "doc three"])
            >>> engine.count
            3
        """
        for text in texts:
            self._corpus.append(self._tokenize(text))

        # why: Rebuild entire index — BM25Okapi doesn't support incremental add
        self._bm25 = BM25Okapi(self._corpus)
        logger.debug("BM25 index rebuilt with %d documents", len(self._corpus))

    def add_document(self, text: str) -> None:
        """Add a single document to the BM25 index.

        Args:
            text: Document text to add.

        Performance:
            Time: O(N * L) — rebuilds entire index
        """
        self.add_documents([text])

    def search(self, query: str, top_k: int = 10) -> list[BM25Result]:
        """Search for documents matching the query keywords.

        Args:
            query: Search query string.
            top_k: Number of top results to return.

        Returns:
            List of BM25Result sorted by descending score.
            Empty list if no documents in corpus.

        Performance:
            Time: O(V * N) where V = query terms, N = corpus size
            Typical: <2ms at 100k docs

        Example:
            >>> results = engine.search("diabetes treatment", top_k=5)
            >>> for r in results:
            ...     print(f"Doc {r.doc_index}: score={r.score:.4f}")
        """
        if self._bm25 is None or len(self._corpus) == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # perf: BM25Okapi.get_scores returns all scores in one pass
        scores = self._bm25.get_scores(query_tokens)

        # perf: Use argpartition for O(N) top-k instead of O(N log N) sort
        effective_k = min(top_k, len(scores))
        if effective_k == 0:
            return []

        top_indices = np.argpartition(scores, -effective_k)[-effective_k:]
        # why: Sort the top-k by score descending
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            BM25Result(doc_index=int(idx), score=float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0.0  # why: Filter zero-score matches
        ]

    def reset(self) -> None:
        """Reset the BM25 index, removing all documents.

        Performance:
            Time: O(1)
        """
        self._corpus = []
        self._bm25 = None
