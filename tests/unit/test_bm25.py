# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under BSL 1.1 — see LICENSE for details.

"""Unit tests for BM25 keyword search engine."""

from __future__ import annotations

from vecforge.core.bm25 import BM25Engine


class TestBM25Engine:
    """Tests for BM25 keyword search."""

    def setup_method(self) -> None:
        self.engine = BM25Engine()

    def test_add_and_search(self) -> None:
        """Add documents and search returns relevant results."""
        self.engine.add_documents(
            [
                "patient with type 2 diabetes mellitus",
                "hip fracture in elderly patient",
                "cardiac arrest emergency response",
            ]
        )

        results = self.engine.search("diabetes", top_k=2)
        assert len(results) >= 1
        assert results[0].doc_index == 0  # diabetes doc should rank first
        assert results[0].score > 0

    def test_empty_search(self) -> None:
        """Searching empty engine returns empty list."""
        results = self.engine.search("anything")
        assert results == []

    def test_no_match(self) -> None:
        """Search with no matching terms returns empty or low scores."""
        self.engine.add_documents(["cat sat on mat"])
        results = self.engine.search("quantum physics relativity")
        # Results may be empty or have zero scores
        for r in results:
            assert r.score >= 0

    def test_count(self) -> None:
        """Count tracks number of documents."""
        assert self.engine.count == 0
        self.engine.add_documents(["doc1", "doc2"])
        assert self.engine.count == 2
        self.engine.add_document("doc3")
        assert self.engine.count == 3

    def test_reset(self) -> None:
        """Reset clears all documents."""
        self.engine.add_documents(["doc1", "doc2"])
        self.engine.reset()
        assert self.engine.count == 0
        assert self.engine.search("doc1") == []

    def test_tokenizer(self) -> None:
        """Tokenizer produces lowercase word tokens."""
        tokens = BM25Engine._tokenize("Hello, World! Test 123.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        assert "123" in tokens

    def test_top_k_limiting(self) -> None:
        """Results limited to top_k."""
        self.engine.add_documents(
            [
                "word alpha",
                "word beta",
                "word gamma",
                "word delta",
                "word epsilon",
            ]
        )
        results = self.engine.search("word", top_k=2)
        assert len(results) <= 2
