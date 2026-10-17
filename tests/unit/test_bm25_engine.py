# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under BSL 1.1 — see LICENSE for details.

"""Tests for the VecForge 2.0 incremental inverted-index BM25 engine."""

from __future__ import annotations

import math

import numpy as np

from vecforge.core.bm25 import BM25Engine


def _reference_scores(
    corpus: list[list[str]], query: list[str], k1: float, b: float
) -> list[float]:
    """Textbook BM25 (Lucene IDF) used to validate the engine."""
    n = len(corpus)
    avgdl = sum(len(d) for d in corpus) / n
    out = []
    for doc in corpus:
        score = 0.0
        for term in query:
            tf = doc.count(term)
            if tf == 0:
                continue
            df = sum(1 for d in corpus if term in d)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / avgdl))
        out.append(score)
    return out


class TestBM25Scoring:
    def test_matches_reference_formula(self) -> None:
        docs = [
            "patient with type 2 diabetes mellitus",
            "hip fracture in elderly patient",
            "cardiac arrest emergency response",
            "diabetes diabetes insulin therapy",
        ]
        engine = BM25Engine(k1=1.5, b=0.75)
        engine.add_documents(docs)
        corpus = [BM25Engine._tokenize(d) for d in docs]
        for query in ["diabetes", "elderly patient", "insulin patient diabetes"]:
            got = engine.get_scores(query)
            expected = _reference_scores(corpus, BM25Engine._tokenize(query), 1.5, 0.75)
            np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_repeated_query_terms_count_once_per_occurrence(self) -> None:
        engine = BM25Engine()
        engine.add_documents(["alpha beta", "gamma"])
        once = engine.get_scores("alpha")[0]
        twice = engine.get_scores("alpha alpha")[0]
        assert twice == np.float32(2.0) * once

    def test_idf_never_negative_for_common_terms(self) -> None:
        engine = BM25Engine()
        engine.add_documents(["the cat", "the dog", "the fish", "a bird"])
        scores = engine.get_scores("the")
        assert (scores[:3] > 0).all()
        assert scores[3] == 0.0


class TestIncrementalAdd:
    def test_incremental_add_equals_batch_add(self) -> None:
        docs = [f"doc {i} about topic {i % 3} and detail {i}" for i in range(50)]
        batch = BM25Engine()
        batch.add_documents(docs)
        incremental = BM25Engine()
        for d in docs:
            incremental.add_document(d)
        for q in ["topic 1", "detail 7", "doc about"]:
            np.testing.assert_allclose(batch.get_scores(q), incremental.get_scores(q))

    def test_positions_are_sequential(self) -> None:
        engine = BM25Engine()
        assert engine.add_documents(["a", "b"]) == [0, 1]
        assert engine.add_document("c") == 2
        assert engine.size == 3
        assert engine.count == 3

    def test_empty_document_is_indexed(self) -> None:
        engine = BM25Engine()
        engine.add_documents(["", "real words here"])
        assert engine.count == 2
        results = engine.search("words")
        assert [r.doc_index for r in results] == [1]

    def test_vocabulary_size(self) -> None:
        engine = BM25Engine()
        engine.add_documents(["one two", "two three"])
        assert engine.vocabulary_size == 3


class TestTombstonesAndScoping:
    def test_remove_excludes_from_results(self) -> None:
        engine = BM25Engine()
        engine.add_documents(["diabetes a", "diabetes b", "other"])
        assert engine.remove(0)
        assert not engine.remove(0)  # already removed
        assert not engine.remove(99)
        assert engine.count == 2
        assert engine.deleted_count == 1
        assert [r.doc_index for r in engine.search("diabetes")] == [1]

    def test_allowed_restricts_results(self) -> None:
        engine = BM25Engine()
        engine.add_documents(["diabetes a", "diabetes b", "diabetes c"])
        results = engine.search("diabetes", allowed=np.array([0, 2]))
        assert {r.doc_index for r in results} == {0, 2}

    def test_allowed_empty_returns_nothing(self) -> None:
        engine = BM25Engine()
        engine.add_documents(["diabetes a"])
        assert engine.search("diabetes", allowed=np.array([], dtype=np.int64)) == []

    def test_search_zero_top_k(self) -> None:
        engine = BM25Engine()
        engine.add_documents(["diabetes a"])
        assert engine.search("diabetes", top_k=0) == []


class TestSerialisation:
    def test_roundtrip_preserves_scores_and_tombstones(self) -> None:
        engine = BM25Engine(k1=1.2, b=0.5)
        engine.add_documents(
            ["alpha beta gamma", "beta delta", "gamma epsilon", "ünïcödé words"]
        )
        engine.remove(1)
        blob = engine.to_bytes()
        restored = BM25Engine.from_bytes(blob)
        assert restored.count == engine.count
        assert restored.size == engine.size
        assert restored.deleted_count == 1
        for q in ["beta", "gamma", "ünïcödé", "missing"]:
            np.testing.assert_allclose(restored.get_scores(q), engine.get_scores(q))
        # restored engine keeps accepting documents
        restored.add_document("new beta doc")
        assert restored.count == engine.count + 1

    def test_roundtrip_empty_engine(self) -> None:
        restored = BM25Engine.from_bytes(BM25Engine().to_bytes())
        assert restored.count == 0
        assert restored.search("x") == []

    def test_rejects_foreign_blob(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="BM25"):
            BM25Engine.from_bytes(b"not a vecforge blob")

    def test_reset_clears_everything(self) -> None:
        engine = BM25Engine()
        engine.add_documents(["a b", "c"])
        engine.remove(0)
        engine.reset()
        assert engine.size == 0
        assert engine.deleted_count == 0
        assert engine.vocabulary_size == 0
