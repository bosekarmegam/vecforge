# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under BSL 1.1 — see LICENSE for details.

"""Tests for the FAISS indexer: selectors, HNSW, serialisation."""

from __future__ import annotations

import numpy as np
import pytest

import vecforge.core.indexer as indexer_mod
from vecforge.core.indexer import FaissIndexer


def _unit(n: int, d: int = 16, seed: int = 0) -> np.ndarray:
    x = np.random.default_rng(seed).random((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class TestSelectors:
    def test_allowed_ids_restrict_results(self) -> None:
        x = _unit(50)
        idx = FaissIndexer(16, index_type="flat")
        idx.add(x)
        allowed = np.array([3, 5, 7], dtype=np.int64)
        d, i = idx.search(x[3], top_k=5, allowed_ids=allowed)
        assert set(i.tolist()) <= set(allowed.tolist())
        assert i[0] == 3
        assert len(i) == 3  # clamped to the allowed pool

    def test_excluded_ids_never_returned(self) -> None:
        x = _unit(50)
        idx = FaissIndexer(16, index_type="flat")
        idx.add(x)
        d, i = idx.search(x[3], top_k=5, excluded_ids=np.array([3], dtype=np.int64))
        assert 3 not in i.tolist()
        assert len(i) == 5

    def test_allowed_and_excluded_combined(self) -> None:
        x = _unit(50)
        idx = FaissIndexer(16, index_type="flat")
        idx.add(x)
        d, i = idx.search(
            x[3],
            top_k=5,
            allowed_ids=np.array([3, 5, 7], dtype=np.int64),
            excluded_ids=np.array([3], dtype=np.int64),
        )
        assert set(i.tolist()) == {5, 7}

    def test_empty_allowed_returns_nothing(self) -> None:
        x = _unit(10)
        idx = FaissIndexer(16, index_type="flat")
        idx.add(x)
        d, i = idx.search(x[0], top_k=3, allowed_ids=np.zeros(0, dtype=np.int64))
        assert len(i) == 0

    def test_search_batch_shape(self) -> None:
        x = _unit(20)
        idx = FaissIndexer(16, index_type="flat")
        idx.add(x)
        d, i = idx.search_batch(x[:4], top_k=3)
        assert d.shape == (4, 3) and i.shape == (4, 3)
        assert (i[:, 0] == np.arange(4)).all()

    def test_dimension_mismatch(self) -> None:
        idx = FaissIndexer(16)
        with pytest.raises(ValueError, match="dimension"):
            idx.add(np.ones((1, 8), dtype=np.float32))
        idx.add(_unit(2))
        with pytest.raises(ValueError, match="dimension"):
            idx.search(np.ones(8, dtype=np.float32))


class TestHNSW:
    def test_hnsw_index_with_selectors(self) -> None:
        x = _unit(300)
        idx = FaissIndexer(16, index_type="hnsw")
        assert idx.index_type == "hnsw"
        idx.add(x)
        d, i = idx.search(x[10], top_k=5)
        assert i[0] == 10
        d, i = idx.search(
            x[10], top_k=3, allowed_ids=np.array([1, 2, 10], dtype=np.int64)
        )
        assert i[0] == 10 and set(i.tolist()) <= {1, 2, 10}
        d, i = idx.search(x[10], top_k=3, excluded_ids=np.array([10], dtype=np.int64))
        assert 10 not in i.tolist()

    def test_auto_upgrades_past_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(indexer_mod, "AUTO_HNSW_THRESHOLD", 100)
        idx = FaissIndexer(16, index_type="auto")
        idx.add(_unit(60))
        assert idx.index_type == "flat"
        idx.add(_unit(60, seed=1))
        assert idx.index_type == "hnsw"
        assert idx.count == 120
        x = idx.reconstruct_all()
        assert x.shape == (120, 16)
        d, i = idx.search(x[5], top_k=1)
        assert i[0] == 5

    def test_reset_keeps_requested_type(self) -> None:
        idx = FaissIndexer(16, index_type="hnsw")
        idx.add(_unit(5))
        idx.reset()
        assert idx.count == 0 and idx.index_type == "hnsw"
        flat = FaissIndexer(16, index_type="flat")
        flat.reset()
        assert flat.index_type == "flat"

    def test_bad_type(self) -> None:
        with pytest.raises(ValueError):
            FaissIndexer(16, index_type="ivf")  # type: ignore[arg-type]


class TestSerialisation:
    @pytest.mark.parametrize("kind", ["flat", "hnsw"])
    def test_roundtrip_detects_type(self, kind: str) -> None:
        x = _unit(40)
        idx = FaissIndexer(16, index_type=kind)  # type: ignore[arg-type]
        idx.add(x)
        restored = FaissIndexer.from_bytes(idx.to_bytes(), 16)
        assert restored.count == 40
        assert restored.index_type == kind
        d, i = restored.search(x[7], top_k=1)
        assert i[0] == 7

    def test_wrong_dimension_rejected(self) -> None:
        idx = FaissIndexer(16)
        idx.add(_unit(3))
        with pytest.raises(ValueError, match="dimension"):
            FaissIndexer.from_bytes(idx.to_bytes(), 32)

    def test_reconstruct_empty(self) -> None:
        assert FaissIndexer(4).reconstruct_all().shape == (0, 4)
