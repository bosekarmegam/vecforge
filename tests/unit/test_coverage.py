# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under BSL 1.1 — see LICENSE for details.

"""Tests for VecForge coverage — batch API, filters, hybrid fusion,
cascade search, audit logging, encryption, and snapshots."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

# ─── Helper ───


def _make_vault():
    """Create an in-memory VecForge with mocked embedder."""
    from vecforge.core.vault import VecForge

    with patch("vecforge.core.vault.Embedder") as mock_cls:
        mock = mock_cls.return_value
        mock.dimension = 384
        mock.encode.side_effect = lambda text, *a, **kw: np.random.randn(
            1 if isinstance(text, str) else len(text), 384
        ).astype(np.float32)
        vault = VecForge(":memory:", model_name="test")
        vault._embedder = mock
        return vault


# ─── Batch Add Tests ───


class TestAddBatch:
    """Tests for VecForge.add_batch() bulk insertion."""

    def test_batch_add_returns_correct_count(self) -> None:
        vault = _make_vault()
        ids = vault.add_batch(["doc1", "doc2", "doc3"])
        assert len(ids) == 3
        assert vault.stats()["documents"] == 3
        vault.close()

    def test_batch_add_empty_list(self) -> None:
        vault = _make_vault()
        ids = vault.add_batch([])
        assert ids == []
        vault.close()

    def test_batch_add_with_metadata(self) -> None:
        vault = _make_vault()
        ids = vault.add_batch(
            ["a", "b"],
            metadata_list=[{"src": "x"}, {"src": "y"}],
        )
        assert len(ids) == 2
        vault.close()

    def test_batch_add_metadata_mismatch_raises(self) -> None:
        vault = _make_vault()
        with pytest.raises(ValueError, match="metadata_list length"):
            vault.add_batch(["a", "b"], metadata_list=[{"x": 1}])
        vault.close()

    def test_batch_add_namespace(self) -> None:
        vault = _make_vault()
        vault.add_batch(["ns doc"], namespace="team_a")
        assert "team_a" in vault.list_namespaces()
        vault.close()

    def test_batch_add_searchable(self) -> None:
        vault = _make_vault()
        vault.add_batch(["cardiac arrest", "hip fracture"])
        results = vault.search("cardiac", top_k=2)
        assert len(results) > 0
        vault.close()


# ─── MetadataFilter Tests ───


class TestMetadataFilter:
    """Tests for metadata filter operators."""

    def test_equality_match(self) -> None:
        from vecforge.search.filters import MetadataFilter

        f = MetadataFilter({"type": "NDA"})
        assert f.matches({"type": "NDA"})
        assert not f.matches({"type": "SLA"})

    def test_gte_operator(self) -> None:
        from vecforge.search.filters import MetadataFilter

        f = MetadataFilter({"year": {"gte": 2023}})
        assert f.matches({"year": 2024})
        assert f.matches({"year": 2023})
        assert not f.matches({"year": 2022})

    def test_lte_operator(self) -> None:
        from vecforge.search.filters import MetadataFilter

        f = MetadataFilter({"year": {"lte": 2023}})
        assert f.matches({"year": 2022})
        assert not f.matches({"year": 2024})

    def test_gt_lt_operators(self) -> None:
        from vecforge.search.filters import MetadataFilter

        f = MetadataFilter({"score": {"gt": 5, "lt": 10}})
        assert f.matches({"score": 7})
        assert not f.matches({"score": 5})
        assert not f.matches({"score": 10})

    def test_in_operator(self) -> None:
        from vecforge.search.filters import MetadataFilter

        f = MetadataFilter({"status": {"in": ["active", "pending"]}})
        assert f.matches({"status": "active"})
        assert not f.matches({"status": "closed"})

    def test_not_in_operator(self) -> None:
        from vecforge.search.filters import MetadataFilter

        f = MetadataFilter({"status": {"not_in": ["deleted"]}})
        assert f.matches({"status": "active"})
        assert not f.matches({"status": "deleted"})

    def test_ne_operator(self) -> None:
        from vecforge.search.filters import MetadataFilter

        f = MetadataFilter({"type": {"ne": "draft"}})
        assert f.matches({"type": "final"})
        assert not f.matches({"type": "draft"})

    def test_missing_key_fails(self) -> None:
        from vecforge.search.filters import MetadataFilter

        f = MetadataFilter({"missing_key": "value"})
        assert not f.matches({"other": "data"})

    def test_filter_results_method(self) -> None:
        from vecforge.search.filters import MetadataFilter

        class FakeResult:
            def __init__(self, meta):
                self.metadata = meta

        f = MetadataFilter({"year": {"gte": 2023}})
        results = [FakeResult({"year": 2024}), FakeResult({"year": 2020})]
        filtered = f.filter_results(results)
        assert len(filtered) == 1

    def test_empty_filter_returns_all(self) -> None:
        from vecforge.search.filters import MetadataFilter

        f = MetadataFilter({})
        assert f.filter_results(["a", "b", "c"]) == ["a", "b", "c"]

    def test_filter_with_callable_getter(self) -> None:
        from vecforge.search.filters import MetadataFilter

        f = MetadataFilter({"x": 1})
        data = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
        filtered = f.filter_results(data, metadata_getter=lambda d: d)
        assert len(filtered) == 1


# ─── Hybrid Fusion Tests ───


class TestHybridFusion:
    """Tests for reciprocal_rank_fusion and weighted_linear_fusion."""

    def test_rrf_basic(self) -> None:
        from vecforge.search.hybrid import reciprocal_rank_fusion

        dense_ids = np.array([0, 2, 5], dtype=np.int64)
        dense_scores = np.array([0.9, 0.7, 0.5], dtype=np.float32)
        sparse_ids = [2, 0, 3]
        sparse_scores = [5.0, 3.0, 1.0]

        fused = reciprocal_rank_fusion(
            dense_ids,
            dense_scores,
            sparse_ids,
            sparse_scores,
        )
        assert isinstance(fused, list)
        assert len(fused) >= 3

    def test_rrf_alpha_zero(self) -> None:
        from vecforge.search.hybrid import reciprocal_rank_fusion

        dense_ids = np.array([0], dtype=np.int64)
        dense_scores = np.array([1.0], dtype=np.float32)
        fused = reciprocal_rank_fusion(
            dense_ids,
            dense_scores,
            [1],
            [1.0],
            alpha=0.0,
        )
        # alpha=0 → dense gets 0 weight, sparse gets full
        scores = dict(fused)
        assert scores.get(1, 0) > scores.get(0, 0)

    def test_rrf_alpha_one(self) -> None:
        from vecforge.search.hybrid import reciprocal_rank_fusion

        dense_ids = np.array([0], dtype=np.int64)
        dense_scores = np.array([1.0], dtype=np.float32)
        fused = reciprocal_rank_fusion(
            dense_ids,
            dense_scores,
            [1],
            [1.0],
            alpha=1.0,
        )
        scores = dict(fused)
        assert scores.get(0, 0) > scores.get(1, 0)

    def test_rrf_skips_negative_ids(self) -> None:
        from vecforge.search.hybrid import reciprocal_rank_fusion

        dense_ids = np.array([-1, 0], dtype=np.int64)
        dense_scores = np.array([0.0, 0.9], dtype=np.float32)
        fused = reciprocal_rank_fusion(
            dense_ids,
            dense_scores,
            [],
            [],
        )
        doc_ids = [x[0] for x in fused]
        assert -1 not in doc_ids

    def test_linear_fusion(self) -> None:
        from vecforge.search.hybrid import weighted_linear_fusion

        dense_ids = np.array([0, 1], dtype=np.int64)
        dense_scores = np.array([0.9, 0.5], dtype=np.float32)
        sparse_ids = [1, 2]
        sparse_scores = [3.0, 1.0]

        fused = weighted_linear_fusion(
            dense_ids,
            dense_scores,
            sparse_ids,
            sparse_scores,
        )
        assert isinstance(fused, list)
        assert len(fused) >= 2

    def test_linear_fusion_empty(self) -> None:
        from vecforge.search.hybrid import weighted_linear_fusion

        dense_ids = np.array([], dtype=np.int64)
        dense_scores = np.array([], dtype=np.float32)

        fused = weighted_linear_fusion(
            dense_ids,
            dense_scores,
            [],
            [],
        )
        assert fused == []


# ─── CascadeSearcher Tests ───


class TestCascadeSearcher:
    """Tests for cascade search pipeline."""

    def test_search_returns_candidates(self) -> None:
        from vecforge.core.bm25 import BM25Engine
        from vecforge.core.indexer import FaissIndexer
        from vecforge.search.cascade import CascadeSearcher

        dim = 32
        indexer = FaissIndexer(dimension=dim)

        vecs = np.random.randn(10, dim).astype(np.float32)
        indexer.add(vecs)

        bm25 = BM25Engine()
        bm25.add_documents([f"document number {i}" for i in range(10)])

        cascade = CascadeSearcher(
            indexer=indexer,
            bm25=bm25,
            reranker=None,
        )
        query = np.random.randn(dim).astype(np.float32)

        results = cascade.search(
            query_vector=query,
            query_text="document",
            top_k=5,
        )
        assert len(results) > 0
        assert all(hasattr(r, "score") for r in results)
        assert all(hasattr(r, "doc_index") for r in results)

    def test_search_empty_indexer(self) -> None:
        from vecforge.core.bm25 import BM25Engine
        from vecforge.core.indexer import FaissIndexer
        from vecforge.search.cascade import CascadeSearcher

        indexer = FaissIndexer(dimension=16)
        bm25 = BM25Engine()
        cascade = CascadeSearcher(indexer=indexer, bm25=bm25)

        query = np.random.randn(16).astype(np.float32)
        results = cascade.search(query_vector=query, query_text="test")
        assert results == []


# ─── AuditLogger Tests ───


class TestAuditLogger:
    """Tests for audit logging."""

    def test_audit_log_writes_jsonl(self) -> None:
        from vecforge.security.audit import AuditLogger

        with tempfile.NamedTemporaryFile(
            suffix=".jsonl",
            delete=False,
            mode="w",
        ) as f:
            log_path = f.name

        try:
            logger = AuditLogger(log_path=log_path)
            logger.log(
                actor="admin",
                operation="add",
                doc_id="test-123",
                namespace="default",
                metadata={"chars": 42},
            )

            with open(log_path, encoding="utf-8") as f:
                lines = f.readlines()

            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["actor"] == "admin"
            assert entry["operation"] == "add"
            assert entry["doc_id"] == "test-123"
        finally:
            os.unlink(log_path)

    def test_audit_log_disabled(self) -> None:
        from vecforge.security.audit import AuditLogger

        logger = AuditLogger(log_path=None)
        assert not logger.enabled
        logger.log(actor="x", operation="add")

    def test_audit_read_log(self) -> None:
        from vecforge.security.audit import AuditLogger

        with tempfile.NamedTemporaryFile(
            suffix=".jsonl",
            delete=False,
            mode="w",
        ) as f:
            log_path = f.name

        try:
            logger = AuditLogger(log_path=log_path)
            logger.log(actor="a1", operation="add")
            logger.log(actor="a2", operation="delete")

            events = logger.read_log()
            assert len(events) == 2

            events = logger.read_log(actor="a1")
            assert len(events) == 1
            assert events[0]["actor"] == "a1"

            events = logger.read_log(operation="delete")
            assert len(events) == 1
        finally:
            os.unlink(log_path)

    def test_audit_read_empty(self) -> None:
        from vecforge.security.audit import AuditLogger

        logger = AuditLogger(log_path=None)
        assert logger.read_log() == []


# ─── Encryption Tests ───


class TestEncryption:
    """Tests for encryption key validation."""

    def test_none_key_returns_none(self) -> None:
        from vecforge.security.encryption import validate_encryption_key

        assert validate_encryption_key(None) is None

    def test_valid_key_returns_key(self) -> None:
        from vecforge.security.encryption import validate_encryption_key

        key = "my-secure-key-12345"
        assert validate_encryption_key(key) == key

    def test_short_key_raises(self) -> None:
        from vecforge.security.encryption import validate_encryption_key

        with pytest.raises(ValueError, match="at least 8 characters"):
            validate_encryption_key("short")

    def test_empty_key_raises(self) -> None:
        from vecforge.security.encryption import validate_encryption_key

        with pytest.raises(ValueError):
            validate_encryption_key("")

    def test_sqlcipher_check(self) -> None:
        from vecforge.security.encryption import check_sqlcipher_available

        result = check_sqlcipher_available()
        assert isinstance(result, bool)


# ─── Snapshots Tests ───


class TestSnapshots:
    """Tests for vault snapshot manager."""

    def test_create_snapshot(self) -> None:
        from vecforge.security.snapshots import SnapshotManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with open(db_path, "w") as f:
                f.write("fake db content")

            manager = SnapshotManager(vault_path=db_path)
            backup_dir = os.path.join(tmpdir, "backups")
            snap_path = manager.create_snapshot(backup_dir)
            assert os.path.exists(snap_path)

    def test_restore_snapshot(self) -> None:
        from vecforge.security.snapshots import SnapshotManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with open(db_path, "w") as f:
                f.write("original content")

            manager = SnapshotManager(vault_path=db_path)
            backup_dir = os.path.join(tmpdir, "backups")
            snap_path = manager.create_snapshot(backup_dir)

            # Modify original
            with open(db_path, "w") as f:
                f.write("modified content")

            manager.restore_snapshot(snap_path)

            with open(db_path) as f:
                assert f.read() == "original content"

    def test_list_snapshots(self) -> None:
        from vecforge.security.snapshots import SnapshotManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with open(db_path, "w") as f:
                f.write("data")

            manager = SnapshotManager(vault_path=db_path)
            backup_dir = os.path.join(tmpdir, "backups")
            manager.create_snapshot(backup_dir)

            snaps = manager.list_snapshots(backup_dir)
            assert len(snaps) >= 1

    def test_snapshot_missing_vault_raises(self) -> None:
        from vecforge.security.snapshots import SnapshotManager

        manager = SnapshotManager(vault_path="/nonexistent/file.db")
        with pytest.raises(FileNotFoundError):
            manager.create_snapshot("/tmp/backups")

    def test_restore_missing_snapshot_raises(self) -> None:
        from vecforge.security.snapshots import SnapshotManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = SnapshotManager(vault_path=db_path)
            with pytest.raises(FileNotFoundError):
                manager.restore_snapshot("/nonexistent/snap.db")

    def test_list_empty_dir(self) -> None:
        from vecforge.security.snapshots import SnapshotManager

        manager = SnapshotManager(vault_path="test.db")
        assert manager.list_snapshots("/nonexistent/dir") == []
