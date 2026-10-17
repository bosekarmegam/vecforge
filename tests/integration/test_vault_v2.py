# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under BSL 1.1 — see LICENSE for details.

"""Integration tests for the VecForge 2.0 vault using the hash embedder.

No model download is needed, so these exercise the real FAISS + BM25 +
SQLite pipeline end to end.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from vecforge import VecForge
from vecforge.core.embedder import HashEmbedder
from vecforge.exceptions import (
    DeletionProtectedError,
    DocumentNotFoundError,
    DuplicateDocumentError,
    NamespaceNotFoundError,
    VaultClosedError,
    VaultEmptyError,
)

DOCS = [
    ("patient with type 2 diabetes mellitus", {"ward": 7, "year": 2024}),
    ("hip fracture in elderly patient", {"ward": 3, "year": 2023}),
    ("cardiac arrest emergency response", {"ward": "icu", "year": 2024}),
    ("stock market crash analysis", {"ward": 0, "year": 2020}),
]


@pytest.fixture
def vault() -> VecForge:
    db = VecForge(":memory:", model_name="hash")
    yield db
    if not db.closed:
        db.close()


def _fill(db: VecForge) -> list[str]:
    return db.add_batch([t for t, _ in DOCS], metadata_list=[m for _, m in DOCS])


class TestSearchSemantics:
    def test_hybrid_search_returns_signals(self, vault: VecForge) -> None:
        _fill(vault)
        res = vault.search("type 2 diabetes", top_k=2)
        assert res[0].text.startswith("patient with type 2 diabetes")
        assert res[0].score == pytest.approx(1.0)
        assert "dense" in res[0].signals and "sparse" in res[0].signals
        assert res[0].dense_score is not None

    def test_namespace_recall_with_many_namespaces(self, vault: VecForge) -> None:
        """1.x filtered namespaces after a fixed top_k*4 fetch and lost recall."""
        for ns in range(40):
            vault.add_batch(
                [f"note {i} for tenant {ns} about billing" for i in range(10)],
                namespace=f"tenant{ns}",
            )
        res = vault.search("billing note", namespace="tenant39", top_k=10)
        assert len(res) == 10
        assert all(r.namespace == "tenant39" for r in res)

    def test_metadata_filter_widens_window(self, vault: VecForge) -> None:
        """A rare filter value still yields top_k matches."""
        vault.add_batch([f"generic document number {i}" for i in range(200)])
        vault.add_batch(
            [f"rare document number {i}" for i in range(5)],
            metadata_list=[{"rare": True} for _ in range(5)],
        )
        res = vault.search("document", filters={"rare": True}, top_k=5)
        assert len(res) == 5
        assert all(r.metadata.get("rare") for r in res)

    def test_filter_with_incomparable_metadata(self, vault: VecForge) -> None:
        _fill(vault)
        res = vault.search("patient", filters={"ward": {"gte": 5}}, top_k=5)
        assert [r.metadata["ward"] for r in res] == [7]

    def test_alpha_extremes(self, vault: VecForge) -> None:
        _fill(vault)
        dense_only = vault.search("diabetes", alpha=1.0, top_k=2)
        sparse_only = vault.search("diabetes", alpha=0.0, top_k=2)
        assert dense_only[0].sparse_score is None
        assert all(r.dense_score is None for r in sparse_only)
        assert sparse_only[0].text.startswith("patient with type 2 diabetes")

    def test_recency_weight_prefers_newer(self, vault: VecForge) -> None:
        _fill(vault)
        res = vault.search("patient", recency_weight=1.0, top_k=4)
        times = [r.timestamp for r in res]
        assert times == sorted(times, reverse=True)

    def test_quantum_rerank_keeps_best_first(self, vault: VecForge) -> None:
        _fill(vault)
        for top_k in (1, 2, 3, 4):
            res = vault.search("type 2 diabetes", top_k=top_k, quantum_rerank=True)
            assert res[0].text.startswith("patient with type 2 diabetes")
            assert res[0].score == pytest.approx(1.0)
            assert len(res) == top_k

    def test_vault_level_quantum_default(self) -> None:
        db = VecForge(":memory:", model_name="hash", quantum=True)
        _fill(db)
        assert db.search("cardiac", top_k=1)[0].text.startswith("cardiac")
        assert db.stats()["quantum"] is True
        db.close()

    def test_search_batch_and_exclude(self, vault: VecForge) -> None:
        ids = _fill(vault)
        batches = vault.search_batch(["diabetes", "cardiac"], top_k=1)
        assert batches[0][0].text.startswith("patient") and batches[1][
            0
        ].text.startswith("cardiac")
        res = vault.search("diabetes", top_k=3, exclude_doc_ids=[ids[0]])
        assert ids[0] not in [r.doc_id for r in res]

    def test_similar(self, vault: VecForge) -> None:
        ids = _fill(vault)
        res = vault.similar(ids[0], top_k=2)
        assert ids[0] not in [r.doc_id for r in res]
        assert res[0].text.startswith("hip fracture")
        with pytest.raises(DocumentNotFoundError):
            vault.similar("missing")

    def test_empty_and_invalid(self, vault: VecForge) -> None:
        with pytest.raises(VaultEmptyError):
            vault.search("x")
        _fill(vault)
        assert vault.search("x", top_k=0) == []
        with pytest.raises(NamespaceNotFoundError):
            vault.search("x", namespace="nope")
        vault.create_namespace("empty")
        assert vault.search("x", namespace="empty") == []

    def test_precomputed_query_vector(self, vault: VecForge) -> None:
        ids = _fill(vault)
        vec = vault.get_embedding(ids[2])
        assert vec is not None
        res = vault.search("", query_vector=vec, alpha=1.0, top_k=1)
        assert res[0].doc_id == ids[2]


class TestMutations:
    def test_delete_tombstones_then_compacts(self, vault: VecForge) -> None:
        ids = _fill(vault)
        assert vault.delete(ids[1])
        assert not vault.delete(ids[1])
        assert not vault.exists(ids[1])
        assert vault.count() == 3
        assert vault.stats()["tombstones"] == 1
        assert ids[1] not in [r.doc_id for r in vault.search("elderly hip", top_k=4)]
        vault.compact()
        assert vault.stats()["tombstones"] == 0
        assert vault.count() == 3
        assert vault.search("cardiac", top_k=1)[0].text.startswith("cardiac")

    def test_delete_many_and_namespace(self, vault: VecForge) -> None:
        ids = _fill(vault)
        vault.add_batch(["tenant doc a", "tenant doc b"], namespace="t")
        assert vault.delete_many(ids[:2] + ["missing"]) == 2
        assert vault.count() == 4
        assert vault.delete_namespace("t") == 2
        assert "t" not in vault.list_namespaces()
        assert vault.count() == 2

    def test_auto_compaction(
        self, vault: VecForge, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import vecforge.core.vault as vault_mod

        monkeypatch.setattr(vault_mod, "_COMPACT_MIN_TOMBSTONES", 2)
        ids = vault.add_batch([f"doc {i}" for i in range(10)])
        vault.delete(ids[0])
        vault.delete(ids[1])
        assert vault.stats()["tombstones"] == 0  # 2 >= 20% of 10 → compacted
        assert vault.count() == 8

    def test_upsert_get_update(self, vault: VecForge) -> None:
        vault.upsert("first version", doc_id="note-1", metadata={"v": 1})
        vault.upsert("second version", doc_id="note-1", metadata={"v": 2})
        assert vault.count() == 1
        got = vault.get("note-1")
        assert (
            got is not None
            and got.text == "second version"
            and got.metadata == {"v": 2}
        )
        meta = vault.update_metadata("note-1", {"tag": "x"})
        assert meta == {"v": 2, "tag": "x"}
        assert vault.update_metadata("note-1", {"only": 1}, merge=False) == {"only": 1}
        with pytest.raises(DocumentNotFoundError):
            vault.update_metadata("missing", {})
        assert vault.get("missing") is None
        assert "note-1" in vault and len(vault) == 1

    def test_duplicate_id_raises(self, vault: VecForge) -> None:
        vault.add("a", doc_id="x")
        with pytest.raises(DuplicateDocumentError):
            vault.add("b", doc_id="x")
        with pytest.raises(DuplicateDocumentError):
            vault.add_batch(["c", "d"], doc_ids=["y", "x"])
        assert vault.count() == 1

    def test_deletion_protection_covers_upsert(self) -> None:
        db = VecForge(":memory:", model_name="hash", deletion_protection=True)
        db.add("v1", doc_id="k")
        with pytest.raises(DeletionProtectedError):
            db.upsert("v2", doc_id="k")
        with pytest.raises(DeletionProtectedError):
            db.delete_many(["k"])
        with pytest.raises(DeletionProtectedError):
            db.delete_namespace("default")
        db.close()

    def test_add_with_precomputed_embedding(self, vault: VecForge) -> None:
        vec = HashEmbedder().embed("hello world")
        doc_id = vault.add("hello world", embedding=vec * 3.0, modality="image")
        stored = vault.get_embedding(doc_id)
        assert stored is not None
        np.testing.assert_allclose(stored, vec, atol=1e-6)
        assert vault.get(doc_id).modality == "image"  # type: ignore[union-attr]
        with pytest.raises(ValueError, match="dimension"):
            vault.add("bad", embedding=np.ones(3, dtype=np.float32))

    def test_batch_validation(self, vault: VecForge) -> None:
        assert vault.add_batch([]) == []
        with pytest.raises(ValueError):
            vault.add_batch(["a"], metadata_list=[{}, {}])
        with pytest.raises(ValueError):
            vault.add_batch(["a"], doc_ids=["x", "y"])

    def test_ingest_uses_batches(self, vault: VecForge, tmp_path: Path) -> None:
        for i in range(5):
            (tmp_path / f"f{i}.txt").write_text(f"file {i} content about vectors")
        assert vault.ingest(str(tmp_path), batch_size=2) == 5
        assert vault.count() == 5

    def test_thread_safety_smoke(self, vault: VecForge) -> None:
        errors: list[BaseException] = []

        def worker(k: int) -> None:
            try:
                for i in range(20):
                    vault.add(f"thread {k} doc {i}")
                    vault.search("thread doc", top_k=3)
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert vault.count() == 80


class TestPersistence:
    def test_snapshot_fast_path_and_rebuild(self, tmp_path: Path) -> None:
        path = str(tmp_path / "v.db")
        db = VecForge(path, model_name="hash")
        ids = _fill(db)
        db.add_batch(["tenant doc"], namespace="t")
        db.delete(ids[3])
        db.close()

        reopened = VecForge(path, model_name="hash")
        assert reopened.count() == 4
        assert reopened.stats()["tombstones"] == 1
        assert not reopened.exists(ids[3])
        assert reopened.search("diabetes", top_k=1)[0].doc_id == ids[0]
        assert reopened.search("doc", namespace="t", top_k=1)[0].namespace == "t"
        reopened.close()

    def test_stale_snapshot_triggers_rebuild(self, tmp_path: Path) -> None:
        path = str(tmp_path / "v.db")
        db = VecForge(path, model_name="hash")
        _fill(db)
        db.close()
        # mutate behind the vault's back → generation changes → snapshot stale
        from vecforge.core.storage import StorageBackend

        raw = StorageBackend(path)
        raw.insert_doc(
            "sneaky doc about tennis",
            HashEmbedder().embed("sneaky doc about tennis"),
            {},
        )
        raw.close()
        reopened = VecForge(path, model_name="hash")
        assert reopened.count() == 5
        assert reopened.search("tennis", top_k=1)[0].text.startswith("sneaky")
        reopened.close()

    def test_legacy_vault_without_snapshot_rebuilds(self, tmp_path: Path) -> None:
        path = str(tmp_path / "v.db")
        db = VecForge(path, model_name="hash")
        _fill(db)
        db.storage.clear_index_state()
        db.close()
        from vecforge.core.storage import StorageBackend

        raw = StorageBackend(path)
        raw.clear_index_state()
        raw.close()
        reopened = VecForge(path, model_name="hash")
        assert reopened.count() == 4
        assert reopened.search("cardiac", top_k=1)[0].text.startswith("cardiac")
        reopened.close()

    def test_auto_save_every(self, tmp_path: Path) -> None:
        path = str(tmp_path / "v.db")
        db = VecForge(path, model_name="hash", auto_save_every=2)
        db.add("one")
        assert db.storage.load_index_state() is None
        db.add("two")
        state = db.storage.load_index_state()
        assert state is not None and state.doc_count == 2
        db.close()

    def test_backup_and_closed_errors(self, tmp_path: Path) -> None:
        path = str(tmp_path / "v.db")
        db = VecForge(path, model_name="hash")
        _fill(db)
        dest = str(tmp_path / "backup.db")
        db.backup(dest)
        copy = VecForge(dest, model_name="hash")
        assert copy.count() == 4
        copy.close()
        db.close()
        db.close()  # idempotent
        assert db.closed
        assert "closed" in repr(db)
        with pytest.raises(VaultClosedError):
            db.search("x")
        with pytest.raises(VaultClosedError):
            db.add("x")

    def test_reembed_with_new_model(self, tmp_path: Path) -> None:
        path = str(tmp_path / "v.db")
        db = VecForge(path, model_name="hash:64")
        _fill(db)
        db.close()
        db2 = VecForge(path, model_name="hash:128")
        assert db2.stats()["dimension"] == 64  # stored vectors reused
        assert db2.reembed() == 4
        assert db2.stats()["dimension"] == 128
        assert db2.search("diabetes", top_k=1)[0].text.startswith("patient")
        db2.close()

    def test_stats_fields(self, vault: VecForge) -> None:
        _fill(vault)
        st = vault.stats()
        assert st["version"]
        assert st["index_type"] == "flat"
        assert st["embedding_model"] == "hash:384"
        assert st["schema_version"] == 2
        assert st["namespace_counts"]["default"] == 4


class TestApiKeys:
    def test_register_revoke_persist(self, tmp_path: Path) -> None:
        from vecforge.exceptions import VecForgePermissionError

        path = str(tmp_path / "keys.db")
        admin = VecForge(path, model_name="hash")
        admin.add("secret doc")
        admin.register_api_key("reader-key", "read-only")
        assert list(admin.list_api_keys().values()) == ["read-only"]
        admin.close()

        reader = VecForge(path, model_name="hash", api_key="reader-key")
        assert reader.search("secret", top_k=1)[0].text == "secret doc"
        with pytest.raises(VecForgePermissionError):
            reader.add("nope")
        with pytest.raises(VecForgePermissionError):
            reader.list_api_keys()
        reader.close()

        stranger = VecForge(path, model_name="hash", api_key="not-registered")
        with pytest.raises(VecForgePermissionError):
            stranger.search("secret")
        stranger.close()

        admin = VecForge(path, model_name="hash")
        assert admin.revoke_api_key("reader-key")
        assert not admin.revoke_api_key("reader-key")
        with pytest.raises(ValueError):
            admin.register_api_key("k", "superuser")
        admin.close()


class TestScaleSafety:
    def test_index_snapshot_is_chunked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import vecforge.core.storage as storage_mod

        monkeypatch.setattr(storage_mod, "_BLOB_CHUNK_BYTES", 1024)
        path = str(tmp_path / "chunks.db")
        db = VecForge(path, model_name="hash")
        db.add_batch([f"chunked document {i}" for i in range(50)])
        db.close()
        from vecforge.core.storage import StorageBackend

        raw = StorageBackend(path)
        conn = raw._conn
        assert conn is not None
        n_chunks = conn.execute("SELECT COUNT(*) FROM index_chunks").fetchone()[0]
        assert n_chunks > 2
        raw.close()
        reopened = VecForge(path, model_name="hash")
        assert reopened.count() == 50
        assert reopened.search("chunked document 7", top_k=1)[0].text.endswith(" 7")
        reopened.close()

    def test_mixed_dimensions_are_skipped_not_fatal(self, tmp_path: Path) -> None:
        path = str(tmp_path / "mixed.db")
        db = VecForge(path, model_name="hash:64")
        db.add("sixty four dims")
        db.storage.clear_index_state()
        db.close()
        from vecforge.core.storage import StorageBackend

        raw = StorageBackend(path)
        raw.insert_doc(
            "thirty two dims", HashEmbedder(dimension=32).embed("thirty two dims"), {}
        )
        raw.clear_index_state()
        raw.close()
        reopened = VecForge(path, model_name="hash:64")
        assert reopened.count() == 1  # the odd row is left out, not fatal
        assert reopened.search("sixty", top_k=1)[0].text == "sixty four dims"
        assert reopened.reembed() == 2
        assert reopened.count() == 2
        reopened.close()


class TestRerankInteractions:
    class _FakeReranker:
        def rerank(self, query: str, texts: list[str], top_k: int | None = None):
            ranked = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            out = [(i, float(100 - len(texts[i]))) for i in ranked]
            return out[:top_k] if top_k else out

    def test_recency_applies_after_cross_encoder(self, vault: VecForge) -> None:
        _fill(vault)
        vault._reranker = self._FakeReranker()  # type: ignore[assignment]
        plain = vault.search("patient", rerank=True, top_k=4)
        assert plain[0].text == min((t for t, _ in DOCS), key=len)
        recent = vault.search("patient", rerank=True, recency_weight=1.0, top_k=4)
        times = [r.timestamp for r in recent]
        assert times == sorted(times, reverse=True)
        assert all(0.0 <= r.score <= 1.0 for r in recent)

    def test_quantum_with_cross_encoder_uses_survivors(self, vault: VecForge) -> None:
        _fill(vault)
        vault._reranker = self._FakeReranker()  # type: ignore[assignment]
        res = vault.search("type 2 diabetes", rerank=True, quantum_rerank=True, top_k=2)
        assert len(res) == 2
        assert all(r.score > 0 for r in res)


class TestSnapshotManagerWithRealVault:
    def test_snapshot_of_open_wal_vault_is_complete(self, tmp_path: Path) -> None:
        from vecforge.security.snapshots import SnapshotManager

        path = str(tmp_path / "live.db")
        db = VecForge(path, model_name="hash")
        db.add_batch([f"row {i}" for i in range(20)])  # still in the WAL
        snap = SnapshotManager(path).create_snapshot(str(tmp_path / "backups"))
        copy = VecForge(snap, model_name="hash")
        assert copy.count() == 20
        copy.close()
        db.close()
        SnapshotManager(path).restore_snapshot(snap)
        restored = VecForge(path, model_name="hash")
        assert restored.count() == 20
        restored.close()
