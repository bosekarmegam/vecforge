# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under BSL 1.1 — see LICENSE for details.

"""Tests for the schema-v2 storage backend."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest

from vecforge.core.storage import (
    IndexState,
    SessionRecord,
    StorageBackend,
    content_hash,
)


def _vec(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(8).astype(np.float32)


class TestBatchInserts:
    def test_insert_docs_batch(self) -> None:
        s = StorageBackend(":memory:")
        docs = s.insert_docs(["a", "b", "c"], np.stack([_vec(i) for i in range(3)]))
        assert len(docs) == 3
        assert s.count_docs() == 3
        assert all(d.content_hash for d in docs)
        s.close()

    def test_batch_is_atomic_on_duplicate(self) -> None:
        s = StorageBackend(":memory:")
        s.insert_doc("first", _vec(), {}, doc_id="dup")
        with pytest.raises(sqlite3.IntegrityError):
            s.insert_docs(
                ["x", "y"], np.stack([_vec(1), _vec(2)]), doc_ids=["new", "dup"]
            )
        assert s.count_docs() == 1
        assert s.get_doc("new") is None
        s.close()

    def test_mismatched_lengths_raise(self) -> None:
        s = StorageBackend(":memory:")
        with pytest.raises(ValueError):
            s.insert_docs(["a", "b"], np.stack([_vec()]))
        with pytest.raises(ValueError):
            s.insert_docs(["a"], np.stack([_vec()]), metadatas=[{}, {}])
        s.close()

    def test_generation_increments_on_mutation(self) -> None:
        s = StorageBackend(":memory:")
        g0 = s.generation
        doc = s.insert_doc("a", _vec(), {})
        assert s.generation == g0 + 1
        # metadata is read from SQLite at query time: no snapshot invalidation
        s.update_doc(doc.doc_id, metadata={"k": 1})
        assert s.generation == g0 + 1
        s.update_doc(doc.doc_id, text="changed")
        assert s.generation == g0 + 2
        s.delete_doc(doc.doc_id)
        assert s.generation == g0 + 3
        s.delete_doc("missing")
        assert s.generation == g0 + 3
        s.close()


class TestReads:
    def test_get_docs_batched(self) -> None:
        s = StorageBackend(":memory:")
        n = 1200  # > one IN chunk
        docs = s.insert_docs(
            [f"doc {i}" for i in range(n)], np.stack([_vec(i) for i in range(n)])
        )
        ids = [d.doc_id for d in docs]
        found = s.get_docs(ids + ["missing"])
        assert len(found) == n
        assert found[ids[7]].text == "doc 7"
        s.close()

    def test_iter_docs_order_and_namespace(self) -> None:
        s = StorageBackend(":memory:")
        s.insert_doc("a", _vec(), {}, namespace="x")
        s.insert_doc("b", _vec(), {}, namespace="y")
        s.insert_doc("c", _vec(), {}, namespace="x")
        assert [d.text for d in s.iter_docs(batch_size=1)] == ["a", "b", "c"]
        assert [d.text for d in s.iter_docs(namespace="x")] == ["a", "c"]
        assert s.count_by_namespace() == {"default": 0, "x": 2, "y": 1}
        s.close()

    def test_find_by_content_hash(self) -> None:
        s = StorageBackend(":memory:")
        s.insert_doc("same text", _vec(), {}, namespace="n")
        hit = s.find_by_content_hash("n", content_hash("n", "same text"))
        assert hit is not None and hit.text == "same text"
        assert (
            s.find_by_content_hash("other", content_hash("other", "same text")) is None
        )
        s.close()


class TestUpdatesAndDeletes:
    def test_update_doc_fields(self) -> None:
        s = StorageBackend(":memory:")
        doc = s.insert_doc("old", _vec(), {"a": 1}, namespace="n")
        assert s.update_doc(
            doc.doc_id, text="new", metadata={"b": 2}, modality="memory"
        )
        got = s.get_doc(doc.doc_id)
        assert got is not None
        assert got.text == "new"
        assert got.metadata == {"b": 2}
        assert got.modality == "memory"
        assert got.updated_at is not None
        assert got.content_hash == content_hash("n", "new")
        assert not s.update_doc("missing", text="x")
        assert not s.update_doc(doc.doc_id)
        s.close()

    def test_delete_many_and_namespace(self) -> None:
        s = StorageBackend(":memory:")
        a = s.insert_doc("a", _vec(), {}, namespace="n1")
        b = s.insert_doc("b", _vec(), {}, namespace="n1")
        s.insert_doc("c", _vec(), {}, namespace="n2")
        s.create_namespace("n2")
        assert s.delete_docs([a.doc_id, b.doc_id, "missing"]) == 2
        assert s.delete_namespace("n2") == 1
        assert "n2" not in s.list_namespaces()
        assert s.delete_namespace("default") == 0
        assert "default" in s.list_namespaces()
        s.close()


class TestIndexState:
    def test_roundtrip(self) -> None:
        s = StorageBackend(":memory:")
        assert s.load_index_state() is None
        state = IndexState(
            faiss_blob=b"faiss",
            bm25_blob=b"bm25",
            doc_ids=["a", "b", "c"],
            doc_namespaces=["default", "x", "x"],
            tombstones=[1],
            dimension=8,
            generation=3,
            doc_count=2,
            model_name="hash:8",
            index_type="flat",
        )
        s.save_index_state(state)
        loaded = s.load_index_state()
        assert loaded is not None
        assert loaded.doc_ids == ["a", "b", "c"]
        assert loaded.doc_namespaces == ["default", "x", "x"]
        assert loaded.tombstones == [1]
        assert loaded.model_name == "hash:8"
        assert loaded.faiss_blob == b"faiss" and loaded.bm25_blob == b"bm25"
        s.clear_index_state()
        assert s.load_index_state() is None
        s.close()


class TestSessionsAndKV:
    def test_sessions(self) -> None:
        s = StorageBackend(":memory:")
        rec = SessionRecord(session_id="s1", agent="claude", started_at=time.time())
        s.upsert_session(rec)
        s.increment_session_turns("s1", 2)
        got = s.get_session("s1")
        assert got is not None and got.turn_count == 2
        rec.summary = "done"
        rec.ended_at = time.time()
        s.upsert_session(rec)
        assert s.get_session("s1").summary == "done"  # type: ignore[union-attr]
        s.upsert_session(
            SessionRecord(session_id="s2", agent="other", started_at=time.time() + 1)
        )
        assert [r.session_id for r in s.list_sessions()] == ["s2", "s1"]
        assert [r.session_id for r in s.list_sessions(agent="claude")] == ["s1"]
        assert s.delete_session("s1") and not s.delete_session("s1")
        s.close()

    def test_kv(self) -> None:
        s = StorageBackend(":memory:")
        s.kv_set("a:1", {"x": [1, 2]})
        s.kv_set("a:2", "str")
        s.kv_set("b", 3)
        assert s.kv_get("a:1") == {"x": [1, 2]}
        assert s.kv_get("missing", "dflt") == "dflt"
        assert s.kv_keys("a:") == ["a:1", "a:2"]
        assert s.kv_delete("b") and not s.kv_delete("b")
        with pytest.raises(ValueError):
            s.kv_set("__doc_generation__", 1)
        s.close()


class TestMaintenanceAndSecurity:
    def test_backup_is_consistent_copy(self, tmp_path: Path) -> None:
        src = StorageBackend(str(tmp_path / "src.db"))
        src.insert_doc("hello", _vec(), {})
        dest = str(tmp_path / "copy.db")
        src.backup(dest)
        src.close()
        copy = StorageBackend(dest)
        assert copy.count_docs() == 1
        copy.close()

    def test_vacuum_runs(self, tmp_path: Path) -> None:
        s = StorageBackend(str(tmp_path / "v.db"))
        d = s.insert_doc("x", _vec(), {})
        s.delete_doc(d.doc_id)
        s.vacuum()
        s.close()
        assert not s.is_open

    def test_pragma_key_escaping(self) -> None:
        assert StorageBackend._escape_pragma_key("pa'ss") == "pa''ss"
        assert StorageBackend._escape_pragma_key("plain") == "plain"

    def test_v1_schema_migrates_in_place(self, tmp_path: Path) -> None:
        path = str(tmp_path / "v1.db")
        conn = sqlite3.connect(path)
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
            INSERT INTO schema_version VALUES (1);
            CREATE TABLE documents (
                doc_id TEXT PRIMARY KEY, text TEXT NOT NULL, embedding BLOB NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                namespace TEXT NOT NULL DEFAULT 'default',
                modality TEXT NOT NULL DEFAULT 'text',
                created_at REAL NOT NULL, updated_at REAL);
            CREATE TABLE namespaces (name TEXT PRIMARY KEY, created_at REAL NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}');
            INSERT INTO namespaces VALUES ('default', 1.0, '{}');
            """)
        conn.execute(
            "INSERT INTO documents (doc_id, text, embedding, metadata_json, namespace,"
            " modality, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("d1", "legacy doc", _vec().tobytes(), '{"k": 1}', "default", "text", 1.0),
        )
        conn.commit()
        conn.close()

        s = StorageBackend(path)
        assert s.schema_version == 2
        doc = s.get_doc("d1")
        assert doc is not None
        assert doc.content_hash == content_hash("default", "legacy doc")
        assert s.find_by_content_hash("default", doc.content_hash) is not None
        assert s.generation >= 0
        s.close()
