# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under BSL 1.1 — see LICENSE for details.

"""Regression tests for issues found by the 2.0 pre-release code review.

Each test reproduces a bug that was confirmed against the first 2.0 build.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from vecforge import MemoryStore, VecForge
from vecforge.cli.main import cli
from vecforge.core.embedder import HashEmbedder
from vecforge.core.storage import StorageBackend
from vecforge.exceptions import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    VaultClosedError,
    VecForgePermissionError,
)
from vecforge.quantum.amplitude_encoder import AmplitudeEncoder
from vecforge.quantum.grover_amplifier import GroverAmplifier
from vecforge.quantum.reranker import QuantumReranker
from vecforge.security.snapshots import SnapshotManager
from vecforge.server.app import create_app

# ─── Core: atomicity and dimensions ───


class TestAtomicWrites:
    def test_wrong_dimension_leaves_no_orphan_row(self, tmp_path: Path) -> None:
        path = str(tmp_path / "v.db")
        db = VecForge(path, model_name="hash")
        with pytest.raises(ValueError, match="dimension"):
            db.add("clip vector doc", embedding=np.ones(512, dtype=np.float32))
        db.add("normal doc")
        db.close()
        raw = sqlite3.connect(path)
        assert raw.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        raw.close()
        again = VecForge(path, model_name="hash")
        assert again.stats()["dimension"] == 384
        assert again.search("normal doc", top_k=1)[0].text == "normal doc"
        again.close()

    def test_failed_upsert_keeps_original(self) -> None:
        db = VecForge(":memory:", model_name="hash")
        db.add("original text", doc_id="k1")
        with pytest.raises(ValueError):
            db.upsert("new text", doc_id="k1", embedding=np.ones(7, dtype=np.float32))
        got = db.get("k1")
        assert got is not None and got.text == "original text"
        assert db.search("original", top_k=1)[0].doc_id == "k1"
        db.close()

    def test_index_failure_rolls_back_storage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = VecForge(":memory:", model_name="hash")
        db.add("first")

        def boom(*a: object, **k: object) -> None:
            raise RuntimeError("index exploded")

        monkeypatch.setattr(db, "_index_documents", boom)
        with pytest.raises(RuntimeError):
            db.add_batch(["second", "third"])
        monkeypatch.undo()
        assert db.count() == 1
        assert db._storage.count_docs() == 1
        db.close()

    def test_auto_save_failure_does_not_fail_the_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = VecForge(str(tmp_path / "a.db"), model_name="hash", auto_save_every=1)

        def broken_save() -> None:
            raise OSError("disk full")

        monkeypatch.setattr(db, "_save", broken_save)
        doc_id = db.add("second doc", doc_id="s2")
        assert doc_id == "s2" and db.exists("s2")
        monkeypatch.undo()
        with pytest.raises(DuplicateDocumentError):
            db.add("again", doc_id="s2")
        db.close()

    def test_failed_statement_does_not_poison_next_transaction(
        self, tmp_path: Path
    ) -> None:
        path = str(tmp_path / "t.db")
        db = VecForge(path, model_name="hash")
        doc = db.add("x")
        blocker = sqlite3.connect(path, timeout=0.05, isolation_level=None)
        blocker.execute("BEGIN IMMEDIATE")
        db._storage._conn.execute("PRAGMA busy_timeout = 50")  # type: ignore[union-attr]
        with pytest.raises(sqlite3.OperationalError):
            db.update_metadata(doc, {"k": 1})
        blocker.execute("ROLLBACK")
        blocker.close()
        db.add("after the failure")  # must not raise "transaction within a transaction"
        assert db.count() == 2
        db.close()

    def test_metadata_edit_keeps_snapshot_valid(self, tmp_path: Path) -> None:
        path = str(tmp_path / "m.db")
        db = VecForge(path, model_name="hash")
        doc = db.add("doc")
        db.close()
        db = VecForge(path, model_name="hash")
        gen = db._storage.generation
        db.update_metadata(doc, {"seen": True})
        assert db._storage.generation == gen
        assert db._mutations_since_save == 0
        db.close()


class TestSearchEdges:
    def test_exclude_ids_outside_scope_do_not_shrink_pool(self) -> None:
        db = VecForge(":memory:", model_name="hash")
        db.add("alpha one", namespace="a")
        b = db.add("alpha two", namespace="b")
        assert len(db.similar(b, namespace="a")) == 1
        assert len(db.search("alpha", namespace="a", exclude_doc_ids=[b])) == 1
        db.add_batch([f"alpha {i}" for i in range(9)], namespace="a")
        res = db.search("alpha", namespace="a", top_k=10, exclude_doc_ids=["nope"] * 8)
        assert len(res) == 10
        db.close()

    def test_delete_many_accepts_generators(self) -> None:
        db = VecForge(":memory:", model_name="hash")
        ids = db.add_batch(["a", "b", "c"])
        assert db.delete_many(i for i in ids[:2]) == 2
        assert db.count() == 1 and not db.exists(ids[0])
        db.close()

    def test_recency_with_identical_timestamps(self) -> None:
        db = VecForge(":memory:", model_name="hash")
        db.add_batch(["red apple", "green apple", "blue sky"])
        res = db.search("apple", top_k=3, quantum_rerank=True, recency_weight=1.0)
        assert all(r.score > 0 for r in res)
        db.close()


class TestQuantumEdgeInputs:
    def test_nan_and_inf(self) -> None:
        out = GroverAmplifier().amplify(np.array([np.nan, 0.5, 0.2]))
        assert np.isfinite(out).all()
        out = GroverAmplifier().amplify(np.array([np.inf, 0.5, 0.2]))
        assert np.isfinite(out).all() and int(np.argmax(out)) == 0
        res = QuantumReranker().rerank(
            "q", ["a", "b", "c"], [np.inf, 0.5, 0.2], top_k=3
        )
        assert res[0][0] == 0
        enc = AmplitudeEncoder().encode(np.array([np.inf, 0.5]))
        assert enc[0] > enc[1]


# ─── Security ───


class TestRbacCoverage:
    def test_unregistered_key_cannot_read_metadata_or_storage(
        self, tmp_path: Path
    ) -> None:
        path = str(tmp_path / "r.db")
        admin = VecForge(path, model_name="hash")
        doc = admin.add("secret", namespace="ward7")
        admin.close()
        db = VecForge(path, model_name="hash", api_key="intruder")
        for call in (
            lambda: db.count(),
            lambda: db.exists(doc),
            lambda: db.list_namespaces(),
            lambda: db.stats(),
            lambda: db.exists_namespace("ward7"),
            lambda: db.similar("missing"),
            lambda: db.compact(),
            lambda: db.save(),
            lambda: db.storage,
        ):
            with pytest.raises(VecForgePermissionError):
                call()
        db.close()  # close still works for any key

    def test_reserved_kv_keys_are_protected(self) -> None:
        s = StorageBackend(":memory:")
        with pytest.raises(ValueError, match="reserved"):
            s.kv_set("__rbac_keys__", {"x": "admin"})
        assert not s.kv_delete("__rbac_keys__")
        s.kv_set("user_1", 1)
        s.kv_set("userX1", 2)
        s.kv_set("50%", 3)
        s.kv_set("Abc", 4)
        assert s.kv_keys("user_") == ["user_1"]
        assert s.kv_keys("50%") == ["50%"]
        assert s.kv_keys("abc") == []
        s.close()

    def test_snapshot_path_with_special_characters(self, tmp_path: Path) -> None:
        # '?' is not allowed in Windows file names; '#', '%' and spaces are
        name = "my#vault 100%.db" if os.name == "nt" else "my#vault? 100%.db"
        path = tmp_path / name
        db = VecForge(str(path), model_name="hash")
        db.add("hello")
        snap = SnapshotManager(str(path)).create_snapshot(str(tmp_path / "bk"))
        db.close()
        copy = sqlite3.connect(snap)
        assert copy.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        copy.close()
        assert not (tmp_path / "my").exists()


class TestEncryptedVault:
    def test_sqlcipher_round_trip_and_encrypted_backup(self, tmp_path: Path) -> None:
        pytest.importorskip("sqlcipher3")
        path = str(tmp_path / "enc.db")
        key = "correct horse 'battery' staple"
        db = VecForge(path, model_name="hash", encryption_key=key)
        assert db.stats()["encrypted"]
        db.add("top secret", doc_id="s1")
        with pytest.raises(DuplicateDocumentError):
            db.add("again", doc_id="s1")
        dest = str(tmp_path / "enc-backup.db")
        db.backup(dest)
        db.close()
        assert b"top secret" not in Path(path).read_bytes()
        assert not Path(dest).read_bytes().startswith(b"SQLite format 3")
        again = VecForge(dest, model_name="hash", encryption_key=key)
        assert again.get("s1").text == "top secret"  # type: ignore[union-attr]
        again.close()


# ─── Memory layer ───


class TestMemoryIsolationAndConcurrency:
    def test_agents_cannot_touch_each_others_memories(self) -> None:
        vault = VecForge(":memory:", model_name="hash")
        a = MemoryStore(vault=vault, agent="a")
        b = MemoryStore(vault=vault, agent="b")
        m = a.remember("alice's secret plan", kind="fact")
        plain = vault.add("plain document")
        with pytest.raises(DocumentNotFoundError):
            b.update(m.memory_id, content="hijacked")
        assert b.forget(m.memory_id) is False
        assert b.forget_many([m.memory_id]) == 0
        assert a.forget(plain) is False
        assert a.get(m.memory_id).content == "alice's secret plan"  # type: ignore[union-attr]
        s = a.start_session()
        a.end_session("alice summary")
        assert b.end_session("bob closes it", session_id=s.session_id) is None
        assert b.get_session(s.session_id) is None
        with pytest.raises(ValueError, match="belongs to agent"):
            b.remember("x", session_id=s.session_id)
        assert a.get_session(s.session_id).summary == "alice summary"  # type: ignore[union-attr]
        vault.close()

    def test_concurrent_stores_on_one_vault(self, tmp_path: Path) -> None:
        vault = VecForge(str(tmp_path / "c.db"), model_name="hash")
        stores = [MemoryStore(vault=vault, agent=f"agent{i}") for i in range(2)]
        errors: list[BaseException] = []

        def work(store: MemoryStore, k: int) -> None:
            try:
                for i in range(40):
                    store.remember(f"agent {k} fact number {i}", dedup=False)
                    store.recall("fact number", top_k=3)
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        threads = [
            threading.Thread(target=work, args=(st, k))
            for k, st in enumerate(stores)
            for _ in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert vault.count() == vault._storage.count_docs() == 160
        vault.close()

    def test_reinforce_counts_are_not_lost(self) -> None:
        vault = VecForge(":memory:", model_name="hash")
        s1 = MemoryStore(vault=vault, agent="x")
        s2 = MemoryStore(vault=vault, agent="x")
        m = s1.remember("shared")

        def bump(store: MemoryStore) -> None:
            for _ in range(50):
                store.reinforce(m.memory_id, delta=0.0)

        ts = [threading.Thread(target=bump, args=(s,)) for s in (s1, s2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert s1.get(m.memory_id).access_count == 100  # type: ignore[union-attr]
        vault.close()

    def test_closed_vault_raises_cleanly(self) -> None:
        vault = VecForge(":memory:", model_name="hash")
        store = MemoryStore(vault=vault, agent="z")
        vault.close()
        with pytest.raises(VaultClosedError):
            store.remember("x")


class TestMemorySemantics:
    def test_import_restores_sessions_and_timestamps(self, tmp_path: Path) -> None:
        src = MemoryStore(":memory:", agent="a", model_name="hash")
        s = src.start_session(metadata={"name": "first"})
        old = src.remember("set up the schema", kind="episode")
        src.end_session("Set up the database schema")
        out = tmp_path / "m.json"
        src.export_json(str(out))
        dst = MemoryStore(":memory:", agent="a", model_name="hash")
        dst.import_json(str(out))
        sess = dst.get_session(s.session_id)
        assert sess is not None and not sess.is_open
        assert sess.summary == "Set up the database schema"
        assert sess.metadata == {"name": "first"}
        restored = [m for m in dst.list_memories() if m.content == old.content][0]
        assert restored.created_at == pytest.approx(old.created_at)
        assert "Last session" in dst.context_pack()
        src.close()
        dst.close()

    def test_exact_dedup_checks_every_hash_match(self) -> None:
        mem = MemoryStore(":memory:", agent="a", model_name="hash", dedup_threshold=1.0)
        mem.remember("same text", kind="fact")
        mem.remember("same text", kind="note")
        again = mem.remember("same text", kind="note")
        assert again.deduplicated
        assert mem.stats()["memories"] == 2
        mem.close()

    def test_featureless_texts_are_distinct(self) -> None:
        emb = HashEmbedder()
        assert float(emb.embed("🎉🎉") @ emb.embed("🔥")) < 0.9
        mem = MemoryStore(":memory:", agent="a", model_name="hash")
        mem.remember("🎉🎉")
        assert not mem.remember("🔥").deduplicated
        mem.close()

    def test_updated_at_tracks_edits_not_reads(self) -> None:
        mem = MemoryStore(":memory:", agent="a", model_name="hash")
        m = mem.remember("fresh fact")
        mem.recall("fresh fact")
        assert mem.get(m.memory_id).updated_at is None  # type: ignore[union-attr]
        edited = mem.update(m.memory_id, content="edited fact")
        assert edited.updated_at is not None
        raw = mem.vault.get(m.memory_id)
        assert raw is not None and "updated_at_ts" not in raw.metadata
        mem.close()

    def test_session_lifecycle_is_idempotent(self) -> None:
        mem = MemoryStore(":memory:", agent="a", model_name="hash")
        s = mem.start_session(session_id="S1")
        mem.remember("turn one")
        mem.end_session("first summary")
        assert mem.end_session(session_id="S1").summary == "first summary"  # type: ignore[union-attr]
        resumed = mem.start_session(session_id="S1")
        assert resumed.summary == "first summary" and resumed.turn_count >= 2
        assert s.session_id == "S1"
        mem.close()

    def test_bounds(self) -> None:
        mem = MemoryStore(":memory:", agent="a", model_name="hash")
        for i in range(3):
            mem.remember(f"fact {i}", dedup=False)
        assert mem.recall("fact", top_k=-1) == []
        assert mem.recall("fact", top_k=0) == []
        assert mem.context_pack("fact", max_chars=-5) == ""
        mem.start_session()
        mem.end_session("the only summary")
        mem.start_session()
        assert mem.context_pack().count("the only summary") == 1
        mem.close()


# ─── REST / MCP / CLI ───


class TestServers:
    def test_rest_restart_and_unicode_key(self, tmp_path: Path) -> None:
        app = create_app(str(tmp_path / "r.db"), model_name="hash")
        with TestClient(app) as c:
            assert (
                c.post("/api/v1/memory/remember", json={"content": "x"}).status_code
                == 200
            )
        app.state.vault = None  # what a lifespan-aware host does on restart
        with TestClient(app) as c:
            r = c.post("/api/v1/memory/remember", json={"content": "after restart"})
            assert r.status_code == 200
        secure = create_app(str(tmp_path / "s.db"), model_name="hash", api_key="sécret")
        with TestClient(secure) as c:
            assert (
                c.get(
                    "/api/v1/health",
                    headers=[(b"X-API-Key", "s\xe9cret".encode())],
                ).status_code
                == 401
            )

    def test_mcp_errors_are_readable_and_since_zero_filters(self) -> None:
        pytest.importorskip("mcp")
        import asyncio

        from vecforge.mcp.server import create_server

        server, app = create_server(
            vault_path=":memory:", agent="t", model_name="hash", warmup=False
        )
        try:
            with pytest.raises(Exception, match="does not exist"):
                asyncio.run(server.call_tool("reinforce", {"memory_id": "nope"}))
            app.remember("old fact")
            assert app.recall("fact", since_days=0)["count"] == 0
            assert app.recall("fact", since_days=None)["count"] == 1
        finally:
            app.close()

    def test_print_config_includes_model(self) -> None:
        res = CliRunner().invoke(
            cli,
            [
                "mcp",
                "--print-config",
                "claude-desktop",
                "--vault",
                "m.db",
                "--model",
                "hash",
            ],
        )
        args = json.loads(res.output)["mcpServers"]["vecforge-memory"]["args"]
        assert args[-2:] == ["--model", "hash"]

    def test_cli_errors_are_friendly_and_paths_must_exist(self, tmp_path: Path) -> None:
        runner = CliRunner()
        empty = str(tmp_path / "empty.db")
        VecForge(empty, model_name="hash").close()
        res = runner.invoke(cli, ["search", "x", "--vault", empty, "--model", "hash"])
        assert res.exit_code == 1 and "contains no documents" in res.output
        assert "Traceback" not in res.output
        res = runner.invoke(
            cli, ["memory", "remember", "   ", "--vault", empty, "--model", "hash"]
        )
        assert res.exit_code == 1 and "must not be empty" in res.output
        for cmd in (["stats"], ["export"], ["compact"]):
            missing = tmp_path / f"missing-{cmd[0]}.db"
            res = runner.invoke(cli, [*cmd, str(missing)])
            assert res.exit_code == 2 and not missing.exists()
        res = runner.invoke(
            cli, ["backup", str(tmp_path / "nope.db"), str(tmp_path / "bk.db")]
        )
        assert res.exit_code == 2 and not (tmp_path / "nope.db").exists()
