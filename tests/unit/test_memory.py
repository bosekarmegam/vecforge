# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under BSL 1.1 — see LICENSE for details.

"""Tests for the AI memory layer."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from vecforge import VecForge
from vecforge.exceptions import DocumentNotFoundError
from vecforge.memory import KINDS, Memory, MemoryStore, Session


@pytest.fixture
def mem() -> MemoryStore:
    store = MemoryStore(":memory:", agent="claude", model_name="hash")
    yield store
    store.close()


class TestRememberRecall:
    def test_remember_and_recall(self, mem: MemoryStore) -> None:
        m = mem.remember(
            "User prefers concise answers",
            kind="preference",
            importance=0.9,
            tags=["style"],
        )
        assert isinstance(m, Memory)
        assert m.memory_id.startswith("mem_")
        assert m.kind == "preference" and m.tags == ["style"]
        mem.remember("Deploys happen on Fridays", kind="fact", tags=["ops"])
        hits = mem.recall("how should I write answers?", top_k=1)
        assert hits[0].content == "User prefers concise answers"
        assert hits[0].score > 0
        assert hits[0].access_count == 1 and hits[0].last_accessed is not None

    def test_unknown_kind_becomes_note_and_importance_clamped(
        self, mem: MemoryStore
    ) -> None:
        m = mem.remember("x", kind="weird", importance=7)
        assert m.kind == "note" and m.importance == 1.0
        assert "note" in KINDS

    def test_empty_content_rejected(self, mem: MemoryStore) -> None:
        with pytest.raises(ValueError):
            mem.remember("   ")

    def test_exact_duplicate_is_reinforced(self, mem: MemoryStore) -> None:
        a = mem.remember("The repo uses ruff", kind="fact", importance=0.4)
        b = mem.remember(
            "The repo uses ruff", kind="fact", importance=0.8, tags=["tooling"]
        )
        assert b.deduplicated and b.memory_id == a.memory_id
        assert b.importance >= 0.8 and "tooling" in b.tags
        assert mem.stats()["memories"] == 1

    def test_near_duplicate_is_reinforced_same_kind_only(
        self, mem: MemoryStore
    ) -> None:
        a = mem.remember("the repo uses black and ruff for linting", kind="fact")
        b = mem.remember("The repo uses black and ruff for linting!", kind="fact")
        c = mem.remember("the repo uses black and ruff for linting", kind="preference")
        assert b.deduplicated and b.memory_id == a.memory_id
        assert not c.deduplicated

    def test_dedup_can_be_disabled(self, mem: MemoryStore) -> None:
        mem.remember("same", dedup=False)
        mem.remember("same", dedup=False)
        assert mem.stats()["memories"] == 2

    def test_recall_filters(self, mem: MemoryStore) -> None:
        mem.remember(
            "budget approved for Q3", kind="decision", tags=["finance"], importance=0.9
        )
        mem.remember(
            "budget spreadsheet lives in drive", kind="fact", tags=["finance", "docs"]
        )
        mem.remember("budget joke from standup", kind="episode", importance=0.1)
        assert all(m.kind == "fact" for m in mem.recall("budget", kinds=["fact"]))
        assert all("docs" in m.tags for m in mem.recall("budget", tags=["docs"]))
        assert all(
            m.importance >= 0.5 for m in mem.recall("budget", min_importance=0.5)
        )
        assert mem.recall("budget", since=time.time() + 100) == []
        assert mem.recall("budget", since=time.time() - 100)

    def test_recall_without_query_ranks_by_importance_and_recency(
        self, mem: MemoryStore
    ) -> None:
        mem.remember("low", importance=0.1)
        mem.remember("high", importance=0.95)
        res = mem.recall("", top_k=2)
        assert res[0].content == "high"

    def test_recall_empty_store(self, mem: MemoryStore) -> None:
        assert mem.recall("anything") == []
        assert mem.context_pack("anything") == ""

    def test_remember_many(self, mem: MemoryStore) -> None:
        out = mem.remember_many(
            ["a thing", {"content": "b thing", "kind": "task"}], kind="fact"
        )
        assert [m.kind for m in out] == ["fact", "task"]


class TestEditing:
    def test_reinforce_update_forget(self, mem: MemoryStore) -> None:
        m = mem.remember("original", importance=0.5)
        r = mem.reinforce(m.memory_id, delta=0.3)
        assert r.importance == pytest.approx(0.8) and r.access_count == 1
        u = mem.update(
            m.memory_id, content="edited", tags=["t"], kind="insight", importance=0.2
        )
        assert u.content == "edited" and u.tags == ["t"] and u.kind == "insight"
        assert u.importance == pytest.approx(0.2)
        assert mem.recall("edited", top_k=1)[0].memory_id == m.memory_id
        assert mem.forget(m.memory_id) and not mem.forget(m.memory_id)
        assert mem.get(m.memory_id) is None
        with pytest.raises(DocumentNotFoundError):
            mem.reinforce("mem_missing")
        with pytest.raises(DocumentNotFoundError):
            mem.update("mem_missing", content="x")

    def test_list_and_clear(self, mem: MemoryStore) -> None:
        mem.remember("a", kind="fact")
        mem.remember("b", kind="task")
        mem.remember("c", kind="task", tags=["x"])
        assert len(mem.list_memories()) == 3
        assert [
            m.content for m in mem.list_memories(kinds=["task"], newest_first=False)
        ] == ["b", "c"]
        assert [m.content for m in mem.list_memories(tags=["x"])] == ["c"]
        assert len(mem.list(limit=2)) == 2
        assert mem.clear(kind="task") == 2
        assert mem.clear() == 1
        assert mem.stats()["memories"] == 0

    def test_get_ignores_other_namespaces(self, mem: MemoryStore) -> None:
        doc_id = mem.vault.add("plain document")
        assert mem.get(doc_id) is None


class TestSessions:
    def test_session_lifecycle_and_context_pack(self, mem: MemoryStore) -> None:
        s = mem.start_session(metadata={"project": "vecforge"})
        assert isinstance(s, Session) and s.is_open
        m = mem.remember("worked on the quantum module", kind="episode")
        assert m.session_id == s.session_id
        assert mem.get_session(s.session_id).turn_count == 1  # type: ignore[union-attr]
        ended = mem.end_session("Fixed the Grover reranker")
        assert ended is not None and not ended.is_open and ended.summary
        assert mem.current_session is None
        assert any(x.kind == "summary" for x in mem.list_memories())

        mem.start_session()
        pack = mem.context_pack("quantum")
        assert "Last session" in pack and "Fixed the Grover reranker" in pack
        assert "### Episodes" in pack and "worked on the quantum module" in pack
        assert len(mem.context_pack("quantum", max_chars=40)) <= 40
        assert mem.last_session().summary == "Fixed the Grover reranker"  # type: ignore[union-attr]
        assert len(mem.session_memories(s.session_id)) == 2

    def test_end_unknown_session(self, mem: MemoryStore) -> None:
        assert mem.end_session("x") is None
        assert mem.end_session("x", session_id="nope") is None

    def test_explicit_session_id_autocreates_record(self, mem: MemoryStore) -> None:
        mem.remember("tied to s9", session_id="s9")
        assert mem.get_session("s9") is not None
        assert [s.session_id for s in mem.list_sessions()] == ["s9"]


class TestPersistenceAndIO:
    def test_memories_survive_reopen(self, tmp_path: Path) -> None:
        path = str(tmp_path / "mem.db")
        store = MemoryStore(path, agent="claude", model_name="hash")
        store.remember("persistent fact about rockets", kind="fact")
        store.start_session()
        store.end_session("session one")
        store.close()
        again = MemoryStore(path, agent="claude", model_name="hash")
        assert again.recall("rockets", top_k=1)[0].content.startswith("persistent")
        assert again.last_session().summary == "session one"  # type: ignore[union-attr]
        other = MemoryStore(vault=again.vault, agent="other")
        assert other.recall("rockets") == []  # agent isolation
        again.close()

    def test_default_path_is_created(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        store = MemoryStore(agent="a", model_name="hash")
        assert (tmp_path / ".vecforge" / "memory.db").exists()
        store.close()

    def test_export_import(self, mem: MemoryStore, tmp_path: Path) -> None:
        mem.remember("exported fact", kind="fact", tags=["io"])
        mem.start_session()
        mem.end_session("exported session")
        out = tmp_path / "mem.json"
        records = mem.export_json(str(out))
        assert len(records) == 2
        payload = json.loads(out.read_text())
        assert payload["agent"] == "claude" and len(payload["sessions"]) == 1
        target = MemoryStore(":memory:", agent="fresh", model_name="hash")
        assert target.import_json(str(out)) == 2
        assert target.import_json(str(out)) == 0  # all duplicates now
        assert target.recall("exported fact", top_k=1)[0].tags == ["io"]
        target.close()

    def test_shared_vault_not_closed_by_store(self) -> None:
        db = VecForge(":memory:", model_name="hash")
        store = MemoryStore(vault=db, agent="a")
        store.remember("x")
        store.close()
        assert not db.closed
        assert "memory:a" in db.list_namespaces()
        db.close()

    def test_repr_and_stats(self, mem: MemoryStore) -> None:
        assert "claude" in repr(mem)
        st = mem.stats()
        assert (
            st["namespace"] == "memory:claude" and st["embedding_model"] == "hash:384"
        )
