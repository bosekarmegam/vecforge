# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under BSL 1.1 — see LICENSE for details.

"""Tests for the MCP server (tool implementations + SDK registration)."""

from __future__ import annotations

import asyncio
import json

import pytest

from vecforge.mcp.server import (
    VecForgeMCP,
    claude_code_config,
    claude_desktop_config,
)

mcp_sdk = pytest.importorskip("mcp", reason="mcp SDK not installed")


@pytest.fixture
def app() -> VecForgeMCP:
    a = VecForgeMCP(
        vault_path=":memory:", agent="test", model_name="hash", warmup=False
    )
    yield a
    a.close()


class TestToolImplementations:
    def test_remember_recall_forget(self, app: VecForgeMCP) -> None:
        out = app.remember(
            "User likes tabs", kind="preference", importance=0.8, tags=["style"]
        )
        assert out["stored"] and out["memory"]["kind"] == "preference"
        dup = app.remember("User likes tabs", kind="preference")
        assert not dup["stored"]
        rec = app.recall("indentation preference", top_k=3)
        assert rec["count"] == 1 and rec["memories"][0]["content"] == "User likes tabs"
        assert app.recall("x", since_days=0.0001)["count"] == 1
        mid = out["memory"]["memory_id"]
        assert app.reinforce(mid)["memory"]["importance"] >= 0.8
        assert app.forget(mid)["deleted"]
        assert app.list_memories()["count"] == 0

    def test_sessions_and_context(self, app: VecForgeMCP) -> None:
        start = app.start_session("first")
        assert start["last_session"] is None
        app.remember("did some work", kind="episode")
        end = app.end_session("finished the work")
        assert end["session"]["summary"] == "finished the work"
        start2 = app.start_session()
        assert start2["last_session"]["summary"] == "finished the work"
        pack = app.context_pack("work")
        assert "finished the work" in pack
        assert app.recent_sessions()["count"] == 2

    def test_vault_tools(self, app: VecForgeMCP) -> None:
        assert app.search_vault("anything")["count"] == 0  # empty vault → graceful
        doc = app.add_document("Project notes about rockets", namespace="notes")
        assert doc["namespace"] == "notes"
        hit = app.search_vault("rockets", namespace="notes", quantum=True)
        assert hit["count"] == 1 and hit["results"][0]["doc_id"] == doc["doc_id"]
        bad = app.search_vault("rockets", namespace="missing")
        assert bad["count"] == 0 and "error" in bad
        stats = app.vault_stats()
        assert stats["vault"]["documents"] == 1 and stats["memory"]["memories"] == 0


class TestServerRegistration:
    def test_tools_resources_prompts_registered(self) -> None:
        from vecforge.mcp.server import create_server

        server, app = create_server(
            vault_path=":memory:", agent="t", model_name="hash", warmup=False
        )
        try:
            tools = asyncio.run(server.list_tools())
            names = {t.name for t in tools}
            assert {
                "remember",
                "recall",
                "forget",
                "reinforce",
                "list_memories",
                "context_pack",
                "start_session",
                "end_session",
                "recent_sessions",
                "search_vault",
                "add_document",
                "vault_stats",
            } <= names
            resources = asyncio.run(server.list_resources())
            assert any(str(r.uri).startswith("vecforge://memory") for r in resources)
            prompts = asyncio.run(server.list_prompts())
            assert any(p.name == "recall_context" for p in prompts)

            result = asyncio.run(
                server.call_tool("remember", {"content": "via sdk", "kind": "fact"})
            )
            assert result is not None
            assert app.store.recall("via sdk", top_k=1)[0].content == "via sdk"
        finally:
            app.close()

    def test_warmup_thread(self) -> None:
        app = VecForgeMCP(
            vault_path=":memory:", agent="w", model_name="hash", warmup=True
        )
        assert app._warm_thread is not None
        app._warm_thread.join(timeout=5)
        app.close()


class TestConfigSnippets:
    def test_claude_desktop_config(self) -> None:
        cfg = claude_desktop_config(
            "/tmp/mem.db", agent="claude", python="/usr/bin/python3"
        )
        server = cfg["mcpServers"]["vecforge-memory"]
        assert server["command"] == "/usr/bin/python3"
        assert "--vault" in server["args"] and "/tmp/mem.db" in server["args"]
        json.dumps(cfg)

    def test_claude_code_config_matches(self) -> None:
        assert claude_code_config(agent="x") == claude_desktop_config(agent="x")
