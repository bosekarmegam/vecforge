# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
Model Context Protocol server exposing VecForge memory and search.

Tools (all local, nothing leaves the machine):

    remember        store a memory (fact / preference / episode / …)
    recall          hybrid-search memories, weighted by importance+recency
    forget          delete a memory
    reinforce       raise a memory's importance
    list_memories   browse without a query
    context_pack    prompt-ready briefing for the start of a session
    start_session   open a session so new memories are attributed to it
    end_session     close it with a summary the next session can recall
    recent_sessions what happened last time
    search_vault    search any namespace of the vault (documents, notes…)
    add_document    add a document to the vault
    vault_stats     vault + memory statistics

Startup is fast: the embedding model loads in a background thread while
the MCP handshake completes, the vault restores its indexes from a
snapshot instead of rebuilding, and the index snapshot is refreshed every
25 writes and at shutdown so the next session opens instantly.

Works with both ``mcp`` 1.x (``FastMCP``) and ``mcp`` 2.x (``MCPServer``).

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import signal
import sys
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

from vecforge.exceptions import VecForgeError
from vecforge.memory import MemoryStore

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8765
_INSTRUCTIONS = (
    "VecForge gives you persistent local memory across sessions. "
    "At the start of a conversation call context_pack (optionally with a topic) "
    "to recall what you know. Use remember to store durable facts, user "
    "preferences, decisions and task state as they come up (one idea per "
    "memory, importance 0-1). Use recall to look things up before answering "
    "questions about past work. Call start_session at the beginning and "
    "end_session with a short summary at the end so the next session can "
    "pick up where you left off."
)


def _load_server_class() -> tuple[Any, int]:
    """Return ``(server_class, major_version)`` for the installed mcp SDK."""
    try:
        from mcp.server.mcpserver import MCPServer  # mcp >= 2.0

        return MCPServer, 2
    except ImportError:
        pass
    try:
        # mcp 1.x
        from mcp.server.fastmcp import (  # type: ignore[attr-defined,unused-ignore]
            FastMCP,
        )

        return FastMCP, 1
    except ImportError as exc:
        raise ImportError(
            "The MCP server requires the 'mcp' package.\n"
            "Install with: pip install vecforge[mcp]\n"
            "VecForge by Suneel Bose K · ArcGX TechLabs"
        ) from exc


class VecForgeMCP:
    """Holds the memory store and implements every tool as a plain method.

    The MCP server registers thin wrappers around these methods, so they
    can also be unit-tested (and reused by other transports) without an
    MCP client.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        vault_path: Vault file (default ``~/.vecforge/memory.db``).
        agent: Agent name for memory isolation.
        model_name: Embedding model name or ``"hash"``.
        encryption_key: Optional SQLCipher key.
        warmup: Load the embedding model in a background thread.
    """

    def __init__(
        self,
        vault_path: str | None = None,
        agent: str = "default",
        model_name: str | None = None,
        encryption_key: str | None = None,
        warmup: bool = True,
    ) -> None:
        self.store = MemoryStore(
            vault_path,
            agent=agent,
            model_name=model_name,
            encryption_key=encryption_key,
            auto_save_every=25,
        )
        self._started_at = time.time()
        self._warm_thread: threading.Thread | None = None
        if warmup:
            self._warm_thread = threading.Thread(
                target=self._warm, name="vecforge-warmup", daemon=True
            )
            self._warm_thread.start()

    def _warm(self) -> None:
        try:
            warm = getattr(self.store.vault.embedder, "warmup", None)
            if callable(warm):
                warm()
            else:
                self.store.vault.embedder.embed("vecforge warmup")
            logger.info("Embedding model warm")
        except Exception as exc:  # pragma: no cover - model download issues
            logger.warning("Embedder warmup failed: %s", exc)

    # ─── Tool implementations ───

    def remember(
        self,
        content: str,
        kind: str = "fact",
        importance: float = 0.5,
        tags: list[str] | None = None,
        session_id: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        m = self.store.remember(
            content,
            kind=kind,
            importance=importance,
            tags=tags,
            session_id=session_id,
            source=source,
        )
        return {"stored": not m.deduplicated, "memory": m.to_dict()}

    def recall(
        self,
        query: str,
        top_k: int = 8,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        since_days: float | None = None,
        min_importance: float = 0.0,
    ) -> dict[str, Any]:
        since = (
            time.time() - float(since_days) * 86400.0
            if since_days is not None
            else None
        )
        memories = self.store.recall(
            query,
            top_k=top_k,
            kinds=kinds,
            tags=tags,
            since=since,
            min_importance=min_importance,
        )
        return {"count": len(memories), "memories": [m.to_dict() for m in memories]}

    def forget(self, memory_id: str) -> dict[str, Any]:
        return {"memory_id": memory_id, "deleted": self.store.forget(memory_id)}

    def reinforce(self, memory_id: str, delta: float = 0.1) -> dict[str, Any]:
        return {"memory": self.store.reinforce(memory_id, delta=delta).to_dict()}

    def list_memories(
        self,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        session_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        memories = self.store.list_memories(
            kinds=kinds, tags=tags, session_id=session_id, limit=limit
        )
        return {"count": len(memories), "memories": [m.to_dict() for m in memories]}

    def context_pack(
        self, query: str = "", max_chars: int = 4000, top_k: int = 12
    ) -> str:
        return self.store.context_pack(query or None, max_chars=max_chars, top_k=top_k)

    def start_session(
        self, name: str = "", metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        meta = dict(metadata or {})
        if name:
            meta["name"] = name
        session = self.store.start_session(metadata=meta)
        last = self.store.last_session()
        return {
            "session": session.to_dict(),
            "last_session": last.to_dict() if last else None,
        }

    def end_session(
        self, summary: str = "", session_id: str | None = None
    ) -> dict[str, Any]:
        session = self.store.end_session(summary=summary, session_id=session_id)
        self.store.save()
        return {"session": session.to_dict() if session else None}

    def recent_sessions(self, limit: int = 5) -> dict[str, Any]:
        sessions = self.store.list_sessions(limit=limit)
        return {"count": len(sessions), "sessions": [s.to_dict() for s in sessions]}

    def search_vault(
        self,
        query: str,
        top_k: int = 5,
        namespace: str | None = None,
        filters: dict[str, Any] | None = None,
        quantum: bool = False,
    ) -> dict[str, Any]:
        try:
            results = self.store.vault.search(
                query,
                top_k=top_k,
                namespace=namespace,
                filters=filters,
                quantum_rerank=quantum,
            )
        except Exception as exc:  # VaultEmptyError, NamespaceNotFoundError…
            return {"count": 0, "results": [], "error": str(exc).splitlines()[0]}
        return {"count": len(results), "results": [r.to_dict() for r in results]}

    def add_document(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        namespace: str = "default",
    ) -> dict[str, Any]:
        doc_id = self.store.vault.add(text, metadata=metadata, namespace=namespace)
        return {"doc_id": doc_id, "namespace": namespace}

    def vault_stats(self) -> dict[str, Any]:
        return {
            "memory": self.store.stats(),
            "vault": self.store.vault.stats(),
            "uptime_seconds": round(time.time() - self._started_at, 1),
        }

    def close(self) -> None:
        if not self.store.vault.closed:
            self.store.close()


def _tool_error_class() -> type[Exception]:
    """Return the SDK's ToolError (its message is shown to the model)."""
    try:
        from mcp.server.mcpserver.exceptions import ToolError

        return ToolError  # type: ignore[no-any-return,unused-ignore]
    except ImportError:
        try:
            from mcp.server.fastmcp.exceptions import ToolError as LegacyToolError

            return LegacyToolError  # type: ignore[no-any-return]
        except ImportError:
            return RuntimeError


_T = TypeVar("_T")


def _safe(fn: Callable[..., _T], *args: Any) -> _T:
    """Call a tool implementation, turning user errors into ToolErrors."""
    try:
        return fn(*args)
    except (VecForgeError, ValueError, TypeError, KeyError) as exc:
        message = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        raise _tool_error_class()(message) from exc


def _register(server: Any, app: VecForgeMCP) -> None:
    """Attach tools, resources and prompts to an MCP server instance."""

    @server.tool(name="remember", description="Store a memory for future sessions.")
    def remember(
        content: str,
        kind: str = "fact",
        importance: float = 0.5,
        tags: list[str] | None = None,
        session_id: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        """Store a durable memory.

        kind: fact | preference | episode | task | summary | note | insight |
        decision. importance: 0.0-1.0. Near-duplicates reinforce the
        existing memory instead of creating a copy.
        """
        return _safe(app.remember, content, kind, importance, tags, session_id, source)

    @server.tool(name="recall", description="Search memories by meaning and keywords.")
    def recall(
        query: str,
        top_k: int = 8,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        since_days: float | None = None,
        min_importance: float = 0.0,
    ) -> dict[str, Any]:
        """Recall memories relevant to a query, ranked by relevance,
        importance and recency."""
        return _safe(app.recall, query, top_k, kinds, tags, since_days, min_importance)

    @server.tool(name="forget", description="Delete a memory by ID.")
    def forget(memory_id: str) -> dict[str, Any]:
        """Permanently delete one memory."""
        return _safe(app.forget, memory_id)

    @server.tool(name="reinforce", description="Mark a memory as important.")
    def reinforce(memory_id: str, delta: float = 0.1) -> dict[str, Any]:
        """Raise a memory's importance so it is recalled more readily."""
        return _safe(app.reinforce, memory_id, delta)

    @server.tool(name="list_memories", description="Browse memories without a query.")
    def list_memories(
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        session_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """List recent memories, optionally filtered by kind, tag or session."""
        return _safe(app.list_memories, kinds, tags, session_id, limit)

    @server.tool(
        name="context_pack",
        description="Prompt-ready briefing of what you remember (session start).",
    )
    def context_pack(query: str = "", max_chars: int = 4000, top_k: int = 12) -> str:
        """Return a markdown briefing: last session summary + relevant memories."""
        return _safe(app.context_pack, query, max_chars, top_k)

    @server.tool(name="start_session", description="Open a memory session.")
    def start_session(
        name: str = "", metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Start a session; returns it plus the previous session (if any)."""
        return _safe(app.start_session, name, metadata)

    @server.tool(name="end_session", description="Close the session with a summary.")
    def end_session(summary: str = "", session_id: str | None = None) -> dict[str, Any]:
        """End the current session and store the summary for next time."""
        return _safe(app.end_session, summary, session_id)

    @server.tool(name="recent_sessions", description="List recent sessions.")
    def recent_sessions(limit: int = 5) -> dict[str, Any]:
        """Show the most recent sessions and their summaries."""
        return _safe(app.recent_sessions, limit)

    @server.tool(name="search_vault", description="Hybrid search over the vault.")
    def search_vault(
        query: str,
        top_k: int = 5,
        namespace: str | None = None,
        filters: dict[str, Any] | None = None,
        quantum: bool = False,
    ) -> dict[str, Any]:
        """Search documents in any namespace (dense + BM25 hybrid)."""
        return _safe(app.search_vault, query, top_k, namespace, filters, quantum)

    @server.tool(name="add_document", description="Add a document to the vault.")
    def add_document(
        text: str, metadata: dict[str, Any] | None = None, namespace: str = "default"
    ) -> dict[str, Any]:
        """Store a document (not a memory) for later search."""
        return _safe(app.add_document, text, metadata, namespace)

    @server.tool(name="vault_stats", description="Vault and memory statistics.")
    def vault_stats() -> dict[str, Any]:
        """Return statistics about the vault and the memory store."""
        return _safe(
            app.vault_stats,
        )

    @server.resource("vecforge://memory/recent")
    def recent_memories_resource() -> str:
        """The 20 most recent memories as JSON."""
        return json.dumps(app.list_memories(limit=20), default=str)

    @server.resource("vecforge://memory/sessions")
    def sessions_resource() -> str:
        """Recent sessions as JSON."""
        return json.dumps(app.recent_sessions(limit=10), default=str)

    @server.resource("vecforge://vault/stats")
    def stats_resource() -> str:
        """Vault statistics as JSON."""
        return json.dumps(app.vault_stats(), default=str)

    @server.prompt(name="recall_context", description="Briefing from memory")
    def recall_context(topic: str = "") -> str:
        """Prompt that injects a memory briefing for a topic."""
        pack = app.context_pack(topic)
        return pack or "No memories stored yet."


def create_server(
    vault_path: str | None = None,
    agent: str = "default",
    model_name: str | None = None,
    encryption_key: str | None = None,
    name: str = "vecforge",
    host: str = "127.0.0.1",
    port: int = _DEFAULT_PORT,
    warmup: bool = True,
) -> tuple[Any, VecForgeMCP]:
    """Build an MCP server wired to a VecForge memory store.

    Args:
        vault_path: Vault file (default ``~/.vecforge/memory.db``).
        agent: Agent name for memory isolation.
        model_name: Embedding model name or ``"hash"``.
        encryption_key: Optional SQLCipher key.
        name: Server name shown to clients.
        host: Bind host for HTTP transports.
        port: Bind port for HTTP transports.
        warmup: Load the embedding model in the background at startup.

    Returns:
        ``(server, app)`` — the SDK server object and the tool implementation.

    Performance:
        Startup: O(index snapshot) + background model load
    """
    server_cls, major = _load_server_class()
    app = VecForgeMCP(
        vault_path=vault_path,
        agent=agent,
        model_name=model_name,
        encryption_key=encryption_key,
        warmup=warmup,
    )
    if major >= 2:
        server = server_cls(name=name, instructions=_INSTRUCTIONS)
    else:
        server = server_cls(name, instructions=_INSTRUCTIONS, host=host, port=port)
    _register(server, app)
    server._vecforge_app = app  # noqa: SLF001 - convenience for callers/tests
    server._vecforge_http = (host, port)  # noqa: SLF001
    return server, app


def run_server(
    vault_path: str | None = None,
    agent: str = "default",
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = _DEFAULT_PORT,
    model_name: str | None = None,
    encryption_key: str | None = None,
) -> None:
    """Start the MCP server and block until the client disconnects.

    Args:
        vault_path: Vault file (default ``~/.vecforge/memory.db``).
        agent: Agent name for memory isolation.
        transport: ``"stdio"`` (Claude Desktop / Claude Code) or
            ``"http"`` (streamable HTTP, several clients).
        host: Bind host for HTTP.
        port: Bind port for HTTP.
        model_name: Embedding model name or ``"hash"``.
        encryption_key: Optional SQLCipher key.
    """
    # security/why: stdio transport owns stdout — log to stderr only
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s vecforge-mcp %(levelname)s %(message)s",
    )
    server, app = create_server(
        vault_path=vault_path,
        agent=agent,
        model_name=model_name,
        encryption_key=encryption_key,
        host=host,
        port=port,
    )
    atexit.register(app.close)
    # why: SIGTERM (service managers, container stop) skips atexit unless it
    # is turned into a normal interpreter exit
    with contextlib.suppress(ValueError):  # not in the main thread
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    logger.info(
        "VecForge MCP server starting (agent=%s, transport=%s, vault=%s)",
        agent,
        transport,
        app.store.vault.path,
    )
    _, major = _load_server_class()
    try:
        if transport in ("http", "streamable-http"):
            if major >= 2:
                server.run("streamable-http", host=host, port=port)
            else:
                server.run("streamable-http")
        else:
            server.run("stdio")
    finally:
        app.close()


def claude_desktop_config(
    vault_path: str | None = None,
    agent: str = "claude",
    python: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Return the ``claude_desktop_config.json`` snippet for this server.

    Args:
        vault_path: Vault file to use.
        agent: Agent name.
        python: Python executable (defaults to the current interpreter).
        model_name: Embedding model (must match the one the vault uses).

    Returns:
        A dict to merge into Claude Desktop's config under ``mcpServers``.
    """
    args = ["-m", "vecforge.cli.main", "mcp", "--agent", agent]
    if vault_path:
        args += ["--vault", vault_path]
    if model_name:
        args += ["--model", model_name]
    return {
        "mcpServers": {
            "vecforge-memory": {
                "command": python or sys.executable,
                "args": args,
            }
        }
    }


def claude_code_config(
    vault_path: str | None = None,
    agent: str = "claude",
    python: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Return the ``.mcp.json`` snippet for Claude Code.

    Args:
        vault_path: Vault file to use.
        agent: Agent name.
        python: Python executable (defaults to the current interpreter).

    Returns:
        A dict to write as ``.mcp.json`` in the project root.
    """
    return claude_desktop_config(
        vault_path=vault_path, agent=agent, python=python, model_name=model_name
    )
