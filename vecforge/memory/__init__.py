# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under the Business Source License 1.1 (BSL 1.1)

"""
VecForge AI Memory Layer.

Persistent, searchable memory for AI assistants and agents. A
:class:`MemoryStore` lets Claude (or any model) remember facts,
preferences, decisions and episodes across sessions and recall them with
hybrid semantic + keyword search, weighted by importance and recency.

Usage::

    from vecforge.memory import MemoryStore

    mem = MemoryStore("~/.vecforge/memory.db", agent="claude")
    mem.remember("User prefers Python over TypeScript", kind="preference")
    for m in mem.recall("which language should I use?"):
        print(m.content)

Expose it to Claude Desktop / Claude Code over MCP with::

    vecforge mcp --vault ~/.vecforge/memory.db

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from vecforge.memory.store import KINDS, Memory, MemoryStore, Session

__all__ = ["KINDS", "Memory", "MemoryStore", "Session"]
