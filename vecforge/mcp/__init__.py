# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under the Business Source License 1.1 (BSL 1.1)

"""
VecForge MCP server — give Claude (and any MCP client) persistent memory.

Run locally over stdio (what Claude Desktop and Claude Code use)::

    vecforge mcp --vault ~/.vecforge/memory.db --agent claude

or over HTTP for several clients::

    vecforge mcp --transport http --port 8765

Requires the ``mcp`` package: ``pip install vecforge[mcp]``.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from vecforge.mcp.server import (
    claude_code_config,
    claude_desktop_config,
    create_server,
    run_server,
)

__all__ = [
    "claude_code_config",
    "claude_desktop_config",
    "create_server",
    "run_server",
]
