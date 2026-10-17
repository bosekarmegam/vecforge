# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
AI Memory Agent Example — VecForge 2.0

Shows how an assistant remembers across sessions:

1. Session one: the user shares preferences and facts; the assistant stores
   them and ends the session with a summary.
2. The process "restarts" (the store is closed and reopened from disk).
3. Session two: the assistant asks for a context pack and recalls what it
   needs — before the user repeats anything.

The same store is what `vecforge mcp` exposes to Claude Desktop / Claude Code.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

Usage:
    python examples/ai_memory_agent.py
"""

from __future__ import annotations

import os
import tempfile

from vecforge import MemoryStore


def session_one(path: str) -> None:
    mem = MemoryStore(path, agent="assistant", model_name="hash")
    session = mem.start_session(metadata={"client": "example"})
    print(f"🟢 Session {session.session_id} started")

    # Things the user said that are worth keeping
    style = "User prefers answers under 150 words with no em-dashes"
    mem.remember(
        "User's name is Suneel and they lead ArcGX TechLabs",
        kind="fact",
        importance=0.9,
    )
    mem.remember(style, kind="preference", importance=0.95)
    mem.remember(
        "Project vecforge uses black, ruff and pytest; CI runs on 3.10-3.13",
        kind="fact",
        tags=["vecforge"],
    )
    mem.remember(
        "Decided to ship version 2.0 with the memory layer and MCP server",
        kind="decision",
        tags=["vecforge"],
    )
    mem.remember(
        "Follow up: write the release notes for 2.0", kind="task", tags=["vecforge"]
    )

    # A duplicate is reinforced, not duplicated
    again = mem.remember(style, kind="preference")
    print(
        f"   duplicate reinforced: {again.deduplicated} "
        f"(importance now {again.importance:.2f})"
    )

    mem.end_session(
        "Set up the vecforge 2.0 release plan and captured the user's answer style"
    )
    print(f"   memories stored: {mem.stats()['memories']}")
    mem.close()


def session_two(path: str) -> None:
    mem = MemoryStore(path, agent="assistant", model_name="hash")
    mem.start_session()

    print("\n📋 Context pack at the start of the new session:\n")
    print(mem.context_pack("vecforge release"))

    print("\n🔎 Targeted recall — 'how should I phrase my answers?'")
    for m in mem.recall("how should I phrase my answers?", top_k=2):
        print(f"   [{m.score:.2f}] ({m.kind}) {m.content}")

    print("\n🔎 Open tasks:")
    for m in mem.list_memories(kinds=["task"]):
        print(f"   - {m.content}")

    mem.end_session("Answered release questions using recalled context")
    mem.close()


def main() -> None:
    print("🧠 VecForge AI Memory Agent Demo")
    print("=" * 50)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "memory.db")
        session_one(path)
        print("\n🔁 process restarted — reopening the store from disk")
        session_two(path)
    print("\n✅ Memory survived the restart.")
    print("VecForge by Suneel Bose K · ArcGX TechLabs")


if __name__ == "__main__":
    main()
