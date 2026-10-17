# 🔌 MCP Server Guide

Connect Claude Desktop, Claude Code, or any Model Context Protocol client to
a local VecForge memory — nothing leaves your machine.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

---

## Install

```bash
pip install "vecforge[mcp]"
```

Works with the official `mcp` Python SDK, both 1.x (`FastMCP`) and 2.x
(`MCPServer`).

---

## Start the server

```bash
# stdio transport — what Claude Desktop and Claude Code use
vecforge mcp --vault ~/.vecforge/memory.db --agent claude

# streamable HTTP for several clients on one machine
vecforge mcp --transport http --host 127.0.0.1 --port 8765
```

Options:

| Option | Default | Description |
|---|---|---|
| `--vault` | `~/.vecforge/memory.db` | Vault file (created on demand) |
| `--agent` | `claude` | Memory namespace |
| `--transport` | `stdio` | `stdio` or `http` |
| `--host` / `--port` | `127.0.0.1` / `8765` | HTTP bind address |
| `--model` | `all-MiniLM-L6-v2` | Embedding model, or `hash` for no download |
| `--print-config` | — | Print a client config snippet instead of starting |

Set `VECFORGE_KEY` to open an encrypted vault.

---

## Claude Desktop

Print the snippet and merge it into `claude_desktop_config.json`:

```bash
vecforge mcp --print-config claude-desktop --vault ~/.vecforge/memory.db
```

```json
{
  "mcpServers": {
    "vecforge-memory": {
      "command": "/usr/bin/python3",
      "args": ["-m", "vecforge.cli.main", "mcp", "--agent", "claude",
               "--vault", "/home/you/.vecforge/memory.db"]
    }
  }
}
```

Config file locations: macOS `~/Library/Application Support/Claude/claude_desktop_config.json`,
Windows `%APPDATA%\Claude\claude_desktop_config.json`.

## Claude Code

Write the same snippet to `.mcp.json` in your project root (or add it with
`claude mcp add`):

```bash
vecforge mcp --print-config claude-code > .mcp.json
```

---

## Tools exposed

| Tool | Purpose |
|---|---|
| `context_pack(query, max_chars, top_k)` | Prompt-ready briefing — call at session start |
| `remember(content, kind, importance, tags, session_id, source)` | Store a memory (near-duplicates reinforce) |
| `recall(query, top_k, kinds, tags, since_days, min_importance)` | Search memories |
| `forget(memory_id)` | Delete a memory |
| `reinforce(memory_id, delta)` | Raise importance |
| `list_memories(kinds, tags, session_id, limit)` | Browse without a query |
| `start_session(name, metadata)` | Open a session; returns the previous one |
| `end_session(summary, session_id)` | Close it with a summary |
| `recent_sessions(limit)` | What happened last time |
| `search_vault(query, top_k, namespace, filters, quantum)` | Hybrid search over any namespace |
| `add_document(text, metadata, namespace)` | Store a document (not a memory) |
| `vault_stats()` | Vault and memory statistics |

Resources: `vecforge://memory/recent`, `vecforge://memory/sessions`,
`vecforge://vault/stats`. Prompt: `recall_context(topic)`.

The server's instructions tell the model to call `context_pack` first,
`remember` durable facts as they come up, and `end_session` with a summary.

---

## Why it is fast

* The vault restores its FAISS and BM25 indexes from a snapshot instead of
  rebuilding them, so opening a vault with 100k memories takes milliseconds.
* The embedding model loads in a background thread while the MCP handshake
  completes.
* Repeated query embeddings are served from an LRU cache.
* The snapshot is refreshed every 25 writes and at shutdown, so the next
  session also opens instantly.

---

## Using the server from Python

```python
from vecforge.mcp.server import create_server, VecForgeMCP

# Tool implementations without any transport (handy for tests and other hosts)
app = VecForgeMCP(vault_path=":memory:", agent="bot", model_name="hash", warmup=False)
app.remember("User likes tabs", kind="preference")
print(app.context_pack("indentation"))

# Full MCP server object (mcp SDK)
server, app = create_server(vault_path="~/.vecforge/memory.db", agent="claude")
server.run("stdio")
```

---

## Security notes

* The stdio transport is only reachable by the process that spawned it.
* The HTTP transport binds `127.0.0.1` by default; put it behind an
  authenticating proxy before exposing it to a network.
* Memories live in the same encrypted vault as your documents when
  `VECFORGE_KEY` is set.
