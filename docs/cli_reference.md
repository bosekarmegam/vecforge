# 🖥️ CLI Reference

VecForge 2.0 command-line interface.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

---

Every command that opens a vault accepts `--model <name>` (or the
`VECFORGE_MODEL` environment variable). Use `--model hash` for the
dependency-free embedder.

## Vault commands

### `vecforge ingest`

```bash
vecforge ingest <path> --vault <vault.db> [--namespace NS] [--chunk-size 1000] [--chunk-overlap 200]
```
Ingests `.txt`, `.md`, `.pdf`, `.docx`, `.html` files or directories in batches.

### `vecforge search`

```bash
vecforge search "<query>" --vault <vault.db> [options]
```

| Option | Default | Description |
|---|---|---|
| `--top-k` | `5` | Number of results |
| `--namespace` | — | Restrict to a namespace |
| `--alpha` | `0.5` | Semantic weight (0.0–1.0) |
| `--rerank` | off | Cross-encoder reranking |
| `--quantum` | off | Grover-inspired reranking |
| `--filters` | — | Metadata filters as JSON, e.g. `'{"year": {"gte": 2024}}'` |
| `--recency` | `0.0` | Recency weight |
| `--json` | off | Print results as JSON |

```bash
vecforge search "diabetes treatment" --vault my.db --quantum --json
vecforge search "contract clause" --vault my.db --top-k 10 --rerank --filters '{"type": "NDA"}'
```

### `vecforge stats <vault.db>`

Shows documents, namespaces, index type, tombstones, embedder and version.

### `vecforge export <vault.db> [-o file.json] [--namespace NS]`

### `vecforge compact <vault.db>`

Rebuilds the indexes without tombstones and refreshes the snapshot.

### `vecforge backup <vault.db> <destination>`

Writes a consistent copy with the SQLite online-backup API — safe while the
vault is in use.

### `vecforge serve`

```bash
vecforge serve --vault <vault.db> [--port 8080] [--host 127.0.0.1] [--api-key KEY]
```

The server binds `127.0.0.1` by default. Pass `--host 0.0.0.0` to expose it
and `--api-key` (or `$VECFORGE_API_KEY`) to require `X-API-Key`.

---

## MCP server

```bash
vecforge mcp [--vault ~/.vecforge/memory.db] [--agent claude] [--transport stdio|http]
             [--host 127.0.0.1] [--port 8765] [--print-config claude-desktop|claude-code]
```

`--print-config` prints the JSON snippet for Claude Desktop or Claude Code
instead of starting the server. See the [MCP Guide](mcp.md).

---

## Memory commands

All take `--vault` (default `~/.vecforge/memory.db`), `--agent` (default
`claude`) and `--model`.

```bash
vecforge memory remember "User prefers tea" --kind preference --importance 0.8 --tags drinks
vecforge memory recall "beverage preference" --top-k 5 [--kinds fact,preference] [--json]
vecforge memory list [--limit 20] [--kinds task]
vecforge memory forget <memory_id>
vecforge memory sessions [--limit 10]
vecforge memory context [--query topic] [--max-chars 4000]
vecforge memory export memories.json
vecforge memory import memories.json
```

---

## Global options

```bash
vecforge --version    # prints the package version
vecforge --help
```
