# 🖥️ CLI Reference

VecForge command-line interface.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

---

## Commands

### `vecforge ingest`

Ingest documents from files or directories.

```bash
vecforge ingest <path> --vault <vault.db> [options]
```

| Option | Default | Description |
|---|---|---|
| `--vault` | required | Path to vault database |
| `--namespace` | `default` | Target namespace |
| `--chunk-size` | `1000` | Chunk size in characters |
| `--chunk-overlap` | `200` | Overlap between chunks |

**Examples:**
```bash
vecforge ingest reports/ --vault my.db
vecforge ingest data.pdf --vault my.db --namespace legal
vecforge ingest docs/ --vault my.db --chunk-size 500
```

---

### `vecforge search`

Search the vault with a natural language query.

```bash
vecforge search "<query>" --vault <vault.db> [options]
```

| Option | Default | Description |
|---|---|---|
| `--vault` | required | Path to vault database |
| `--top-k` | `5` | Number of results |
| `--namespace` | `None` | Restrict to namespace |
| `--alpha` | `0.5` | Semantic weight (0.0–1.0) |
| `--rerank` | `False` | Enable cross-encoder reranking |

**Examples:**
```bash
vecforge search "diabetes treatment" --vault my.db
vecforge search "contract clause" --vault my.db --top-k 10 --rerank
vecforge search "patient" --vault my.db --namespace ward_7
```

---

### `vecforge stats`

Show vault statistics.

```bash
vecforge stats <vault.db>
```

**Example:**
```
═══════════════════════════════════════════════
VecForge Vault Statistics
═══════════════════════════════════════════════
Path:           my.db
Documents:      1500
Encrypted:      True
Quantum:        False
Protection:     False
Namespaces:     default, ward_7, ICU
Index vectors:  1500
BM25 docs:      1500
```

---

### `vecforge export`

Export vault data to JSON.

```bash
vecforge export <vault.db> [options]
```

| Option | Default | Description |
|---|---|---|
| `--format` | `json` | Export format |
| `-o, --output` | stdout | Output file path |
| `--namespace` | `None` | Export specific namespace |

**Examples:**
```bash
vecforge export my.db -o data.json
vecforge export my.db --namespace ward_7 -o ward7.json
```

---

### `vecforge serve`

Start the VecForge REST API server.

```bash
vecforge serve --vault <vault.db> [options]
```

| Option | Default | Description |
|---|---|---|
| `--vault` | required | Path to vault database |
| `--port` | `8080` | Server port |
| `--host` | `0.0.0.0` | Server host |

**Example:**
```bash
vecforge serve --vault my.db --port 8080
```

---

## Global Options

```bash
vecforge --version    # Show version
vecforge --help       # Show help
```
