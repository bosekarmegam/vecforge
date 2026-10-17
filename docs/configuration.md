# ⚙️ Configuration Reference

All VecForge 2.0 configuration options in one place.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

---

## VecForge Constructor Options

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | `":memory:"` | Vault database file path |
| `encryption_key` | `str \| None` | `None` | SQLCipher AES-256 encryption key |
| `audit_log` | `str \| None` | `None` | Path to audit log file (JSONL) |
| `quantum` | `bool` | `False` | Default for `search(quantum_rerank=...)` |
| `deletion_protection` | `bool` | `False` | Block delete / replacing upsert |
| `api_key` | `str \| None` | `None` | API key for RBAC (None = admin; unregistered keys get no access) |
| `model_name` | `str \| None` | `all-MiniLM-L6-v2` | Embedding model, `"hash"`, or `"hash:<dim>"` |
| `embedder` | `EmbedderLike` | `None` | Custom embedder object |
| `index_type` | `str` | `"auto"` | `flat`, `hnsw`, or `auto` (HNSW past 100k docs) |
| `auto_save_every` | `int` | `0` | Snapshot cadence in mutations (0 = on save/close) |
| `rerank_model` | `str \| None` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |

---

## Environment Variables

| Variable | Description |
|---|---|
| `VECFORGE_KEY` | Encryption key (used by `vecforge mcp`; pass explicitly elsewhere) |
| `VECFORGE_MODEL` | Default embedding model name (e.g. `hash` for offline/CI) |
| `VECFORGE_API_KEY` | API key required by `vecforge serve` / `create_app` |

---

## Embedding Models

| Model | Dimensions | Speed | Quality |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | ⚡ Fast | Good (default) |
| `all-MiniLM-L12-v2` | 384 | Medium | Better |
| `all-mpnet-base-v2` | 768 | Slower | Best |
| `paraphrase-MiniLM-L6-v2` | 384 | ⚡ Fast | Good |
| `hash` / `hash:<dim>` | 384 (configurable) | ⚡⚡ Instant | Lexical only, no download |

Switching models on an existing vault: open it with the new model and call
`db.reembed()`.

---

## Index Types

| `index_type` | Search | Memory at 1M×384 | Notes |
|---|---|---|---|
| `flat` | exact, ~30 ms at 1M | 1.5 GB | default below 100k docs |
| `hnsw` | approximate, <1 ms at 1M | ~2 GB | graph build takes minutes at 1M |
| `auto` | flat → HNSW at 100k | — | one-time upgrade when the threshold is crossed |

---

## Ingestion Settings

| Parameter | Default | Range | Description |
|---|---|---|---|
| `chunk_size` | `1000` | 100–10000 | Characters per chunk |
| `chunk_overlap` | `200` | 0–chunk_size | Overlap between chunks |
| `batch_size` | `64` | ≥1 | Chunks embedded per batch |

---

## Search Settings

| Parameter | Default | Range | Description |
|---|---|---|---|
| `top_k` | `10` | 1–1000 | Results to return |
| `alpha` | `0.5` | 0.0–1.0 | Semantic vs keyword weight |
| `rerank` | `False` | — | Cross-encoder reranking |
| `quantum_rerank` | vault default | — | Grover reranking with consensus oracle |
| `recency_weight` | `0.0` | 0.0–1.0 | Favour recent documents (applied after reranking) |

Metadata filter operators: `gte`, `lte`, `gt`, `lt`, `in`, `not_in`, `ne`,
`contains`, `any`, `all`, `exists`.

---

## Memory Settings

| Parameter | Default | Description |
|---|---|---|
| `agent` | `"default"` | Namespace `memory:<agent>` |
| `dedup_threshold` | `0.92` | Cosine similarity treated as a duplicate |
| `auto_save_every` | `25` | Snapshot cadence for memory stores |
| `recency_weight` / `importance_weight` | `0.15` / `0.25` | Recall score blend |

---

## RBAC Roles

| Role | Permissions |
|---|---|
| `admin` | read, write, delete, create_namespace, manage_keys, backup |
| `read-write` | read, write, delete |
| `read-only` | read |
| `none` | nothing — the role of any key that was not registered |

Register keys with `db.register_api_key(key, role)`; only SHA-256 digests are stored.

---

## File Formats

| Extension | Handler | Required Package |
|---|---|---|
| `.txt`, `.md` | Built-in | — |
| `.pdf` | PyMuPDF | `pymupdf` |
| `.docx` | python-docx | `python-docx` |
| `.html`, `.htm` | BeautifulSoup | `beautifulsoup4` |
