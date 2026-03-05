# ⚙️ Configuration Reference

All VecForge configuration options in one place.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

---

## VecForge Constructor Options

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | `":memory:"` | Vault database file path |
| `encryption_key` | `str \| None` | `None` | SQLCipher AES-256 encryption key |
| `audit_log` | `str \| None` | `None` | Path to audit log file (JSONL) |
| `quantum` | `bool` | `False` | Enable quantum-inspired acceleration |
| `deletion_protection` | `bool` | `False` | Block delete operations |
| `api_key` | `str \| None` | `None` | API key for RBAC (None = admin) |
| `model_name` | `str` | `"all-MiniLM-L6-v2"` | Embedding model name |

---

## Environment Variables

| Variable | Description |
|---|---|
| `VECFORGE_KEY` | Encryption key for SQLCipher |

---

## Embedding Models

| Model | Dimensions | Speed | Quality |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | ⚡ Fast | Good |
| `all-MiniLM-L12-v2` | 384 | Medium | Better |
| `all-mpnet-base-v2` | 768 | Slower | Best |
| `paraphrase-MiniLM-L6-v2` | 384 | ⚡ Fast | Good |

---

## Ingestion Settings

| Parameter | Default | Range | Description |
|---|---|---|---|
| `chunk_size` | `1000` | 100–10000 | Characters per chunk |
| `chunk_overlap` | `200` | 0–chunk_size | Overlap between chunks |

---

## Search Settings

| Parameter | Default | Range | Description |
|---|---|---|---|
| `top_k` | `10` | 1–1000 | Results to return |
| `alpha` | `0.5` | 0.0–1.0 | Semantic vs keyword weight |
| `rerank` | `False` | — | Cross-encoder reranking |
| `recency_weight` | `0.0` | 0.0–1.0 | Favour recent documents |

---

## RBAC Roles

| Role | Permissions |
|---|---|
| `admin` | read, write, delete, create_namespace, manage_keys, backup |
| `read-write` | read, write, delete |
| `read-only` | read |

---

## File Formats

| Extension | Handler | Required Package |
|---|---|---|
| `.txt`, `.md` | Built-in | — |
| `.pdf` | PyMuPDF | `pymupdf` |
| `.docx` | python-docx | `python-docx` |
| `.html`, `.htm` | BeautifulSoup | `beautifulsoup4` |
