# 📖 API Reference

Complete Python API for VecForge.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

---

## `VecForge` Class

```python
from vecforge import VecForge
```

### Constructor

```python
VecForge(
    path: str = ":memory:",
    encryption_key: str | None = None,
    audit_log: str | None = None,
    quantum: bool = False,
    deletion_protection: bool = False,
    api_key: str | None = None,
    model_name: str = "all-MiniLM-L6-v2",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | `":memory:"` | Vault path or `":memory:"` |
| `encryption_key` | `str \| None` | `None` | SQLCipher AES-256 key |
| `audit_log` | `str \| None` | `None` | Path to JSONL audit log |
| `quantum` | `bool` | `False` | Enable quantum-inspired acceleration |
| `deletion_protection` | `bool` | `False` | Prevent accidental deletions |
| `api_key` | `str \| None` | `None` | API key for RBAC |
| `model_name` | `str` | `"all-MiniLM-L6-v2"` | Sentence-transformers model |

---

### `add()`

```python
db.add(
    text: str,
    metadata: dict | None = None,
    namespace: str = "default",
    doc_id: str | None = None,
) -> str
```

Add a document to the vault. Returns the document ID.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | required | Document text |
| `metadata` | `dict \| None` | `None` | Key-value metadata |
| `namespace` | `str` | `"default"` | Target namespace |
| `doc_id` | `str \| None` | `None` | Custom ID (auto-UUID if None) |

**Returns:** `str` — Document ID

**Raises:** `VecForgePermissionError` if API key lacks write permission.

---

### `search()`

```python
db.search(
    query: str,
    top_k: int = 10,
    alpha: float = 0.5,
    rerank: bool = False,
    namespace: str | None = None,
    filters: dict | None = None,
    recency_weight: float = 0.0,
) -> list[SearchResult]
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | required | Natural language query |
| `top_k` | `int` | `10` | Number of results |
| `alpha` | `float` | `0.5` | Semantic weight (0.0–1.0) |
| `rerank` | `bool` | `False` | Enable cross-encoder reranking |
| `namespace` | `str \| None` | `None` | Namespace filter |
| `filters` | `dict \| None` | `None` | Metadata filters |
| `recency_weight` | `float` | `0.0` | Recency bias (0.0–1.0) |

**Returns:** `list[SearchResult]` sorted by descending relevance.

**Raises:**
- `VaultEmptyError` — Vault has no documents
- `InvalidAlphaError` — Alpha outside [0.0, 1.0]
- `NamespaceNotFoundError` — Namespace doesn't exist
- `VecForgePermissionError` — API key lacks read permission

---

### `ingest()`

```python
db.ingest(
    path: str,
    namespace: str = "default",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> int
```

**Returns:** `int` — Number of chunks ingested.

---

### `delete()`

```python
db.delete(doc_id: str) -> bool
```

**Returns:** `bool` — True if deleted.

**Raises:** `DeletionProtectedError`, `VecForgePermissionError`

---

### `create_namespace()`

```python
db.create_namespace(name: str) -> None
```

---

### `list_namespaces()`

```python
db.list_namespaces() -> list[str]
```

---

### `stats()`

```python
db.stats() -> dict[str, Any]
```

Returns: `{"documents": int, "namespaces": list, "encrypted": bool, ...}`

---

### `save()` / `close()`

```python
db.save()   # Persist FAISS index
db.close()  # Save + close connection
```

---

## `SearchResult` Dataclass

```python
from vecforge import SearchResult
```

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Document text |
| `score` | `float` | Relevance score |
| `metadata` | `dict` | User metadata |
| `namespace` | `str` | Namespace |
| `doc_id` | `str` | Document UUID |
| `modality` | `str` | Content type |
| `timestamp` | `float` | Creation time |

---

## Exceptions

```python
from vecforge import (
    VecForgeError,
    VaultEmptyError,
    NamespaceNotFoundError,
    VecForgePermissionError,
    InvalidAlphaError,
    EncryptionKeyError,
    DeletionProtectedError,
    IngestError,
)
```

All exceptions inherit from `VecForgeError`. All messages include actionable guidance.
