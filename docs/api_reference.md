# 📖 API Reference

Complete Python API for VecForge 2.0.

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
    model_name: str | None = None,
    embedder: EmbedderLike | None = None,
    index_type: str = "auto",
    auto_save_every: int = 0,
    rerank_model: str | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | `":memory:"` | Vault path or `":memory:"` |
| `encryption_key` | `str \| None` | `None` | SQLCipher AES-256 key |
| `audit_log` | `str \| None` | `None` | Path to JSONL audit log |
| `quantum` | `bool` | `False` | Default for `search(quantum_rerank=...)` |
| `deletion_protection` | `bool` | `False` | Block deletes and replacing upserts |
| `api_key` | `str \| None` | `None` | API key for RBAC (None = local admin) |
| `model_name` | `str \| None` | `all-MiniLM-L6-v2` | Sentence-transformers model, or `"hash"` / `"hash:<dim>"`; `$VECFORGE_MODEL` overrides the default |
| `embedder` | `EmbedderLike` | `None` | Pre-built embedder (overrides `model_name`) |
| `index_type` | `str` | `"auto"` | `"flat"` (exact), `"hnsw"` (approximate), `"auto"` (flat, HNSW past 100k docs) |
| `auto_save_every` | `int` | `0` | Save the index snapshot every N mutations |
| `rerank_model` | `str \| None` | ms-marco MiniLM | Cross-encoder for `rerank=True` |

---

### Writing

```python
db.add(text, metadata=None, namespace="default", doc_id=None,
       modality="text", embedding=None) -> str
```
Add one document. `embedding` accepts a precomputed vector (for example a CLIP image embedding); `text` is still indexed for BM25. Raises `DuplicateDocumentError` if `doc_id` exists.

```python
db.add_batch(texts, metadata_list=None, namespace="default",
             doc_ids=None, embeddings=None, modality="text") -> list[str]
```
One embedding call, one SQLite transaction, one index insert. 10–50× faster than sequential `add()`. Atomic: nothing is inserted if a custom ID collides.

```python
db.upsert(text, doc_id, metadata=None, namespace="default",
          modality="text", embedding=None) -> str
```
Insert, or replace the document with that ID.

```python
db.update_metadata(doc_id, metadata, merge=True) -> dict
```
Edit metadata without re-embedding. Raises `DocumentNotFoundError`.

```python
db.ingest(path, namespace="default", chunk_size=1000,
          chunk_overlap=200, batch_size=64) -> int
```
Ingest `.txt`, `.md`, `.pdf`, `.docx`, `.html` files or directories in batches.

---

### Reading

```python
db.search(
    query: str,
    top_k: int = 10,
    alpha: float = 0.5,
    rerank: bool = False,
    quantum_rerank: bool | None = None,
    namespace: str | None = None,
    filters: dict | None = None,
    recency_weight: float = 0.0,
    query_vector: ndarray | None = None,
    exclude_doc_ids: Sequence[str] | None = None,
) -> list[SearchResult]
```

| Parameter | Description |
|---|---|
| `alpha` | `0.0` keyword only, `1.0` semantic only, `0.5` balanced |
| `rerank` | Cross-encoder reranking |
| `quantum_rerank` | Grover amplification with a consensus oracle; `None` uses the vault default |
| `namespace` | Enforced inside the retrievers (no recall loss) |
| `filters` | Metadata filters; the candidate window widens automatically until `top_k` matches are found |
| `recency_weight` | Applied after reranking on min-max normalised scores |
| `query_vector` | Precomputed query embedding (image queries, `similar()`) |
| `exclude_doc_ids` | Leave these documents out |

Raises `VaultEmptyError`, `InvalidAlphaError`, `NamespaceNotFoundError`, `VecForgePermissionError`.

```python
db.search_batch(queries, **kwargs) -> list[list[SearchResult]]   # one embedding call
db.similar(doc_id, top_k=10, namespace=None, filters=None)      # neighbours of a document
db.get(doc_id) -> SearchResult | None
db.get_embedding(doc_id) -> ndarray | None
db.exists(doc_id) -> bool          # also: doc_id in db
db.count(namespace=None) -> int    # also: len(db)
```

---

### Deleting

```python
db.delete(doc_id) -> bool                 # O(1) tombstone, auto-compaction at 20% tombstones
db.delete_many(doc_ids) -> int
db.delete_namespace(name) -> int
db.compact() -> None                      # rebuild indexes without tombstones
```
All raise `DeletionProtectedError` when `deletion_protection=True`.

---

### Namespaces, keys, maintenance

```python
db.create_namespace(name)
db.list_namespaces() -> list[str]
db.exists_namespace(name) -> bool

db.register_api_key(api_key, role)        # role: admin | read-write | read-only; digest stored
db.revoke_api_key(api_key) -> bool
db.list_api_keys() -> dict[str, str]      # digest → role

db.stats() -> dict                        # documents, namespace_counts, index_type, tombstones, model, …
db.save()                                 # persist FAISS + BM25 snapshot for instant reopen
db.backup(dest_path) -> str               # consistent copy (SQLite online backup)
db.reembed(batch_size=256) -> int         # re-index after changing the embedding model
db.close()                                # save + close (also via `with VecForge(...) as db:`)
```

Properties: `db.embedder`, `db.storage`, `db.path`, `db.closed`.

---

## `SearchResult`

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Document text |
| `score` | `float` | Relevance score (higher = better) |
| `metadata` | `dict` | User metadata |
| `namespace` | `str` | Namespace |
| `doc_id` | `str` | Document ID |
| `modality` | `str` | `text`, `image`, `audio`, `memory`, … |
| `timestamp` | `float` | Creation time |
| `dense_score` | `float \| None` | Normalised semantic score, `None` if not a dense hit |
| `sparse_score` | `float \| None` | Normalised BM25 score, `None` if not a keyword hit |
| `signals` | `list[str]` | `["dense"]`, `["sparse"]` or both |
| `updated_at` | `float \| None` | Last edit time |

`SearchResult.to_dict()` returns a JSON-serialisable dict.

---

## Embedders

```python
from vecforge import Embedder, HashEmbedder, CallableEmbedder, EmbedderLike, get_embedder
```

| Class | Description |
|---|---|
| `Embedder(model_name, device=None, cache_size=512)` | sentence-transformers, lazy load, LRU cache |
| `HashEmbedder(dimension=384)` | Deterministic feature hashing, no model, no download |
| `CallableEmbedder(fn, dimension=None, name="callable")` | Wrap any `texts -> vectors` function (OpenAI, Voyage, Ollama, ONNX…) |
| `get_embedder(name_or_object)` | Resolve `"hash"`, `"hash:256"`, a model name, or an object |

Every engine exposes `name`, `dimension`, `encode(texts)`, `embed(text)` and `clear_cache()`.

---

## `MemoryStore`

```python
from vecforge import MemoryStore, Memory, Session
```

```python
MemoryStore(path="~/.vecforge/memory.db", agent="default", model_name=None,
            embedder=None, encryption_key=None, dedup_threshold=0.92,
            auto_save_every=25, quantum=False, vault=None)
```

| Method | Description |
|---|---|
| `remember(content, kind="fact", importance=0.5, tags=None, metadata=None, session_id=None, source=None, dedup=True) -> Memory` | Store or reinforce |
| `remember_many(items, **defaults) -> list[Memory]` | Bulk store |
| `recall(query, top_k=8, kinds=None, tags=None, session_id=None, since=None, min_importance=0.0, recency_weight=0.15, importance_weight=0.25, alpha=0.6, touch=True) -> list[Memory]` | Ranked recall |
| `get(memory_id)`, `list_memories(kinds, session_id, tags, limit, newest_first)` | Lookup / browse |
| `reinforce(memory_id, delta=0.1)`, `update(memory_id, content=…, importance=…, tags=…, kind=…)` | Edit |
| `forget(memory_id)`, `forget_many(ids)`, `clear(kind=None)` | Delete |
| `start_session(session_id=None, metadata=None) -> Session` | Begin attributing memories |
| `end_session(summary="", session_id=None) -> Session \| None` | Close with a summary memory |
| `last_session()`, `list_sessions(limit)`, `get_session(id)`, `session_memories(id)` | Session history |
| `context_pack(query=None, max_chars=4000, top_k=12, kinds=None, include_last_session=True) -> str` | Prompt-ready briefing |
| `export_json(path=None)`, `import_json(path, dedup=True)` | Portability |
| `stats()`, `save()`, `close()` | Housekeeping |

See the [Memory Guide](memory.md).

---

## Lower-level modules

| Module | Highlights |
|---|---|
| `vecforge.core.bm25.BM25Engine` | Incremental inverted index: `add_documents`, `search(query, top_k, allowed=None)`, `remove(pos)`, `get_scores`, `to_bytes` / `from_bytes` |
| `vecforge.core.indexer.FaissIndexer` | `search(query, top_k, allowed_ids=None, excluded_ids=None)`, `index_type`, `to_bytes` / `from_bytes` |
| `vecforge.core.storage.StorageBackend` | Batch inserts, `get_docs`, sessions, key/value store, `backup`, index snapshots |
| `vecforge.quantum` | `AmplitudeEncoder`, `GroverAmplifier(method="closed_form")`, `QuantumReranker(rerank(..., marked=...))` |
| `vecforge.search.filters.MetadataFilter` | Operators `gte lte gt lt in not_in ne contains any all exists` |
| `vecforge.security.snapshots.SnapshotManager` | WAL-safe snapshots via the SQLite backup API |
| `vecforge.mcp` | `create_server`, `run_server`, `VecForgeMCP`, config snippets |

---

## Exceptions

```python
from vecforge import (
    VecForgeError, VaultEmptyError, VaultClosedError, NamespaceNotFoundError,
    DocumentNotFoundError, DuplicateDocumentError, VecForgePermissionError,
    InvalidAlphaError, EncryptionKeyError, DeletionProtectedError, IngestError,
)
```

All inherit from `VecForgeError`. All messages include actionable guidance.
