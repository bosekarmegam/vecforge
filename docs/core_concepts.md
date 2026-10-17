# 🧠 Core Concepts

Understand VecForge's architecture and key concepts.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

---

## Vaults

A **vault** is VecForge's storage unit — a single (optionally encrypted)
SQLite file containing documents, embeddings, metadata, namespaces, sessions
and a snapshot of the search indexes.

```python
from vecforge import VecForge

db = VecForge("my_vault.db")                                   # persistent
db = VecForge(":memory:")                                      # throw-away
db = VecForge("secure.db", encryption_key=os.environ["VECFORGE_KEY"])
```

### What's inside a vault (schema v2)

- **documents** — text, embedding, metadata, namespace, modality, timestamps, content hash
- **namespaces** — tenant boundaries
- **index_state / index_chunks** — FAISS + BM25 snapshot, position map, tombstones, generation
- **sessions** — memory sessions for AI agents
- **kv** — small key/value store (generation counter, API key registry)

Version-1 vaults are migrated automatically on open.

---

## Documents

| Field | Type | Description |
|---|---|---|
| `doc_id` | `str` | Unique ID (auto UUID or custom) |
| `text` | `str` | Document text |
| `metadata` | `dict` | Free-form key-value pairs |
| `namespace` | `str` | Tenant namespace (default `"default"`) |
| `modality` | `str` | `text`, `image`, `audio`, `memory` |
| `timestamp` | `float` | Creation time |
| `updated_at` | `float \| None` | Last edit |

```python
doc_id = db.add("Patient P4821 — Type 2 diabetes",
                metadata={"ward": "7", "year": 2026}, namespace="hospital_a")
db.upsert("Patient P4821 — Type 2 diabetes, stable", doc_id=doc_id)
db.update_metadata(doc_id, {"reviewed": True})
```

---

## Namespaces

Namespaces provide **multi-tenant data isolation**. Searches restricted to a
namespace are scoped inside the retrievers, so small namespaces are never
starved by large ones.

```python
db.create_namespace("hospital_a")
db.add("Patient A data", namespace="hospital_a")
results = db.search("patient", namespace="hospital_a")
db.delete_namespace("hospital_a")
```

---

## Indexes and persistence

- **FAISS** holds the dense vectors: exact `flat` search below 100k documents,
  graph-based `hnsw` above (configurable with `index_type`).
- **BM25** is VecForge's own incremental inverted index: adding a document is
  O(tokens), searching is O(matching postings), and the index serialises to a
  compact NumPy blob.
- **Deletes are tombstones.** `delete()` is O(1); the indexes are compacted
  automatically once tombstones exceed 20% of the vault, or on `compact()`.
- **Snapshots.** `save()` (and `close()`) store the FAISS index, the BM25
  index, the position map and the tombstones in the vault. The next
  `VecForge(path)` validates the snapshot against a generation counter and
  restores it in milliseconds instead of rebuilding from every embedding.

---

## Hybrid Search

1. **Dense retrieval (FAISS)** — semantic recall.
2. **Sparse retrieval (BM25)** — keyword precision.
3. **Fusion** — `alpha · dense + (1 − alpha) · sparse` on min-max normalised scores.
4. **Hydration + metadata filters** — adaptive candidate window.
5. **Reranking** — cross-encoder and/or Grover amplification (optional).
6. **Recency** — optional, applied last.

```python
db.search("query", alpha=0.0)   # keyword only
db.search("query", alpha=0.5)   # balanced (default)
db.search("query", alpha=1.0)   # semantic only
```

Every `SearchResult` reports `signals` (which retrievers matched),
`dense_score` and `sparse_score`.

---

## Embeddings

Embedders are pluggable (`EmbedderLike`):

```python
db = VecForge("vault", model_name="all-mpnet-base-v2")     # sentence-transformers
db = VecForge("vault", model_name="hash")                  # no download, instant
db = VecForge("vault", embedder=CallableEmbedder(my_fn))   # any API / model
db = VecForge("vault", embedder=ImageEmbedder())           # CLIP space for images
```

Models load lazily and repeated queries hit an LRU cache. The embedder name
is recorded in the snapshot; opening a vault with a different model logs a
warning and `db.reembed()` re-indexes everything.

---

## AI Memory

`MemoryStore` layers remember / recall / sessions / context packs on top of a
vault, one namespace per agent, and `vecforge mcp` exposes it to Claude over
the Model Context Protocol. See the [Memory Guide](memory.md) and the
[MCP Guide](mcp.md).

---

## Context Manager and thread safety

```python
with VecForge("my_vault.db") as db:
    db.add("hello world")
    results = db.search("hello")
# index snapshot saved, connection closed
```

All public `VecForge` methods are serialised with a re-entrant lock, so one
vault can be shared by the REST server's thread pool or several threads in
your application.
