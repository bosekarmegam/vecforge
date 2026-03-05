# 🧠 Core Concepts

Understand VecForge's architecture and key concepts.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

---

## Vaults

A **vault** is VecForge's storage unit — a single encrypted database file containing all your documents, embeddings, metadata, and indexes.

```python
from vecforge import VecForge

# File-based vault (persistent)
db = VecForge("my_vault.db")

# In-memory vault (for testing)
db = VecForge(":memory:")

# Encrypted vault
db = VecForge("secure.db", encryption_key=os.environ["VECFORGE_KEY"])
```

### What's Inside a Vault?
- **Documents** — Original text with metadata
- **Embeddings** — Dense vectors from sentence-transformers
- **FAISS Index** — Fast vector similarity search index
- **BM25 Index** — Keyword search index (in-memory, rebuilt on load)
- **Namespaces** — Tenant isolation boundaries
- **Audit Log** — Record of all operations (optional)

---

## Documents

Documents are the basic unit of storage. Each document has:

| Field | Type | Description |
|---|---|---|
| `doc_id` | `str` | Unique UUID (auto-generated or custom) |
| `text` | `str` | The document text content |
| `metadata` | `dict` | Free-form key-value pairs |
| `namespace` | `str` | Tenant namespace (default: `"default"`) |
| `modality` | `str` | Content type: `text`, `image`, `audio` |
| `timestamp` | `float` | Creation time (Unix timestamp) |

```python
doc_id = db.add(
    "Patient P4821 — Type 2 diabetes",
    metadata={"ward": "7", "year": 2026, "priority": "high"},
    namespace="hospital_a",
)
```

---

## Namespaces

Namespaces provide **multi-tenant data isolation**. Documents in one namespace are invisible to searches in another.

```python
# Create namespaces
db.create_namespace("hospital_a")
db.create_namespace("hospital_b")

# Add data to specific namespaces
db.add("Patient A data", namespace="hospital_a")
db.add("Patient B data", namespace="hospital_b")

# Searches are isolated — hospital_a never sees hospital_b
results = db.search("patient", namespace="hospital_a")
# Only returns hospital_a documents
```

### Isolation Guarantee
- All SQL queries are scoped with `WHERE namespace = ?`
- No API exists to bypass namespace boundaries
- Namespace isolation is tested in the security test suite

---

## Hybrid Search

VecForge uses a **4-stage cascading search pipeline** combining dense (semantic) and sparse (keyword) retrieval:

### Stage 1: Dense Retrieval (FAISS)
Embeds the query using sentence-transformers and finds nearest neighbours in the FAISS index. High recall — catches semantically similar documents even without exact keyword matches.

### Stage 2: Sparse Merge (BM25)
Runs BM25 keyword search and merges results with dense scores using **Reciprocal Rank Fusion (RRF)**. Boosts precision for keyword-heavy queries.

### Stage 3: Metadata Filtering
Applies user-specified filters on metadata fields. Supports operators: `eq`, `gte`, `lte`, `gt`, `lt`, `in`, `not_in`, `ne`.

### Stage 4: Cross-Encoder Reranking (Optional)
Reranks top candidates using a cross-encoder model for maximum precision. Adds ~20-50ms latency but significantly improves result quality.

### The Alpha Parameter

```python
# alpha controls semantic vs keyword balance
results = db.search("query", alpha=0.0)   # 100% keyword (BM25 only)
results = db.search("query", alpha=0.5)   # 50/50 balanced (default)
results = db.search("query", alpha=1.0)   # 100% semantic (FAISS only)
results = db.search("query", alpha=0.7)   # 70% semantic, 30% keyword
```

---

## Embeddings

VecForge uses **sentence-transformers** for local text embedding. Models are downloaded once and cached.

### Default Model
`all-MiniLM-L6-v2` — 384 dimensions, fast, good quality.

### Custom Models
```python
db = VecForge("vault", model_name="all-mpnet-base-v2")   # 768-dim, higher quality
db = VecForge("vault", model_name="all-MiniLM-L6-v2")    # 384-dim, faster
```

### Lazy Loading
Models load on first `add()` or `search()` call, keeping VecForge init instant.

---

## SearchResult

Every search returns a list of `SearchResult` objects:

```python
results = db.search("diabetes")

for r in results:
    print(r.text)       # Document text
    print(r.score)      # Relevance score (higher = better)
    print(r.metadata)   # {"ward": "7", "year": 2026}
    print(r.namespace)  # "default"
    print(r.doc_id)     # "a1b2c3d4-..."
    print(r.modality)   # "text"
    print(r.timestamp)  # 1709654400.0
```

---

## Context Manager

VecForge supports Python's `with` statement for automatic cleanup:

```python
with VecForge("my_vault.db") as db:
    db.add("hello world")
    results = db.search("hello")
# Automatically saves index and closes connection
```
