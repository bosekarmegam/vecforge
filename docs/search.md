# 🔍 Search Guide

Master VecForge's 4-stage cascading search pipeline.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

---

## Basic Search

```python
from vecforge import VecForge

db = VecForge("my_vault")
results = db.search("diabetes treatment", top_k=10)

for r in results:
    print(f"[{r.score:.4f}] {r.text[:100]}")
```

---

## Tuning Alpha — Semantic vs Keyword

The `alpha` parameter controls the balance between semantic (dense) and keyword (sparse) search:

| Alpha | Behaviour | Best For |
|---|---|---|
| `0.0` | 100% keyword (BM25) | Exact term matching |
| `0.3` | 70% keyword, 30% semantic | Known terminology |
| `0.5` | Balanced (default) | General purpose |
| `0.7` | 70% semantic, 30% keyword | Natural language queries |
| `1.0` | 100% semantic (FAISS) | Conceptual/fuzzy search |

```python
# Exact keyword matching — "find this exact term"
results = db.search("ICD-10 E11.65", alpha=0.0)

# Conceptual search — "find similar ideas"
results = db.search("elderly patient with bone injury", alpha=0.8)
```

---

## Metadata Filtering

Filter results based on metadata values:

### Equality

```python
results = db.search("patient", filters={"type": "NDA"})
```

### Range Operators

```python
# Documents from 2024 onwards
results = db.search("patient", filters={"year": {"gte": 2024}})

# Score between 0.5 and 0.9
results = db.search("patient", filters={"score": {"gte": 0.5, "lte": 0.9}})

# Strict greater/less than
results = db.search("patient", filters={"priority": {"gt": 3, "lt": 10}})
```

### Set Operators

```python
# Include specific values
results = db.search("patient", filters={"status": {"in": ["active", "pending"]}})

# Exclude specific values
results = db.search("patient", filters={"status": {"not_in": ["deleted", "archived"]}})
```

### Not Equal

```python
results = db.search("patient", filters={"type": {"ne": "draft"}})
```

### Combined Filters (AND Logic)

```python
results = db.search(
    "diabetes",
    filters={
        "type": "admission",
        "year": {"gte": 2024},
        "ward": {"in": ["7", "ICU"]},
    },
)
```

---

## Cross-Encoder Reranking

Enable reranking for maximum precision on critical queries:

```python
results = db.search(
    "elderly diabetic hip fracture treatment protocol",
    rerank=True,         # Enable cross-encoder
    top_k=5,
)
```

### When to Use Reranking

| Scenario | Rerank? | Why |
|---|---|---|
| Browsing / exploration | ❌ No | Speed matters more |
| Critical decision support | ✅ Yes | Accuracy matters most |
| Top-1 accuracy needed | ✅ Yes | Cross-encoder excels here |
| Real-time autocomplete | ❌ No | Latency-sensitive |
| Batch analysis | ✅ Yes | Latency acceptable |

Reranking adds ~20-50ms latency but significantly improves relevance ordering.

---

## Namespace-Scoped Search

```python
# Search only within a specific namespace
results = db.search("patient", namespace="hospital_a")

# All results guaranteed to be from hospital_a
for r in results:
    assert r.namespace == "hospital_a"
```

---

## Recency Weighting

Bias results towards recently added documents:

```python
results = db.search(
    "diabetes update",
    recency_weight=0.3,  # 30% recency influence
)
```

| Weight | Effect |
|---|---|
| `0.0` | Recency ignored (default) |
| `0.3` | Mild recency boost |
| `0.7` | Strong recency preference |
| `1.0` | Most recent documents dominate |

---

## Search Result Fields

```python
result = results[0]

result.text       # str — Document text
result.score      # float — Relevance score (higher = better)
result.metadata   # dict — User-provided metadata
result.namespace  # str — Namespace
result.doc_id     # str — UUID
result.modality   # str — "text", "image", etc.
result.timestamp  # float — Unix creation time
```
