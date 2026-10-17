# 🔍 Search Guide

Master VecForge's cascading search pipeline.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

---

## Basic Search

```python
from vecforge import VecForge

db = VecForge("my_vault")
results = db.search("diabetes treatment", top_k=10)

for r in results:
    print(f"[{r.score:.4f}] {r.signals} {r.text[:100]}")
```

`r.signals` tells you which retrievers matched: `["dense"]`, `["sparse"]`
or both. `r.dense_score` and `r.sparse_score` carry the normalised
per-retriever scores (`None` when that retriever did not return the
document).

---

## The pipeline

1. **Dense retrieval (FAISS).** The query embedding is compared with every
   document vector. Namespace restrictions and deleted documents are applied
   *inside* FAISS with ID selectors, so a namespace with 10 documents in a
   vault of a million still returns all 10.
2. **Sparse retrieval (BM25).** VecForge's incremental inverted index
   scores keyword matches in O(postings), scoped to the same namespace.
3. **Fusion.** Both score sets are min-max normalised over the candidate
   window and combined as `alpha · dense + (1 − alpha) · sparse`.
4. **Hydration + metadata filters.** Candidates are loaded in one batched
   query and filtered. If a filter rejects too many candidates, the window
   widens (×4, up to the whole namespace) until `top_k` matches are found.
5. **Reranking (optional).** Cross-encoder (`rerank=True`) and/or Grover
   amplification with a consensus oracle (`quantum_rerank=True`).
6. **Recency (optional).** Applied last, on min-max normalised scores, so it
   is never discarded by a reranker.

---

## Tuning Alpha — Semantic vs Keyword

| Alpha | Behaviour | Best For |
|---|---|---|
| `0.0` | 100% keyword (BM25) | Exact term matching, IDs, codes |
| `0.3` | 70% keyword, 30% semantic | Known terminology |
| `0.5` | Balanced (default) | General purpose |
| `0.7` | 70% semantic, 30% keyword | Natural language queries |
| `1.0` | 100% semantic (FAISS) | Conceptual/fuzzy search, image queries |

```python
results = db.search("ICD-10 E11.65", alpha=0.0)
results = db.search("elderly patient with bone injury", alpha=0.8)
```

When one retriever returns nothing (for example BM25 with no keyword
overlap) the other receives the full weight, so scores stay on the `[0, 1]`
scale.

---

## Metadata Filtering

```python
db.search("patient", filters={"type": "NDA"})                        # equality
db.search("patient", filters={"year": {"gte": 2024}})                # range
db.search("patient", filters={"score": {"gte": 0.5, "lte": 0.9}})
db.search("patient", filters={"status": {"in": ["active", "pending"]}})
db.search("patient", filters={"status": {"not_in": ["deleted"]}})
db.search("patient", filters={"type": {"ne": "draft"}})
db.search("patient", filters={"tags": {"contains": "ops"}})          # list or substring
db.search("patient", filters={"tags": {"any": ["ops", "infra"]}})    # list overlap
db.search("patient", filters={"tags": {"all": ["ops", "prod"]}})
db.search("patient", filters={"reviewed": {"exists": False}})        # key missing / None
```

Several keys combine with AND. Comparisons between incompatible types (a
string field against a numeric threshold) simply do not match — they never
raise.

---

## Cross-Encoder Reranking

```python
results = db.search("elderly diabetic hip fracture treatment protocol", rerank=True, top_k=5)
```

| Scenario | Rerank? | Why |
|---|---|---|
| Browsing / exploration | ❌ No | Speed matters more |
| Critical decision support | ✅ Yes | Accuracy matters most |
| Top-1 accuracy needed | ✅ Yes | Cross-encoder excels here |
| Real-time autocomplete | ❌ No | Latency-sensitive |

Reranking adds ~20–50 ms. Combine with `quantum_rerank=True` to run the
cross-encoder only on the Grover-selected survivors.

---

## Quantum-Inspired Reranking

```python
results = db.search("space exploration", quantum_rerank=True)
```

Grover amplitude amplification boosts candidates that both retrievers agreed
on and sharpens the score gap, keeping the best candidate first. See the
[Quantum Guide](quantum.md).

---

## Namespace-Scoped Search

```python
results = db.search("patient", namespace="hospital_a")
assert all(r.namespace == "hospital_a" for r in results)
```

---

## Recency Weighting

```python
results = db.search("diabetes update", recency_weight=0.3)
```

| Weight | Effect |
|---|---|
| `0.0` | Recency ignored (default) |
| `0.3` | Mild recency boost |
| `0.7` | Strong recency preference |
| `1.0` | Most recent documents dominate |

---

## More ways to query

```python
db.search_batch(["diabetes", "cardiac arrest"], top_k=5)   # one embedding call
db.similar(doc_id, top_k=5)                                 # neighbours of a document
db.search("", query_vector=vec, alpha=1.0)                  # precomputed vector (images)
db.search("diabetes", exclude_doc_ids=[seen_id])
```

---

## Search Result Fields

```python
r.text          # str — document text
r.score         # float — relevance (higher = better)
r.metadata      # dict
r.namespace     # str
r.doc_id        # str
r.modality      # "text", "image", "audio", "memory"
r.timestamp     # float — creation time
r.dense_score   # float | None
r.sparse_score  # float | None
r.signals       # ["dense", "sparse"]
r.updated_at    # float | None
r.to_dict()     # JSON-serialisable
```
