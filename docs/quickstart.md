# ⚡ Quickstart Guide

Get VecForge running in under 5 minutes. No API keys. No cloud. No config files.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

---

## Install

```bash
pip install vecforge
```

## Your First Vault — 5 Lines

```python
from vecforge import VecForge

db = VecForge("my_vault")
db.add("Patient admitted with type 2 diabetes", metadata={"ward": "7"})
results = db.search("diabetic patient")
print(results[0].text)
# → 'Patient admitted with type 2 diabetes'
```

That's it. Your data stays on your machine, forever.

---

## Add Multiple Documents

```python
from vecforge import VecForge

db = VecForge("medical_vault")

# Add documents with rich metadata
db.add(
    "Patient P4821 — Type 2 diabetes, hip fracture admission 2024-09-14",
    metadata={"ward": "7", "year": 2024, "type": "admission"},
)
db.add(
    "Patient P5102 — Cardiac arrest, ICU transfer 2024-10-01",
    metadata={"ward": "ICU", "year": 2024, "type": "emergency"},
)
db.add(
    "Patient P3890 — Routine checkup, diabetes follow-up 2023-06-15",
    metadata={"ward": "7", "year": 2023, "type": "checkup"},
)
```

## Search with Filters

```python
# Hybrid search — semantic + keyword
results = db.search("diabetes", top_k=5)

# Filter by metadata
results = db.search(
    "diabetes",
    filters={"year": {"gte": 2024}},
    top_k=5,
)

# Namespace-scoped search
results = db.search("diabetes", namespace="ward_7", top_k=5)

# High-precision with reranking
results = db.search("elderly diabetic hip fracture", rerank=True, top_k=3)
```

## Ingest Entire Directories

```python
# Auto-detects format: PDF, DOCX, TXT, MD, HTML
count = db.ingest("medical_records/")
print(f"Ingested {count} chunks")
```

## Enable Encryption

```python
import os

db = VecForge(
    "secure_vault",
    encryption_key=os.environ["VECFORGE_KEY"],
    audit_log="audit.jsonl",
)
```

## Use the CLI

```bash
vecforge ingest my_docs/ --vault my.db
vecforge search "diabetes" --vault my.db --top-k 5
vecforge stats my.db
```

---

## Next Steps

- [Core Concepts](core_concepts.md) — Understand vaults, namespaces, and hybrid search
- [Search Guide](search.md) — Master the 4-stage search pipeline
- [Security Guide](security.md) — Encryption, RBAC, and audit logging
- [API Reference](api_reference.md) — Full Python API
