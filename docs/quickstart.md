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

That's it. Your data stays on your machine, forever. Reopen the vault later
and it restores its indexes in milliseconds.

> Offline or in CI? `VecForge("my_vault", model_name="hash")` needs no model
> download.

---

## Add Many Documents

```python
db.add_batch(
    [
        "Patient P4821 — Type 2 diabetes, hip fracture admission 2024-09-14",
        "Patient P5102 — Cardiac arrest, ICU transfer 2024-10-01",
        "Patient P3890 — Routine checkup, diabetes follow-up 2023-06-15",
    ],
    metadata_list=[
        {"ward": "7", "year": 2024, "type": "admission"},
        {"ward": "ICU", "year": 2024, "type": "emergency"},
        {"ward": "7", "year": 2023, "type": "checkup"},
    ],
)
```

## Search with Filters

```python
results = db.search("diabetes", top_k=5)                                  # hybrid
results = db.search("diabetes", filters={"year": {"gte": 2024}}, top_k=5)  # filtered
results = db.search("diabetes", namespace="ward_7", top_k=5)              # scoped
results = db.search("elderly diabetic hip fracture", rerank=True, top_k=3) # precise
results = db.search("elderly diabetic hip fracture", quantum_rerank=True)  # Grover
for r in results:
    print(r.score, r.signals, r.text)
```

## Ingest Entire Directories

```python
count = db.ingest("medical_records/")   # PDF, DOCX, TXT, MD, HTML
print(f"Ingested {count} chunks")
```

## Give an AI assistant memory

```python
from vecforge import MemoryStore

mem = MemoryStore("~/.vecforge/memory.db", agent="claude")
mem.remember("User prefers concise answers", kind="preference", importance=0.9)
print(mem.context_pack("how should I answer?"))
```

Then let Claude use it directly:

```bash
pip install "vecforge[mcp]"
vecforge mcp --print-config claude-desktop   # paste into claude_desktop_config.json
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
vecforge search "diabetes" --vault my.db --top-k 5 --quantum
vecforge stats my.db
vecforge memory remember "Deploys happen on Fridays" --kind fact
```

---

## Next Steps

- [Core Concepts](core_concepts.md) — vaults, namespaces, indexes, hybrid search
- [Search Guide](search.md) — alpha, filters, reranking, signals
- [Memory Guide](memory.md) and [MCP Guide](mcp.md) — memory for Claude and other agents
- [Security Guide](security.md) — encryption, RBAC, audit logging
- [API Reference](api_reference.md) — full Python API
