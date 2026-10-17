# 🌐 REST API Reference

FastAPI REST server for VecForge 2.0.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

---

## Starting the Server

```bash
vecforge serve --vault my.db --port 8080                 # binds 127.0.0.1
vecforge serve --vault my.db --host 0.0.0.0 --api-key "$VECFORGE_API_KEY"
```

Interactive docs: `http://localhost:8080/docs`. When an API key is configured
(`--api-key` or `$VECFORGE_API_KEY`) every request must send
`X-API-Key: <key>`; otherwise the server answers `401`.

Endpoints run in FastAPI's thread pool, so embedding and search never block
the event loop. The vault is closed cleanly on shutdown.

---

## Documents

### `POST /api/v1/add`

```json
{"text": "Patient with type 2 diabetes", "metadata": {"ward": "7"},
 "namespace": "default", "doc_id": null, "modality": "text"}
```
→ `{"doc_id": "…", "message": "Document added successfully"}`. `409` if the ID exists.

### `POST /api/v1/add_batch`

```json
{"texts": ["a", "b"], "metadata_list": [{"k": 1}, {"k": 2}], "namespace": "default"}
```
→ `{"doc_ids": ["…", "…"], "count": 2}`

### `POST /api/v1/search`

```json
{"query": "diabetes treatment", "top_k": 5, "alpha": 0.5, "namespace": null,
 "filters": {"year": {"gte": 2024}}, "rerank": false,
 "quantum_rerank": true, "recency_weight": 0.0}
```

```json
{"results": [{"text": "…", "score": 1.0, "metadata": {"year": 2026},
              "namespace": "default", "doc_id": "…", "modality": "text",
              "timestamp": 1760000000.0, "dense_score": 1.0,
              "sparse_score": 0.8, "signals": ["dense", "sparse"]}],
 "count": 1}
```

`404` for an empty vault or unknown namespace.

### `GET /api/v1/docs/{doc_id}` — fetch one document (`404` if missing)
### `PUT /api/v1/docs/{doc_id}` — upsert `{"text", "metadata", "namespace", "modality"}`
### `PATCH /api/v1/docs/{doc_id}/metadata` — `{"metadata": {...}, "merge": true}`
### `DELETE /api/v1/docs/{doc_id}` — `{"deleted": true, "doc_id": "…"}`
### `GET /api/v1/docs/{doc_id}/similar?top_k=10` — neighbours of a document

---

## Vault

### `GET /api/v1/stats`

```json
{"path": "my.db", "version": "2.0.0", "documents": 1500,
 "namespaces": ["default", "ward_7"], "namespace_counts": {"default": 1400, "ward_7": 100},
 "encrypted": false, "quantum": false, "deletion_protection": false,
 "index_vectors": 1502, "index_type": "flat", "tombstones": 2,
 "bm25_documents": 1500, "embedding_model": "all-MiniLM-L6-v2",
 "dimension": 384, "schema_version": 2}
```

### `POST /api/v1/save` — persist the index snapshot
### `POST /api/v1/compact` — rebuild indexes without tombstones
### `GET /api/v1/health` — `{"status": "healthy", "version": "2.0.0", "documents": 1500, …}`

---

## Namespaces

### `POST /api/v1/namespaces` — `{"name": "ward_7"}`
### `GET /api/v1/namespaces` — `{"namespaces": ["default", "ward_7"]}`
### `DELETE /api/v1/namespaces/{name}` — `{"deleted": "ward_7", "documents": 100}`

---

## Memory

All memory endpoints take an `agent` (default `"default"`); each agent has its
own isolated memory namespace.

### `POST /api/v1/memory/remember`

```json
{"content": "User prefers dark mode", "kind": "preference", "importance": 0.8,
 "tags": ["ui"], "session_id": null, "source": "user", "agent": "claude"}
```
→ `{"stored": true, "memory": {...}}` (`stored` is `false` when a duplicate was reinforced).

### `POST /api/v1/memory/recall`

```json
{"query": "theme preference", "top_k": 5, "kinds": ["preference"],
 "tags": null, "session_id": null, "min_importance": 0.0, "agent": "claude"}
```
→ `{"count": 1, "memories": [{...}]}`

### `GET /api/v1/memory/context?agent=claude&query=theme&max_chars=4000`
→ `{"context": "## Memory for agent 'claude' …"}`

### `DELETE /api/v1/memory/{memory_id}?agent=claude`
### `POST /api/v1/memory/sessions/start` — `{"agent": "claude", "metadata": {}}` → session + last session
### `POST /api/v1/memory/sessions/end` — `{"agent": "claude", "session_id": "…", "summary": "…"}`
### `GET /api/v1/memory/sessions?agent=claude&limit=10`

---

## Error codes

| Status | When |
|---|---|
| `400` | Validation error, invalid alpha, empty memory content |
| `401` | Missing or wrong `X-API-Key` |
| `403` | Permission denied (RBAC) or deletion-protected vault |
| `404` | Empty vault, unknown document, namespace, memory or session |
| `409` | Duplicate document ID |

---

## Embedding in your own app

```python
from fastapi import FastAPI
from vecforge import VecForge
from vecforge.server.app import create_app

app = create_app("my.db", api_key="secret", model_name="hash")
# or reuse an open vault (it will not be closed by the app):
db = VecForge("my.db")
app = create_app(vault=db)
```
