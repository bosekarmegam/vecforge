# 🌐 REST API Reference

FastAPI REST server endpoints for VecForge.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

---

## Starting the Server

```bash
vecforge serve --vault my.db --port 8080
```

Interactive docs available at: `http://localhost:8080/docs`

---

## Endpoints

### `POST /api/v1/add` — Add Document

```bash
curl -X POST http://localhost:8080/api/v1/add \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Patient with type 2 diabetes",
    "metadata": {"ward": "7", "year": 2026},
    "namespace": "default"
  }'
```

**Response:**
```json
{"doc_id": "a1b2c3d4-...", "message": "Document added successfully"}
```

---

### `POST /api/v1/search` — Search Vault

```bash
curl -X POST http://localhost:8080/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "diabetes treatment",
    "top_k": 5,
    "alpha": 0.5,
    "namespace": null,
    "filters": {"year": {"gte": 2024}},
    "rerank": false
  }'
```

**Response:**
```json
{
  "results": [
    {
      "text": "Patient with type 2 diabetes",
      "score": 0.8542,
      "metadata": {"ward": "7", "year": 2026},
      "namespace": "default",
      "doc_id": "a1b2c3d4-...",
      "modality": "text"
    }
  ],
  "count": 1
}
```

---

### `DELETE /api/v1/docs/{doc_id}` — Delete Document

```bash
curl -X DELETE http://localhost:8080/api/v1/docs/a1b2c3d4-...
```

**Response:**
```json
{"deleted": true, "doc_id": "a1b2c3d4-..."}
```

---

### `GET /api/v1/stats` — Vault Statistics

```bash
curl http://localhost:8080/api/v1/stats
```

**Response:**
```json
{
  "path": "my.db",
  "documents": 1500,
  "namespaces": ["default", "ward_7"],
  "encrypted": false,
  "quantum": false
}
```

---

### `POST /api/v1/namespaces` — Create Namespace

```bash
curl -X POST http://localhost:8080/api/v1/namespaces \
  -H "Content-Type: application/json" \
  -d '{"name": "ward_7"}'
```

---

### `GET /api/v1/namespaces` — List Namespaces

```bash
curl http://localhost:8080/api/v1/namespaces
```

**Response:**
```json
{"namespaces": ["default", "ward_7"]}
```

---

### `GET /api/v1/health` — Health Check

```bash
curl http://localhost:8080/api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "VecForge",
  "version": "0.2.0",
  "built_by": "Suneel Bose K · ArcGX TechLabs"
}
```
