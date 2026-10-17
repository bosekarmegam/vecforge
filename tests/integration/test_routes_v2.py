# VecForge — Tests for the 2.0 REST API (hash embedder, no model download)
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from vecforge.server.app import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(str(tmp_path / "api.db"), model_name="hash")
    with TestClient(app) as c:
        yield c


def test_health_reports_version(client: TestClient) -> None:
    from vecforge import __version__

    body = client.get("/api/v1/health").json()
    assert body["version"] == __version__
    assert body["documents"] == 0


def test_batch_get_upsert_patch_similar_delete(client: TestClient) -> None:
    r = client.post(
        "/api/v1/add_batch",
        json={
            "texts": ["patient with diabetes", "hip fracture", "cardiac arrest"],
            "metadata_list": [{"ward": 7}, {"ward": 3}, {"ward": 1}],
        },
    )
    assert r.status_code == 200 and r.json()["count"] == 3
    ids = r.json()["doc_ids"]

    r = client.get(f"/api/v1/docs/{ids[0]}")
    assert r.status_code == 200 and r.json()["metadata"] == {"ward": 7}
    assert client.get("/api/v1/docs/missing").status_code == 404

    r = client.put(
        "/api/v1/docs/custom-1", json={"text": "custom note", "metadata": {"a": 1}}
    )
    assert r.status_code == 200 and r.json()["doc_id"] == "custom-1"
    r = client.patch("/api/v1/docs/custom-1/metadata", json={"metadata": {"b": 2}})
    assert r.json()["metadata"] == {"a": 1, "b": 2}
    assert (
        client.patch("/api/v1/docs/nope/metadata", json={"metadata": {}}).status_code
        == 404
    )

    r = client.get(f"/api/v1/docs/{ids[0]}/similar?top_k=2")
    assert r.status_code == 200 and ids[0] not in [
        x["doc_id"] for x in r.json()["results"]
    ]

    r = client.post(
        "/api/v1/search",
        json={
            "query": "diabetic patient",
            "top_k": 2,
            "quantum_rerank": True,
            "recency_weight": 0.1,
        },
    )
    assert r.status_code == 200
    top = r.json()["results"][0]
    assert top["doc_id"] == ids[0] and "dense" in top["signals"]

    r = client.post(
        "/api/v1/search", json={"query": "patient", "filters": {"ward": {"gte": 5}}}
    )
    assert [x["metadata"]["ward"] for x in r.json()["results"]] == [7]

    assert client.delete(f"/api/v1/docs/{ids[1]}").json()["deleted"] is True
    stats = client.get("/api/v1/stats").json()
    assert (
        stats["documents"] == 3
        and stats["tombstones"] == 1
        and stats["embedding_model"] == "hash:384"
    )
    assert client.post("/api/v1/compact").json()["documents"] == 3
    assert client.post("/api/v1/save").status_code == 200


def test_duplicate_and_namespace_errors(client: TestClient) -> None:
    assert (
        client.post("/api/v1/add", json={"text": "a", "doc_id": "x"}).status_code == 200
    )
    assert (
        client.post("/api/v1/add", json={"text": "b", "doc_id": "x"}).status_code == 409
    )
    assert (
        client.post(
            "/api/v1/search", json={"query": "q", "namespace": "nope"}
        ).status_code
        == 404
    )
    client.post("/api/v1/namespaces", json={"name": "t"})
    client.post("/api/v1/add", json={"text": "tenant doc", "namespace": "t"})
    r = client.delete("/api/v1/namespaces/t")
    assert r.json() == {"deleted": "t", "documents": 1}
    assert "t" not in client.get("/api/v1/namespaces").json()["namespaces"]


def test_memory_endpoints(client: TestClient) -> None:
    r = client.post(
        "/api/v1/memory/remember",
        json={
            "content": "user prefers dark mode",
            "kind": "preference",
            "agent": "bot",
        },
    )
    assert r.status_code == 200 and r.json()["stored"]
    mid = r.json()["memory"]["memory_id"]
    r = client.post(
        "/api/v1/memory/recall", json={"query": "theme preference", "agent": "bot"}
    )
    assert r.json()["count"] == 1
    assert (
        client.post(
            "/api/v1/memory/recall", json={"query": "theme", "agent": "other"}
        ).json()["count"]
        == 0
    )

    s = client.post("/api/v1/memory/sessions/start", json={"agent": "bot"}).json()
    assert s["last_session"] is None
    r = client.post(
        "/api/v1/memory/sessions/end",
        json={
            "agent": "bot",
            "session_id": s["session"]["session_id"],
            "summary": "done",
        },
    )
    assert r.json()["session"]["summary"] == "done"
    assert (
        client.post(
            "/api/v1/memory/sessions/end", json={"agent": "bot", "session_id": "nope"}
        ).status_code
        == 404
    )
    assert len(client.get("/api/v1/memory/sessions?agent=bot").json()["sessions"]) == 1
    ctx = client.get("/api/v1/memory/context?agent=bot&query=theme").json()["context"]
    assert "dark mode" in ctx and "done" in ctx
    assert client.delete(f"/api/v1/memory/{mid}?agent=bot").json()["deleted"]
    assert client.delete(f"/api/v1/memory/{mid}?agent=bot").status_code == 404
    assert (
        client.post(
            "/api/v1/memory/remember", json={"content": "   ", "agent": "bot"}
        ).status_code
        == 400
    )


def test_api_key_protection(tmp_path) -> None:
    app = create_app(str(tmp_path / "secure.db"), model_name="hash", api_key="s3cret")
    with TestClient(app) as c:
        assert c.get("/api/v1/health").status_code == 401
        assert (
            c.get("/api/v1/health", headers={"X-API-Key": "wrong"}).status_code == 401
        )
        assert (
            c.get("/api/v1/health", headers={"X-API-Key": "s3cret"}).status_code == 200
        )


def test_lifespan_closes_vault(tmp_path) -> None:
    app = create_app(str(tmp_path / "life.db"), model_name="hash")
    with TestClient(app) as c:
        c.post("/api/v1/add", json={"text": "x"})
    assert app.state.vault is None


def test_reuses_existing_vault(tmp_path) -> None:
    from vecforge import VecForge

    db = VecForge(":memory:", model_name="hash")
    db.add("shared doc")
    app = create_app(vault=db)
    with TestClient(app) as c:
        assert c.get("/api/v1/stats").json()["documents"] == 1
    assert not db.closed
    db.close()
