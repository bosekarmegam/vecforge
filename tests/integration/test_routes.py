# VecForge — Tests for REST API
import pytest
from fastapi.testclient import TestClient

from vecforge.server.app import create_app


@pytest.fixture
def client(tmp_path):
    vault_path = str(tmp_path / "test_api.db")
    app = create_app(vault_path)
    return TestClient(app)


def test_api_health(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_api_namespaces(client):
    response = client.post("/api/v1/namespaces", json={"name": "test_ns"})
    assert response.status_code == 200

    response = client.get("/api/v1/namespaces")
    assert response.status_code == 200
    assert "test_ns" in response.json()["namespaces"]


def test_api_add_and_search(client):
    # Add doc
    payload = {
        "text": "This is a test medical record for John Doe.",
        "metadata": {"type": "medical"},
        "namespace": "default",
    }
    response = client.post("/api/v1/add", json=payload)
    assert response.status_code == 200
    doc_id = response.json()["doc_id"]

    # Search doc
    search_payload = {
        "query": "medical record",
        "top_k": 1,
        "namespace": "default",
        "filters": {"type": "medical"},
    }
    response = client.post("/api/v1/search", json=search_payload)
    assert response.status_code == 200
    results = response.json()
    assert results["count"] == 1
    assert results["results"][0]["doc_id"] == doc_id

    # Stats
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    assert response.json()["documents"] == 1

    # Delete
    response = client.delete(f"/api/v1/docs/{doc_id}")
    assert response.status_code == 200
    assert response.json()["deleted"] is True


def test_api_errors(client):
    # Search empty vault
    search_payload = {"query": "anything"}
    response = client.post("/api/v1/search", json=search_payload)
    assert response.status_code == 404

    # Delete non-existent
    response = client.delete("/api/v1/docs/does-not-exist")
    assert response.status_code == 404
