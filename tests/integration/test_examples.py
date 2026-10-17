# VecForge — Phase 2 Example Verification
# Runs all example flows with mocked embedder (no PyTorch needed)
# Built by Suneel Bose K · ArcGX TechLabs Private Limited

from __future__ import annotations

from unittest.mock import patch

import numpy as np


def _make_vault():
    from vecforge.core.vault import VecForge

    with patch("vecforge.core.vault.Embedder") as mock_cls:
        mock = mock_cls.return_value
        mock.dimension = 384
        mock.encode.side_effect = lambda text, *a, **kw: np.random.randn(
            1 if isinstance(text, str) else len(text), 384
        ).astype(np.float32)
        vault = VecForge(":memory:", model_name="test")
        vault._embedder = mock
        return vault


def test_hospital_search():
    """Hospital search: add, search, namespace, filter, rerank."""
    db = _make_vault()

    db.create_namespace("ward_7")
    db.create_namespace("ward_icu")

    db.add(
        "Patient P4821 — Type 2 diabetes mellitus, HbA1c 9.2%",
        metadata={"ward": "7", "year": 2026, "type": "admission"},
        namespace="ward_7",
    )
    db.add(
        "Patient P6102 — Cardiac arrest, ROSC achieved",
        metadata={"ward": "ICU", "year": 2026, "type": "emergency"},
        namespace="ward_icu",
    )
    db.add(
        "Patient P4455 — Hip fracture, diabetes comorbidity",
        metadata={"ward": "3", "year": 2026, "type": "admission"},
        namespace="ward_7",
    )

    # Search all
    results = db.search("diabetic patient", top_k=3)
    assert len(results) > 0, "Search returned no results"

    # Namespace search
    results = db.search("emergency", namespace="ward_icu", top_k=2)
    assert all(r.namespace == "ward_icu" for r in results)

    # Metadata filter
    results = db.search("diabetes", filters={"year": 2026}, top_k=3)
    assert all(r.metadata.get("year") == 2026 for r in results)

    # Stats
    stats = db.stats()
    assert stats["documents"] == 3

    db.close()
    print("  ✅ Hospital search — PASS")


def test_legal_documents():
    """Legal doc search: add, filter by type."""
    db = _make_vault()

    db.add(
        "Non-Disclosure Agreement between ArcGX and Client",
        metadata={"type": "NDA", "year": 2025},
    )
    db.add("Software License Agreement v3.0", metadata={"type": "SLA", "year": 2024})
    db.add(
        "Employment contract for engineering team",
        metadata={"type": "contract", "year": 2026},
    )

    results = db.search("agreement", top_k=3)
    assert len(results) > 0

    results = db.search("agreement", filters={"type": "NDA"}, top_k=3)
    assert all(r.metadata.get("type") == "NDA" for r in results)

    db.close()
    print("  ✅ Legal documents — PASS")


def test_gis_data():
    """GIS data search: spatial metadata filtering."""
    db = _make_vault()

    db.add(
        "USGS Landsat 8 Collection — 30m resolution global coverage",
        metadata={"source": "USGS", "resolution": 30, "category": "satellite"},
    )
    db.add(
        "Sentinel-2 MSI — 10m resolution multispectral",
        metadata={"source": "ESA", "resolution": 10, "category": "satellite"},
    )
    db.add(
        "OpenStreetMap vector data — roads and buildings",
        metadata={"source": "OSM", "resolution": 1, "category": "vector"},
    )

    results = db.search("satellite imagery", top_k=3)
    assert len(results) > 0

    results = db.search("high resolution", filters={"category": "satellite"}, top_k=2)
    assert all(r.metadata.get("category") == "satellite" for r in results)

    db.close()
    print("  ✅ GIS data search — PASS")


def test_multi_tenant():
    """Multi-tenant SaaS: namespace isolation."""
    db = _make_vault()

    db.create_namespace("tenant_a")
    db.create_namespace("tenant_b")

    db.add(
        "Tenant A confidential report Q1 2026",
        namespace="tenant_a",
        metadata={"tenant": "a"},
    )
    db.add(
        "Tenant B internal memo — hiring plan",
        namespace="tenant_b",
        metadata={"tenant": "b"},
    )

    # Tenant A can only see their data
    results = db.search("report", namespace="tenant_a", top_k=5)
    assert all(r.namespace == "tenant_a" for r in results)

    # Tenant B can only see their data
    results = db.search("memo", namespace="tenant_b", top_k=5)
    assert all(r.namespace == "tenant_b" for r in results)

    ns = db.list_namespaces()
    assert "tenant_a" in ns and "tenant_b" in ns

    db.close()
    print("  ✅ Multi-tenant SaaS — PASS")


def test_batch_add():
    """Phase 2 batch add API."""
    db = _make_vault()

    texts = [
        "Batch doc one — cardiac monitoring",
        "Batch doc two — diabetes management",
        "Batch doc three — ortho post-op care",
    ]
    meta = [
        {"src": "a"},
        {"src": "b"},
        {"src": "c"},
    ]

    ids = db.add_batch(texts, metadata_list=meta, namespace="bulk")
    assert len(ids) == 3

    stats = db.stats()
    assert stats["documents"] == 3
    assert "bulk" in stats["namespaces"]

    results = db.search("cardiac", top_k=2)
    assert len(results) > 0

    db.close()
    print("  ✅ Batch add (Phase 2) — PASS")


def test_codebase_assistant():
    """Codebase assistant: code doc search."""
    db = _make_vault()

    db.add(
        "def add(self, text): Adds a document to the vault",
        metadata={"file": "vault.py", "type": "function"},
    )
    db.add(
        "class VecForge: Main vector database class",
        metadata={"file": "vault.py", "type": "class"},
    )
    db.add(
        "def search(self, query): Performs hybrid search",
        metadata={"file": "vault.py", "type": "function"},
    )

    results = db.search("how to add documents", top_k=2)
    assert len(results) > 0

    db.close()
    print("  ✅ Codebase assistant — PASS")


if __name__ == "__main__":
    print("=" * 55)
    print("VecForge Phase 2 — Example Verification Suite")
    print("Built by Suneel Bose K · ArcGX TechLabs")
    print("=" * 55)
    print()

    test_hospital_search()
    test_legal_documents()
    test_gis_data()
    test_multi_tenant()
    test_batch_add()
    test_codebase_assistant()

    print()
    print("=" * 55)
    print("🎉 ALL 6 EXAMPLES VERIFIED — Phase 2 is SOLID!")
    print("=" * 55)
