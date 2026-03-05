# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under BSL 1.1 — see LICENSE for details.

"""Unit tests for SQLite storage backend."""

from __future__ import annotations

import numpy as np

from vecforge.core.storage import StorageBackend


class TestStorageBackend:
    """Tests for StorageBackend CRUD operations."""

    def setup_method(self) -> None:
        self.storage = StorageBackend(":memory:")

    def teardown_method(self) -> None:
        self.storage.close()

    def test_insert_and_get_doc(self) -> None:
        """Insert and retrieve a document."""
        embedding = np.random.randn(384).astype(np.float32)
        doc = self.storage.insert_doc(
            text="hello world",
            embedding=embedding,
            metadata={"source": "test"},
            namespace="default",
        )
        assert doc.doc_id
        assert doc.text == "hello world"

        retrieved = self.storage.get_doc(doc.doc_id)
        assert retrieved is not None
        assert retrieved.text == "hello world"
        assert retrieved.metadata == {"source": "test"}
        np.testing.assert_array_almost_equal(retrieved.embedding, embedding)

    def test_get_nonexistent_doc(self) -> None:
        """Getting a non-existent doc returns None."""
        result = self.storage.get_doc("nonexistent")
        assert result is None

    def test_delete_doc(self) -> None:
        """Delete a document."""
        embedding = np.random.randn(384).astype(np.float32)
        doc = self.storage.insert_doc("to delete", embedding, {}, "default")
        assert self.storage.delete_doc(doc.doc_id)
        assert self.storage.get_doc(doc.doc_id) is None

    def test_delete_nonexistent(self) -> None:
        """Deleting non-existent doc returns False."""
        assert not self.storage.delete_doc("nonexistent")

    def test_count_docs(self) -> None:
        """Count documents total and by namespace."""
        embedding = np.random.randn(384).astype(np.float32)
        self.storage.insert_doc("doc1", embedding, {}, "ns_a")
        self.storage.insert_doc("doc2", embedding, {}, "ns_a")
        self.storage.insert_doc("doc3", embedding, {}, "ns_b")

        assert self.storage.count_docs() == 3
        assert self.storage.count_docs(namespace="ns_a") == 2
        assert self.storage.count_docs(namespace="ns_b") == 1

    def test_namespace_scoped_queries(self) -> None:
        """Documents are correctly scoped to namespaces."""
        embedding = np.random.randn(384).astype(np.float32)
        self.storage.create_namespace("tenant_a")
        self.storage.create_namespace("tenant_b")

        self.storage.insert_doc("secret A", embedding, {}, "tenant_a")
        self.storage.insert_doc("secret B", embedding, {}, "tenant_b")

        docs_a = self.storage.get_docs_by_namespace("tenant_a")
        docs_b = self.storage.get_docs_by_namespace("tenant_b")

        assert len(docs_a) == 1
        assert docs_a[0].text == "secret A"
        assert len(docs_b) == 1
        assert docs_b[0].text == "secret B"

    def test_list_namespaces(self) -> None:
        """List all namespaces."""
        self.storage.create_namespace("ns_x")
        self.storage.create_namespace("ns_y")
        namespaces = self.storage.list_namespaces()
        assert "default" in namespaces
        assert "ns_x" in namespaces
        assert "ns_y" in namespaces

    def test_faiss_index_persistence(self) -> None:
        """Save and load FAISS index data."""
        data = b"fake_faiss_index_data"
        self.storage.save_faiss_index(data, dimension=384, count=100)

        loaded = self.storage.load_faiss_index()
        assert loaded is not None
        index_data, dim, count = loaded
        assert index_data == data
        assert dim == 384
        assert count == 100

    def test_faiss_index_not_saved(self) -> None:
        """Loading unsaved index returns None."""
        assert self.storage.load_faiss_index() is None
