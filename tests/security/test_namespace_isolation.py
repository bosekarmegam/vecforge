# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under BSL 1.1 — see LICENSE for details.

"""Security tests for namespace isolation.

Tenants must NEVER see each other's data — this is the #1
security invariant of VecForge multi-tenancy.
"""

from __future__ import annotations

import numpy as np
import pytest

from vecforge.core.storage import StorageBackend
from vecforge.exceptions import NamespaceNotFoundError
from vecforge.security.namespaces import NamespaceManager


class TestNamespaceIsolation:
    """Tests for multi-tenant data isolation."""

    def setup_method(self) -> None:
        self.storage = StorageBackend(":memory:")
        self.ns_mgr = NamespaceManager(self.storage)

    def teardown_method(self) -> None:
        self.storage.close()

    def test_tenants_cannot_see_each_other(self) -> None:
        """Tenants must never see each other's data — ever."""
        self.ns_mgr.create("acme")
        self.ns_mgr.create("globex")

        embedding = np.random.randn(384).astype(np.float32)
        self.storage.insert_doc("Acme secret revenue data", embedding, {}, "acme")
        self.storage.insert_doc("Globex secret strategy doc", embedding, {}, "globex")

        acme_docs = self.storage.get_docs_by_namespace("acme")
        assert len(acme_docs) == 1
        assert acme_docs[0].text == "Acme secret revenue data"
        assert acme_docs[0].namespace == "acme"

        globex_docs = self.storage.get_docs_by_namespace("globex")
        assert len(globex_docs) == 1
        assert globex_docs[0].text == "Globex secret strategy doc"
        assert globex_docs[0].namespace == "globex"

        # CRITICAL: Acme must never see Globex data
        assert not any("globex" in d.text.lower() for d in acme_docs)
        assert not any("acme" in d.text.lower() for d in globex_docs)

    def test_namespace_validation(self) -> None:
        """Accessing non-existent namespace raises error."""
        with pytest.raises(NamespaceNotFoundError):
            self.ns_mgr.validate("nonexistent")

    def test_default_namespace_exists(self) -> None:
        """Default namespace always exists."""
        self.ns_mgr.validate("default")  # should not raise

    def test_create_and_list_namespaces(self) -> None:
        """Create and list namespaces."""
        self.ns_mgr.create("ns_a")
        self.ns_mgr.create("ns_b")
        namespaces = self.ns_mgr.list_all()
        assert "default" in namespaces
        assert "ns_a" in namespaces
        assert "ns_b" in namespaces

    def test_namespace_exists_check(self) -> None:
        """exists() correctly identifies namespaces."""
        self.ns_mgr.create("real_ns")
        assert self.ns_mgr.exists("real_ns")
        assert not self.ns_mgr.exists("fake_ns")

    def test_count_scoped_to_namespace(self) -> None:
        """Counts are correctly scoped per namespace."""
        embedding = np.random.randn(384).astype(np.float32)
        self.ns_mgr.create("team_a")
        self.ns_mgr.create("team_b")

        for i in range(5):
            self.storage.insert_doc(f"doc_a_{i}", embedding, {}, "team_a")
        for i in range(3):
            self.storage.insert_doc(f"doc_b_{i}", embedding, {}, "team_b")

        assert self.storage.count_docs(namespace="team_a") == 5
        assert self.storage.count_docs(namespace="team_b") == 3
        assert self.storage.count_docs() == 8
