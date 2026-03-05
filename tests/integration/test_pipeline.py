# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under BSL 1.1 — see LICENSE for details.

"""Integration test for the full VecForge add → search pipeline.

Uses mocked embeddings to avoid loading the sentence-transformers
model in CI. This tests the full data flow through all subsystems.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


class TestPipelineIntegration:
    """End-to-end integration test with mocked embeddings."""

    def _make_vault(self):
        """Create a VecForge instance with mocked embedder."""
        from vecforge.core.vault import VecForge

        # why: Mock the embedder to avoid downloading models in tests
        with patch("vecforge.core.vault.Embedder") as mock_embedder_cls:
            mock_embedder = mock_embedder_cls.return_value
            mock_embedder.dimension = 384
            mock_embedder.encode.side_effect = lambda text, *a, **kw: np.random.randn(
                1 if isinstance(text, str) else len(text), 384
            ).astype(np.float32)

            vault = VecForge(":memory:", model_name="test")
            vault._embedder = mock_embedder
            return vault

    def test_add_and_search(self) -> None:
        """Full pipeline: add documents, search, verify results."""
        vault = self._make_vault()

        # Add documents
        doc1 = vault.add("Patient with type 2 diabetes", metadata={"ward": "7"})
        doc2 = vault.add("Hip fracture in elderly patient", metadata={"ward": "3"})
        doc3 = vault.add("Cardiac arrest emergency", metadata={"ward": "ICU"})

        assert doc1
        assert doc2
        assert doc3

        # Search should return results
        results = vault.search("diabetes", top_k=3)
        assert len(results) > 0
        assert all(hasattr(r, "text") for r in results)
        assert all(hasattr(r, "score") for r in results)
        assert all(hasattr(r, "doc_id") for r in results)

        vault.close()

    def test_namespace_search_isolation(self) -> None:
        """Search results respect namespace boundaries."""
        vault = self._make_vault()

        vault.add("Secret A data", namespace="team_a")
        vault.add("Secret B data", namespace="team_b")

        results_a = vault.search("secret", namespace="team_a", top_k=10)
        for r in results_a:
            assert r.namespace == "team_a"

        results_b = vault.search("secret", namespace="team_b", top_k=10)
        for r in results_b:
            assert r.namespace == "team_b"

        vault.close()

    def test_metadata_filtering(self) -> None:
        """Metadata filters are applied correctly."""
        vault = self._make_vault()

        vault.add("Old doc", metadata={"year": 2020})
        vault.add("New doc", metadata={"year": 2024})

        results = vault.search(
            "doc",
            filters={"year": {"gte": 2023}},
            top_k=10,
        )
        for r in results:
            assert r.metadata.get("year", 0) >= 2023

        vault.close()

    def test_stats(self) -> None:
        """Stats returns correct vault information."""
        vault = self._make_vault()

        vault.add("doc 1")
        vault.add("doc 2")

        info = vault.stats()
        assert info["documents"] == 2
        assert "default" in info["namespaces"]
        assert info["built_by"] == "Suneel Bose K · ArcGX TechLabs Private Limited"

        vault.close()

    def test_delete(self) -> None:
        """Delete removes document from vault."""
        vault = self._make_vault()

        doc_id = vault.add("to be deleted")
        assert vault.stats()["documents"] == 1

        vault.delete(doc_id)
        assert vault.stats()["documents"] == 0

        vault.close()

    def test_deletion_protection(self) -> None:
        """Deletion protection prevents deletes."""
        from vecforge.exceptions import DeletionProtectedError

        vault = self._make_vault()
        vault._deletion_protection = True

        doc_id = vault.add("protected doc")
        with pytest.raises(DeletionProtectedError):
            vault.delete(doc_id)

        vault.close()

    def test_context_manager(self) -> None:
        """Context manager works for auto-close."""
        vault = self._make_vault()
        with vault:
            vault.add("context test")
            results = vault.search("context", top_k=1)
            assert len(results) > 0

    def test_vault_repr(self) -> None:
        """Repr shows useful info."""
        vault = self._make_vault()
        repr_str = repr(vault)
        assert "VecForge" in repr_str
        vault.close()

    def test_empty_vault_search_raises(self) -> None:
        """Searching empty vault raises VaultEmptyError."""
        from vecforge.exceptions import VaultEmptyError

        vault = self._make_vault()

        with pytest.raises(VaultEmptyError):
            vault.search("anything")

        vault.close()

    def test_invalid_alpha_raises(self) -> None:
        """Alpha outside [0, 1] raises InvalidAlphaError."""
        from vecforge.exceptions import InvalidAlphaError

        vault = self._make_vault()
        vault.add("test")

        with pytest.raises(InvalidAlphaError):
            vault.search("test", alpha=1.5)

        vault.close()
