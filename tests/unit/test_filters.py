# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under BSL 1.1 — see LICENSE for details.

"""Unit tests for metadata filtering."""

from __future__ import annotations

from vecforge.search.filters import MetadataFilter


class TestMetadataFilter:
    """Tests for metadata filter operators."""

    def test_equality_match(self) -> None:
        """Simple equality filter works."""
        f = MetadataFilter({"type": "NDA"})
        assert f.matches({"type": "NDA"})
        assert not f.matches({"type": "Contract"})

    def test_missing_key(self) -> None:
        """Missing key fails the filter."""
        f = MetadataFilter({"type": "NDA"})
        assert not f.matches({"category": "legal"})

    def test_gte_operator(self) -> None:
        """Greater-than-or-equal operator."""
        f = MetadataFilter({"year": {"gte": 2023}})
        assert f.matches({"year": 2024})
        assert f.matches({"year": 2023})
        assert not f.matches({"year": 2022})

    def test_lte_operator(self) -> None:
        """Less-than-or-equal operator."""
        f = MetadataFilter({"year": {"lte": 2023}})
        assert f.matches({"year": 2022})
        assert f.matches({"year": 2023})
        assert not f.matches({"year": 2024})

    def test_gt_lt_operators(self) -> None:
        """Strict greater-than and less-than."""
        f = MetadataFilter({"score": {"gt": 0.5, "lt": 1.0}})
        assert f.matches({"score": 0.7})
        assert not f.matches({"score": 0.5})  # not strictly greater
        assert not f.matches({"score": 1.0})  # not strictly less

    def test_in_operator(self) -> None:
        """In-list operator."""
        f = MetadataFilter({"status": {"in": ["active", "pending"]}})
        assert f.matches({"status": "active"})
        assert f.matches({"status": "pending"})
        assert not f.matches({"status": "closed"})

    def test_not_in_operator(self) -> None:
        """Not-in-list operator."""
        f = MetadataFilter({"status": {"not_in": ["deleted", "archived"]}})
        assert f.matches({"status": "active"})
        assert not f.matches({"status": "deleted"})

    def test_ne_operator(self) -> None:
        """Not-equal operator."""
        f = MetadataFilter({"type": {"ne": "draft"}})
        assert f.matches({"type": "final"})
        assert not f.matches({"type": "draft"})

    def test_combined_filters(self) -> None:
        """Multiple filter conditions (AND logic)."""
        f = MetadataFilter(
            {
                "type": "NDA",
                "year": {"gte": 2023},
                "status": {"in": ["active", "pending"]},
            }
        )
        assert f.matches({"type": "NDA", "year": 2024, "status": "active"})
        assert not f.matches({"type": "NDA", "year": 2022, "status": "active"})
        assert not f.matches({"type": "Contract", "year": 2024, "status": "active"})

    def test_empty_filter_matches_all(self) -> None:
        """Empty filter matches everything."""
        f = MetadataFilter({})
        assert f.matches({"anything": "goes"})
        assert f.matches({})

    def test_range_filter(self) -> None:
        """Range with both gte and lte."""
        f = MetadataFilter({"score": {"gte": 0.5, "lte": 0.9}})
        assert f.matches({"score": 0.7})
        assert f.matches({"score": 0.5})
        assert f.matches({"score": 0.9})
        assert not f.matches({"score": 0.4})
        assert not f.matches({"score": 1.0})
