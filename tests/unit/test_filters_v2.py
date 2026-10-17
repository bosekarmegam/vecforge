# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under BSL 1.1 — see LICENSE for details.

"""Tests for the VecForge 2.0 metadata filter operators."""

from __future__ import annotations

from vecforge.search.filters import MetadataFilter


class TestNewOperators:
    def test_contains_string_and_list(self) -> None:
        f = MetadataFilter({"tags": {"contains": "ops"}})
        assert f.matches({"tags": ["ops", "infra"]})
        assert not f.matches({"tags": ["dev"]})
        assert MetadataFilter({"title": {"contains": "Fract"}}).matches(
            {"title": "Hip Fracture"}
        )

    def test_any_and_all(self) -> None:
        assert MetadataFilter({"tags": {"any": ["a", "z"]}}).matches(
            {"tags": ["a", "b"]}
        )
        assert not MetadataFilter({"tags": {"any": ["z"]}}).matches(
            {"tags": ["a", "b"]}
        )
        assert MetadataFilter({"tags": {"all": ["a", "b"]}}).matches(
            {"tags": ["a", "b", "c"]}
        )
        assert not MetadataFilter({"tags": {"all": ["a", "z"]}}).matches(
            {"tags": ["a", "b"]}
        )

    def test_exists(self) -> None:
        f_yes = MetadataFilter({"year": {"exists": True}})
        f_no = MetadataFilter({"year": {"exists": False}})
        assert f_yes.matches({"year": 2024})
        assert not f_yes.matches({"other": 1})
        assert f_no.matches({"other": 1})
        assert not f_no.matches({"year": 2024})
        assert f_no.matches({"year": None})


class TestRobustness:
    def test_incomparable_types_do_not_match_or_raise(self) -> None:
        f = MetadataFilter({"ward": {"gte": 5}})
        assert not f.matches({"ward": "icu"})
        assert f.matches({"ward": 7})

    def test_contains_on_non_container(self) -> None:
        assert not MetadataFilter({"n": {"contains": "x"}}).matches({"n": 5})
        assert not MetadataFilter({"n": {"any": ["x"]}}).matches({"n": 5})
        assert not MetadataFilter({"n": {"all": ["x"]}}).matches({"n": None})

    def test_in_with_unhashable_threshold(self) -> None:
        assert MetadataFilter({"k": {"in": [1, 2]}}).matches({"k": 2})
        assert not MetadataFilter({"k": {"in": 5}}).matches({"k": 2})


class TestFusionTieBreak:
    def test_consensus_wins_ties(self) -> None:
        import numpy as np

        from vecforge.search.hybrid import weighted_linear_fusion_detailed

        # doc 1: best dense, no sparse → 0.5 ; doc 2: worst dense + best sparse → 0.5
        hits = weighted_linear_fusion_detailed(
            dense_ids=np.array([1, 3, 2]),
            dense_scores=np.array([0.9, 0.5, 0.1], dtype=np.float32),
            sparse_ids=[2],
            sparse_scores=[4.0],
            alpha=0.5,
        )
        assert hits[0].doc_index == 2 and hits[0].signals == ["dense", "sparse"]
        assert hits[0].score == hits[1].score
