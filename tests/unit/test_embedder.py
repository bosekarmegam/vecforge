# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under BSL 1.1 — see LICENSE for details.

"""Tests for the pluggable embedding engines."""

from __future__ import annotations

import numpy as np
import pytest

from vecforge.core.embedder import (
    CallableEmbedder,
    Embedder,
    EmbedderLike,
    HashEmbedder,
    get_embedder,
)


class TestHashEmbedder:
    def test_shape_norm_and_determinism(self) -> None:
        emb = HashEmbedder()
        v = emb.embed("patient with diabetes")
        assert v.shape == (384,)
        assert v.dtype == np.float32
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5
        np.testing.assert_array_equal(v, HashEmbedder().embed("patient with diabetes"))

    def test_lexical_similarity(self) -> None:
        emb = HashEmbedder()
        a = emb.embed("patient with type 2 diabetes")
        assert float(a @ emb.embed("diabetic patient")) > float(
            a @ emb.embed("stock market")
        )

    def test_fuzzy_spelling_shares_signal(self) -> None:
        emb = HashEmbedder()
        assert float(emb.embed("diabetes") @ emb.embed("diabetic")) > 0.2

    def test_encode_batch_and_cache(self) -> None:
        emb = HashEmbedder(cache_size=4)
        batch = emb.encode(["a b", "c d", "a b"])
        assert batch.shape == (3, 384)
        np.testing.assert_array_equal(batch[0], batch[2])
        np.testing.assert_array_equal(emb.embed("a b"), batch[0])
        emb.clear_cache()
        np.testing.assert_array_equal(emb.embed("a b"), batch[0])

    def test_empty_text_is_valid(self) -> None:
        v = HashEmbedder().embed("")
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5

    def test_custom_dimension_and_name(self) -> None:
        emb = HashEmbedder(dimension=64)
        assert emb.dimension == 64
        assert emb.name == "hash:64"
        assert emb.embed("x").shape == (64,)

    def test_rejects_tiny_dimension(self) -> None:
        with pytest.raises(ValueError):
            HashEmbedder(dimension=2)

    def test_satisfies_protocol(self) -> None:
        assert isinstance(HashEmbedder(), EmbedderLike)


class TestCallableEmbedder:
    def test_wraps_function_and_normalises(self) -> None:
        emb = CallableEmbedder(lambda ts: [[3.0, 4.0]] * len(ts), name="toy")
        assert emb.dimension == 2
        np.testing.assert_allclose(emb.embed("x"), [0.6, 0.8])
        assert emb.name == "toy"
        assert isinstance(emb, EmbedderLike)

    def test_probes_dimension_lazily(self) -> None:
        calls: list[list[str]] = []

        def fn(ts: list[str]) -> np.ndarray:
            calls.append(ts)
            return np.ones((len(ts), 5), dtype=np.float32)

        emb = CallableEmbedder(fn)
        assert emb.dimension == 5
        assert len(calls) == 1


class TestFactory:
    def test_resolves_hash_specs(self) -> None:
        assert isinstance(get_embedder("hash"), HashEmbedder)
        assert get_embedder("HASH:128").dimension == 128

    def test_bad_hash_spec(self) -> None:
        with pytest.raises(ValueError):
            get_embedder("hash:abc")

    def test_passes_through_objects(self) -> None:
        emb = HashEmbedder()
        assert get_embedder(emb) is emb

    def test_rejects_unknown_objects(self) -> None:
        with pytest.raises(TypeError):
            get_embedder(object())  # type: ignore[arg-type]

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VECFORGE_MODEL", "hash:32")
        assert get_embedder(None).dimension == 32

    def test_model_name_gives_lazy_embedder(self) -> None:
        emb = get_embedder("all-MiniLM-L6-v2")
        assert isinstance(emb, Embedder)
        assert emb.name == "all-MiniLM-L6-v2"
        assert not emb.is_loaded
