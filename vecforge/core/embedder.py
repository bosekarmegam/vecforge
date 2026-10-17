# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
Embedding engines for VecForge.

VecForge 2.0 makes the embedder pluggable so the database adapts to any
AI stack:

* :class:`Embedder` — local sentence-transformers models (default).
* :class:`HashEmbedder` — deterministic feature-hashing embeddings with no
  model download and no PyTorch. Instant startup; ideal for tests, CI,
  constrained machines and as the always-available fallback.
* :class:`CallableEmbedder` — wrap any function ``texts -> vectors`` (an
  OpenAI/Voyage/Ollama client, a custom ONNX model, …).

All engines expose the same :class:`EmbedderLike` protocol, cache repeated
query embeddings in a small LRU, and return L2-normalised float32 vectors
so FAISS inner-product search is cosine similarity.

Select an engine with ``VecForge(model_name=...)``: a sentence-transformers
model name, ``"hash"`` (or ``"hash:512"`` for a custom dimension), or an
object implementing :class:`EmbedderLike`. The ``VECFORGE_MODEL``
environment variable overrides the default model name.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# perf: Default model balances quality and speed for most use cases
_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_ENV_MODEL = "VECFORGE_MODEL"
_DEFAULT_CACHE_SIZE = 512
_HASH_TOKEN_RE = re.compile(r"\w+")


@runtime_checkable
class EmbedderLike(Protocol):
    """Protocol every VecForge embedding engine satisfies."""

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        ...

    @property
    def name(self) -> str:
        """Human-readable engine/model name (persisted in the vault)."""
        ...

    def encode(self, texts: list[str] | str) -> NDArray[np.float32]:
        """Encode one or many texts to a ``(n, dimension)`` float32 array."""
        ...

    def embed(self, text: str) -> NDArray[np.float32]:
        """Encode one text to a ``(dimension,)`` float32 vector."""
        ...


class _QueryCache:
    """Tiny thread-safe LRU for repeated query/document embeddings."""

    def __init__(self, maxsize: int) -> None:
        self._maxsize = max(0, int(maxsize))
        self._data: OrderedDict[str, NDArray[np.float32]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> NDArray[np.float32] | None:
        if self._maxsize == 0:
            return None
        with self._lock:
            vec = self._data.get(key)
            if vec is not None:
                self._data.move_to_end(key)
            return vec

    def put(self, key: str, vec: NDArray[np.float32]) -> None:
        if self._maxsize == 0:
            return
        with self._lock:
            self._data[key] = vec
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class _CachedEmbedderBase:
    """Shared caching + convenience layer for all embedders."""

    _cache: _QueryCache

    def _encode_uncached(self, texts: list[str]) -> NDArray[np.float32]:
        raise NotImplementedError

    @property
    def dimension(self) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    def _encode_with_cache(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        out: list[NDArray[np.float32] | None] = [None] * len(texts)
        misses: list[int] = []
        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is None:
                misses.append(i)
            else:
                out[i] = cached

        if misses:
            # perf: de-duplicate identical texts inside one batch
            unique_texts = list(dict.fromkeys(texts[i] for i in misses))
            vectors = self._encode_uncached(unique_texts)
            lookup = {t: vectors[j] for j, t in enumerate(unique_texts)}
            for i in misses:
                vec = lookup[texts[i]]
                self._cache.put(texts[i], vec)
                out[i] = vec

        return np.stack([v for v in out if v is not None]).astype(np.float32)

    def encode(
        self,
        texts: list[str] | str,
        batch_size: int = 64,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> NDArray[np.float32]:
        """Encode texts into dense embedding vectors.

        Args:
            texts: Single string or list of strings to embed.
            batch_size: Batch size hint for model-backed engines.
            normalize: Kept for API compatibility; vectors are always
                L2-normalised.
            show_progress: Kept for API compatibility.

        Returns:
            NumPy array of shape ``(n_texts, dimension)`` with float32 vectors.

        Performance:
            Time: O(n · d); cache hits are O(1) per text.
        """
        del batch_size, normalize, show_progress  # handled per engine
        if isinstance(texts, str):
            texts = [texts]
        return self._encode_with_cache(list(texts))

    def embed(self, text: str) -> NDArray[np.float32]:
        """Encode one text into a 1-D ``(dimension,)`` float32 vector.

        Args:
            text: Text to embed.

        Returns:
            Unit-norm float32 vector.

        Performance:
            Time: O(d) on cache hit, one model call on miss.
        """
        vec: NDArray[np.float32] = self.encode([text])[0]
        return vec

    def clear_cache(self) -> None:
        """Drop all cached embeddings.

        Performance:
            Time: O(1)
        """
        self._cache.clear()


class Embedder(_CachedEmbedderBase):
    """Local text embedding engine using sentence-transformers.

    Lazily loads the model on first use to keep VecForge init fast.
    All processing runs locally — zero cloud dependency.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        model_name: Name of the sentence-transformers model.
            Defaults to 'all-MiniLM-L6-v2' (384-dim, fast, good quality).
        device: Device to run on ('cpu', 'cuda'). Auto-detected if None.
        cache_size: Number of recent texts whose embeddings are cached.

    Performance:
        Time: O(n * d) where n = number of texts, d = model dimension
        Typical: ~5ms per text on CPU, ~0.5ms on GPU; ~0.01ms on cache hit

    Example:
        >>> embedder = Embedder()
        >>> vectors = embedder.encode(["hello world", "vector search"])
        >>> vectors.shape
        (2, 384)
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str | None = None,
        cache_size: int = _DEFAULT_CACHE_SIZE,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None  # Lazy-loaded SentenceTransformer
        self._dimension: int | None = None
        self._cache = _QueryCache(cache_size)
        self._load_lock = threading.Lock()

    @property
    def name(self) -> str:
        """Return the model name.

        Performance:
            Time: O(1)
        """
        return self._model_name

    @property
    def dimension(self) -> int:
        """Return embedding dimension, loading model if needed.

        Returns:
            Integer dimension of the embedding vectors.

        Performance:
            Time: O(1) after first call
        """
        if self._dimension is None:
            self._load_model()
        assert self._dimension is not None  # guaranteed after _load_model
        return self._dimension

    @property
    def is_loaded(self) -> bool:
        """Return whether the model has been loaded yet.

        Performance:
            Time: O(1)
        """
        return self._model is not None

    def warmup(self) -> None:
        """Load the model eagerly (e.g. in a background thread at startup).

        Performance:
            Time: one-time model load (~1-3 s)
        """
        self._load_model()

    def _load_model(self) -> None:
        """Lazily load the sentence-transformer model.

        Performance:
            Time: O(1) — one-time cost of ~1-3 seconds for model loading
        """
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for VecForge embeddings.\n"
                    "Install with: pip install sentence-transformers\n"
                    "Or use the dependency-free engine: VecForge(model_name='hash')\n"
                    "VecForge by Suneel Bose K · ArcGX TechLabs"
                ) from e

            logger.info("Loading embedding model: %s", self._model_name)
            model = SentenceTransformer(self._model_name, device=self._device)
            self._dimension = int(model.get_sentence_embedding_dimension() or 0)
            self._model = model
            logger.info(
                "Embedding model loaded: %s (dim=%d)",
                self._model_name,
                self._dimension,
            )

    def _encode_uncached(self, texts: list[str]) -> NDArray[np.float32]:
        self._load_model()
        # perf: sentence-transformers handles batching internally
        vectors: NDArray[np.float32] = self._model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashEmbedder(_CachedEmbedderBase):
    """Deterministic feature-hashing text embedder (no model, no download).

    Builds a sparse bag of word unigrams, word bigrams and character
    trigrams, hashes each feature into ``dimension`` buckets with a signed
    hash (to cancel collision bias), applies ``log(1 + tf)`` weighting and
    L2-normalises. Semantically it is a lexical model — it captures shared
    vocabulary and fuzzy spelling, not meaning — but it is fully
    deterministic across processes and machines, needs nothing but NumPy,
    and starts in microseconds.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        dimension: Vector size. Defaults to 384 to match MiniLM vaults.
        cache_size: Number of recent texts whose embeddings are cached.

    Performance:
        Time: O(L) per text, ~20µs for a sentence

    Example:
        >>> emb = HashEmbedder()
        >>> v = emb.embed("patient with diabetes")
        >>> v.shape
        (384,)
        >>> sim = lambda t: float(v @ emb.embed(t))
        >>> sim("diabetic patient") > sim("stock market")
        True
    """

    def __init__(
        self, dimension: int = 384, cache_size: int = _DEFAULT_CACHE_SIZE
    ) -> None:
        if dimension < 8:
            raise ValueError(
                "HashEmbedder dimension must be at least 8.\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            )
        self._dimension = int(dimension)
        self._cache = _QueryCache(cache_size)

    @property
    def name(self) -> str:
        """Return the engine name (``hash:<dimension>``).

        Performance:
            Time: O(1)
        """
        return f"hash:{self._dimension}"

    @property
    def dimension(self) -> int:
        """Return the embedding dimension.

        Performance:
            Time: O(1)
        """
        return self._dimension

    @staticmethod
    def _features(text: str) -> list[str]:
        tokens = _HASH_TOKEN_RE.findall(text.lower())
        feats: list[str] = []
        for tok in tokens:
            feats.append("w:" + tok)
            if len(tok) >= 4:
                padded = f"#{tok}#"
                feats.extend("c:" + padded[i : i + 3] for i in range(len(padded) - 2))
        feats.extend(f"b:{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False))
        if not feats:
            # why: emoji/punctuation-only text must not collapse onto one vector
            feats = ["r:" + ch for ch in text if not ch.isspace()]
        return feats

    def _vector(self, text: str) -> NDArray[np.float32]:
        vec = np.zeros(self._dimension, dtype=np.float64)
        counts: dict[str, int] = {}
        for f in self._features(text):
            counts[f] = counts.get(f, 0) + 1
        for feat, tf in counts.items():
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "little")
            bucket = h % self._dimension
            sign = 1.0 if (h >> 63) & 1 else -1.0
            weight = math.log1p(tf)
            if feat.startswith("c:"):
                weight *= 0.5  # char trigrams: fuzzy signal, lower weight
            vec[bucket] += sign * weight
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            vec[0] = 1.0
            norm = 1.0
        return (vec / norm).astype(np.float32)

    def _encode_uncached(self, texts: list[str]) -> NDArray[np.float32]:
        return np.stack([self._vector(t) for t in texts]).astype(np.float32)


class CallableEmbedder(_CachedEmbedderBase):
    """Adapter that turns any ``texts -> vectors`` function into an engine.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        fn: Callable accepting a list of strings and returning an array-like
            of shape ``(n, dimension)`` (any float dtype).
        dimension: Vector size. If None, it is probed with one call.
        name: Name recorded in the vault (e.g. ``"text-embedding-3-small"``).
        cache_size: Number of recent texts whose embeddings are cached.

    Example:
        >>> emb = CallableEmbedder(lambda ts: [[1.0, 0.0]] * len(ts), name="toy")
        >>> emb.embed("x").shape
        (2,)
    """

    def __init__(
        self,
        fn: Callable[[list[str]], Any],
        dimension: int | None = None,
        name: str = "callable",
        cache_size: int = _DEFAULT_CACHE_SIZE,
    ) -> None:
        self._fn = fn
        self._dimension = dimension
        self._name = name
        self._cache = _QueryCache(cache_size)

    @property
    def name(self) -> str:
        """Return the configured engine name."""
        return self._name

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (probed on first use if unknown)."""
        if self._dimension is None:
            probe = self._encode_uncached(["vecforge dimension probe"])
            self._dimension = int(probe.shape[1])
        return self._dimension

    def _encode_uncached(self, texts: list[str]) -> NDArray[np.float32]:
        raw = np.asarray(self._fn(texts), dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        out: NDArray[np.float32] = (raw / norms).astype(np.float32)
        return out


def get_embedder(
    model: str | EmbedderLike | None = None,
    cache_size: int = _DEFAULT_CACHE_SIZE,
) -> EmbedderLike:
    """Resolve an embedder from a name, an object, or the environment.

    Args:
        model: A sentence-transformers model name, ``"hash"`` /
            ``"hash:<dim>"`` for :class:`HashEmbedder`, an object that
            already implements :class:`EmbedderLike`, or None to use the
            ``VECFORGE_MODEL`` environment variable / default model.
        cache_size: LRU size for engines created here.

    Returns:
        An :class:`EmbedderLike` instance.

    Performance:
        Time: O(1) — models load lazily on first encode.

    Example:
        >>> get_embedder("hash:128").dimension
        128
    """
    if model is None or (isinstance(model, str) and not model.strip()):
        model = os.environ.get(_ENV_MODEL, _DEFAULT_MODEL)

    if not isinstance(model, str):
        if isinstance(model, EmbedderLike):
            return model
        raise TypeError(
            "model must be a model name string or an EmbedderLike object "
            f"(got {type(model).__name__}).\n"
            "VecForge by Suneel Bose K · ArcGX TechLabs"
        )

    lowered = model.strip().lower()
    if lowered == "hash" or lowered.startswith("hash:"):
        dim = 384
        if ":" in lowered:
            try:
                dim = int(lowered.split(":", 1)[1])
            except ValueError as e:
                raise ValueError(
                    f"Invalid hash embedder spec '{model}'. "
                    "Use 'hash' or 'hash:<dim>'.\n"
                    "VecForge by Suneel Bose K · ArcGX TechLabs"
                ) from e
        return HashEmbedder(dimension=dim, cache_size=cache_size)

    return Embedder(model_name=model, cache_size=cache_size)


__all__ = [
    "CallableEmbedder",
    "Embedder",
    "EmbedderLike",
    "HashEmbedder",
    "get_embedder",
]
