# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
FAISS index management for VecForge.

Provides nearest-neighbour search using FAISS with two index types:

* ``flat``  — ``IndexFlatIP``, exact inner-product search. Best for
  collections up to ~100k vectors.
* ``hnsw``  — ``IndexHNSWFlat``, graph-based approximate search with
  sub-millisecond queries at millions of vectors.

``index_type="auto"`` (the default) starts flat and upgrades itself to
HNSW once the collection crosses :data:`AUTO_HNSW_THRESHOLD` vectors.

Searches accept an *allowed* set and an *excluded* set of positions,
implemented with FAISS ID selectors, so namespace scoping and tombstoned
deletes are applied **inside** the index rather than by post-filtering a
fixed top-k (which silently lost recall in VecForge 1.x).

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from typing import Literal

import faiss
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

IndexType = Literal["flat", "hnsw", "auto"]

# perf: Switch flat → HNSW when the collection exceeds this many vectors
AUTO_HNSW_THRESHOLD = 50_000
_HNSW_M = 32
_HNSW_EF_CONSTRUCTION = 200
_HNSW_EF_SEARCH = 64
# why: graph search under a restrictive filter can miss members; small
# allowed sets (one namespace) are searched exactly instead — still cheap
_EXACT_SUBSET_MAX = 20_000


class FaissIndexer:
    """FAISS-based vector index for fast nearest-neighbour search.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        dimension: Embedding vector dimension.
        index_type: ``"flat"`` (exact), ``"hnsw"`` (approximate) or
            ``"auto"`` (flat, upgrading to HNSW past the threshold).
        hnsw_m: HNSW graph degree. Defaults to 32.
        ef_search: HNSW search breadth (higher = better recall, slower).
        ef_construction: HNSW build breadth.

    Performance:
        Flat index: O(N * d) search — exact, ~30ms at 1M×384 on CPU
        HNSW index: O(log N * d) — approximate, <1ms at 1M
        Namespace / tombstone filtering: applied inside FAISS via selectors

    Example:
        >>> indexer = FaissIndexer(dimension=384)
        >>> indexer.add(np.random.randn(100, 384).astype(np.float32))
        >>> distances, indices = indexer.search(query_vec, top_k=5)
    """

    def __init__(
        self,
        dimension: int,
        index_type: IndexType = "auto",
        hnsw_m: int = _HNSW_M,
        ef_search: int = _HNSW_EF_SEARCH,
        ef_construction: int = _HNSW_EF_CONSTRUCTION,
    ) -> None:
        if index_type not in ("flat", "hnsw", "auto"):
            raise ValueError(
                f"Unknown index_type '{index_type}'. Use 'flat', 'hnsw' or 'auto'.\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            )
        self._dimension = int(dimension)
        self._requested_type: IndexType = index_type
        self._hnsw_m = int(hnsw_m)
        self._ef_search = int(ef_search)
        self._ef_construction = int(ef_construction)
        self._count = 0
        self._index: faiss.Index
        self._is_hnsw = False
        if index_type == "hnsw":
            self._index = self._new_hnsw()
            self._is_hnsw = True
        else:
            self._index = faiss.IndexFlatIP(self._dimension)

    # ─── Construction helpers ───

    def _new_hnsw(self) -> faiss.Index:
        index = faiss.IndexHNSWFlat(
            self._dimension, self._hnsw_m, faiss.METRIC_INNER_PRODUCT
        )
        index.hnsw.efConstruction = self._ef_construction
        index.hnsw.efSearch = self._ef_search
        return index

    def _maybe_upgrade(self) -> None:
        """Upgrade flat → HNSW when ``auto`` crosses the threshold."""
        if (
            self._requested_type != "auto"
            or self._is_hnsw
            or self._count < AUTO_HNSW_THRESHOLD
        ):
            return
        logger.info(
            "FaissIndexer: %d vectors — upgrading flat index to HNSW (one-time)",
            self._count,
        )
        vectors = self.reconstruct_all()
        hnsw = self._new_hnsw()
        hnsw.add(vectors)
        self._index = hnsw
        self._is_hnsw = True

    # ─── Properties ───

    @property
    def count(self) -> int:
        """Return number of vectors in the index.

        Performance:
            Time: O(1)
        """
        return self._count

    @property
    def dimension(self) -> int:
        """Return vector dimension.

        Performance:
            Time: O(1)
        """
        return self._dimension

    @property
    def index_type(self) -> str:
        """Return the concrete index type in use (``flat`` or ``hnsw``).

        Performance:
            Time: O(1)
        """
        return "hnsw" if self._is_hnsw else "flat"

    @property
    def requested_type(self) -> str:
        """Return the index type requested at construction.

        Performance:
            Time: O(1)
        """
        return self._requested_type

    # ─── Mutation ───

    def add(self, vectors: NDArray[np.float32]) -> int:
        """Add vectors to the index.

        Args:
            vectors: Array of shape (n, dimension) with float32 vectors.

        Returns:
            Position of the first vector added (positions are sequential).

        Performance:
            Time: O(n * d) for flat; O(n * log N * d) for HNSW

        Example:
            >>> indexer = FaissIndexer(384)
            >>> indexer.add(np.random.randn(50, 384).astype(np.float32))
            0
            >>> indexer.count
            50
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        # security: Validate dimensions match
        if vectors.shape[1] != self._dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self._dimension}, "
                f"got {vectors.shape[1]}.\n"
                f"VecForge by Suneel Bose K · ArcGX TechLabs"
            )

        # perf: Ensure contiguous float32 for FAISS
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        start = self._count
        self._index.add(vectors)
        self._count += vectors.shape[0]
        self._maybe_upgrade()
        return start

    def reset(self) -> None:
        """Reset the index, removing all vectors.

        Performance:
            Time: O(1)
        """
        if self._requested_type == "hnsw":
            self._index = self._new_hnsw()
            self._is_hnsw = True
        else:
            self._index = faiss.IndexFlatIP(self._dimension)
            self._is_hnsw = False
        self._count = 0

    # ─── Search ───

    @staticmethod
    def _build_selector(
        allowed_ids: NDArray[np.int64] | None,
        excluded_ids: NDArray[np.int64] | None,
    ) -> faiss.IDSelector | None:
        selectors: list[faiss.IDSelector] = []
        if allowed_ids is not None:
            arr = np.ascontiguousarray(np.asarray(allowed_ids, dtype=np.int64))
            selectors.append(faiss.IDSelectorBatch(arr))
        if excluded_ids is not None and len(excluded_ids) > 0:
            arr = np.ascontiguousarray(np.asarray(excluded_ids, dtype=np.int64))
            selectors.append(faiss.IDSelectorNot(faiss.IDSelectorBatch(arr)))
        if not selectors:
            return None
        if len(selectors) == 1:
            return selectors[0]
        return faiss.IDSelectorAnd(selectors[0], selectors[1])

    def _search_params(
        self, sel: faiss.IDSelector | None, ef: int | None = None
    ) -> faiss.SearchParameters | None:
        if sel is None and not self._is_hnsw:
            return None
        if self._is_hnsw:
            params = faiss.SearchParametersHNSW()  # type: ignore[attr-defined]
            params.efSearch = int(ef or self._ef_search)
            if sel is not None:
                params.sel = sel
            hnsw_params: faiss.SearchParameters = params
            return hnsw_params
        return faiss.SearchParameters(sel=sel)  # type: ignore[call-arg]

    def search_batch(
        self,
        queries: NDArray[np.float32],
        top_k: int = 10,
        allowed_ids: NDArray[np.int64] | None = None,
        excluded_ids: NDArray[np.int64] | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Search several query vectors at once.

        Args:
            queries: Array of shape (q, dimension).
            top_k: Neighbours per query.
            allowed_ids: Optional positions to restrict the search to.
            excluded_ids: Optional positions to exclude (tombstones).

        Returns:
            ``(distances, indices)`` arrays of shape (q, k). Slots that
            could not be filled hold ``-1`` in ``indices``.

        Performance:
            Time: O(q * N * d) flat, O(q * log N * d) HNSW
        """
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if queries.shape[1] != self._dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self._dimension}, "
                f"got {queries.shape[1]}.\n"
                f"VecForge by Suneel Bose K · ArcGX TechLabs"
            )

        pool = self._count
        if allowed_ids is not None:
            pool = min(pool, len(allowed_ids))
        effective_k = min(top_k, pool)
        if effective_k <= 0:
            return (
                np.zeros((queries.shape[0], 0), dtype=np.float32),
                np.zeros((queries.shape[0], 0), dtype=np.int64),
            )

        if (
            self._is_hnsw
            and allowed_ids is not None
            and len(allowed_ids) <= _EXACT_SUBSET_MAX
        ):
            return self._exact_subset_search(
                queries, effective_k, allowed_ids, excluded_ids
            )

        # why: selectors need extra search breadth to fill k slots
        ef = self._ef_search
        if self._is_hnsw and (allowed_ids is not None or excluded_ids is not None):
            ef = max(self._ef_search, 4 * effective_k)

        sel = self._build_selector(allowed_ids, excluded_ids)
        params = self._search_params(sel, ef)
        if params is None:
            distances, indices = self._index.search(queries, effective_k)
        else:
            distances, indices = self._index.search(queries, effective_k, params=params)
        return distances.astype(np.float32), indices.astype(np.int64)

    def _exact_subset_search(
        self,
        queries: NDArray[np.float32],
        k: int,
        allowed_ids: NDArray[np.int64],
        excluded_ids: NDArray[np.int64] | None,
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Exact inner-product search over a subset of positions."""
        ids = np.asarray(allowed_ids, dtype=np.int64)
        if excluded_ids is not None and len(excluded_ids) > 0:
            ids = ids[~np.isin(ids, np.asarray(excluded_ids, dtype=np.int64))]
        q = queries.shape[0]
        out_d = np.full((q, k), -np.inf, dtype=np.float32)
        out_i = np.full((q, k), -1, dtype=np.int64)
        if ids.size == 0:
            return out_d, out_i
        vectors = np.asarray(self._index.reconstruct_batch(ids), dtype=np.float32)
        scores = queries @ vectors.T  # (q, |ids|)
        kk = min(k, ids.size)
        for row in range(q):
            sc = scores[row]
            top = np.argpartition(sc, -kk)[-kk:] if kk < sc.size else np.arange(sc.size)
            top = top[np.argsort(sc[top])[::-1]]
            out_d[row, :kk] = sc[top]
            out_i[row, :kk] = ids[top]
        return out_d, out_i

    def search(
        self,
        query: NDArray[np.float32],
        top_k: int = 10,
        allowed_ids: NDArray[np.int64] | None = None,
        excluded_ids: NDArray[np.int64] | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Search for nearest neighbours to a query vector.

        Args:
            query: Query vector of shape (dimension,) or (1, dimension).
            top_k: Number of nearest neighbours to return.
            allowed_ids: Optional positions to restrict the search to
                (e.g. one namespace).
            excluded_ids: Optional positions to exclude (tombstones).

        Returns:
            Tuple of (distances, indices) 1-D arrays. Distances are inner
            product scores (higher = more similar). Indices are 0-based
            positions in the order vectors were added; ``-1`` marks empty
            slots and is filtered out.

        Performance:
            Time: O(N * d) for flat, O(log N * d) for HNSW
            Typical: <5ms at 100k docs

        Example:
            >>> distances, indices = indexer.search(query_vec, top_k=5)
            >>> print(f"Best match: index={indices[0]}, score={distances[0]:.4f}")
        """
        distances, indices = self.search_batch(
            query, top_k=top_k, allowed_ids=allowed_ids, excluded_ids=excluded_ids
        )
        d, i = distances[0], indices[0]
        keep = i >= 0
        return d[keep], i[keep]

    # ─── Serialisation ───

    def reconstruct_all(self) -> NDArray[np.float32]:
        """Return every stored vector as an ``(N, dimension)`` array.

        Performance:
            Time: O(N * d)
        """
        if self._count == 0:
            return np.zeros((0, self._dimension), dtype=np.float32)
        out: NDArray[np.float32] = self._index.reconstruct_n(0, self._count)
        return np.asarray(out, dtype=np.float32)

    def to_bytes(self) -> bytes:
        """Serialize the FAISS index to bytes for storage.

        Uses ``faiss.write_index`` through a temporary file, which is
        several times faster than the SWIG in-memory writer for large
        indexes.

        Returns:
            Raw bytes of the serialized FAISS index.

        Performance:
            Time: O(N * d) — proportional to index size
        """
        fd, path = tempfile.mkstemp(suffix=".faiss", prefix="vecforge-")
        os.close(fd)
        try:
            faiss.write_index(self._index, path)
            with open(path, "rb") as f:
                return f.read()
        finally:
            with contextlib.suppress(OSError):
                os.unlink(path)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        dimension: int,
        index_type: IndexType = "auto",
    ) -> FaissIndexer:
        """Deserialize a FAISS index from bytes.

        Args:
            data: Serialized FAISS index bytes.
            dimension: Expected embedding dimension.
            index_type: Requested type (``auto`` keeps upgrading later).

        Returns:
            Reconstructed FaissIndexer instance.

        Raises:
            ValueError: If the stored dimension does not match.

        Performance:
            Time: O(N * d) — proportional to index size
        """
        fd, path = tempfile.mkstemp(suffix=".faiss", prefix="vecforge-")
        os.close(fd)
        try:
            with open(path, "wb") as f:
                f.write(data)
            index = faiss.read_index(path)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(path)
        if int(index.d) != int(dimension):
            raise ValueError(
                f"Stored index dimension {index.d} != expected {dimension}.\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            )

        instance = cls(dimension, index_type=index_type)
        instance._index = index
        instance._count = int(index.ntotal)
        instance._is_hnsw = (
            isinstance(index, faiss.IndexHNSW) or "HNSW" in type(index).__name__
        )
        if instance._is_hnsw:
            index.hnsw.efSearch = instance._ef_search  # type: ignore[attr-defined]
        return instance
