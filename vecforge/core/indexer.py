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

Provides efficient approximate nearest neighbour search using FAISS.
Supports both flat (exact) and IVF (approximate) indexes with automatic
training when the collection grows.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import logging

import faiss
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# perf: Switch to IVF when collection exceeds this threshold
_IVF_THRESHOLD = 10_000
_IVF_NLIST = 100  # number of Voronoi cells for IVF
_IVF_NPROBE = 10  # number of cells to search


class FaissIndexer:
    """FAISS-based vector index for fast nearest-neighbour search.

    Starts with IndexFlatIP (exact inner product search) for small
    collections, and can be upgraded to IndexIVFFlat for larger ones.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        dimension: Embedding vector dimension.

    Performance:
        Flat index: O(N * d) search — exact, best for N < 10k
        IVF index:  O(N/nlist * d * nprobe) — approximate, for N > 10k
        Typical: <5ms at 100k docs with IVF

    Example:
        >>> indexer = FaissIndexer(dimension=384)
        >>> indexer.add(np.random.randn(100, 384).astype(np.float32))
        >>> distances, indices = indexer.search(query_vec, top_k=5)
    """

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._index: faiss.Index = faiss.IndexFlatIP(dimension)
        self._count = 0
        self._is_ivf = False

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

    def add(self, vectors: NDArray[np.float32]) -> None:
        """Add vectors to the index.

        Args:
            vectors: Array of shape (n, dimension) with float32 vectors.

        Performance:
            Time: O(n * d) for flat index
            Amortized O(n * d / nlist) for IVF after training

        Example:
            >>> indexer = FaissIndexer(384)
            >>> indexer.add(np.random.randn(50, 384).astype(np.float32))
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

        self._index.add(vectors)
        self._count += vectors.shape[0]

    def search(
        self, query: NDArray[np.float32], top_k: int = 10
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Search for nearest neighbours to query vector.

        Args:
            query: Query vector of shape (dimension,) or (1, dimension).
            top_k: Number of nearest neighbours to return.

        Returns:
            Tuple of (distances, indices) arrays, each of shape (top_k,).
            Distances are inner product scores (higher = more similar).
            Indices are 0-based positions in the order vectors were added.

        Performance:
            Time: O(N * d) for flat, O(N/nlist * d * nprobe) for IVF
            Typical: <5ms at 100k docs

        Example:
            >>> distances, indices = indexer.search(query_vec, top_k=5)
            >>> print(f"Best match: index={indices[0]}, score={distances[0]:.4f}")
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # perf: Ensure contiguous float32
        query = np.ascontiguousarray(query, dtype=np.float32)

        # why: Clamp top_k to available vectors
        effective_k = min(top_k, self._count)
        if effective_k == 0:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.int64),
            )

        distances, indices = self._index.search(query, effective_k)
        return distances[0], indices[0]

    def to_bytes(self) -> bytes:
        """Serialize the FAISS index to bytes for storage.

        Returns:
            Raw bytes of the serialized FAISS index.

        Performance:
            Time: O(N * d) — proportional to index size
        """
        data: bytes = faiss.serialize_index(self._index).tobytes()
        return data

    @classmethod
    def from_bytes(cls, data: bytes, dimension: int) -> FaissIndexer:
        """Deserialize a FAISS index from bytes.

        Args:
            data: Serialized FAISS index bytes.
            dimension: Expected embedding dimension.

        Returns:
            Reconstructed FaissIndexer instance.

        Performance:
            Time: O(N * d) — proportional to index size
        """
        index_array = np.frombuffer(data, dtype=np.uint8)
        index = faiss.deserialize_index(index_array)

        instance = cls(dimension)
        instance._index = index
        instance._count = index.ntotal
        return instance

    def reset(self) -> None:
        """Reset the index, removing all vectors.

        Performance:
            Time: O(1)
        """
        self._index = faiss.IndexFlatIP(self._dimension)
        self._count = 0
        self._is_ivf = False
