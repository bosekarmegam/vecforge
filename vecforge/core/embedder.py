# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
Embedding engine for VecForge.

Wraps sentence-transformers for local text embedding. No internet
required — models are downloaded once and cached locally.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# perf: Default model balances quality and speed for most use cases
_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """Local text embedding engine using sentence-transformers.

    Lazily loads the model on first use to keep VecForge init fast.
    All processing runs locally — zero cloud dependency.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        model_name: Name of the sentence-transformers model.
            Defaults to 'all-MiniLM-L6-v2' (384-dim, fast, good quality).
        device: Device to run on ('cpu', 'cuda'). Auto-detected if None.

    Performance:
        Time: O(n * d) where n = number of texts, d = model dimension
        Typical: ~5ms per text on CPU, ~0.5ms on GPU

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
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None  # Lazy-loaded SentenceTransformer
        self._dimension: int | None = None

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

    def _load_model(self) -> None:
        """Lazily load the sentence-transformer model.

        Performance:
            Time: O(1) — one-time cost of ~1-3 seconds for model loading
        """
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for VecForge embeddings.\n"
                "Install with: pip install sentence-transformers\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            ) from e

        logger.info("Loading embedding model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name, device=self._device)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Embedding model loaded: %s (dim=%d)",
            self._model_name,
            self._dimension,
        )

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
            batch_size: Batch size for encoding. Defaults to 64.
            normalize: If True, L2-normalize vectors for cosine similarity.
                Defaults to True.
            show_progress: Show progress bar for large batches.

        Returns:
            NumPy array of shape (n_texts, dimension) with float32 vectors.

        Performance:
            Time: O(n * d) where n = len(texts), d = model dimension
            Typical: ~5ms per text on CPU with default model

        Example:
            >>> embedder = Embedder()
            >>> vec = embedder.encode("patient with diabetes")
            >>> vec.shape
            (1, 384)
        """
        self._load_model()

        if isinstance(texts, str):
            texts = [texts]

        # perf: sentence-transformers handles batching internally
        vectors: NDArray[np.float32] = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        return vectors.astype(np.float32)
