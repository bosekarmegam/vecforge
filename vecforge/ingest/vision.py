# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
Image embedding via OpenCLIP (local CLIP model).

Provides ImageEmbedder for generating 512-dim embeddings from images,
enabling cross-modal search between text and images without any cloud API.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Lazy imports — only required when ImageEmbedder is used
_open_clip: object | None = None
_torch: object | None = None
_PIL_Image: object | None = None


def _ensure_deps() -> None:
    """Lazy-import open_clip and torch — raise clear message if missing."""
    global _open_clip, _torch, _PIL_Image
    if _open_clip is None:
        try:
            import open_clip

            _open_clip = open_clip
        except ImportError as exc:
            raise ImportError(
                "ImageEmbedder requires the [multimodal] extra.\n"
                "Install with: pip install vecforge[multimodal]\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            ) from exc
    if _torch is None:
        import torch

        _torch = torch
    if _PIL_Image is None:
        from PIL import Image

        _PIL_Image = Image


class ImageEmbedder:
    """Embed images into the shared VecForge vector space using CLIP.

    Uses OpenCLIP (local, no internet) to produce 512-dim float32 embeddings
    compatible with the existing text embedding space for cross-modal search.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        model_name: OpenCLIP model name. Default ``ViT-B-32``.
        pretrained: Pretrained weights name. Default ``openai``.

    Performance:
        Embedding: ~20ms per image on CPU, ~5ms on GPU
        Memory:    ~150MB model load (cached after first call)

    Example::

        >>> embedder = ImageEmbedder()
        >>> vec = embedder.embed("photo.jpg")
        >>> vec.shape
        (512,)
        >>> vec.dtype
        dtype('float32')
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
    ) -> None:
        self._model_name = model_name
        self._pretrained = pretrained
        self._model: object | None = None
        self._preprocess: object | None = None
        self._device: str | None = None

    def _load(self) -> None:
        """Lazy-load model on first use to avoid startup cost."""
        if self._model is not None:
            return
        _ensure_deps()
        import open_clip
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        model, _, preprocess = open_clip.create_model_and_transforms(
            self._model_name, pretrained=self._pretrained
        )
        model.eval()
        if hasattr(model, "to"):
            model = model.to(device)
        self._model = model
        self._preprocess = preprocess
        logger.info(
            "ImageEmbedder: loaded %s/%s on %s",
            self._model_name,
            self._pretrained,
            device,
        )

    def embed(
        self,
        image: str | Path | PIL.Image.Image,  # noqa: F821
    ) -> NDArray[np.float32]:
        """Generate a 512-dim CLIP embedding for an image.

        Args:
            image: File path (str or Path) or a PIL.Image.Image object.

        Returns:
            Unit-norm float32 array of shape (512,).

        Raises:
            ImportError: If ``vecforge[multimodal]`` is not installed.
            FileNotFoundError: If the image path does not exist.

        Performance:
            ~20ms CPU, ~5ms GPU. Model is cached after first load.

        Example::

            >>> vec = ImageEmbedder().embed("sunset.jpg")
            >>> vec.shape
            (512,)
        """
        self._load()
        import torch
        from PIL import Image as PILImage

        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(
                    f"Image not found: {path}\n"
                    "VecForge by Suneel Bose K · ArcGX TechLabs"
                )
            img = PILImage.open(path).convert("RGB")
        else:
            img = image.convert("RGB")

        # Preprocess: resize, normalize, batch-dim
        preprocessed = self._preprocess(img).unsqueeze(0)  # type: ignore[operator]
        if self._device:
            preprocessed = preprocessed.to(self._device)

        with torch.no_grad():
            features = self._model.encode_image(preprocessed)  # type: ignore[union-attr]
            # L2 normalise to unit sphere
            features = features / features.norm(dim=-1, keepdim=True)

        vec: NDArray[np.float32] = features.squeeze(0).cpu().numpy().astype(np.float32)
        logger.debug("ImageEmbedder: embedded image → shape %s", vec.shape)
        return vec

    def embed_text(self, text: str) -> NDArray[np.float32]:
        """Generate a CLIP text embedding for cross-modal search.

        Args:
            text: Query string to embed in CLIP's shared text/image space.

        Returns:
            Unit-norm float32 array of shape (512,).

        Example::

            >>> vec = ImageEmbedder().embed_text("a photo of a cat")
            >>> vec.shape
            (512,)
        """
        self._load()
        import open_clip
        import torch

        tokens = open_clip.tokenize([text])
        if self._device:
            tokens = tokens.to(self._device)

        with torch.no_grad():
            features = self._model.encode_text(tokens)  # type: ignore[union-attr]
            features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze(0).cpu().numpy().astype(np.float32)
