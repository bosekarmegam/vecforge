# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under the Business Source License 1.1 (BSL 1.1)

"""all-MiniLM-L6-v2 through ONNX Runtime, for the retrieval-quality experiments.

Mirrors sentence-transformers: WordPiece tokenisation (max 256 tokens), mean
pooling over the attention mask, then L2 normalisation. Point MINILM_DIR at a
folder holding ``onnx/model.onnx`` and ``tokenizer.json`` for the model.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class MiniLM:
    dimension = 384

    def __init__(self, model_dir: str | None = None, max_length: int = 256, batch_size: int = 64) -> None:
        d = Path(model_dir or os.environ["MINILM_DIR"])
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 1
        self._session = ort.InferenceSession(str(d / "onnx" / "model.onnx"), opts, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._session.get_inputs()}
        self._tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self._tok.enable_truncation(max_length=max_length)
        self._tok.enable_padding(pad_id=0, pad_token="[PAD]")
        self._batch = batch_size

    def encode(self, texts: list[str] | str) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for i in range(0, len(texts), self._batch):
            enc = self._tok.encode_batch(texts[i : i + self._batch])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self._session.run(None, feeds)[0]
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            pooled /= np.linalg.norm(pooled, axis=1, keepdims=True)
            out.append(pooled.astype(np.float32))
        return np.concatenate(out) if out else np.zeros((0, self.dimension), np.float32)

    __call__ = encode
