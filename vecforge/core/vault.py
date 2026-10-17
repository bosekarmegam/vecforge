# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
Main VecForge class — the 5-line API surface.

This is the primary entry point for all VecForge operations. Designed
to make any core feature usable in 5 lines of Python or fewer.

VecForge 2.0 highlights:

* namespace scoping and deletes are applied **inside** the FAISS/BM25
  retrievers (no more post-filter recall loss);
* metadata filters adaptively widen the candidate window until ``top_k``
  matches are found;
* deletes are O(1) tombstones with automatic compaction;
* reopening a vault restores the FAISS and BM25 indexes from a snapshot
  instead of rebuilding them;
* pluggable embedders (sentence-transformers, dependency-free hashing,
  or any callable) and a query-embedding cache;
* all public methods are thread-safe.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

Example::

    from vecforge import VecForge

    db = VecForge("my_vault")
    db.add("Patient admitted with type 2 diabetes")
    results = db.search("diabetic patient")
    print(results[0].text)
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vecforge.core.bm25 import BM25Engine
from vecforge.core.embedder import Embedder, EmbedderLike, get_embedder
from vecforge.core.indexer import FaissIndexer, IndexType
from vecforge.core.reranker import Reranker
from vecforge.core.storage import IndexState, StorageBackend, StoredDocument
from vecforge.exceptions import (
    DeletionProtectedError,
    DocumentNotFoundError,
    DuplicateDocumentError,
    InvalidAlphaError,
    VaultClosedError,
    VaultEmptyError,
)
from vecforge.quantum.reranker import QuantumReranker as _QuantumReranker
from vecforge.search.cascade import CascadeCandidate, CascadeSearcher
from vecforge.search.filters import MetadataFilter
from vecforge.security.audit import AuditLogger
from vecforge.security.encryption import validate_encryption_key
from vecforge.security.namespaces import NamespaceManager
from vecforge.security.rbac import RBACManager

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_ENV_MODEL = "VECFORGE_MODEL"
_MIN_FETCH = 20
_COMPACT_MIN_TOMBSTONES = 1_000
_COMPACT_RATIO = 0.2
_RESTORE_BATCH = 2_000
_RBAC_KV_KEY = "__rbac_keys__"


@dataclass
class SearchResult:
    """A single search result from VecForge.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Attributes:
        text: Document text content.
        score: Relevance score (higher = more relevant).
        metadata: User-provided metadata dictionary.
        namespace: Namespace this document belongs to.
        doc_id: Unique document identifier.
        modality: Content modality (text, image, audio, etc.).
        timestamp: Document creation timestamp.
        dense_score: Normalised dense (semantic) score, or None if the
            dense retriever did not return this document.
        sparse_score: Normalised BM25 (keyword) score, or None if the
            sparse retriever did not return this document.
        signals: Retrievers that matched: ``["dense"]``, ``["sparse"]``
            or both.
        updated_at: Last update timestamp, if the document was edited.
    """

    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    namespace: str = "default"
    doc_id: str = ""
    modality: str = "text"
    timestamp: float = 0.0
    dense_score: float | None = None
    sparse_score: float | None = None
    signals: list[str] = field(default_factory=list)
    updated_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary of the result.

        Performance:
            Time: O(1)
        """
        return asdict(self)

    def __repr__(self) -> str:
        preview = self.text[:80] + "..." if len(self.text) > 80 else self.text
        return (
            f"SearchResult(score={self.score:.4f}, "
            f"doc_id='{self.doc_id[:8]}...', "
            f"text='{preview}')"
        )


class VecForge:
    """Universal local-first vector database.

    The 5-line API: create, ingest, search, done.
    Every feature is usable in 5 lines of Python or fewer.

    Built by Suneel Bose K, Founder & CEO, ArcGX TechLabs Private Limited.

    Args:
        path: Vault path. Use ':memory:' for in-memory storage.
        encryption_key: SQLCipher AES-256 encryption key. Use
            ``os.environ['VECFORGE_KEY']`` — never hardcode.
        audit_log: Path to JSONL audit log file. If None, auditing off.
        quantum: Default for ``search(quantum_rerank=...)``. Defaults to
            False.
        deletion_protection: Prevent accidental deletions. Defaults to False.
        api_key: API key for RBAC. None = local admin.
        model_name: Embedding model: a sentence-transformers model name
            (default ``all-MiniLM-L6-v2`` or ``$VECFORGE_MODEL``), or
            ``"hash"`` / ``"hash:<dim>"`` for the dependency-free engine.
        embedder: Pre-built embedder object (anything implementing
            :class:`~vecforge.core.embedder.EmbedderLike`). Overrides
            ``model_name``.
        index_type: FAISS index: ``"flat"`` (exact), ``"hnsw"``
            (approximate) or ``"auto"`` (flat, upgrading past 100k docs).
        auto_save_every: Persist the index snapshot automatically after
            this many mutations (0 = only on ``save()``/``close()``).
        rerank_model: Cross-encoder model name for ``rerank=True``.

    Performance:
        Init: O(1) — lazy model loading, O(snapshot) index restore
        Search: O(log N) + O(k) rerank
        Add: O(d) embedding + O(1) storage

    Example:
        >>> from vecforge import VecForge
        >>> db = VecForge("my_vault")
        >>> db.add("Patient admitted with type 2 diabetes", metadata={"ward": "7"})
        >>> results = db.search("diabetic patient")
        >>> print(results[0].text)
        'Patient admitted with type 2 diabetes'
    """

    def __init__(
        self,
        path: str = ":memory:",
        encryption_key: str | None = None,
        audit_log: str | None = None,
        quantum: bool = False,
        deletion_protection: bool = False,
        api_key: str | None = None,
        model_name: str | None = None,
        embedder: EmbedderLike | None = None,
        index_type: IndexType = "auto",
        auto_save_every: int = 0,
        rerank_model: str | None = None,
    ) -> None:
        self._path = path
        self._quantum = quantum
        self._deletion_protection = deletion_protection
        self._index_type: IndexType = index_type
        self._auto_save_every = max(0, int(auto_save_every))
        self._lock = threading.RLock()
        self._closed = False

        # security: Validate encryption key
        validated_key = validate_encryption_key(encryption_key)

        # ─── Initialize subsystems ───
        self._storage = StorageBackend(path=path, encryption_key=validated_key)
        self._embedder: EmbedderLike = self._resolve_embedder(model_name, embedder)
        self._indexer: FaissIndexer | None = None  # why: Lazy — needs dimension
        self._bm25 = BM25Engine()
        self._reranker = Reranker(rerank_model) if rerank_model else Reranker()
        self._rbac = RBACManager(
            api_key=api_key, key_roles=self._storage.kv_get(_RBAC_KV_KEY, {}) or {}
        )
        self._audit = AuditLogger(log_path=audit_log)
        self._namespace_mgr = NamespaceManager(self._storage)

        # why: position bookkeeping — FAISS/BM25 position → doc
        self._index_to_doc_id: list[str] = []
        self._doc_id_to_index: dict[str, int] = {}
        self._index_namespace: list[str] = []
        self._tombstones: set[int] = set()
        self._tombstone_arr: NDArray[np.int64] | None = None
        self._ns_positions: dict[str, set[int]] = {}
        self._ns_arr_cache: dict[str, NDArray[np.int64]] = {}
        self._mutations_since_save = 0

        # why: Try to restore from persisted state
        self._restore_state()

        logger.info(
            "VecForge vault opened: %s (encrypted=%s, docs=%d, model=%s)",
            path,
            self._storage.is_encrypted,
            self.count(),
            self._embedder.name,
        )

    # ─── Setup helpers ───

    @staticmethod
    def _resolve_embedder(
        model_name: str | None, embedder: EmbedderLike | None
    ) -> EmbedderLike:
        if embedder is not None:
            return get_embedder(embedder)
        name = model_name or os.environ.get(_ENV_MODEL) or _DEFAULT_MODEL
        lowered = name.strip().lower()
        if lowered == "hash" or lowered.startswith("hash:"):
            return get_embedder(name)
        # why: use the module-level Embedder symbol so it stays patchable
        return Embedder(model_name=name)

    def _check_open(self) -> None:
        if self._closed:
            raise VaultClosedError(self._path)

    def _restore_state(self) -> None:
        """Restore FAISS index and BM25 from persisted storage.

        Tries the index snapshot first (O(blob size)); falls back to a
        full rebuild from the stored embeddings (O(N·d), no re-embedding).

        Performance:
            Time: O(snapshot) fast path, O(N * d) rebuild
        """
        total = self._storage.count_docs()
        if total == 0:
            return

        state = self._storage.load_index_state()
        if state is not None and self._try_fast_restore(state, total):
            return

        self._rebuild_from_storage()

    def _try_fast_restore(self, state: IndexState, total: int) -> bool:
        live = len(state.doc_ids) - len(state.tombstones)
        if (
            state.generation != self._storage.generation
            or state.doc_count != total
            or live != total
            or len(state.doc_namespaces) != len(state.doc_ids)
        ):
            logger.info("Index snapshot is stale — rebuilding indexes from storage")
            return False
        try:
            bm25 = BM25Engine.from_bytes(state.bm25_blob)
            if state.faiss_blob:
                indexer = FaissIndexer.from_bytes(
                    state.faiss_blob, state.dimension, index_type=self._index_type
                )
            else:
                indexer = self._flat_index_from_storage(state)
        except Exception as exc:  # pragma: no cover - corrupt snapshot
            logger.warning("Index snapshot unreadable (%s) — rebuilding", exc)
            return False
        if indexer.count != len(state.doc_ids) or bm25.size != len(state.doc_ids):
            logger.warning("Index snapshot inconsistent — rebuilding")
            return False

        if state.model_name and state.model_name != self._embedder.name:
            logger.warning(
                "Vault was indexed with embedder '%s' but opened with '%s'. "
                "Stored vectors are reused; queries may not match well. "
                "Call db.reembed() to re-index with the new model.",
                state.model_name,
                self._embedder.name,
            )

        self._indexer = indexer
        self._bm25 = bm25
        self._index_to_doc_id = list(state.doc_ids)
        self._index_namespace = list(state.doc_namespaces)
        self._tombstones = set(state.tombstones)
        self._rebuild_position_maps()
        logger.info("Restored %d documents from index snapshot (fast path)", total)
        return True

    def _flat_index_from_storage(self, state: IndexState) -> FaissIndexer:
        """Rebuild a flat index from stored vectors in snapshot position order.

        Positions are insertion order (SQLite rowid order), so the vectors
        are streamed straight into a fresh ``IndexFlatIP`` — a memcpy, no
        embedding and no tokenising. Tombstoned positions get a zero vector
        placeholder (they are excluded from every search by the selector).
        """
        requested: IndexType = (
            "flat" if self._index_type == "auto" else self._index_type
        )
        indexer = FaissIndexer(dimension=state.dimension, index_type=requested)
        tombstones = set(state.tombstones)
        zero = np.zeros(state.dimension, dtype=np.float32)
        rows = self._storage.iter_embeddings(batch_size=_RESTORE_BATCH)
        batch: list[NDArray[np.float32]] = []
        for pos, did in enumerate(state.doc_ids):
            if pos in tombstones:
                batch.append(zero)
            else:
                row = next(rows, None)
                if row is None or row[0] != did or len(row[1]) != state.dimension:
                    raise ValueError("snapshot position order does not match storage")
                batch.append(row[1])
            if len(batch) >= _RESTORE_BATCH:
                indexer.add(np.stack(batch))
                batch = []
        if batch:
            indexer.add(np.stack(batch))
        if next(rows, None) is not None:
            raise ValueError("storage has documents missing from the snapshot")
        # why: restore the 'auto' upgrade behaviour after a flat restore
        indexer._requested_type = self._index_type
        return indexer

    def _rebuild_from_storage(self) -> None:
        """Rebuild all in-memory indexes from the stored embeddings."""
        self._indexer = None
        self._bm25 = BM25Engine()
        self._index_to_doc_id = []
        self._index_namespace = []
        self._tombstones = set()

        batch_ids: list[str] = []
        batch_ns: list[str] = []
        batch_texts: list[str] = []
        batch_vecs: list[NDArray[np.float32]] = []

        skipped = 0

        def flush() -> None:
            nonlocal skipped
            if not batch_ids:
                return
            dim = (
                len(batch_vecs[0]) if self._indexer is None else self._indexer.dimension
            )
            keep = [i for i, v in enumerate(batch_vecs) if len(v) == dim]
            if len(keep) != len(batch_ids):
                # why: a model switch left rows with another dimension — index
                # what fits and tell the user how to repair the rest
                skipped += len(batch_ids) - len(keep)
                for seq in (batch_ids, batch_ns, batch_texts, batch_vecs):
                    seq[:] = [seq[i] for i in keep]
                if not batch_ids:
                    return
            vectors = np.stack(batch_vecs).astype(np.float32)
            if self._indexer is None:
                self._indexer = FaissIndexer(
                    dimension=vectors.shape[1], index_type=self._index_type
                )
            self._indexer.add(vectors)
            self._bm25.add_documents(batch_texts)
            self._index_to_doc_id.extend(batch_ids)
            self._index_namespace.extend(batch_ns)
            batch_ids.clear()
            batch_ns.clear()
            batch_texts.clear()
            batch_vecs.clear()

        for doc in self._storage.iter_docs(batch_size=_RESTORE_BATCH):
            batch_ids.append(doc.doc_id)
            batch_ns.append(doc.namespace)
            batch_texts.append(doc.text)
            batch_vecs.append(doc.embedding)
            if len(batch_ids) >= _RESTORE_BATCH:
                flush()
        flush()

        self._rebuild_position_maps()
        self._mutations_since_save = 1  # why: snapshot is stale → save on close
        if skipped:
            logger.warning(
                "%d documents have a different embedding dimension and were left "
                "out of the index. Run db.reembed() to re-index them with the "
                "current model.",
                skipped,
            )
        logger.info("Rebuilt indexes for %d documents", len(self._index_to_doc_id))

    def _rebuild_position_maps(self) -> None:
        self._doc_id_to_index = {
            did: pos
            for pos, did in enumerate(self._index_to_doc_id)
            if pos not in self._tombstones
        }
        self._ns_positions = {}
        for pos, ns in enumerate(self._index_namespace):
            if pos in self._tombstones:
                continue
            self._ns_positions.setdefault(ns, set()).add(pos)
        self._ns_arr_cache = {}
        self._tombstone_arr = None

    def _ensure_indexer(self) -> FaissIndexer:
        """Ensure FAISS indexer is initialized.

        Returns:
            FaissIndexer instance.

        Performance:
            Time: O(1) if already initialized, O(model_load) on first call
        """
        if self._indexer is None:
            dim = self._embedder.dimension
            self._indexer = FaissIndexer(dimension=dim, index_type=self._index_type)
        return self._indexer

    # ─── Position bookkeeping ───

    def _index_documents(
        self,
        doc_ids: Sequence[str],
        namespace: str,
        embeddings: NDArray[np.float32],
        texts: Sequence[str],
    ) -> None:
        indexer = self._ensure_indexer()
        start = indexer.add(embeddings)
        self._bm25.add_documents(list(texts))
        ns_set = self._ns_positions.setdefault(namespace, set())
        for offset, did in enumerate(doc_ids):
            pos = start + offset
            self._index_to_doc_id.append(did)
            self._index_namespace.append(namespace)
            self._doc_id_to_index[did] = pos
            ns_set.add(pos)
        self._ns_arr_cache.pop(namespace, None)
        self._note_mutation(len(doc_ids))

    def _tombstone(self, doc_id: str) -> None:
        pos = self._doc_id_to_index.pop(doc_id, None)
        if pos is None:
            return
        self._tombstones.add(pos)
        self._tombstone_arr = None
        self._bm25.remove(pos)
        ns = self._index_namespace[pos]
        positions = self._ns_positions.get(ns)
        if positions is not None:
            positions.discard(pos)
            self._ns_arr_cache.pop(ns, None)

    def _tombstone_array(self) -> NDArray[np.int64] | None:
        if not self._tombstones:
            return None
        if self._tombstone_arr is None:
            self._tombstone_arr = np.fromiter(
                sorted(self._tombstones), dtype=np.int64, count=len(self._tombstones)
            )
        return self._tombstone_arr

    def _namespace_array(self, namespace: str) -> NDArray[np.int64]:
        cached = self._ns_arr_cache.get(namespace)
        if cached is None:
            positions = self._ns_positions.get(namespace, set())
            cached = np.fromiter(
                sorted(positions), dtype=np.int64, count=len(positions)
            )
            self._ns_arr_cache[namespace] = cached
        return cached

    def _note_mutation(self, n: int = 1) -> None:
        self._mutations_since_save += n
        if (
            self._auto_save_every
            and self._mutations_since_save >= self._auto_save_every
        ):
            self.save()

    def _maybe_compact(self) -> None:
        size = len(self._index_to_doc_id)
        if len(self._tombstones) >= _COMPACT_MIN_TOMBSTONES and len(
            self._tombstones
        ) >= _COMPACT_RATIO * max(size, 1):
            self.compact()

    def _embed_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        vectors = self._embedder.encode(list(texts))
        return np.asarray(vectors, dtype=np.float32)

    def _validate_embedding(self, embedding: Any) -> NDArray[np.float32]:
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if self._indexer is not None and vec.shape[0] != self._indexer.dimension:
            raise ValueError(
                f"Embedding dimension {vec.shape[0]} does not match vault "
                f"dimension {self._indexer.dimension}.\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            )
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec = vec / norm
        return vec.astype(np.float32)

    # ─── Write API ───

    def add(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        namespace: str = "default",
        doc_id: str | None = None,
        modality: str = "text",
        embedding: NDArray[np.float32] | Sequence[float] | None = None,
    ) -> str:
        """Add a text document to the vault.

        Args:
            text: Document text content.
            metadata: Optional metadata dictionary.
            namespace: Target namespace. Defaults to 'default'.
            doc_id: Optional custom document ID.
            modality: Content modality label (``text``, ``image``, ...).
            embedding: Optional precomputed vector (e.g. a CLIP image
                embedding). When given, ``text`` is still indexed for BM25.

        Returns:
            Document ID (UUID string).

        Raises:
            VecForgePermissionError: If API key lacks write permission.
            DuplicateDocumentError: If ``doc_id`` already exists.

        Performance:
            Time: O(d) for embedding + O(1) for storage
            Typical: ~10ms per document

        Example:
            >>> doc_id = db.add(
            ...     "Patient P4821 — Type 2 diabetes",
            ...     metadata={"ward": "7", "year": 2024},
            ...     namespace="ward_7"
            ... )
        """
        with self._lock:
            self._check_open()
            # security: Check write permission
            self._rbac.require("write")

            if metadata is None:
                metadata = {}

            # why: Ensure namespace exists, create if needed
            if not self._namespace_mgr.exists(namespace):
                self._namespace_mgr.create(namespace)

            # ─── Embed the text ───
            if embedding is None:
                embedding_vec = self._embed_texts([text])[0]
            else:
                embedding_vec = self._validate_embedding(embedding)

            # ─── Store in SQLite ───
            try:
                stored = self._storage.insert_doc(
                    text=text,
                    embedding=embedding_vec,
                    metadata=metadata,
                    namespace=namespace,
                    modality=modality,
                    doc_id=doc_id,
                )
            except sqlite3.IntegrityError as exc:
                raise DuplicateDocumentError(str(doc_id)) from exc

            # ─── Update in-memory indexes ───
            self._index_documents(
                [stored.doc_id], namespace, embedding_vec.reshape(1, -1), [text]
            )

            # security: Emit audit event
            self._audit.log(
                actor=self._rbac.key_id,
                operation="add",
                doc_id=stored.doc_id,
                namespace=namespace,
                metadata={"chars": len(text), "modality": modality},
            )

            logger.debug("Added doc %s to namespace '%s'", stored.doc_id, namespace)
            return stored.doc_id

    def add_batch(
        self,
        texts: list[str],
        metadata_list: list[dict[str, Any]] | None = None,
        namespace: str = "default",
        doc_ids: Sequence[str | None] | None = None,
        embeddings: NDArray[np.float32] | None = None,
        modality: str = "text",
    ) -> list[str]:
        """Add multiple documents in one efficient batch operation.

        Embeds all texts in a single model call, inserts them in one
        SQLite transaction and adds them to FAISS/BM25 once, achieving
        10-50x the throughput of sequential ``add()``.

        Built by Suneel Bose K · ArcGX TechLabs Private Limited.

        Args:
            texts: List of document text strings.
            metadata_list: Optional list of metadata dicts (one per
                text). If None, all documents get empty metadata.
            namespace: Target namespace. Defaults to 'default'.
            doc_ids: Optional custom IDs (None entries auto-generate).
            embeddings: Optional precomputed vectors ``(len(texts), d)``.
            modality: Content modality label for the batch.

        Returns:
            List of document IDs.

        Raises:
            VecForgePermissionError: If API key lacks write access.
            ValueError: If metadata_list length != texts length.
            DuplicateDocumentError: If any custom ID already exists
                (nothing is inserted).

        Performance:
            Time: O(B * d) single model call + O(B) storage
            Typical: ~5x faster than sequential add() for B > 10

        Example:
            >>> ids = db.add_batch(
            ...     ["First doc", "Second doc", "Third doc"],
            ...     namespace="bulk",
            ... )
            >>> len(ids)
            3
        """
        with self._lock:
            self._check_open()
            # security: Check write permission
            self._rbac.require("write")

            if not texts:
                return []

            if metadata_list is None:
                metadata_list = [{} for _ in texts]

            if len(metadata_list) != len(texts):
                raise ValueError(
                    f"metadata_list length ({len(metadata_list)}) "
                    f"must match texts length ({len(texts)})"
                )
            if doc_ids is not None and len(doc_ids) != len(texts):
                raise ValueError(
                    f"doc_ids length ({len(doc_ids)}) must match texts ({len(texts)})"
                )

            # why: Ensure namespace exists
            if not self._namespace_mgr.exists(namespace):
                self._namespace_mgr.create(namespace)

            # perf: Batch embed all texts in one model call
            if embeddings is None:
                vectors = self._embed_texts(texts)
            else:
                vectors = np.stack(
                    [self._validate_embedding(e) for e in np.asarray(embeddings)]
                ).astype(np.float32)
                if vectors.shape[0] != len(texts):
                    raise ValueError(
                        f"embeddings rows ({vectors.shape[0]}) must match "
                        f"texts ({len(texts)})"
                    )

            # ─── Store in one transaction ───
            try:
                stored = self._storage.insert_docs(
                    texts=texts,
                    embeddings=vectors,
                    metadatas=metadata_list,
                    namespace=namespace,
                    modality=modality,
                    doc_ids=doc_ids,
                )
            except sqlite3.IntegrityError as exc:
                raise DuplicateDocumentError("<batch>") from exc

            ids = [s.doc_id for s in stored]
            self._index_documents(ids, namespace, vectors, texts)

            # security: Audit batch operation
            self._audit.log(
                actor=self._rbac.key_id,
                operation="add_batch",
                namespace=namespace,
                metadata={
                    "count": len(texts),
                    "chars": sum(len(t) for t in texts),
                },
            )

            logger.info("Batch added %d docs to '%s'", len(texts), namespace)
            return ids

    def upsert(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
        namespace: str = "default",
        modality: str = "text",
        embedding: NDArray[np.float32] | Sequence[float] | None = None,
    ) -> str:
        """Insert a document, or replace it if ``doc_id`` already exists.

        Args:
            text: Document text content.
            doc_id: Document identifier (required).
            metadata: Optional metadata dictionary.
            namespace: Target namespace.
            modality: Content modality label.
            embedding: Optional precomputed vector.

        Returns:
            The document ID.

        Raises:
            DeletionProtectedError: If replacing an existing document in a
                deletion-protected vault.

        Performance:
            Time: O(d) embedding + O(1) storage

        Example:
            >>> db.upsert("Updated note", doc_id="note-1")
            'note-1'
        """
        with self._lock:
            self._check_open()
            self._rbac.require("write")
            if self._storage.get_doc(doc_id) is not None:
                if self._deletion_protection:
                    raise DeletionProtectedError(doc_id)
                self._storage.delete_doc(doc_id)
                self._tombstone(doc_id)
                self._note_mutation()
            return self.add(
                text,
                metadata=metadata,
                namespace=namespace,
                doc_id=doc_id,
                modality=modality,
                embedding=embedding,
            )

    def update_metadata(
        self, doc_id: str, metadata: dict[str, Any], merge: bool = True
    ) -> dict[str, Any]:
        """Update a document's metadata without re-embedding.

        Args:
            doc_id: Document identifier.
            metadata: Keys to set.
            merge: If True (default) merge into the existing metadata;
                if False replace it entirely.

        Returns:
            The new metadata dictionary.

        Raises:
            DocumentNotFoundError: If the document does not exist.

        Performance:
            Time: O(log N)
        """
        with self._lock:
            self._check_open()
            self._rbac.require("write")
            doc = self._storage.get_doc(doc_id)
            if doc is None:
                raise DocumentNotFoundError(doc_id)
            new_meta = {**doc.metadata, **metadata} if merge else dict(metadata)
            self._storage.update_doc(doc_id, metadata=new_meta)
            self._note_mutation()
            self._audit.log(
                actor=self._rbac.key_id,
                operation="update_metadata",
                doc_id=doc_id,
                namespace=doc.namespace,
            )
            return new_meta

    # ─── Read API ───

    def get(self, doc_id: str) -> SearchResult | None:
        """Fetch a document by ID.

        Args:
            doc_id: Document identifier.

        Returns:
            A :class:`SearchResult` (score 1.0) or None if not found.

        Performance:
            Time: O(log N)
        """
        with self._lock:
            self._check_open()
            self._rbac.require("read")
            doc = self._storage.get_doc(doc_id)
            return self._to_result(doc, 1.0) if doc is not None else None

    def exists(self, doc_id: str) -> bool:
        """Return whether a document ID exists.

        Performance:
            Time: O(1)
        """
        with self._lock:
            self._check_open()
            return doc_id in self._doc_id_to_index

    def exists_namespace(self, name: str) -> bool:
        """Return whether a namespace exists.

        Performance:
            Time: O(1) with cache
        """
        with self._lock:
            self._check_open()
            return self._namespace_mgr.exists(name)

    def get_embedding(self, doc_id: str) -> NDArray[np.float32] | None:
        """Return the stored embedding vector of a document.

        Args:
            doc_id: Document identifier.

        Returns:
            float32 vector, or None if the document does not exist.

        Performance:
            Time: O(log N)
        """
        with self._lock:
            self._check_open()
            self._rbac.require("read")
            doc = self._storage.get_doc(doc_id)
            return None if doc is None else np.asarray(doc.embedding, dtype=np.float32)

    def count(self, namespace: str | None = None) -> int:
        """Return the number of live documents.

        Args:
            namespace: Optional namespace to count.

        Performance:
            Time: O(1)
        """
        with self._lock:
            if namespace is None:
                return len(self._doc_id_to_index)
            return len(self._ns_positions.get(namespace, ()))

    def __len__(self) -> int:
        return self.count()

    def __contains__(self, doc_id: object) -> bool:
        return isinstance(doc_id, str) and self.exists(doc_id)

    @staticmethod
    def _to_result(
        doc: StoredDocument,
        score: float,
        candidate: CascadeCandidate | None = None,
    ) -> SearchResult:
        return SearchResult(
            text=doc.text,
            score=float(score),
            metadata=doc.metadata,
            namespace=doc.namespace,
            doc_id=doc.doc_id,
            modality=doc.modality,
            timestamp=doc.created_at,
            dense_score=candidate.dense_score if candidate else None,
            sparse_score=candidate.sparse_score if candidate else None,
            signals=list(candidate.signals) if candidate else [],
            updated_at=doc.updated_at,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        rerank: bool = False,
        quantum_rerank: bool | None = None,
        namespace: str | None = None,
        filters: dict[str, Any] | None = None,
        recency_weight: float = 0.0,
        query_vector: NDArray[np.float32] | None = None,
        exclude_doc_ids: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        """Perform hybrid cascading search across the vault.

        Runs the cascade pipeline: FAISS dense retrieval → BM25 keyword
        merge → metadata filter → optional cross-encoder / Grover
        reranking. Namespace scoping and deletes are enforced inside the
        retrievers, and the candidate window widens automatically when a
        metadata filter rejects too many candidates.

        Built by Suneel Bose K · ArcGX TechLabs Private Limited.

        Args:
            query: Natural language search query.
            top_k: Number of results to return. Defaults to 10.
            alpha: Semantic weight (0.0 = keyword only, 1.0 = semantic only).
                Defaults to 0.5 (balanced hybrid).
            rerank: If True, applies cross-encoder reranking.
                Adds ~20-50ms. Defaults to False.
            quantum_rerank: If True, applies Grover amplitude amplification
                with a *consensus oracle* (candidates matched by both the
                dense and the sparse retriever). None uses the vault's
                ``quantum`` default.
            namespace: Restrict to this namespace. None = all accessible.
            filters: Metadata key-value filters.
                E.g. {"type": "NDA", "year": {"gte": 2023}}.
            recency_weight: Weight for document recency (0.0–1.0).
            query_vector: Precomputed query embedding (e.g. a CLIP image
                vector). ``query`` is then used only for BM25.
            exclude_doc_ids: Document IDs to leave out of the results.

        Returns:
            List of SearchResult sorted by descending relevance.

        Raises:
            VaultEmptyError: If vault contains no documents.
            InvalidAlphaError: If alpha outside [0.0, 1.0].
            NamespaceNotFoundError: If namespace does not exist.
            VecForgePermissionError: If API key lacks read permission.

        Performance:
            Time: O(log N) FAISS + O(k) rerank where k << N
            Classic rerank: <15ms at 100k docs
            Quantum rerank: O(k) closed-form — <1ms

        Example::

            >>> results = db.search(
            ...     "elderly diabetic hip fracture",
            ...     namespace="ward_7",
            ...     quantum_rerank=True,
            ...     top_k=5,
            ... )
            >>> print(results[0].text)
        """
        with self._lock:
            self._check_open()
            # security: Check read permission
            self._rbac.require("read")

            # why: Validate inputs
            if not 0.0 <= alpha <= 1.0:
                raise InvalidAlphaError(alpha)
            if top_k <= 0:
                return []
            recency_weight = float(min(max(recency_weight, 0.0), 1.0))
            if quantum_rerank is None:
                quantum_rerank = self._quantum

            # why: Validate namespace if specified
            if namespace is not None:
                self._namespace_mgr.validate(namespace)

            # why: Check vault has data
            indexer = self._ensure_indexer()
            if indexer.count == 0 or not self._doc_id_to_index:
                raise VaultEmptyError(self._path)

            allowed: NDArray[np.int64] | None = None
            if namespace is not None:
                allowed = self._namespace_array(namespace)
                if allowed.size == 0:
                    return []
            excluded = self._tombstone_array()
            if exclude_doc_ids:
                extra = [
                    self._doc_id_to_index[d]
                    for d in exclude_doc_ids
                    if d in self._doc_id_to_index
                ]
                if extra:
                    excl_list = (
                        excluded.tolist() if excluded is not None else []
                    ) + extra
                    excluded = np.asarray(sorted(set(excl_list)), dtype=np.int64)
            pool = (
                int(allowed.size) if allowed is not None else len(self._doc_id_to_index)
            )
            if exclude_doc_ids:
                pool = max(0, pool - len(exclude_doc_ids))
            if pool <= 0:
                return []

            # ─── Embed query ───
            if query_vector is None:
                query_vec = self._embedder.encode(query)[0]
            else:
                query_vec = self._validate_embedding(query_vector)

            cascade = CascadeSearcher(indexer=indexer, bm25=self._bm25, reranker=None)
            meta_filter = MetadataFilter(filters) if filters else None

            # ─── Retrieve with adaptive over-fetch ───
            fetch_k = min(max(top_k * 4, _MIN_FETCH), pool)
            results: list[SearchResult] = []
            while True:
                candidates = cascade.search(
                    query_vector=query_vec,
                    query_text=query,
                    top_k=fetch_k,
                    alpha=alpha,
                    over_fetch=1,
                    allowed_ids=allowed,
                    excluded_ids=excluded,
                )
                results = self._hydrate(candidates, namespace, meta_filter)
                if len(results) >= top_k or fetch_k >= pool:
                    break
                fetch_k = min(fetch_k * 4, pool)

            # ─── Quantum-inspired reranking (Grover, consensus oracle) ───
            if quantum_rerank and results:
                results = self._quantum_rerank(query, results, top_k, use_cross=rerank)
            # ─── Cross-encoder reranking (final pass) ───
            elif rerank and results:
                reranked = self._reranker.rerank(
                    query, [r.text for r in results], top_k=top_k
                )
                reranked_results = []
                for orig_idx, rerank_score in reranked:
                    if orig_idx < len(results):
                        r = results[orig_idx]
                        r.score = float(rerank_score)
                        reranked_results.append(r)
                results = reranked_results

            # ─── Recency weighting (after reranking so it is never discarded) ───
            if recency_weight > 0.0 and results:
                self._apply_recency(results, recency_weight)

            # ─── Trim to top_k ───
            results = results[:top_k]

            # security: Audit the search
            self._audit.log(
                actor=self._rbac.key_id,
                operation="search",
                namespace=namespace,
                metadata={
                    "query": query,
                    "top_k": top_k,
                    "results": len(results),
                    "quantum_rerank": bool(quantum_rerank),
                },
            )

            return results

    @staticmethod
    def _apply_recency(results: list[SearchResult], weight: float) -> None:
        """Blend min-max normalised scores with a recency factor, in place."""
        now = time.time()
        scores = [r.score for r in results]
        s_min, s_max = min(scores), max(scores)
        s_range = (s_max - s_min) or 1.0
        max_age = max((now - r.timestamp) for r in results) or 1.0
        for r in results:
            norm = (r.score - s_min) / s_range if s_max > s_min else 1.0
            age_factor = 1.0 - min(1.0, max(0.0, now - r.timestamp) / max_age)
            blended = (1.0 - weight) * norm + weight * age_factor
            r.score = float(min(1.0, max(0.0, blended)))
        results.sort(key=lambda r: r.score, reverse=True)

    def _hydrate(
        self,
        candidates: list[CascadeCandidate],
        namespace: str | None,
        meta_filter: MetadataFilter | None,
    ) -> list[SearchResult]:
        """Load candidate documents in one batched query and filter them."""
        pairs: list[tuple[CascadeCandidate, str]] = []
        for c in candidates:
            if c.doc_index < 0 or c.doc_index >= len(self._index_to_doc_id):
                continue
            if c.doc_index in self._tombstones:
                continue
            pairs.append((c, self._index_to_doc_id[c.doc_index]))

        docs = self._storage.get_docs([did for _, did in pairs])
        results: list[SearchResult] = []
        for c, did in pairs:
            doc = docs.get(did)
            if doc is None:
                continue
            # security: Namespace filtering (defence in depth)
            if namespace is not None and doc.namespace != namespace:
                continue
            if meta_filter is not None and not meta_filter.matches(doc.metadata):
                continue
            results.append(self._to_result(doc, c.score, c))
        return results

    def _quantum_rerank(
        self, query: str, results: list[SearchResult], top_k: int, use_cross: bool
    ) -> list[SearchResult]:
        q_reranker = _QuantumReranker(
            classical_reranker=self._reranker if use_cross else None
        )
        scores = [r.score for r in results]
        texts = [r.text for r in results]
        consensus = [
            i
            for i, r in enumerate(results)
            if r.dense_score is not None and r.sparse_score is not None
        ]
        marked = consensus if 0 < len(consensus) < len(results) else None
        q_ranked = q_reranker.rerank(query, texts, scores, top_k=top_k, marked=marked)
        quantum_results = []
        for orig_idx, new_score in q_ranked:
            if orig_idx < len(results):
                r = results[orig_idx]
                r.score = float(new_score)
                quantum_results.append(r)
        logger.debug(
            "Quantum rerank: %d results, oracle=%s, %s",
            len(quantum_results),
            "consensus" if marked else "top-sqrt",
            q_reranker.last_explanation,
        )
        return quantum_results

    def search_batch(
        self, queries: Sequence[str], **kwargs: Any
    ) -> list[list[SearchResult]]:
        """Run several searches, embedding all queries in one model call.

        Args:
            queries: Query strings.
            **kwargs: Passed to :meth:`search`.

        Returns:
            One result list per query.

        Performance:
            Time: one embedding call + O(Q) searches
        """
        with self._lock:
            self._check_open()
            if not queries:
                return []
            vectors = self._embedder.encode(list(queries))
            return [
                self.search(q, query_vector=vectors[i], **kwargs)
                for i, q in enumerate(queries)
            ]

    def similar(
        self,
        doc_id: str,
        top_k: int = 10,
        namespace: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Find documents similar to an existing document.

        Args:
            doc_id: Reference document.
            top_k: Number of results.
            namespace: Optional namespace restriction.
            filters: Optional metadata filters.

        Returns:
            Results excluding the reference document itself.

        Raises:
            DocumentNotFoundError: If the document does not exist.

        Performance:
            Time: same as :meth:`search` (dense only)
        """
        with self._lock:
            self._check_open()
            doc = self._storage.get_doc(doc_id)
            if doc is None:
                raise DocumentNotFoundError(doc_id)
            return self.search(
                doc.text,
                top_k=top_k,
                alpha=1.0,
                namespace=namespace,
                filters=filters,
                query_vector=doc.embedding,
                exclude_doc_ids=[doc_id],
            )

    # ─── Ingestion ───

    def ingest(
        self,
        path: str,
        namespace: str = "default",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 64,
    ) -> int:
        """Ingest documents from a file or directory.

        Auto-detects format and chunks documents for embedding.
        Supports: .txt, .md, .pdf, .docx, .html

        Built by Suneel Bose K · ArcGX TechLabs Private Limited.

        Args:
            path: File or directory path to ingest.
            namespace: Target namespace. Defaults to 'default'.
            chunk_size: Max chars per chunk. Defaults to 1000.
            chunk_overlap: Overlap between chunks. Defaults to 200.
            batch_size: Chunks embedded per batch. Defaults to 64.

        Returns:
            Number of chunks ingested.

        Raises:
            VecForgePermissionError: If API key lacks write permission.

        Performance:
            Time: O(F * S) where F = files, S = avg file size
            Typical: <5min for 1000 PDFs

        Example:
            >>> count = db.ingest("my_documents/", namespace="legal")
            >>> print(f"Ingested {count} chunks")
        """
        with self._lock:
            self._check_open()
            # security: Check write permission
            self._rbac.require("write")

            from vecforge.ingest.dispatcher import IngestDispatcher

            dispatcher = IngestDispatcher(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks = dispatcher.ingest(path)

            count = 0
            step = max(1, int(batch_size))
            for i in range(0, len(chunks), step):
                batch = chunks[i : i + step]
                self.add_batch(
                    [c.text for c in batch],
                    metadata_list=[c.metadata for c in batch],
                    namespace=namespace,
                )
                count += len(batch)

            logger.info("Ingested %d chunks from %s", count, path)
            return count

    # ─── Delete API ───

    def delete(self, doc_id: str) -> bool:
        """Delete a document by ID.

        The document is removed from storage immediately and tombstoned
        in the in-memory indexes (O(1)); indexes are compacted
        automatically once tombstones exceed 20% of the vault.

        Args:
            doc_id: Document identifier.

        Returns:
            True if document was deleted.

        Raises:
            DeletionProtectedError: If vault has deletion protection.
            VecForgePermissionError: If API key lacks delete permission.

        Performance:
            Time: O(log N)

        Example:
            >>> db.delete("a1b2c3d4-...")
            True
        """
        with self._lock:
            self._check_open()
            # security: Check delete permission
            self._rbac.require("delete")

            if self._deletion_protection:
                raise DeletionProtectedError(doc_id)

            deleted = self._storage.delete_doc(doc_id)

            if deleted:
                self._tombstone(doc_id)
                self._note_mutation()
                # security: Audit the deletion
                self._audit.log(
                    actor=self._rbac.key_id,
                    operation="delete",
                    doc_id=doc_id,
                )
                self._maybe_compact()

            return deleted

    def delete_many(self, doc_ids: Sequence[str]) -> int:
        """Delete several documents in one transaction.

        Args:
            doc_ids: Document identifiers.

        Returns:
            Number of documents deleted.

        Raises:
            DeletionProtectedError: If vault has deletion protection.

        Performance:
            Time: O(K log N)
        """
        with self._lock:
            self._check_open()
            self._rbac.require("delete")
            if self._deletion_protection:
                raise DeletionProtectedError(doc_ids[0] if doc_ids else "<batch>")
            if not doc_ids:
                return 0
            deleted = self._storage.delete_docs(list(doc_ids))
            for did in doc_ids:
                self._tombstone(did)
            if deleted:
                self._note_mutation(deleted)
                self._audit.log(
                    actor=self._rbac.key_id,
                    operation="delete_many",
                    metadata={"count": deleted},
                )
                self._maybe_compact()
            return deleted

    def delete_namespace(self, name: str) -> int:
        """Delete a namespace and every document in it.

        Args:
            name: Namespace name.

        Returns:
            Number of documents deleted.

        Raises:
            DeletionProtectedError: If vault has deletion protection.

        Performance:
            Time: O(K log N)
        """
        with self._lock:
            self._check_open()
            self._rbac.require("delete")
            if self._deletion_protection:
                raise DeletionProtectedError(f"namespace:{name}")
            positions = list(self._ns_positions.get(name, set()))
            doc_ids = [self._index_to_doc_id[p] for p in positions]
            deleted = self._storage.delete_namespace(name)
            for did in doc_ids:
                self._tombstone(did)
            self._ns_positions.pop(name, None)
            self._ns_arr_cache.pop(name, None)
            self._namespace_mgr.refresh()
            if deleted:
                self._note_mutation(deleted)
            self._audit.log(
                actor=self._rbac.key_id,
                operation="delete_namespace",
                namespace=name,
                metadata={"count": deleted},
            )
            self._maybe_compact()
            return deleted

    def compact(self) -> None:
        """Rebuild the in-memory indexes without tombstones and save.

        Performance:
            Time: O(N * d)
        """
        with self._lock:
            self._check_open()
            self._rebuild_from_storage()
            self.save()
            logger.info("Vault compacted: %d live documents", self.count())

    def reembed(self, batch_size: int = 256) -> int:
        """Re-embed every document with the current embedder and rebuild.

        Use after switching ``model_name`` on an existing vault.

        Args:
            batch_size: Documents embedded per model call.

        Returns:
            Number of documents re-embedded.

        Performance:
            Time: O(N * d) model calls
        """
        with self._lock:
            self._check_open()
            self._rbac.require("write")
            docs = list(self._storage.iter_docs())
            for i in range(0, len(docs), batch_size):
                batch = docs[i : i + batch_size]
                vectors = self._embed_texts([d.text for d in batch])
                for doc, vec in zip(batch, vectors, strict=True):
                    self._storage.update_doc(doc.doc_id, embedding=vec)
            self._rebuild_from_storage()
            self.save()
            return len(docs)

    # ─── API keys (RBAC) ───

    def register_api_key(self, api_key: str, role: str) -> None:
        """Register an API key with a role, persisted in the vault.

        Only the SHA-256 digest of the key is stored. Keys that are not
        registered get no access when passed as ``api_key``.

        Args:
            api_key: The key to register.
            role: ``admin``, ``read-write`` or ``read-only``.

        Raises:
            VecForgePermissionError: If the current key cannot manage keys.
            ValueError: If the role is invalid.

        Performance:
            Time: O(1)

        Example:
            >>> db.register_api_key(os.environ["TEAM_KEY"], "read-only")
        """
        with self._lock:
            self._check_open()
            self._rbac.register_key(api_key, role)
            self._storage.kv_set(_RBAC_KV_KEY, self._rbac.key_roles)
            self._audit.log(
                actor=self._rbac.key_id,
                operation="register_api_key",
                metadata={"role": role},
            )

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke a registered API key.

        Args:
            api_key: The key (or its digest) to revoke.

        Returns:
            True if the key was registered.

        Performance:
            Time: O(1)
        """
        with self._lock:
            self._check_open()
            removed = self._rbac.revoke_key(api_key)
            self._storage.kv_set(_RBAC_KV_KEY, self._rbac.key_roles)
            if removed:
                self._audit.log(actor=self._rbac.key_id, operation="revoke_api_key")
            return removed

    def list_api_keys(self) -> dict[str, str]:
        """List registered keys as ``digest → role`` (raw keys are never stored).

        Performance:
            Time: O(K)
        """
        with self._lock:
            self._check_open()
            self._rbac.require("manage_keys")
            return self._rbac.key_roles

    # ─── Namespaces & stats ───

    def create_namespace(self, name: str) -> None:
        """Create a new namespace for tenant isolation.

        Args:
            name: Namespace name.

        Raises:
            VecForgePermissionError: If API key lacks create_namespace.

        Performance:
            Time: O(1)

        Example:
            >>> db.create_namespace("ward_7")
        """
        with self._lock:
            self._check_open()
            self._rbac.require("create_namespace")
            self._namespace_mgr.create(name)

            self._audit.log(
                actor=self._rbac.key_id,
                operation="create_namespace",
                namespace=name,
            )

    def list_namespaces(self) -> list[str]:
        """List all namespaces in the vault.

        Returns:
            Sorted list of namespace names.

        Performance:
            Time: O(K)
        """
        with self._lock:
            self._check_open()
            return self._namespace_mgr.list_all()

    def stats(self) -> dict[str, Any]:
        """Get vault statistics.

        Returns:
            Dictionary with vault metadata and statistics.

        Performance:
            Time: O(K) namespaces

        Example:
            >>> db.stats()
            {'documents': 1500, 'namespaces': ['default', 'ward_7'], ...}
        """
        with self._lock:
            self._check_open()
            from vecforge import __version__

            ns_counts = self._storage.count_by_namespace()
            return {
                "path": self._path,
                "version": __version__,
                "documents": self._storage.count_docs(),
                "namespaces": sorted(ns_counts.keys()),
                "namespace_counts": ns_counts,
                "encrypted": self._storage.is_encrypted,
                "quantum": self._quantum,
                "deletion_protection": self._deletion_protection,
                "index_vectors": self._indexer.count if self._indexer else 0,
                "index_type": (
                    self._indexer.index_type if self._indexer else self._index_type
                ),
                "tombstones": len(self._tombstones),
                "bm25_documents": self._bm25.count,
                "embedding_model": self._embedder.name,
                "dimension": self._indexer.dimension if self._indexer else None,
                "schema_version": self._storage.schema_version,
                "built_by": "Suneel Bose K · ArcGX TechLabs Private Limited",
            }

    # ─── Persistence ───

    def save(self) -> None:
        """Persist the FAISS + BM25 index snapshot for fast reopening.

        Documents are always durable in SQLite; this stores the derived
        indexes so the next ``VecForge(path)`` skips the rebuild.

        Performance:
            Time: O(N * d)
        """
        with self._lock:
            self._check_open()
            if self._path == ":memory:":
                self._mutations_since_save = 0
                return
            if self._indexer is None or self._indexer.count == 0:
                self._storage.clear_index_state()
                self._mutations_since_save = 0
                return
            # perf: a flat index is just the stored vectors in position order,
            # so only graph indexes (HNSW) need their bytes persisted
            faiss_blob = (
                b"" if self._indexer.index_type == "flat" else self._indexer.to_bytes()
            )
            state = IndexState(
                faiss_blob=faiss_blob,
                bm25_blob=self._bm25.to_bytes(),
                doc_ids=list(self._index_to_doc_id),
                doc_namespaces=list(self._index_namespace),
                tombstones=sorted(self._tombstones),
                dimension=self._indexer.dimension,
                generation=self._storage.generation,
                doc_count=self._storage.count_docs(),
                model_name=self._embedder.name,
                index_type=self._indexer.index_type,
            )
            self._storage.save_index_state(state)
            self._mutations_since_save = 0
            logger.info("Index snapshot saved (%d vectors)", self._indexer.count)

    def backup(self, dest_path: str) -> str:
        """Write a consistent copy of the vault file (safe while open).

        Args:
            dest_path: Destination path.

        Returns:
            The destination path.

        Performance:
            Time: O(file size)
        """
        with self._lock:
            self._check_open()
            self._rbac.require("backup")
            self.save()
            return self._storage.backup(dest_path)

    def close(self) -> None:
        """Save state and close the vault.

        Performance:
            Time: O(N * d) for index save + O(1) for connection close
        """
        with self._lock:
            if self._closed:
                return
            try:
                if self._mutations_since_save or self._path != ":memory:":
                    self.save()
            finally:
                self._storage.close()
                self._closed = True
            logger.info("VecForge vault closed: %s", self._path)

    @property
    def closed(self) -> bool:
        """Return whether the vault has been closed.

        Performance:
            Time: O(1)
        """
        return self._closed

    @property
    def embedder(self) -> EmbedderLike:
        """Return the active embedding engine.

        Performance:
            Time: O(1)
        """
        return self._embedder

    @property
    def storage(self) -> StorageBackend:
        """Return the storage backend (sessions, key/value store, backups).

        Performance:
            Time: O(1)
        """
        return self._storage

    @property
    def path(self) -> str:
        """Return the vault path.

        Performance:
            Time: O(1)
        """
        return self._path

    def __enter__(self) -> VecForge:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit — auto-save and close."""
        self.close()

    def __repr__(self) -> str:
        if self._closed:
            return f"VecForge(path='{self._path}', closed=True)"
        count = self._storage.count_docs()
        return (
            f"VecForge(path='{self._path}', docs={count}, "
            f"encrypted={self._storage.is_encrypted})"
        )
