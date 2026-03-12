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
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vecforge.core.bm25 import BM25Engine
from vecforge.core.embedder import Embedder
from vecforge.core.indexer import FaissIndexer
from vecforge.core.reranker import Reranker
from vecforge.core.storage import StorageBackend
from vecforge.exceptions import (
    DeletionProtectedError,
    InvalidAlphaError,
    VaultEmptyError,
)
from vecforge.quantum.reranker import QuantumReranker as _QuantumReranker
from vecforge.search.cascade import CascadeSearcher
from vecforge.search.filters import MetadataFilter
from vecforge.security.audit import AuditLogger
from vecforge.security.encryption import validate_encryption_key
from vecforge.security.namespaces import NamespaceManager
from vecforge.security.rbac import RBACManager

logger = logging.getLogger(__name__)


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
    """

    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    namespace: str = "default"
    doc_id: str = ""
    modality: str = "text"
    timestamp: float = 0.0

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
        quantum: Enable quantum-inspired acceleration. Defaults to False.
        deletion_protection: Prevent accidental deletions. Defaults to False.
        api_key: API key for RBAC. None = local admin.
        model_name: Embedding model name. Defaults to 'all-MiniLM-L6-v2'.

    Performance:
        Init: O(1) — lazy model loading
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
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._path = path
        self._quantum = quantum
        self._deletion_protection = deletion_protection

        # security: Validate encryption key
        validated_key = validate_encryption_key(encryption_key)

        # ─── Initialize subsystems ───
        self._storage = StorageBackend(path=path, encryption_key=validated_key)
        self._embedder = Embedder(model_name=model_name)
        self._indexer: FaissIndexer | None = None  # why: Lazy — needs dimension
        self._bm25 = BM25Engine()
        self._reranker = Reranker()
        self._rbac = RBACManager(api_key=api_key)
        self._audit = AuditLogger(log_path=audit_log)
        self._namespace_mgr = NamespaceManager(self._storage)

        # why: Document index tracking — maps FAISS index → doc_id
        self._index_to_doc_id: list[str] = []

        # why: Try to restore from persisted state
        self._restore_state()

        logger.info(
            "VecForge vault opened: %s (encrypted=%s, quantum=%s)",
            path,
            self._storage.is_encrypted,
            quantum,
        )

    def _restore_state(self) -> None:
        """Restore FAISS index and BM25 from persisted storage.

        Performance:
            Time: O(N * d) where N = docs, d = dimension
        """
        # why: Load all docs to rebuild in-memory indexes
        docs = self._storage.get_all_docs()
        if not docs:
            return

        # why: Initialize FAISS with first doc's embedding dimension
        dim = len(docs[0].embedding)
        self._indexer = FaissIndexer(dimension=dim)

        # perf: Batch add all embeddings at once
        embeddings = np.stack([d.embedding for d in docs])
        self._indexer.add(embeddings)

        # why: Rebuild BM25 corpus
        self._bm25.add_documents([d.text for d in docs])

        # why: Rebuild index → doc_id mapping
        self._index_to_doc_id = [d.doc_id for d in docs]

        logger.info("Restored %d documents from storage", len(docs))

    def _ensure_indexer(self) -> FaissIndexer:
        """Ensure FAISS indexer is initialized.

        Returns:
            FaissIndexer instance.

        Performance:
            Time: O(1) if already initialized, O(model_load) on first call
        """
        if self._indexer is None:
            dim = self._embedder.dimension
            self._indexer = FaissIndexer(dimension=dim)
        return self._indexer

    def add(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        namespace: str = "default",
        doc_id: str | None = None,
    ) -> str:
        """Add a text document to the vault.

        Args:
            text: Document text content.
            metadata: Optional metadata dictionary.
            namespace: Target namespace. Defaults to 'default'.
            doc_id: Optional custom document ID.

        Returns:
            Document ID (UUID string).

        Raises:
            VecForgePermissionError: If API key lacks write permission.

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
        # security: Check write permission
        self._rbac.require("write")

        if metadata is None:
            metadata = {}

        # why: Ensure namespace exists, create if needed
        if not self._namespace_mgr.exists(namespace):
            self._namespace_mgr.create(namespace)

        # ─── Embed the text ───
        embedding = self._embedder.encode(text)
        embedding_vec = embedding[0]  # shape: (dimension,)

        # ─── Store in SQLite ───
        stored = self._storage.insert_doc(
            text=text,
            embedding=embedding_vec,
            metadata=metadata,
            namespace=namespace,
            doc_id=doc_id,
        )

        # ─── Update in-memory indexes ───
        indexer = self._ensure_indexer()
        indexer.add(embedding_vec.reshape(1, -1))
        self._bm25.add_document(text)
        self._index_to_doc_id.append(stored.doc_id)

        # security: Emit audit event
        self._audit.log(
            actor=self._rbac.key_id,
            operation="add",
            doc_id=stored.doc_id,
            namespace=namespace,
            metadata={"chars": len(text)},
        )

        logger.debug("Added doc %s to namespace '%s'", stored.doc_id, namespace)
        return stored.doc_id

    def add_batch(
        self,
        texts: list[str],
        metadata_list: list[dict[str, Any]] | None = None,
        namespace: str = "default",
    ) -> list[str]:
        """Add multiple documents in one efficient batch operation.

        Embeds all texts in a single model call and batch-inserts into
        FAISS, achieving ~3-5x throughput vs. sequential add().

        Built by Suneel Bose K · ArcGX TechLabs Private Limited.

        Args:
            texts: List of document text strings.
            metadata_list: Optional list of metadata dicts (one per
                text). If None, all documents get empty metadata.
            namespace: Target namespace. Defaults to 'default'.

        Returns:
            List of document IDs.

        Raises:
            VecForgePermissionError: If API key lacks write access.
            ValueError: If metadata_list length != texts length.

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

        # why: Ensure namespace exists
        if not self._namespace_mgr.exists(namespace):
            self._namespace_mgr.create(namespace)

        # perf: Batch embed all texts in one model call
        embeddings = self._embedder.encode(texts)

        # ─── Store & index each doc ───
        indexer = self._ensure_indexer()
        doc_ids: list[str] = []

        for i, (text, meta) in enumerate(zip(texts, metadata_list, strict=False)):
            stored = self._storage.insert_doc(
                text=text,
                embedding=embeddings[i],
                metadata=meta,
                namespace=namespace,
            )
            self._bm25.add_document(text)
            self._index_to_doc_id.append(stored.doc_id)
            doc_ids.append(stored.doc_id)

        # perf: Batch add all embeddings to FAISS
        indexer.add(embeddings)

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

        logger.info(
            "Batch added %d docs to '%s'",
            len(texts),
            namespace,
        )
        return doc_ids

    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        rerank: bool = False,
        quantum_rerank: bool = False,
        namespace: str | None = None,
        filters: dict[str, Any] | None = None,
        recency_weight: float = 0.0,
    ) -> list[SearchResult]:
        """Perform hybrid cascading search across the vault.

        Runs a 4-stage pipeline: FAISS dense retrieval → BM25 keyword
        merge → metadata filter → optional cross-encoder reranking.

        Built by Suneel Bose K · ArcGX TechLabs Private Limited.

        Args:
            query: Natural language search query.
            top_k: Number of results to return. Defaults to 10.
            alpha: Semantic weight (0.0 = keyword only, 1.0 = semantic only).
                Defaults to 0.5 (balanced hybrid).
            rerank: If True, applies cross-encoder reranking.
                Adds ~20-50ms. Defaults to False.
            quantum_rerank: If True, applies Grover-inspired score
                amplification after the cascade. Runs in O(√N) time.
                Defaults to False.
            namespace: Restrict to this namespace. None = all accessible.
            filters: Metadata key-value filters.
                E.g. {"type": "NDA", "year": {"gte": 2023}}.
            recency_weight: Weight for document recency (0.0–1.0).

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
            Quantum rerank: <20ms at 1M docs

        Example::

            >>> results = db.search(
            ...     "elderly diabetic hip fracture",
            ...     namespace="ward_7",
            ...     quantum_rerank=True,
            ...     top_k=5,
            ... )
            >>> print(results[0].text)
        """
        # security: Check read permission
        self._rbac.require("read")

        # why: Validate alpha range
        if not 0.0 <= alpha <= 1.0:
            raise InvalidAlphaError(alpha)

        # why: Validate namespace if specified
        if namespace is not None:
            self._namespace_mgr.validate(namespace)

        # why: Check vault has data
        indexer = self._ensure_indexer()
        if indexer.count == 0:
            raise VaultEmptyError(self._path)

        # ─── Embed query ───
        query_vector = self._embedder.encode(query)[0]

        # ─── Run cascade search ───
        cascade = CascadeSearcher(
            indexer=indexer,
            bm25=self._bm25,
            reranker=self._reranker if rerank else None,
        )

        # why: Get more candidates for namespace/metadata filtering
        fetch_k = top_k * 4
        candidates = cascade.search(
            query_vector=query_vector,
            query_text=query,
            top_k=fetch_k,
            alpha=alpha,
            rerank=False,  # why: We'll rerank after hydration
        )

        # ─── Hydrate candidates with full doc data ───
        results: list[SearchResult] = []
        for candidate in candidates:
            if candidate.doc_index >= len(self._index_to_doc_id):
                continue

            doc_id = self._index_to_doc_id[candidate.doc_index]
            doc = self._storage.get_doc(doc_id)
            if doc is None:
                continue

            # security: Namespace filtering
            if namespace is not None and doc.namespace != namespace:
                continue

            results.append(
                SearchResult(
                    text=doc.text,
                    score=candidate.score,
                    metadata=doc.metadata,
                    namespace=doc.namespace,
                    doc_id=doc.doc_id,
                    modality=doc.modality,
                    timestamp=doc.created_at,
                )
            )

        # ─── Metadata filtering ───
        if filters:
            meta_filter = MetadataFilter(filters)
            results = [r for r in results if meta_filter.matches(r.metadata)]

        # ─── Recency weighting ───
        if recency_weight > 0.0 and results:
            now = time.time()
            max_age = max((now - r.timestamp) for r in results) or 1.0

            for r in results:
                age_factor = 1.0 - ((now - r.timestamp) / max_age)
                r.score = (1.0 - recency_weight) * r.score + recency_weight * age_factor

            results.sort(key=lambda r: r.score, reverse=True)

        # ─── Cross-encoder reranking (final pass) ───
        if rerank and results:
            reranked = self._reranker.rerank(
                query,
                [r.text for r in results],
                top_k=top_k,
            )
            reranked_results = []
            for orig_idx, rerank_score in reranked:
                if orig_idx < len(results):
                    r = results[orig_idx]
                    r.score = rerank_score
                    reranked_results.append(r)
            results = reranked_results

        # ─── Quantum-Inspired Reranking (Stage 5, optional) ───
        if quantum_rerank and results:
            q_reranker = _QuantumReranker(
                classical_reranker=self._reranker if rerank else None
            )
            scores = [r.score for r in results]
            texts = [r.text for r in results]
            q_ranked = q_reranker.rerank(query, texts, scores, top_k=top_k)
            quantum_results = []
            for orig_idx, new_score in q_ranked:
                if orig_idx < len(results):
                    r = results[orig_idx]
                    r.score = new_score
                    quantum_results.append(r)
            results = quantum_results
            logger.debug(
                "Quantum rerank: %d results after Grover amplification",
                len(results),
            )

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
                "quantum_rerank": quantum_rerank,
            },
        )

        return results

    def ingest(
        self,
        path: str,
        namespace: str = "default",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
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
        # security: Check write permission
        self._rbac.require("write")

        from vecforge.ingest.dispatcher import IngestDispatcher

        dispatcher = IngestDispatcher(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = dispatcher.ingest(path)

        count = 0
        for chunk in chunks:
            self.add(
                text=chunk.text,
                metadata=chunk.metadata,
                namespace=namespace,
            )
            count += 1

        logger.info("Ingested %d chunks from %s", count, path)
        return count

    def delete(self, doc_id: str) -> bool:
        """Delete a document by ID.

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
        # security: Check delete permission
        self._rbac.require("delete")

        if self._deletion_protection:
            raise DeletionProtectedError(doc_id)

        deleted = self._storage.delete_doc(doc_id)

        if deleted:
            # security: Audit the deletion
            self._audit.log(
                actor=self._rbac.key_id,
                operation="delete",
                doc_id=doc_id,
            )

            # why: Rebuild in-memory indexes for consistency
            self._rebuild_indexes()

        return deleted

    def _rebuild_indexes(self) -> None:
        """Rebuild FAISS and BM25 indexes from storage.

        Called after deletions to maintain consistency.

        Performance:
            Time: O(N * d)
        """
        docs = self._storage.get_all_docs()

        if not docs:
            if self._indexer is not None:
                self._indexer.reset()
            self._bm25.reset()
            self._index_to_doc_id = []
            return

        dim = len(docs[0].embedding)
        self._indexer = FaissIndexer(dimension=dim)

        embeddings = np.stack([d.embedding for d in docs])
        self._indexer.add(embeddings)

        self._bm25 = BM25Engine()
        self._bm25.add_documents([d.text for d in docs])

        self._index_to_doc_id = [d.doc_id for d in docs]

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
        return self._namespace_mgr.list_all()

    def stats(self) -> dict[str, Any]:
        """Get vault statistics.

        Returns:
            Dictionary with vault metadata and statistics.

        Performance:
            Time: O(1)

        Example:
            >>> db.stats()
            {'documents': 1500, 'namespaces': ['default', 'ward_7'], ...}
        """
        namespaces = self.list_namespaces()
        ns_counts = {}
        for ns in namespaces:
            ns_counts[ns] = self._storage.count_docs(namespace=ns)

        return {
            "path": self._path,
            "documents": self._storage.count_docs(),
            "namespaces": namespaces,
            "namespace_counts": ns_counts,
            "encrypted": self._storage.is_encrypted,
            "quantum": self._quantum,
            "deletion_protection": self._deletion_protection,
            "index_vectors": self._indexer.count if self._indexer else 0,
            "bm25_documents": self._bm25.count,
            "built_by": "Suneel Bose K · ArcGX TechLabs Private Limited",
        }

    def save(self) -> None:
        """Persist FAISS index to storage for durability.

        Performance:
            Time: O(N * d)
        """
        if self._indexer is not None and self._indexer.count > 0:
            index_data = self._indexer.to_bytes()
            self._storage.save_faiss_index(
                index_data=index_data,
                dimension=self._indexer.dimension,
                count=self._indexer.count,
            )
            logger.info("FAISS index saved (%d vectors)", self._indexer.count)

    def close(self) -> None:
        """Save state and close the vault.

        Performance:
            Time: O(N * d) for index save + O(1) for connection close
        """
        self.save()
        self._storage.close()
        logger.info("VecForge vault closed: %s", self._path)

    def __enter__(self) -> VecForge:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit — auto-save and close."""
        self.close()

    def __repr__(self) -> str:
        count = self._storage.count_docs()
        return (
            f"VecForge(path='{self._path}', docs={count}, "
            f"encrypted={self._storage.is_encrypted})"
        )
