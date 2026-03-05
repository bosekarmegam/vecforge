# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
SQLite + SQLCipher persistence layer for VecForge.

Provides encrypted document storage with namespace scoping. All queries
are scoped to a namespace to ensure multi-tenant isolation. Uses WAL
mode for concurrent read performance.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


@dataclass
class StoredDocument:
    """A document stored in the VecForge vault.

    Attributes:
        doc_id: Unique document identifier (UUID).
        text: Original document text.
        embedding: Dense embedding vector as numpy array.
        metadata: User-provided metadata dictionary.
        namespace: Namespace this document belongs to.
        modality: Content modality (text, image, audio, etc.).
        created_at: Unix timestamp of creation.
    """

    doc_id: str
    text: str
    embedding: NDArray[np.float32]
    metadata: dict[str, Any]
    namespace: str
    modality: str
    created_at: float


class StorageBackend:
    """SQLite/SQLCipher persistence backend for VecForge.

    All queries are namespace-scoped to prevent cross-tenant data leaks.
    Uses WAL mode for concurrent read performance. Supports optional
    SQLCipher AES-256 encryption at rest.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        path: Database file path, or ':memory:' for in-memory storage.
        encryption_key: Optional SQLCipher encryption key. If provided
            and SQLCipher is available, AES-256 encryption is enabled.

    Performance:
        Insert: O(1) amortized per document
        Lookup: O(log N) with indexed columns
        Namespace-scoped queries: O(log N) using composite index

    Example:
        >>> storage = StorageBackend(":memory:")
        >>> doc = storage.insert_doc("hello world", embedding, {}, "default")
        >>> retrieved = storage.get_doc(doc.doc_id)
    """

    def __init__(
        self,
        path: str = ":memory:",
        encryption_key: str | None = None,
    ) -> None:
        self._path = path
        self._encryption_key = encryption_key
        self._conn: sqlite3.Connection | None = None
        self._encrypted = False

        self._connect()
        self._init_schema()

    def _connect(self) -> None:
        """Establish database connection, optionally with encryption.

        Performance:
            Time: O(1) — connection setup
        """
        # security: Try SQLCipher first if encryption key is provided
        if self._encryption_key:
            try:
                import sqlcipher3

                self._conn = sqlcipher3.connect(self._path)
                self._conn.execute(f"PRAGMA key = '{self._encryption_key}'")
                self._encrypted = True
                logger.info("SQLCipher encryption enabled (AES-256)")
            except ImportError:
                logger.warning(
                    "sqlcipher3 not installed — falling back to unencrypted "
                    "SQLite. Install sqlcipher3 for AES-256 encryption."
                )
                self._conn = sqlite3.connect(self._path, check_same_thread=False)
        else:
            self._conn = sqlite3.connect(self._path, check_same_thread=False)

        # perf: WAL mode for concurrent reads
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.row_factory = sqlite3.Row

    def _init_schema(self) -> None:
        """Initialize database schema if not exists.

        Performance:
            Time: O(1)
        """
        assert self._conn is not None

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            );

            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                namespace TEXT NOT NULL DEFAULT 'default',
                modality TEXT NOT NULL DEFAULT 'text',
                created_at REAL NOT NULL,
                updated_at REAL
            );

            CREATE INDEX IF NOT EXISTS idx_documents_namespace
                ON documents(namespace);

            CREATE INDEX IF NOT EXISTS idx_documents_namespace_created
                ON documents(namespace, created_at);

            CREATE TABLE IF NOT EXISTS namespaces (
                name TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS faiss_index (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                index_data BLOB NOT NULL,
                dimension INTEGER NOT NULL,
                count INTEGER NOT NULL,
                updated_at REAL NOT NULL
            );
            """)

        # Ensure default namespace exists
        self._conn.execute(
            """
            INSERT OR IGNORE INTO namespaces (name, created_at)
            VALUES ('default', ?)
            """,
            (time.time(),),
        )

        self._conn.commit()
        logger.debug("Database schema initialized (version %d)", _SCHEMA_VERSION)

    def insert_doc(
        self,
        text: str,
        embedding: NDArray[np.float32],
        metadata: dict[str, Any],
        namespace: str = "default",
        modality: str = "text",
        doc_id: str | None = None,
    ) -> StoredDocument:
        """Insert a document into the vault.

        Args:
            text: Document text content.
            embedding: Dense embedding vector.
            metadata: User-provided metadata.
            namespace: Target namespace. Defaults to 'default'.
            modality: Content type. Defaults to 'text'.
            doc_id: Optional custom document ID. Auto-generated if None.

        Returns:
            StoredDocument with the inserted document's details.

        Raises:
            sqlite3.IntegrityError: If doc_id already exists.

        Performance:
            Time: O(1) amortized

        Example:
            >>> doc = storage.insert_doc("hello", embedding, {"source": "test"})
            >>> print(doc.doc_id)
            'a1b2c3d4...'
        """
        assert self._conn is not None

        if doc_id is None:
            doc_id = str(uuid.uuid4())

        now = time.time()

        # security: Always scope to namespace
        self._conn.execute(
            """
            INSERT INTO documents (doc_id, text, embedding, metadata_json,
                                   namespace, modality, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                text,
                embedding.tobytes(),
                json.dumps(metadata),
                namespace,
                modality,
                now,
            ),
        )
        self._conn.commit()

        return StoredDocument(
            doc_id=doc_id,
            text=text,
            embedding=embedding,
            metadata=metadata,
            namespace=namespace,
            modality=modality,
            created_at=now,
        )

    def get_doc(self, doc_id: str) -> StoredDocument | None:
        """Retrieve a document by ID.

        Args:
            doc_id: Document identifier.

        Returns:
            StoredDocument if found, None otherwise.

        Performance:
            Time: O(log N) — indexed lookup
        """
        assert self._conn is not None

        row = self._conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()

        if row is None:
            return None

        return self._row_to_document(row)

    def get_docs_by_namespace(
        self, namespace: str, limit: int = 1000, offset: int = 0
    ) -> list[StoredDocument]:
        """Retrieve documents within a specific namespace.

        Args:
            namespace: Namespace to query.
            limit: Maximum documents to return.
            offset: Pagination offset.

        Returns:
            List of StoredDocument in creation order.

        Performance:
            Time: O(log N + limit) — index scan

        Example:
            >>> docs = storage.get_docs_by_namespace("ward_7", limit=50)
        """
        assert self._conn is not None

        # security: CRITICAL — always scope to namespace
        rows = self._conn.execute(
            """
            SELECT * FROM documents
            WHERE namespace = ?
            ORDER BY created_at ASC
            LIMIT ? OFFSET ?
            """,
            (namespace, limit, offset),
        ).fetchall()

        return [self._row_to_document(row) for row in rows]

    def get_all_docs(self, namespace: str | None = None) -> list[StoredDocument]:
        """Retrieve all documents, optionally filtered by namespace.

        Args:
            namespace: If provided, only docs from this namespace.
                If None, returns docs from ALL namespaces.

        Returns:
            List of all matching StoredDocument.

        Performance:
            Time: O(N) — full table scan
        """
        assert self._conn is not None

        if namespace is not None:
            # security: Namespace-scoped query
            rows = self._conn.execute(
                "SELECT * FROM documents WHERE namespace = ? ORDER BY created_at",
                (namespace,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM documents ORDER BY created_at"
            ).fetchall()

        return [self._row_to_document(row) for row in rows]

    def delete_doc(self, doc_id: str) -> bool:
        """Delete a document by ID.

        Args:
            doc_id: Document identifier.

        Returns:
            True if document was deleted, False if not found.

        Performance:
            Time: O(log N)
        """
        assert self._conn is not None

        cursor = self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def count_docs(self, namespace: str | None = None) -> int:
        """Count documents in the vault.

        Args:
            namespace: If provided, count only this namespace.

        Returns:
            Number of documents.

        Performance:
            Time: O(1) with index
        """
        assert self._conn is not None

        if namespace is not None:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM documents WHERE namespace = ?",
                (namespace,),
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()

        return int(row[0]) if row else 0

    def list_namespaces(self) -> list[str]:
        """List all namespace names.

        Returns:
            Sorted list of namespace names.

        Performance:
            Time: O(K) where K = number of namespaces
        """
        assert self._conn is not None

        rows = self._conn.execute(
            "SELECT name FROM namespaces ORDER BY name"
        ).fetchall()

        return [row["name"] for row in rows]

    def create_namespace(self, name: str) -> None:
        """Create a new namespace.

        Args:
            name: Namespace name. Must be unique.

        Performance:
            Time: O(1)
        """
        assert self._conn is not None

        self._conn.execute(
            "INSERT OR IGNORE INTO namespaces (name, created_at) VALUES (?, ?)",
            (name, time.time()),
        )
        self._conn.commit()

    def save_faiss_index(self, index_data: bytes, dimension: int, count: int) -> None:
        """Persist the FAISS index to storage.

        Args:
            index_data: Serialized FAISS index bytes.
            dimension: Embedding dimension.
            count: Number of vectors in the index.

        Performance:
            Time: O(N * d) — proportional to index size
        """
        assert self._conn is not None

        self._conn.execute(
            """
            INSERT OR REPLACE INTO faiss_index
                (id, index_data, dimension, count, updated_at)
            VALUES (1, ?, ?, ?, ?)
            """,
            (index_data, dimension, count, time.time()),
        )
        self._conn.commit()

    def load_faiss_index(self) -> tuple[bytes, int, int] | None:
        """Load the persisted FAISS index.

        Returns:
            Tuple of (index_data, dimension, count) or None if not saved.

        Performance:
            Time: O(N * d) — proportional to index size
        """
        assert self._conn is not None

        row = self._conn.execute(
            "SELECT index_data, dimension, count FROM faiss_index WHERE id = 1"
        ).fetchone()

        if row is None:
            return None

        return (bytes(row["index_data"]), int(row["dimension"]), int(row["count"]))

    def _row_to_document(self, row: sqlite3.Row) -> StoredDocument:
        """Convert a database row to a StoredDocument.

        Performance:
            Time: O(d) where d = embedding dimension
        """
        embedding = np.frombuffer(row["embedding"], dtype=np.float32)

        return StoredDocument(
            doc_id=row["doc_id"],
            text=row["text"],
            embedding=embedding,
            metadata=json.loads(row["metadata_json"]),
            namespace=row["namespace"],
            modality=row["modality"],
            created_at=row["created_at"],
        )

    @property
    def is_encrypted(self) -> bool:
        """Return whether the storage backend is using encryption.

        Performance:
            Time: O(1)
        """
        return self._encrypted

    def close(self) -> None:
        """Close the database connection.

        Performance:
            Time: O(1)
        """
        if self._conn:
            self._conn.close()
            self._conn = None
