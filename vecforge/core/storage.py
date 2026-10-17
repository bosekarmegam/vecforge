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

Schema version 2 (VecForge 2.0) adds:

* ``content_hash`` on documents for O(1) duplicate detection;
* an ``index_state`` table that snapshots the FAISS index, the BM25
  inverted index, the position→doc_id map and tombstones, so reopening a
  vault is a single blob read instead of an O(N·d) rebuild;
* ``sessions`` and ``kv`` tables backing the AI memory layer;
* a monotonically increasing document *generation* counter used to
  validate index snapshots.

Version-1 vaults are migrated in place on open.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import sqlite3
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 2
_IN_CHUNK = 500
# why: SQLite caps a single BLOB at ~1 GB; a 1M×384 float32 FAISS index is
# 1.5 GB, so index snapshots are stored in fixed-size chunks.
_BLOB_CHUNK_BYTES = 64 * 1024 * 1024
_GENERATION_KEY = "__doc_generation__"
# security: keys starting with this prefix are owned by VecForge itself
# (generation counter, API-key registry) and cannot be set through kv_set
_RESERVED_PREFIX = "__"


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
        updated_at: Unix timestamp of the last update (None if never).
        content_hash: SHA-256 of ``namespace + text`` for dedup.
    """

    doc_id: str
    text: str
    embedding: NDArray[np.float32]
    metadata: dict[str, Any]
    namespace: str
    modality: str
    created_at: float
    updated_at: float | None = None
    content_hash: str = ""


@dataclass
class IndexState:
    """Snapshot of the in-memory indexes persisted alongside the documents.

    Attributes:
        faiss_blob: Serialised FAISS index.
        bm25_blob: Serialised BM25 inverted index.
        doc_ids: doc_id for every index position (including tombstones).
        doc_namespaces: namespace for every index position.
        tombstones: Index positions that are deleted.
        dimension: Embedding dimension.
        generation: Document generation counter at snapshot time.
        doc_count: Live document count at snapshot time.
        model_name: Embedder name at snapshot time.
        index_type: FAISS index type (``flat`` / ``hnsw``).
        updated_at: Unix timestamp of the snapshot.
    """

    faiss_blob: bytes
    bm25_blob: bytes
    doc_ids: list[str]
    doc_namespaces: list[str]
    tombstones: list[int]
    dimension: int
    generation: int
    doc_count: int
    model_name: str = ""
    index_type: str = "flat"
    updated_at: float = 0.0


@dataclass
class SessionRecord:
    """A memory session (one conversation / agent run).

    Attributes:
        session_id: Unique session identifier.
        agent: Agent name the session belongs to.
        started_at: Unix timestamp when the session started.
        ended_at: Unix timestamp when it ended (None while open).
        summary: Free-text summary written at session end.
        metadata: Arbitrary metadata.
        turn_count: Number of memories written during the session.
    """

    session_id: str
    agent: str
    started_at: float
    ended_at: float | None = None
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    turn_count: int = 0


def content_hash(namespace: str, text: str) -> str:
    """Compute the dedup hash for a document.

    Args:
        namespace: Document namespace.
        text: Document text.

    Returns:
        Hex SHA-256 digest of ``namespace + "\\x00" + text``.

    Performance:
        Time: O(len(text))
    """
    h = hashlib.sha256()
    h.update(namespace.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


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
        Insert: O(1) amortized per document; batch inserts run in one
            transaction.
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
        self._db: Any = sqlite3  # driver module: sqlite3 or sqlcipher3
        self._encrypted = False
        self._schema_version = 0
        self._tx_depth = 0

        self._connect()
        self._init_schema()

    # ─── Connection ───

    @staticmethod
    def _escape_pragma_key(key: str) -> str:
        """Escape a key for use inside a single-quoted PRAGMA literal.

        SQLCipher's ``PRAGMA key`` does not accept bound parameters, so the
        key is embedded in SQL; doubling single quotes is the SQL-standard
        escape and prevents a key containing ``'`` from breaking out of
        the literal.
        """
        return key.replace("'", "''")

    def _connect(self) -> None:
        """Establish database connection, optionally with encryption.

        Performance:
            Time: O(1) — connection setup
        """
        # security: Try SQLCipher first if encryption key is provided
        if self._encryption_key:
            try:
                import sqlcipher3

                self._db = sqlcipher3
                self._conn = sqlcipher3.connect(
                    self._path, check_same_thread=False, isolation_level=None
                )
                escaped = self._escape_pragma_key(self._encryption_key)
                self._conn.execute(f"PRAGMA key = '{escaped}'")
                # why: fail fast on a wrong key instead of at first query
                self._conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
                self._encrypted = True
                logger.info("SQLCipher encryption enabled (AES-256)")
            except ImportError:
                logger.warning(
                    "sqlcipher3 not installed — falling back to unencrypted "
                    "SQLite. Install sqlcipher3 for AES-256 encryption."
                )
                self._conn = sqlite3.connect(
                    self._path, check_same_thread=False, isolation_level=None
                )
        else:
            # why: autocommit + explicit transactions (see _tx) — Python's
            # implicit transactions leak after a failed statement and break
            # the next explicit BEGIN
            self._conn = sqlite3.connect(
                self._path, check_same_thread=False, isolation_level=None
            )

        # perf: WAL mode for concurrent reads (no-op for :memory:)
        if self._path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.row_factory = self._db.Row

    @contextmanager
    def _tx(self) -> Iterator[None]:
        """Run the enclosed statements in one transaction (re-entrant)."""
        assert self._conn is not None
        if self._tx_depth:
            self._tx_depth += 1
            try:
                yield
            finally:
                self._tx_depth -= 1
            return
        self._conn.execute("BEGIN IMMEDIATE")
        self._tx_depth = 1
        try:
            yield
        except BaseException:
            self._tx_depth = 0
            self._conn.execute("ROLLBACK")
            raise
        self._tx_depth = 0
        self._conn.execute("COMMIT")

    @property
    def IntegrityError(self) -> type[Exception]:  # noqa: N802 - mirrors DB-API
        """The driver's IntegrityError class (sqlite3 or sqlcipher3)."""
        err: type[Exception] = self._db.IntegrityError
        return err

    @property
    def Error(self) -> type[Exception]:  # noqa: N802 - mirrors DB-API
        """The driver's base Error class (sqlite3 or sqlcipher3)."""
        err: type[Exception] = self._db.Error
        return err

    def _init_schema(self) -> None:
        """Initialize or migrate the database schema.

        Performance:
            Time: O(1) for fresh vaults; O(N) once when migrating v1 → v2
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
                updated_at REAL,
                content_hash TEXT NOT NULL DEFAULT ''
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

            CREATE TABLE IF NOT EXISTS index_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                faiss_blob BLOB NOT NULL,
                bm25_blob BLOB NOT NULL,
                doc_ids BLOB NOT NULL,
                doc_namespaces BLOB NOT NULL DEFAULT x'',
                tombstones TEXT NOT NULL DEFAULT '[]',
                dimension INTEGER NOT NULL,
                generation INTEGER NOT NULL,
                doc_count INTEGER NOT NULL,
                model_name TEXT NOT NULL DEFAULT '',
                index_type TEXT NOT NULL DEFAULT 'flat',
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS index_chunks (
                kind TEXT NOT NULL,
                seq INTEGER NOT NULL,
                data BLOB NOT NULL,
                PRIMARY KEY (kind, seq)
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                agent TEXT NOT NULL DEFAULT 'default',
                started_at REAL NOT NULL,
                ended_at REAL,
                summary TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                turn_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_agent_started
                ON sessions(agent, started_at);

            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            """)

        # ─── Migration: v1 vaults lack content_hash ───
        columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(documents)").fetchall()
        }
        if "content_hash" not in columns:
            self._conn.execute(
                "ALTER TABLE documents ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''"
            )
        if "updated_at" not in columns:
            self._conn.execute("ALTER TABLE documents ADD COLUMN updated_at REAL")

        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_content_hash "
            "ON documents(namespace, content_hash)"
        )

        row = self._conn.execute(
            "SELECT MAX(version) AS v FROM schema_version"
        ).fetchone()
        current = int(row["v"]) if row and row["v"] is not None else 0

        if current < 2:
            # why: backfill dedup hashes for v1 rows (one-time, O(N))
            missing = self._conn.execute(
                "SELECT doc_id, namespace, text FROM documents WHERE content_hash = ''"
            ).fetchall()
            if missing:
                self._conn.executemany(
                    "UPDATE documents SET content_hash = ? WHERE doc_id = ?",
                    [
                        (content_hash(r["namespace"], r["text"]), r["doc_id"])
                        for r in missing
                    ],
                )
                logger.info("Migrated %d documents to schema v2", len(missing))
            self._conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
                (_SCHEMA_VERSION,),
            )

        # Ensure default namespace exists
        self._conn.execute(
            "INSERT OR IGNORE INTO namespaces (name, created_at) VALUES ('default', ?)",
            (time.time(),),
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO kv (key, value_json, updated_at) VALUES (?, '0', ?)",
            (_GENERATION_KEY, time.time()),
        )
        self._conn.commit()
        self._schema_version = _SCHEMA_VERSION
        logger.debug("Database schema initialized (version %d)", _SCHEMA_VERSION)

    # ─── Generation counter ───

    def _bump_generation(self) -> int:
        """Increment the document generation counter (inside the caller's txn)."""
        assert self._conn is not None
        self._conn.execute(
            "UPDATE kv SET value_json = "
            "CAST((CAST(value_json AS INTEGER) + 1) AS TEXT), "
            "updated_at = ? WHERE key = ?",
            (time.time(), _GENERATION_KEY),
        )
        return self.generation

    @property
    def generation(self) -> int:
        """Return the document generation counter.

        Increments on every mutation; index snapshots record the generation
        they were taken at so a stale snapshot is never trusted.

        Performance:
            Time: O(1)
        """
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT value_json FROM kv WHERE key = ?", (_GENERATION_KEY,)
        ).fetchone()
        return int(row["value_json"]) if row else 0

    @property
    def schema_version(self) -> int:
        """Return the on-disk schema version.

        Performance:
            Time: O(1)
        """
        return self._schema_version

    @property
    def path(self) -> str:
        """Return the vault path.

        Performance:
            Time: O(1)
        """
        return self._path

    # ─── Documents: write ───

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
        docs = self.insert_docs(
            texts=[text],
            embeddings=np.asarray(embedding, dtype=np.float32).reshape(1, -1),
            metadatas=[metadata],
            namespace=namespace,
            modality=modality,
            doc_ids=[doc_id] if doc_id is not None else None,
        )
        return docs[0]

    def insert_docs(
        self,
        texts: Sequence[str],
        embeddings: NDArray[np.float32],
        metadatas: Sequence[dict[str, Any]] | None = None,
        namespace: str = "default",
        modality: str = "text",
        doc_ids: Sequence[str | None] | None = None,
    ) -> list[StoredDocument]:
        """Insert many documents in a single transaction.

        Args:
            texts: Document texts.
            embeddings: Array of shape ``(len(texts), dimension)``.
            metadatas: One metadata dict per text (None = empty dicts).
            namespace: Target namespace.
            modality: Content type for every document.
            doc_ids: Optional custom IDs (None entries are auto-generated).

        Returns:
            List of StoredDocument in input order.

        Raises:
            sqlite3.IntegrityError: If any doc_id already exists (nothing
                is inserted in that case).

        Performance:
            Time: O(B) in one transaction — ~50x faster than B commits
        """
        assert self._conn is not None
        if not texts:
            return []

        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        if emb.shape[0] != len(texts):
            raise ValueError(
                f"embeddings rows ({emb.shape[0]}) must match texts ({len(texts)}).\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            )
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if len(metadatas) != len(texts):
            raise ValueError(
                f"metadatas length ({len(metadatas)}) must match "
                f"texts ({len(texts)}).\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            )

        now = time.time()
        stored: list[StoredDocument] = []
        rows: list[tuple[Any, ...]] = []
        for i, text in enumerate(texts):
            did = doc_ids[i] if doc_ids is not None and doc_ids[i] else None
            did = did or str(uuid.uuid4())
            meta = metadatas[i] if metadatas[i] is not None else {}
            chash = content_hash(namespace, text)
            rows.append(
                (
                    did,
                    text,
                    np.ascontiguousarray(emb[i]).tobytes(),
                    json.dumps(meta, default=str),
                    namespace,
                    modality,
                    now,
                    chash,
                )
            )
            stored.append(
                StoredDocument(
                    doc_id=did,
                    text=text,
                    embedding=emb[i],
                    metadata=meta,
                    namespace=namespace,
                    modality=modality,
                    created_at=now,
                    content_hash=chash,
                )
            )

        with self._tx():
            self._conn.executemany(
                """
                INSERT INTO documents (doc_id, text, embedding, metadata_json,
                                       namespace, modality, created_at, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._bump_generation()
        return stored

    def update_doc(
        self,
        doc_id: str,
        text: str | None = None,
        embedding: NDArray[np.float32] | None = None,
        metadata: dict[str, Any] | None = None,
        modality: str | None = None,
    ) -> bool:
        """Update fields of an existing document.

        Args:
            doc_id: Document identifier.
            text: New text (also refreshes ``content_hash``).
            embedding: New embedding vector.
            metadata: Replacement metadata dictionary.
            modality: New modality.

        Returns:
            True if a row was updated.

        Performance:
            Time: O(log N)
        """
        assert self._conn is not None
        sets: list[str] = []
        params: list[Any] = []
        if text is not None:
            row = self._conn.execute(
                "SELECT namespace FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if row is None:
                return False
            sets.append("text = ?")
            params.append(text)
            sets.append("content_hash = ?")
            params.append(content_hash(row["namespace"], text))
        if embedding is not None:
            sets.append("embedding = ?")
            params.append(np.ascontiguousarray(embedding, dtype=np.float32).tobytes())
        if metadata is not None:
            sets.append("metadata_json = ?")
            params.append(json.dumps(metadata, default=str))
        if modality is not None:
            sets.append("modality = ?")
            params.append(modality)
        if not sets:
            return False
        sets.append("updated_at = ?")
        params.append(time.time())
        params.append(doc_id)
        with self._tx():
            cursor = self._conn.execute(
                f"UPDATE documents SET {', '.join(sets)} WHERE doc_id = ?", params
            )
            # why: metadata/modality are read from SQLite at query time, so
            # only text/vector changes invalidate the index snapshot
            if cursor.rowcount > 0 and (text is not None or embedding is not None):
                self._bump_generation()
        return cursor.rowcount > 0

    def delete_doc(self, doc_id: str) -> bool:
        """Delete a document by ID.

        Args:
            doc_id: Document identifier.

        Returns:
            True if document was deleted, False if not found.

        Performance:
            Time: O(log N)
        """
        return self.delete_docs([doc_id]) > 0

    def delete_docs(self, doc_ids: Sequence[str]) -> int:
        """Delete many documents in one transaction.

        Args:
            doc_ids: Document identifiers.

        Returns:
            Number of rows deleted.

        Performance:
            Time: O(K log N)
        """
        assert self._conn is not None
        if not doc_ids:
            return 0
        deleted = 0
        with self._tx():
            for i in range(0, len(doc_ids), _IN_CHUNK):
                chunk = list(doc_ids[i : i + _IN_CHUNK])
                placeholders = ",".join("?" for _ in chunk)
                cursor = self._conn.execute(
                    f"DELETE FROM documents WHERE doc_id IN ({placeholders})", chunk
                )
                deleted += cursor.rowcount
            if deleted:
                self._bump_generation()
        return deleted

    def delete_namespace_docs(self, namespace: str) -> int:
        """Delete every document in a namespace.

        Args:
            namespace: Namespace to clear.

        Returns:
            Number of rows deleted.

        Performance:
            Time: O(K log N)
        """
        assert self._conn is not None
        with self._tx():
            cursor = self._conn.execute(
                "DELETE FROM documents WHERE namespace = ?", (namespace,)
            )
            if cursor.rowcount:
                self._bump_generation()
        return int(cursor.rowcount)

    # ─── Documents: read ───

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

    def get_docs(
        self, doc_ids: Sequence[str], include_embedding: bool = True
    ) -> dict[str, StoredDocument]:
        """Retrieve many documents by ID in a few batched queries.

        Args:
            doc_ids: Document identifiers.
            include_embedding: If False, skip reading the embedding blobs
                (search hydration does not need them); ``embedding`` is
                then an empty array.

        Returns:
            Mapping of doc_id → StoredDocument for the IDs that exist.

        Performance:
            Time: O(K log N) with ``K / 500`` round trips
        """
        assert self._conn is not None
        out: dict[str, StoredDocument] = {}
        ids = list(dict.fromkeys(doc_ids))
        columns = (
            "*"
            if include_embedding
            else "doc_id, text, metadata_json, namespace, modality, created_at, "
            "updated_at, content_hash"
        )
        for i in range(0, len(ids), _IN_CHUNK):
            chunk = ids[i : i + _IN_CHUNK]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                f"SELECT {columns} FROM documents WHERE doc_id IN ({placeholders})",
                chunk,
            ).fetchall()
            for row in rows:
                doc = self._row_to_document(row)
                out[doc.doc_id] = doc
        return out

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
            ORDER BY rowid ASC
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
            List of all matching StoredDocument in insertion order.

        Performance:
            Time: O(N) — full table scan
        """
        return list(self.iter_docs(namespace=namespace))

    def iter_docs(
        self, namespace: str | None = None, batch_size: int = 2000
    ) -> Iterator[StoredDocument]:
        """Stream documents in insertion order without loading all at once.

        Args:
            namespace: Optional namespace filter.
            batch_size: Rows fetched per round trip.

        Yields:
            StoredDocument objects.

        Performance:
            Time: O(N); memory O(batch_size)
        """
        assert self._conn is not None
        if namespace is not None:
            cursor = self._conn.execute(
                "SELECT * FROM documents WHERE namespace = ? ORDER BY rowid",
                (namespace,),
            )
        else:
            cursor = self._conn.execute("SELECT * FROM documents ORDER BY rowid")
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            for row in rows:
                yield self._row_to_document(row)

    def iter_embeddings(
        self, batch_size: int = 5000
    ) -> Iterator[tuple[str, NDArray[np.float32]]]:
        """Stream ``(doc_id, embedding)`` pairs in insertion order.

        Reads only the two columns the index restore needs — several times
        faster than :meth:`iter_docs` for large vaults.

        Args:
            batch_size: Rows fetched per round trip.

        Yields:
            ``(doc_id, embedding)`` tuples.

        Performance:
            Time: O(N); memory O(batch_size · d)
        """
        assert self._conn is not None
        cursor = self._conn.execute(
            "SELECT doc_id, embedding FROM documents ORDER BY rowid"
        )
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            for row in rows:
                yield row[0], np.frombuffer(row[1], dtype=np.float32)

    def find_by_content_hash(self, namespace: str, chash: str) -> StoredDocument | None:
        """Find a document by its content hash within a namespace.

        Args:
            namespace: Namespace to search.
            chash: Hash from :func:`content_hash`.

        Returns:
            The first matching StoredDocument, or None.

        Performance:
            Time: O(log N)
        """
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT * FROM documents WHERE namespace = ? AND content_hash = ? "
            "ORDER BY rowid LIMIT 1",
            (namespace, chash),
        ).fetchone()
        return self._row_to_document(row) if row is not None else None

    def find_all_by_content_hash(
        self, namespace: str, chash: str
    ) -> list[StoredDocument]:
        """Return every document in a namespace with this content hash.

        Performance:
            Time: O(log N + matches)
        """
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM documents WHERE namespace = ? AND content_hash = ? "
            "ORDER BY rowid",
            (namespace, chash),
        ).fetchall()
        return [self._row_to_document(r) for r in rows]

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

    def count_by_namespace(self) -> dict[str, int]:
        """Count documents grouped by namespace.

        Returns:
            Mapping namespace → document count (namespaces with zero
            documents are included).

        Performance:
            Time: O(K) with the namespace index
        """
        assert self._conn is not None
        counts = {name: 0 for name in self.list_namespaces()}
        rows = self._conn.execute(
            "SELECT namespace, COUNT(*) AS c FROM documents GROUP BY namespace"
        ).fetchall()
        for row in rows:
            counts[row["namespace"]] = int(row["c"])
        return counts

    # ─── Namespaces ───

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

    def delete_namespace(self, name: str) -> int:
        """Delete a namespace and every document in it.

        Args:
            name: Namespace name. The ``default`` namespace cannot be
                deleted (its documents are removed but the row stays).

        Returns:
            Number of documents deleted.

        Performance:
            Time: O(K log N)
        """
        assert self._conn is not None
        deleted = self.delete_namespace_docs(name)
        if name != "default":
            self._conn.execute("DELETE FROM namespaces WHERE name = ?", (name,))
            self._conn.commit()
        return deleted

    # ─── Index persistence (legacy v1 API kept for compatibility) ───

    def save_faiss_index(self, index_data: bytes, dimension: int, count: int) -> None:
        """Persist a raw FAISS index blob (legacy v1 table).

        Prefer :meth:`save_index_state`, which also stores the BM25 index
        and the position map.

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
        """Load the legacy raw FAISS index blob.

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

    def _write_chunks(self, kind: str, data: bytes) -> int:
        assert self._conn is not None
        self._conn.execute("DELETE FROM index_chunks WHERE kind = ?", (kind,))
        view = memoryview(data)
        rows = [
            (kind, seq, view[offset : offset + _BLOB_CHUNK_BYTES])
            for seq, offset in enumerate(range(0, len(data), _BLOB_CHUNK_BYTES))
        ]
        if rows:
            self._conn.executemany(
                "INSERT INTO index_chunks (kind, seq, data) VALUES (?, ?, ?)", rows
            )
        return len(rows)

    def _read_chunks(self, kind: str) -> bytes:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT data FROM index_chunks WHERE kind = ? ORDER BY seq", (kind,)
        ).fetchall()
        return b"".join(bytes(r["data"]) for r in rows)

    def save_index_state(self, state: IndexState) -> None:
        """Persist a complete in-memory index snapshot.

        Large blobs (FAISS, BM25) are written in 64 MB chunks so vaults
        with millions of vectors stay below SQLite's single-BLOB limit.

        Args:
            state: Snapshot to store (replaces any previous one).

        Performance:
            Time: O(size of blobs)
        """
        assert self._conn is not None
        with self._tx():
            self._write_chunks("faiss", state.faiss_blob)
            self._write_chunks("bm25", state.bm25_blob)
            self._conn.execute(
                """
                INSERT OR REPLACE INTO index_state
                    (id, faiss_blob, bm25_blob, doc_ids, doc_namespaces, tombstones,
                     dimension, generation, doc_count, model_name, index_type,
                     updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    b"",
                    b"",
                    "\n".join(state.doc_ids).encode("utf-8"),
                    "\n".join(state.doc_namespaces).encode("utf-8"),
                    json.dumps(sorted(int(t) for t in state.tombstones)),
                    int(state.dimension),
                    int(state.generation),
                    int(state.doc_count),
                    state.model_name,
                    state.index_type,
                    time.time(),
                ),
            )

    def load_index_state(self) -> IndexState | None:
        """Load the persisted index snapshot, if any.

        Returns:
            IndexState or None.

        Performance:
            Time: O(size of blobs)
        """
        assert self._conn is not None
        row = self._conn.execute("SELECT * FROM index_state WHERE id = 1").fetchone()
        if row is None:
            return None
        raw_ids = bytes(row["doc_ids"]).decode("utf-8")
        doc_ids = raw_ids.split("\n") if raw_ids else []
        raw_ns = bytes(row["doc_namespaces"]).decode("utf-8")
        doc_namespaces = raw_ns.split("\n") if raw_ns else []
        faiss_blob = bytes(row["faiss_blob"]) or self._read_chunks("faiss")
        bm25_blob = bytes(row["bm25_blob"]) or self._read_chunks("bm25")
        return IndexState(
            faiss_blob=faiss_blob,
            bm25_blob=bm25_blob,
            doc_ids=doc_ids,
            doc_namespaces=doc_namespaces,
            tombstones=[int(t) for t in json.loads(row["tombstones"])],
            dimension=int(row["dimension"]),
            generation=int(row["generation"]),
            doc_count=int(row["doc_count"]),
            model_name=str(row["model_name"]),
            index_type=str(row["index_type"]),
            updated_at=float(row["updated_at"]),
        )

    def clear_index_state(self) -> None:
        """Drop the persisted index snapshot.

        Performance:
            Time: O(1)
        """
        assert self._conn is not None
        with self._tx():
            self._conn.execute("DELETE FROM index_state")
            self._conn.execute("DELETE FROM index_chunks")

    # ─── Sessions (memory layer) ───

    def upsert_session(self, session: SessionRecord) -> None:
        """Create or update a session record.

        Args:
            session: Session to persist.

        Performance:
            Time: O(log S)
        """
        assert self._conn is not None
        self._conn.execute(
            """
            INSERT INTO sessions (session_id, agent, started_at, ended_at, summary,
                                  metadata_json, turn_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                agent = excluded.agent,
                started_at = excluded.started_at,
                ended_at = excluded.ended_at,
                summary = excluded.summary,
                metadata_json = excluded.metadata_json,
                turn_count = excluded.turn_count
            """,
            (
                session.session_id,
                session.agent,
                session.started_at,
                session.ended_at,
                session.summary,
                json.dumps(session.metadata, default=str),
                session.turn_count,
            ),
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Fetch a session by ID.

        Performance:
            Time: O(log S)
        """
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return self._row_to_session(row) if row is not None else None

    def list_sessions(
        self, agent: str | None = None, limit: int = 20
    ) -> list[SessionRecord]:
        """List sessions, newest first.

        Args:
            agent: Optional agent filter.
            limit: Maximum sessions to return.

        Performance:
            Time: O(limit) with the (agent, started_at) index
        """
        assert self._conn is not None
        if agent is not None:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE agent = ? "
                "ORDER BY started_at DESC LIMIT ?",
                (agent, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_session(r) for r in rows]

    def increment_session_turns(self, session_id: str, by: int = 1) -> None:
        """Add to a session's turn counter.

        Performance:
            Time: O(log S)
        """
        assert self._conn is not None
        self._conn.execute(
            "UPDATE sessions SET turn_count = turn_count + ? WHERE session_id = ?",
            (by, session_id),
        )
        self._conn.commit()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session record (memories are untouched).

        Performance:
            Time: O(log S)
        """
        assert self._conn is not None
        cursor = self._conn.execute(
            "DELETE FROM sessions WHERE session_id = ?", (session_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # ─── Key/value store ───

    def kv_set(self, key: str, value: Any, _internal: bool = False) -> None:
        """Store a JSON-serialisable value under a key.

        Keys starting with ``__`` are reserved for VecForge.

        Performance:
            Time: O(log K)
        """
        assert self._conn is not None
        if key == _GENERATION_KEY or (
            key.startswith(_RESERVED_PREFIX) and not _internal
        ):
            raise ValueError(
                f"Key '{key}' is reserved.\nVecForge by Suneel Bose K · ArcGX TechLabs"
            )
        self._conn.execute(
            "INSERT OR REPLACE INTO kv (key, value_json, updated_at) VALUES (?, ?, ?)",
            (key, json.dumps(value, default=str), time.time()),
        )
        self._conn.commit()

    def kv_get(self, key: str, default: Any = None) -> Any:
        """Fetch a value by key.

        Performance:
            Time: O(log K)
        """
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT value_json FROM kv WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row["value_json"]) if row is not None else default

    def kv_delete(self, key: str) -> bool:
        """Delete a key.

        Performance:
            Time: O(log K)
        """
        assert self._conn is not None
        if key.startswith(_RESERVED_PREFIX):
            return False
        cursor = self._conn.execute("DELETE FROM kv WHERE key = ?", (key,))
        self._conn.commit()
        return cursor.rowcount > 0

    def kv_keys(self, prefix: str = "") -> list[str]:
        """List keys with an optional prefix.

        Performance:
            Time: O(K)
        """
        assert self._conn is not None
        # why: exact, case-sensitive prefix match (LIKE treats _ and % as
        # wildcards and ignores case)
        rows = self._conn.execute(
            "SELECT key FROM kv WHERE substr(key, 1, ?) = ? ORDER BY key",
            (len(prefix), prefix),
        ).fetchall()
        return [r["key"] for r in rows if not r["key"].startswith(_RESERVED_PREFIX)]

    # ─── Maintenance ───

    def backup(self, dest_path: str) -> str:
        """Write a consistent copy of the vault using SQLite's backup API.

        Unlike copying the file, this is safe while the vault is open and
        in WAL mode: it produces a single self-contained database file.

        Args:
            dest_path: Destination file path (overwritten).

        Returns:
            The destination path.

        Performance:
            Time: O(file size)
        """
        assert self._conn is not None
        if self._encrypted:
            # security: export through SQLCipher so the copy is encrypted
            # with the same key (a plain sqlite3 target would be plaintext)
            if Path(dest_path).exists():
                Path(dest_path).unlink()
            self._conn.execute(
                "ATTACH DATABASE ? AS vf_backup KEY ?",
                (dest_path, self._encryption_key),
            )
            try:
                self._conn.execute("SELECT sqlcipher_export('vf_backup')")
            finally:
                self._conn.execute("DETACH DATABASE vf_backup")
        else:
            dest = sqlite3.connect(dest_path)
            try:
                self._conn.backup(dest)
            finally:
                dest.close()
        logger.info("Vault backed up to %s", dest_path)
        return dest_path

    def vacuum(self) -> None:
        """Reclaim disk space after deletions.

        Performance:
            Time: O(file size)
        """
        assert self._conn is not None
        self._conn.commit()
        self._conn.execute("VACUUM")

    def _row_to_document(self, row: sqlite3.Row) -> StoredDocument:
        """Convert a database row to a StoredDocument.

        Performance:
            Time: O(d) where d = embedding dimension
        """
        keys = row.keys()
        if "embedding" in keys:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
        else:
            embedding = np.zeros(0, dtype=np.float32)
        return StoredDocument(
            doc_id=row["doc_id"],
            text=row["text"],
            embedding=embedding,
            metadata=json.loads(row["metadata_json"]),
            namespace=row["namespace"],
            modality=row["modality"],
            created_at=row["created_at"],
            updated_at=row["updated_at"] if "updated_at" in keys else None,
            content_hash=row["content_hash"] if "content_hash" in keys else "",
        )

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> SessionRecord:
        return SessionRecord(
            session_id=row["session_id"],
            agent=row["agent"],
            started_at=float(row["started_at"]),
            ended_at=float(row["ended_at"]) if row["ended_at"] is not None else None,
            summary=row["summary"] or "",
            metadata=json.loads(row["metadata_json"] or "{}"),
            turn_count=int(row["turn_count"]),
        )

    @property
    def is_encrypted(self) -> bool:
        """Return whether the storage backend is using encryption.

        Performance:
            Time: O(1)
        """
        return self._encrypted

    @property
    def is_open(self) -> bool:
        """Return whether the connection is open.

        Performance:
            Time: O(1)
        """
        return self._conn is not None

    def close(self) -> None:
        """Close the database connection.

        Performance:
            Time: O(1)
        """
        if self._conn:
            with contextlib.suppress(self._db.Error):
                self._conn.commit()
            self._conn.close()
            self._conn = None
