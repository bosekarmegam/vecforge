# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
MemoryStore — long-term memory for AI assistants, built on VecForge.

Design:

* Every memory is a VecForge document in the namespace ``memory:<agent>``
  with structured metadata (kind, importance, tags, session, access
  statistics). The vault's hybrid search, encryption, audit log and
  snapshots apply unchanged.
* ``remember()`` de-duplicates: an exact or near-duplicate (cosine ≥
  ``dedup_threshold``) of an existing memory reinforces that memory
  instead of inserting a copy, so the store stays compact over hundreds
  of sessions.
* ``recall()`` blends relevance, importance and recency and records each
  access, so frequently useful memories rise and stale ones fade.
* Sessions give an assistant a notion of "last time": ``start_session``
  / ``end_session(summary)`` plus ``context_pack()`` produce a compact,
  prompt-ready briefing at the start of the next conversation.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vecforge.core.embedder import EmbedderLike
from vecforge.core.storage import SessionRecord, content_hash
from vecforge.core.vault import SearchResult, VecForge
from vecforge.exceptions import (
    DocumentNotFoundError,
    VaultClosedError,
    VaultEmptyError,
)

logger = logging.getLogger(__name__)

KINDS: frozenset[str] = frozenset(
    {"fact", "preference", "episode", "task", "summary", "note", "insight", "decision"}
)
_DEFAULT_KIND = "note"
_DEFAULT_PATH = "~/.vecforge/memory.db"
_RECENCY_HALF_LIFE_DAYS = 30.0
_KIND_ORDER = {
    "preference": 0,
    "decision": 1,
    "fact": 2,
    "insight": 3,
    "task": 4,
    "summary": 5,
    "episode": 6,
    "note": 7,
}
_KIND_LABELS = {
    "preference": "Preferences",
    "decision": "Decisions",
    "fact": "Facts",
    "insight": "Insights",
    "task": "Tasks",
    "summary": "Summaries",
    "episode": "Episodes",
    "note": "Notes",
}


@dataclass
class Memory:
    """A single remembered item.

    Attributes:
        memory_id: Unique identifier (``mem_…``).
        content: The remembered text.
        kind: One of :data:`KINDS` (fact, preference, episode, task,
            summary, note, insight, decision).
        importance: 0.0–1.0 weight used during recall.
        tags: Free-form tags.
        created_at: Unix timestamp.
        updated_at: Unix timestamp of the last edit, if any.
        last_accessed: Unix timestamp of the last recall hit, if any.
        access_count: Number of times recall returned this memory.
        session_id: Session that produced the memory, if any.
        source: Where it came from (``"user"``, ``"assistant"``, a URL…).
        agent: Agent the memory belongs to.
        score: Recall score (0 when not produced by ``recall``).
        metadata: Extra user metadata.
        deduplicated: True when ``remember()`` merged into this existing
            memory instead of creating a new one.
    """

    memory_id: str
    content: str
    kind: str = _DEFAULT_KIND
    importance: float = 0.5
    tags: list[str] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float | None = None
    last_accessed: float | None = None
    access_count: int = 0
    session_id: str | None = None
    source: str | None = None
    agent: str = "default"
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    deduplicated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return asdict(self)

    @property
    def age_days(self) -> float:
        """Days since the memory was created."""
        return max(0.0, (time.time() - self.created_at) / 86400.0)


@dataclass
class Session:
    """A conversation / agent run tracked by the memory store.

    Attributes:
        session_id: Unique identifier.
        agent: Owning agent.
        started_at: Unix timestamp.
        ended_at: Unix timestamp, or None while open.
        summary: Summary written by ``end_session``.
        metadata: Arbitrary metadata.
        turn_count: Memories written during the session.
    """

    session_id: str
    agent: str = "default"
    started_at: float = 0.0
    ended_at: float | None = None
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    turn_count: int = 0

    @property
    def is_open(self) -> bool:
        """Whether the session has not been ended."""
        return self.ended_at is None

    @property
    def duration_seconds(self) -> float:
        """Seconds between start and end (or now while open)."""
        end = self.ended_at if self.ended_at is not None else time.time()
        return max(0.0, end - self.started_at)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        d = asdict(self)
        d["is_open"] = self.is_open
        return d

    @classmethod
    def _from_record(cls, rec: SessionRecord) -> Session:
        return cls(
            session_id=rec.session_id,
            agent=rec.agent,
            started_at=rec.started_at,
            ended_at=rec.ended_at,
            summary=rec.summary,
            metadata=dict(rec.metadata),
            turn_count=rec.turn_count,
        )

    def _to_record(self) -> SessionRecord:
        return SessionRecord(
            session_id=self.session_id,
            agent=self.agent,
            started_at=self.started_at,
            ended_at=self.ended_at,
            summary=self.summary,
            metadata=dict(self.metadata),
            turn_count=self.turn_count,
        )


class MemoryStore:
    """Persistent memory for AI assistants (Claude, GPT, local models).

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        path: Vault file. Defaults to ``~/.vecforge/memory.db`` (created on
            demand). ``":memory:"`` gives a throw-away store.
        agent: Agent name; memories are isolated per agent in the
            namespace ``memory:<agent>``.
        model_name: Embedding model (see :class:`~vecforge.VecForge`).
            ``"hash"`` needs no model download.
        embedder: Pre-built embedder (overrides ``model_name``).
        encryption_key: Optional SQLCipher key.
        dedup_threshold: Cosine similarity above which ``remember()``
            treats new content as a duplicate of an existing memory of the
            same kind. Defaults to 0.92.
        auto_save_every: Index snapshot cadence in mutations (default 25).
        quantum: Use Grover reranking during recall. Defaults to False.
        vault: Reuse an existing :class:`~vecforge.VecForge` instead of
            opening ``path``.

    Performance:
        remember: one embedding + O(1) storage (+ one dedup search)
        recall:   one hybrid search + O(k) rescoring

    Example::

        >>> mem = MemoryStore(":memory:", model_name="hash")
        >>> m = mem.remember("The user's name is Suneel", kind="fact")
        >>> mem.recall("what is the user's name?")[0].content
        "The user's name is Suneel"
    """

    def __init__(
        self,
        path: str | None = None,
        agent: str = "default",
        model_name: str | None = None,
        embedder: EmbedderLike | None = None,
        encryption_key: str | None = None,
        dedup_threshold: float = 0.92,
        auto_save_every: int = 25,
        quantum: bool = False,
        vault: VecForge | None = None,
    ) -> None:
        self._agent = agent or "default"
        self._namespace = f"memory:{self._agent}"
        self._dedup_threshold = float(dedup_threshold)
        self._current_session: Session | None = None

        if vault is not None:
            self._vault = vault
            self._owns_vault = False
        else:
            resolved = self._resolve_path(path)
            self._vault = VecForge(
                resolved,
                model_name=model_name,
                embedder=embedder,
                encryption_key=encryption_key,
                auto_save_every=auto_save_every,
                quantum=quantum,
            )
            self._owns_vault = True

        # why: every store on a vault (REST, MCP, several agents) must
        # serialise with the vault itself, which owns the SQLite connection
        self._lock = self._vault.lock

        if not self._vault.exists_namespace(self._namespace):
            self._vault.create_namespace(self._namespace)

    @staticmethod
    def _resolve_path(path: str | None) -> str:
        if path is None or path == "":
            path = _DEFAULT_PATH
        if path == ":memory:":
            return path
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

    # ─── Properties ───

    @property
    def agent(self) -> str:
        """Agent name this store belongs to."""
        return self._agent

    @property
    def namespace(self) -> str:
        """Vault namespace holding this agent's memories."""
        return self._namespace

    @property
    def vault(self) -> VecForge:
        """Underlying VecForge vault (shared with other namespaces)."""
        return self._vault

    @property
    def current_session(self) -> Session | None:
        """The session started by :meth:`start_session`, if any."""
        return self._current_session

    @contextmanager
    def _guard(self) -> Iterator[None]:
        """Hold the vault lock and fail cleanly if the vault is closed."""
        with self._lock:
            if self._vault.closed:
                raise VaultClosedError(self._vault.path)
            yield

    # ─── Conversions ───

    def _to_memory(self, r: SearchResult, score: float | None = None) -> Memory:
        meta = r.metadata or {}
        return Memory(
            memory_id=r.doc_id,
            content=r.text,
            kind=str(meta.get("kind", _DEFAULT_KIND)),
            importance=float(meta.get("importance", 0.5)),
            tags=list(meta.get("tags", [])),
            created_at=float(meta.get("created_at", r.timestamp)),
            updated_at=meta.get("edited_at"),
            last_accessed=meta.get("last_accessed"),
            access_count=int(meta.get("access_count", 0)),
            session_id=meta.get("session_id"),
            source=meta.get("source"),
            agent=str(meta.get("agent", self._agent)),
            score=float(score if score is not None else r.score),
            metadata=dict(meta.get("extra", {})),
        )

    @staticmethod
    def _normalise_kind(kind: str | None) -> str:
        k = (kind or _DEFAULT_KIND).strip().lower()
        return k if k in KINDS else _DEFAULT_KIND

    @staticmethod
    def _clamp(x: float) -> float:
        return float(min(max(x, 0.0), 1.0))

    # ─── Write ───

    def remember(
        self,
        content: str,
        kind: str = "fact",
        importance: float = 0.5,
        tags: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        source: str | None = None,
        dedup: bool = True,
        created_at: float | None = None,
        _count_turn: bool = True,
    ) -> Memory:
        """Store a memory (or reinforce an existing near-duplicate).

        Args:
            content: Text to remember.
            kind: Memory kind (fact, preference, episode, task, summary,
                note, insight, decision). Unknown kinds become ``note``.
            importance: 0.0–1.0 weight for recall.
            tags: Optional tags.
            metadata: Extra metadata stored with the memory.
            session_id: Session to attribute the memory to. Defaults to
                the current session, if one was started.
            source: Origin label (``"user"``, ``"assistant"``, a URL…).
            dedup: Merge into an existing duplicate instead of inserting.
            created_at: Original creation time (used when importing).

        Returns:
            The stored (or reinforced) :class:`Memory`.

        Performance:
            Time: one embedding + O(1) insert (+ one search when dedup)

        Example::

            >>> mem.remember("Deploys happen on Fridays", kind="fact", tags=["ops"])
        """
        content = (content or "").strip()
        if not content:
            raise ValueError(
                "Memory content must not be empty.\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            )
        with self._guard():
            kind = self._normalise_kind(kind)
            importance = self._clamp(importance)
            tag_list = sorted({t.strip() for t in (tags or []) if t and t.strip()})
            if session_id is None and self._current_session is not None:
                session_id = self._current_session.session_id
            if session_id is not None:
                self._require_own_session(session_id, create=True)

            if dedup:
                existing = self._find_duplicate(content, kind)
                if existing is not None:
                    merged = self._reinforce(
                        existing,
                        delta=0.05,
                        extra_tags=tag_list,
                        importance_floor=importance,
                    )
                    merged.deduplicated = True
                    logger.debug("Memory deduplicated into %s", merged.memory_id)
                    return merged

            now = time.time() if created_at is None else float(created_at)
            meta: dict[str, Any] = {
                "kind": kind,
                "importance": importance,
                "tags": tag_list,
                "created_at": now,
                "last_accessed": None,
                "access_count": 0,
                "session_id": session_id,
                "source": source,
                "agent": self._agent,
                "extra": dict(metadata or {}),
            }
            memory_id = "mem_" + uuid.uuid4().hex[:16]
            self._vault.add(
                content,
                metadata=meta,
                namespace=self._namespace,
                doc_id=memory_id,
                modality="memory",
            )
            if session_id and _count_turn:
                self._bump_session(session_id)
            return Memory(
                memory_id=memory_id,
                content=content,
                kind=kind,
                importance=importance,
                tags=tag_list,
                created_at=now,
                session_id=session_id,
                source=source,
                agent=self._agent,
                metadata=dict(metadata or {}),
            )

    def remember_many(
        self, items: Sequence[dict[str, Any] | str], **defaults: Any
    ) -> list[Memory]:
        """Store several memories.

        Args:
            items: Strings or dicts with :meth:`remember` keyword arguments.
            **defaults: Defaults applied to every item.

        Returns:
            Stored memories in input order.
        """
        out: list[Memory] = []
        for item in items:
            if isinstance(item, str):
                out.append(self.remember(item, **defaults))
            else:
                kwargs = {**defaults, **item}
                content = kwargs.pop("content")
                out.append(self.remember(content, **kwargs))
        return out

    def _find_duplicate(self, content: str, kind: str) -> Memory | None:
        storage = self._vault._storage
        for exact in storage.find_all_by_content_hash(
            self._namespace, content_hash(self._namespace, content)
        ):
            if exact.metadata.get("kind") == kind:
                result = self._vault.get(exact.doc_id)
                if result is not None:
                    return self._to_memory(result, score=1.0)

        if self._dedup_threshold >= 1.0 or self._vault.count(self._namespace) == 0:
            return None

        query_vec = self._vault.embedder.embed(content)
        try:
            hits = self._vault.search(
                content,
                top_k=3,
                alpha=1.0,
                namespace=self._namespace,
                filters={"kind": kind},
                query_vector=query_vec,
            )
        except VaultEmptyError:
            return None
        for hit in hits:
            vec = self._vault.get_embedding(hit.doc_id)
            if vec is None:
                continue
            cosine = float(np.dot(vec, query_vec))
            if cosine >= self._dedup_threshold:
                return self._to_memory(hit, score=cosine)
        return None

    def _reinforce(
        self,
        memory: Memory,
        delta: float,
        extra_tags: Iterable[str] = (),
        importance_floor: float | None = None,
    ) -> Memory:
        new_importance = self._clamp(memory.importance + delta)
        if importance_floor is not None:
            new_importance = max(new_importance, self._clamp(importance_floor))
        tags = sorted(set(memory.tags) | {t for t in extra_tags if t})
        now = time.time()
        self._vault.update_metadata(
            memory.memory_id,
            {
                "importance": new_importance,
                "tags": tags,
                "access_count": memory.access_count + 1,
                "last_accessed": now,
            },
        )
        memory.importance = new_importance
        memory.tags = tags
        memory.access_count += 1
        memory.last_accessed = now
        return memory

    def reinforce(self, memory_id: str, delta: float = 0.1) -> Memory:
        """Increase a memory's importance and access count.

        Args:
            memory_id: Memory to reinforce.
            delta: Importance increment (clamped to 1.0).

        Returns:
            The updated memory.

        Raises:
            DocumentNotFoundError: If the memory does not exist.
        """
        with self._guard():
            memory = self.get(memory_id)
            if memory is None:
                raise DocumentNotFoundError(memory_id)
            return self._reinforce(memory, delta)

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        tags: Iterable[str] | None = None,
        kind: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Edit a memory. Changing ``content`` re-embeds it.

        Args:
            memory_id: Memory to edit.
            content: New text.
            importance: New importance (0–1).
            tags: Replacement tag list.
            kind: New kind.
            metadata: Replacement extra metadata.

        Returns:
            The updated memory.

        Raises:
            DocumentNotFoundError: If the memory does not exist.
        """
        with self._guard():
            current = self._vault.get(memory_id)
            if current is None or current.namespace != self._namespace:
                raise DocumentNotFoundError(memory_id)
            meta = dict(current.metadata)
            meta["edited_at"] = time.time()
            if importance is not None:
                meta["importance"] = self._clamp(importance)
            if tags is not None:
                meta["tags"] = sorted({t.strip() for t in tags if t and t.strip()})
            if kind is not None:
                meta["kind"] = self._normalise_kind(kind)
            if metadata is not None:
                meta["extra"] = dict(metadata)
            if content is not None and content.strip() and content != current.text:
                self._vault.upsert(
                    content.strip(),
                    doc_id=memory_id,
                    metadata=meta,
                    namespace=self._namespace,
                    modality="memory",
                )
            else:
                self._vault.update_metadata(memory_id, meta, merge=False)
            refreshed = self._vault.get(memory_id)
            assert refreshed is not None
            return self._to_memory(refreshed, score=0.0)

    def forget(self, memory_id: str) -> bool:
        """Delete a memory.

        Args:
            memory_id: Memory to delete.

        Returns:
            True if it existed.
        """
        with self._guard():
            if self.get(memory_id) is None:
                return False
            return self._vault.delete(memory_id)

    def forget_many(self, memory_ids: Sequence[str]) -> int:
        """Delete several memories.

        Returns:
            Number deleted.
        """
        with self._guard():
            own = [m for m in dict.fromkeys(memory_ids) if self.get(m) is not None]
            return self._vault.delete_many(own) if own else 0

    def clear(self, kind: str | None = None) -> int:
        """Delete all memories (optionally only one kind).

        Args:
            kind: Restrict to this kind.

        Returns:
            Number deleted.
        """
        with self._guard():
            ids = [
                m.memory_id
                for m in self.list_memories(kinds=[kind] if kind else None, limit=0)
            ]
            return self._vault.delete_many(ids) if ids else 0

    # ─── Read ───

    def get(self, memory_id: str) -> Memory | None:
        """Fetch a memory by ID.

        Returns:
            The memory, or None.
        """
        with self._guard():
            r = self._vault.get(memory_id)
            if r is None or r.namespace != self._namespace:
                return None
            return self._to_memory(r, score=0.0)

    def list_memories(
        self,
        kinds: Sequence[str] | None = None,
        session_id: str | None = None,
        tags: Sequence[str] | None = None,
        limit: int = 50,
        newest_first: bool = True,
    ) -> list[Memory]:
        """List memories without a query.

        Args:
            kinds: Restrict to these kinds.
            session_id: Restrict to one session.
            tags: Require at least one of these tags.
            limit: Maximum returned (0 = all).
            newest_first: Sort order.

        Returns:
            Memories sorted by creation time.

        Performance:
            Time: O(M) over this agent's memories
        """
        with self._guard():
            kind_set = {self._normalise_kind(k) for k in kinds} if kinds else None
            tag_set = set(tags) if tags else None
            out: list[Memory] = []
            for doc in self._vault._storage.iter_docs(namespace=self._namespace):
                meta = doc.metadata
                if kind_set and meta.get("kind") not in kind_set:
                    continue
                if session_id is not None and meta.get("session_id") != session_id:
                    continue
                if tag_set and not (tag_set & set(meta.get("tags", []))):
                    continue
                out.append(
                    self._to_memory(
                        SearchResult(
                            text=doc.text,
                            score=0.0,
                            metadata=meta,
                            namespace=doc.namespace,
                            doc_id=doc.doc_id,
                            modality=doc.modality,
                            timestamp=doc.created_at,
                            updated_at=doc.updated_at,
                        ),
                        score=0.0,
                    )
                )
            out.sort(key=lambda m: m.created_at, reverse=newest_first)
            return out[:limit] if limit and limit > 0 else out

    @staticmethod
    def _recency_factor(timestamp: float, now: float) -> float:
        age_days = max(0.0, (now - timestamp) / 86400.0)
        return float(math.exp(-math.log(2.0) * age_days / _RECENCY_HALF_LIFE_DAYS))

    def recall(
        self,
        query: str,
        top_k: int = 8,
        kinds: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
        session_id: str | None = None,
        since: float | None = None,
        min_importance: float = 0.0,
        recency_weight: float = 0.15,
        importance_weight: float = 0.25,
        alpha: float = 0.6,
        touch: bool = True,
    ) -> list[Memory]:
        """Recall the memories most relevant to a query.

        Score = ``(1 - rw - iw) * relevance + iw * importance + rw * recency``
        where recency decays with a 30-day half-life.

        Args:
            query: What you need to remember.
            top_k: Number of memories to return.
            kinds: Restrict to these kinds.
            tags: Require at least one of these tags.
            session_id: Restrict to one session.
            since: Only memories created after this Unix timestamp.
            min_importance: Drop memories below this importance.
            recency_weight: Weight of recency in the final score.
            importance_weight: Weight of importance in the final score.
            alpha: Semantic vs keyword balance for the underlying search.
            touch: Record the access (updates ``access_count`` and
                ``last_accessed``).

        Returns:
            Memories sorted by descending score (empty if none stored).

        Performance:
            Time: one hybrid search + O(k) rescoring

        Example::

            >>> for m in mem.recall("deployment schedule", kinds=["fact"]):
            ...     print(m.score, m.content)
        """
        query = (query or "").strip()
        with self._guard():
            if top_k <= 0 or self._vault.count(self._namespace) == 0:
                return []
            filters: dict[str, Any] = {}
            if kinds:
                filters["kind"] = {"in": [self._normalise_kind(k) for k in kinds]}
            if tags:
                filters["tags"] = {"any": list(tags)}
            if session_id is not None:
                filters["session_id"] = session_id
            if since is not None:
                filters["created_at"] = {"gt": float(since)}
            if min_importance > 0.0:
                filters["importance"] = {"gte": float(min_importance)}

            if not query:
                items = self.list_memories(
                    kinds=kinds, session_id=session_id, tags=tags, limit=0
                )
                items = [m for m in items if m.importance >= min_importance]
                if since is not None:
                    items = [m for m in items if m.created_at > since]
                now = time.time()
                for m in items:
                    m.score = importance_weight * m.importance + (
                        1.0 - importance_weight
                    ) * self._recency_factor(m.created_at, now)
                items.sort(key=lambda m: m.score, reverse=True)
                return items[:top_k]

            try:
                hits = self._vault.search(
                    query,
                    top_k=max(top_k * 3, 10),
                    alpha=alpha,
                    namespace=self._namespace,
                    filters=filters or None,
                )
            except VaultEmptyError:
                return []

            rw = self._clamp(recency_weight)
            iw = self._clamp(importance_weight)
            base = max(0.0, 1.0 - rw - iw)
            now = time.time()
            memories: list[Memory] = []
            for hit in hits:
                m = self._to_memory(hit)
                recency = self._recency_factor(m.last_accessed or m.created_at, now)
                m.score = base * hit.score + iw * m.importance + rw * recency
                memories.append(m)
            memories.sort(key=lambda m: m.score, reverse=True)
            memories = memories[:top_k]

            if touch and memories:
                for m in memories:
                    m.access_count += 1
                    m.last_accessed = now
                    self._vault.update_metadata(
                        m.memory_id,
                        {"access_count": m.access_count, "last_accessed": now},
                    )
            return memories

    # ─── Sessions ───

    def start_session(
        self,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Begin a session; subsequent memories are attributed to it.

        Args:
            session_id: Custom ID (auto-generated if None).
            metadata: Arbitrary metadata (client, project, …).

        Returns:
            The new session.
        """
        with self._guard():
            if session_id is not None:
                existing = self._require_own_session(session_id, create=False)
                if existing is not None:
                    # why: resume instead of wiping summary and turn count
                    existing.ended_at = None
                    existing.metadata.update(metadata or {})
                    self._vault._storage.upsert_session(existing._to_record())
                    self._current_session = existing
                    return existing
            session = Session(
                session_id=session_id or "sess_" + uuid.uuid4().hex[:12],
                agent=self._agent,
                started_at=time.time(),
                metadata=dict(metadata or {}),
            )
            self._vault._storage.upsert_session(session._to_record())
            self._current_session = session
            return session

    def end_session(
        self, summary: str = "", session_id: str | None = None
    ) -> Session | None:
        """End a session and optionally store its summary as a memory.

        Args:
            summary: What happened; stored as a ``summary`` memory so the
                next session can recall it.
            session_id: Session to end (defaults to the current one).

        Returns:
            The ended session, or None if it does not exist.
        """
        with self._guard():
            sid = session_id or (
                self._current_session.session_id if self._current_session else None
            )
            if sid is None:
                return None
            rec = self._vault._storage.get_session(sid)
            if rec is None or rec.agent != self._agent:
                return None
            session = Session._from_record(rec)
            new_summary = summary.strip()
            if session.ended_at is not None and not new_summary:
                # why: ending twice must not wipe the stored summary
                if self._current_session and self._current_session.session_id == sid:
                    self._current_session = None
                return session
            session.ended_at = time.time()
            session.summary = new_summary or session.summary
            if new_summary:
                self.remember(
                    session.summary,
                    kind="summary",
                    importance=0.6,
                    session_id=sid,
                    source="session_summary",
                    dedup=False,
                )
                session.turn_count += 1
            self._vault._storage.upsert_session(session._to_record())
            if self._current_session and self._current_session.session_id == sid:
                self._current_session = None
            return session

    def _require_own_session(self, session_id: str, create: bool) -> Session | None:
        """Return this agent's session, creating it if asked.

        Raises:
            ValueError: If the session belongs to another agent.
        """
        storage = self._vault._storage
        rec = storage.get_session(session_id)
        if rec is not None and rec.agent != self._agent:
            raise ValueError(
                f"Session '{session_id}' belongs to agent '{rec.agent}'.\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            )
        if rec is None and create:
            rec = SessionRecord(
                session_id=session_id, agent=self._agent, started_at=time.time()
            )
            storage.upsert_session(rec)
        return Session._from_record(rec) if rec is not None else None

    def _bump_session(self, session_id: str) -> None:
        storage = self._vault._storage
        storage.increment_session_turns(session_id)
        if self._current_session and self._current_session.session_id == session_id:
            self._current_session.turn_count += 1

    def get_session(self, session_id: str) -> Session | None:
        """Fetch one of this agent's sessions by ID."""
        with self._guard():
            rec = self._vault._storage.get_session(session_id)
            if rec is None or rec.agent != self._agent:
                return None
            return Session._from_record(rec)

    def list_sessions(self, limit: int = 10) -> list[Session]:
        """List this agent's sessions, newest first."""
        with self._guard():
            return [
                Session._from_record(r)
                for r in self._vault._storage.list_sessions(
                    agent=self._agent, limit=max(0, limit)
                )
            ]

    def last_session(self, exclude_current: bool = True) -> Session | None:
        """Return the most recent session (excluding the current one).

        Args:
            exclude_current: Skip the session started by this store.

        Returns:
            The previous session, or None.
        """
        for s in self.list_sessions(limit=5):
            if (
                exclude_current
                and self._current_session
                and (s.session_id == self._current_session.session_id)
            ):
                continue
            return s
        return None

    def session_memories(self, session_id: str, limit: int = 100) -> list[Memory]:
        """List the memories written during a session (oldest first)."""
        return self.list_memories(
            session_id=session_id, limit=limit, newest_first=False
        )

    # ─── Context packing ───

    def context_pack(
        self,
        query: str | None = None,
        max_chars: int = 4000,
        top_k: int = 12,
        kinds: Sequence[str] | None = None,
        include_last_session: bool = True,
        touch: bool = False,
    ) -> str:
        """Build a prompt-ready briefing from memory.

        The pack lists the previous session's summary, then the most
        relevant memories grouped by kind (preferences and decisions
        first), trimmed to ``max_chars``.

        Args:
            query: Topic to focus recall on. None returns the most
                important recent memories.
            max_chars: Character budget (~4 chars per token).
            top_k: Maximum memories considered.
            kinds: Restrict to these kinds.
            include_last_session: Prepend the last session summary.
            touch: Record accesses for the memories included.

        Returns:
            Markdown text, or an empty string when nothing is stored.

        Performance:
            Time: one recall + O(k) formatting
        """
        if max_chars <= 0:
            return ""
        with self._guard():
            memories = self.recall(query or "", top_k=top_k, kinds=kinds, touch=touch)
            lines: list[str] = []
            header = f"## Memory for agent '{self._agent}'" + (
                f" — relevant to: {query.strip()}" if query and query.strip() else ""
            )
            lines.append(header)

            if include_last_session:
                last = self.last_session()
                if last is not None and last.summary:
                    when = time.strftime(
                        "%Y-%m-%d %H:%M", time.localtime(last.started_at)
                    )
                    lines.append(f"\n### Last session ({when})\n{last.summary}")
                    memories = [
                        m
                        for m in memories
                        if not (m.kind == "summary" and m.content == last.summary)
                    ]

            if memories:
                grouped: dict[str, list[Memory]] = {}
                for m in memories:
                    grouped.setdefault(m.kind, []).append(m)
                for kind in sorted(grouped, key=lambda k: _KIND_ORDER.get(k, 99)):
                    lines.append(f"\n### {_KIND_LABELS.get(kind, kind.title())}")
                    for m in grouped[kind]:
                        tag_str = f" [{', '.join(m.tags)}]" if m.tags else ""
                        lines.append(f"- {m.content}{tag_str}")
            elif len(lines) == 1:
                return ""

            text = "\n".join(lines)
            if len(text) > max_chars:
                text = text[: max(0, max_chars - 1)].rstrip() + "…"
            return text

    # ─── Import / export ───

    def export_json(self, path: str | None = None) -> list[dict[str, Any]]:
        """Export every memory (and session) to JSON.

        Args:
            path: File to write. None returns the records only.

        Returns:
            The exported memory records.
        """
        with self._guard():
            memories = [
                m.to_dict() for m in self.list_memories(limit=0, newest_first=False)
            ]
            payload = {
                "agent": self._agent,
                "exported_at": time.time(),
                "memories": memories,
                "sessions": [s.to_dict() for s in self.list_sessions(limit=10_000)],
            }
            if path:
                Path(path).expanduser().write_text(
                    json.dumps(payload, indent=2, default=str), encoding="utf-8"
                )
            return memories

    def import_json(self, path: str, dedup: bool = True) -> int:
        """Import memories exported by :meth:`export_json`.

        Args:
            path: JSON file.
            dedup: Skip duplicates of existing memories.

        Returns:
            Number of memories imported (duplicates excluded).
        """
        with self._guard():
            payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
            records = (
                payload.get("memories", payload)
                if isinstance(payload, dict)
                else payload
            )
            sessions = payload.get("sessions", []) if isinstance(payload, dict) else []
            for sess in sessions:
                existing = self._vault._storage.get_session(sess["session_id"])
                if existing is not None and existing.agent != self._agent:
                    continue
                self._vault._storage.upsert_session(
                    SessionRecord(
                        session_id=sess["session_id"],
                        agent=self._agent,
                        started_at=float(sess.get("started_at", 0.0)),
                        ended_at=sess.get("ended_at"),
                        summary=sess.get("summary", ""),
                        metadata=sess.get("metadata") or {},
                        turn_count=int(sess.get("turn_count", 0)),
                    )
                )
            imported = 0
            for rec in records:
                m = self.remember(
                    rec["content"],
                    kind=rec.get("kind", _DEFAULT_KIND),
                    importance=float(rec.get("importance", 0.5)),
                    tags=rec.get("tags") or [],
                    metadata=rec.get("metadata") or {},
                    session_id=rec.get("session_id"),
                    source=rec.get("source"),
                    dedup=dedup,
                    created_at=rec.get("created_at"),
                    _count_turn=False,
                )
                if not m.deduplicated:
                    imported += 1
            return imported

    # ─── Stats / lifecycle ───

    def stats(self) -> dict[str, Any]:
        """Return memory statistics for this agent."""
        with self._guard():
            memories = self.list_memories(limit=0)
            by_kind: dict[str, int] = {}
            for m in memories:
                by_kind[m.kind] = by_kind.get(m.kind, 0) + 1
            sessions = self.list_sessions(limit=10_000)
            return {
                "agent": self._agent,
                "namespace": self._namespace,
                "path": self._vault.path,
                "memories": len(memories),
                "by_kind": dict(sorted(by_kind.items())),
                "sessions": len(sessions),
                "open_sessions": sum(1 for s in sessions if s.is_open),
                "embedding_model": self._vault.embedder.name,
                "encrypted": self._vault.stats()["encrypted"],
            }

    def save(self) -> None:
        """Persist the vault's index snapshot."""
        self._vault.save()

    def close(self) -> None:
        """Close the store (and the vault, if this store opened it)."""
        with self._lock:
            if self._owns_vault and not self._vault.closed:
                self._vault.close()

    # why: ``list`` reads naturally at call sites; defined last so the
    # builtin ``list`` type stays available for annotations in this class.
    list = list_memories

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"MemoryStore(agent='{self._agent}', path='{self._vault.path}')"
