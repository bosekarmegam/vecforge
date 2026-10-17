# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
FastAPI route definitions for VecForge REST API.

Provides endpoints for documents (add, batch add, get, search, update
metadata, delete), namespaces, vault statistics and the AI memory layer.

Endpoints are plain ``def`` functions so FastAPI runs them in its thread
pool; the vault serialises access with its own lock, so the event loop is
never blocked by embedding or search work.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import hmac
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from vecforge.core.vault import SearchResult, VecForge
from vecforge.exceptions import (
    DeletionProtectedError,
    DocumentNotFoundError,
    DuplicateDocumentError,
    NamespaceNotFoundError,
    VaultEmptyError,
    VecForgeError,
    VecForgePermissionError,
)
from vecforge.memory import MemoryStore

# ─── Request/Response Models ───


class AddRequest(BaseModel):
    """Request body for adding a document."""

    text: str = Field(..., description="Document text content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")
    namespace: str = Field(default="default", description="Target namespace")
    doc_id: str | None = Field(default=None, description="Custom document ID")
    modality: str = Field(default="text", description="Content modality")


class AddResponse(BaseModel):
    """Response for a successful add operation."""

    doc_id: str
    message: str = "Document added successfully"


class AddBatchRequest(BaseModel):
    """Request body for adding many documents at once."""

    texts: list[str] = Field(..., description="Document texts")
    metadata_list: list[dict[str, Any]] | None = Field(
        default=None, description="One metadata dict per text"
    )
    namespace: str = Field(default="default", description="Target namespace")
    doc_ids: list[str | None] | None = Field(default=None, description="Custom IDs")


class AddBatchResponse(BaseModel):
    """Response for a batch add."""

    doc_ids: list[str]
    count: int


class UpsertRequest(BaseModel):
    """Request body for inserting-or-replacing a document."""

    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    namespace: str = "default"
    modality: str = "text"


class MetadataUpdateRequest(BaseModel):
    """Request body for updating document metadata."""

    metadata: dict[str, Any]
    merge: bool = Field(default=True, description="Merge into existing metadata")


class SearchRequest(BaseModel):
    """Request body for searching the vault."""

    query: str = Field(..., description="Search query")
    top_k: int = Field(default=10, ge=1, le=1000, description="Number of results")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Semantic weight")
    namespace: str | None = Field(default=None, description="Restrict to namespace")
    filters: dict[str, Any] | None = Field(default=None, description="Metadata filters")
    rerank: bool = Field(default=False, description="Enable cross-encoder reranking")
    quantum_rerank: bool = Field(default=False, description="Enable Grover reranking")
    recency_weight: float = Field(default=0.0, ge=0.0, le=1.0, description="Recency")


class SearchResultItem(BaseModel):
    """A single search result."""

    text: str
    score: float
    metadata: dict[str, Any]
    namespace: str
    doc_id: str
    modality: str
    timestamp: float = 0.0
    dense_score: float | None = None
    sparse_score: float | None = None
    signals: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Response for a search operation."""

    results: list[SearchResultItem]
    count: int


class StatsResponse(BaseModel):
    """Response for vault statistics."""

    path: str
    version: str
    documents: int
    namespaces: list[str]
    namespace_counts: dict[str, int]
    encrypted: bool
    quantum: bool
    deletion_protection: bool
    index_vectors: int
    index_type: str
    tombstones: int
    bm25_documents: int
    embedding_model: str
    dimension: int | None
    schema_version: int


class NamespaceRequest(BaseModel):
    """Request body for creating a namespace."""

    name: str = Field(..., description="Namespace name")


class RememberRequest(BaseModel):
    """Request body for storing a memory."""

    content: str
    kind: str = "fact"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    session_id: str | None = None
    source: str | None = None
    agent: str = "default"


class RecallRequest(BaseModel):
    """Request body for recalling memories."""

    query: str
    top_k: int = Field(default=8, ge=1, le=100)
    kinds: list[str] | None = None
    tags: list[str] | None = None
    session_id: str | None = None
    min_importance: float = Field(default=0.0, ge=0.0, le=1.0)
    agent: str = "default"


class SessionStartRequest(BaseModel):
    """Request body for starting a memory session."""

    agent: str = "default"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionEndRequest(BaseModel):
    """Request body for ending a memory session."""

    agent: str = "default"
    session_id: str | None = None
    summary: str = ""


# ─── Helpers ───


def _item(r: SearchResult) -> SearchResultItem:
    return SearchResultItem(
        text=r.text,
        score=r.score,
        metadata=r.metadata,
        namespace=r.namespace,
        doc_id=r.doc_id,
        modality=r.modality,
        timestamp=r.timestamp,
        dense_score=r.dense_score,
        sparse_score=r.sparse_score,
        signals=r.signals,
    )


def _http_error(exc: VecForgeError) -> HTTPException:
    message = str(exc)
    if isinstance(
        exc, VaultEmptyError | DocumentNotFoundError | NamespaceNotFoundError
    ):
        return HTTPException(status_code=404, detail=message)
    if isinstance(exc, VecForgePermissionError | DeletionProtectedError):
        return HTTPException(status_code=403, detail=message)
    if isinstance(exc, DuplicateDocumentError):
        return HTTPException(status_code=409, detail=message)
    return HTTPException(status_code=400, detail=message)


def get_vault(request: Request) -> VecForge:
    """FastAPI dependency returning the application's vault."""
    vault: VecForge = request.app.state.vault
    return vault


def get_memory(request: Request, agent: str) -> MemoryStore:
    """Return (and cache) the memory store for an agent."""
    stores: dict[str, MemoryStore] = request.app.state.memory_stores
    store = stores.get(agent)
    if store is None:
        store = MemoryStore(vault=request.app.state.vault, agent=agent)
        stores[agent] = store
    return store


def require_api_key(
    request: Request, x_api_key: str | None = Header(default=None)
) -> None:
    """Reject requests without the configured ``X-API-Key`` header."""
    expected: str | None = getattr(request.app.state, "api_key", None)
    if not expected:
        return
    if x_api_key is None or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


# ─── Router ───


def create_router() -> APIRouter:
    """Create the API router.

    The vault is read from ``request.app.state.vault`` so several apps can
    coexist in one process and the vault is closed on shutdown.

    Returns:
        Configured APIRouter.
    """
    router = APIRouter(
        prefix="/api/v1", tags=["VecForge"], dependencies=[Depends(require_api_key)]
    )

    # ─── Documents ───

    @router.post("/add", response_model=AddResponse)
    def add_document(
        request: AddRequest, vault: VecForge = Depends(get_vault)
    ) -> AddResponse:
        """Add a document to the vault."""
        try:
            doc_id = vault.add(
                text=request.text,
                metadata=request.metadata,
                namespace=request.namespace,
                doc_id=request.doc_id,
                modality=request.modality,
            )
            return AddResponse(doc_id=doc_id)
        except VecForgeError as e:
            raise _http_error(e) from e

    @router.post("/add_batch", response_model=AddBatchResponse)
    def add_batch(
        request: AddBatchRequest, vault: VecForge = Depends(get_vault)
    ) -> AddBatchResponse:
        """Add many documents in one call."""
        try:
            ids = vault.add_batch(
                request.texts,
                metadata_list=request.metadata_list,
                namespace=request.namespace,
                doc_ids=request.doc_ids,
            )
            return AddBatchResponse(doc_ids=ids, count=len(ids))
        except (VecForgeError, ValueError) as e:
            if isinstance(e, VecForgeError):
                raise _http_error(e) from e
            raise HTTPException(status_code=400, detail=str(e)) from e

    @router.post("/search", response_model=SearchResponse)
    def search_vault(
        request: SearchRequest, vault: VecForge = Depends(get_vault)
    ) -> SearchResponse:
        """Search the vault with a natural language query."""
        try:
            results = vault.search(
                query=request.query,
                top_k=request.top_k,
                alpha=request.alpha,
                namespace=request.namespace,
                filters=request.filters,
                rerank=request.rerank,
                quantum_rerank=request.quantum_rerank,
                recency_weight=request.recency_weight,
            )
            items = [_item(r) for r in results]
            return SearchResponse(results=items, count=len(items))
        except VecForgeError as e:
            raise _http_error(e) from e

    @router.get("/docs/{doc_id}", response_model=SearchResultItem)
    def get_document(
        doc_id: str, vault: VecForge = Depends(get_vault)
    ) -> SearchResultItem:
        """Fetch a document by ID."""
        try:
            result = vault.get(doc_id)
        except VecForgeError as e:
            raise _http_error(e) from e
        if result is None:
            raise HTTPException(
                status_code=404, detail=f"Document '{doc_id}' not found"
            )
        return _item(result)

    @router.put("/docs/{doc_id}", response_model=AddResponse)
    def upsert_document(
        doc_id: str, request: UpsertRequest, vault: VecForge = Depends(get_vault)
    ) -> AddResponse:
        """Insert or replace a document."""
        try:
            vault.upsert(
                request.text,
                doc_id=doc_id,
                metadata=request.metadata,
                namespace=request.namespace,
                modality=request.modality,
            )
            return AddResponse(doc_id=doc_id, message="Document upserted")
        except VecForgeError as e:
            raise _http_error(e) from e

    @router.patch("/docs/{doc_id}/metadata")
    def update_metadata(
        doc_id: str,
        request: MetadataUpdateRequest,
        vault: VecForge = Depends(get_vault),
    ) -> dict[str, Any]:
        """Update a document's metadata."""
        try:
            meta = vault.update_metadata(doc_id, request.metadata, merge=request.merge)
            return {"doc_id": doc_id, "metadata": meta}
        except VecForgeError as e:
            raise _http_error(e) from e

    @router.delete("/docs/{doc_id}")
    def delete_document(
        doc_id: str, vault: VecForge = Depends(get_vault)
    ) -> dict[str, Any]:
        """Delete a document by ID."""
        try:
            deleted = vault.delete(doc_id)
        except VecForgeError as e:
            raise _http_error(e) from e
        if not deleted:
            raise HTTPException(
                status_code=404, detail=f"Document '{doc_id}' not found"
            )
        return {"deleted": True, "doc_id": doc_id}

    @router.get("/docs/{doc_id}/similar", response_model=SearchResponse)
    def similar_documents(
        doc_id: str, top_k: int = 10, vault: VecForge = Depends(get_vault)
    ) -> SearchResponse:
        """Find documents similar to an existing one."""
        try:
            results = vault.similar(doc_id, top_k=top_k)
        except VecForgeError as e:
            raise _http_error(e) from e
        items = [_item(r) for r in results]
        return SearchResponse(results=items, count=len(items))

    # ─── Vault ───

    @router.get("/stats", response_model=StatsResponse)
    def get_stats(vault: VecForge = Depends(get_vault)) -> StatsResponse:
        """Get vault statistics."""
        info = vault.stats()
        return StatsResponse(**{k: info[k] for k in StatsResponse.model_fields})

    @router.post("/save")
    def save_vault(vault: VecForge = Depends(get_vault)) -> dict[str, str]:
        """Persist the index snapshot now."""
        vault.save()
        return {"status": "saved"}

    @router.post("/compact")
    def compact_vault(vault: VecForge = Depends(get_vault)) -> dict[str, Any]:
        """Rebuild indexes without tombstones."""
        vault.compact()
        return {"status": "compacted", "documents": vault.count()}

    # ─── Namespaces ───

    @router.post("/namespaces")
    def create_namespace(
        request: NamespaceRequest, vault: VecForge = Depends(get_vault)
    ) -> dict[str, str]:
        """Create a new namespace."""
        try:
            vault.create_namespace(request.name)
            return {"created": request.name}
        except VecForgeError as e:
            raise _http_error(e) from e

    @router.get("/namespaces")
    def list_namespaces(vault: VecForge = Depends(get_vault)) -> dict[str, list[str]]:
        """List all namespaces."""
        return {"namespaces": vault.list_namespaces()}

    @router.delete("/namespaces/{name}")
    def delete_namespace(
        name: str, vault: VecForge = Depends(get_vault)
    ) -> dict[str, Any]:
        """Delete a namespace and its documents."""
        try:
            deleted = vault.delete_namespace(name)
            return {"deleted": name, "documents": deleted}
        except VecForgeError as e:
            raise _http_error(e) from e

    # ─── Memory ───

    @router.post("/memory/remember")
    def memory_remember(request: RememberRequest, req: Request) -> dict[str, Any]:
        """Store a memory for an agent."""
        store = get_memory(req, request.agent)
        try:
            m = store.remember(
                request.content,
                kind=request.kind,
                importance=request.importance,
                tags=request.tags,
                session_id=request.session_id,
                source=request.source,
            )
        except (VecForgeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"stored": not m.deduplicated, "memory": m.to_dict()}

    @router.post("/memory/recall")
    def memory_recall(request: RecallRequest, req: Request) -> dict[str, Any]:
        """Recall memories for an agent."""
        store = get_memory(req, request.agent)
        memories = store.recall(
            request.query,
            top_k=request.top_k,
            kinds=request.kinds,
            tags=request.tags,
            session_id=request.session_id,
            min_importance=request.min_importance,
        )
        return {"count": len(memories), "memories": [m.to_dict() for m in memories]}

    @router.get("/memory/context")
    def memory_context(
        req: Request, agent: str = "default", query: str = "", max_chars: int = 4000
    ) -> dict[str, str]:
        """Prompt-ready briefing for an agent."""
        store = get_memory(req, agent)
        return {"context": store.context_pack(query or None, max_chars=max_chars)}

    @router.delete("/memory/{memory_id}")
    def memory_forget(
        memory_id: str, req: Request, agent: str = "default"
    ) -> dict[str, Any]:
        """Delete a memory."""
        store = get_memory(req, agent)
        try:
            deleted = store.forget(memory_id)
        except VecForgeError as e:
            raise _http_error(e) from e
        if not deleted:
            raise HTTPException(
                status_code=404, detail=f"Memory '{memory_id}' not found"
            )
        return {"deleted": True, "memory_id": memory_id}

    @router.post("/memory/sessions/start")
    def memory_session_start(
        request: SessionStartRequest, req: Request
    ) -> dict[str, Any]:
        """Start a memory session."""
        store = get_memory(req, request.agent)
        session = store.start_session(metadata=request.metadata)
        last = store.last_session()
        return {
            "session": session.to_dict(),
            "last_session": last.to_dict() if last else None,
        }

    @router.post("/memory/sessions/end")
    def memory_session_end(request: SessionEndRequest, req: Request) -> dict[str, Any]:
        """End a memory session with a summary."""
        store = get_memory(req, request.agent)
        session = store.end_session(
            summary=request.summary, session_id=request.session_id
        )
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"session": session.to_dict()}

    @router.get("/memory/sessions")
    def memory_sessions(
        req: Request, agent: str = "default", limit: int = 10
    ) -> dict[str, Any]:
        """List recent memory sessions."""
        store = get_memory(req, agent)
        return {"sessions": [s.to_dict() for s in store.list_sessions(limit=limit)]}

    # ─── Health ───

    @router.get("/health")
    def health_check(vault: VecForge = Depends(get_vault)) -> dict[str, Any]:
        """Health check endpoint."""
        from vecforge import __version__

        return {
            "status": "healthy",
            "service": "VecForge",
            "version": __version__,
            "documents": vault.count(),
            "built_by": "Suneel Bose K · ArcGX TechLabs",
        }

    return router
