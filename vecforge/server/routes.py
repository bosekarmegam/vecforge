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

Provides endpoints for add, search, delete, stats, and namespace
management.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from vecforge.core.vault import VecForge
from vecforge.exceptions import (
    VaultEmptyError,
    VecForgeError,
    VecForgePermissionError,
)

# ─── Request/Response Models ───


class AddRequest(BaseModel):
    """Request body for adding a document."""

    text: str = Field(..., description="Document text content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")
    namespace: str = Field(default="default", description="Target namespace")


class AddResponse(BaseModel):
    """Response for a successful add operation."""

    doc_id: str
    message: str = "Document added successfully"


class SearchRequest(BaseModel):
    """Request body for searching the vault."""

    query: str = Field(..., description="Search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Semantic weight")
    namespace: str | None = Field(default=None, description="Restrict to namespace")
    filters: dict[str, Any] | None = Field(default=None, description="Metadata filters")
    rerank: bool = Field(default=False, description="Enable reranking")


class SearchResultItem(BaseModel):
    """A single search result."""

    text: str
    score: float
    metadata: dict[str, Any]
    namespace: str
    doc_id: str
    modality: str


class SearchResponse(BaseModel):
    """Response for a search operation."""

    results: list[SearchResultItem]
    count: int


class StatsResponse(BaseModel):
    """Response for vault statistics."""

    path: str
    documents: int
    namespaces: list[str]
    encrypted: bool
    quantum: bool


class NamespaceRequest(BaseModel):
    """Request body for creating a namespace."""

    name: str = Field(..., description="Namespace name")


# ─── Router ───

_vault_instance: VecForge | None = None


def create_router(vault_path: str) -> APIRouter:
    """Create API router with vault instance.

    Args:
        vault_path: Path to the vault database.

    Returns:
        Configured APIRouter.
    """
    global _vault_instance
    _vault_instance = VecForge(vault_path)

    router = APIRouter(prefix="/api/v1", tags=["VecForge"])

    @router.post("/add", response_model=AddResponse)
    async def add_document(request: AddRequest) -> AddResponse:
        """Add a document to the vault."""
        assert _vault_instance is not None
        try:
            doc_id = _vault_instance.add(
                text=request.text,
                metadata=request.metadata,
                namespace=request.namespace,
            )
            return AddResponse(doc_id=doc_id)
        except VecForgeError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @router.post("/search", response_model=SearchResponse)
    async def search_vault(request: SearchRequest) -> SearchResponse:
        """Search the vault with a natural language query."""
        assert _vault_instance is not None
        try:
            results = _vault_instance.search(
                query=request.query,
                top_k=request.top_k,
                alpha=request.alpha,
                namespace=request.namespace,
                filters=request.filters,
                rerank=request.rerank,
            )
            items = [
                SearchResultItem(
                    text=r.text,
                    score=r.score,
                    metadata=r.metadata,
                    namespace=r.namespace,
                    doc_id=r.doc_id,
                    modality=r.modality,
                )
                for r in results
            ]
            return SearchResponse(results=items, count=len(items))
        except VaultEmptyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except VecForgePermissionError as e:
            raise HTTPException(status_code=403, detail=str(e)) from e
        except VecForgeError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @router.delete("/docs/{doc_id}")
    async def delete_document(doc_id: str) -> dict[str, Any]:
        """Delete a document by ID."""
        assert _vault_instance is not None
        try:
            deleted = _vault_instance.delete(doc_id)
            if not deleted:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document '{doc_id}' not found",
                )
            return {"deleted": True, "doc_id": doc_id}
        except VecForgeError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @router.get("/stats", response_model=StatsResponse)
    async def get_stats() -> StatsResponse:
        """Get vault statistics."""
        assert _vault_instance is not None
        info = _vault_instance.stats()
        return StatsResponse(
            path=info["path"],
            documents=info["documents"],
            namespaces=info["namespaces"],
            encrypted=info["encrypted"],
            quantum=info["quantum"],
        )

    @router.post("/namespaces")
    async def create_namespace(request: NamespaceRequest) -> dict[str, str]:
        """Create a new namespace."""
        assert _vault_instance is not None
        try:
            _vault_instance.create_namespace(request.name)
            return {"created": request.name}
        except VecForgeError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @router.get("/namespaces")
    async def list_namespaces() -> dict[str, list[str]]:
        """List all namespaces."""
        assert _vault_instance is not None
        return {"namespaces": _vault_instance.list_namespaces()}

    @router.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "VecForge",
            "version": "1.0.0",
            "built_by": "Suneel Bose K · ArcGX TechLabs",
        }

    return router
