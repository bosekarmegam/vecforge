# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
FastAPI REST server for VecForge.

Provides a REST API for vault operations. Designed for local-first
deployment — no cloud dependency required.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from vecforge.core.vault import VecForge
from vecforge.server.routes import create_router

_ENV_API_KEY = "VECFORGE_API_KEY"


def create_app(
    vault_path: str = ":memory:",
    api_key: str | None = None,
    model_name: str | None = None,
    vault: VecForge | None = None,
    encryption_key: str | None = None,
) -> FastAPI:
    """Create the FastAPI application.

    Args:
        vault_path: Path to the vault database.
        api_key: If set (or ``$VECFORGE_API_KEY`` is set), every request
            must carry a matching ``X-API-Key`` header.
        model_name: Embedding model name or ``"hash"``.
        vault: Reuse an already-open vault instead of opening
            ``vault_path``.
        encryption_key: Optional SQLCipher key for the vault.

    Returns:
        FastAPI application instance. The vault lives on ``app.state.vault``
        and is closed when the application shuts down.

    Example:
        >>> app = create_app("my_vault.db")
        >>> # Run with: uvicorn vecforge.server.app:app
    """
    from vecforge import __version__

    owns_vault = vault is None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.vault is None:
            app.state.vault = VecForge(
                vault_path,
                model_name=model_name,
                encryption_key=encryption_key,
                auto_save_every=500,
            )
        try:
            yield
        finally:
            # why: cached memory stores point at this vault; drop them so a
            # restarted app builds fresh ones
            app.state.memory_stores.clear()
            if owns_vault and app.state.vault is not None:
                app.state.vault.close()
                app.state.vault = None

    app = FastAPI(
        title="VecForge API",
        description=(
            "VecForge — Universal Local-First Vector Database REST API.\n\n"
            "Built by Suneel Bose K · ArcGX TechLabs Private Limited.\n"
            "Licensed under BSL 1.1."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.state.vault = vault
    app.state.memory_stores = {}
    app.state.api_key = api_key or os.environ.get(_ENV_API_KEY) or None

    # why: open eagerly too, so the app works without a lifespan-aware host
    if app.state.vault is None:
        app.state.vault = VecForge(
            vault_path,
            model_name=model_name,
            encryption_key=encryption_key,
            auto_save_every=500,
        )

    app.include_router(create_router())
    return app
