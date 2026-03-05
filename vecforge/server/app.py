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

from fastapi import FastAPI

from vecforge.server.routes import create_router


def create_app(vault_path: str = ":memory:") -> FastAPI:
    """Create the FastAPI application.

    Args:
        vault_path: Path to the vault database.

    Returns:
        FastAPI application instance.

    Example:
        >>> app = create_app("my_vault.db")
        >>> # Run with: uvicorn vecforge.server.app:app
    """
    app = FastAPI(
        title="VecForge API",
        description=(
            "VecForge — Universal Local-First Vector Database REST API.\n\n"
            "Built by Suneel Bose K · ArcGX TechLabs Private Limited.\n"
            "Licensed under BSL 1.1."
        ),
        version="0.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    router = create_router(vault_path)
    app.include_router(router)

    return app
