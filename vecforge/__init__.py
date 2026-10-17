# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
VecForge — Forge your vector database. Own it forever.

A universal, local-first Python vector database with enterprise security,
multimodal ingestion, quantum-inspired reranking, and a persistent memory
layer for AI agents (Claude, GPT, local models) exposed over MCP.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

Quick Start::

    from vecforge import VecForge

    db = VecForge("my_vault")
    db.add("Patient admitted with type 2 diabetes", metadata={"ward": "7"})
    results = db.search("diabetic patient")
    print(results[0].text)

AI memory across sessions::

    from vecforge import MemoryStore

    mem = MemoryStore("~/.vecforge/memory.db", agent="claude")
    mem.remember("User prefers concise answers", kind="preference")
    print(mem.context_pack("how should I answer?"))
"""

from __future__ import annotations

from vecforge.core.embedder import (
    CallableEmbedder,
    Embedder,
    EmbedderLike,
    HashEmbedder,
    get_embedder,
)
from vecforge.core.vault import SearchResult, VecForge
from vecforge.exceptions import (
    DeletionProtectedError,
    DocumentNotFoundError,
    DuplicateDocumentError,
    EncryptionKeyError,
    IngestError,
    InvalidAlphaError,
    NamespaceNotFoundError,
    VaultClosedError,
    VaultEmptyError,
    VecForgeError,
    VecForgePermissionError,
)
from vecforge.memory import Memory, MemoryStore, Session

__all__ = [
    "VecForge",
    "SearchResult",
    "MemoryStore",
    "Memory",
    "Session",
    "Embedder",
    "EmbedderLike",
    "HashEmbedder",
    "CallableEmbedder",
    "get_embedder",
    "VecForgeError",
    "VaultEmptyError",
    "VaultClosedError",
    "NamespaceNotFoundError",
    "DocumentNotFoundError",
    "DuplicateDocumentError",
    "VecForgePermissionError",
    "InvalidAlphaError",
    "EncryptionKeyError",
    "DeletionProtectedError",
    "IngestError",
]

__version__ = "2.0.0"
__author__ = "Suneel Bose K"
__company__ = "ArcGX TechLabs Private Limited"
__license__ = "BSL-1.1"
__copyright__ = "Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs"
