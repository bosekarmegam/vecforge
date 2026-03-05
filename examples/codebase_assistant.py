# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
Codebase Assistant Example — VecForge

Demonstrates building a code documentation search system:
- Index code documentation and comments
- Semantic search for function discovery
- Metadata tagging by language and module

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

Usage:
    python examples/codebase_assistant.py
"""

from __future__ import annotations

from vecforge import VecForge


def main() -> None:
    """Run the codebase assistant demo."""

    db = VecForge(":memory:")

    print("💻 VecForge Codebase Assistant Demo")
    print("=" * 50)

    # ─── Index code documentation ───
    print("\n📝 Indexing code documentation...")

    docs = [
        ("VecForge.add() — Add a text document to the vault. "
         "Accepts text, metadata dict, namespace, and optional doc_id. "
         "Returns the document UUID. Requires write permission.",
         {"module": "core.vault", "language": "python", "type": "method"}),

        ("VecForge.search() — Perform hybrid cascading search. "
         "4-stage pipeline: FAISS dense → BM25 merge → metadata filter → rerank. "
         "Alpha controls semantic vs keyword weight. Returns list of SearchResult.",
         {"module": "core.vault", "language": "python", "type": "method"}),

        ("FaissIndexer — FAISS-based vector index for nearest-neighbour search. "
         "Starts with IndexFlatIP for small collections. Supports serialization "
         "to/from bytes for persistent storage.",
         {"module": "core.indexer", "language": "python", "type": "class"}),

        ("BM25Engine — BM25 keyword search using Okapi BM25. "
         "Maintains in-memory inverted index. Uses argpartition for O(N) top-k. "
         "Supports add_documents(), add_document(), search(), and reset().",
         {"module": "core.bm25", "language": "python", "type": "class"}),

        ("StorageBackend — SQLite/SQLCipher persistence backend. "
         "All queries namespace-scoped. WAL mode for concurrent reads. "
         "Stores documents, embeddings, metadata, and FAISS index.",
         {"module": "core.storage", "language": "python", "type": "class"}),

        ("MetadataFilter — Filter search results by metadata conditions. "
         "Supports operators: eq, gte, lte, gt, lt, in, not_in, ne. "
         "Multiple filters combined with AND logic.",
         {"module": "search.filters", "language": "python", "type": "class"}),

        ("RBACManager — Role-based access control with three roles: "
         "admin (full access), read-write (CRUD), read-only (search only). "
         "Maps API keys to roles. require() raises VecForgePermissionError.",
         {"module": "security.rbac", "language": "python", "type": "class"}),

        ("AuditLogger — Append-only JSONL audit log for compliance. "
         "Logs actor, operation, doc_id, namespace, and metadata. "
         "Supports filtered reading by time range, actor, and operation.",
         {"module": "security.audit", "language": "python", "type": "class"}),

        ("reciprocal_rank_fusion() — Fuse dense and sparse search results "
         "using weighted Reciprocal Rank Fusion (RRF). Each document score: "
         "alpha * 1/(k+dense_rank) + (1-alpha) * 1/(k+sparse_rank).",
         {"module": "search.hybrid", "language": "python", "type": "function"}),
    ]

    for text, metadata in docs:
        db.add(text, metadata=metadata)
    print(f"  ✅ Indexed {len(docs)} code documentation entries")

    # ─── Developer queries ───
    queries = [
        "How do I add a document to the database?",
        "What search algorithm does VecForge use?",
        "How does permission checking work?",
        "How are dense and sparse scores combined?",
        "What metadata filtering operators are available?",
    ]

    for query in queries:
        print(f"\n{'─' * 50}")
        print(f"❓ Developer asks: {query}")

        results = db.search(query, top_k=2, alpha=0.6)
        for i, r in enumerate(results, 1):
            print(f"\n  📌 Result {i} [{r.score:.4f}]")
            print(f"     Module: {r.metadata.get('module', 'N/A')}")
            print(f"     Type:   {r.metadata.get('type', 'N/A')}")
            print(f"     {r.text[:100]}...")

    db.close()
    print(f"\n{'=' * 50}")
    print("✅ Codebase assistant demo complete!")
    print("VecForge by Suneel Bose K · ArcGX TechLabs")


if __name__ == "__main__":
    main()
