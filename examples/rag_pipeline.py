# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
RAG Pipeline Example — VecForge

Demonstrates building a Retrieval-Augmented Generation (RAG) pipeline:
1. Ingest documents into VecForge
2. User asks a question
3. VecForge retrieves relevant context
4. Context is formatted for an LLM prompt

This example uses VecForge as the retrieval backend.
The LLM call is shown as a template — plug in any LLM you want.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

Usage:
    python examples/rag_pipeline.py
"""

from __future__ import annotations

from vecforge import VecForge


def build_rag_prompt(question: str, context_chunks: list[str]) -> str:
    """Build a RAG prompt with retrieved context.

    Args:
        question: User's question.
        context_chunks: Retrieved text chunks from VecForge.

    Returns:
        Formatted prompt string ready for an LLM.
    """
    context = "\n\n---\n\n".join(context_chunks)
    return f"""You are a helpful assistant. Answer the question based ONLY on the
provided context. If the answer is not in the context, say "I don't have
enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""


def main() -> None:
    """Run the RAG pipeline demo."""

    db = VecForge(":memory:")

    print("🤖 VecForge RAG Pipeline Demo")
    print("=" * 50)

    # ─── Step 1: Ingest knowledge base ───
    print("\n📚 Building knowledge base...")

    knowledge = [
        ("VecForge uses FAISS for dense vector similarity search. "
         "FAISS supports both flat (exact) indexes for small datasets "
         "and IVF (approximate) indexes for larger collections. "
         "The default index type is IndexFlatIP using inner product similarity."),

        ("VecForge combines dense retrieval with BM25 keyword search "
         "using Reciprocal Rank Fusion (RRF). The alpha parameter controls "
         "the balance: alpha=0.0 is keyword-only, alpha=1.0 is semantic-only, "
         "and alpha=0.5 is a balanced hybrid (recommended for most use cases)."),

        ("VecForge supports AES-256 encryption at rest using SQLCipher. "
         "To enable encryption, pass an encryption_key when creating the vault. "
         "Always use environment variables for keys, never hardcode them. "
         "Without SQLCipher installed, VecForge falls back to unencrypted SQLite."),

        ("VecForge's namespace feature provides multi-tenant data isolation. "
         "Each namespace is a logical partition within the vault. Documents in "
         "one namespace are invisible to searches in another namespace. "
         "All SQL queries are scoped with WHERE namespace = ? for security."),

        ("VecForge supports cross-encoder reranking for improved search precision. "
         "When rerank=True is passed to search(), the top candidates are "
         "reranked using a cross-encoder model. This adds ~20-50ms latency but "
         "significantly improves the relevance ordering of results."),

        ("VecForge is built by Suneel Bose K, Founder and CEO of "
         "ArcGX TechLabs Private Limited. It is licensed under the Business "
         "Source License 1.1 (BSL). Free for personal, research, and "
         "non-commercial use. Commercial use requires a separate license."),
    ]

    for text in knowledge:
        db.add(text, metadata={"source": "vecforge_docs"})
    print(f"  ✅ Added {len(knowledge)} knowledge chunks")

    # ─── Step 2: User questions ───
    questions = [
        "How does VecForge handle encryption?",
        "What is the alpha parameter in search?",
        "How does namespace isolation work?",
        "Who built VecForge?",
    ]

    for question in questions:
        print(f"\n{'─' * 50}")
        print(f"❓ Question: {question}")
        print(f"{'─' * 50}")

        # ─── Step 3: Retrieve relevant context ───
        results = db.search(question, top_k=2, alpha=0.6)
        context_chunks = [r.text for r in results]

        print(f"\n📎 Retrieved {len(results)} relevant chunks:")
        for i, r in enumerate(results, 1):
            print(f"   {i}. [{r.score:.4f}] {r.text[:60]}...")

        # ─── Step 4: Build RAG prompt ───
        prompt = build_rag_prompt(question, context_chunks)

        print(f"\n📝 Generated RAG prompt ({len(prompt)} chars):")
        print(f"   {prompt[:150]}...")
        print("\n   → Send this to any LLM (GPT-4, Claude, Llama, etc.)")

    db.close()
    print(f"\n{'=' * 50}")
    print("✅ RAG pipeline demo complete!")
    print("VecForge by Suneel Bose K · ArcGX TechLabs")


if __name__ == "__main__":
    main()
