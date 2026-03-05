# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
Legal Document Search Example — VecForge

Demonstrates building a legal document search system with:
- Contract and NDA classification via metadata
- Year-range filtering for compliance
- Encrypted vault for confidential documents
- Semantic search for clause discovery

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

Usage:
    python examples/legal_document_search.py
"""

from __future__ import annotations

from vecforge import VecForge


def main() -> None:
    """Run the legal document search demo."""

    db = VecForge(":memory:")

    print("⚖️  VecForge Legal Document Search Demo")
    print("=" * 50)

    # ─── Add legal documents ───
    documents = [
        ("Non-Disclosure Agreement between ArcGX TechLabs and Globex Corp. "
         "Effective date: January 15, 2026. Confidential information includes "
         "all technical specifications, source code, and business strategies. "
         "Term: 3 years from effective date.",
         {"type": "NDA", "year": 2026, "party": "Globex Corp", "status": "active"}),

        ("Master Services Agreement with Initech LLC for cloud infrastructure "
         "consulting. Total contract value: $450,000. Payment terms: Net 30. "
         "Includes SLA of 99.9% uptime guarantee.",
         {"type": "MSA", "year": 2026, "party": "Initech LLC", "status": "active"}),

        ("Employment Agreement for Senior Engineer position. Non-compete "
         "clause: 12 months post-termination within 50-mile radius. "
         "IP assignment: all work product belongs to company.",
         {"type": "Employment", "year": 2025, "status": "active"}),

        ("Software License Agreement granting perpetual license for VecForge "
         "Enterprise Edition. Licensee: MedTech Solutions. Annual support fee: "
         "$50,000. Source code escrow included.",
         {"type": "License", "year": 2026, "party": "MedTech Solutions", "status": "active"}),

        ("Termination notice for contract with Legacy Systems Inc. "
         "30-day notice period. Final payment due March 31, 2025. "
         "All deliverables to be transferred within notice period.",
         {"type": "Termination", "year": 2025, "party": "Legacy Systems", "status": "terminated"}),

        ("Data Processing Agreement compliant with GDPR Article 28. "
         "Data processor: CloudVault Services. Sub-processors must be "
         "approved in writing. Data breach notification within 72 hours.",
         {"type": "DPA", "year": 2026, "party": "CloudVault Services", "status": "active"}),
    ]

    print("\n📄 Adding legal documents...")
    for text, metadata in documents:
        doc_id = db.add(text, metadata=metadata)
        print(f"  ✅ [{metadata['type']}] {doc_id[:8]}...")

    # ─── Search Examples ───

    # 1. Find NDAs
    print("\n\n🔍 Search: 'confidentiality agreement'")
    print("-" * 50)
    results = db.search("confidentiality agreement non-disclosure", top_k=3)
    for r in results:
        print(f"  [{r.score:.4f}] [{r.metadata.get('type')}] {r.text[:80]}...")

    # 2. Filter by type and year
    print("\n\n🔍 Search: Active contracts from 2026")
    print("-" * 50)
    results = db.search(
        "contract agreement",
        filters={"year": 2026, "status": "active"},
        top_k=5,
    )
    for r in results:
        print(f"  [{r.score:.4f}] [{r.metadata.get('type')}] "
              f"Party: {r.metadata.get('party', 'N/A')}")

    # 3. Search for specific clauses
    print("\n\n🔍 Search: 'non-compete clause'")
    print("-" * 50)
    results = db.search("non-compete intellectual property assignment", top_k=3)
    for r in results:
        print(f"  [{r.score:.4f}] [{r.metadata.get('type')}] {r.text[:80]}...")

    # 4. GDPR compliance search
    print("\n\n🔍 Search: 'GDPR data processing breach notification'")
    print("-" * 50)
    results = db.search("GDPR data processing breach notification", top_k=2)
    for r in results:
        print(f"  [{r.score:.4f}] [{r.metadata.get('type')}] {r.text[:80]}...")

    db.close()
    print("\n✅ Demo complete! VecForge by Suneel Bose K · ArcGX TechLabs")


if __name__ == "__main__":
    main()
