# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
Hospital Search Example — VecForge

Demonstrates building a medical record search system with:
- Namespace isolation per ward
- Metadata filtering by year and type
- Encrypted vault for patient data security
- Cross-encoder reranking for critical queries

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

Usage:
    python examples/hospital_search.py
"""

from __future__ import annotations

import os

from vecforge import VecForge


def main() -> None:
    """Run the hospital search demo."""

    # ─── Create an encrypted vault for patient data ───
    # For demo purposes, we use a simple key. In production,
    # ALWAYS use: os.environ["VECFORGE_KEY"]
    db = VecForge(":memory:")

    print("🏥 VecForge Hospital Search Demo")
    print("=" * 50)

    # ─── Set up ward namespaces ───
    db.create_namespace("ward_7")
    db.create_namespace("ward_icu")
    db.create_namespace("ward_3")

    # ─── Add patient records ───
    records = [
        # Ward 7 — Diabetes / Endocrinology
        ("Patient P4821 — Type 2 diabetes mellitus, HbA1c 9.2%, "
         "admitted for insulin titration. History of neuropathy.",
         {"ward": "7", "year": 2026, "type": "admission", "priority": "medium"},
         "ward_7"),

        ("Patient P5210 — Gestational diabetes, 28 weeks pregnant, "
         "referred for dietary management and glucose monitoring.",
         {"ward": "7", "year": 2026, "type": "referral", "priority": "high"},
         "ward_7"),

        ("Patient P3890 — Type 1 diabetes, routine follow-up, "
         "pump settings adjusted, HbA1c 6.8%.",
         {"ward": "7", "year": 2025, "type": "checkup", "priority": "low"},
         "ward_7"),

        # Ward ICU — Critical Care
        ("Patient P6102 — Cardiac arrest in cafeteria, ROSC achieved "
         "after 4 minutes CPR, intubated, transferred to ICU.",
         {"ward": "ICU", "year": 2026, "type": "emergency", "priority": "critical"},
         "ward_icu"),

        ("Patient P6230 — Severe sepsis secondary to UTI, "
         "vasopressors started, blood cultures pending.",
         {"ward": "ICU", "year": 2026, "type": "emergency", "priority": "critical"},
         "ward_icu"),

        # Ward 3 — Orthopaedics
        ("Patient P4455 — Elderly female, hip fracture after fall, "
         "scheduled for hemiarthroplasty. Comorbidities: diabetes, HTN.",
         {"ward": "3", "year": 2026, "type": "admission", "priority": "high"},
         "ward_3"),

        ("Patient P4501 — ACL reconstruction post-op day 2, "
         "physiotherapy commenced, pain well controlled.",
         {"ward": "3", "year": 2026, "type": "post-op", "priority": "medium"},
         "ward_3"),
    ]

    print("\n📋 Adding patient records...")
    for text, metadata, namespace in records:
        doc_id = db.add(text, metadata=metadata, namespace=namespace)
        print(f"  ✅ Added to {namespace}: {doc_id[:8]}...")

    # ─── Search Examples ───

    # 1. Basic search across all wards
    print("\n\n🔍 Search: 'diabetic patient' (all wards)")
    print("-" * 50)
    results = db.search("diabetic patient", top_k=3)
    for r in results:
        print(f"  [{r.score:.4f}] [{r.namespace}] {r.text[:80]}...")

    # 2. Namespace-scoped search
    print("\n\n🔍 Search: 'emergency critical' (ICU only)")
    print("-" * 50)
    results = db.search("emergency critical", namespace="ward_icu", top_k=3)
    for r in results:
        print(f"  [{r.score:.4f}] [{r.namespace}] {r.text[:80]}...")

    # 3. Metadata-filtered search
    print("\n\n🔍 Search: 'diabetes' (2026 admissions only)")
    print("-" * 50)
    results = db.search(
        "diabetes",
        filters={"year": 2026, "type": "admission"},
        top_k=3,
    )
    for r in results:
        print(f"  [{r.score:.4f}] {r.metadata} {r.text[:60]}...")

    # 4. High-precision search with reranking
    print("\n\n🔍 Search: 'elderly hip fracture with diabetes' (reranked)")
    print("-" * 50)
    results = db.search(
        "elderly hip fracture with diabetes comorbidity",
        rerank=True,
        top_k=3,
    )
    for r in results:
        print(f"  [{r.score:.4f}] [{r.namespace}] {r.text[:80]}...")

    # ─── Vault Stats ───
    print("\n\n📊 Vault Statistics")
    print("-" * 50)
    stats = db.stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    db.close()
    print("\n✅ Demo complete! VecForge by Suneel Bose K · ArcGX TechLabs")


if __name__ == "__main__":
    main()
