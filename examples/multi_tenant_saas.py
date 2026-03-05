# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
Multi-Tenant SaaS Example — VecForge

Demonstrates building a multi-tenant application with:
- Hard namespace isolation between tenants
- RBAC for tenant-specific API keys
- Audit logging for compliance
- Each tenant's data is invisible to others

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

Usage:
    python examples/multi_tenant_saas.py
"""

from __future__ import annotations

from vecforge import VecForge
from vecforge.exceptions import VecForgePermissionError


def main() -> None:
    """Run the multi-tenant SaaS demo."""

    print("🏢 VecForge Multi-Tenant SaaS Demo")
    print("=" * 50)

    # ─── Create vault with audit logging ───
    db = VecForge(":memory:", audit_log="saas_audit.jsonl")

    # ─── Onboard tenants ───
    tenants = {
        "acme_corp": {
            "docs": [
                "Acme Corp Q4 2026 revenue: $45M, up 12% YoY",
                "Acme Corp hiring plan: 50 engineers in Q1 2026",
                "Acme Corp product roadmap: AI assistant launch March 2026",
            ],
        },
        "globex_inc": {
            "docs": [
                "Globex Inc merger with Initech pending board approval",
                "Globex Inc patent portfolio: 127 active patents",
                "Globex Inc R&D budget: $12M for autonomous systems",
            ],
        },
        "wayne_enterprises": {
            "docs": [
                "Wayne Enterprises defence contract: night vision systems",
                "Wayne Enterprises charity gala raised $2.1M for Gotham",
                "Wayne Enterprises R&D: advanced materials division expanding",
            ],
        },
    }

    print("\n👥 Onboarding tenants...")
    for tenant, data in tenants.items():
        db.create_namespace(tenant)
        for doc in data["docs"]:
            db.add(doc, namespace=tenant, metadata={"tenant": tenant})
        print(f"  ✅ {tenant}: {len(data['docs'])} documents")

    # ─── Demonstrate namespace isolation ───
    print("\n\n🔐 Namespace Isolation Test")
    print("-" * 50)

    # Search as Acme Corp — should ONLY see Acme data
    print("\n  Acme Corp searches for 'revenue':")
    results = db.search("revenue financial results", namespace="acme_corp", top_k=5)
    for r in results:
        print(f"    [{r.namespace}] {r.text[:60]}...")
    assert all(r.namespace == "acme_corp" for r in results), "ISOLATION BREACH!"
    print("  ✅ Isolation verified — no cross-tenant leaks")

    # Search as Globex — should ONLY see Globex data
    print("\n  Globex Inc searches for 'patent':")
    results = db.search("patent intellectual property", namespace="globex_inc", top_k=5)
    for r in results:
        print(f"    [{r.namespace}] {r.text[:60]}...")
    assert all(r.namespace == "globex_inc" for r in results), "ISOLATION BREACH!"
    print("  ✅ Isolation verified — no cross-tenant leaks")

    # ─── Cross-tenant search attempt ───
    print("\n\n🛡️  Cross-Tenant Protection")
    print("-" * 50)
    print("  Wayne Enterprises searches for 'Acme revenue':")
    results = db.search("Acme revenue", namespace="wayne_enterprises", top_k=5)
    acme_leaks = [r for r in results if "acme" in r.text.lower()]
    if not acme_leaks:
        print("  ✅ No Acme data leaked to Wayne Enterprises!")
    else:
        print("  ❌ DATA LEAK DETECTED!")

    # ─── Vault stats per tenant ───
    print("\n\n📊 Per-Tenant Statistics")
    print("-" * 50)
    stats = db.stats()
    for ns, count in stats["namespace_counts"].items():
        if ns != "default":
            print(f"  {ns}: {count} documents")

    # ─── Audit log ───
    print("\n\n📋 Audit Log (last 5 events)")
    print("-" * 50)
    from vecforge.security.audit import AuditLogger
    audit = AuditLogger("saas_audit.jsonl")
    events = audit.read_log()
    for event in events[-5:]:
        print(f"  [{event['operation']}] ns={event.get('namespace', 'N/A')} "
              f"doc={str(event.get('doc_id', ''))[:8]}...")

    db.close()

    # Cleanup audit file
    import os
    if os.path.exists("saas_audit.jsonl"):
        os.remove("saas_audit.jsonl")

    print("\n✅ Multi-tenant demo complete!")
    print("VecForge by Suneel Bose K · ArcGX TechLabs")


if __name__ == "__main__":
    main()
