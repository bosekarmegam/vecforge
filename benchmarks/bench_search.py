# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
VecForge benchmark suite.

Measures search latency and ingestion throughput at varying collection
sizes to verify North Star performance targets.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

Usage:
    python -m benchmarks.bench_search
"""

from __future__ import annotations

import statistics
import time

import numpy as np

from vecforge.core.bm25 import BM25Engine
from vecforge.core.indexer import FaissIndexer

DIMENSION = 384


def _random_vecs(n: int, d: int = DIMENSION) -> np.ndarray:
    """Generate random float32 vectors."""
    vecs = np.random.randn(n, d).astype(np.float32)
    # why: Normalise for inner-product search
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-8)


def bench_faiss_search(n_docs: int, n_queries: int = 100, top_k: int = 10) -> dict:
    """Benchmark FAISS search at given collection size."""
    print(f"\n{'─' * 50}")
    print(f"FAISS Search Benchmark — {n_docs:,} docs, {n_queries} queries, top_k={top_k}")
    print(f"{'─' * 50}")

    vecs = _random_vecs(n_docs)
    indexer = FaissIndexer(dimension=DIMENSION)
    indexer.add(vecs)

    queries = _random_vecs(n_queries)
    latencies: list[float] = []

    for q in queries:
        start = time.perf_counter()
        indexer.search(q, top_k=top_k)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(0.95 * len(latencies))]
    p99 = sorted(latencies)[int(0.99 * len(latencies))]
    mean = statistics.mean(latencies)

    print(f"  Mean:  {mean:.2f}ms")
    print(f"  p50:   {p50:.2f}ms")
    print(f"  p95:   {p95:.2f}ms")
    print(f"  p99:   {p99:.2f}ms")

    return {"n_docs": n_docs, "mean_ms": mean, "p50_ms": p50, "p95_ms": p95, "p99_ms": p99}


def bench_faiss_ingest(n_docs: int) -> dict:
    """Benchmark FAISS ingestion throughput."""
    print(f"\n{'─' * 50}")
    print(f"FAISS Ingest Benchmark — {n_docs:,} docs")
    print(f"{'─' * 50}")

    vecs = _random_vecs(n_docs)
    indexer = FaissIndexer(dimension=DIMENSION)

    start = time.perf_counter()
    indexer.add(vecs)
    elapsed = time.perf_counter() - start

    throughput = n_docs / elapsed

    print(f"  Time:       {elapsed:.3f}s")
    print(f"  Throughput: {throughput:,.0f} docs/sec")

    return {"n_docs": n_docs, "time_s": elapsed, "throughput": throughput}


def bench_bm25_search(n_docs: int, n_queries: int = 100, top_k: int = 10) -> dict:
    """Benchmark BM25 search at given collection size."""
    print(f"\n{'─' * 50}")
    print(f"BM25 Search Benchmark — {n_docs:,} docs, {n_queries} queries")
    print(f"{'─' * 50}")

    bm25 = BM25Engine()
    # why: Generate fake docs of ~50 words each
    words = ["patient", "diabetes", "emergency", "fracture", "cardiac",
             "hospital", "treatment", "admission", "discharged", "medication"]

    docs = [" ".join(np.random.choice(words, size=50).tolist()) for _ in range(n_docs)]
    bm25.add_documents(docs)

    queries_list = [
        "diabetic patient treatment",
        "cardiac emergency hospital",
        "fracture admission medication",
    ] * (n_queries // 3 + 1)

    latencies: list[float] = []
    for q in queries_list[:n_queries]:
        start = time.perf_counter()
        bm25.search(q, top_k=top_k)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    p50 = statistics.median(latencies)
    mean = statistics.mean(latencies)

    print(f"  Mean: {mean:.2f}ms")
    print(f"  p50:  {p50:.2f}ms")

    return {"n_docs": n_docs, "mean_ms": mean, "p50_ms": p50}


def run_all() -> None:
    """Run all benchmarks and print summary."""
    print("=" * 60)
    print("VecForge Benchmark Suite")
    print("Built by Suneel Bose K · ArcGX TechLabs Private Limited")
    print("=" * 60)

    results = []

    # North Star targets:
    #   Search 100k docs: <15ms
    #   Ingest 1000 PDFs: <5min
    #   Encrypt + search: <20ms overhead

    for n in [1_000, 10_000, 100_000]:
        results.append(bench_faiss_search(n))

    for n in [1_000, 10_000, 100_000]:
        bench_faiss_ingest(n)

    for n in [1_000, 10_000]:
        bench_bm25_search(n)

    # ─── Summary ───
    print(f"\n{'═' * 60}")
    print("Summary vs North Star Targets")
    print(f"{'═' * 60}")

    for r in results:
        target = "< 15ms" if r["n_docs"] >= 100_000 else "—"
        status = "✅" if r["p50_ms"] < 15 else "⚠️"
        print(f"  {r['n_docs']:>8,} docs: p50={r['p50_ms']:.2f}ms  target={target}  {status}")


if __name__ == "__main__":
    run_all()
