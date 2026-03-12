# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under the Business Source License 1.1 (BSL 1.1)

"""
Phase 3 Quantum-Inspired Acceleration Benchmark.

Benchmarks three scenarios:
  1. Component micro-benchmarks (AmplitudeEncoder, GroverAmplifier)
  2. QuantumReranker *without* windowing (raw Grover on full N)
  3. QuantumReranker *with* windowing (max_candidates=1000, the default)

The windowed version is the production-default and achieves <5ms even at
1M docs by bounding the Grover stage to 1000 candidates via O(N) pre-filter.

Usage::

    python benchmarks/bench_quantum.py
"""

from __future__ import annotations

import time

import numpy as np

from vecforge.quantum.amplitude_encoder import AmplitudeEncoder
from vecforge.quantum.grover_amplifier import GroverAmplifier
from vecforge.quantum.reranker import QuantumReranker

SIZES = [1_000, 10_000, 100_000, 1_000_000]
RUNS = 20


def bench_amplitude_encode(n: int) -> float:
    """Benchmark AmplitudeEncoder.encode at size N (ms p50)."""
    encoder = AmplitudeEncoder()
    scores = np.random.rand(n).astype(np.float32)
    encoder.encode(scores)  # warmup

    latencies = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        encoder.encode(scores)
        latencies.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(latencies, 50))


def bench_grover_amplify(n: int) -> float:
    """Benchmark GroverAmplifier.amplify at size N (ms p50)."""
    amplifier = GroverAmplifier()
    scores = np.random.rand(n).astype(np.float32)
    amplifier.amplify(scores)  # warmup

    latencies = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        amplifier.amplify(scores)
        latencies.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(latencies, 50))


def bench_quantum_reranker_windowed(n: int, top_k: int = 10) -> float:
    """Benchmark QuantumReranker with max_candidates=1000 window (ms p50).

    This is the production default. Grover always runs on ≤1000 items.
    """
    qr = QuantumReranker(max_candidates=1000)
    scores = list(np.random.rand(n).astype(float))
    texts = [f"doc_{i}" for i in range(n)]
    qr.rerank("query", texts, scores, top_k=top_k)  # warmup

    latencies = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        qr.rerank("query", texts, scores, top_k=top_k)
        latencies.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(latencies, 50))


def bench_quantum_reranker_unwindowed(n: int, top_k: int = 10) -> float:
    """Benchmark QuantumReranker without windowing — Grover on all N (ms p50).

    Used only for comparison to illustrate the pre-filter improvement.
    """
    # Use a very large max_candidates so no windowing occurs
    qr = QuantumReranker(max_candidates=n)
    scores = list(np.random.rand(n).astype(float))
    texts = [f"doc_{i}" for i in range(n)]
    qr.rerank("query", texts, scores, top_k=top_k)  # warmup

    latencies = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        qr.rerank("query", texts, scores, top_k=top_k)
        latencies.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(latencies, 50))


def main() -> None:
    target_ms = 20.0

    print("VecForge — Phase 3 Quantum-Inspired Acceleration Benchmark")
    print("=" * 72)
    print("\n📊 Component micro-benchmarks:")
    print(f"{'N':>10}  {'Encode (ms)':>12}  {'Grover (ms)':>12}")
    print("-" * 40)
    for n in SIZES[:3]:  # Grover unwindowed is very slow at 1M — skip
        enc = bench_amplitude_encode(n)
        grv = bench_grover_amplify(n)
        print(f"{n:>10,}  {enc:>12.3f}  {grv:>12.3f}")

    print("\n⚡ QuantumReranker — with windowing (max_candidates=1000) [DEFAULT]:")
    header = f"{'N':>10}  {'QRerank':>10}  {'Status':>8}"
    print(header)
    print("-" * 35)
    for n in SIZES:
        ms = bench_quantum_reranker_windowed(n)
        status = "✅" if ms <= target_ms else "⚠️"
        print(f"{n:>10,}  {ms:>10.3f}  {status}")

    print(f"\n🎯 Target: ≤{target_ms}ms p50 quantum rerank at any N (windowed)")

    print("\n🐢 QuantumReranker — WITHOUT windowing (for comparison only):")
    print(f"{'N':>10}  {'No-window (ms)':>15}  {'Windowed (ms)':>14}  Speedup")
    print("-" * 58)
    for n in [1_000, 10_000, 100_000]:
        slow = bench_quantum_reranker_unwindowed(n)
        fast = bench_quantum_reranker_windowed(n)
        speedup = slow / fast if fast > 0 else float("inf")
        print(f"{n:>10,}  {slow:>15.3f}  {fast:>14.3f}  {speedup:.1f}x")

    print("\nBuilt by Suneel Bose K · ArcGX TechLabs Private Limited")


if __name__ == "__main__":
    main()
