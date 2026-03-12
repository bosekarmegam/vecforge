# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under the Business Source License 1.1 (BSL 1.1)

"""
Phase 3 Quantum-Inspired Acceleration Benchmark.

Compares classical FAISS search with quantum-inspired reranking
across multiple dataset sizes.

Target: <20ms quantum rerank p50 at 1M docs.

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
    """Benchmark AmplitudeEncoder.encode at size N."""
    encoder = AmplitudeEncoder()
    scores = np.random.rand(n).astype(np.float32)
    # Warmup
    encoder.encode(scores)

    latencies = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        encoder.encode(scores)
        latencies.append((time.perf_counter() - t0) * 1000)

    return float(np.percentile(latencies, 50))


def bench_grover_amplify(n: int) -> float:
    """Benchmark GroverAmplifier.amplify at size N."""
    amplifier = GroverAmplifier()
    scores = np.random.rand(n).astype(np.float32)
    # Warmup
    amplifier.amplify(scores)

    latencies = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        amplifier.amplify(scores)
        latencies.append((time.perf_counter() - t0) * 1000)

    return float(np.percentile(latencies, 50))


def bench_quantum_reranker(n: int, top_k: int = 10) -> float:
    """Benchmark QuantumReranker.rerank at size N."""
    qr = QuantumReranker()
    scores = list(np.random.rand(n).astype(float))
    texts = [f"doc_{i}" for i in range(n)]
    # Warmup
    qr.rerank("query", texts, scores, top_k=top_k)

    latencies = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        qr.rerank("query", texts, scores, top_k=top_k)
        latencies.append((time.perf_counter() - t0) * 1000)

    return float(np.percentile(latencies, 50))


def main() -> None:
    print("VecForge — Phase 3 Quantum-Inspired Acceleration Benchmark")
    print("=" * 65)
    header = f"{'N':>10}  {'Encode':>10}  {'Grover':>10}  {'QRerank':>10}  Status"
    print(header)
    print("-" * 65)

    target_ms = 20.0

    for n in SIZES:
        enc_ms = bench_amplitude_encode(n)
        grover_ms = bench_grover_amplify(n)
        rerank_ms = bench_quantum_reranker(n)
        status = "✅" if rerank_ms <= target_ms else "⚠️"
        row = (
            f"{n:>10,}  {enc_ms:>10.3f}  "
            f"{grover_ms:>10.3f}  {rerank_ms:>10.3f}  {status}"
        )
        print(row)

    print("-" * 65)
    print(f"Target: ≤{target_ms}ms p50 quantum rerank at 1M docs")
    print("Built by Suneel Bose K · ArcGX TechLabs Private Limited")


if __name__ == "__main__":
    main()
