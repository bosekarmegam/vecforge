# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under the Business Source License 1.1 (BSL 1.1)

"""
Quantum-inspired reranking benchmark (VecForge 2.0).

Measures:
  1. AmplitudeEncoder.encode at several sizes
  2. GroverAmplifier — closed form (O(N), production) vs iterative reference
  3. QuantumReranker end to end with the default 1000-candidate window,
     including the O(N) pre-filter at 1M candidates
  4. Ranking quality: the amplifier must keep the best candidate first and
     move oracle-marked candidates to the top (the 1.x reranker reversed
     rankings for roughly half of all candidate-set sizes)

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


def _p50(fn, *args, runs: int = RUNS) -> float:
    fn(*args)  # warmup
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        latencies.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(latencies, 50))


def bench_amplitude_encode(n: int) -> float:
    encoder = AmplitudeEncoder()
    scores = np.random.rand(n).astype(np.float32)
    return _p50(encoder.encode, scores)


def bench_grover(n: int, method: str) -> float:
    amp = GroverAmplifier(method=method)  # type: ignore[arg-type]
    a = np.random.rand(n).astype(np.float32)
    a /= np.linalg.norm(a)
    mask = GroverAmplifier.default_marked(a)
    return _p50(
        lambda: amp.amplify(a, marked=mask), runs=5 if method == "iterative" else RUNS
    )


def bench_reranker(n: int, top_k: int = 10) -> float:
    qr = QuantumReranker(max_candidates=1000)
    scores = np.random.rand(n).astype(np.float32)
    return _p50(lambda: qr.amplify_scores(scores, top_k=top_k))


def quality_check() -> None:
    qr = QuantumReranker()
    reversed_in_1x = [2, 3, 4, 5, 6, 15, 20, 26, 41, 58]
    ok = True
    for n in reversed_in_1x:
        scores = list(np.linspace(1.0, 0.05, n))
        res = qr.rerank("q", [str(i) for i in range(n)], scores, top_k=3)
        ok &= res[0][0] == 0
    scores = list(np.linspace(1.0, 0.05, 20))
    res = qr.rerank("q", [str(i) for i in range(20)], scores, top_k=5, marked=[3, 7])
    ok &= [i for i, _ in res][:2] == [3, 7]
    status = "✅" if ok else "❌"
    print(f"\n🧪 Ranking quality: best-first for N∈{reversed_in_1x} → {status}")
    print(f"   consensus oracle [3, 7] → top-2 = {[i for i, _ in res][:2]}")
    print(f"   {qr.last_explanation}")


def main() -> None:
    print("VecForge 2.0 — Quantum-Inspired Reranking Benchmark")
    print("=" * 72)
    print("\n📊 Component micro-benchmarks (ms p50):")
    header = f"{'N':>10}  {'Encode':>8}  {'Grover closed-form':>18}  "
    print(header + f"{'Grover iterative':>16}")
    print("-" * 60)
    for n in SIZES[:3]:
        enc = bench_amplitude_encode(n)
        closed = bench_grover(n, "closed_form")
        iterative = bench_grover(n, "iterative")
        print(f"{n:>10,}  {enc:>8.3f}  {closed:>18.3f}  {iterative:>16.3f}")

    print("\n⚡ QuantumReranker (window=1000, O(N) pre-filter + O(K) closed form):")
    print(f"{'N':>10}  {'ms p50':>8}  Status")
    print("-" * 32)
    for n in SIZES:
        ms = bench_reranker(n)
        print(f"{n:>10,}  {ms:>8.3f}  {'✅' if ms <= 20.0 else '⚠️'}")

    quality_check()
    print("\nBuilt by Suneel Bose K · ArcGX TechLabs Private Limited")


if __name__ == "__main__":
    main()
