# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under the Business Source License 1.1 (BSL 1.1)

"""
VecForge Quantum-Inspired Acceleration Module.

Provides classical implementations of quantum-inspired algorithms for
ultra-fast reranking. No quantum hardware required — all algorithms
run on standard NumPy/SciPy on any CPU.

Components:
    AmplitudeEncoder — Maps scores into quantum amplitude space (L2-norm)
    GroverAmplifier  — Grover-inspired score amplification (O(√N) effective)
    QuantumReranker  — Drop-in reranker using the above for <20ms at 1M docs

Usage::

    from vecforge.quantum import QuantumReranker

    reranker = QuantumReranker()
    results = reranker.rerank(
        query="diabetes", texts=candidates, scores=scores, top_k=10
    )

Or via the VecForge search API::

    results = db.search("query", quantum_rerank=True)

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from vecforge.quantum.amplitude_encoder import AmplitudeEncoder
from vecforge.quantum.grover_amplifier import GroverAmplifier
from vecforge.quantum.reranker import QuantumReranker

__all__ = [
    "AmplitudeEncoder",
    "GroverAmplifier",
    "QuantumReranker",
]
