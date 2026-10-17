# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under BSL 1.1 — see LICENSE for details.

"""Correctness tests for the VecForge 2.0 Grover amplitude amplification.

VecForge 1.x applied the diffusion operator without an oracle, which is a
pure reflection: for candidate sets of size 2-6, 15-26, 41-58, … the
ranking was *reversed* (worst result first) and for all other sizes the
step was a no-op. These tests pin the corrected behaviour.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from vecforge.quantum.amplitude_encoder import AmplitudeEncoder
from vecforge.quantum.grover_amplifier import GroverAmplifier
from vecforge.quantum.reranker import QuantumReranker

REVERSED_SIZES_IN_1X = [2, 3, 4, 5, 6, 15, 16, 20, 26, 41, 50, 58]


class TestGroverAmplifierMath:
    def test_closed_form_matches_iterative_reference(self) -> None:
        rng = np.random.default_rng(42)
        closed = GroverAmplifier(method="closed_form")
        iterative = GroverAmplifier(method="iterative")
        for _ in range(100):
            n = int(rng.integers(2, 200))
            m = int(rng.integers(1, n))
            a = rng.random(n).astype(np.float32)
            a /= np.linalg.norm(a)
            mask = np.zeros(n, dtype=bool)
            mask[rng.choice(n, m, replace=False)] = True
            np.testing.assert_allclose(
                closed.amplify(a, marked=mask),
                iterative.amplify(a, marked=mask),
                atol=1e-5,
            )

    def test_output_is_unit_norm(self) -> None:
        amp = GroverAmplifier()
        for n in [2, 7, 33, 100]:
            a = np.random.default_rng(n).random(n).astype(np.float32)
            out = amp.amplify(a)
            assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-5

    def test_marked_probability_increases(self) -> None:
        amp = GroverAmplifier()
        rng = np.random.default_rng(7)
        for n in [4, 10, 50, 200]:
            a = rng.random(n).astype(np.float32)
            a /= np.linalg.norm(a)
            mask = GroverAmplifier.default_marked(a)
            before = amp.marked_probability(a, mask)
            after = amp.marked_probability(amp.amplify(a, marked=mask), mask)
            assert after >= before - 1e-6
            if n >= 50:
                # mass in the unmarked residual can never rotate, so the bound
                # is 1 - ||r_U||²; random starts still exceed 0.7 comfortably
                assert after > 0.7 and after > before

    def test_marked_order_is_preserved(self) -> None:
        amp = GroverAmplifier()
        rng = np.random.default_rng(3)
        for n in [5, 20, 100]:
            a = rng.random(n).astype(np.float32)
            a /= np.linalg.norm(a)
            mask = GroverAmplifier.default_marked(a, m=max(2, n // 4))
            out = amp.amplify(a, marked=mask)
            assert np.array_equal(np.argsort(out[mask]), np.argsort(a[mask]))
            assert (out[mask] > 0).all()

    @pytest.mark.parametrize("n", REVERSED_SIZES_IN_1X)
    def test_top_candidate_survives_for_sizes_reversed_in_1x(self, n: int) -> None:
        scores = np.linspace(1.0, 0.05, n).astype(np.float32)
        out = GroverAmplifier().amplify(scores)
        assert int(np.argmax(out)) == 0
        assert int(np.argmax(out**2)) == 0

    def test_optimal_iterations_uniform_start(self) -> None:
        # Classic Grover: N=4, M=1 → exactly one iteration finds the target
        assert GroverAmplifier._optimal_iterations(4, 1) == 1
        assert GroverAmplifier._optimal_iterations(1, 1) == 0
        assert GroverAmplifier._optimal_iterations(10, 10) == 0

    def test_no_signal_returns_input(self) -> None:
        amp = GroverAmplifier()
        a = np.array([0.6, 0.3, 0.1], dtype=np.float32)
        unit = a / np.linalg.norm(a)
        np.testing.assert_allclose(
            amp.amplify(a, marked=np.zeros(3, bool)), unit, atol=1e-6
        )
        np.testing.assert_allclose(
            amp.amplify(a, marked=np.ones(3, bool)), unit, atol=1e-6
        )

    def test_bad_mask_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="mask shape"):
            GroverAmplifier().amplify(
                np.ones(4, dtype=np.float32), marked=np.ones(3, bool)
            )

    def test_bad_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method"):
            GroverAmplifier(method="magic")  # type: ignore[arg-type]

    def test_explicit_iterations_override(self) -> None:
        a = np.array([0.9, 0.2, 0.1, 0.1], dtype=np.float32)
        a /= np.linalg.norm(a)
        mask = np.array([True, False, False, False])
        zero = GroverAmplifier().amplify(a, marked=mask, iterations=0)
        np.testing.assert_allclose(zero, a, atol=1e-6)
        fixed = GroverAmplifier(max_iterations=0).amplify(a, marked=mask)
        np.testing.assert_allclose(fixed, a, atol=1e-6)


class TestAmplitudeEncoder:
    def test_negative_scores_are_shifted_not_flipped(self) -> None:
        enc = AmplitudeEncoder()
        out = enc.encode(np.array([-2.0, 0.0, 2.0], dtype=np.float32))
        assert out[0] == 0.0
        assert out[2] > out[1] > out[0]

    def test_probabilities_sum_to_one(self) -> None:
        p = AmplitudeEncoder.probabilities(np.array([0.6, -0.8], dtype=np.float32))
        assert abs(float(p.sum()) - 1.0) < 1e-6
        assert p[1] > p[0]

    def test_nan_scores_are_sanitised(self) -> None:
        out = AmplitudeEncoder().encode(np.array([np.nan, 1.0], dtype=np.float32))
        assert not np.any(np.isnan(out))

    def test_empty_input(self) -> None:
        assert AmplitudeEncoder().encode(np.zeros(0, dtype=np.float32)).shape == (0,)


class TestQuantumRerankerBehaviour:
    def test_default_oracle_keeps_order_and_sharpens(self) -> None:
        qr = QuantumReranker()
        scores = [0.9, 0.7, 0.5, 0.3, 0.1]
        res = qr.rerank("q", ["a", "b", "c", "d", "e"], scores, top_k=5)
        assert [i for i, _ in res] == [0, 1, 2, 3, 4]
        new = [s for _, s in res]
        assert new[0] == pytest.approx(1.0)
        assert new == sorted(new, reverse=True)
        assert all(0.0 <= s <= 1.0 for s in new)
        # the gap between best and worst widens relative to the input
        assert (new[0] - new[-1]) / new[0] > (scores[0] - scores[-1]) / scores[0]

    @pytest.mark.parametrize("n", REVERSED_SIZES_IN_1X)
    def test_regression_1x_reversal(self, n: int) -> None:
        scores = list(np.linspace(1.0, 0.05, n))
        res = QuantumReranker().rerank("q", [str(i) for i in range(n)], scores, top_k=3)
        assert res[0][0] == 0

    def test_consensus_oracle_lifts_marked_candidates(self) -> None:
        qr = QuantumReranker()
        n = 20
        scores = list(np.linspace(1.0, 0.05, n))
        res = qr.rerank(
            "q", [str(i) for i in range(n)], scores, top_k=5, marked=[3, 7, 12]
        )
        top = [i for i, _ in res]
        assert top[:3] == [3, 7, 12]  # marked, in original relative order
        assert top[3:] == [0, 1]  # then the best unmarked, original order
        assert qr.last_explanation["marked"] == 3
        assert (
            qr.last_explanation["marked_probability_after"]
            > qr.last_explanation["marked_probability_before"]
        )

    def test_boolean_mask_oracle(self) -> None:
        mask = np.zeros(6, dtype=bool)
        mask[4] = True
        res = QuantumReranker().rerank(
            "q", list("abcdef"), [6, 5, 4, 3, 2, 1], top_k=2, marked=mask
        )
        assert res[0][0] == 4

    def test_oracle_with_no_signal_falls_back(self) -> None:
        res = QuantumReranker().rerank(
            "q", list("abcd"), [0.4, 0.3, 0.2, 0.1], top_k=2, marked=[]
        )
        assert [i for i, _ in res] == [0, 1]

    def test_windowing_keeps_indices_valid_and_is_fast(self) -> None:
        n = 1_000_000
        scores = np.random.default_rng(1).random(n).astype(np.float32)
        qr = QuantumReranker(max_candidates=1000)
        t0 = time.perf_counter()
        idx, new_scores, marked = qr.amplify_scores(scores, top_k=10)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert len(idx) == 1000
        assert int(idx[0]) == int(np.argmax(scores))
        assert new_scores[0] == pytest.approx(1.0)
        assert elapsed_ms < 2000
        assert qr.last_explanation["window"] == 1000

    def test_cross_encoder_runs_only_on_survivors(self) -> None:
        class FakeReranker:
            def __init__(self) -> None:
                self.calls: list[list[str]] = []

            def rerank(self, query: str, texts: list[str], top_k: int | None = None):
                self.calls.append(list(texts))
                ranked = sorted(
                    range(len(texts)), key=lambda i: len(texts[i]), reverse=True
                )
                return [(i, float(len(texts[i]))) for i in ranked][:top_k]

        fake = FakeReranker()
        qr = QuantumReranker(classical_reranker=fake)
        texts = [f"{'x' * (i + 1)}" for i in range(30)]
        scores = list(np.linspace(1.0, 0.1, 30))
        res = qr.rerank("q", texts, scores, top_k=3, marked=[2, 5])
        assert len(fake.calls) == 1
        assert len(fake.calls[0]) == 3  # 2 marked survivors + 1 filled to top_k
        assert len(res) == 3

    def test_top_k_zero_and_empty(self) -> None:
        qr = QuantumReranker()
        assert qr.rerank("q", [], [], top_k=5) == []
        assert qr.rerank("q", ["a"], [1.0], top_k=0) == []

    def test_sqrt_default_oracle_size(self) -> None:
        qr = QuantumReranker()
        qr.amplify_scores(np.linspace(1, 0, 100).astype(np.float32), top_k=3)
        assert qr.last_explanation["marked"] == math.ceil(math.sqrt(100))
