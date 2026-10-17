# 🌀 Quantum-Inspired Reranking

VecForge's `quantum_rerank=True` runs a classical simulation of **Grover
amplitude amplification** over the search candidates. No quantum hardware
is involved; the algorithm is linear algebra on NumPy and, since 2.0, runs
in closed form in **O(N)**.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

---

## What changed in 2.0

VecForge 1.x applied Grover's *diffusion* operator without an *oracle*.
Diffusion alone, `a' = 2·mean(a) − a`, is a reflection: it preserves the
mean, so applying it twice is the identity and applying it once **reverses**
the ranking. The 1.x iteration count `k = ⌊π / (4·asin(1/√N))⌋` is odd for
roughly half of all candidate-set sizes (2–6, 15–26, 41–58, …), so
`quantum_rerank=True` returned the worst candidate first for those sizes and
did nothing for the rest. 2.0 fixes the algorithm and adds regression tests
for every size the old code reversed.

---

## The algorithm

1. **Amplitude encoding.** Candidate scores `s` are L2-normalised to a
   unit vector `|ψ⟩ = s / ‖s‖`. Squared amplitudes are probabilities.
2. **Oracle.** A boolean mask marks the candidates to boost. In
   `VecForge.search` the oracle is *consensus*: candidates returned by
   **both** the dense retriever and BM25. When there is no consensus the
   top `√K` by score are marked.
3. **Grover iterate.** `G = D·O` where `O` flips the sign of marked
   amplitudes and `D = 2|u⟩⟨u| − I` is inversion about the mean. Each
   iterate rotates the state toward the marked subspace by `2θ`, with
   `sin θ = √(M/N)`.
4. **Closed form.** The state decomposes into the uniform components over
   the marked and unmarked sets (a 2-D rotation) plus residuals that `G`
   leaves unchanged (marked) or negates (unmarked). The optimal number of
   iterations from the initial angle `φ₀` is `round((π/2 − φ₀) / 2θ)` and
   the result is evaluated analytically — no loop over `√N` iterations.
5. **Scores.** Marked candidates get `p_i = (a_i + c)²`, the same shift
   for every marked item, so their relative order is **preserved** while
   their probability mass grows. Unmarked candidates share the residual
   mass in proportion to their original probabilities. Scores are
   max-normalised so the best candidate is `1.0`.
6. **Optional cross-encoder.** With `rerank=True` as well, the cross-encoder
   runs only on the marked survivors (O(M) model calls instead of O(N)).

Guarantees (covered by `tests/unit/test_quantum_correctness.py`):

* closed form equals the iterative reference to float precision;
* the best candidate stays first for every candidate-set size;
* marked candidates move above unmarked ones in their original order;
* the probability mass on the marked set never decreases.

---

## Usage

```python
from vecforge import VecForge

db = VecForge("my_vault")
db.add("NASA's Artemis aims to return humans to the Moon")
db.add("Authentic Italian pizza requires a wood-fired oven")
db.add("The quick brown fox jumps over the lazy dog")

results = db.search("space exploration", quantum_rerank=True)
for r in results:
    print(f"{r.score:.3f}  {r.signals}  {r.text}")
```

`VecForge(quantum=True)` makes it the default for every search.

Combine with the cross-encoder:

```python
results = db.search(
    "elderly diabetic hip fracture",
    rerank=True,          # cross-encoder on the marked survivors only
    quantum_rerank=True,  # Grover selects the survivors
    top_k=5,
)
```

Direct use with your own oracle:

```python
from vecforge.quantum import QuantumReranker

qr = QuantumReranker(max_candidates=1000)
ranked = qr.rerank(
    query="space",
    texts=["NASA article", "Pizza recipe", "Fox story", "Moon landing"],
    scores=[0.9, 0.5, 0.4, 0.3],
    top_k=4,
    marked=[0, 3],            # oracle: indices (or a boolean mask)
)
# → [(0, 1.0), (3, 0.7…), (1, …), (2, …)]
print(qr.last_explanation)  # marked probability before/after, window size
```

Lower-level pieces:

```python
from vecforge.quantum import AmplitudeEncoder, GroverAmplifier

amps = AmplitudeEncoder().encode(scores)
mask = GroverAmplifier.default_marked(amps, m=5)
out = GroverAmplifier().amplify(amps, marked=mask)      # closed form, O(N)
ref = GroverAmplifier(method="iterative").amplify(amps, marked=mask)
probs = AmplitudeEncoder.probabilities(out)
```

---

## Performance

`python -m benchmarks.bench_quantum` on a 4-core CPU container (p50):

| Candidates | Encode | Grover closed form | Grover iterative (reference) | Full rerank, window 1000 |
|---:|---:|---:|---:|---:|
| 1,000 | 0.005 ms | 0.03 ms | 0.03 ms | 0.11 ms |
| 10,000 | 0.02 ms | 0.07 ms | 0.11 ms | 0.13 ms |
| 100,000 | 0.2 ms | 2.0 ms | 2.2 ms | 0.29 ms |
| 1,000,000 | — | — | — | 1.7 ms |

With the default `√K` oracle the optimal iteration count is small, so the
iterative reference is nearly as fast as the closed form; the closed form
matters when a strong oracle marks few candidates in a large window (its
cost never depends on `k`).

The full reranker bounds the amplification stage with a 1000-candidate
window selected by an O(N) `argpartition`, so cost at 1M candidates is
dominated by that single pass. Inside `VecForge.search` the reranker only
sees the (at most `4·top_k`, widened for filters) candidates the cascade
produced, so its cost is well under a millisecond.

---

## Optional Qiskit extra

```bash
pip install "vecforge[quantum]"
```

Installs `qiskit` and `qiskit-aer` for genuine circuit experiments. The
built-in reranker never needs them.
