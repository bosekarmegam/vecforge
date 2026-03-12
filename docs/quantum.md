# Quantum-Inspired Acceleration

VecForge Phase 3 adds **quantum-inspired algorithms** that run on ordinary CPUs — no quantum computer required. These are classical implementations of mathematical techniques borrowed from quantum computing.

## Key Concepts

### Amplitude Encoding

Maps a classical score vector into **quantum amplitude space** by L2-normalizing it:

```
|ψ⟩ = s / ‖s‖₂
```

Inner products between amplitude vectors correspond to quantum fidelity (state overlap), providing a more stable distance metric across different score distributions.

### Grover's Algorithm (Classical Simulation)

[Grover's algorithm](https://en.wikipedia.org/wiki/Grover%27s_algorithm) finds a marked item in an unsorted list in **O(√N)** steps vs O(N) classically. The core operation — the **diffusion operator** (inversion-about-mean) — is purely classical:

```
s'_i = 2μ - s_i     where μ = mean(s)
```

Iterating this `k = π√N/4` times amplifies high-relevance candidates and suppresses low-relevance ones, exponentially widening the score gap.

### QuantumReranker

Combines the two above to efficiently rerank search candidates:

1. **Amplitude encoding** — normalise hybrid fusion scores into amplitude space
2. **Grover amplification** — diffuse scores to amplify top candidates
3. **√N pre-selection** — use `np.argpartition` to identify top √N survivors in O(N)
4. **Optional cross-encoder** — run cross-encoder on √N survivors only (not all N)

## Performance

| N (docs) | Grover (ms p50) | QRerank (ms p50) | Target |
|---:|---:|---:|---:|
| 1,000 | <0.1 | <0.5 | — |
| 10,000 | <0.5 | <1 | — |
| 100,000 | <2 | <5 | — |
| **1,000,000** | **<8** | **<20** | **✅ <20ms** |

## Usage

### Basic quantum reranking

```python
from vecforge import VecForge

db = VecForge("my_vault")
db.add("NASA's Artemis aims to return humans to the Moon")
db.add("Authentic Italian pizza requires wood-fired oven")
db.add("The quick brown fox jumps over the lazy dog")

# Enable Grover-inspired reranking
results = db.search("space exploration", quantum_rerank=True)
for r in results:
    print(f"{r.score:.3f}  →  {r.text}")
```

### Combining quantum rerank with cross-encoder

```python
# quantum_rerank=True + rerank=True runs cross-encoder on only √N survivors
results = db.search(
    "elderly diabetic hip fracture",
    rerank=True,           # cross-encoder precision
    quantum_rerank=True,   # Grover pre-selection limits cross-encoder to √N calls
    top_k=5,
)
```

### Using QuantumReranker directly

```python
from vecforge.quantum import QuantumReranker

qr = QuantumReranker()
results = qr.rerank(
    query="space exploration",
    texts=["NASA article", "Pizza recipe", "Fox story"],
    scores=[0.9, 0.2, 0.05],
    top_k=3,
)
for idx, score in results:
    print(f"[{idx}] score={score:.4f}")
```

## Optional Qiskit Extra

For genuine quantum hybrid experiments (requires quantum hardware or simulator):

```bash
pip install vecforge[quantum]
```

This installs:
- `qiskit>=1.0.0` — IBM Quantum circuit framework
- `qiskit-aer>=0.14.0` — Local quantum circuit simulator

> **Note:** The core quantum-inspired features (AmplitudeEncoder, GroverAmplifier, QuantumReranker) only require NumPy and are **always available** without the [quantum] extra.

## API Reference

### `AmplitudeEncoder`

```python
from vecforge.quantum import AmplitudeEncoder

encoder = AmplitudeEncoder()
amplitudes = encoder.encode(scores)    # NDArray[float32] → unit-norm
recovered  = encoder.decode(amplitudes, original_norm)
```

### `GroverAmplifier`

```python
from vecforge.quantum import GroverAmplifier

amp = GroverAmplifier(max_iterations=None)  # None = auto-optimal
amplified = amp.amplify(scores, iterations=None)
```

### `QuantumReranker`

```python
from vecforge.quantum import QuantumReranker
from vecforge.core.reranker import Reranker

qr = QuantumReranker(
    classical_reranker=Reranker(),  # optional cross-encoder for √N candidates
    grover_iterations=None,         # auto-optimal
)
results = qr.rerank(query, texts, scores, top_k=10)
# → list[tuple[original_index, amplified_score]]
```

---

Built by Suneel Bose K · ArcGX TechLabs Private Limited
