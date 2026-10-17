<p align="center">
  <img src="https://raw.githubusercontent.com/bosekarmegam/vecforge/main/assets/brand/vecforge-mark.svg" alt="VecForge logo" width="120" height="120">
  <h1 align="center">VecForge</h1>
  <p align="center"><strong>Forge your vector database. Own it forever.</strong></p>
  <p align="center">
    Local-first · Encrypted · Hybrid Search · AI Memory over MCP · Zero Cloud Dependency
  </p>
</p>

---

**VecForge** is a universal, local-first Python vector database with enterprise security, multimodal ingestion, quantum-inspired reranking, and — new in 2.0 — a **persistent memory layer for AI assistants** that Claude Desktop, Claude Code and any MCP client can use to remember everything between sessions.

Built by **Suneel Bose K** — Founder & CEO, [ArcGX TechLabs Private Limited](https://arcgx.in)

[![PyPI version](https://img.shields.io/pypi/v/vecforge.svg)](https://pypi.org/project/vecforge/)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10--3.13-green.svg)](https://python.org)
[![Tests](https://github.com/bosekarmegam/vecforge/actions/workflows/tests.yml/badge.svg)](https://github.com/bosekarmegam/vecforge/actions/workflows/tests.yml)
[![Coverage](https://img.shields.io/badge/Coverage-90%25-brightgreen.svg)](#-benchmarks)
[![Ruff](https://img.shields.io/badge/Linting-Ruff%20✅-brightgreen.svg)](https://github.com/astral-sh/ruff)
[![MCP](https://img.shields.io/badge/MCP-Claude%20ready-blueviolet.svg)](#-ai-memory-for-claude-and-other-agents)
[![Quantum](https://img.shields.io/badge/Quantum%20Rerank-1.7ms%20@%201M-blueviolet.svg)](#-quantum-inspired-search)

---

## ⚡ 5-Line Quickstart

```python
from vecforge import VecForge

db = VecForge("my_vault")
db.add("Patient admitted with type 2 diabetes", metadata={"ward": "7"})
results = db.search("diabetic patient")
print(results[0].text)
```

That's it. No API keys. No cloud. No config files. **Your data stays on your machine**, and the vault reopens in milliseconds next time.

> Offline, in CI, or on a small machine? `VecForge("my_vault", model_name="hash")` needs no model download at all.

---

## 🧠 AI Memory for Claude and other agents

VecForge 2.0 gives any assistant a long-term memory: facts, preferences, decisions and session summaries that survive the end of a conversation and come back ranked by relevance, importance and recency.

```python
from vecforge import MemoryStore

mem = MemoryStore("~/.vecforge/memory.db", agent="claude")
mem.remember("User prefers concise answers without em-dashes", kind="preference", importance=0.9)
mem.remember("Deploys happen on Fridays", kind="fact", tags=["ops"])

mem.start_session()
print(mem.context_pack("how should I answer?"))   # prompt-ready briefing
mem.end_session("Reviewed the deployment plan")   # remembered for next time
```

Expose it to Claude over the **Model Context Protocol** — nothing leaves your machine:

```bash
pip install "vecforge[mcp]"
vecforge mcp --print-config claude-desktop     # paste into claude_desktop_config.json
vecforge mcp --print-config claude-code > .mcp.json
```

Tools: `context_pack`, `remember`, `recall`, `forget`, `reinforce`, `list_memories`, `start_session`, `end_session`, `recent_sessions`, `search_vault`, `add_document`, `vault_stats`. Near-duplicates reinforce existing memories instead of piling up, the embedding model warms up in the background, and the index snapshot is refreshed every few writes so the next session opens instantly.

> 📖 [Memory Guide](docs/memory.md) · [MCP Guide](docs/mcp.md) · [`examples/ai_memory_agent.py`](examples/ai_memory_agent.py)

---

## 🔥 Why VecForge?

| Feature | Pinecone | ChromaDB | **VecForge** |
|---|---|---|---|
| Local-first | ❌ Cloud-only | ✅ | ✅ **Always** |
| Encryption at rest | ❌ | ❌ | ✅ **AES-256** |
| Hybrid search | ✅ | ❌ | ✅ **Dense + BM25, explainable signals** |
| AI memory + MCP server | ❌ | ❌ | ✅ **Built-in** |
| Quantum reranking | ❌ | ❌ | ✅ **Grover, closed form** |
| Multimodal search | ❌ | ❌ | ✅ **CLIP + Whisper** |
| Namespace isolation | ✅ Cloud | ❌ | ✅ **Inside the index** |
| RBAC + audit log | ✅ Cloud | ❌ | ✅ **Built-in** |
| Pluggable embedders | ❌ | ✅ | ✅ **Any model, any API, or none** |
| Price | $$$$ | Free | ✅ **Free** |

---

## 📦 Install

```bash
pip install vecforge                 # core
pip install "vecforge[mcp]"          # + MCP server for Claude
pip install "vecforge[encryption]"   # + SQLCipher AES-256
pip install "vecforge[multimodal]"   # + CLIP images, Whisper audio
```

From source: `git clone https://github.com/bosekarmegam/vecforge.git && cd vecforge && pip install -e ".[dev]"`

> **Windows users:** PyTorch needs the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe). See the [Installation Guide](docs/installation.md).

---

## 🔍 Hybrid Search

```python
results = db.search(
    "elderly diabetic hip fracture",
    top_k=5,
    alpha=0.7,                     # 70% semantic, 30% keyword
    rerank=True,                   # cross-encoder precision boost
    quantum_rerank=True,           # Grover amplification of consensus hits
    namespace="ward_7",            # enforced inside FAISS + BM25
    filters={"year": {"gte": 2023}, "tags": {"any": ["ortho", "geriatric"]}},
)
for r in results:
    print(r.score, r.signals, r.dense_score, r.sparse_score, r.text)
```

Namespace scoping and deletes are applied **inside** the retrievers, so a namespace with ten documents in a vault of a million returns all ten; metadata filters widen the candidate window automatically until `top_k` matches are found.

More: `db.add_batch`, `db.upsert`, `db.get`, `db.update_metadata`, `db.similar(doc_id)`, `db.search_batch`, `db.delete_many`, `db.delete_namespace`, `db.compact`, `db.backup`.

> 📖 [Search Guide](docs/search.md) · [API Reference](docs/api_reference.md)

---

## 🔁 Adapts to any AI stack

```python
from vecforge import VecForge, CallableEmbedder, HashEmbedder

VecForge("v.db", model_name="all-mpnet-base-v2")             # any sentence-transformers model
VecForge("v.db", model_name="hash")                          # deterministic, no download, instant
VecForge("v.db", embedder=CallableEmbedder(my_api_embed))     # OpenAI / Voyage / Ollama / ONNX
```

Switch models later with `db.reembed()`. Every engine caches repeated queries, and vectors can be supplied directly: `db.add("caption", embedding=clip_vec, modality="image")`.

---

## 🌀 Quantum-Inspired Search

`quantum_rerank=True` runs **Grover amplitude amplification** over the candidates with a *consensus oracle* (hits returned by both the dense and the keyword retriever), evaluated in **closed form in O(N)**. Marked candidates rise above unmarked ones while keeping their order; the best candidate always stays first.

> 2.0 fixes the 1.x implementation, which applied Grover's diffusion step without an oracle and therefore *reversed* the ranking for roughly half of all candidate-set sizes. Regression tests cover every size the old code reversed.

| Candidates | Full rerank (window 1000) |
|---:|---:|
| 1,000 | 0.11 ms |
| 100,000 | 0.29 ms |
| 1,000,000 | **1.7 ms** |

> 📖 [Quantum Guide](docs/quantum.md)

---

## 🔐 Encrypted Vault, RBAC, Audit

```python
import os
from vecforge import VecForge

db = VecForge(
    "secure_vault",
    encryption_key=os.environ["VECFORGE_KEY"],   # SQLCipher AES-256, indexes included
    audit_log="audit.jsonl",
    deletion_protection=True,
)
db.register_api_key(os.environ["TEAM_KEY"], "read-only")   # digests only; unknown keys get nothing
```

> 📖 [Security Guide](docs/security.md)

---

## 🖼️ Multimodal Search

```python
from vecforge.ingest.vision import ImageEmbedder

clip = ImageEmbedder()
db = VecForge("gallery.db", embedder=clip)                       # one CLIP space for text + images
db.add("ward photo", embedding=clip.embed_image("ward.jpg"), modality="image")
db.search("patients in beds", alpha=1.0)                         # text → image
```

Audio goes through Whisper: `AudioEmbedder().transcribe_and_embed("meeting.mp3")`.

> 📖 [Multimodal Guide](docs/multimodal.md)

---

## 📄 Auto-Ingest Documents

```python
db.ingest("medical_records/")   # PDF, DOCX, TXT, MD, HTML — chunked and batched
```

> 📖 [Ingestion Guide](docs/ingestion.md)

---

## 🖥️ CLI & 🌐 REST API

```bash
vecforge ingest my_docs/ --vault my.db
vecforge search "diabetes" --vault my.db --top-k 5 --quantum --json
vecforge memory remember "User prefers tea" --kind preference
vecforge mcp --vault ~/.vecforge/memory.db --agent claude
vecforge serve --vault my.db --port 8080 --api-key "$VECFORGE_API_KEY"
```

```bash
curl -X POST http://localhost:8080/api/v1/search -H "X-API-Key: $VECFORGE_API_KEY" \
  -H "Content-Type: application/json" -d '{"query": "diabetes", "top_k": 5, "quantum_rerank": true}'
curl -X POST http://localhost:8080/api/v1/memory/remember -H "Content-Type: application/json" \
  -d '{"content": "User prefers dark mode", "kind": "preference", "agent": "claude"}'
```

> 📖 [CLI Reference](docs/cli_reference.md) · [REST API Reference](docs/rest_api.md)

---

## 🧪 Examples

| Example | Description |
|---|---|
| [🧠 AI Memory Agent](examples/ai_memory_agent.py) | An assistant that remembers across a process restart |
| [🏥 Hospital Search](examples/hospital_search.py) | Medical record search with namespace isolation per ward |
| [⚖️ Legal Documents](examples/legal_document_search.py) | NDA and contract search with type/year filtering |
| [🌍 GIS Data Search](examples/gis_data_search.py) | Geospatial dataset discovery |
| [🤖 RAG Pipeline](examples/rag_pipeline.py) | Retrieval-Augmented Generation with VecForge as backend |
| [🏢 Multi-Tenant SaaS](examples/multi_tenant_saas.py) | Namespace isolation, RBAC, and audit logging |
| [💻 Codebase Assistant](examples/codebase_assistant.py) | Code documentation semantic search |

---

## 📄 Citing VecForge

If VecForge helps your work, please cite the paper ([PDF](paper/VecForge-2.0-Paper.pdf); GitHub's "Cite this repository" button uses [`CITATION.cff`](CITATION.cff)):

```bibtex
@misc{bosek2026vecforge,
  author = {Suneel Bose K},
  title  = {VecForge: A Local-First Hybrid Vector Database with Persistent
            Agent Memory and Order-Preserving Grover-Inspired Reranking},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/bosekarmegam/vecforge/blob/main/paper/VecForge-2.0-Paper.pdf}
}
```

---

## 📚 Documentation

- **[📄 Research paper (PDF)](paper/VecForge-2.0-Paper.pdf)** · [Word](paper/VecForge-2.0-Paper.docx) · [experiments](paper/experiments/): design, proofs and a reproducible evaluation of VecForge 2.0
- **[📘 Complete Instructions (PDF, 26 pages)](docs/VecForge-Instructions.pdf)**: install, search, AI memory, MCP setup for Claude, REST, CLI and security in one guide
- [⚡ Quickstart](docs/quickstart.md) · [📦 Installation](docs/installation.md) · [🧠 Core Concepts](docs/core_concepts.md)
- [🔍 Search](docs/search.md) · [🧠 Memory](docs/memory.md) · [🔌 MCP](docs/mcp.md) · [🌀 Quantum](docs/quantum.md) · [🖼️ Multimodal](docs/multimodal.md)
- [🔐 Security](docs/security.md) · [📄 Ingestion](docs/ingestion.md) · [⚙️ Configuration](docs/configuration.md)
- [📖 API Reference](docs/api_reference.md) · [🖥️ CLI Reference](docs/cli_reference.md) · [🌐 REST API](docs/rest_api.md)
- [📝 Changelog](CHANGELOG.md) — including 2.0 migration notes
- 🎨 Brand kit: logo SVG/PNG exports and the 2.0 reel in [`assets/brand/`](assets/brand/)

---

## 📊 Benchmarks

Measured on a 4-core CPU container with the built-in hash embedder (so the numbers show database cost, not model cost), 100,000 documents, `python -m benchmarks.bench_search` / `bench_quantum`:

| Operation | VecForge 2.0 | VecForge 1.1 |
|---|---|---|
| Hybrid search, 100k docs (flat) | **7.3 ms** p50 | ~11 ms (no namespace scoping) |
| Hybrid search, 100k docs (HNSW, `index_type="auto"`) | **7.3 ms** p50, FAISS <1 ms | — |
| Namespace-scoped search, 500 of 100k docs | **0.8 ms** p50, full recall | lost recall past 4·top_k |
| Ingest 100k docs (database cost) | **4,500 docs/s** | O(N²) BM25 rebuild per add |
| Delete | **0.1 ms** (tombstone) | full index rebuild |
| Reopen 5k-doc vault | **30 ms** | rebuild |
| Reopen 100k-doc vault | **1.5 s** (snapshot) | 2.3 s (rebuild) |
| Quantum rerank, 1M candidates | **1.7 ms** | 3.3 s (and reversed rankings) |

### Quality Gates

| Check | Result |
|---|---|
| Ruff lint | ✅ All checks passed |
| Black | ✅ |
| Pytest | ✅ 346 tests (+10 that need the sentence-transformers download) |
| Coverage | 89% |

---

## ⚖️ License

**Business Source License 1.1 (BSL)**

- ✅ Free for personal, research, open-source, and non-commercial use
- ✅ Read, modify, and share freely
- 📋 Commercial use requires a license from ArcGX TechLabs

Contact: [suneelbose@arcgx.in](mailto:suneelbose@arcgx.in)

---

<p align="center">
  Built with ❤️ by <strong>Suneel Bose K</strong> · <strong>ArcGX TechLabs Private Limited</strong>
</p>
