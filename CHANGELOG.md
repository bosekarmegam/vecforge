# VecForge — Changelog

All notable changes to VecForge will be documented in this file.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

## [1.1.0] — 2026-03-12
### Added — Phase 4: Multimodal & Advanced

* 🖼️ **`vecforge/ingest/vision.py`** — `ImageEmbedder` using local OpenCLIP (ViT-B-32). Generates 512-dim CLIP embeddings from JPG / PNG / WebP files. GPU-accelerated when CUDA available, CPU fallback. Lazy-loads model on first use.
* 🎤 **`vecforge/ingest/audio.py`** — `AudioEmbedder` using local OpenAI Whisper (`base` model). Transcribes MP3 / WAV / FLAC → text, then embeds transcript for semantic search.
* 🔗 **`vecforge/search/crossmodal.py`** — `CrossModalSearcher`. Auto-detects query modality (text / image path / audio path) and routes to the correct embedder. Enables text→image, image→text, audio→text search in the unified vector space.
* ☁️ **`vecforge/sync/cloud.py`** — `CloudSync` — opt-in encrypted vault backup to S3, GCS, or Azure Blob. Vault is AES-256 encrypted locally **before** any upload; decryption key never leaves the machine. SHA-256 checksum verified after upload.
* 📦 **`[multimodal]` extra** — `open-clip-torch`, `openai-whisper`, `Pillow`
* 📦 **`[cloud]` extra** — `boto3`, `google-cloud-storage`, `azure-storage-blob`
* 🧪 **`tests/unit/test_multimodal.py`** — 15 new tests covering CrossModalSearcher modality detection, ImageEmbedder/AudioEmbedder import guards, and windowed QuantumReranker at 1M docs

### Fixed — Quantum large-dataset performance
* ⚡ **`QuantumReranker.max_candidates`** — New `max_candidates=1000` pre-filter in `QuantumReranker`: uses O(N) `numpy.argpartition` to clip to top-K candidates before Grover runs. Reduces 1M-doc rerank from **3.3s → <5ms** (660x speedup). Grover now correctly runs in O(K·√K) as spec'd in `VecForge_INSTRUCTIONS.md`.
* 📊 Quantum rerank now meets the **<20ms at 1M docs** North Star target when using the pre-filter window.

### Changed
* 🔼 **Classifier** bumped Alpha → Beta → Production/Stable
* 🔢 **Version** bumped 1.0.1 → 1.1.0


### Added — Phase 3: Quantum-Inspired Acceleration
* 🌀 **`vecforge.quantum` module** — New `AmplitudeEncoder`, `GroverAmplifier`, and `QuantumReranker` classes running entirely on classical hardware (NumPy only)
* ⚛️ **Grover-inspired score amplification** — Inversion-about-mean diffusion operator (O(√N) effective steps) that widens the relevance gap between top and bottom candidates
* 📐 **Amplitude encoding** — L2-normalises scores into unit-norm quantum amplitude space for numerically stable score fusion
* 🔌 **`quantum_rerank=True` param** — New flag on `VecForge.search()` that runs Grover amplification as Stage 5 of the cascade pipeline
* 📦 **`vecforge[quantum]` extra** — Optional `qiskit>=1.0.0` and `qiskit-aer>=0.14.0` extras for genuine quantum hybrid experiments
* ⚡ **`benchmarks/bench_quantum.py`** — New benchmark suite; measured results on CPU:
  - 1k docs: Encode=0.006ms, Grover=0.391ms, QRerank=0.551ms ✅
  - 10k docs: Encode=0.009ms, Grover=1.636ms, QRerank=3.137ms ✅
  - 100k docs: Encode=0.054ms, Grover=18.527ms, QRerank=32.682ms
  - 1M docs: AmplitudeEncoder alone=4.230ms ✅
* 📖 **`docs/quantum.md`** — Full documentation covering algorithm theory, usage, performance, and API reference
* 🔼 **Bumped classifier** from Alpha → Beta

## [1.0.0] — 2026-03-05
### Changed
* 🚀 Vector score normalization (fixed a bug where `SearchCascade` hybrid fusion resulted in squashed score differences). Scores are now perfectly min-max normalized between `0.0` and `1.0`. All search relevancies accurately reflect semantic similarity.
* Bumped major version due to stabilization of API surfaces and core scoring algorithms.

## [0.2.0] — 2026-03-05

### Added — Phase 2: Performance & Polish
- **Batch add API** (`add_batch()`) — single-call embedding + FAISS insert for ~5x throughput
- **Benchmark suite** (`benchmarks/bench_search.py`) — FAISS + BM25 latency & throughput tests
- **40 new tests** — `test_coverage.py` covering filters, hybrid fusion, cascade, audit, encryption, snapshots
- **`.gitignore`** with comprehensive Python project ignores
- **README badges** — Tests, Coverage, Ruff, Mypy, Benchmark

### Fixed
- **Ruff lint**: B904 exception chaining, N806 naming, F841 unused vars, E501 line length (8 files)
- **Mypy type errors**: unused `type: ignore` comments, missing type params, `no-any-return` (4 files)
- **`MetadataFilter`**: Rewrote 303-char one-liner into readable if-chain
- **CLI export**: Removed unused VecForge context manager

### Performance
- **100k doc search**: p50 = 11.31ms ✅ (target <15ms)
- **FAISS ingest**: 2.9M docs/sec throughput
- **BM25 10k search**: p50 = 9.40ms

### Quality Gates
- Ruff: All checks passed ✅
- Mypy: 0 errors in 27 files ✅
- Pytest: 110/110 tests pass ✅
- Coverage: 61% (core modules 75-100%)

---

## [0.1.0] — 2026-03-05

### Added — Phase 1: Core Engine
- **Core VecForge class** with 5-line API (add, search, ingest, delete)
- **FAISS dense retrieval** with IndexFlatIP for vector similarity search
- **BM25 keyword search** via rank-bm25 for sparse retrieval
- **Hybrid search fusion** using Reciprocal Rank Fusion (RRF) and weighted linear combination
- **4-stage cascade search pipeline**: FAISS → BM25 merge → metadata filter → cross-encoder rerank
- **SQLite persistence** with WAL mode for concurrent reads
- **SQLCipher AES-256 encryption** at rest (optional, graceful fallback)
- **Namespace isolation** for multi-tenant data separation
- **Role-based access control (RBAC)** with admin, read-write, read-only roles
- **Audit logging** with append-only JSONL events
- **Metadata filtering** with equality, range, in/not_in operators
- **Document ingestion** for .txt, .md, .pdf, .docx, .html with smart chunking
- **Cross-encoder reranking** for precision search refinement
- **Vault snapshots** for backup and restore
- **Click CLI** with ingest, search, stats, export, serve commands
- **FastAPI REST server** with /add, /search, /stats, /docs endpoints
- **Sentence-transformers** embedding with lazy model loading

### Author
- Suneel Bose K — Founder & CEO, ArcGX TechLabs Private Limited
