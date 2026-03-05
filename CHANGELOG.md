# VecForge — Changelog

All notable changes to VecForge will be documented in this file.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

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
