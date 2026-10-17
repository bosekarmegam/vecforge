# VecForge — Changelog

All notable changes to VecForge will be documented in this file.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

## [2.0.0] — 2026-10-17
### Phase 5: AI Memory, MCP, and a corrected, faster core

#### Added — AI memory for Claude and other agents
* 🧠 **`vecforge.memory.MemoryStore`** — persistent memory on top of a vault: `remember` / `recall` / `forget` / `reinforce` / `update`, memory kinds (fact, preference, decision, insight, task, episode, summary, note), importance + recency weighted recall, exact and near-duplicate de-duplication, sessions (`start_session` / `end_session` with summaries, `last_session`), `context_pack()` prompt-ready briefings, JSON export/import, per-agent namespaces.
* 🔌 **`vecforge.mcp`** — local Model Context Protocol server (`vecforge mcp`, stdio or streamable HTTP) exposing `remember`, `recall`, `forget`, `reinforce`, `list_memories`, `context_pack`, `start_session`, `end_session`, `recent_sessions`, `search_vault`, `add_document`, `vault_stats`, plus resources and a `recall_context` prompt. Works with `mcp` 1.x and 2.x. `--print-config claude-desktop|claude-code` prints the client snippet. Background model warm-up and periodic snapshots make the next session open instantly.
* 🖥️ **CLI** — `vecforge mcp`, `vecforge memory remember|recall|list|forget|sessions|context|export|import`, `vecforge compact`, `vecforge backup`; `search --quantum --filters --recency --json`; `--model` on every vault command; version read from the package.
* 🌐 **REST** — `add_batch`, `GET/PUT docs/{id}`, `PATCH docs/{id}/metadata`, `docs/{id}/similar`, `DELETE namespaces/{name}`, `save`, `compact`, memory endpoints (`/memory/remember|recall|context|sessions`), `quantum_rerank` and `recency_weight` on search, `X-API-Key` authentication (`--api-key` / `$VECFORGE_API_KEY`), full `stats`, vault closed on shutdown, endpoints run in the thread pool.

#### Added — AI-database adaptability
* 🔁 **Pluggable embedders** (`EmbedderLike`): `Embedder` (sentence-transformers), `HashEmbedder` (`model_name="hash"`, deterministic, no model download, instant start), `CallableEmbedder` (wrap OpenAI / Voyage / Ollama / ONNX functions), `ImageEmbedder` now implements the protocol so a whole vault can live in CLIP space. `VECFORGE_MODEL` env override, LRU query-embedding cache, `db.reembed()` after a model switch.
* 📝 **Vault API** — `upsert`, `get`, `get_embedding`, `exists`, `count`, `update_metadata`, `search_batch`, `similar`, `delete_many`, `delete_namespace`, `compact`, `backup`, `register_api_key` / `revoke_api_key` / `list_api_keys`, `add(embedding=..., modality=...)`, `search(query_vector=..., exclude_doc_ids=...)`; `len(db)` and `doc_id in db`.
* 🔎 **Explainable results** — `SearchResult.dense_score`, `sparse_score`, `signals`, `updated_at`, `to_dict()`.
* 🧩 **Metadata filters** — new operators `contains`, `any`, `all`, `exists`; incompatible comparisons no longer raise.

#### Changed — performance
* ⚡ **BM25 engine rewritten** as an incremental inverted index (no `rank-bm25`): O(tokens) adds instead of a full rebuild per document (1.x ingestion was O(N²)), O(postings) queries instead of O(terms × N), tombstones, pickle-free serialisation.
* ⚡ **Namespace scoping and deletes inside the retrievers** via FAISS ID selectors — small namespaces are never starved by a fixed `4·top_k` window (1.x lost recall), and metadata filters widen the window adaptively.
* ⚡ **Index snapshots** — `save()` / `close()` persist FAISS + BM25 + position map + tombstones (in 64 MB chunks, inside the encrypted vault); reopening validates a generation counter and restores in milliseconds instead of rebuilding from every embedding.
* ⚡ **Batch everything** — `add_batch` / `ingest` use one SQLite transaction and one index insert; result hydration is one batched query without embedding blobs.
* ⚡ **O(1) deletes** with tombstones and automatic compaction at 20%.
* ⚡ **FAISS `index_type`** — `flat`, `hnsw`, or `auto` (HNSW past 100k documents).
* 🔒 All public `VecForge` methods are thread-safe (re-entrant lock).

#### Fixed — quantum reranking (correctness)
* 🌀 **`GroverAmplifier` applied the diffusion operator without an oracle.** Diffusion alone is a reflection: for candidate sets of size 2–6, 15–26, 41–58, … (odd optimal `k`) the ranking was **reversed**, for the rest it was a no-op. 2.0 implements the real oracle + diffusion iterate, evaluates it in **closed form in O(N)**, keeps the order inside the marked set, and `VecForge.search` uses a *consensus oracle* (candidates matched by both retrievers). Scores are max-normalised probabilities in `[0, 1]`. With `rerank=True` the cross-encoder runs only on the Grover survivors instead of twice.
* 🧪 Regression tests for every candidate-set size the 1.x code reversed; closed form verified against the iterative reference.

#### Fixed — other
* `CrossModalSearcher` / `AudioEmbedder` called a non-existent `Embedder.embed()`; `embed()` now exists on every embedder and `CrossModalSearcher(text_embedder=db.embedder)` shares the vault's model.
* SQLCipher `PRAGMA key` was built with an unescaped f-string; keys containing quotes now work and wrong keys fail at open time.
* `SnapshotManager` copied a live WAL database (incomplete backups) and left stale journals on restore; it now uses the SQLite online-backup API.
* `recency_weight` was discarded by `rerank`/`quantum_rerank`; it is now applied after reranking on normalised scores. Negative `top_k` returned garbage; it now returns `[]`.
* `VecForge(quantum=True)` had no effect; it is now the default for `search(quantum_rerank=...)`.
* Version strings diverged (package 1.1.0, CLI/REST 1.0.0); all derive from `vecforge.__version__`.
* Vaults with mixed embedding dimensions (after a model switch) no longer fail to open; mismatched rows are skipped with a warning until `reembed()`.
* Persisted FAISS index was written on every close but never read; `IVF` constants were dead code.
* Stray root scripts moved to `scripts/`; `tests/verify_examples.py` is now collected as `tests/integration/test_examples.py`.

#### Security
* 🔑 Unregistered API keys now get **no** access (1.x granted read-only). Keys are registered with `db.register_api_key()` and persisted as SHA-256 digests.
* 🛡️ `vecforge serve` binds `127.0.0.1` by default and supports `X-API-Key`.

#### Breaking changes / migration
* Schema v2 (`content_hash`, `index_state`, `index_chunks`, `sessions`, `kv`): 1.x vaults migrate automatically on first open; the first open rebuilds the indexes once, after which snapshots apply.
* `quantum_rerank=True` returns different (correct) orderings and `[0, 1]` scores.
* `VecForge(api_key=...)` with an unregistered key is denied; register keys as admin first.
* `vecforge serve` default host changed from `0.0.0.0` to `127.0.0.1`.
* `ImageEmbedder.embed()` still accepts image paths; `embed_image()` is the explicit method and `embed()` on plain text now encodes text (CLIP text tower).
* Dependencies removed: `rank-bm25`, `numba`, `joblib`. New extras: `[mcp]`, `[encryption]`.
* `CascadeSearcher.search()` gained `allowed_ids` / `excluded_ids` / `over_fetch`; `SearchResult` gained fields (dataclass positional order unchanged).
* `db.storage` (raw SQLite access) now requires an admin key; `save()` and `compact()` require write; `exists`, `count`, `stats`, `similar` and namespace listing require read.
* `Storage.kv_set()` rejects keys starting with `__` (reserved for VecForge internals).
* CLI `stats`, `export`, `compact` and `backup` refuse a vault path that does not exist instead of creating an empty vault.

#### Hardening (pre-release review)
* Atomic writes: a dimension mismatch is rejected before anything is stored, a failed index update rolls the rows back, and `upsert` replaces a document in one transaction.
* SQLCipher vaults: `sqlite3.Row` replaced by the driver's own row type, explicit transactions throughout, and encrypted `backup()` via `sqlcipher_export`.
* Snapshots stay valid across metadata-only updates; auto-save failures are logged instead of raised; snapshot source URIs work with spaces and non-ASCII paths.
* `MemoryStore` shares the vault lock, refuses use after close, and isolates agents: `forget`, `update`, sessions and dedup only touch the store's own namespace. Imports keep sessions and timestamps; `context_pack` no longer repeats the last session summary.
* Quantum rerank and amplitude encoding handle `NaN` and `±inf` scores (`+inf` ranks first); ties keep their input order.
* `MemoryStore.recall(since=...)` keeps memories created strictly after the timestamp, as documented.
* MCP tools return readable errors, shut down cleanly on SIGTERM, and `--print-config` keeps `--model`. REST API keys are compared as bytes, so non-ASCII keys return 401 instead of 500.
* `HashEmbedder` gives punctuation-only text a non-zero vector; `ImageEmbedder.embed()` only treats strings as paths when they look like images or exist on disk.
* mypy strict passes with no errors; 386 tests, 90% coverage, plus a randomized differential fuzzer against a brute-force reference.

#### Brand and docs
* New VecForge logo: SVG mark and horizontal lockups (light and dark) with PNG, favicon and app-icon exports in `assets/brand/`.
* Website (`index.html`) updated with the new logo and 2.0 content: AI memory section, MCP tools, quantum rerank, comparison rows. Plain-language copy, and a mobile layout that no longer scrolls sideways (the hero glow widened the page on phones; the comparison table now fits a 320 px screen).
* Complete instructions guide: `docs/VecForge-Instructions.pdf` (26 pages, bookmarked), built from `docs/guide/` by `scripts/build_guide.py`.
* 45-second vertical brand reel `assets/brand/vecforge-reel.mp4` (1080x1920, "Powered by ArcGX TechLabs") with an original soundtrack generated by `scripts/make_reel_music.py` and rendered by `scripts/render_reel.cjs`.

## [1.1.0] — 2026-03-12
### Added — Phase 4: Multimodal & Advanced

* 🖼️ **`vecforge/ingest/vision.py`** — `ImageEmbedder` using local OpenCLIP (ViT-B-32). Generates 512-dim CLIP embeddings from JPG / PNG / WebP files. GPU-accelerated when CUDA available, CPU fallback. Lazy-loads model on first use.
* 🎤 **`vecforge/ingest/audio.py`** — `AudioEmbedder` using local OpenAI Whisper (`base` model). Transcribes MP3 / WAV / FLAC → text, then embeds transcript for semantic search.
* 🔗 **`vecforge/search/crossmodal.py`** — `CrossModalSearcher`. Auto-detects query modality (text / image path / audio path) and routes to the correct embedder.
* ☁️ **`vecforge/sync/cloud.py`** — `CloudSync` — opt-in vault backup upload to S3, GCS, or Azure Blob with SHA-256 checksum metadata.
* 📦 **`[multimodal]` extra** — `open-clip-torch`, `openai-whisper`, `Pillow`
* 📦 **`[cloud]` extra** — `boto3`, `google-cloud-storage`, `azure-storage-blob`
* 🧪 **`tests/unit/test_multimodal.py`** — tests covering CrossModalSearcher modality detection, ImageEmbedder/AudioEmbedder import guards, and the windowed QuantumReranker at 1M docs

### Changed — Quantum large-dataset performance
* ⚡ **`QuantumReranker.max_candidates`** — pre-filter to the top-K candidates with O(N) `numpy.argpartition` before amplification (default 1000).

### Changed
* 🔼 Classifier bumped to Production/Stable; version 1.0.1 → 1.1.0

## [1.0.1] — 2026-03-08
### Added — Phase 3: Quantum-Inspired Acceleration
* 🌀 **`vecforge.quantum` module** — `AmplitudeEncoder`, `GroverAmplifier`, and `QuantumReranker` running on NumPy (superseded by the corrected 2.0 implementation)
* 🔌 **`quantum_rerank=True` param** on `VecForge.search()`
* 📦 **`vecforge[quantum]` extra** — optional `qiskit` / `qiskit-aer`
* ⚡ **`benchmarks/bench_quantum.py`** and 📖 **`docs/quantum.md`**

## [1.0.0] — 2026-03-05
### Changed
* 🚀 Vector score normalization (fixed a bug where `SearchCascade` hybrid fusion resulted in squashed score differences). Scores are min-max normalized between `0.0` and `1.0`.
* Bumped major version due to stabilization of API surfaces and core scoring algorithms.

## [0.2.0] — 2026-03-05

### Added — Phase 2: Performance & Polish
- **Batch add API** (`add_batch()`) — single-call embedding + FAISS insert
- **Benchmark suite** (`benchmarks/bench_search.py`) — FAISS + BM25 latency & throughput tests
- **40 new tests** — `test_coverage.py` covering filters, hybrid fusion, cascade, audit, encryption, snapshots
- **`.gitignore`** with comprehensive Python project ignores
- **README badges** — Tests, Coverage, Ruff, Mypy, Benchmark

### Fixed
- **Ruff lint**: B904 exception chaining, N806 naming, F841 unused vars, E501 line length (8 files)
- **Mypy type errors**: unused `type: ignore` comments, missing type params, `no-any-return` (4 files)
- **`MetadataFilter`**: Rewrote 303-char one-liner into readable if-chain
- **CLI export**: Removed unused VecForge context manager

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
