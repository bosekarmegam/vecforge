# 📦 Installation Guide

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

---

## Requirements

- **Python 3.10+** (3.11–3.13 tested in CI)
- **pip** package manager

## Quick Install

```bash
pip install vecforge
```

## Install from Source (Development)

```bash
git clone https://github.com/bosekarmegam/vecforge.git
cd vecforge
pip install -e ".[dev]"
```

## Extras

| Extra | Installs | For |
|---|---|---|
| `vecforge[mcp]` | `mcp` | The MCP memory server for Claude Desktop / Claude Code |
| `vecforge[multimodal]` | open-clip-torch, openai-whisper, Pillow | Image and audio search |
| `vecforge[encryption]` | sqlcipher3-binary | AES-256 encrypted vaults |
| `vecforge[cloud]` | boto3, google-cloud-storage, azure-storage-blob | Encrypted backups to S3 / GCS / Azure |
| `vecforge[gpu]` | faiss-gpu, cupy | GPU acceleration |
| `vecforge[quantum]` | qiskit, qiskit-aer | Genuine quantum-circuit experiments |
| `vecforge[dev]` | pytest, mypy, ruff, black, httpx, mcp | Development |

```bash
pip install "vecforge[mcp,encryption]"
```

## No-download mode

The default embedding model is downloaded once from Hugging Face. On an
offline machine, in CI, or for instant startup, use the built-in hashing
embedder — no PyTorch model required:

```python
db = VecForge("vault.db", model_name="hash")
```

or set `VECFORGE_MODEL=hash`. (PyTorch is still installed as a dependency of
`sentence-transformers`; it is simply never loaded.)

## Optional: SQLCipher for Encryption

```bash
pip install "vecforge[encryption]"      # sqlcipher3-binary wheels
# or, with the system library:
pip install sqlcipher3
```

> **Note:** without SQLCipher, VecForge falls back to standard (unencrypted)
> SQLite and emits a warning when an `encryption_key` is passed.

**macOS:** `brew install sqlcipher && pip install sqlcipher3`
**Ubuntu/Debian:** `sudo apt-get install libsqlcipher-dev && pip install sqlcipher3`

## Verify Installation

```python
import vecforge
print(vecforge.__version__)   # 2.0.0
print(vecforge.__author__)    # Suneel Bose K
print(vecforge.__company__)   # ArcGX TechLabs Private Limited
```

```bash
vecforge --version
vecforge mcp --print-config claude-desktop
```

## System Dependencies

> VecForge uses `sentence-transformers` for embeddings, which requires **PyTorch**.

### Windows

PyTorch requires the **Microsoft Visual C++ Redistributable**:
https://aka.ms/vs/17/release/vc_redist.x64.exe. Without it you will see
`OSError: [WinError 126] The specified module could not be found` when loading `c10.dll`.

### macOS

No additional system dependencies.

### Linux (Ubuntu/Debian)

```bash
sudo apt-get install -y build-essential
```

### Python Dependencies

| Dependency | Purpose | Required? |
|---|---|---|
| `torch` | Deep learning backend | ✅ Auto-installed |
| `sentence-transformers` | Text embeddings | ✅ Auto-installed |
| `faiss-cpu` | Vector similarity search | ✅ Auto-installed |
| `numpy` | Core math, built-in BM25 and Grover | ✅ Auto-installed |
| `pymupdf`, `python-docx`, `beautifulsoup4` | Document ingestion | ✅ Auto-installed |
| `click` | CLI interface | ✅ Auto-installed |
| `fastapi` + `uvicorn` | REST API server | ✅ Auto-installed |
| `mcp` | MCP server | ⚙️ `vecforge[mcp]` |
| `sqlcipher3` | AES-256 encryption | ⚙️ `vecforge[encryption]` |
| `faiss-gpu` | GPU acceleration | ⚙️ `vecforge[gpu]` |

VecForge 2.0 no longer depends on `rank-bm25`, `numba` or `joblib`.
