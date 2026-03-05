# 📦 Installation Guide

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

---

## Requirements

- **Python 3.10+** (3.11 or 3.12 recommended)
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

## Install Options

### CPU-Only (Default)

```bash
pip install vecforge
```

### With GPU Acceleration

```bash
pip install vecforge[gpu]
```

Requires CUDA toolkit installed on your system.

### With Development Tools

```bash
pip install vecforge[dev]
```

Includes: pytest, mypy, ruff, black, coverage.

### Full Install

```bash
pip install vecforge[dev,gpu]
```

## Optional: SQLCipher for Encryption

For AES-256 encrypted vaults, install SQLCipher:

```bash
pip install sqlcipher3
```

> **Note:** `sqlcipher3` requires the SQLCipher C library. On Windows, this may
> require additional setup. VecForge works without it — encryption is optional.
> Without SQLCipher, VecForge falls back to standard (unencrypted) SQLite.

### Platform-Specific SQLCipher Installation

**macOS:**
```bash
brew install sqlcipher
pip install sqlcipher3
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libsqlcipher-dev
pip install sqlcipher3
```

**Windows:**
```bash
# Pre-built wheel (if available)
pip install sqlcipher3
# Or build from source with Visual Studio Build Tools
```

## Verify Installation

```python
import vecforge
print(vecforge.__version__)   # 0.2.0
print(vecforge.__author__)    # Suneel Bose K
print(vecforge.__company__)   # ArcGX TechLabs Private Limited
```

## System Dependencies

> **Important:** VecForge uses `sentence-transformers` for embeddings, which requires **PyTorch**.
> PyTorch has platform-specific system requirements listed below.

### Windows

PyTorch requires the **Microsoft Visual C++ Redistributable** (latest supported):

```bash
# Download and install from:
# https://aka.ms/vs/17/release/vc_redist.x64.exe
```

> Without MSVC C++ Redistributable, you will see:
> `OSError: [WinError 126] The specified module could not be found` when loading `c10.dll`

### macOS

No additional system dependencies required. PyTorch installs cleanly via pip.

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
| `numpy` | Core math | ✅ Auto-installed |
| `rank-bm25` | Keyword search | ✅ Auto-installed |
| `pymupdf` | PDF ingestion | ✅ Auto-installed |
| `click` | CLI interface | ✅ Auto-installed |
| `fastapi` + `uvicorn` | REST API server | ✅ Auto-installed |
| `sqlcipher3` | AES-256 encryption | ⚙️ Optional |
| `faiss-gpu` | GPU acceleration | ⚙️ Optional |
