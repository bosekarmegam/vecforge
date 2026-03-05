# 📄 Ingestion Guide

Auto-ingest documents from files and directories.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

---

## Supported Formats

| Format | Extensions | Parser |
|---|---|---|
| Plain Text | `.txt` | Built-in |
| Markdown | `.md` | Built-in |
| PDF | `.pdf` | PyMuPDF (fitz) |
| Word | `.docx` | python-docx |
| HTML | `.html`, `.htm` | BeautifulSoup |

---

## Quick Ingest

```python
from vecforge import VecForge

db = VecForge("my_vault")

# Ingest a single file
db.ingest("report.pdf")

# Ingest an entire directory (recursive)
count = db.ingest("documents/")
print(f"Ingested {count} chunks")
```

---

## Chunking Configuration

Documents are split into overlapping chunks for better search precision:

```python
count = db.ingest(
    "documents/",
    chunk_size=1000,      # Max characters per chunk
    chunk_overlap=200,    # Overlap between consecutive chunks
)
```

### Choosing Chunk Size

| Chunk Size | Best For |
|---|---|
| 500 | Short passages, high precision |
| 1000 | General purpose (default) |
| 2000 | Long-form documents, more context |

### How Chunking Works
1. Text is split at sentence/paragraph boundaries when possible
2. Chunks overlap by `chunk_overlap` characters for continuity
3. Each chunk gets metadata: `source`, `chunk_index`, `char_start`, `char_end`

---

## Namespace Targeting

Ingest directly into a specific namespace:

```python
db.ingest("hospital_a_records/", namespace="hospital_a")
db.ingest("hospital_b_records/", namespace="hospital_b")
```

---

## Chunk Metadata

Each ingested chunk includes automatic metadata:

```python
results = db.search("diabetes")

print(results[0].metadata)
# {
#     "source": "medical_records/report.pdf",
#     "chunk_index": 3,
#     "char_start": 3000,
#     "char_end": 4000,
#     "page": 2,          # PDF only
# }
```

---

## Using the Dispatcher Directly

For advanced use, access the ingestion dispatcher directly:

```python
from vecforge.ingest.dispatcher import IngestDispatcher

dispatcher = IngestDispatcher(chunk_size=500, chunk_overlap=100)

# Get raw chunks without adding to vault
chunks = dispatcher.ingest("my_documents/")

for chunk in chunks:
    print(f"Source: {chunk.metadata['source']}")
    print(f"Text: {chunk.text[:100]}...")
    print()
```

---

## Supported Extensions

```python
from vecforge.ingest.dispatcher import IngestDispatcher

print(IngestDispatcher.supported_extensions())
# ['.docx', '.htm', '.html', '.md', '.pdf', '.txt']
```
