# 🖼️ Multimodal Guide

Images and audio in a VecForge vault.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

---

## Install

```bash
pip install "vecforge[multimodal]"   # open-clip-torch, openai-whisper, Pillow
```

Whisper also needs `ffmpeg` on the system path.

---

## How vectors line up

A vault has **one** embedding dimension. The default text model
(`all-MiniLM-L6-v2`) produces 384-dim vectors; CLIP produces 512-dim
vectors. To search images with text (or text with images) everything in the
vault must live in the same space, so build the vault **with CLIP as its
embedder**:

```python
from vecforge import VecForge
from vecforge.ingest.vision import ImageEmbedder

clip = ImageEmbedder()                      # ViT-B-32 / openai, 512-dim
db = VecForge("gallery.db", embedder=clip)  # text queries use CLIP's text tower

# Text documents are embedded with CLIP text
db.add("A photo of a hospital ward", metadata={"type": "caption"})

# Images are embedded with CLIP vision and stored with modality="image"
for path in ["ward.jpg", "lab.png"]:
    db.add(
        f"image {path}",                         # text used for BM25 / display
        embedding=clip.embed_image(path),
        modality="image",
        metadata={"path": path},
    )

results = db.search("patients in beds", alpha=1.0, top_k=5)   # text → image
for r in results:
    print(r.modality, r.metadata.get("path"), round(r.score, 3))
```

Use `alpha=1.0` (semantic only) for image results: BM25 has nothing to match
in an image. Query **by image** with a precomputed vector:

```python
vec = clip.embed_image("query.jpg")
results = db.search("", query_vector=vec, alpha=1.0, top_k=5)
```

`ImageEmbedder` implements the same `EmbedderLike` protocol as the text
embedders (`name`, `dimension`, `encode`, `embed`), which is why it can be
passed as `embedder=`.

---

## Audio

`AudioEmbedder` transcribes audio with Whisper and embeds the transcript
with the vault's text model, so audio lives naturally in a text vault:

```python
from vecforge import VecForge
from vecforge.ingest.audio import AudioEmbedder

db = VecForge("notes.db")
audio = AudioEmbedder(whisper_model="base", text_embedder=db.embedder)

transcript, vec = audio.transcribe_and_embed("meeting.mp3")
db.add(transcript, embedding=vec, modality="audio", metadata={"path": "meeting.mp3"})

db.search("what did we decide about the budget?")
```

Supported formats: `mp3`, `wav`, `flac`, `m4a`, `ogg`, `webm`.

---

## CrossModalSearcher

`CrossModalSearcher` turns any query (text, image path, audio path) into a
vector by detecting the modality from the file extension:

```python
from vecforge.search.crossmodal import CrossModalSearcher

cs = CrossModalSearcher(text_embedder=db.embedder)   # share the vault's model
vec = cs.encode_query("ward_photo.jpg")               # CLIP image vector
vec = cs.encode_query("patient_recording.mp3")        # Whisper → text vector
vec = cs.encode_query("hip fracture elderly")         # text vector
results = db.search("", query_vector=vec, alpha=1.0)
```

Pass `text_embedder=db.embedder` so text queries reuse the vault's model and
cache. Remember the dimension rule above: image vectors only match a vault
built with `embedder=ImageEmbedder()`.

---

## Opt-in cloud backup

```bash
pip install "vecforge[cloud]"
```

```python
from vecforge.sync import CloudSync

db.backup("vault-backup.db")             # consistent copy via SQLite backup API
sync = CloudSync(backend="s3", bucket="my-vecforge-backups")
sync.upload("vault-backup.db.enc")       # upload an encrypted copy only
```

`CloudSync` refuses files that look like plaintext vaults (`.db` suffix).
Encrypt the backup before uploading — for example with SQLCipher
(`VecForge(..., encryption_key=...)` produces an encrypted `.db` you can
rename to `.db.enc`) or with your own tooling. The upload itself never
encrypts.
