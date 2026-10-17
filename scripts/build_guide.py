# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under the Business Source License 1.1 (BSL 1.1)

"""
Build docs/VecForge-Instructions.pdf from docs/guide/VecForge-Instructions.html.

Steps:
  1. Print the HTML to PDF with headless Chromium (fonts: Inter, DejaVu Mono).
  2. Stamp a footer with page numbers on every page except the cover.
  3. Add bookmarks for every chapter and set document metadata.

Usage::

    python scripts/build_guide.py [--chrome /path/to/chrome]

Requires: Chromium/Chrome, pypdf, reportlab.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "guide" / "VecForge-Instructions.html"
OUT = ROOT / "docs" / "VecForge-Instructions.pdf"
CHROME_CANDIDATES = [
    os.environ.get("CHROME", ""),
    "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
    shutil.which("chromium") or "",
    shutil.which("chromium-browser") or "",
    shutil.which("google-chrome") or "",
]
CHAPTERS = [
    "What VecForge is",
    "Installation",
    "Quickstart",
    "Core concepts",
    "Adding and managing data",
    "Searching",
    "AI memory",
    "Connecting Claude over MCP",
    "Embedding models",
    "Security",
    "REST API",
    "Command line",
    "Multimodal",
    "Performance and tuning",
    "Upgrading from 1.x",
    "Troubleshooting",
    "Quick reference",
]


def find_chrome(explicit: str | None) -> str:
    for c in [explicit or "", *CHROME_CANDIDATES]:
        if c and Path(c).exists():
            return c
    raise SystemExit("Chromium/Chrome not found; pass --chrome PATH")


def print_pdf(chrome: str, src: Path, dest: Path) -> None:
    subprocess.run(
        [
            chrome,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={dest}",
            src.as_uri(),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def footer_page(width: float, height: float, number: int, total: int) -> PdfReader:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(width, height))
    c.setStrokeColor(HexColor("#E4E6F2"))
    c.setLineWidth(0.6)
    c.line(51, 40, width - 51, 40)
    c.setFillColor(HexColor("#8B90A8"))
    c.setFont("Helvetica", 7.5)
    c.drawString(51, 28, "VecForge 2.0 — Complete Instructions")
    c.drawCentredString(width / 2, 28, "Powered by ArcGX TechLabs")
    c.drawRightString(width - 51, 28, f"{number} / {total}")
    c.save()
    buf.seek(0)
    return PdfReader(buf)


def chapter_pages(reader: PdfReader) -> dict[str, int]:
    """Find the first page whose text starts a chapter heading."""
    found: dict[str, int] = {}
    for idx, page in enumerate(reader.pages):
        if idx < 2:
            continue
        text = page.extract_text() or ""
        head = " ".join(text.split()[:12])
        for n, title in enumerate(CHAPTERS, 1):
            if title in found:
                continue
            if re.match(rf"^{n}\s*{re.escape(title)}", head):
                found[title] = idx
    return found


def build(chrome: str) -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.pdf"
        print_pdf(chrome, SRC, raw)
        reader = PdfReader(str(raw))
        total = len(reader.pages)
        # why: detect headings before the footer text is merged into pages
        chapters = chapter_pages(reader)
        writer = PdfWriter()
        for i, page in enumerate(reader.pages):
            if i > 0:
                w = float(page.mediabox.width)
                h = float(page.mediabox.height)
                page.merge_page(footer_page(w, h, i + 1, total).pages[0])
            writer.add_page(page)

        writer.add_outline_item("Cover", 0)
        writer.add_outline_item("Contents", 1)
        for title, idx in sorted(chapters.items(), key=lambda kv: kv[1]):
            writer.add_outline_item(f"{CHAPTERS.index(title) + 1}. {title}", idx)

        writer.add_metadata(
            {
                "/Title": "VecForge 2.0 — Complete Instructions",
                "/Author": "Suneel Bose K · ArcGX TechLabs Private Limited",
                "/Subject": "Installation, search, AI memory, MCP, REST, CLI, security",
                "/Keywords": "VecForge, vector database, MCP, Claude, AI memory",
                "/Creator": "scripts/build_guide.py",
            }
        )
        writer.page_mode = "/UseOutlines"
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT, "wb") as f:
            writer.write(f)
    return OUT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chrome", default=None)
    args = parser.parse_args()
    out = build(find_chrome(args.chrome))
    print(f"wrote {out} ({len(PdfReader(str(out)).pages)} pages)")


if __name__ == "__main__":
    main()
