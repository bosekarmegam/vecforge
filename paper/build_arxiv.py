# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under the Business Source License 1.1 (BSL 1.1)

"""
Build the arXiv LaTeX source package of the VecForge 2.0 paper from vecforge-paper.md.

    python3 paper/build_arxiv.py

Needs pandoc (3.x) and pdflatex (TeX Live 2023 or later, with mathptmx,
orcidlink, booktabs, longtable, caption, hyperref and microtype). Writes a
clean paper/arxiv/ directory holding main.tex, the figures the paper
references as PNG files at the root, and main.pdf for review, then packs
main.tex and the PNG files into paper/VecForge-2.0-arXiv-source.tar.gz with
every file at the archive root, which is the layout arXiv expects.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "vecforge-paper.md"
ARXIV = HERE / "arxiv"
TEX = ARXIV / "main.tex"
TAR = HERE / "VecForge-2.0-arXiv-source.tar.gz"
# why: arXiv reads the log and runs more passes while it sees this message
RERUN_HINT = "\\typeout{get arXiv to do 4 passes: Label(s) may have changed. Rerun}"
PANDOC = [
    "pandoc",
    str(SRC),
    "--citeproc",
    "--bibliography",
    str(HERE / "references.bib"),
    "--csl",
    str(HERE / "ieee-numeric.csl"),
    "--number-sections",
    "--template",
    str(HERE / "arxiv-template.tex"),
    "--lua-filter",
    str(HERE / "arxiv-captions.lua"),
    "-t",
    "latex",
    "-o",
    str(TEX),
]


def referenced_figures() -> list[Path]:
    """The image files the Markdown uses, in order of first use."""
    text = SRC.read_text(encoding="utf-8")
    seen: dict[str, Path] = {}
    for rel in re.findall(r"!\[[^\]]*\]\(([^)\s]+)", text):
        path = HERE / rel
        if not path.is_file():
            raise SystemExit(f"figure not found: {path}")
        if path.suffix.lower() != ".png":
            raise SystemExit(f"arXiv package takes PNG figures only: {rel}")
        if path.name in seen and seen[path.name] != path:
            raise SystemExit(f"two figures would share the name {path.name}")
        seen[path.name] = path
    return list(seen.values())


def copy_figures() -> list[str]:
    names = []
    for src in referenced_figures():
        shutil.copyfile(src, ARXIV / src.name)
        names.append(src.name)
    return names


def write_tex() -> None:
    subprocess.run(PANDOC, check=True)
    text = TEX.read_text(encoding="utf-8").rstrip("\n")
    if "\\today" in text:
        raise SystemExit("main.tex contains \\today; arXiv must not stamp a new date")
    TEX.write_text(f"{text}\n{RERUN_HINT}\n", encoding="utf-8")


def compile_tex(passes: int = 3) -> str:
    env = {**os.environ, "max_print_line": "1000"}
    for _ in range(passes):
        run = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=ARXIV,
            env=env,
            capture_output=True,
            text=True,
        )
        log = (ARXIV / "main.log").read_text(encoding="utf-8", errors="replace")
        errors = [line for line in log.splitlines() if line.startswith("!")]
        if run.returncode != 0 or errors:
            print(run.stdout[-3000:])
            detail = "\n".join(errors or ["(no ! line, see main.log)"])
            raise SystemExit(f"pdflatex failed:\n{detail}")
    return log


def report(log: str) -> int:
    overfull = re.findall(r"^Overfull \\hbox \(([\d.]+)pt too wide\)", log, flags=re.M)
    underfull = re.findall(r"^Underfull \\[hv]box", log, flags=re.M)
    pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)
    if not pages:
        raise SystemExit("could not read the page count from main.log")
    wide = sum(1 for pt in overfull if float(pt) > 10)
    print(f"Overfull \\hbox: {len(overfull)} ({wide} wider than 10pt)")
    print(f"Underfull boxes: {len(underfull)}")
    return int(pages.group(1))


def write_tarball(figures: list[str]) -> None:
    # why: arXiv wants main.tex and the figures at the root, and nothing else
    with tarfile.open(TAR, "w:gz") as tar:
        for name in ["main.tex", *figures]:
            tar.add(ARXIV / name, arcname=name)


if __name__ == "__main__":
    shutil.rmtree(ARXIV, ignore_errors=True)
    ARXIV.mkdir()
    figures = copy_figures()
    write_tex()
    pages = report(compile_tex())
    write_tarball(figures)
    print(f"wrote {TAR.name} ({len(figures)} figures), arxiv/main.pdf ({pages} pages)")
