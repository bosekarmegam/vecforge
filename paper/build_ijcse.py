# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under the Business Source License 1.1 (BSL 1.1)

"""
Build the IJCSE journal version of the paper in the journal's own Word template.

    python paper/build_ijcse.py

The text comes from vecforge-paper.md and is adapted to the IJCSE author
instructions: a 250-word abstract, five keywords in alphabetical order,
"Fig. N" captions below figures and table captions above tables, citations
after punctuation, the first mention of each figure and table in bold,
numbered display equations, and Conflicts of Interest and Funding sections
before the references. The body is converted with pandoc and then placed
inside paper/ijcse/IJCSE-IEEE-INDEX-PAPER-FORMAT.docx, so the journal's
banner, headers, fonts and heading numbering are kept. Wide figures and all
tables span both columns through continuous section breaks.

Needs pandoc 3 and python-docx; LibreOffice (soffice) for the PDF preview.
Outputs paper/VecForge-2.0-IJCSE.docx and paper/VecForge-2.0-IJCSE.pdf.
"""

from __future__ import annotations

import copy
import io
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn

HERE = Path(__file__).resolve().parent
SRC = HERE / "vecforge-paper.md"
TEMPLATE = HERE / "ijcse" / "IJCSE-IEEE-INDEX-PAPER-FORMAT.docx"
OUT = HERE / "VecForge-2.0-IJCSE.docx"
PDF = HERE / "VecForge-2.0-IJCSE.pdf"

TITLE = "VecForge: A Local-First Hybrid Vector Database with Persistent Agent Memory and Order-Preserving Grover-Inspired Reranking"
AUTHOR = "Suneel Bose K"
AFFILIATION = "ArcGX TechLabs Private Limited, Tiruchirappalli, Tamil Nadu, India."
EMAIL = "suneelbose@arcgx.in"
ORCID = "https://orcid.org/0009-0000-9282-2287"
KEYWORDS = "agent memory, amplitude amplification, hybrid retrieval, Model Context Protocol, vector database"
ABSTRACT = (
    "Retrieval-augmented applications and AI assistants need a vector store that runs inside the user's own process, "
    "keeps data encrypted on disk, combines semantic and keyword retrieval, and remembers information across "
    "conversations. This paper presents VecForge 2.0, an embedded Python vector database built for these needs. One "
    "SQLite file, optionally encrypted with SQLCipher, holds documents, vectors, metadata, access keys, agent sessions "
    "and index snapshots. Queries fuse FAISS dense search with an incremental BM25 index, and namespace and deletion "
    "constraints apply inside both indexes. A memory layer, served over the Model Context Protocol, ranks memories by "
    "relevance, importance and recency. The paper also proves that the Grover-inspired reranker of the previous "
    "release applies only the diffusion half of Grover's algorithm, so it can only keep or reverse a ranking, and "
    "replaces it with an O(N) closed form of the full algorithm that preserves order. Eighteen experiments evaluate "
    "every feature. On the SciFact benchmark, hybrid fusion reaches an nDCG@10 of 0.707, against 0.645 for dense "
    "search, 0.665 for BM25 and 0.681 for reciprocal rank fusion. Compared with the previous release, in-index "
    "constraints raise recall in small namespaces from 0.5% to 100%, incremental indexing makes insertion 183 times "
    "faster, and snapshots make reopening 9 to 17 times faster, while encryption and access control add little cost. "
    "The evaluation also reports where the design falls short, including slow, highly selective filters and memory "
    "deduplication that confuses paraphrases with changed facts. A single local file can therefore serve both "
    "retrieval and long-term memory for AI agents."
)
BACK_MATTER = """# Data Availability {.unnumbered}

The source code, documentation and experiment scripts are available at <https://github.com/bosekarmegam/vecforge>, and the package installs with `pip install vecforge` (version 2.0.0). The scripts `paper/experiments/run_experiments.py` and `paper/experiments/run_experiments2.py` reproduce every number in this paper, and their raw output is released with them. The SciFact benchmark is publicly available in the BEIR collection.

# Conflicts of Interest {.unnumbered}

The author is affiliated with ArcGX TechLabs Private Limited, which develops VecForge and distributes it under the Business Source License 1.1. The author declares no other conflict of interest.

# Funding Statement {.unnumbered}

This research received no external funding.

"""
UNNUMBERED = {"Data Availability", "Conflicts of Interest", "Funding Statement", "References"}
COLUMN_FIGURES = {"fig-grover-geometry.png", "fig-ingest.png"}  # small enough to sit in one column

# page geometry: Letter, 1 inch top and bottom, 0.25 inch column gap; the sides use the template's own
# 0.5 inch margins, because the journal banner in its header is laid out for that width
SIDE_TW = 720
TEXT_WIDTH_TW = 12240 - 2 * SIDE_TW
COL_GAP_TW = 360
COL_WIDTH_TW = (TEXT_WIDTH_TW - COL_GAP_TW) // 2
EMU_PER_TW = 635
HEADER_FIRST, HEADER_DEFAULT, FOOTER = "rId7", "rId42", "rId8"  # parts that exist in the template


# ─── text ─────────────────────────────────────────────────────────────
def journal_markdown() -> tuple[str, list[tuple[str, float]]]:
    """Adapt the paper's Markdown to the journal's rules. Returns the text and (image, width %) in order."""
    text = SRC.read_text(encoding="utf-8")
    body = text.split("\n---\n", 1)[1]
    renames = {
        "# Evaluation Method\n": "# Materials and Methods\n",
        "# Results\n": "# Results and Discussion\n",
        "# Discussion and Limitations\n": "# Limitations and Threats to Validity\n",
        "Section 10 sets out the evaluation method and Section 11 the results. Section 12 discusses limitations and threats to validity":
        "Section 10 sets out the materials and methods and Section 11 the results. Section 12 discusses limitations and threats to validity",
    }
    for old, new in renames.items():
        assert old in body, old
        body = body.replace(old, new)
    # the journal's back matter replaces the Availability section
    start, end = body.index("# Availability\n"), body.index("# References {.unnumbered}")
    body = body[:start] + BACK_MATTER + body[end:]
    # multi-letter names in equations as operators, so every renderer sets them as one word
    body = re.sub(r"\\mathrm\{([a-z]{2,})\}", r"\\operatorname{\1}", body)
    # citations go after the punctuation that follows them
    body = re.sub(r"\s*(\[@[^\]]+\])([.,;:])", r"\2 \1", body)
    # the first mention of each figure and table in the text is bold
    seen: set[str] = set()
    out = []
    for line in body.split("\n"):
        if line.startswith(("![", "Table:", "|", "#")):
            out.append(line)
            continue

        def bold(m: re.Match[str]) -> str:
            key = f"{m.group(1)} {m.group(2)}"
            if key in seen:
                return m.group(0)
            seen.add(key)
            return f"**{m.group(0)}**"

        out.append(re.sub(r"(?<!\*)\b(Figure|Table) (\d+)([ab]?)\b(?!\.\*\*)", bold, line))
    body = "\n".join(out)
    images = [(Path(p).name, float(w)) for p, w in re.findall(r"!\[.*?\]\(([^)]+)\)\{width=(\d+)%\}", body)]
    return body, images


def pandoc_docx(markdown: str, out: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, dir=HERE) as f:
        f.write(markdown)
        md = Path(f.name)
    try:
        subprocess.run(
            ["pandoc", str(md), "-o", str(out), "--citeproc", "--bibliography", str(HERE / "references.bib"),
             "--csl", str(HERE / "ieee-numeric.csl"), "--resource-path", str(HERE)],
            check=True,
        )
    finally:
        md.unlink()


# ─── XML helpers ──────────────────────────────────────────────────────
NS = nsdecls("w", "r", "wp", "a", "pic", "m")
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
RPR_ORDER = ["rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps", "strike", "dstrike", "outline", "shadow",
             "emboss", "imprint", "noProof", "snapToGrid", "vanish", "webHidden", "color", "spacing", "w", "kern",
             "position", "sz", "szCs", "highlight", "u", "effect", "bdr", "shd", "fitText", "vertAlign", "rtl", "cs",
             "em", "lang", "eastAsianLayout", "specVanish", "oMath"]


def el(xml: str):
    return parse_xml(xml.replace("<w:p>", f"<w:p {NS}>", 1) if xml.startswith("<w:p>") else xml)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def set_rpr(run, **props) -> None:
    """Merge properties into a run's rPr and keep the children in schema order."""
    rpr = run.find(qn("w:rPr"))
    if rpr is None:
        rpr = parse_xml(f"<w:rPr {nsdecls('w')}/>")
        # in a math run the w:rPr follows m:rPr; in a text run it comes first
        mrpr = run.find(qn("m:rPr"))
        if mrpr is not None:
            mrpr.addnext(rpr)
        else:
            run.insert(0, rpr)
    for name, value in props.items():
        for old in rpr.findall(qn(f"w:{name}")):
            rpr.remove(old)
        if value is None:
            continue
        rpr.append(parse_xml(f"<w:{name} {nsdecls('w')} {value}/>"))
    kids = sorted(rpr, key=lambda c: RPR_ORDER.index(c.tag.split("}")[1]) if c.tag.split("}")[1] in RPR_ORDER else 99)
    for c in list(rpr):
        rpr.remove(c)
    for c in kids:
        rpr.append(c)


def size_runs(p, half_points: int, bold: bool | None = None) -> None:
    for r in p.iter(qn("w:r"), qn("m:r")):
        props = {"sz": f'w:val="{half_points}"', "szCs": f'w:val="{half_points}"'}
        if bold is not None and r.tag == qn("w:r"):
            props.update({"b": "" if bold else None, "bCs": "" if bold else None})
        set_rpr(r, **props)


def new_ppr(p, *, style: str | None = None, keep_next: bool = False, num: tuple[int, int] | None = None,
            tabs: str = "", spacing: str = "", ind: str = "", jc: str | None = None, extra_rpr: str = "") -> None:
    """Replace a paragraph's properties with a clean, schema-ordered set."""
    old = p.find(qn("w:pPr"))
    if old is not None:
        p.remove(old)
    parts = []
    if style:
        parts.append(f'<w:pStyle w:val="{style}"/>')
    if keep_next:
        parts.append("<w:keepNext/>")
    parts.append("<w:widowControl/>")
    if num:
        parts.append(f'<w:numPr><w:ilvl w:val="{num[1]}"/><w:numId w:val="{num[0]}"/></w:numPr>')
    if tabs:
        parts.append(f"<w:tabs>{tabs}</w:tabs>")
    if spacing:
        parts.append(f"<w:spacing {spacing}/>")
    if ind:
        parts.append(f"<w:ind {ind}/>")
    if jc:
        parts.append(f'<w:jc w:val="{jc}"/>')
    if extra_rpr:
        parts.append(f"<w:rPr>{extra_rpr}</w:rPr>")
    p.insert(0, parse_xml(f"<w:pPr {nsdecls('w')}>{''.join(parts)}</w:pPr>"))


def sect_pr(columns: int, *, first: bool = False) -> str:
    refs = (f'<w:headerReference w:type="first" r:id="{HEADER_FIRST}"/>' if first else "") + \
        f'<w:headerReference w:type="default" r:id="{HEADER_DEFAULT}"/>' + \
        (f'<w:footerReference w:type="first" r:id="{FOOTER}"/>' if first else "") + \
        f'<w:footerReference w:type="default" r:id="{FOOTER}"/>'
    cols = f'<w:cols w:num="2" w:space="{COL_GAP_TW}"/>' if columns == 2 else '<w:cols w:space="720"/>'
    return (f"<w:sectPr {nsdecls('w', 'r')}>{refs}" + ("" if first else '<w:type w:val="continuous"/>') +
            '<w:pgSz w:w="12240" w:h="15840"/>'
            f'<w:pgMar w:top="1440" w:right="{SIDE_TW}" w:bottom="1440" w:left="{SIDE_TW}" w:header="720" w:footer="720" w:gutter="0"/>' +
            ('<w:pgNumType w:start="1"/>' if first else "") + cols + ("<w:titlePg/>" if first else "") + "</w:sectPr>")


def section_break(columns: int, *, first: bool = False):
    """An almost invisible paragraph that ends the current section."""
    return parse_xml(
        f'<w:p {NS}><w:pPr><w:spacing w:before="0" w:after="0" w:line="20" w:lineRule="exact"/>'
        f'<w:rPr><w:sz w:val="2"/></w:rPr>{sect_pr(columns, first=first)}</w:pPr></w:p>'
    )


# ─── title block ──────────────────────────────────────────────────────
def title_block(logo_xml: str) -> list:
    sup = '<w:vertAlign w:val="superscript"/>'
    return [
        el(f'<w:p><w:pPr><w:pStyle w:val="Title"/><w:spacing w:before="120" w:after="200"/><w:ind w:left="0" w:right="0"/>'
           f'<w:jc w:val="center"/></w:pPr>{logo_xml}<w:r><w:t>{esc(TITLE)}</w:t></w:r></w:p>'),
        el(f'<w:p><w:pPr><w:spacing w:after="80"/><w:jc w:val="center"/></w:pPr>'
           f'<w:r><w:rPr><w:sz w:val="22"/></w:rPr><w:t>{AUTHOR}</w:t></w:r><w:r><w:rPr><w:sz w:val="22"/>{sup}</w:rPr><w:t>1</w:t></w:r></w:p>'),
        el(f'<w:p><w:pPr><w:spacing w:after="80"/><w:jc w:val="center"/></w:pPr>'
           f'<w:r><w:rPr><w:i/><w:sz w:val="20"/>{sup}</w:rPr><w:t>1</w:t></w:r>'
           f'<w:r><w:rPr><w:i/><w:sz w:val="20"/></w:rPr><w:t>{esc(AFFILIATION)}</w:t></w:r></w:p>'),
        el(f'<w:p><w:pPr><w:spacing w:after="40"/><w:jc w:val="center"/></w:pPr>'
           f'<w:r><w:rPr><w:i/><w:color w:val="0000FF"/><w:sz w:val="18"/>{sup}</w:rPr><w:t>1</w:t></w:r>'
           f'<w:r><w:rPr><w:i/><w:color w:val="0000FF"/><w:sz w:val="18"/></w:rPr><w:t>{EMAIL}</w:t></w:r></w:p>'),
        el(f'<w:p><w:pPr><w:spacing w:after="160"/><w:jc w:val="center"/></w:pPr>'
           f'<w:r><w:rPr><w:sz w:val="18"/></w:rPr><w:t xml:space="preserve">ORCID iD: {ORCID}</w:t></w:r></w:p>'),
        el('<w:p><w:pPr><w:tabs><w:tab w:val="left" w:pos="2900"/><w:tab w:val="left" w:pos="5600"/>'
           '<w:tab w:val="left" w:pos="8100"/></w:tabs><w:spacing w:after="200"/><w:ind w:left="560"/></w:pPr>'
           '<w:r><w:rPr><w:sz w:val="18"/></w:rPr><w:t>Received:</w:t><w:tab/><w:t>Revised:</w:t><w:tab/>'
           '<w:t>Accepted:</w:t><w:tab/><w:t>Published:</w:t></w:r></w:p>'),
        el(f'<w:p><w:pPr><w:widowControl/><w:spacing w:after="160"/><w:jc w:val="both"/></w:pPr>'
           f'<w:r><w:rPr><w:b/><w:i/><w:sz w:val="20"/></w:rPr><w:t xml:space="preserve">Abstract - </w:t></w:r>'
           f'<w:r><w:rPr><w:i/><w:sz w:val="20"/></w:rPr><w:t>{esc(ABSTRACT)}</w:t></w:r></w:p>'),
        el(f'<w:p><w:pPr><w:widowControl/><w:spacing w:after="240"/><w:jc w:val="both"/></w:pPr>'
           f'<w:r><w:rPr><w:b/><w:i/><w:sz w:val="20"/></w:rPr><w:t xml:space="preserve">Keywords - </w:t></w:r>'
           f'<w:r><w:rPr><w:i/><w:sz w:val="20"/></w:rPr><w:t>{esc(KEYWORDS)}</w:t></w:r></w:p>'),
    ]


# ─── assembly ─────────────────────────────────────────────────────────
def kind(p) -> str:
    """Classify a pandoc body element."""
    if p.tag == qn("w:tbl"):
        return "table"
    style = p.find(f"{{{W}}}pPr/{{{W}}}pStyle")
    s = style.get(qn("w:val")) if style is not None else ""
    if p.find(".//{http://schemas.openxmlformats.org/officeDocument/2006/math}oMathPara") is not None:
        return "equation"
    if p.find(f".//{{{W}}}drawing") is not None:
        return "figure"
    return {"Heading1": "h1", "Heading2": "h2", "Heading3": "h3", "ImageCaption": "figcaption",
            "TableCaption": "tablecaption", "Bibliography": "bib"}.get(s, "list" if p.find(f"{{{W}}}pPr/{{{W}}}numPr") is not None else "text")


def build() -> None:
    markdown, images = journal_markdown()
    words = len(ABSTRACT.split())
    assert words <= 250, f"abstract has {words} words"
    with tempfile.TemporaryDirectory() as tmp:
        mid = Path(tmp) / "body.docx"
        pandoc_docx(markdown, mid)
        src = Document(str(mid))
        doc = Document(str(TEMPLATE))
        body = doc.element.body

        # keep the journal logo from the template's first page, then clear the template's body
        logo = next(r for r in body.iter(qn("w:r")) if r.find(f".//{{{W}}}drawing") is not None)
        logo_xml = re.sub(r'\s(?:w14|wp14):[A-Za-z]+="[^"]*"', "", logo.xml)
        logo_xml = re.sub(r"<w:r [^>]*>", "<w:r>", logo_xml, count=1)
        for child in list(body):
            body.remove(child)
        for p in title_block(logo_xml):
            body.append(p)
        body.append(section_break(1, first=True))

        # list numbering: add a bullet and a decimal definition to the template's numbering part
        numbering = doc.part.numbering_part.element
        abs_ids = [int(x.get(qn("w:abstractNumId"))) for x in numbering.findall(qn("w:abstractNum"))]
        num_ids = [int(x.get(qn("w:numId"))) for x in numbering.findall(qn("w:num"))]
        list_nums = {}
        for i, (fmt, text) in enumerate((("bullet", "•"), ("decimal", "%1."))):
            aid, nid = max(abs_ids) + 1 + i, max(num_ids) + 1 + i
            absnum = parse_xml(
                f'<w:abstractNum {nsdecls("w")} w:abstractNumId="{aid}"><w:multiLevelType w:val="singleLevel"/>'
                f'<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="{fmt}"/><w:lvlText w:val="{text}"/>'
                f'<w:lvlJc w:val="left"/><w:pPr><w:ind w:left="360" w:hanging="270"/></w:pPr></w:lvl></w:abstractNum>')
            first_num = numbering.find(qn("w:num"))
            first_num.addprevious(absnum)
            numbering.append(parse_xml(f'<w:num {nsdecls("w")} w:numId="{nid}"><w:abstractNumId w:val="{aid}"/></w:num>'))
            list_nums[fmt] = nid
        src_formats = {}
        for num in src.part.numbering_part.element.findall(qn("w:num")):
            aid = num.find(qn("w:abstractNumId")).get(qn("w:val"))
            absn = src.part.numbering_part.element.find(f'{{{W}}}abstractNum[@{{{W}}}abstractNumId="{aid}"]')
            fmt = absn.find(f"{{{W}}}lvl/{{{W}}}numFmt").get(qn("w:val"))
            src_formats[num.get(qn("w:numId"))] = "bullet" if fmt == "bullet" else "decimal"

        mode, image_index, equation, figure_pending = 2, 0, 0, None
        stream = list(src.element.body)
        for node in stream:
            if node.tag == qn("w:sectPr"):
                continue
            if node.tag not in (qn("w:p"), qn("w:tbl")):  # body-level bookmarks around sections
                body.append(copy.deepcopy(node))
                continue
            k = kind(node)
            node = copy.deepcopy(node)

            # wide items span the page: tables (with their captions) and most figures (with theirs)
            wide = False
            if k in ("tablecaption", "table"):
                wide = True
            elif k == "figure":
                name = images[image_index][0]
                wide = name not in COLUMN_FIGURES
                figure_pending = wide
            elif k == "figcaption":
                wide = bool(figure_pending)
            want = 1 if wide else 2
            if want != mode:
                body.append(section_break(mode))
                mode = want

            # hyperlinks and images point at the pandoc file's relationships; move them over
            for h in node.iter(qn("w:hyperlink")):
                rid = h.get(qn("r:id"))
                if rid:
                    h.set(qn("r:id"), doc.part.relate_to(src.part.rels[rid].target_ref, RT.HYPERLINK, is_external=True))
            for blip in node.iter(qn("a:blip")):
                blob = src.part.related_parts[blip.get(qn("r:embed"))].blob
                rid, _ = doc.part.get_or_add_image(io.BytesIO(blob))
                blip.set(qn("r:embed"), rid)
            # character styles from pandoc do not exist in the template; use direct formatting instead
            for r in node.iter(qn("w:r")):
                rs = r.find(f"{{{W}}}rPr/{{{W}}}rStyle")
                if rs is not None:
                    name = rs.get(qn("w:val"))
                    if name == "VerbatimChar":
                        set_rpr(r, rStyle=None, rFonts='w:ascii="Courier New" w:hAnsi="Courier New" w:cs="Courier New"',
                                sz='w:val="18"', szCs='w:val="18"')
                    elif name == "Hyperlink":
                        set_rpr(r, rStyle=None, color='w:val="0563C1"', u='w:val="single"')
                    else:
                        set_rpr(r, rStyle=None)

            if k == "h1":
                text = "".join(t.text or "" for t in node.iter(qn("w:t"))).strip()
                numbered = text not in UNNUMBERED
                new_ppr(node, style="Heading1", keep_next=True, num=(5, 0) if numbered else None,
                        spacing='w:before="200" w:after="80"', ind='w:left="360" w:hanging="360"' if numbered else 'w:left="0" w:firstLine="0"')
                size_runs(node, 24, bold=True)
            elif k in ("h2", "h3"):
                level = 1 if k == "h2" else 2
                new_ppr(node, style="Heading2" if k == "h2" else "Heading3", keep_next=True, num=(5, level),
                        spacing='w:before="120" w:after="60"', ind='w:left="0" w:firstLine="0"')
                size_runs(node, 20, bold=k == "h2")
                for r in node.iter(qn("w:r")):
                    set_rpr(r, i="", iCs="")
            elif k == "equation":
                equation += 1
                maths = [copy.deepcopy(m) for m in node.iter(qn("m:oMath"))]
                chars = sum(len(t.text or "") for m in maths for t in m.iter(qn("m:t")))
                long_eq = chars > 34  # too wide to centre and still leave room for the number
                eq = parse_xml(f'<w:p {NS}>' + ("" if long_eq else "<w:r><w:tab/></w:r>") + "</w:p>")
                for m in maths:
                    eq.append(m)
                eq.append(parse_xml(f'<w:r {nsdecls("w")}><w:tab/><w:t>({equation})</w:t></w:r>'))
                centre = "" if long_eq else f'<w:tab w:val="center" w:pos="{COL_WIDTH_TW // 2}"/>'
                new_ppr(eq, tabs=f'{centre}<w:tab w:val="right" w:pos="{COL_WIDTH_TW}"/>',
                        spacing='w:before="80" w:after="80"')
                size_runs(eq, 16 if long_eq else 18)
                node = eq
            elif k == "figure":
                name, pct = images[image_index]
                image_index += 1
                width_tw = (COL_WIDTH_TW * min(1.0, pct / 100 * 1.6) if name in COLUMN_FIGURES
                            else TEXT_WIDTH_TW * min(1.0, pct / 100))
                for ext in list(node.iter(qn("wp:extent"))) + list(node.iter(qn("a:ext"))):
                    cx, cy = int(ext.get("cx")), int(ext.get("cy"))
                    new_cx = int(width_tw * EMU_PER_TW)
                    ext.set("cx", str(new_cx))
                    ext.set("cy", str(int(cy * new_cx / cx)))
                new_ppr(node, keep_next=True, spacing='w:before="160" w:after="60"', jc="center")
            elif k == "figcaption":
                first_t = next(t for t in node.iter(qn("w:t")) if t.text and t.text.strip())
                first_t.text = re.sub(r"^\s*Figure (\d+)\.\s*", r"Fig. \1 ", first_t.text)
                if first_t.text.startswith("Fig. ") and not first_t.text.endswith(" "):
                    first_t.text += " "
                first_t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                new_ppr(node, spacing='w:before="0" w:after="200"', jc="center")
                size_runs(node, 16, bold=True)
                figure_pending = None
            elif k == "tablecaption":
                new_ppr(node, keep_next=True, spacing='w:before="160" w:after="80"', jc="center")
                size_runs(node, 16, bold=True)
            elif k == "table":
                tblpr = node.find(qn("w:tblPr"))
                keep_tail = [c for c in tblpr if c.tag in (qn("w:tblCaption"), qn("w:tblDescription"))]
                borders = "".join(f'<w:{b} w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
                                  for b in ("top", "left", "bottom", "right", "insideH", "insideV"))
                fresh = parse_xml(
                    f'<w:tblPr {nsdecls("w")}><w:tblW w:w="5000" w:type="pct"/><w:jc w:val="center"/>'
                    f'<w:tblBorders>{borders}</w:tblBorders><w:tblLayout w:type="autofit"/>'
                    f'<w:tblCellMar><w:left w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar>'
                    f'<w:tblLook w:val="0000"/></w:tblPr>')
                for c in keep_tail:
                    fresh.append(c)
                tblpr.addprevious(fresh)
                node.remove(tblpr)
                rows = list(node.iter(qn("w:tr")))
                for ri, tr in enumerate(rows):
                    trpr = tr.find(qn("w:trPr"))
                    if trpr is None:
                        trpr = parse_xml(f'<w:trPr {nsdecls("w")}/>')
                        tr.insert(0, trpr)
                    trpr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
                    if ri == 0:
                        trpr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
                    for cp in tr.iter(qn("w:p")):
                        jc = cp.find(f"{{{W}}}pPr/{{{W}}}jc")
                        new_ppr(cp, keep_next=ri < len(rows) - 1, spacing='w:before="20" w:after="20"',
                                jc=jc.get(qn("w:val")) if jc is not None else None)
                        size_runs(cp, 18, bold=True if ri == 0 else None)
            elif k == "bib":
                new_ppr(node, spacing='w:before="0" w:after="40" w:line="276" w:lineRule="auto"', ind='w:left="400" w:hanging="400"', jc="both")
                size_runs(node, 18)
            elif k == "list":
                nid = node.find(f"{{{W}}}pPr/{{{W}}}numPr/{{{W}}}numId").get(qn("w:val"))
                new_ppr(node, style="BodyText", num=(list_nums[src_formats.get(nid, "bullet")], 0),
                        spacing='w:before="0" w:after="60"', jc="both")
                size_runs(node, 20)
            else:  # running text; left-aligned when a long code name would leave wide gaps in justified lines
                long_code = any(len(t.text or "") > 24 for r in node.iter(qn("w:r")) if r.find(f"{{{W}}}rPr/{{{W}}}rFonts") is not None
                                for t in r.iter(qn("w:t")))
                new_ppr(node, style="BodyText", spacing='w:before="0" w:after="60"', ind='w:firstLine="284"',
                        jc="left" if long_code else "both")
                size_runs(node, 20)
            body.append(node)

        body.append(parse_xml(sect_pr(mode)))

        # pandoc writes m:mcJc before m:count; the schema wants count first
        for mcpr in body.iter(qn("m:mcPr")):
            count = mcpr.find(qn("m:count"))
            if count is not None:
                mcpr.remove(count)
                mcpr.insert(0, count)

        # and m:dPr children in begChr, sepChr, endChr, grow, shp, ctrlPr order
        dpr_order = ["begChr", "sepChr", "endChr", "grow", "shp", "ctrlPr"]
        for dpr in body.iter(qn("m:dPr")):
            kids = sorted(dpr, key=lambda c: dpr_order.index(c.tag.split("}")[1]) if c.tag.split("}")[1] in dpr_order else 9)
            for c in list(dpr):
                dpr.remove(c)
            for c in kids:
                dpr.append(c)

        # unique ids for every drawing
        for n, d in enumerate(body.iter(qn("wp:docPr")), start=1):
            d.set("id", str(n))
        for n, d in enumerate(body.iter(qn("pic:cNvPr")), start=1):
            d.set("id", str(n))

        # keep only the template parts this document uses
        keep = {HEADER_FIRST, HEADER_DEFAULT, FOOTER}
        used = set(re.findall(r'r:(?:embed|id)="([^"]+)"', body.xml))
        for rid, rel in list(doc.part.rels.items()):
            if rel.reltype in (RT.IMAGE, RT.HEADER, RT.FOOTER) and rid not in used and rid not in keep:
                doc.part.drop_rel(rid)

        # header text: this is a research article by one author
        for rel in doc.part.rels.values():
            if rel.reltype == RT.HEADER:
                hdr = rel.target_part.element
                for t in hdr.iter(qn("w:t")):
                    t.text = {"(Review": "(Research", "Corresponding": "Suneel", "Author": "Bose", "et": "K",
                              "al. /": "/", "2023": "2026"}.get(t.text, t.text)

        doc.core_properties.title = TITLE
        doc.core_properties.author = AUTHOR
        doc.core_properties.keywords = KEYWORDS
        doc.core_properties.subject = "IJCSE submission"
        doc.save(str(OUT))
    print(f"wrote {OUT.name} (abstract {words} words, {image_index} figures, {equation} equations)")


def preview_pdf() -> None:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        print("soffice not found; skipped the PDF preview")
        return
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", tmp, str(OUT)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.move(str(Path(tmp) / f"{OUT.stem}.pdf"), PDF)
    print(f"wrote {PDF.name}")


if __name__ == "__main__":
    build()
    preview_pdf()
