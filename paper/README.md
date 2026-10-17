# VecForge 2.0 paper

**VecForge: A Local-First Hybrid Vector Database with Persistent Agent Memory and Order-Preserving Grover-Inspired Reranking**
Suneel Bose K, ArcGX TechLabs Private Limited

| File | What it is |
| --- | --- |
| `VecForge-2.0-Paper.pdf` | Final paper, A4, ready to upload |
| `VecForge-2.0-Paper.docx` | The same paper in Word, for editing or for venues that want .docx |
| `VecForge-2.0-arXiv-source.tar.gz` | LaTeX source package for arXiv (`main.tex` plus figures), built by `build_arxiv.py` |
| `VecForge-2.0-IJCSE.docx` | The paper in the IJCSE journal template (two-column, US Letter), built by `build_ijcse.py`; `VecForge-2.0-IJCSE.pdf` is a LibreOffice preview |
| `PUBLISHING.md` | Step-by-step guide: arXiv endorsement, licence, upload, then ORCID, Google Scholar and ResearchGate |
| `vecforge-paper.md` | Source text (Markdown with LaTeX math and citation keys) |
| `references.bib` | Bibliography (BibTeX) |
| `figures/` | Figures; `figures/src/` holds the diagram sources (Mermaid and SVG) |
| `experiments/` | Scripts that produce every number in the evaluation, with `results.json` and `results2.json` |

## Rebuild

```bash
# E1 to E7: VecForge 2.0 against the 1.x code (about 15 minutes on 4 cores)
python paper/experiments/run_experiments.py
python paper/experiments/make_figures.py

# E8 to E18: SciFact retrieval quality, encryption, deletes, filters, ingestion,
# memory deduplication, MCP, REST, security overhead, stage breakdown, threads
export SCIFACT_DIR=/path/to/scifact      # BEIR SciFact: corpus.jsonl, queries.jsonl, qrels/
export MINILM_DIR=/path/to/minilm        # all-MiniLM-L6-v2 as ONNX (model.onnx, tokenizer.json)
python paper/experiments/run_experiments2.py            # or --only E9,E10 to rerun a subset
python paper/experiments/make_figures2.py

# Diagrams (Mermaid and hand-drawn SVG), then the paper itself
node paper/figures/src/render_diagrams.cjs              # needs MERMAID_JS pointing at mermaid.min.js
node paper/figures/src/render_svg_figs.cjs
python paper/build_paper.py                             # PDF + DOCX
python paper/build_arxiv.py                             # arXiv LaTeX source package (needs pdflatex)
python paper/build_ijcse.py                             # IJCSE journal version in the journal's Word template
```

`build_paper.py` needs pandoc 3, Node with `playwright`, KaTeX (set `KATEX_DIR`), pypdf and reportlab.
The experiments need `onnxruntime` and `tokenizers` for the MiniLM encoder and the `mcp` client package.

## What is in the paper

- One colour system for every figure (`figures/src/palette.json`): the Okabe-Ito colour-blind-safe hues, each with one meaning across the paper, checked under simulated protanopia, deuteranopia and tritanopia.
- 21 figures in several forms: block diagram, flowcharts, sequence diagrams, an entity-relationship
  diagram, a state diagram, a geometric diagram, a permission matrix, a methodology diagram, and bar,
  line, scatter, histogram, cumulative-distribution, stacked and precision-recall charts.
- 10 tables, including retrieval quality with bootstrap confidence intervals and significance tests.
- 18 experiments covering every feature, with the negative results reported alongside the positive ones.

## Where to publish for free

All of these cost nothing to submit or to read.

| Venue | What you get | Format | Notes |
| --- | --- | --- | --- |
| **arXiv** (cs.DB, cross-list cs.IR and cs.AI) | The standard preprint server for computer science; indexed by Google Scholar and Semantic Scholar within days | PDF accepted; LaTeX source preferred | Since January 2026 a first-time submitter needs a **personal endorsement** from an established arXiv author in computer science unless they have both an academic email and a prior arXiv paper. See `PUBLISHING.md`. |
| **Zenodo** | A permanent **DOI** in minutes, run by CERN | PDF + Word + code archive | Best first step. Link the GitHub repository to Zenodo and every GitHub release gets its own DOI automatically. Choose "Publication / Preprint" for the paper. |
| **TechRxiv** (IEEE) | Engineering preprint server with a DOI | PDF or Word | Light moderation, no endorsement needed. |
| **OSF Preprints** | Preprint with a DOI | PDF or Word | Simple and broad. |
| **Peer-reviewed workshops** (for example SIGIR, CIKM, VLDB or NeurIPS workshops on retrieval, data systems or agents) | Peer review and a citable workshop paper | Usually the ACM or IEEE two-column template, 4 to 8 pages | Submission is free; attending may cost money. Check each call for papers. |

Venues to be careful with:

- **Journal of Open Source Software (JOSS)** is free and peer-reviewed, but only accepts software under an OSI-approved open-source licence. VecForge uses the Business Source License 1.1, so it is not eligible unless the licence changes.
- Avoid any journal that emails you first, promises acceptance in days, or asks for a fee after acceptance. Check a journal in the DOAJ (doaj.org) before submitting.

## Recommended path

The detailed arXiv walkthrough, including the endorsement rules that changed in January 2026, is in `PUBLISHING.md`.

1. **Zenodo now.** Upload the PDF, the Word file and a ZIP of the repository at tag v2.0.0. You get a DOI immediately. Add the DOI to `CITATION.cff`, the README and the website.
2. **arXiv next.** Get an endorsement for cs.DB, then upload `VecForge-2.0-arXiv-source.tar.gz` with cs.IR and cs.AI as cross-lists. Use the abstract from the paper (arXiv limits it to 1,920 characters; this one is 1,916).
3. **A workshop later.** Reformat into the venue's template (ACM `acmart` or IEEE `IEEEtran`), extend the SciFact evaluation to more BEIR datasets as Section 13 suggests, and submit for peer review.

## Format of this paper

The paper follows the usual structure for a systems paper in computer science:

1. Title, author, affiliation, abstract (about 250 words), keywords
2. Introduction, ending with a numbered list of contributions
3. Background and related work
4. System design
5. The core technical contribution, with propositions and proofs (Grover reranking)
6. A second contribution (agent memory)
7. Evaluation: setup, one subsection per question, tables and figures, honest negative results
8. Discussion and limitations
9. Conclusion
10. Availability (code, package, data)
11. References in IEEE numeric style

## Before you submit

- [ ] Read the whole paper once more and check every number against `experiments/results.json` and `experiments/results2.json`.
- [ ] Decide on an AI-assistance statement. arXiv and most publishers do not allow an AI tool to be listed as an author, and many ask authors to disclose substantial use of AI tools in a short acknowledgement. Check the policy of the venue you choose.
- [ ] Add funding or acknowledgements if any apply.
- [x] ORCID iD 0009-0000-9282-2287 is on the author line, in `CITATION.cff` and on the website. Link it to your arXiv and Zenodo accounts too.
- [ ] After publishing, add the DOI or arXiv ID to `CITATION.cff`, the README, the website and your portfolio.
