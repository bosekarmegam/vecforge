# VecForge 2.0 paper

**VecForge: A Local-First Hybrid Vector Database with Persistent Agent Memory and a Corrected Grover-Inspired Reranker**
Suneel Bose K, ArcGX TechLabs Private Limited

| File | What it is |
| --- | --- |
| `VecForge-2.0-Paper.pdf` | Final paper, A4, ready to upload |
| `VecForge-2.0-Paper.docx` | The same paper in Word, for editing or for venues that want .docx |
| `vecforge-paper.md` | Source text (Markdown with LaTeX math and citation keys) |
| `references.bib` | Bibliography (BibTeX) |
| `figures/` | Figures; `figures/src/` holds the architecture diagram source |
| `experiments/` | Scripts that produce every number in the evaluation, and `results.json` |

## Rebuild

```bash
python paper/experiments/run_experiments.py      # about 15 minutes on 4 cores
python paper/experiments/make_figures.py
python paper/build_paper.py                      # PDF + DOCX
```

`build_paper.py` needs pandoc 3, Node with `playwright`, pypdf and reportlab.

## Where to publish for free

All of these cost nothing to submit or to read.

| Venue | What you get | Format | Notes |
| --- | --- | --- | --- |
| **arXiv** (cs.DB, cross-list cs.IR and cs.AI) | The standard preprint server for computer science; indexed by Google Scholar and Semantic Scholar within days | PDF accepted; LaTeX source preferred | First-time submitters in a category need an **endorsement** from an existing arXiv author in that category. Ask a colleague or a professor who has published in cs.DB or cs.IR. |
| **Zenodo** | A permanent **DOI** in minutes, run by CERN | PDF + Word + code archive | Best first step. Link the GitHub repository to Zenodo and every GitHub release gets its own DOI automatically. Choose "Publication / Preprint" for the paper. |
| **TechRxiv** (IEEE) | Engineering preprint server with a DOI | PDF or Word | Light moderation, no endorsement needed. |
| **OSF Preprints** | Preprint with a DOI | PDF or Word | Simple and broad. |
| **Peer-reviewed workshops** (for example SIGIR, CIKM, VLDB or NeurIPS workshops on retrieval, data systems or agents) | Peer review and a citable workshop paper | Usually the ACM or IEEE two-column template, 4 to 8 pages | Submission is free; attending may cost money. Check each call for papers. |

Venues to be careful with:

- **Journal of Open Source Software (JOSS)** is free and peer-reviewed, but only accepts software under an OSI-approved open-source licence. VecForge uses the Business Source License 1.1, so it is not eligible unless the licence changes.
- Avoid any journal that emails you first, promises acceptance in days, or asks for a fee after acceptance. Check a journal in the DOAJ (doaj.org) before submitting.

## Recommended path

1. **Zenodo now.** Upload the PDF, the Word file and a ZIP of the repository at tag v2.0.0. You get a DOI immediately. Add the DOI to `CITATION.cff`, the README and the website.
2. **arXiv next.** Get an endorsement for cs.DB, then submit the PDF with cs.IR and cs.AI as cross-lists. Use the abstract from the paper (arXiv limits it to 1,920 characters; this one fits).
3. **A workshop later.** Reformat into the venue's template (ACM `acmart` or IEEE `IEEEtran`), add the BEIR retrieval-quality evaluation described in Section 7, and submit for peer review.

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

- [ ] Read the whole paper once more and check every number against `experiments/results.json`.
- [ ] Decide on an AI-assistance statement. arXiv and most publishers do not allow an AI tool to be listed as an author, and many ask authors to disclose substantial use of AI tools in a short acknowledgement. Check the policy of the venue you choose.
- [ ] Add funding or acknowledgements if any apply.
- [ ] Add an ORCID iD (free at orcid.org) to the author line and to your Zenodo and arXiv profiles.
- [ ] After publishing, add the DOI or arXiv ID to `CITATION.cff`, the README, the website and your portfolio.
