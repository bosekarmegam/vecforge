# Publishing the VecForge 2.0 paper

A step-by-step guide for putting the paper on arXiv, and for making sure it then shows up on your ORCID record, Google Scholar and ResearchGate. Everything here was checked against the live arXiv, Google Scholar and ResearchGate help pages in October 2026.

Files you will use:

| File | Use |
| --- | --- |
| `VecForge-2.0-arXiv-source.tar.gz` | What you upload to arXiv: `main.tex` plus the figure PNGs, nothing else |
| `VecForge-2.0-Paper.pdf` | The reading copy for your website, ResearchGate and anywhere that wants a PDF |
| `VecForge-2.0-Paper.docx` | The Word version, for venues that ask for .docx |
| `arxiv/main.pdf` | What arXiv will produce from the source, so you can check it first |

Rebuild the arXiv package after any change to the paper with `python3 paper/build_arxiv.py` (it runs pdflatex three times and refuses to package a build with errors).

## 1. Before you start: three things that take time

1. **Endorsement.** Since 21 January 2026 arXiv endorses a first-time submitter automatically only if they have **both** an email address at an academic or research institution **and** already own an arXiv paper in the same subject domain. A company address such as arcgx.in does not count, and this is your first arXiv paper, so you will need a **personal endorsement** from an established arXiv author in computer science. Section 3 explains how to get one. Start this first; it is the only step that depends on someone else.
2. **ORCID.** You have an iD (0009-0000-9282-2287). Link it to your arXiv account at <https://arxiv.org/user/confirm_orcid_id> before you submit, so the paper is tied to you and not just to your name.
3. **Google Scholar and ResearchGate accounts.** Create them now if you have not; both take minutes, and Scholar needs a few days after the paper appears to pick it up.

## 2. Choose the licence (decide once, it cannot be changed)

arXiv asks you to pick a licence during submission. The choice is permanent for that version.

| Licence | Choose it if |
| --- | --- |
| **arXiv.org perpetual, non-exclusive licence** (recommended) | You may later submit the paper to a conference or journal. It gives arXiv only the right to distribute the paper and keeps every other right with you, which is what most publishers expect. |
| CC BY 4.0 | You want the widest possible reuse and are sure no future publisher will object. Many publishers accept it, but some do not. |
| CC BY-NC-SA or CC BY-NC-ND | You want to stop commercial reuse. |
| CC0 | Not recommended; you give up copyright entirely. |

The paper's licence is separate from the software licence. VecForge stays under the Business Source License whatever you choose for the paper.

## 3. Get endorsed

1. Go to <https://arxiv.org/submit>, start a new submission, and choose the primary category **cs.DB (Databases)**. arXiv will tell you that you need endorsement and email you an **endorsement request** with a link and a six-character code.
2. Find an endorser. The best choice is someone you know who publishes in computer science on arXiv. If you do not know one:
   - Open the arXiv page of a paper you cite (for example the BEIR paper, arXiv:2104.08663, or the FAISS paper, arXiv:2401.08281) and click **"Which authors of this paper are endorsers?"** at the bottom of the abstract page.
   - Look for computer science faculty near you (NIT Tiruchirappalli, IIT Madras, Anna University) who have recent arXiv papers in cs.IR, cs.DB or cs.AI; their arXiv abstract pages show the same link.
   - Any endorser for the **cs** domain can endorse you for cs.DB; the domain is the whole of computer science, not the single category.
3. Write to **one or two** people, not a list. A short message works: who you are, one sentence on the paper, the endorsement link from arXiv's email, a link to your ORCID record (<https://orcid.org/0009-0000-9282-2287>) and to the PDF on <https://bosekarmegam.github.io/vecforge/>. arXiv explicitly says not to mass-mail endorsers or to send repeat requests.
4. One positive endorsement is enough. You will get an email from arXiv when it is recorded; then continue the submission.

If you cannot find an endorser within a couple of weeks, **TechRxiv** (IEEE) and **Zenodo** need no endorsement and give a DOI the same day; the paper's README lists them. You can still submit to arXiv later.

## 4. Check the source package

The package was built to arXiv's rules, but check it once yourself:

```bash
tar -tzf paper/VecForge-2.0-arXiv-source.tar.gz     # should list main.tex and PNG files only
pdffonts paper/arxiv/main.pdf                       # every font "emb yes", no "Type 3"
```

What is already taken care of: one `main.tex` at the root; figures flattened to the root as PNG; no `.bib` or `.bbl` (the references are written into `main.tex`); no `\today`; no hidden files; no comments; page numbers and the "Preprint" line come from the template, not from a journal style.

## 5. Submit

1. **Start**: <https://arxiv.org/submit>. Confirm you are the author and agree to the policies.
2. **Licence**: pick the one you chose in Section 2.
3. **Upload**: choose the `.tar.gz` file. arXiv unpacks it and lists the files; it should show `main.tex` and 21 PNG files. If it flags any file as unnecessary, remove it.
4. **Process**: pick **PDFLaTeX** when asked for the processor (arXiv will suggest it). Read the TeX log to the end; there must be no errors. Open the PDF arXiv produced and page through it. It should match `paper/arxiv/main.pdf`.
5. **Metadata**. Copy the text exactly, with no LaTeX and no line breaks:
   - Title: `VecForge: A Local-First Hybrid Vector Database with Persistent Agent Memory and Order-Preserving Grover-Inspired Reranking`
   - Authors: `Suneel Bose K`
   - Abstract: the abstract from the paper, as one paragraph with single spaces. It is under arXiv's 1,920-character limit.
   - Comments (optional): `25 pages, 21 figures, 10 tables. Code and experiments: https://github.com/bosekarmegam/vecforge`
   - Primary category: `cs.DB`. Cross-lists: `cs.IR` and `cs.AI`.
   - ACM class (optional): `H.3.3; H.2.4`
   - Report number, journal reference and DOI: leave empty. arXiv assigns its own DOI.
6. **Preview and submit**. Submissions made before 14:00 US Eastern time on a weekday are announced that evening (about 20:00 Eastern, which is 05:30 the next morning in India). Submissions after the cut-off go out the next working day. You receive the arXiv identifier (for example `2610.NNNNN`) by email at announcement.

## 6. After it is announced

1. **DOI.** Every arXiv paper gets a DataCite DOI automatically: `https://doi.org/10.48550/arXiv.<id>`. It points to the abstract page and survives new versions.
2. **Update the records.** Put the arXiv identifier and DOI into `CITATION.cff` (`identifiers` and `preferred-citation`), the README's citation block, `index.html`, your portfolio and GitHub profile. Replacing the paper later (a v2) keeps the same identifier and DOI.
3. **Claim ownership.** The paper is linked to the account that submitted it. Keep the paper password from the confirmation email; it is needed if you ever change accounts.
4. **ORCID record.** With your ORCID linked to arXiv, add the paper to your ORCID record from **Works, Add, Search and link, DataCite**, or paste the DOI under **Add DOI**. If you enable DataCite as a trusted organisation in ORCID (**Account settings, Trusted organisations**), later DataCite DOIs are added automatically.

## 7. Google Scholar

Google Scholar indexes arXiv itself. There is nothing to upload; the paper usually appears in Scholar a few days to a few weeks after announcement.

To have it appear on **your profile** automatically:

1. Sign in at <https://scholar.google.com/citations> with a personal Google account (Scholar recommends personal over institutional, so you keep the profile if you change jobs). Create the profile with your name as it is on the paper, `Suneel Bose K`, and your affiliation.
2. In the profile menu choose **Configure article updates** and select **Apply updates automatically**. New articles that Scholar matches to your name are then added without you doing anything; citation counts update automatically in every mode.
3. Set the profile to **public** (Edit next to your name, tick "Make my profile public"). Note: a profile is listed in Scholar search results only when it also has a verified email at a university or research institute; a company address may not qualify. The profile still works and can be linked from your website either way.
4. If the paper has not appeared after about three weeks, use **Add articles** in the profile menu and search for the title; if Scholar has not indexed it yet, **Add article manually** and Scholar merges the record when it finds the arXiv version.
5. Add the profile link to your portfolio once it is public (`https://scholar.google.com/citations?user=<id>`).

## 8. ResearchGate

ResearchGate has no automatic feed from arXiv. It builds publication pages from public metadata and uses name matching to suggest items to you, so your paper may turn up by itself after a few weeks, but do not wait for that.

1. Sign in at <https://www.researchgate.net>. If you are asked to verify that you are a researcher, the arXiv page and the ORCID record are accepted evidence.
2. First search ResearchGate for the paper title or its DOI. If a page already exists (created from the arXiv metadata), open it, click **More** under the stats and choose **Claim authorship, is this your work?**.
3. If there is no page: **Add new** (top right), choose **Preprint**, fill in the title, abstract, date and the arXiv DOI, add yourself as author, and upload `VecForge-2.0-Paper.pdf` as a public full text. The arXiv licence you chose allows this because you remain the copyright holder.
4. When the paper is later published in a venue, use **Add published version** under the preprint to link the two pages.
5. In **Settings, Account**, connect your ORCID iD so ResearchGate and ORCID stay in step.

## 9. One-page checklist

- [ ] ORCID linked to the arXiv account
- [ ] Endorsement requested and received for cs.DB
- [ ] Licence chosen (arXiv non-exclusive unless you have a reason to choose CC BY)
- [ ] `VecForge-2.0-arXiv-source.tar.gz` uploaded, processed with PDFLaTeX, log clean, PDF checked page by page
- [ ] Metadata entered as plain text; cs.DB primary, cs.IR and cs.AI cross-lists
- [ ] After announcement: arXiv id and DOI added to CITATION.cff, README, website, portfolio, profile
- [ ] Paper added to the ORCID record
- [ ] Google Scholar profile public with automatic updates on
- [ ] ResearchGate preprint page claimed or created, full text uploaded
