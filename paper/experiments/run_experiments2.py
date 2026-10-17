# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under the Business Source License 1.1 (BSL 1.1)

"""
Extended experiments for the VecForge 2.0 paper (E8 to E18).

    MINILM_DIR=/path/to/minilm SCIFACT_DIR=/path/to/scifact/data \\
        python paper/experiments/run_experiments2.py [--only E8,E9]

E8   Retrieval quality on SciFact with all-MiniLM-L6-v2 (dense, BM25, hybrid,
     RRF, Grover consensus rerank), alpha tuned on train, reported on dev,
     with bootstrap intervals and paired randomisation tests
E9   SQLCipher encryption overhead
E10  Deletes: tombstones, search cost, and automatic compaction
E11  Metadata filters: adaptive widening vs a fixed candidate window
E12  Ingestion throughput by file format
E13  Memory deduplication: precision and recall over the similarity threshold
E14  MCP server: cold start and tool latency over stdio
E15  REST API latency over HTTP
E16  Audit log and role check overhead
E17  Per-stage latency breakdown of a search
E18  Concurrent search throughput
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from dedup_set import ROWS  # noqa: E402
from run_experiments import p50, p95, zipf_corpus  # noqa: E402

import vecforge.core.vault as vault_mod  # noqa: E402
import vecforge.search.cascade as cascade_mod  # noqa: E402
from vecforge import MemoryStore, VecForge  # noqa: E402
from vecforge.core.bm25 import BM25Engine  # noqa: E402
from vecforge.core.embedder import CallableEmbedder  # noqa: E402
from vecforge.core.indexer import FaissIndexer  # noqa: E402

OUT = HERE / "results2.json"
RNG = np.random.default_rng(20261017)


# ─── helpers ──────────────────────────────────────────────────────────
def minilm():  # type: ignore[no-untyped-def]
    from minilm import MiniLM

    return MiniLM()


def ndcg_at(ranked: list[str], rel: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in rel)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), k)))
    return float(dcg / idcg) if idcg else 0.0


def recall_at(ranked: list[str], rel: set[str], k: int = 100) -> float:
    return len(set(ranked[:k]) & rel) / len(rel) if rel else 0.0


def mrr_at(ranked: list[str], rel: set[str], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k]):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0


def bootstrap_ci(x: np.ndarray, n: int = 10_000, seed: int = 1) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_randomisation(a: np.ndarray, b: np.ndarray, n: int = 10_000, seed: int = 2) -> float:
    """Two-sided paired sign-flip randomisation test on the mean difference."""
    rng = np.random.default_rng(seed)
    d = a - b
    obs = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], size=(n, len(d)))
    return float(((np.abs((flips * d).mean(axis=1)) >= obs).sum() + 1) / (n + 1))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ─── E8 + E17 ─────────────────────────────────────────────────────────
def load_scifact(root: Path) -> tuple[dict[str, str], list[tuple[str, set[str]]], list[tuple[str, set[str]]]]:
    corpus = {}
    for line in open(root / "corpus.jsonl", encoding="utf-8"):
        d = json.loads(line)
        corpus[str(d["doc_id"])] = (d["title"] + " " + " ".join(d["abstract"])).strip()

    def claims(name: str) -> list[tuple[str, set[str]]]:
        out = []
        for line in open(root / name, encoding="utf-8"):
            c = json.loads(line)
            rel = {str(x) for x in c.get("cited_doc_ids", [])}
            if rel:
                out.append((c["claim"], rel))
        return out

    return corpus, claims("claims_train.jsonl"), claims("claims_dev.jsonl")


def latency_stages(path: str, model, best: float, dev) -> tuple[dict, dict]:  # type: ignore[no-untyped-def]
    """End-to-end latency and per-stage time for hybrid search with and without
    Grover reranking. The two modes use separate vault handles (so neither sees
    the other's query cache), are warmed up first, and run interleaved query by
    query with the order alternating, so drift on the machine hits both alike."""
    modes = (("hybrid", {}), ("hybrid + Grover", {"quantum_rerank": True}))
    stage: dict[str, dict[str, float]] = {m: {} for m, _ in modes}
    current = {"mode": modes[0][0]}

    def add(label: str, dt: float) -> None:
        st = stage[current["mode"]]
        st[label] = st.get(label, 0.0) + dt

    def wrap(obj, name, label):  # type: ignore[no-untyped-def]
        orig = getattr(obj, name)

        def timed(*a, **kw):  # type: ignore[no-untyped-def]
            t = time.perf_counter()
            try:
                return orig(*a, **kw)
            finally:
                add(label, time.perf_counter() - t)

        setattr(obj, name, timed)
        return orig

    dbs = {}
    for label, _ in modes:
        db = VecForge(path, embedder=CallableEmbedder(model.encode, dimension=384), index_type="flat")
        enc = db._embedder
        orig_encode = enc.encode

        def enc_timed(*a, _o=orig_encode, **k):  # type: ignore[no-untyped-def]
            t = time.perf_counter()
            try:
                return _o(*a, **k)
            finally:
                add("query embedding (MiniLM)", time.perf_counter() - t)

        enc.encode = enc_timed
        dbs[label] = db
    for i in range(30):
        for label, kw in modes:
            dbs[label].search(f"warm up query number {i} {label}", top_k=10, alpha=best, **kw)
    origs = [
        (FaissIndexer, "search", wrap(FaissIndexer, "search", "dense search (FAISS)")),
        (BM25Engine, "search", wrap(BM25Engine, "search", "sparse search (BM25)")),
        (cascade_mod, "weighted_linear_fusion_detailed",
         wrap(cascade_mod, "weighted_linear_fusion_detailed", "fusion")),
        (vault_mod.VecForge, "_hydrate", wrap(vault_mod.VecForge, "_hydrate", "hydration (SQLite)")),
        (vault_mod.VecForge, "_quantum_rerank", wrap(vault_mod.VecForge, "_quantum_rerank", "Grover rerank")),
    ]
    for st in stage.values():
        st.clear()
    ms: dict[str, list[float]] = {m: [] for m, _ in modes}
    try:
        for qi, (q, _) in enumerate(dev):
            order = modes if qi % 2 == 0 else tuple(reversed(modes))
            for label, kw in order:
                current["mode"] = label
                t = time.perf_counter()
                dbs[label].search(q, top_k=10, alpha=best, **kw)
                ms[label].append((time.perf_counter() - t) * 1000)
    finally:
        for obj, name, orig in origs:
            setattr(obj, name, orig)
        for db in dbs.values():
            db.close()
    lat, breakdown = {}, {}
    for label, _ in modes:
        parts = {k: v * 1000 / len(dev) for k, v in stage[label].items()}
        parts["other"] = max(0.0, float(np.mean(ms[label])) - sum(parts.values()))
        breakdown[label] = parts
        lat[label] = {"p50_ms": p50(ms[label]), "p95_ms": p95(ms[label]), "all_ms": ms[label]}
    return lat, breakdown


def e17_only(work: str) -> tuple[dict, dict]:
    """Rebuild the SciFact vault and rerun only the latency measurement."""
    corpus, _, dev = load_scifact(Path(os.environ["SCIFACT_DIR"]))
    model = minilm()
    ids = list(corpus)
    path = os.path.join(work, "scifact_lat.db")
    db = VecForge(path, embedder=CallableEmbedder(model.encode, dimension=384), index_type="flat")
    for i in range(0, len(ids), 512):
        db.add_batch([corpus[x] for x in ids[i : i + 512]], doc_ids=ids[i : i + 512])
    db.close()
    best = float(json.loads(OUT.read_text())["E8"]["best_alpha"]) if OUT.exists() else 0.5
    return latency_stages(path, model, best, dev)


def e8_e17(work: str) -> tuple[dict, dict]:
    root = Path(os.environ["SCIFACT_DIR"])
    corpus, train, dev = load_scifact(root)
    model = minilm()
    emb = CallableEmbedder(model.encode, dimension=384)
    ids = list(corpus)
    texts = [corpus[i] for i in ids]
    path = os.path.join(work, "scifact.db")
    db = VecForge(path, embedder=emb, index_type="flat")
    t0 = time.perf_counter()
    for i in range(0, len(texts), 512):
        db.add_batch(texts[i : i + 512], doc_ids=ids[i : i + 512])
    ingest_s = time.perf_counter() - t0

    alphas = [round(a, 1) for a in np.linspace(0, 1, 11)]

    def run(queries, alpha, quantum=False, k=100):  # type: ignore[no-untyped-def]
        return [[r.doc_id for r in db.search(q, top_k=k, alpha=alpha, quantum_rerank=quantum)] for q, _ in queries]

    def scores(runs, queries):  # type: ignore[no-untyped-def]
        rel = [r for _, r in queries]
        return {
            "ndcg10": np.array([ndcg_at(x, r) for x, r in zip(runs, rel, strict=True)]),
            "recall100": np.array([recall_at(x, r) for x, r in zip(runs, rel, strict=True)]),
            "mrr10": np.array([mrr_at(x, r) for x, r in zip(runs, rel, strict=True)]),
        }

    sweep = {"train": {}, "dev": {}}
    runs_dev = {}
    for a in alphas:
        sweep["train"][a] = float(scores(run(train, a), train)["ndcg10"].mean())
        runs_dev[a] = run(dev, a)
        sweep["dev"][a] = float(scores(runs_dev[a], dev)["ndcg10"].mean())
    best = max(alphas[1:-1], key=lambda a: sweep["train"][a])

    def rrf(dense, sparse, k=60):  # type: ignore[no-untyped-def]
        out = []
        for d, s in zip(dense, sparse, strict=True):
            sc: dict[str, float] = {}
            for lst in (d, s):
                for r, doc in enumerate(lst):
                    sc[doc] = sc.get(doc, 0.0) + 1.0 / (k + r + 1)
            out.append([x for x, _ in sorted(sc.items(), key=lambda kv: -kv[1])][:100])
        return out

    methods = {
        "BM25 only": runs_dev[0.0],
        "Dense only (MiniLM)": runs_dev[1.0],
        "Hybrid, alpha = 0.5": runs_dev[0.5],
        f"Hybrid, alpha = {best} (tuned)": runs_dev[best],
        "RRF (k = 60)": rrf(runs_dev[1.0], runs_dev[0.0]),
        f"Hybrid {best} + Grover consensus": run(dev, best, quantum=True),
    }
    table = {}
    per_query = {}
    for name, rr in methods.items():
        sc = scores(rr, dev)
        per_query[name] = sc["ndcg10"]
        table[name] = {m: float(v.mean()) for m, v in sc.items()}
        table[name]["ndcg10_ci"] = bootstrap_ci(sc["ndcg10"])
    hyb = per_query[f"Hybrid, alpha = {best} (tuned)"]
    tests = {
        "hybrid_vs_dense_p": paired_randomisation(hyb, per_query["Dense only (MiniLM)"]),
        "hybrid_vs_bm25_p": paired_randomisation(hyb, per_query["BM25 only"]),
        "hybrid_vs_rrf_p": paired_randomisation(hyb, per_query["RRF (k = 60)"]),
        "grover_vs_hybrid_p": paired_randomisation(per_query[f"Hybrid {best} + Grover consensus"], hyb),
    }
    db.close()

    lat, breakdown = latency_stages(path, model, best, dev)

    q_ms = []
    for q, _ in dev[:200]:
        t = time.perf_counter()
        model.encode(q)
        q_ms.append((time.perf_counter() - t) * 1000)

    e8 = {
        "corpus": len(corpus), "train_queries": len(train), "dev_queries": len(dev),
        "ingest_s": ingest_s, "ingest_docs_per_s": len(corpus) / ingest_s,
        "alpha_sweep": sweep, "best_alpha": best, "methods": table, "tests": tests,
        "query_embed_p50_ms": p50(q_ms), "search_latency": lat,
    }
    return e8, {"mean_ms_per_query": breakdown}


# ─── E9 ───────────────────────────────────────────────────────────────
def e9_encryption(work: str) -> dict:
    texts = zipf_corpus(20_000, seed=91, length=40)
    queries = [" ".join(t.split()[:4]) + f" q{i}" for i, t in enumerate(zipf_corpus(200, seed=92, length=40))]
    modes = (("plain SQLite", None), ("SQLCipher AES-256", "correct horse battery staple"))
    # Warm-up run so imports and first-use costs are not charged to either mode.
    warm = VecForge(os.path.join(work, "enc_warm.db"), model_name="hash", index_type="flat")
    warm.add_batch(texts[:2000])
    warm.search("warm up", top_k=10)
    warm.close()
    acc: dict[str, dict[str, list]] = {m: {"ingest": [], "open": [], "ms": [], "mb": [], "enc": []} for m, _ in modes}
    for rep in range(3):
        order = modes if rep % 2 == 0 else tuple(reversed(modes))
        for label, key in order:
            path = os.path.join(work, f"enc_{'on' if key else 'off'}_{rep}.db")
            t0 = time.perf_counter()
            db = VecForge(path, model_name="hash", encryption_key=key, index_type="flat")
            for i in range(0, len(texts), 2000):
                db.add_batch(texts[i : i + 2000])
            db.close()
            acc[label]["ingest"].append(len(texts) / (time.perf_counter() - t0))
            for _ in range(3):
                t0 = time.perf_counter()
                VecForge(path, model_name="hash", encryption_key=key, index_type="flat").close()
                acc[label]["open"].append(time.perf_counter() - t0)
            db = VecForge(path, model_name="hash", encryption_key=key, index_type="flat")
            ms = []
            for q in queries:
                t0 = time.perf_counter()
                db.search(q + f" r{rep}", top_k=10)
                ms.append((time.perf_counter() - t0) * 1000)
            acc[label]["enc"].append(bool(db.stats().get("encrypted")))
            db.close()
            acc[label]["ms"] += ms[10:]
            acc[label]["mb"].append(os.path.getsize(path) / 1e6)
    rows = []
    for label, _ in modes:
        a = acc[label]
        rows.append({"mode": label, "encrypted": all(a["enc"]), "ingest_docs_per_s": float(np.median(a["ingest"])),
                     "ingest_runs": a["ingest"], "open_s": float(np.median(a["open"])),
                     "search_p50_ms": p50(a["ms"]), "search_p95_ms": p95(a["ms"]), "file_mb": float(np.median(a["mb"]))})
    return {"docs": len(texts), "repeats": 3, "rows": rows}


# ─── E10 ──────────────────────────────────────────────────────────────
def e10_deletes(work: str) -> dict:
    n = 50_000
    texts = zipf_corpus(n, seed=101, length=40)
    path = os.path.join(work, "del.db")
    db = VecForge(path, model_name="hash", index_type="flat")
    ids: list[str] = []
    for i in range(0, n, 5000):
        ids += db.add_batch(texts[i : i + 5000])
    queries = [" ".join(t.split()[:4]) for t in zipf_corpus(120, seed=102, length=40)]
    order = list(ids)
    random.Random(7).shuffle(order)
    deleted: set[str] = set()
    for i in range(40):  # warm the query path before the first measurement
        db.search(f"warm up {i}", top_k=10)
    rows = []
    done = 0
    for frac in (0.0, 0.05, 0.10, 0.15, 0.19, 0.25):
        target = int(n * frac)
        batch = order[done:target]
        t0 = time.perf_counter()
        if batch:
            db.delete_many(batch)
        del_s = time.perf_counter() - t0
        deleted.update(batch)
        done = target
        ms, leaks = [], 0
        for qi, q in enumerate(queries):
            t0 = time.perf_counter()
            res = db.search(q + f" f{frac} {qi}", top_k=10)
            ms.append((time.perf_counter() - t0) * 1000)
            leaks += sum(1 for r in res if r.doc_id in deleted)
        st = db.stats()
        rows.append({"deleted_fraction": frac, "deleted_in_step": len(batch), "delete_s": del_s,
                     "delete_us_per_doc": (del_s / len(batch) * 1e6) if batch else None,
                     "tombstones": st.get("tombstones", st.get("deleted", None)),
                     "search_p50_ms": p50(ms[10:]), "deleted_docs_returned": leaks})
    db.close()
    return {"docs": n, "rows": rows}


# ─── E11 ──────────────────────────────────────────────────────────────
def e11_filters(work: str) -> dict:
    n = 50_000
    texts = zipf_corpus(n, seed=111, length=40)
    path = os.path.join(work, "filt.db")
    db = VecForge(path, model_name="hash", index_type="flat")
    for i in range(0, n, 5000):
        db.add_batch(texts[i : i + 5000], metadata_list=[{"bucket": (i + j) % 1000} for j in range(5000)])
    queries = [" ".join(t.split()[:4]) for t in zipf_corpus(100, seed=112, length=40)]
    calls: list[int] = []
    orig = vault_mod.VecForge._hydrate

    def counting(self, candidates, namespace, meta_filter):  # type: ignore[no-untyped-def]
        out = orig(self, candidates, namespace, meta_filter)
        calls.append(len(out))
        return out

    vault_mod.VecForge._hydrate = counting
    rows = []
    try:
        for sel in (0.5, 0.1, 0.01, 0.001):
            limit = int(1000 * sel)
            got, rounds, first, ms = [], [], [], []
            for qi, q in enumerate(queries):
                calls.clear()
                t0 = time.perf_counter()
                res = db.search(q + f" s{sel} {qi}", top_k=10, filters={"bucket": {"lt": limit}})
                ms.append((time.perf_counter() - t0) * 1000)
                got.append(len(res))
                rounds.append(len(calls))
                first.append(min(calls[0], 10) if calls else 0)
            rows.append({"selectivity": sel, "results_mean": float(np.mean(got)),
                         "fixed_window_results_mean": float(np.mean(first)),
                         "rounds_mean": float(np.mean(rounds)), "rounds_max": int(np.max(rounds)),
                         "p50_ms": p50(ms[5:]), "p95_ms": p95(ms[5:])})
    finally:
        vault_mod.VecForge._hydrate = orig
    db.close()
    return {"docs": n, "rows": rows}


# ─── E12 ──────────────────────────────────────────────────────────────
def e12_ingest(work: str) -> dict:
    from docx import Document
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    docs = zipf_corpus(60, seed=121, length=900)  # ~5 kB of text each
    rows = []
    for fmt in ("txt", "md", "html", "docx", "pdf"):
        d = Path(work) / f"ingest_{fmt}"
        d.mkdir()
        chars = 0
        for i, body in enumerate(docs):
            words = body.split()
            paras = [" ".join(words[j : j + 90]) + "." for j in range(0, len(words), 90)]
            chars += sum(len(p) for p in paras)
            f = d / f"doc{i}.{fmt}"
            if fmt == "txt":
                f.write_text("\n\n".join(paras))
            elif fmt == "md":
                f.write_text(f"# Document {i}\n\n" + "\n\n".join(f"## Part {k}\n\n{p}" for k, p in enumerate(paras)))
            elif fmt == "html":
                body_html = "".join(f"<p>{p}</p>" for p in paras)
                f.write_text(f"<html><head><script>var x=1;</script></head><body><nav>menu</nav>{body_html}<footer>f</footer></body></html>")
            elif fmt == "docx":
                doc = Document()
                for p in paras:
                    doc.add_paragraph(p)
                doc.save(str(f))
            else:
                c = canvas.Canvas(str(f), pagesize=A4)
                y = 800
                for p in paras:
                    for k in range(0, len(p), 95):
                        c.drawString(40, y, p[k : k + 95])
                        y -= 14
                        if y < 60:
                            c.showPage()
                            y = 800
                c.save()
        times = []
        for rep in range(4):  # the first pass warms parsers and caches and is discarded
            db = VecForge(os.path.join(work, f"ing_{fmt}_{rep}.db"), model_name="hash", index_type="flat")
            t0 = time.perf_counter()
            n_chunks = db.ingest(str(d))
            times.append(time.perf_counter() - t0)
            db.close()
        dt = float(np.median(times[1:]))
        rows.append({"format": fmt, "files": len(docs), "chars": chars, "chunks": n_chunks,
                     "seconds": dt, "files_per_s": len(docs) / dt, "chars_per_s": chars / dt})
    return {"rows": rows}


# ─── E13 ──────────────────────────────────────────────────────────────
_NEGATIONS = {"not", "no", "never", "dislikes", "dislike", "avoids", "without", "closed"}


def salient(text: str) -> set[str]:
    """Numbers, capitalised words after the first, and negation markers."""
    import re

    words = re.findall(r"[A-Za-z][A-Za-z0-9.-]*|\d+(?:[.,]\d+)*", text)
    out = {w for w in words if w[0].isdigit()}
    out |= {w.lower().rstrip(".") for w in words[1:] if w[0].isupper()}
    out |= {w.lower() for w in words if w.lower() in _NEGATIONS}
    return out


def conflict(a: str, b: str) -> bool:
    return salient(a) != salient(b)


def e13_dedup(work: str) -> dict:
    model = minilm()
    facts = [r[0] for r in ROWS]
    para = [p for r in ROWS for p in (r[1], r[2])]
    near = [r[3] for r in ROWS]
    fv, pv, nv = model.encode(facts), model.encode(para), model.encode(near)
    pos_sims, neg_sims = pv @ fv.T, nv @ fv.T
    pos = pos_sims.max(axis=1)  # should merge
    neg = neg_sims.max(axis=1)  # should not
    pos_best = [facts[j] for j in pos_sims.argmax(axis=1)]
    neg_best = [facts[j] for j in neg_sims.argmax(axis=1)]
    # value-conflict guard: refuse to merge when numbers, names or negations differ
    pos_ok = np.array([not conflict(a, b) for a, b in zip(para, pos_best, strict=True)])
    neg_ok = np.array([not conflict(a, b) for a, b in zip(near, neg_best, strict=True)])

    def curve_for(guard: bool) -> list[dict]:
        rows = []
        for t in np.round(np.arange(0.60, 0.981, 0.02), 2):
            pm = (pos >= t) & (pos_ok if guard else True)
            nm = (neg >= t) & (neg_ok if guard else True)
            tp, fp = int(pm.sum()), int(nm.sum())
            prec = tp / (tp + fp) if tp + fp else 1.0
            rec = tp / len(pos)
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            rows.append({"threshold": float(t), "precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp})
        return rows

    curve = curve_for(False)
    curve_guard = curve_for(True)
    # end-to-end through MemoryStore at the default threshold
    mem = MemoryStore(os.path.join(work, "dedup.db"), agent="e13",
                      embedder=CallableEmbedder(model.encode, dimension=384))
    for f in facts:
        mem.remember(f, kind="fact")
    merged_para = sum(1 for p in para if mem.remember(p, kind="fact").deduplicated)
    merged_near = sum(1 for p in near if mem.remember(p, kind="fact").deduplicated)
    stored = mem.stats()["memories"] if "memories" in mem.stats() else None
    mem.close()
    return {"pairs": len(pos), "near_misses": len(neg), "pos_sim": pos.tolist(), "neg_sim": neg.tolist(),
            "curve": curve, "curve_guard": curve_guard, "default_threshold": 0.92,
            "guard_blocks_paraphrases": int((~pos_ok).sum()), "guard_blocks_near_misses": int((~neg_ok).sum()),
            "memorystore_default": {"paraphrases_merged": merged_para, "paraphrases": len(para),
                                    "near_misses_merged": merged_near, "near_misses": len(near),
                                    "stored_memories": stored}}


# ─── E14 ──────────────────────────────────────────────────────────────
async def _mcp_session(vault: str, calls: int) -> dict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=sys.executable,
                                   args=["-m", "vecforge.cli.main", "mcp", "--vault", vault, "--agent", "bench", "--model", "hash"],
                                   cwd=str(ROOT))
    t0 = time.perf_counter()
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        t_init = time.perf_counter() - t0
        tools = await s.list_tools()
        await s.call_tool("recall", {"query": "warm up", "top_k": 5})
        t_first = time.perf_counter() - t0
        lat: dict[str, list[float]] = {"remember": [], "recall": [], "context_pack": []}
        for i in range(calls):
            for name, args in (("remember", {"content": f"benchmark fact number {i} about topic {i % 17}", "kind": "fact"}),
                               ("recall", {"query": f"topic {i % 17}", "top_k": 5}),
                               ("context_pack", {"query": f"topic {i % 17}", "max_chars": 1500})):
                t = time.perf_counter()
                await s.call_tool(name, args)
                lat[name].append((time.perf_counter() - t) * 1000)
    return {"tools": len(tools.tools), "initialize_s": t_init, "first_tool_result_s": t_first,
            "tool_p50_ms": {k: p50(v[3:]) for k, v in lat.items() if len(v) > 3},
            "tool_p95_ms": {k: p95(v[3:]) for k, v in lat.items() if len(v) > 3}}


def e14_mcp(work: str) -> dict:
    out = {}
    facts = zipf_corpus(5000, seed=141, length=15)
    for n in (0, 5000):
        path = os.path.join(work, f"mcp_{n}.db")
        if n:
            mem = MemoryStore(path, agent="bench", model_name="hash")
            for f in facts[:n]:
                mem.remember(f, kind="fact", dedup=False)
            mem.close()
        runs = [asyncio.run(_mcp_session(path, 40 if i == 0 else 1)) for i in range(3)]
        out[str(n)] = {"cold_start_first_result_s": float(np.median([r["first_tool_result_s"] for r in runs])),
                       "initialize_s": float(np.median([r["initialize_s"] for r in runs])),
                       "tools": runs[0]["tools"], "tool_p50_ms": runs[0]["tool_p50_ms"], "tool_p95_ms": runs[0]["tool_p95_ms"]}
    return out


# ─── E15 ──────────────────────────────────────────────────────────────
def e15_rest(work: str) -> dict:
    import httpx

    path = os.path.join(work, "rest.db")
    db = VecForge(path, model_name="hash", index_type="flat")
    texts = zipf_corpus(10_000, seed=151, length=40)
    for i in range(0, len(texts), 5000):
        db.add_batch(texts[i : i + 5000])
    db.close()
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "vecforge.cli.main", "serve", "--vault", path, "--port", str(port), "--model", "hash"],
                            cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        base = f"http://127.0.0.1:{port}/api/v1"
        with httpx.Client(timeout=30, trust_env=False) as c:
            for _ in range(200):
                try:
                    if c.get(f"{base}/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(0.1)
            lat: dict[str, list[float]] = {"search": [], "add": [], "health": []}
            qs = [" ".join(t.split()[:4]) for t in zipf_corpus(220, seed=152, length=40)]
            for i, q in enumerate(qs):
                for name, fn in (("search", lambda: c.post(f"{base}/search", json={"query": q + f" r{i}", "top_k": 10})),
                                 ("add", lambda: c.post(f"{base}/add", json={"text": f"rest document {i} {q}"})),
                                 ("health", lambda: c.get(f"{base}/health"))):
                    t = time.perf_counter()
                    resp = fn()
                    lat[name].append((time.perf_counter() - t) * 1000)
                    assert resp.status_code == 200, (name, resp.status_code, resp.text[:200])
    finally:
        proc.terminate()
        proc.wait(timeout=20)
    return {"p50_ms": {k: p50(v[20:]) for k, v in lat.items()}, "p95_ms": {k: p95(v[20:]) for k, v in lat.items()}}


# ─── E16 ──────────────────────────────────────────────────────────────
def e16_overhead(work: str) -> dict:
    texts = zipf_corpus(10_000, seed=161, length=40)
    base = os.path.join(work, "ov.db")
    db = VecForge(base, model_name="hash", index_type="flat")
    for i in range(0, len(texts), 5000):
        db.add_batch(texts[i : i + 5000])
    db.register_api_key("bench-key-123", "read-write")
    db.close()
    qs = [" ".join(t.split()[:4]) for t in zipf_corpus(300, seed=162, length=40)]
    rows = []
    for label, kw in (("baseline (local admin, no audit)", {}),
                      ("audit log on", {"audit_log": os.path.join(work, "audit.jsonl")}),
                      ("API key (read-write role)", {"api_key": "bench-key-123"}),
                      ("audit log + API key", {"audit_log": os.path.join(work, "audit2.jsonl"), "api_key": "bench-key-123"})):
        db = VecForge(base, model_name="hash", index_type="flat", **kw)
        s_ms, a_ms = [], []
        for i, q in enumerate(qs):
            t = time.perf_counter()
            db.search(q + f" {label} {i}", top_k=10)
            s_ms.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            db.add(f"overhead document {i} {q}", namespace="default")
            a_ms.append((time.perf_counter() - t) * 1000)
        db.close()
        rows.append({"mode": label, "search_p50_ms": p50(s_ms[20:]), "add_p50_ms": p50(a_ms[20:])})
    return {"rows": rows}


# ─── E18 ──────────────────────────────────────────────────────────────
def e18_threads(work: str) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    texts = zipf_corpus(50_000, seed=181, length=40)
    path = os.path.join(work, "thr.db")
    db = VecForge(path, model_name="hash", index_type="flat")
    for i in range(0, len(texts), 5000):
        db.add_batch(texts[i : i + 5000])
    qs = [" ".join(t.split()[:4]) for t in zipf_corpus(400, seed=182, length=40)]
    for i in range(40):
        db.search(f"warm up {i}", top_k=10)
    # Correctness under concurrency: 8 threads must return what one thread returns.
    check = [q + " check" for q in qs[:200]]
    expected = [[r.doc_id for r in db.search(q, top_k=10)] for q in check]
    with ThreadPoolExecutor(8) as ex:
        got = list(ex.map(lambda q: [r.doc_id for r in db.search(q, top_k=10)], check))
    mismatches = sum(1 for e, g in zip(expected, got, strict=True) if e != g)
    qps: dict[int, list[float]] = {1: [], 2: [], 4: [], 8: []}
    orders = [(1, 2, 4, 8), (8, 4, 2, 1), (2, 8, 1, 4)]
    for rep, order in enumerate(orders):
        for threads in order:
            work_items = [q + f" t{threads} r{rep} {i}" for i, q in enumerate(qs)]
            t0 = time.perf_counter()
            with ThreadPoolExecutor(threads) as ex:
                list(ex.map(lambda q: db.search(q, top_k=10), work_items))
            qps[threads].append(len(work_items) / (time.perf_counter() - t0))
    db.close()
    rows = [{"threads": t, "qps": float(np.median(v)), "qps_runs": v} for t, v in qps.items()]
    return {"docs": len(texts), "repeats": len(orders), "rows": rows,
            "concurrency_check": {"queries": len(check), "threads": 8, "mismatches": mismatches}}


def main() -> None:
    only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
    results: dict = json.loads(OUT.read_text()) if OUT.exists() else {}
    work = tempfile.mkdtemp(prefix="vfbench2_")
    steps = [("E8", None), ("E9", e9_encryption), ("E10", e10_deletes), ("E11", e11_filters), ("E12", e12_ingest),
             ("E13", e13_dedup), ("E14", e14_mcp), ("E15", e15_rest), ("E16", e16_overhead), ("E17", None), ("E18", e18_threads)]
    try:
        for name, fn in steps:
            if only and name not in only:
                continue
            t0 = time.perf_counter()
            if name == "E8":
                results["E8"], results["E17"] = e8_e17(work)
            elif name == "E17":
                lat, parts = e17_only(work)
                results.setdefault("E8", {})["search_latency"] = lat
                results["E17"] = {"mean_ms_per_query": parts}
            else:
                results[name] = fn(work)
            print(f"{name} done in {time.perf_counter() - t0:.1f}s", flush=True)
            OUT.write_text(json.dumps(results, indent=1, default=float))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
