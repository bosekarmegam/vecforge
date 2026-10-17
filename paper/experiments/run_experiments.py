# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under the Business Source License 1.1 (BSL 1.1)

"""
Experiments for the VecForge 2.0 paper.

Every number in the paper's evaluation section comes from this script.
1.x baselines are loaded verbatim from git commit a0d25ab (VecForge 1.1.0).

    python paper/experiments/run_experiments.py           # writes results.json
    python paper/experiments/make_figures.py              # writes figures/

E1  Rank preservation of the Grover reranker, 1.x vs 2.0
E2  Reranker latency (closed form vs iterative; end to end up to 1M candidates)
E3  Keyword index: incremental BM25 vs full rebuild per insert (1.x)
E4  Namespace-scoped recall: in-index ID selector vs 4*top_k post-filter (1.x)
E5  Reopen time: index snapshot vs rebuild from stored vectors
E6  End-to-end hybrid search latency (flat and HNSW)
E7  Agent memory: write throughput and recall latency
"""

from __future__ import annotations

import gc
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import faiss
import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vecforge import MemoryStore, VecForge  # noqa: E402
from vecforge.core.bm25 import BM25Engine  # noqa: E402
from vecforge.quantum.grover_amplifier import GroverAmplifier  # noqa: E402
from vecforge.quantum.reranker import QuantumReranker  # noqa: E402

OUT = Path(__file__).resolve().parent / "results.json"
BASELINE_COMMIT = "a0d25ab"
RNG = np.random.default_rng(20261017)


def p50(xs: list[float]) -> float:
    return float(np.percentile(xs, 50))


def p95(xs: list[float]) -> float:
    return float(np.percentile(xs, 95))


def timed(fn, runs: int, warmup: int = 1) -> list[float]:  # type: ignore[no-untyped-def]
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def load_v1() -> tuple[object, object]:
    """Load the 1.x quantum reranker and BM25 index from git, unmodified."""
    tmp = Path(tempfile.mkdtemp(prefix="vf1_"))
    pkg = tmp / "vf1"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for src, name in [
        ("vecforge/quantum/amplitude_encoder.py", "amplitude_encoder.py"),
        ("vecforge/quantum/grover_amplifier.py", "grover_amplifier.py"),
        ("vecforge/quantum/reranker.py", "reranker.py"),
        ("vecforge/core/bm25.py", "bm25.py"),
    ]:
        code = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{BASELINE_COMMIT}:{src}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        code = code.replace("from vecforge.quantum.", "from vf1.").replace(
            "from vecforge.core.", "from vf1."
        )
        (pkg / name).write_text(code)
    sys.path.insert(0, str(tmp))
    rer = importlib.import_module("vf1.reranker")
    bm = importlib.import_module("vf1.bm25")
    return rer.QuantumReranker, bm.BM25Engine


# ─── E1 ───────────────────────────────────────────────────────────────
def e1_rank_preservation(QR1) -> dict:  # type: ignore[no-untyped-def]
    sizes = list(range(2, 257))
    out: dict[str, dict[str, list[float]]] = {"1.x": {}, "2.0": {}}
    for label in out:
        out[label] = {
            "top1_linear": [],
            "tau_linear": [],
            "top1_random": [],
            "tau_random": [],
        }
    trials = 20
    for n in sizes:
        lin = list(np.linspace(1.0, 0.05, n))
        rand_sets = [
            sorted(RNG.uniform(0.05, 1.0, n).tolist(), reverse=True)
            for _ in range(trials)
        ]
        for label, qr in (("1.x", QR1()), ("2.0", QuantumReranker())):
            res = qr.rerank("q", [str(i) for i in range(n)], lin, top_k=n)
            order = [i for i, _ in res]
            out[label]["top1_linear"].append(float(order[0] == 0))
            out[label]["tau_linear"].append(
                float(kendalltau(range(n), order)[0])
                if n > 2
                else (1.0 if order == [0, 1] else -1.0)
            )
            t1, tau = [], []
            for s in rand_sets:
                r = qr.rerank("q", [str(i) for i in range(n)], s, top_k=n)
                o = [i for i, _ in r]
                t1.append(float(o[0] == 0))
                tau.append(
                    float(kendalltau(range(n), o)[0])
                    if n > 2
                    else (1.0 if o == [0, 1] else -1.0)
                )
            out[label]["top1_random"].append(float(np.mean(t1)))
            out[label]["tau_random"].append(float(np.mean(tau)))
    summary = {}
    for label in out:
        tl = np.array(out[label]["top1_linear"])
        tau = np.array(out[label]["tau_linear"])
        summary[label] = {
            "sizes": len(sizes),
            "top1_kept_share_linear": float(tl.mean()),
            "reversed_share_linear": float((tau < -0.99).mean()),
            "unchanged_share_linear": float((tau > 0.99).mean()),
            "top1_kept_share_random": float(np.mean(out[label]["top1_random"])),
            "mean_tau_random": float(np.mean(out[label]["tau_random"])),
        }
    # marked-set lift: consensus oracle (2.0 only)
    lifts = []
    for n in (20, 50, 100, 500, 1000):
        scores = np.linspace(1.0, 0.05, n)
        marked = sorted(
            RNG.choice(np.arange(1, n), size=max(2, n // 10), replace=False).tolist()
        )
        res = QuantumReranker().rerank(
            "q", [str(i) for i in range(n)], scores.tolist(), top_k=n, marked=marked
        )
        order = [i for i, _ in res]
        lifts.append(
            {
                "n": n,
                "marked": len(marked),
                "marked_in_top": sum(
                    1 for i in order[: len(marked)] if i in set(marked)
                ),
                "marked_order_kept": [i for i in order if i in set(marked)] == marked,
            }
        )
    return {"sizes": sizes, "series": out, "summary": summary, "consensus_lift": lifts}


# ─── E2 ───────────────────────────────────────────────────────────────
def e2_latency() -> dict:
    res: dict[str, list[dict[str, float]]] = {"amplifier": [], "reranker": []}
    for n in (1_000, 10_000, 100_000):
        a = RNG.random(n).astype(np.float32)
        a /= np.linalg.norm(a)
        mask = GroverAmplifier.default_marked(a)
        closed = timed(
            lambda: GroverAmplifier(method="closed_form").amplify(a, marked=mask), 30
        )
        it = timed(
            lambda: GroverAmplifier(method="iterative").amplify(a, marked=mask),
            5 if n == 100_000 else 15,
        )
        res["amplifier"].append(
            {
                "n": n,
                "closed_ms": p50(closed),
                "iterative_ms": p50(it),
                "k": GroverAmplifier._optimal_iterations(n, int(mask.sum())),
            }
        )
    # a single marked candidate needs k ~ (pi/4) sqrt(N) iterations
    res["amplifier_single_marked"] = []
    for n in (1_000, 10_000, 100_000):
        a = RNG.random(n).astype(np.float32)
        a /= np.linalg.norm(a)
        mask = np.zeros(n, dtype=bool)
        mask[int(np.argmax(a))] = True
        closed = timed(
            lambda: GroverAmplifier(method="closed_form").amplify(a, marked=mask), 30
        )
        it = timed(
            lambda: GroverAmplifier(method="iterative").amplify(a, marked=mask), 5
        )
        res["amplifier_single_marked"].append(
            {
                "n": n,
                "closed_ms": p50(closed),
                "iterative_ms": p50(it),
                "k": GroverAmplifier._optimal_iterations(n, 1),
            }
        )
    for n in (1_000, 10_000, 100_000, 1_000_000):
        s = RNG.random(n).astype(np.float32)
        qr = QuantumReranker(max_candidates=1000)
        ms = timed(lambda: qr.amplify_scores(s, top_k=10), 30)
        res["reranker"].append({"n": n, "p50_ms": p50(ms), "p95_ms": p95(ms)})
    return res


# ─── E3 ───────────────────────────────────────────────────────────────
def zipf_corpus(
    n_docs: int, vocab: int = 20_000, length: int = 60, seed: int = 7
) -> list[str]:
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, vocab + 1)
    p = 1.0 / ranks**1.07
    p /= p.sum()
    words = np.array([f"w{i}" for i in range(vocab)])
    return [" ".join(words[rng.choice(vocab, size=length, p=p)]) for _ in range(n_docs)]


def e3_bm25(BM1) -> dict:  # type: ignore[no-untyped-def]
    out = {"insert": [], "query": []}
    corpus = zipf_corpus(4000)
    for n in (250, 500, 1000, 2000, 4000):
        docs = corpus[:n]
        runs = []
        for _ in range(3):  # median of three: the incremental build is short and noisy
            t0 = time.perf_counter()
            e = BM25Engine()
            for d in docs:
                e.add_document(d)
            runs.append(time.perf_counter() - t0)
        new_s = float(np.median(runs))
        old_s = None
        if n <= 2000:  # the 1.x rebuild is quadratic; larger sizes take minutes
            t0 = time.perf_counter()
            o = BM1()
            for d in docs:
                o.add_document(d)
            old_s = time.perf_counter() - t0
        out["insert"].append({"n": n, "v2_s": new_s, "v1_s": old_s})
    big = zipf_corpus(100_000, seed=11)
    queries = [" ".join(q.split()[:3]) for q in zipf_corpus(50, seed=3)]
    for n in (10_000, 100_000):
        e = BM25Engine()
        e.add_documents(big[:n])
        ms = []
        for q in queries:
            t0 = time.perf_counter()
            e.search(q, top_k=10)
            ms.append((time.perf_counter() - t0) * 1000)
        o = BM1()
        o.add_documents(big[:n])
        ms1 = []
        for q in queries[:20]:
            t0 = time.perf_counter()
            o.search(q, top_k=10)
            ms1.append((time.perf_counter() - t0) * 1000)
        # equivalence check: same top-10 on shared queries (both are Okapi BM25)
        overlap = np.mean(
            [
                len(
                    {r.doc_index for r in e.search(q, 10)}
                    & {r.doc_index for r in o.search(q, 10)}
                )
                / 10
                for q in queries[:20]
            ]
        )
        out["query"].append(
            {
                "n": n,
                "v2_p50_ms": p50(ms),
                "v1_p50_ms": p50(ms1),
                "top10_overlap": float(overlap),
            }
        )
        del e, o
        gc.collect()
    return out


# ─── E4 ───────────────────────────────────────────────────────────────
def clustered(
    n: int, d: int, clusters: int, spread: float, rng: np.random.Generator
) -> np.ndarray:
    """Unit vectors drawn from a Gaussian mixture, closer to real text embeddings than isotropic noise."""
    centers = rng.standard_normal((clusters, d)).astype(np.float32)
    v = centers[rng.integers(0, clusters, n)] + spread * rng.standard_normal(
        (n, d)
    ).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def e4_namespace_recall() -> dict:
    n, d, top_k = 100_000, 384, 10
    rng = np.random.default_rng(404)
    x = clustered(n, d, 2_000, 0.6, rng)
    index = faiss.IndexFlatIP(d)
    index.add(x)
    hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 80
    hnsw.add(x)
    q = clustered(200, d, 2_000, 0.6, np.random.default_rng(405))
    # reference: unfiltered HNSW recall@10 against exact search
    _, exact_all = index.search(q, top_k)
    unfiltered = {}
    for ef in (64, 256):
        hnsw.hnsw.efSearch = ef
        _, h_all = hnsw.search(q, top_k)
        unfiltered[ef] = float(
            np.mean(
                [
                    len(set(a.tolist()) & set(b.tolist())) / top_k
                    for a, b in zip(h_all, exact_all)
                ]
            )
        )
    rows = []
    for frac in (0.001, 0.01, 0.05, 0.1, 0.5):
        size = int(n * frac)
        members = np.sort(RNG.choice(n, size=size, replace=False)).astype(np.int64)
        member_set = set(members.tolist())
        sub = x[members]
        truth = [set(members[np.argsort(-(sub @ qi))[:top_k]].tolist()) for qi in q]
        # 1.x: global top 4*top_k, then drop other namespaces
        _, gi = index.search(q, top_k * 4)
        post = [
            len({int(j) for j in row if int(j) in member_set} & t) / top_k
            for row, t in zip(gi, truth)
        ]
        # 2.0 flat: IDSelectorBatch inside FAISS
        sel = faiss.IDSelectorBatch(members)
        params = faiss.SearchParameters(sel=sel)
        t0 = time.perf_counter()
        _, si = index.search(q, top_k, params=params)
        flat_ms = (time.perf_counter() - t0) * 1000 / len(q)
        pre = [len(set(row.tolist()) & t) / top_k for row, t in zip(si, truth)]
        # 2.0 HNSW: exact over reconstructed subset when small, else filtered graph search
        if size <= 20_000:
            t0 = time.perf_counter()
            hs = [set(members[np.argsort(-(sub @ qi))[:top_k]].tolist()) for qi in q]
            hnsw_ms = (time.perf_counter() - t0) * 1000 / len(q)
            hnsw_rec = [len(a & t) / top_k for a, t in zip(hs, truth)]
            hnsw_mode = "exact subset"
        else:
            hnsw_rec, hnsw_ms = [], 0.0
            per_ef = {}
            for ef in (64, 256):
                hp = faiss.SearchParametersHNSW()
                hp.sel = sel
                hp.efSearch = ef
                t0 = time.perf_counter()
                _, hi = hnsw.search(q, top_k, params=hp)
                ms = (time.perf_counter() - t0) * 1000 / len(q)
                rec = [len(set(row.tolist()) & t) / top_k for row, t in zip(hi, truth)]
                per_ef[ef] = {"recall": float(np.mean(rec)), "ms": ms}
                hnsw_rec, hnsw_ms = rec, ms
            hnsw_mode = "filtered graph (efSearch 256)"
        rows.append(
            {
                "fraction": frac,
                "size": size,
                "recall_postfilter_v1": float(np.mean(post)),
                "recall_flat_selector_v2": float(np.mean(pre)),
                "recall_hnsw_v2": float(np.mean(hnsw_rec)),
                "hnsw_mode": hnsw_mode,
                "flat_selector_ms_per_query": flat_ms,
                "hnsw_ms_per_query": hnsw_ms,
                "hnsw_filtered_by_ef": per_ef if size > 20_000 else None,
            }
        )
    return {
        "n": n,
        "dim": d,
        "top_k": top_k,
        "queries": int(len(q)),
        "data": "Gaussian mixture, 2000 clusters, spread 0.6",
        "unfiltered_hnsw_recall": unfiltered,
        "rows": rows,
    }


# ─── E5 / E6 ──────────────────────────────────────────────────────────
def build_vault(path: str, n: int, index_type: str = "flat") -> None:
    texts = zipf_corpus(n, seed=21, length=40)
    db = VecForge(path, model_name="hash", index_type=index_type)
    step = 5000
    for i in range(0, n, step):
        db.add_batch(texts[i : i + step], namespace="docs" if i % 2 == 0 else "notes")
    db.close()


def e5_reopen(workdir: str) -> dict:
    rows = []
    for n in (10_000, 50_000, 100_000):
        path = os.path.join(workdir, f"v{n}.db")
        t0 = time.perf_counter()
        build_vault(path, n)
        build_s = time.perf_counter() - t0
        snap = []
        for _ in range(3):
            t0 = time.perf_counter()
            VecForge(path, model_name="hash").close()
            snap.append(time.perf_counter() - t0)
        rebuild = []
        for _ in range(3):
            db = VecForge(path, model_name="hash")
            db._storage.clear_index_state()
            # why: close the file without writing a fresh snapshot, forcing a rebuild
            db._closed = True
            db._storage.close()
            t0 = time.perf_counter()
            db2 = VecForge(path, model_name="hash")
            rebuild.append(time.perf_counter() - t0)
            db2.close()  # writes the snapshot back
        rows.append(
            {
                "n": n,
                "ingest_s": build_s,
                "ingest_docs_per_s": n / build_s,
                "open_snapshot_s": float(np.median(snap)),
                "open_rebuild_s": float(np.median(rebuild)),
                "file_mb": os.path.getsize(path) / 1e6,
            }
        )
    return {"rows": rows}


def e6_search(workdir: str) -> dict:
    rows = []
    queries = [" ".join(q.split()[:4]) for q in zipf_corpus(210, seed=5, length=40)]
    modes = (
        ("hybrid", {}),
        ("hybrid+namespace", {"namespace": "docs"}),
        ("hybrid+quantum", {"quantum_rerank": True}),
    )
    for n, index_type in ((10_000, "flat"), (100_000, "flat"), (100_000, "hnsw")):
        path = os.path.join(workdir, f"s{n}_{index_type}.db")
        if not os.path.exists(path):
            build_vault(path, n, index_type=index_type)
        db = VecForge(path, model_name="hash", index_type=index_type)
        times: dict[str, list[float]] = {label: [] for label, _ in modes}
        # why: rotate the mode order every query so each mode runs first, second and third
        # equally often; the first search of a query warms CPU caches for the others
        for qi, q in enumerate(queries):
            for label, kw in modes[qi % 3 :] + modes[: qi % 3]:
                text = (
                    q + f" {label} {n} {index_type} v{qi}"
                )  # unique text defeats the embedding cache
                t0 = time.perf_counter()
                db.search(text, top_k=10, **kw)
                times[label].append((time.perf_counter() - t0) * 1000)
        for label, _ in modes:
            ms = times[label][10:]
            rows.append(
                {
                    "n": n,
                    "index": index_type,
                    "mode": label,
                    "p50_ms": p50(ms),
                    "p95_ms": p95(ms),
                    "queries": len(ms),
                }
            )
        db.close()
    return {"rows": rows}


# ─── E7 ───────────────────────────────────────────────────────────────
def e7_memory(workdir: str) -> dict:
    rows = []
    facts = zipf_corpus(5000, seed=33, length=18)
    path = os.path.join(workdir, "mem.db")
    mem = MemoryStore(path, agent="bench", model_name="hash")
    written = 0
    for target in (1000, 5000):
        t0 = time.perf_counter()
        while written < target:
            mem.remember(facts[written], kind="fact", importance=0.5, dedup=False)
            written += 1
        write_s = time.perf_counter() - t0
        ms = timed(lambda: mem.recall("w1 w2 w3 w40", top_k=8), 40)
        cp = timed(lambda: mem.context_pack("w1 w2 w3", max_chars=2000), 20)
        rows.append(
            {
                "memories": target,
                "remember_per_s": (target - (1000 if target == 5000 else 0)) / write_s,
                "recall_p50_ms": p50(ms),
                "context_pack_p50_ms": p50(cp),
            }
        )
    mem.close()
    return {"rows": rows}


def environment() -> dict:
    cpu = "unknown"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "cpu": cpu,
        "cores": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "faiss": faiss.__version__,
        "os": platform.platform(),
        "baseline_commit": BASELINE_COMMIT,
    }


def main() -> None:
    faiss.omp_set_num_threads(os.cpu_count() or 1)
    QR1, BM1 = load_v1()
    only = (
        set(sys.argv[sys.argv.index("--only") + 1].split(","))
        if "--only" in sys.argv
        else None
    )
    results: dict = json.loads(OUT.read_text()) if only and OUT.exists() else {}
    results["environment"] = environment()
    work = tempfile.mkdtemp(prefix="vfbench_")
    try:
        for name, fn in (
            ("E1", lambda: e1_rank_preservation(QR1)),
            ("E2", e2_latency),
            ("E3", lambda: e3_bm25(BM1)),
            ("E4", e4_namespace_recall),
            ("E5", lambda: e5_reopen(work)),
            ("E6", lambda: e6_search(work)),
            ("E7", lambda: e7_memory(work)),
        ):
            if only and name not in only:
                continue
            t0 = time.perf_counter()
            results[name] = fn()
            print(f"{name} done in {time.perf_counter() - t0:.1f}s", flush=True)
            OUT.write_text(json.dumps(results, indent=1))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
