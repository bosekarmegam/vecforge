# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under the Business Source License 1.1 (BSL 1.1)

"""Figures for the extended evaluation (results2.json) and the explanatory
plots that do not depend on measurements (Grover geometry, role matrix)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Arc, FancyArrowPatch  # noqa: E402
from palette import INK, MUTED, RC, ROLE, S  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
R1 = json.loads((HERE / "results.json").read_text())
R2 = json.loads((HERE / "results2.json").read_text()) if (HERE / "results2.json").exists() else {}

NEUTRAL = ROLE["neutral"]["base"]  # axes, reference vectors and the unit circle in the geometry figure
plt.rcParams.update(RC)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


# ─── explanatory figures ──────────────────────────────────────────────
def fig_grover_geometry() -> None:
    theta, phi0 = math.radians(11), math.radians(20)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.set_aspect("equal")
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_patch(Arc((0, 0), 2, 2, theta1=-30, theta2=95, color=NEUTRAL, lw=0.8, alpha=0.5))
    ax.annotate("", xy=(1.18, 0), xytext=(-0.05, 0), arrowprops=dict(arrowstyle="-|>", color=NEUTRAL, lw=1))
    ax.annotate("", xy=(0, 1.18), xytext=(0, -0.05), arrowprops=dict(arrowstyle="-|>", color=NEUTRAL, lw=1))
    ax.text(1.2, -0.07, r"$u_U$ (unmarked)", fontsize=9, color=INK, ha="left", va="top")
    ax.text(0.03, 1.21, r"$u_M$ (marked)", fontsize=9, color=INK, ha="left")

    def vec(angle: float, color: str, label: str, style: str = "-", lw: float = 1.8, off=(0.04, 0.0), text: str | None = None) -> None:
        x, y = math.cos(angle), math.sin(angle)
        ax.add_patch(FancyArrowPatch((0, 0), (x, y), arrowstyle="-|>", mutation_scale=11, color=color, lw=lw, linestyle=style))
        ax.text(x + off[0], y + off[1], label, color=text or color, fontsize=9)

    # label text uses the shade of a light hue so it stays readable on white
    vec(theta, NEUTRAL, r"$s$ (uniform)", "--", 1.1, (0.05, -0.06), text=ROLE["neutral"]["shade"])
    vec(phi0, S["dense"], r"$a$ (encoded scores)", "-", 1.8, (0.05, 0.0))
    vec(-phi0, S["threshold"], r"$O_M a$ (oracle flips sign)", ":", 1.4, (0.05, -0.06))
    iterates = {1: (ROLE["accent"]["base"], ROLE["accent"]["shade"]), 2: (S["v2"], S["v2"]), 3: (S["v2_grover"], S["v2_grover"])}
    for k in (1, 2, 3):
        vec(phi0 + 2 * k * theta, iterates[k][0], "" if k == 3 else rf"$G^{k} a$", "-", 1.8 if k == 3 else 1.2, (0.03, 0.03), text=iterates[k][1])
    ang3 = phi0 + 6 * theta
    ax.text(math.cos(ang3) - 0.11, math.sin(ang3) + 0.02, r"$G^{3} a$, $k^\star = 3$", color=S["v2_grover"], fontsize=9, ha="right")
    ax.add_patch(Arc((0, 0), 0.62, 0.62, theta1=math.degrees(phi0), theta2=math.degrees(phi0 + 2 * theta), color=S["v2"], lw=1.2))
    mid = phi0 + theta
    ax.text(0.36 * math.cos(mid) + 0.02, 0.36 * math.sin(mid), r"$2\theta$", color=S["v2"], fontsize=9)
    ax.add_patch(Arc((0, 0), 0.8, 0.8, theta1=0, theta2=math.degrees(theta), color=NEUTRAL, lw=1))
    ax.text(0.43, 0.037, r"$\theta$", color=NEUTRAL, fontsize=8, va="center")  # sits between the axis and s, clear of both
    ax.set_xlim(-0.55, 1.75)
    ax.set_ylim(-0.48, 1.3)
    save(fig, "fig-grover-geometry")


def fig_rbac() -> None:
    roles = ["admin", "read-write", "read-only", "unregistered key"]
    ops = ["Search and read", "Add, update,\nsave, compact", "Delete", "Create\nnamespace", "Backup", "Manage keys,\nraw storage"]
    grant = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    fig, ax = plt.subplots(figsize=(6.2, 2.2))
    ax.grid(False)
    cmap = matplotlib.colors.ListedColormap([S["denied_fill"], S["allowed_fill"]])
    ax.imshow(grant, cmap=cmap, aspect="auto")
    for i in range(len(roles)):
        for j in range(len(ops)):
            ax.text(j, i, "allowed" if grant[i, j] else "denied", ha="center", va="center",
                    fontsize=8.5, color=S["allowed_text"] if grant[i, j] else S["denied_text"], fontweight="bold" if grant[i, j] else "normal")
    ax.set_xticks(range(len(ops)), ops, fontsize=8.5, color=INK)
    ax.set_yticks(range(len(roles)), roles, fontsize=9, color=INK)
    ax.xaxis.tick_top()
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(ops)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(roles)), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    save(fig, "fig-rbac")


# ─── measured results ─────────────────────────────────────────────────
def fig_alpha() -> None:
    e = R2["E8"]
    a = sorted(float(x) for x in e["alpha_sweep"]["dev"])
    dev = [e["alpha_sweep"]["dev"][str(x) if str(x) in e["alpha_sweep"]["dev"] else x] for x in a]
    train = [e["alpha_sweep"]["train"][str(x) if str(x) in e["alpha_sweep"]["train"] else x] for x in a]
    rrf = e["methods"]["RRF (k = 60)"]["ndcg10"]
    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    ax.plot(a, train, "o-", color=S["train"], ms=3.5, label="train claims (tuning)")
    ax.plot(a, dev, "s-", color=S["dev"], ms=3.5, label="dev claims (reported)")
    ax.axhline(rrf, color=S["rrf"], ls="--", lw=1.2, label="RRF, dev")
    ax.axvline(e["best_alpha"], color=MUTED, ls=":", lw=1)
    ax.set_xlabel(r"fusion weight $\alpha$ (0 = BM25, 1 = dense)")
    ax.set_ylabel("nDCG@10")
    ax.legend(loc="lower center", facecolor="white", edgecolor="none", framealpha=1)  # hides the dotted line behind the text
    save(fig, "fig-alpha-sweep")


def fig_methods() -> None:
    m = dict(R2["E8"]["methods"])
    best = R2["E8"]["best_alpha"]
    tuned = f"Hybrid, alpha = {best} (tuned)"
    if tuned in m and m[tuned]["ndcg10"] == m.get("Hybrid, alpha = 0.5", {}).get("ndcg10"):
        m.pop(tuned)  # the tuned weight equals the default, so one bar is enough
    pretty = {"BM25 only": "BM25 only", "Dense only (MiniLM)": "Dense only (MiniLM)",
              "Hybrid, alpha = 0.5": r"Hybrid, $\alpha$ = 0.5 (default)", tuned: rf"Hybrid, $\alpha$ = {best} (tuned)",
              "RRF (k = 60)": "RRF, k = 60", f"Hybrid {best} + Grover consensus": "Hybrid + Grover consensus"}
    names = list(m)
    vals = [m[n]["ndcg10"] for n in names]
    lo = [m[n]["ndcg10"] - m[n]["ndcg10_ci"][0] for n in names]
    hi = [m[n]["ndcg10_ci"][1] - m[n]["ndcg10"] for n in names]
    rec = [m[n]["recall100"] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3), sharey=True, gridspec_kw={"width_ratios": [1.35, 1]})
    y = np.arange(len(names))[::-1]
    cmap = {"BM25 only": S["bm25"], "Dense only (MiniLM)": S["dense"], "RRF (k = 60)": S["rrf"]}
    colors = [cmap.get(n, S["v2_grover"] if "Grover" in n else S["hybrid"]) for n in names]
    axes[0].barh(y, vals, xerr=[lo, hi], color=colors, height=0.6, error_kw=dict(lw=0.9, capsize=2.5, ecolor=INK))
    for yi, v, h in zip(y, vals, hi, strict=True):
        axes[0].text(v + h + 0.008, yi, f"{v:.3f}", va="center", fontsize=8)
    axes[0].set_yticks(y, [pretty.get(n, n) for n in names])
    axes[0].set_xlabel("nDCG@10 with 95% bootstrap CI")
    axes[0].set_xlim(0.4, 0.82)
    axes[1].barh(y, rec, color=colors, height=0.6)
    for yi, v in zip(y, rec, strict=True):
        axes[1].text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=8)
    axes[1].set_xlabel("Recall@100")
    axes[1].set_xlim(0.8, 1.0)
    for ax in axes:
        ax.grid(axis="y", visible=False)
    save(fig, "fig-scifact-methods")


def fig_stages() -> None:
    b = R2["E17"]["mean_ms_per_query"]
    order = ["query embedding (MiniLM)", "dense search (FAISS)", "sparse search (BM25)", "fusion", "hydration (SQLite)", "Grover rerank", "other"]
    stage_color = dict(zip(order, (S["embed"], S["dense"], S["bm25"], S["fusion"], S["hydration"], S["grover"], S["other"]), strict=True))
    labels = list(b)
    fig, ax = plt.subplots(figsize=(6.6, 1.9))
    y = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    for st in order:
        w = np.array([b[lab].get(st, 0.0) for lab in labels])
        ax.barh(y, w, left=left, color=stage_color[st], height=0.55, label=st)
        left += w
    for yi, tot in zip(y, left, strict=True):
        ax.text(tot + 0.1, yi, f"{tot:.1f} ms", va="center", fontsize=8.5)
    ax.set_yticks(y, labels)
    ax.set_xlabel("mean time per query (ms)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.38), fontsize=8)
    save(fig, "fig-stage-breakdown")


def fig_latency_cdf() -> None:
    lat = R2["E8"]["search_latency"]
    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    for (name, d), c in zip(lat.items(), (S["hybrid"], S["v2_grover"]), strict=False):
        x = np.sort(np.array(d["all_ms"]))
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, lw=1.6, ls="--" if "Grover" in name else "-",
                label=f"{name} (p50 {d['p50_ms']:.1f} ms)")
    ax.set_xlabel("end-to-end search latency (ms)")
    ax.set_ylabel("fraction of queries")
    ax.set_xlim(left=0)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig-latency-cdf")


def fig_encryption() -> None:
    rows = R2["E9"]["rows"]
    metrics = [("ingest_docs_per_s", "Ingest (docs/s)"), ("open_s", "Open from snapshot (s)"),
               ("search_p50_ms", "Search p50 (ms)"), ("file_mb", "File size (MB)")]
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.0))
    for ax, (k, title) in zip(axes, metrics, strict=True):
        v = [r[k] for r in rows]
        bars = ax.bar([0, 1], v, color=[S["plain"], S["encrypted"]], width=0.6)
        ax.set_title(title, fontsize=8.8)
        ax.set_xticks([0, 1], ["plain", "AES-256"], fontsize=8.3)
        ax.grid(axis="x", visible=False)
        top = max(v)
        for b_, val in zip(bars, v, strict=True):
            ax.text(b_.get_x() + b_.get_width() / 2, val + top * 0.02, f"{val:,.0f}" if val >= 100 else f"{val:.2f}", ha="center", fontsize=7.8)
        ax.set_ylim(0, top * 1.2)
        ax.tick_params(axis="y", labelsize=7.5)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig-encryption")


def fig_deletes() -> None:
    rows = R2["E10"]["rows"]
    x = [r["deleted_fraction"] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    ax.plot(x, [r["search_p50_ms"] for r in rows], "o-", color=S["v2"], ms=4, lw=1.6, label="search p50")
    ax.axvline(20, color=S["threshold"], ls="--", lw=1)
    ax.set_ylim(0, max(r["search_p50_ms"] for r in rows) * 1.3)
    ax.text(19.4, ax.get_ylim()[1] * 0.97, "compaction\nat 20%", color=ROLE["baseline"]["shade"], fontsize=8, va="top", ha="right")
    ax.set_xlabel("documents deleted (% of 50,000)")
    ax.set_ylabel("search latency p50 (ms)")
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    save(fig, "fig-deletes")


def fig_filters() -> None:
    rows = R2["E11"]["rows"]
    lab = [f"{r['selectivity'] * 100:g}%" for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    ax.bar(x - 0.18, [r["fixed_window_results_mean"] for r in rows], 0.36, color=S["v1"], label="fixed 4k window")
    ax.bar(x + 0.18, [r["results_mean"] for r in rows], 0.36, color=S["v2"], label="adaptive widening")
    ax.set_xticks(x, lab)
    ax.set_xlabel("share of documents passing the filter")
    ax.set_ylabel("results returned (of k = 10)")
    ax.set_ylim(0, 11.5)
    ax.grid(axis="x", visible=False)
    ax2 = ax.twinx()
    ax2.plot(x, [r["rounds_mean"] for r in rows], "D--", color=S["bm25"], ms=3.5, label="search rounds")
    ax2.set_ylabel("search rounds (mean)", color=ROLE["sparse"]["shade"])
    ax2.tick_params(axis="y", color=S["bm25"], labelcolor=ROLE["sparse"]["shade"])
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(S["bm25"])
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.5,
              handlelength=1.4, columnspacing=0.9)
    save(fig, "fig-filters")


def fig_ingest() -> None:
    rows = R2["E12"]["rows"]
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    names = [r["format"].upper() for r in rows]
    v = [r["chars_per_s"] / 1000 for r in rows]
    colors = [S["format_slow"] if r["format"] == "docx" else S["format_fast"] for r in rows]  # DOCX is the slow one
    bars = ax.bar(names, v, color=colors, width=0.62)
    for b_, val in zip(bars, v, strict=True):
        ax.text(b_.get_x() + b_.get_width() / 2, val + max(v) * 0.02, f"{val:,.0f}", ha="center", fontsize=8, color=INK)
    ax.set_ylabel("thousand characters per second")
    ax.grid(axis="x", visible=False)
    save(fig, "fig-ingest")


def fig_dedup() -> None:
    e = R2["E13"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    bins = np.linspace(0.4, 1.0, 25)
    axes[0].hist(e["pos_sim"], bins=bins, color=S["paraphrase"], alpha=0.75, label="paraphrase (should merge)")
    axes[0].hist(e["neg_sim"], bins=bins, color=S["near_miss"], alpha=0.65, label="near miss (must not merge)")
    axes[0].axvline(e["default_threshold"], color=INK, ls="--", lw=1)
    axes[0].text(e["default_threshold"] + 0.01, axes[0].get_ylim()[1] * 1.0, "default\n0.92", ha="left", fontsize=7.8, va="top")
    axes[0].set_xlabel("cosine similarity to the stored fact (MiniLM)")
    axes[0].set_ylabel("statements")
    axes[0].legend(frameon=False, fontsize=7.8, loc="upper left")
    axes[0].set_xlim(0.4, 1.06)
    for curve, c, lab in ((e["curve"], S["v2"], "cosine only"), (e["curve_guard"], S["guard"], "cosine + value guard")):
        axes[1].plot([p["recall"] for p in curve], [p["precision"] for p in curve], "o-", ms=3, color=c, label=lab)
    place = {0.92: dict(xytext=(4, 9), ha="center", va="bottom"), 0.7: dict(xytext=(-8, 7))}  # 0.92 sits above its point, clear of the zig-zag
    for p in e["curve"]:
        for t, kw in place.items():
            if abs(p["threshold"] - t) < 1e-6:
                axes[1].annotate(f"t = {t:.2f}", (p["recall"], p["precision"]), textcoords="offset points", fontsize=7.5, **kw)
    axes[1].set_xlabel("recall of correct merges")
    axes[1].set_ylabel("precision of merges")
    axes[1].set_xlim(-0.02, 1.0)
    axes[1].set_ylim(0.4, 1.04)
    axes[1].legend(frameon=False, loc="lower right", fontsize=7.8)
    fig.tight_layout(w_pad=2)
    save(fig, "fig-dedup")


def fig_mcp_rest() -> None:
    m = R2["E14"]
    r = R2["E15"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3))
    keys = list(m)
    v = [m[k]["cold_start_first_result_s"] for k in keys]
    bars = axes[0].bar([f"{int(k):,} memories" for k in keys], v, color=[ROLE["accent"]["base"], S["mcp"]], width=0.55)
    for b_, val in zip(bars, v, strict=True):
        axes[0].text(b_.get_x() + b_.get_width() / 2, val + max(v) * 0.02, f"{val:.2f} s", ha="center", fontsize=8)
    axes[0].set_ylabel("start to first tool result (s)")
    axes[0].set_ylim(0, max(v) * 1.2)
    axes[0].grid(axis="x", visible=False)
    tools = ["remember", "recall", "context_pack"]
    x = np.arange(len(tools))
    big = keys[-1]
    axes[1].bar(x - 0.2, [m[big]["tool_p50_ms"][t] for t in tools], 0.4, color=S["mcp"], label="MCP over stdio")
    rest = {"remember": r["p50_ms"]["add"], "recall": r["p50_ms"]["search"], "context_pack": None}
    axes[1].bar(x + 0.2, [rest[t] or 0 for t in tools], 0.4, color=S["rest"], label="REST over HTTP (add, search)")
    axes[1].set_xticks(x, tools)
    axes[1].set_ylabel("latency p50 (ms)")
    axes[1].grid(axis="x", visible=False)
    top = max(max(m[big]["tool_p50_ms"][t] for t in tools), r["p50_ms"]["search"])
    axes[1].set_ylim(0, top * 1.45)
    axes[1].legend(frameon=False, fontsize=7.8, loc="upper left", ncol=1)
    fig.tight_layout(w_pad=2)
    save(fig, "fig-mcp-rest")


def fig_threads() -> None:
    rows = R2["E18"]["rows"]
    t = [r["threads"] for r in rows]
    med = np.array([r["qps"] for r in rows])
    lo = med - np.array([min(r.get("qps_runs", [r["qps"]])) for r in rows])
    hi = np.array([max(r.get("qps_runs", [r["qps"]])) for r in rows]) - med
    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    ax.errorbar(t, med, yerr=[lo, hi], fmt="o-", color=S["v2"], ms=4, capsize=3, lw=1.6, elinewidth=0.9, label="median, range of 3 runs")
    ax.set_xscale("log", base=2)
    ax.set_xticks(t, [str(x) for x in t])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("client threads sharing one vault")
    ax.set_ylabel("queries per second")
    ax.set_ylim(0, max(med + hi) * 1.15)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig-threads")


def fig_reopen() -> None:
    rows = R1["E5"]["rows"]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    ax.bar(x - 0.18, [r["open_rebuild_s"] for r in rows], 0.36, color=S["v1"], label="rebuild from vectors")
    ax.bar(x + 0.18, [r["open_snapshot_s"] for r in rows], 0.36, color=S["v2"], label="load snapshot")
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{r['n'] // 1000}k docs" for r in rows])
    ax.set_ylabel("time to open (s, log scale)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=7.8)
    save(fig, "fig-reopen-times")


def compose(left: str, right: str, out: str) -> None:
    """Place two single-column charts side by side as panels (a) and (b)."""
    from PIL import Image, ImageDraw, ImageFont

    a, b = Image.open(FIG / f"{left}.png").convert("RGB"), Image.open(FIG / f"{right}.png").convert("RGB")
    h = max(a.height, b.height)
    a = a.resize((round(a.width * h / a.height), h), Image.LANCZOS)
    b = b.resize((round(b.width * h / b.height), h), Image.LANCZOS)
    gap, top = 90, 110
    canvas = Image.new("RGB", (a.width + b.width + gap, h + top), "white")
    canvas.paste(a, (0, top))
    canvas.paste(b, (a.width + gap, top))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf", 64)
    draw.text((10, 10), "(a)", fill=INK, font=font)
    draw.text((a.width + gap + 10, 10), "(b)", fill=INK, font=font)
    canvas.save(FIG / f"{out}.png", dpi=(300, 300))


def compose_all() -> None:
    compose("fig-alpha-sweep", "fig-latency-cdf", "fig-pair-alpha-cdf")
    compose("fig4-namespace-recall", "fig-filters", "fig-pair-namespace-filters")
    compose("fig3-bm25-insert", "fig-reopen-times", "fig-pair-bm25-reopen")
    compose("fig-deletes", "fig-threads", "fig-pair-deletes-threads")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig_grover_geometry()
    fig_rbac()
    fig_reopen()
    for fn, key in ((fig_alpha, "E8"), (fig_methods, "E8"), (fig_stages, "E17"), (fig_latency_cdf, "E8"),
                    (fig_encryption, "E9"), (fig_deletes, "E10"), (fig_filters, "E11"), (fig_ingest, "E12"),
                    (fig_dedup, "E13"), (fig_mcp_rest, "E14"), (fig_threads, "E18")):
        if key in R2 and (key != "E14" or "E15" in R2):
            fn()
    compose_all()
    print("figures written to", FIG)
