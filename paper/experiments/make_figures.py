# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under the Business Source License 1.1 (BSL 1.1)

"""Draw the evaluation figures for the VecForge 2.0 paper from results.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

from palette import RC, S

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
R = json.loads((HERE / "results.json").read_text())

plt.rcParams.update(RC)


def fig_rank() -> None:
    e = R["E1"]
    n = e["sizes"]
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.scatter(
        n,
        e["series"]["1.x"]["tau_linear"],
        s=6,
        color=S["v1"],
        label="1.x (diffusion only)",
        zorder=5,
    )
    ax.plot(
        n,
        e["series"]["2.0"]["tau_linear"],
        color=S["v2"],
        lw=1.2,
        label="2.0 (oracle and diffusion)",
        zorder=3,
    )
    ax.set_xlabel("number of candidates N")
    ax.set_ylabel("Kendall τ vs. input order")
    ax.set_ylim(-1.15, 1.15)
    ax.set_yticks([-1, 0, 1])
    ax.legend(loc="center right", frameon=False)
    fig.savefig(FIG / "fig2-rank-preservation.png")
    plt.close(fig)


def fig_bm25() -> None:
    rows = R["E3"]["insert"]
    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    ax.plot(
        [r["n"] for r in rows if r["v1_s"]],
        [r["v1_s"] for r in rows if r["v1_s"]],
        "o-",
        color=S["v1"],
        ms=3.5,
        label="1.x rebuild per insert",
    )
    ax.plot(
        [r["n"] for r in rows],
        [r["v2_s"] for r in rows],
        "s-",
        color=S["v2"],
        ms=3.5,
        label="2.0 incremental",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [r["n"] for r in rows]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:,}" for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("documents inserted one by one")
    ax.set_ylabel("total time (s)")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(FIG / "fig3-bm25-insert.png")
    plt.close(fig)


def fig_namespace() -> None:
    rows = R["E4"]["rows"]
    x = [r["fraction"] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    ax.plot(
        x,
        [r["recall_postfilter_v1"] for r in rows],
        "o-",
        color=S["v1"],
        ms=3.5,
        label="1.x post-filter (4k window)",
    )
    ax.plot(
        x,
        [r["recall_flat_selector_v2"] for r in rows],
        "s-",
        color=S["v2"],
        ms=3.5,
        label="2.0 flat, ID selector",
    )
    ax.plot(
        x,
        [r["recall_hnsw_v2"] for r in rows],
        "^--",
        color=S["v2_hnsw"],
        ms=3.5,
        label="2.0 HNSW",
    )
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in x])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("namespace size (% of vault)")
    ax.set_ylabel("recall@10 within namespace")
    ax.set_ylim(-0.04, 1.06)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.55))
    fig.savefig(FIG / "fig4-namespace-recall.png")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig_rank()
    fig_bm25()
    fig_namespace()
    print("figures written to", FIG)
