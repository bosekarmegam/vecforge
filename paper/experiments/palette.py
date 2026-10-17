# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under the Business Source License 1.1 (BSL 1.1)

"""The paper's colour system, shared by every chart script.

The single source of truth is paper/figures/src/palette.json, which the
diagram renderers read as well, so charts and diagrams always agree.
"""

from __future__ import annotations

import json
from pathlib import Path

_P = json.loads((Path(__file__).resolve().parents[1] / "figures" / "src" / "palette.json").read_text())

INK: str = _P["ink"]
MUTED: str = _P["muted"]
GRID: str = _P["grid"]
RULE: str = _P["rule"]
ROLE: dict[str, dict[str, str]] = _P["roles"]
S: dict[str, str] = _P["series"]  # semantic series colours, e.g. S["v1"], S["bm25"]

# Okabe-Ito order for anything that has no semantic colour of its own.
CATEGORICAL: list[str] = [
    ROLE["proposed"]["base"], ROLE["dense"]["base"], ROLE["sparse"]["base"], ROLE["alt"]["base"],
    ROLE["accent"]["base"], ROLE["baseline"]["base"], ROLE["neutral"]["base"],
]

RC: dict[str, object] = {
    "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "grid.color": GRID, "grid.linewidth": 0.7, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.axisbelow": True, "hatch.linewidth": 0.6,
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Serif", "mathtext.it": "Liberation Serif:italic",
    "mathtext.bf": "Liberation Serif:bold", "mathtext.cal": "Liberation Serif:italic", "mathtext.sf": "Liberation Sans",
    "mathtext.tt": "DejaVu Sans Mono", "axes.prop_cycle": __import__("cycler").cycler(color=CATEGORICAL),
}
