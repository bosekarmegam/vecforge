// Render the paper's Mermaid diagrams (*.mmd in this folder) to PNG in ../.
// Usage: MERMAID_JS=/path/to/mermaid.min.js NODE_PATH=$(npm root -g) node render_diagrams.cjs
"use strict";
const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");
const HERE = __dirname;
const OUT = path.join(HERE, "..");
const MERMAID = process.env.MERMAID_JS || "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js";
// why: palette.json is the one colour system shared by every chart and diagram in the paper
const PALETTE = JSON.parse(fs.readFileSync(path.join(HERE, "palette.json"), "utf8"));
const D = PALETTE.diagram;
const ROLE = PALETTE.roles;
const FONT = "Inter, DejaVu Sans, sans-serif";
const THEME = {
  theme: "base",
  // why: message text has no background, so a white halo keeps it legible where it crosses a lifeline
  themeCSS: ".messageText{paint-order:stroke;stroke:#FFFFFF;stroke-width:6px;stroke-linejoin:round} .edgeLabel p{padding:0 6px}",
  themeVariables: {
    fontFamily: FONT, fontSize: "16px",
    primaryColor: D.process_fill, primaryBorderColor: D.process_stroke, primaryTextColor: PALETTE.ink,
    secondaryColor: D.data_fill, secondaryBorderColor: D.data_stroke,
    tertiaryColor: D.decision_fill, tertiaryBorderColor: D.decision_stroke,
    lineColor: D.arrow, textColor: PALETTE.ink,
    clusterBkg: D.cluster_fill, clusterBorder: D.cluster_stroke, edgeLabelBackground: PALETTE.paper,
    actorBkg: D.terminal_fill, actorBorder: D.terminal_stroke, actorTextColor: PALETTE.ink,
    signalColor: D.arrow, signalTextColor: PALETTE.ink,
    noteBkgColor: D.note_fill, noteBorderColor: D.note_stroke, noteTextColor: PALETTE.ink,
    activationBkgColor: ROLE.accent.tint, activationBorderColor: ROLE.accent.base,
    labelBoxBkgColor: D.terminal_fill, labelBoxBorderColor: D.terminal_stroke, loopTextColor: ROLE.neutral.shade,
    sequenceNumberColor: PALETTE.paper,
    attributeBackgroundColorOdd: PALETTE.paper, attributeBackgroundColorEven: D.terminal_fill,
  },
  flowchart: { curve: "basis", padding: 14, nodeSpacing: 34, rankSpacing: 42 },
  sequence: { actorMargin: 34, mirrorActors: false, showSequenceNumbers: true },
  er: { fontSize: 13 },
};
(async () => {
  const files = fs.readdirSync(HERE).filter((f) => f.endsWith(".mmd")).sort();
  const browser = await chromium.launch({ args: ["--no-sandbox"] });
  const page = await browser.newPage({ deviceScaleFactor: 2.5, viewport: { width: 1400, height: 1000 } });
  // why: Mermaid measures sequence text in an off-screen element that inherits the body font, so the body must use the theme font or notes overflow their box
  await page.setContent(`<!doctype html><html><head><meta charset="utf-8"><style>body{margin:0;background:#fff;font-family:${FONT}} #c{display:inline-block;padding:18px}</style></head><body><div id="c"></div></body></html>`);
  await page.addScriptTag({ path: fs.existsSync(MERMAID) ? MERMAID : undefined, url: fs.existsSync(MERMAID) ? undefined : MERMAID });
  // why: Mermaid measures labels when it renders, so the font must be loaded first
  await page.evaluate(async () => {
    for (const w of ["400", "600"]) await document.fonts.load(`${w} 16px Inter`);
  });
  await page.evaluate((cfg) => window.mermaid.initialize({ startOnLoad: false, ...cfg }), THEME);
  for (const f of files) {
    const src = fs.readFileSync(path.join(HERE, f), "utf8");
    const svg = await page.evaluate(async (code) => (await window.mermaid.render("d" + Math.random().toString(36).slice(2), code)).svg, src);
    await page.evaluate((s) => {
      const c = document.getElementById("c");
      c.innerHTML = s;
      const el = c.querySelector("svg");
      // why: Mermaid sets width 100% plus a max-width; use the natural width instead
      const natural = parseFloat(el.style.maxWidth) || el.viewBox.baseVal.width;
      el.style.width = natural + "px";
      el.style.maxWidth = "none";
      el.removeAttribute("width");
    }, svg);
    await page.evaluate(() => document.fonts.ready);
    const out = path.join(OUT, f.replace(/\.mmd$/, ".png"));
    await (await page.$("#c")).screenshot({ path: out });
    console.log("wrote", path.basename(out));
  }
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
