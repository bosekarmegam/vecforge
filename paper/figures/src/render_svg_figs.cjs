// Render the hand-drawn SVG figures (*.html in this folder) to PNG in ../.
"use strict";
const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");
(async () => {
  const b = await chromium.launch({ args: ["--no-sandbox"] });
  const p = await b.newPage({ deviceScaleFactor: 2.5, viewport: { width: 1100, height: 800 } });
  for (const f of fs.readdirSync(__dirname).filter((x) => x.endsWith(".html")).sort()) {
    await p.goto("file://" + path.join(__dirname, f));
    await p.evaluate(() => document.fonts.ready);
    const out = path.join(__dirname, "..", f.replace(/\.html$/, ".png"));
    await (await p.$("svg")).screenshot({ path: out });
    console.log("wrote", path.basename(out));
  }
  await b.close();
})().catch((e) => { console.error(e); process.exit(1); });
