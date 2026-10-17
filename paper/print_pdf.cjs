// Print paper HTML to PDF once KaTeX has typeset every formula and fonts are ready.
// Usage: node print_pdf.cjs <file-url> <out.pdf>   (needs the playwright package)
"use strict";
const { chromium } = require("playwright");
(async () => {
  const [url, out] = process.argv.slice(2);
  const launch = { args: ["--no-sandbox"] };
  if (process.env.CHROME) launch.executablePath = process.env.CHROME;
  const browser = await chromium.launch(launch);
  const page = await browser.newPage();
  await page.goto(url, { waitUntil: "load" });
  await page.waitForFunction(() => {
    const math = document.querySelectorAll("span.math");
    return math.length === 0 || [...math].every((m) => m.querySelector(".katex"));
  }, null, { timeout: 30000 });
  await page.evaluate(() => document.fonts.ready);
  await page.pdf({ path: out, preferCSSPageSize: true, printBackground: true });
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
