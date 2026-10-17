// VecForge — Universal Local-First Vector Database
// Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
// Licensed under the Business Source License 1.1 (BSL 1.1)
//
// Render assets/brand/reel/vecforge-reel.html to an MP4 reel.
//
//   python scripts/make_reel_music.py /tmp/reel-music.wav
//   NODE_PATH=$(npm root -g) node scripts/render_reel.cjs --audio /tmp/reel-music.wav
//
// Options: --fps 30, --out path.mp4, --audio track.wav (silent track if omitted).
// Needs Node with the `playwright` package, a Chromium build and ffmpeg.
// Each frame calls render(t) in the page, so output is deterministic.
"use strict";
const path = require("path");
const { spawn } = require("child_process");
const { chromium } = require("playwright");

const ROOT = path.resolve(__dirname, "..");
const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : def;
};
const FPS = Number(opt("--fps", "30"));
const OUT = path.resolve(opt("--out", path.join(ROOT, "assets/brand/vecforge-reel.mp4")));
const AUDIO = opt("--audio", null);
const POSTER = path.join(path.dirname(OUT), "vecforge-reel-poster.png");
const SRC = "file://" + path.join(ROOT, "assets/brand/reel/vecforge-reel.html") + "?static&t=0";

(async () => {
  const launch = { args: ["--no-sandbox", "--force-color-profile=srgb"] };
  if (process.env.CHROME) launch.executablePath = process.env.CHROME;
  const browser = await chromium.launch(launch);
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 }, deviceScaleFactor: 1 });
  await page.goto(SRC);
  await page.evaluate(() => document.fonts.ready);
  const duration = await page.evaluate(() => window.DURATION);
  const frames = Math.round(duration * FPS);

  const ff = spawn("ffmpeg", [
    "-y", "-loglevel", "error",
    "-f", "image2pipe", "-framerate", String(FPS), "-c:v", "png", "-i", "-",
    ...(AUDIO ? ["-i", path.resolve(AUDIO)] : ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]),
    "-map", "0:v", "-map", "1:a", "-shortest",
    // why: social apps normalise to about -14 LUFS; match it so nothing gets squashed
    ...(AUDIO ? ["-af", "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000"] : []),
    "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-profile:v", "high",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    "-c:a", "aac", "-b:a", "192k",
    OUT,
  ], { stdio: ["pipe", "inherit", "inherit"] });
  const done = new Promise((res, rej) => ff.on("close", c => (c === 0 ? res() : rej(new Error("ffmpeg exit " + c)))));

  for (let f = 0; f < frames; f++) {
    await page.evaluate(t => window.render(t), f / FPS);
    const png = await page.screenshot({ type: "png" });
    if (!ff.stdin.write(png)) await new Promise(r => ff.stdin.once("drain", r));
    if (f % FPS === 0) process.stdout.write(`\rframe ${f}/${frames}`);
  }
  ff.stdin.end();
  await done;

  await page.evaluate(() => window.render(40.0));
  await page.screenshot({ path: POSTER, type: "png" });
  await browser.close();
  console.log(`\nwrote ${OUT} (${frames} frames @ ${FPS} fps) and ${POSTER}`);
})().catch(err => {
  console.error(err);
  process.exit(1);
});
