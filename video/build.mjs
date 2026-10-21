// Build the 60 s VecForge introduction reel.
//
//   npm install
//   node build.mjs                 # everything below, in order
//   node build.mjs --skip-audio    # reuse hf/assets/audio/bed.wav
//   node build.mjs --skip-remotion # reuse hf/assets/remotion/*.mp4
//
// 1. Voice (Kokoro-82M, scripts/make_reel_voice.py) and soundtrack
//    (scripts/make_reel_music.py), mixed into hf/assets/audio/bed.wav.
//    Needs KOKORO_DIR pointing at the sherpa-onnx kokoro-int8-en-v0_19 folder.
// 2. Remotion renders the motion-graphics segments into hf/assets/remotion/.
// 3. HyperFrames renders hf/index.html (the master timeline: Remotion clips,
//    GSAP code scenes, captions, transitions, sound effects) to the final MP4.
//
// Set REMOTION_BROWSER / HYPERFRAMES_BROWSER_PATH to reuse a Chrome Headless
// Shell already on the machine; otherwise each tool downloads its own.
import {execFileSync} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '..');
const hf = path.join(here, 'hf');
const args = process.argv.slice(2);
const out = path.resolve(args.includes('--out') ? args[args.indexOf('--out') + 1] : path.join(repo, 'assets', 'brand', 'vecforge-intro-reel.mp4'));
const run = (cmd, argv, opts = {}) => execFileSync(cmd, argv, {stdio: 'inherit', ...opts});

// GSAP and the HyperFrames sound effects come from node_modules, not git.
fs.mkdirSync(path.join(hf, 'vendor'), {recursive: true});
fs.copyFileSync(path.join(here, 'node_modules/gsap/dist/gsap.min.js'), path.join(hf, 'vendor/gsap.min.js'));
const sfxSrc = path.join(here, 'node_modules/hyperframes/dist/skills/media-use/audio/assets/sfx');
fs.mkdirSync(path.join(hf, 'assets/sfx'), {recursive: true});
for (const name of ['glitch-1', 'sparkle', 'whoosh', 'whoosh-short', 'pop', 'typing', 'ping', 'chime']) {
  fs.copyFileSync(path.join(sfxSrc, `${name}.mp3`), path.join(hf, 'assets/sfx', `${name}.mp3`));
}

if (!args.includes('--skip-audio')) {
  const kokoro = process.env.KOKORO_DIR;
  if (!kokoro) throw new Error('Set KOKORO_DIR to the kokoro-int8-en-v0_19 model folder (or pass --skip-audio).');
  const tmp = fs.mkdtempSync(path.join(here, '.voice-'));
  const voice = path.join(tmp, 'voice.wav');
  const timings = path.join(tmp, 'timings.json');
  run('python3', [path.join(repo, 'scripts/make_reel_voice.py'), '--model-dir', kokoro, '--script', path.join(hf, 'narration.js'), '--out', voice, '--timings', timings]);
  fs.writeFileSync(path.join(hf, 'voice-timings.js'), `window.VOICE_TIMINGS = ${JSON.stringify(JSON.parse(fs.readFileSync(timings, 'utf8')))};\n`);
  fs.mkdirSync(path.join(hf, 'assets/audio'), {recursive: true});
  run('python3', [path.join(repo, 'scripts/make_reel_music.py'), path.join(hf, 'assets/audio/bed.wav'), '--voice', voice]);
  fs.rmSync(tmp, {recursive: true, force: true});
}

if (!args.includes('--skip-remotion')) run('node', [path.join(here, 'render-remotion.mjs')]);

run('npx', ['hyperframes', 'lint'], {cwd: hf});
// Render at high quality, then encode a delivery copy small enough to share and commit.
const master = path.join(here, '.reel-master.mp4');
run('npx', ['hyperframes', 'render', '-q', 'high', '-f', '30', '-o', master], {cwd: hf});
run('ffmpeg', ['-v', 'error', '-y', '-i', master, '-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-c:a', 'aac', '-b:a', '192k', out]);
fs.rmSync(master, {force: true});
console.log('reel:', out);
