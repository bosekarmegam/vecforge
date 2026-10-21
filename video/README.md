# VecForge introduction reel

A 60-second vertical (1080 × 1920, 30 fps) video that explains what VecForge is and how to use it. Output: [`assets/brand/vecforge-intro-reel.mp4`](../assets/brand/vecforge-intro-reel.mp4).

| Time | Scene | Built with |
| --- | --- | --- |
| 0–5 s | A chat forgets what it was told; "Your AI forgets." | Remotion |
| 5–10 s | The VecForge mark is forged: shockwaves, god-rays, orbiting particles | Remotion |
| 10–17.5 s | Search by meaning: text becomes a vector and lands in a rotating 3D cloud | Remotion |
| 17.5–25 s | Step 1, install: `pip install vecforge`, open a vault | HyperFrames + GSAP |
| 25–32.5 s | Step 2, add: documents fly into the vault | HyperFrames + GSAP |
| 32.5–40 s | Step 3, search: meaning and keyword retrievers fuse, results rank | HyperFrames + GSAP |
| 40–47.5 s | Memory for AI agents over the built-in MCP server | Remotion |
| 47.5–55 s | Encrypted, fast, private | Remotion |
| 55–60 s | End card: `pip install vecforge` | Remotion |

HyperFrames is the master timeline (`hf/index.html`). It places the Remotion clips, draws the three code scenes with GSAP, and adds the karaoke captions, cut transitions, film grain, progress bar and sound effects. Then it renders the final MP4.

## Layout

```
video/
├── build.mjs              # runs every step below
├── render-remotion.mjs    # Remotion segments -> hf/assets/remotion/*.mp4
├── remotion/              # React scenes (fx.tsx holds the shared effects)
└── hf/                    # HyperFrames project
    ├── index.html         # master composition
    ├── narration.js       # voiceover script, also drives the captions
    └── voice-timings.js   # spoken length of each line (written by the build)
```

## Build

Needs Node 22 or later, FFmpeg, Python 3 with `numpy`, `scipy` and `sherpa-onnx`, and the sherpa-onnx [`kokoro-int8-en-v0_19`](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models) voice model.

```bash
cd video
npm install
KOKORO_DIR=/path/to/kokoro-int8-en-v0_19 node build.mjs
```

`--skip-audio` reuses the last voice and music mix. `--skip-remotion` reuses the last Remotion clips. If a Chrome Headless Shell is already installed, set `REMOTION_BROWSER` and `HYPERFRAMES_BROWSER_PATH` to it to skip the downloads. A full build takes about 8 minutes on 4 CPU cores.

To change what is said, edit `hf/narration.js` and rebuild. The captions follow the new text and timing automatically.

## Credits and licences

- Voice: [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0) through sherpa-onnx, voice `af_bella`.
- Music: synthesized by `scripts/make_reel_music.py`. It is original and carries no licence terms.
- Sound effects: the Pixabay set bundled with HyperFrames (Pixabay Content License). The build copies them from `node_modules`, so they are not stored in this repository.
- [HyperFrames](https://github.com/heygen-com/hyperframes): Apache-2.0.
- [GSAP](https://gsap.com): GreenSock standard no-charge licence.
- [Remotion](https://www.remotion.dev): free for individuals and for companies with up to three people. Larger companies need a [company licence](https://www.remotion.dev/license).
- Fonts: Inter (SIL OFL) and DejaVu Sans Mono.
