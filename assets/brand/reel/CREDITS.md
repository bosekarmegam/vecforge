# VecForge reel credits

- **Design, animation and script:** VecForge, Suneel Bose K · ArcGX TechLabs Private Limited.
- **Music:** original, synthesized by `scripts/make_reel_music.py`. No samples or third-party audio.
- **Voice:** generated locally with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
  (Apache-2.0), voice `af_bella`, through [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
  (Apache-2.0) using its `kokoro-int8-en-v0_19` model package.
- **Fonts:** Inter (SIL Open Font License) and DejaVu Sans Mono (Bitstream Vera license).

## Rebuild

```bash
pip install sherpa-onnx scipy numpy
python scripts/make_reel_voice.py --model-dir kokoro-int8-en-v0_19 --out /tmp/voice.wav
python scripts/make_reel_music.py /tmp/reel-mix.wav --voice /tmp/voice.wav
NODE_PATH=$(npm root -g) node scripts/render_reel.cjs --audio /tmp/reel-mix.wav
```

The narration text lives in `narration.js`; the captions in `vecforge-reel.html` read the same file.
