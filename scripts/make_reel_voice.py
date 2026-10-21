# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under the Business Source License 1.1 (BSL 1.1)

"""
Generate the voiceover for the VecForge brand reel with a local neural voice.

Reads the script from assets/brand/reel/narration.js (the same file the
reel's captions use), speaks every line with Kokoro-82M through sherpa-onnx,
fits each line inside its time slot, and writes one 44.1 kHz mono WAV.

Model: the sherpa-onnx "kokoro-int8-en-v0_19" folder (model.int8.onnx,
voices.bin, tokens.txt, espeak-ng-data). Kokoro-82M is Apache-2.0.

Usage::

    pip install sherpa-onnx
    python scripts/make_reel_voice.py --model-dir kokoro-int8-en-v0_19 --out voice.wav
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "assets" / "brand" / "reel" / "narration.js"
SR = 44_100
DURATION = 60.0
MAX_SPEED = 1.25


def load_script(path: Path = SCRIPT) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    lines: list[dict[str, object]] = json.loads(
        text[text.index("[") : text.rindex("]") + 1]
    )
    return lines


def make_tts(model_dir: Path, threads: int):  # type: ignore[no-untyped-def]
    import sherpa_onnx

    kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
        model=str(model_dir / "model.int8.onnx"),
        voices=str(model_dir / "voices.bin"),
        tokens=str(model_dir / "tokens.txt"),
        data_dir=str(model_dir / "espeak-ng-data"),
    )
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(kokoro=kokoro, num_threads=threads)
    )
    return sherpa_onnx.OfflineTts(config)


def speak(tts, text: str, sid: int, speed: float) -> np.ndarray:  # type: ignore[no-untyped-def]
    audio = tts.generate(text, sid=sid, speed=speed)
    x = np.asarray(audio.samples, dtype=np.float64)
    # resample 24 kHz -> 44.1 kHz
    return resample_poly(x, SR // 100, audio.sample_rate // 100)


def trim(x: np.ndarray, threshold: float = 0.004) -> np.ndarray:
    idx = np.flatnonzero(np.abs(x) > threshold)
    if idx.size == 0:
        return x
    pad = int(0.03 * SR)
    return x[max(0, idx[0] - pad) : idx[-1] + pad]


def polish(x: np.ndarray) -> np.ndarray:
    """High-pass rumble, gentle compression, normalise to -1 dBFS."""
    x = sosfilt(butter(2, 75, btype="high", fs=SR, output="sos"), x)
    peak = float(np.max(np.abs(x))) or 1.0
    x = x / peak
    x = np.tanh(1.8 * x) / np.tanh(1.8)
    return x * 10 ** (-1 / 20) / max(float(np.max(np.abs(x))), 1e-9)


def build(
    model_dir: Path,
    sid: int,
    threads: int,
    script: Path = SCRIPT,
    timings: list | None = None,
) -> np.ndarray:
    """Speak every line into a 60 s track; log each spoken length to ``timings``."""
    tts = make_tts(model_dir, threads)
    out = np.zeros(int(SR * DURATION))
    print(f"{'start':>6} {'slot':>5} {'len':>5} {'speed':>5}  line")
    for line in load_script(script):
        start, end = float(line["start"]), float(line["end"])  # type: ignore[arg-type]
        say = str(line.get("say") or line["text"])
        slot = end - start
        speed = 1.05
        clip = trim(speak(tts, say, sid, speed))
        while len(clip) / SR > slot and speed < MAX_SPEED:
            speed = min(MAX_SPEED, speed * min(1.08, len(clip) / SR / slot + 0.01))
            clip = trim(speak(tts, say, sid, speed))
        i = int(start * SR)
        j = min(len(out), i + len(clip))
        out[i:j] += clip[: j - i]
        seconds = len(clip) / SR
        if timings is not None:
            timings.append({"start": start, "spoken": round(seconds, 3)})
        print(f"{start:6.1f} {slot:5.1f} {seconds:5.2f} {speed:5.2f}  {line['text']}")
    return polish(out)


def write_wav(path: str, x: np.ndarray) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--out", default="vecforge-reel-voice.wav")
    ap.add_argument("--speaker", type=int, default=1, help="1 = af_bella (default)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument(
        "--script",
        type=Path,
        default=SCRIPT,
        help="narration file (default: brand reel)",
    )
    ap.add_argument(
        "--timings", type=Path, help="write each line's spoken length as JSON"
    )
    args = ap.parse_args()
    timings: list = []
    write_wav(
        args.out,
        build(args.model_dir, args.speaker, args.threads, args.script, timings),
    )
    if args.timings:
        args.timings.write_text(json.dumps(timings, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
