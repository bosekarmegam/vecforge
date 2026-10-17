# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under the Business Source License 1.1 (BSL 1.1)

"""
Synthesize the original soundtrack for the VecForge brand reel.

45 s at 96 BPM (18 bars of 2.5 s), A minor, laid out to match
assets/brand/reel/vecforge-reel.html:

  0–5 s    pad intro and rising swell; impact + bell on the 5 s downbeat
  5–10 s   bass and half-time kick under the wordmark
  10–35 s  full groove (four-on-the-floor, claps, hats, echoing arpeggio)
           with a chord change on every bar and a swell into each scene
  35–45 s  end card: bell, drums thin out, final Am9 rings and fades

Everything is generated here, so the track carries no licensing terms.

Usage::

    python scripts/make_reel_music.py out.wav
"""

from __future__ import annotations

import sys
import wave

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

SR = 44_100
BPM = 96.0
BEAT = 60.0 / BPM  # 0.625 s
BAR = 4 * BEAT  # 2.5 s
DUR = 45.0
N = int(SR * DUR)
RNG = np.random.default_rng(20261017)

# chord per bar (MIDI): (bass root, pad voicing)
AM7 = (45, [57, 60, 64, 67, 71])
FMAJ7 = (41, [53, 57, 60, 64, 69])
CMAJ7 = (48, [55, 59, 60, 64, 67])
G6 = (43, [55, 59, 62, 64, 67])
AM9 = (45, [57, 60, 64, 67, 71, 76])
PROG = [AM7, AM7] + [AM7, FMAJ7, CMAJ7, G6] * 3 + [FMAJ7, G6, AM9, AM9]


def hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def at(t: float) -> int:
    return int(round(t * SR))


def place(buf: np.ndarray, sig: np.ndarray, t: float) -> None:
    i = at(t)
    j = min(len(buf), i + len(sig))
    if j > i:
        buf[i:j] += sig[: j - i]


def env_ad(n: int, attack: float, decay: float) -> np.ndarray:
    tt = np.arange(n) / SR
    a = np.clip(tt / max(attack, 1e-4), 0, 1)
    return a * np.exp(-np.maximum(tt - attack, 0) / decay)


def saw(f: float, n: int, phase: float = 0.0, harmonics: int = 10) -> np.ndarray:
    tt = np.arange(n) / SR
    out = np.zeros(n)
    for k in range(1, harmonics + 1):
        if f * k > 9000:
            break
        out += np.sin(2 * np.pi * f * k * tt + phase * k) / k * np.exp(-0.18 * k)
    return out


def lowpass(x: np.ndarray, cutoff: float, order: int = 2) -> np.ndarray:
    sos = butter(order, cutoff, btype="low", fs=SR, output="sos")
    return sosfilt(sos, x)


def bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    sos = butter(2, [lo, hi], btype="band", fs=SR, output="sos")
    return sosfilt(sos, x)


def highpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    sos = butter(2, cutoff, btype="high", fs=SR, output="sos")
    return sosfilt(sos, x)


def reverb(x: np.ndarray, seed_ms: float) -> np.ndarray:
    """Schroeder reverb: four combs in parallel, two allpasses in series."""
    out = np.zeros_like(x)
    for ms, g in ((29.7, 0.83), (37.1, 0.81), (41.1, 0.79), (43.7, 0.77)):
        d = int(SR * (ms + seed_ms) / 1000)
        a = np.zeros(d + 1)
        a[0], a[d] = 1.0, -g
        out += lfilter([1.0], a, x)
    for ms, g in ((5.0, 0.7), (1.7, 0.7)):
        d = int(SR * ms / 1000)
        b = np.zeros(d + 1)
        b[0], b[d] = -g, 1.0
        a = np.zeros(d + 1)
        a[0], a[d] = 1.0, -g
        out = lfilter(b, a, out)
    return lowpass(out, 6000) * 0.25


# ── instruments ─────────────────────────────────────────────────────
def pad(left: np.ndarray, right: np.ndarray) -> None:
    for bar, (_, notes) in enumerate(PROG):
        t0 = bar * BAR
        length = BAR + 1.2
        n = at(length)
        tt = np.arange(n) / SR
        env = np.clip(tt / 0.45, 0, 1) * np.clip((length - tt) / 1.0, 0, 1)
        if bar >= 16:  # final chord rings out
            n = at(DUR - t0)
            tt = np.arange(n) / SR
            env = np.clip(tt / 0.6, 0, 1) * np.exp(-tt / 6.0)
        for i, m in enumerate(notes):
            f = hz(m)
            for side, cents in ((left, -6.0), (right, 6.0)):
                sig = saw(f * 2 ** (cents / 1200), n, phase=i * 0.7 + cents)
                side_sig = sig * env * 0.055
                place(side, side_sig, t0)


def bass(buf: np.ndarray, start: float, end: float) -> None:
    pattern = [0.0, 1.5, 2.0, 3.5]  # beats: 1, 2&, 3, 4&
    for bar, (root, _) in enumerate(PROG):
        for b in pattern:
            t = bar * BAR + b * BEAT
            if not start <= t < end:
                continue
            n = at(0.42)
            tt = np.arange(n) / SR
            f = hz(root - 12)
            sig = np.sin(2 * np.pi * f * tt) + 0.35 * np.sin(2 * np.pi * 2 * f * tt)
            sig = np.tanh(1.6 * sig) * env_ad(n, 0.006, 0.22) * 0.32
            place(buf, sig, t)


def kick(n_len: float = 0.45) -> np.ndarray:
    n = at(n_len)
    tt = np.arange(n) / SR
    f = 46 + 90 * np.exp(-tt / 0.035)
    phase = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(phase) * env_ad(n, 0.002, 0.16) * 0.9


def clap() -> np.ndarray:
    n = at(0.25)
    noise = bandpass(RNG.standard_normal(n), 1100, 5200)
    tone = np.sin(2 * np.pi * 185 * np.arange(n) / SR) * env_ad(n, 0.001, 0.04)
    return (noise * env_ad(n, 0.002, 0.07) * 0.5 + tone * 0.2) * 0.55


def hat() -> np.ndarray:
    n = at(0.08)
    return highpass(RNG.standard_normal(n), 7500) * env_ad(n, 0.001, 0.018) * 0.16


def pluck(f: float) -> np.ndarray:
    n = at(0.5)
    tt = np.arange(n) / SR
    sig = np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * 2 * f * tt + 0.3)
    return sig * env_ad(n, 0.003, 0.14) * 0.11


def bell(f: float, length: float = 3.0) -> np.ndarray:
    n = at(length)
    tt = np.arange(n) / SR
    index = 3.0 * np.exp(-tt / 0.6)
    mod = np.sin(2 * np.pi * f * 3.5 * tt) * index
    return np.sin(2 * np.pi * f * tt + mod) * env_ad(n, 0.002, 0.9) * 0.16


def swell(length: float, peak: float = 0.22) -> np.ndarray:
    """Filtered-noise riser that opens up and stops dead at its end."""
    n = at(length)
    noise = RNG.standard_normal(n)
    out = np.zeros(n)
    y = 0.0
    for i in range(n):  # one-pole lowpass with a rising cutoff
        p = i / n
        a = 1 - np.exp(-2 * np.pi * (300 + 7000 * p**2) / SR)
        y += a * (noise[i] - y)
        out[i] = y
    return out * (np.linspace(0, 1, n) ** 2.2) * peak


def impact() -> np.ndarray:
    n = at(2.2)
    tt = np.arange(n) / SR
    boom = np.sin(2 * np.pi * (38 + 40 * np.exp(-tt / 0.08)) * tt) * env_ad(n, 0.002, 0.7)
    return boom * 0.7 + np.pad(kick(), (0, n - at(0.45)))


def build() -> np.ndarray:
    left, right = np.zeros(N), np.zeros(N)
    mono = np.zeros(N)
    drums = np.zeros(N)
    send = np.zeros(N)  # reverb send
    kick_env = np.zeros(N)

    pad(left, right)

    # intro: swell into the spark at 5.0 s
    place(mono, swell(2.4, 0.12), 2.6)
    place(mono, impact(), 5.0)
    b = bell(hz(81)) + 0.6 * bell(hz(88))
    place(send, b, 5.0)
    place(mono, b * 0.8, 5.0)

    bass(mono, 5.0, 40.0)

    k = kick()
    for i in range(int(DUR / BEAT)):
        t = i * BEAT
        beat_in_bar = i % 4
        full = 10.0 <= t < 35.0
        half = (5.0 <= t < 10.0 or 35.0 <= t < 40.0) and beat_in_bar in (0, 2)
        if full or half:
            if t >= 5.01 or t < 4.99:
                place(drums, k, t)
                place(kick_env, env_ad(at(0.3), 0.002, 0.12), t)
        if full and beat_in_bar in (1, 3):
            place(drums, clap(), t)
            place(send, clap() * 0.5, t)
        if 7.5 <= t < 37.5:
            place(drums, hat(), t + BEAT / 2)
            if full:
                place(drums, hat() * 0.45, t + BEAT * 0.75)

    # arpeggio with dotted-eighth echo, 10–35 s
    pattern = [0, 2, 3, 4, 3, 2, 1, 2]
    echo = np.zeros(N)
    for bar in range(4, 14):
        _, notes = PROG[bar]
        for s_ in range(8):
            t = bar * BAR + s_ * BEAT / 2
            place(echo, pluck(hz(notes[pattern[s_] % len(notes)] + 12)), t)
    d = at(BEAT * 0.75)
    arp_l, arp_r = echo.copy(), echo.copy()
    tap = echo.copy()
    for rep in range(1, 4):
        tap = np.concatenate([np.zeros(d), tap[:-d]]) * 0.42
        (arp_l if rep % 2 else arp_r)[:] += tap
    send += echo * 0.6

    # scene swells and the end-card bell
    for t in (15.0, 20.0, 25.0, 30.0, 35.0):
        place(mono, swell(1.0, 0.06), t - 1.0)
    eb = bell(hz(76), 4.0) + 0.5 * bell(hz(83), 4.0)
    place(send, eb, 35.0)
    place(mono, eb * 0.7, 35.0)

    # sidechain: duck pad and bass under the kick
    duck = 1.0 - 0.45 * np.clip(kick_env, 0, 1)
    left *= duck
    right *= duck
    mono *= 0.85 + 0.15 * duck

    rl, rr = reverb(send + 0.5 * (left + right), 0.0), reverb(send + 0.5 * (left + right), 2.3)
    out_l = left + mono + drums + arp_l + rl * 1.4
    out_r = right + mono + drums + arp_r + rr * 1.4
    stereo = np.stack([out_l, out_r], axis=1)

    # master: gentle saturation, fade in/out, normalise to -1 dBFS
    stereo = np.tanh(stereo * 1.2) / np.tanh(1.2)
    tt = np.arange(N) / SR
    fade = np.clip(tt / 0.4, 0, 1) * np.clip((DUR - tt) / 1.6, 0, 1)
    stereo *= fade[:, None]
    stereo /= np.max(np.abs(stereo)) / 10 ** (-1 / 20)
    return stereo


def write_wav(path: str, data: np.ndarray) -> None:
    pcm = (np.clip(data, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "vecforge-reel-music.wav"
    write_wav(out, build())
    print(f"wrote {out} ({DUR:.0f} s, {BPM:.0f} BPM)")
