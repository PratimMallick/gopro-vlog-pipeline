"""Stage F — music analysis: tempo, beat grid, bars/phrases and energy curve.

Pure NumPy (spectral-flux onsets, autocorrelation tempo, Ellis-style DP beat tracking),
so no dependency on librosa/lzma.
"""
from __future__ import annotations

import subprocess

import numpy as np

from common import PROJECT_DIR, REPORTS, get_logger, read_json, write_json

log = get_logger("stage_f")
MUSIC = PROJECT_DIR / "music"
SR, HOP, NFFT = 22050, 512, 2048


def decode(path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def onset_envelope(y: np.ndarray) -> np.ndarray:
    n = 1 + (len(y) - NFFT) // HOP
    win = np.hanning(NFFT).astype(np.float32)
    prev = None
    env = np.zeros(n, dtype=np.float32)
    for i in range(n):
        mag = np.log1p(10 * np.abs(np.fft.rfft(y[i * HOP:i * HOP + NFFT] * win)))
        if prev is not None:
            env[i] = np.maximum(0, mag - prev).sum()
        prev = mag
    env -= np.convolve(env, np.ones(16) / 16, mode="same")
    return np.maximum(env, 0) / (env.std() + 1e-9)


def estimate_period(env: np.ndarray) -> float:
    fps = SR / HOP
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    lags = np.arange(len(ac))
    bpm = 60 * fps / np.maximum(lags, 1)
    weight = np.exp(-0.5 * (np.log2(bpm / 115) / 0.9) ** 2)  # prior centred on 115 bpm
    ok = (bpm > 60) & (bpm < 180)
    return float(lags[ok][np.argmax((ac * weight)[ok])])


def track_beats(env: np.ndarray, period: float, tightness: float = 100) -> np.ndarray:
    n = len(env)
    score = env.copy()
    back = -np.ones(n, dtype=int)
    lo, hi = int(round(period / 2)), int(round(period * 2))
    for i in range(hi, n):
        prev = np.arange(i - hi, i - lo)
        pen = -tightness * np.log((i - prev) / period) ** 2
        j = int(np.argmax(score[prev] + pen))
        score[i] = env[i] + score[prev][j] + pen[j]
        back[i] = prev[j]
    i = int(np.argmax(score[-hi:])) + n - hi
    beats = []
    while i >= 0:
        beats.append(i)
        i = back[i]
    return np.array(beats[::-1]) * HOP / SR


def analyse(path) -> dict:
    y = decode(path)
    env = onset_envelope(y)
    period = estimate_period(env)
    beats = track_beats(env, period)
    fr = np.clip((beats * SR / HOP).astype(int), 0, len(env) - 1)
    strength = env[fr]
    phase = int(np.argmax([strength[k::4].mean() for k in range(4)])) if len(beats) >= 8 else 0
    bars = beats[phase::4]
    energy = []
    for t0 in np.arange(0, len(y) / SR, 2.0):
        seg = y[int(t0 * SR):int((t0 + 2) * SR)]
        energy.append([round(float(t0), 1), round(float(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9)), 1)])
    return {"file": path.name, "duration_s": round(len(y) / SR, 3),
            "tempo_bpm": round(60 * SR / HOP / period, 1),
            "beats": [round(float(b), 3) for b in beats], "bars": [round(float(b), 3) for b in bars],
            "phrases": [round(float(b), 3) for b in bars[::2]], "energy_2s": energy}


def main() -> None:
    out = REPORTS / "music_analysis.json"
    res = read_json(out, {}) or {}
    for p in sorted(MUSIC.glob("*.mp3")):
        if p.name in res:
            continue
        log.info("analysing %s", p.name)
        res[p.name] = a = analyse(p)
        e = [v for _, v in a["energy_2s"]]
        log.info("  %.0f bpm, %.0fs, %d beats, energy min/med/max %.0f/%.0f/%.0f dB", a["tempo_bpm"],
                 a["duration_s"], len(a["beats"]), min(e), float(np.median(e)), max(e))
        write_json(out, res)


if __name__ == "__main__":
    main()
