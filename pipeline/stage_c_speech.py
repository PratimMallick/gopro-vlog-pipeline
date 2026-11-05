"""Stage C — local speech recognition (Whisper large-v3 via MLX), per clip, resumable.

Two passes per clip with speech: verbatim transcription in the detected language
(Hindi / English / code-switched) and Whisper's own speech-to-English translation.
Segments get acoustic QA metrics so hallucinations on wind/silence can be rejected.
"""
from __future__ import annotations

import time

import numpy as np
import soundfile as sf

from common import CFG, REPORTS, WORK, fmt_tc, get_logger, load_inventory, read_json, run, write_json

log = get_logger("stage_c")
T = CFG["transcription"]
AUD = WORK / "audio16k"
TR = WORK / "transcripts"
AUD.mkdir(parents=True, exist_ok=True)
TR.mkdir(parents=True, exist_ok=True)
HALLUCINATIONS = ("subscribe", "सब्सक्राइब", "thank you for watching", "thanks for watching",
                  "like and share", "please like", "amara.org", "चैनल को", "धन्यवाद देखने",
                  "welcome to our channel", "welcome to my channel", "बातचीत हिंदी में")
PROMPT_ECHO = "बातचीत हिंदी में"  # text of an earlier initial_prompt that Whisper parroted on noise
# Whole-segment texts Whisper emits on wind/surf noise in Hindi mode.
NOISE_TEXTS = {"कर दो", "अजय को", "अजय को अ", "कि अ", "अ", "हैं", "है", "झाल", "प्रस्तुति", "प्रश्न"}


def extract_audio(clip: dict):
    out = AUD / f"{clip['clip_id']}.wav"
    if not out.exists():
        run(["ffmpeg", "-v", "error", "-y", "-i", clip["source_path"], "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", str(out)])
    return out


def band_metrics(x: np.ndarray, sr: int = 16000) -> dict:
    if len(x) < sr // 10:
        return {"rms_db": -99.0, "voice_ratio": 0.0, "wind_ratio": 0.0}
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    tot = spec.sum() + 1e-12
    return {"rms_db": round(20 * np.log10(rms), 1),
            "voice_ratio": round(float(spec[(f > 250) & (f < 3500)].sum() / tot), 3),
            "wind_ratio": round(float(spec[f < 150].sum() / tot), 3)}


def qa(seg: dict, x: np.ndarray, sr: int = 16000) -> dict:
    a = x[int(seg["start"] * sr): int(seg["end"] * sr)]
    m = band_metrics(a)
    text = seg["text"].strip()
    flags = []
    if seg.get("no_speech_prob", 0) > 0.6 and seg.get("avg_logprob", 0) < -0.8:
        flags.append("likely_no_speech")
    if seg.get("compression_ratio", 1) > 2.4:
        flags.append("repetitive")
    if seg.get("avg_logprob", 0) < -1.0:
        flags.append("low_confidence")
    if any(h in text.lower() for h in HALLUCINATIONS):
        flags.append("known_hallucination")
    if m["rms_db"] < -48:
        flags.append("too_quiet")
    if m["wind_ratio"] > 0.6:
        flags.append("wind")
    if len(text) < 2 or text.strip(" .।") in NOISE_TEXTS or set(text.replace("कर दो", "").strip()) <= set(" ।."):
        flags.append("empty")
    words = text.split()
    if len(words) >= 6 and len(set(words)) / len(words) < 0.35:
        flags.append("repetitive")
    if seg["end"] <= seg["start"]:
        flags.append("bad_timing")
    keep = not ({"likely_no_speech", "known_hallucination", "repetitive", "empty", "too_quiet",
                 "bad_timing"} & set(flags))
    return {**m, "flags": flags, "keep": keep}


def transcribe(path, task: str, language: str | None):
    import mlx_whisper
    # No initial prompt: with one, Whisper echoes the prompt text over wind/surf.
    return mlx_whisper.transcribe(str(path), path_or_hf_repo=T["model"], task=task, language=language,
                                  condition_on_previous_text=False, initial_prompt=None, word_timestamps=False,
                                  no_speech_threshold=0.6, logprob_threshold=-1.0,
                                  compression_ratio_threshold=2.4, temperature=(0.0, 0.2, 0.4))


def process(clip: dict) -> dict:
    cid = clip["clip_id"]
    out = TR / f"{cid}.json"
    cached = read_json(out)
    if cached and any(PROMPT_ECHO in s["text"] for s in cached.get("segments", [])):
        log.info("%s: cached transcript is an initial-prompt echo; re-transcribing", cid)
        cached = None
    if cached:
        if cached.get("segments") or cached.get("skipped"):
            x, _ = sf.read(AUD / f"{cid}.wav", dtype="float32")
            for key in ("segments", "translation"):
                for s in cached.get(key, []):
                    s.update(qa(s, x))
        return cached
    wav = extract_audio(clip)
    x, sr = sf.read(wav, dtype="float32")
    whole = band_metrics(x)
    res = {"clip_id": cid, "audio": whole, "language": None, "segments": [], "translation": []}
    if whole["rms_db"] < T["min_speech_rms_db"]:
        res["skipped"] = "near-silent"
        write_json(out, res)
        return res
    tr = transcribe(wav, "transcribe", None)
    lang = tr.get("language")
    if lang not in ("hi", "en"):   # Hindi is often misdetected (Urdu, or noise -> random languages)
        tr = transcribe(wav, "transcribe", "hi")
        res["language_redetected_from"] = lang
        lang = "hi"
    res["language"] = lang
    for s in tr["segments"]:
        seg = {"start": round(s["start"], 2), "end": round(min(s["end"], clip["duration_s"]), 2),
               "text": s["text"].strip(), "avg_logprob": round(s.get("avg_logprob", 0), 3),
               "no_speech_prob": round(s.get("no_speech_prob", 0), 3),
               "compression_ratio": round(s.get("compression_ratio", 0), 2)}
        seg.update(qa(seg, x, sr))
        res["segments"].append(seg)
    if any(s["keep"] for s in res["segments"]) and lang != "en":
        tl = transcribe(wav, "translate", lang)
        for s in tl["segments"]:
            seg = {"start": round(s["start"], 2), "end": round(min(s["end"], clip["duration_s"]), 2),
                   "text": s["text"].strip(), "avg_logprob": round(s.get("avg_logprob", 0), 3),
                   "no_speech_prob": round(s.get("no_speech_prob", 0), 3),
                   "compression_ratio": round(s.get("compression_ratio", 0), 2)}
            seg.update(qa(seg, x, sr))
            res["translation"].append(seg)
    write_json(out, res)
    return res


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main() -> None:
    clips = [c for c in load_inventory()
             if c["source_kind"] != "unusable" and c["has_audio"] and c.get("episode_date_ok", True)]
    t0 = time.time()
    allres = []
    for n, c in enumerate(clips, 1):
        try:
            allres.append(process(c))
        except Exception as exc:
            log.error("%s failed: %s", c["clip_id"], exc)
            continue
        el = time.time() - t0
        r = allres[-1]
        log.info("[%d/%d] %s lang=%s kept=%d/%d (%.0fs, ~%.0fs left)", n, len(clips), c["clip_id"],
                 r["language"], sum(s["keep"] for s in r["segments"]), len(r["segments"]), el,
                 el / n * (len(clips) - n))
    # Per-source-clip transcript exports (source timecodes).
    srt, lines = [], []
    k = 1
    for r in allres:
        kept = [s for s in r["segments"] if s["keep"]]
        if not kept:
            continue
        lines.append(f"\n=== {r['clip_id']}  language={r['language']}  clip_rms={r['audio']['rms_db']} dB ===")
        for s in r["segments"]:
            mark = "" if s["keep"] else f"   [REJECTED: {','.join(s['flags'])}]"
            lines.append(f"[{fmt_tc(s['start'])}-{fmt_tc(s['end'])}] {s['text']}{mark}")
            if s["keep"]:
                srt.append(f"{k}\n{srt_time(s['start'])} --> {srt_time(s['end'])}\n[{r['clip_id']}] {s['text']}\n")
                k += 1
        for s in r["translation"]:
            if s["keep"]:
                lines.append(f"    EN(tr) [{fmt_tc(s['start'])}-{fmt_tc(s['end'])}] {s['text']}")
    (REPORTS / "source_transcripts.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORTS / "source_transcripts.srt").write_text("\n".join(srt), encoding="utf-8")
    write_json(REPORTS / "source_transcripts.json", allres)
    log.info("Transcripts written for %d clips", len(allres))


if __name__ == "__main__":
    main()
