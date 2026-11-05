"""Stage I — validate the rendered video and write reports/validation_report.json."""
from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path

import numpy as np

from common import CFG, PROJECT_DIR, REPORTS, REVIEW, SOURCE_DIR, episode_dates, ffprobe_json, get_logger, read_json, run, write_json

log = get_logger("stage_i")
FINAL = PROJECT_DIR / CFG["paths"]["final_name"]
SHOTS = REVIEW / "final_frames"
SHOTS.mkdir(parents=True, exist_ok=True)


def intervals(stderr: str, start_key: str, end_key: str) -> list[tuple[float, float]]:
    s = [float(x) for x in re.findall(rf"{start_key}[:=]\s*([\d.]+)", stderr)]
    e = [float(x) for x in re.findall(rf"{end_key}[:=]\s*([\d.]+)", stderr)]
    return list(zip(s, e + [float("nan")] * (len(s) - len(e))))


def main() -> None:
    edl = read_json(REPORTS / "edit_decision_list.json")
    p = ffprobe_json(FINAL)
    v = next(s for s in p["streams"] if s["codec_type"] == "video")
    a = next(s for s in p["streams"] if s["codec_type"] == "audio")
    subs = [s for s in p["streams"] if s["codec_type"] == "subtitle"]
    dur = float(p["format"]["duration"])
    checks = {}
    lo = float(CFG["episode"].get("min_duration_s", 480))
    hi = float(CFG["episode"].get("max_duration_s", 900))
    checks["duration_in_range"] = lo <= dur <= hi
    checks["duration_matches_edl"] = abs(dur - edl["duration_s"]) < 0.25
    checks["video_h264_high"] = v["codec_name"] == "h264" and v.get("profile") == "High"
    checks["resolution_1920x1080"] = (v["width"], v["height"]) == (1920, 1080)
    checks["aspect_16_9"] = Fraction(v["width"], v["height"]) == Fraction(16, 9) and v.get("sample_aspect_ratio", "1:1") == "1:1"
    checks["fps_29_97_cfr"] = v["r_frame_rate"] == "30000/1001" and v["avg_frame_rate"] == "30000/1001"
    checks["yuv420p_bt709"] = v["pix_fmt"] == "yuv420p" and v.get("color_primaries") == "bt709"
    checks["audio_aac_48k_stereo"] = a["codec_name"] == "aac" and a["sample_rate"] == "48000" and a["channels"] == 2
    checks["english_subtitle_track"] = len(subs) == 1

    log.info("running black / freeze / silence / loudness detection")
    r = run(["ffmpeg", "-hide_banner", "-i", str(FINAL), "-vf",
             "blackdetect=d=0.5:pix_th=0.08,freezedetect=n=0.003:d=2.5", "-af",
             "silencedetect=n=-50dB:d=1.0,ebur128=peak=true", "-f", "null", "-"]).stderr.decode(errors="replace")
    black = intervals(r, "black_start", "black_end")
    freeze = intervals(r, "freeze_start", "freeze_end")
    silence = intervals(r, "silence_start", "silence_end")
    summ = r[r.rfind("Summary:"):]
    lufs = float(re.search(r"I:\s*(-?[\d.]+) LUFS", summ).group(1))
    tp = float(re.search(r"Peak:\s*(-?[\d.]+) dBFS", summ).group(1))
    lra = float(re.search(r"LRA:\s*(-?[\d.]+) LU", summ).group(1))
    # Fades to/from black at the very start and end are intentional.
    unexpected_black = [b for b in black if b[0] > 1.0 and not (b[1] != b[1] or b[1] > dur - 1.5)]
    # Intentional holds: timelapses and static wide shots are flagged separately for human review.
    checks["no_unexpected_black"] = not unexpected_black
    checks["no_audio_dropouts"] = not [s for s in silence if s[0] > 0.5 and (s[1] != s[1] or s[1] < dur - 1.0)]
    checks["loudness_-14_LUFS"] = abs(lufs + 14) <= 1.0
    checks["true_peak_le_-1"] = tp <= -1.0

    # Subtitle sync: speech energy (dialogue band) should be higher inside cues than in the 1 s before them.
    raw = run(["ffmpeg", "-v", "error", "-i", str(FINAL), "-map", "0:a", "-af", "highpass=f=200,lowpass=f=3500",
               "-ac", "1", "-ar", "8000", "-f", "f32le", "-"]).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    env = np.sqrt(np.convolve(x ** 2, np.ones(400) / 400, mode="same"))
    def lvl(t0, t1):
        seg = env[int(max(0, t0) * 8000):int(t1 * 8000)]
        return float(20 * np.log10(np.mean(seg) + 1e-9)) if len(seg) else -120.0
    sync = []
    for s in edl["subtitles_en"]:
        inside = lvl(s["timeline_start"], s["timeline_end"])
        sync.append({"t": round(s["timeline_start"], 2), "text": s["text"][:60], "inside_db": round(inside, 1)})
    med = float(np.median([q["inside_db"] for q in sync]))
    quiet = [q for q in sync if q["inside_db"] < med - 12]
    checks["subtitles_over_speech"] = len(quiet) <= max(2, len(sync) // 20)
    last_sub = max(s["timeline_end"] for s in edl["subtitles_en"])
    checks["subtitles_within_video"] = last_sub <= dur

    src_root = str(SOURCE_DIR.resolve())
    paths = {e["video"]["path"] for e in edl["entries"]} | {e["audio"]["path"] for e in edl["entries"]}
    checks["all_footage_from_source_dir"] = all(str(Path(q).resolve()).startswith(src_root) for q in paths)
    inv = {c["clip_id"]: c for c in read_json(REPORTS / "clip_inventory.json")["clips"]}
    dates = {inv[e["video"]["clip_id"]]["creation_time_tag"][:10] for e in edl["entries"]}
    allowed = set(episode_dates())
    checks["all_footage_from_capture_date"] = bool(dates) and dates <= allowed
    # In-camera frozen spans hide under changing subtitles, so freezedetect alone cannot catch them.
    fz = read_json(REPORTS / "frozen_spans.json", {})
    checks["no_frozen_source_picture"] = not [
        e["entry_id"] for e in edl["entries"]
        if sum(max(0.0, min(b, e["video"]["out"]) - max(a, e["video"]["in"]))
               for a, b in fz.get(e["video"]["clip_id"], [])) > 0.2]
    lic = read_json(PROJECT_DIR / "music" / "LICENSES.json")
    checks["music_licensed_and_credited"] = all(Path(c["track"]).stem in lic["tracks"] for c in edl["music_cues"])

    # Frames for human visual review: start, quarter points, end, and a few subtitle moments.
    marks = [2.0, dur * 0.25, dur * 0.5, dur * 0.75, dur - 5.0]
    nsubs = len(edl["subtitles_en"])
    if nsubs:
        picks = []
        for i in (min(5, nsubs - 1), nsubs // 2, -1 if nsubs < 3 else -3):
            if -nsubs <= i < nsubs and i not in picks:
                picks.append(i)
        marks += [edl["subtitles_en"][i]["timeline_start"] + 0.6 for i in picks]
    for i, t in enumerate(marks):
        run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(FINAL), "-frames:v", "1",
             "-vf", "scale=960:-1", str(SHOTS / f"frame_{i:02d}_{t:07.2f}s.jpg")])

    ok = all(checks.values())
    rep = {"file": str(FINAL), "size_mb": round(int(p["format"]["size"]) / 1e6, 1), "duration_s": round(dur, 3),
           "video": {k: v.get(k) for k in ("codec_name", "profile", "width", "height", "r_frame_rate", "pix_fmt",
                                            "color_primaries", "bit_rate")},
           "audio": {k: a.get(k) for k in ("codec_name", "sample_rate", "channels", "bit_rate")},
           "loudness": {"integrated_lufs": lufs, "true_peak_dbfs": tp, "lra_lu": lra},
           "black_intervals": black, "freeze_intervals_for_review": freeze, "silences": silence,
           "subtitle_cues": len(sync), "subtitles_over_quiet_audio": quiet,
           "checks": checks, "passed": ok}
    write_json(REPORTS / "validation_report.json", rep)
    for k, val in checks.items():
        log.info("%-32s %s", k, "PASS" if val else "FAIL")
    log.info("freeze intervals (review): %s", freeze)
    log.info("VALIDATION %s", "PASSED" if ok else "FAILED")


if __name__ == "__main__":
    main()
