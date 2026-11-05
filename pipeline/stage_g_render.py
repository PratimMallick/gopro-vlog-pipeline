"""Stage G/I — render segments, mix audio, assemble and mux the final MP4.

Resumable: every segment / audio group is cached under work/ by a content hash, so a
failed or interrupted run continues where it stopped. Sources are only ever read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf

from common import CFG, PROJECT_DIR, REPORTS, WORK, assert_not_source, get_logger, read_json, run, write_json
from overlays import overlay_png

log = get_logger("stage_g")
R = CFG["render"]
LOOK = CFG["look"]
FPS = Fraction(R["fps"])
SR = R["audio_rate"]
SEG = WORK / "segments"
AUD = WORK / "audio_groups"
SEG.mkdir(parents=True, exist_ok=True)
AUD.mkdir(parents=True, exist_ok=True)
FINAL = PROJECT_DIR / CFG["paths"]["final_name"]
RENDER_VERSION = 4


def h(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


# ---------------------------------------------------------------- video segments
def video_filters(e: dict) -> tuple[str, list[str]]:
    v = e["video"]
    W, H = R["width"], R["height"]
    chain = []
    if abs(v["width"] / v["height"] - 16 / 9) > 0.002:   # 4:3 masters and 848x480 proxies -> centre 16:9
        chain.append("crop=iw:trunc(iw*9/16/2)*2")
    if e.get("stabilize"):
        chain.append("deshake")
    chain.append(f"scale={W}:{H}:flags=lanczos:out_range=tv")
    if v["source_kind"] == "proxy_only":                   # low-bitrate 480p: tame blocks, restore edges
        chain.append("hqdn3d=2:1.5:4:3,unsharp=5:5:0.55:5:5:0")
    sp = e["speed"]
    chain.append(f"setpts=(PTS-STARTPTS)/{sp}" if sp != 1 else "setpts=PTS-STARTPTS")
    chain.append(f"fps={R['fps']}")
    g = {"contrast": LOOK["contrast"], "saturation": LOOK["saturation"], "gamma": LOOK["gamma"],
         "brightness": 0.0, **e.get("grade", {})}
    chain.append(f"eq=contrast={g['contrast']}:saturation={g['saturation']}:gamma={g['gamma']}"
                 f":brightness={g['brightness']}")
    chain.append("format=yuv420p")
    dur = float(e["frames"] / FPS)
    graph = [f"[0:v]{','.join(chain)}[b0]"]
    last = "b0"
    pngs = []
    k = 1
    for o in e["overlays"]:
        a = max(0.0, o["timeline_start"] - e["timeline_start"])
        b = min(dur, o["timeline_end"] - e["timeline_start"])
        if b - a < 0.05:
            continue
        fd = 0.12 if o["kind"] == "subtitle" else 0.45
        fx = []
        if o["timeline_start"] >= e["timeline_start"] - 1e-3:
            fx.append(f"fade=t=in:st={a:.3f}:d={fd}:alpha=1")
        if o["timeline_end"] <= e["timeline_end"] + 1e-3:
            fx.append(f"fade=t=out:st={max(a, b - fd):.3f}:d={fd}:alpha=1")
        keep = ("kind", "text", "detail", "translated", "here", "phase",
                "style", "basemap", "points", "pad", "leg", "caption")
        spec = {kk: o[kk] for kk in keep if kk in o and o[kk] is not None}
        pngs.append(str(overlay_png(spec)))
        graph.append(f"[{k}:v]format=rgba{',' + ','.join(fx) if fx else ''}[o{k}]")
        graph.append(f"[{last}][o{k}]overlay=0:0:enable='between(t,{a:.3f},{b:.3f})':format=auto[b{k}]")
        last = f"b{k}"
        k += 1
    post = []
    if e.get("fade_in"):
        post.append(f"fade=t=in:st=0:d={e['fade_in']}")
    if e.get("fade_out"):
        post.append(f"fade=t=out:st={dur - e['fade_out']:.3f}:d={e['fade_out']}")
    post += ["tpad=stop_mode=clone:stop_duration=1", "setsar=1"]
    graph.append(f"[{last}]{','.join(post)}[vout]")
    return ";".join(graph), pngs


def render_segment(e: dict) -> Path:
    key = h({"v": RENDER_VERSION, "video": e["video"], "frames": e["frames"], "speed": e["speed"],
             "overlays": e["overlays"], "grade": e.get("grade"), "fi": e.get("fade_in"), "fo": e.get("fade_out"),
             "stab": e.get("stabilize"), "R": R, "LOOK": LOOK})
    out = SEG / f"{e['entry_id']}_{key}.mp4"
    if out.exists():
        return out
    same = [p for p in SEG.glob(f"*_{key}.mp4") if ".part" not in p.name]
    if same:   # identical content rendered under an earlier entry number
        shutil.copyfile(same[0], out)
        return out
    graph, pngs = video_filters(e)
    v = e["video"]
    src_dur = (v["out"] - v["in"]) + 0.5
    dur = float(e["frames"] / FPS)
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if v["codec"] == "hevc":
        cmd += ["-hwaccel", "videotoolbox"]
    cmd += ["-ss", f"{v['in']:.4f}", "-t", f"{src_dur:.4f}", "-i", v["path"]]
    for p in pngs:
        cmd += ["-loop", "1", "-framerate", R["fps"], "-t", f"{dur + 1:.3f}", "-i", p]
    tmp = out.with_suffix(".part.mp4")
    assert_not_source(tmp)
    cmd += ["-filter_complex", graph, "-map", "[vout]", "-frames:v", str(e["frames"]), "-an",
            "-c:v", "libx264", "-preset", R["preset"], "-crf", str(R["crf"]), "-profile:v", R["x264_profile"],
            "-pix_fmt", "yuv420p", "-r", R["fps"], "-g", "60", "-bf", "2", "-threads", "4",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            "-video_track_timescale", "30000", str(tmp)]
    run(cmd)
    tmp.replace(out)
    return out


# ---------------------------------------------------------------- audio
def audio_groups(entries: list[dict]) -> list[dict]:
    """Contiguous pieces of the same take (split only by B-roll) are extracted as one block."""
    groups = []
    for e in entries:
        a = e["audio"]
        g = groups[-1] if groups else None
        if (g and g["clip_id"] == a["clip_id"] and abs(g["out"] - a["in"]) < 1e-3 and g["speed"] == e["speed"]
                and g["mode"] == e["mode"]):
            g["out"] = a["out"]
            g["frames"] += e["frames"]
            g["entries"].append(e["entry_id"])
            g["fade_out"] = e.get("fade_out", 0)
        else:
            groups.append({"clip_id": a["clip_id"], "path": a["path"], "has_audio": a["has_audio"], "in": a["in"],
                           "out": a["out"], "speed": e["speed"], "mode": e["mode"], "frames": e["frames"],
                           "timeline_start": e["timeline_start"], "entries": [e["entry_id"]],
                           "nat_gain_db": e.get("nat_gain_db", 0.0), "fade_in": e.get("fade_in", 0),
                           "fade_out": e.get("fade_out", 0)})
    return groups


def extract_group(g: dict) -> np.ndarray:
    n = round(g["frames"] / FPS * SR)
    if not g["has_audio"] or g["speed"] != 1.0:
        return np.zeros((n, 2), dtype=np.float32)
    key = h({"v": RENDER_VERSION, "p": g["path"], "in": g["in"], "n": n, "mode": g["mode"]})
    out = AUD / f"{g['clip_id']}_{key}.wav"
    if not out.exists():
        if g["mode"] == "dialogue":   # wind-rumble high-pass + gentle broadband denoise
            af = "highpass=f=90:p=2,afftdn=nr=10:nf=-42:tn=1,lowpass=f=15000"
        else:                          # ambience: only remove sub-bass wind thump
            af = "highpass=f=45"
        run(["ffmpeg", "-v", "error", "-y", "-ss", f"{g['in']:.4f}", "-i", g["path"], "-vn",
             "-af", f"{af},aresample={SR},apad", "-ac", "2", "-ar", str(SR), "-t", f"{n / SR:.6f}",
             "-c:a", "pcm_f32le", str(out)])
    x, _ = sf.read(out, dtype="float32", always_2d=True)
    if len(x) < n:
        x = np.pad(x, ((0, n - len(x)), (0, 0)))
    return x[:n]


def db(x: float) -> float:
    return 10 ** (x / 20)


def active_rms(x: np.ndarray, top: float = 0.4) -> float:
    mono = x.mean(axis=1)
    hop = SR // 20
    if len(mono) < hop:
        return 1e-6
    fr = np.sqrt(np.mean(mono[: len(mono) // hop * hop].reshape(-1, hop) ** 2, axis=1))
    fr = np.sort(fr)[::-1]
    return float(np.mean(fr[: max(1, int(len(fr) * top))]) + 1e-9)


def decode_music(path: str, offset: float, dur: float) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{offset:.3f}", "-t", f"{dur + 0.1:.3f}", "-i", path,
                          "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()


def mix(edl: dict) -> Path:
    total_n = round(sum(e["frames"] for e in edl["entries"]) / FPS * SR)
    key = h({"v": RENDER_VERSION, "e": [(e["audio"], e["mode"], e["frames"], e.get("nat_gain_db"),
                                         e.get("fade_in"), e.get("fade_out")) for e in edl["entries"]],
             "m": edl["music_cues"], "s": [(s["timeline_start"], s["timeline_end"]) for s in edl["subtitles_en"]],
             "R": R})
    out = WORK / f"mix_{key}.wav"
    if out.exists():
        return out
    nat = np.zeros((total_n, 2), dtype=np.float32)
    report = []
    for g in audio_groups(edl["entries"]):
        x = extract_group(g)
        if g["mode"] == "dialogue":
            gain = float(np.clip(-20.0 - 20 * np.log10(active_rms(x)), -6, 15))
        else:
            gain = float(np.clip(-25.0 - 20 * np.log10(active_rms(x, 0.8)), -10, 9))
            if g["mode"] == "music":
                gain -= 14
        gain += g["nat_gain_db"]
        x = x * db(gain)
        ramp = int(0.012 * SR)
        fi, fo = max(ramp, int(g["fade_in"] * SR)), max(ramp, int(g["fade_out"] * SR))
        x[:fi] *= np.linspace(0, 1, fi)[:, None]
        x[-fo:] *= np.linspace(1, 0, fo)[:, None]
        s = round(g["timeline_start"] * SR)
        nat[s:s + len(x)] += x[: total_n - s]
        report.append({"clip": g["clip_id"], "mode": g["mode"], "timeline_start": g["timeline_start"],
                       "gain_db": round(gain, 1)})

    # Music level automation (100 Hz control rate) from entry modes plus speech (subtitle) intervals.
    cr = 100
    n_ctl = int(total_n / SR * cr) + 2
    level = np.full(n_ctl, R["music_db_ambient"], dtype=np.float64)
    for e in edl["entries"]:
        a, b = int(e["timeline_start"] * cr), int(e["timeline_end"] * cr)
        level[a:b] = {"music": R["music_db_open"], "ambient": R["music_db_ambient"],
                      "dialogue": R["music_db_under_speech"] + 3}[e["mode"]]
    for s in edl["subtitles_en"]:
        a, b = max(0, int((s["timeline_start"] - 0.35) * cr)), int((s["timeline_end"] + 0.5) * cr)
        level[a:b] = np.minimum(level[a:b], R["music_db_under_speech"])
    k = int(0.5 * cr)
    level = np.convolve(np.pad(level, (k, k), mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]
    env = np.interp(np.arange(total_n) / SR * cr, np.arange(len(level)), level).astype(np.float32)

    bed = np.zeros((total_n, 2), dtype=np.float32)
    for i, c in enumerate(edl["music_cues"]):
        dur = c["timeline_end"] - c["timeline_start"]
        y = decode_music(c["path"], c["track_offset"], dur)
        y *= db(-18.0 - 20 * np.log10(active_rms(y, 0.8)))   # level-match the tracks
        s = round(c["timeline_start"] * SR)
        y = y[: max(0, min(len(y), total_n - s))]
        fin = int((0.25 if i == 0 else 2.5) * SR)
        fout = int((0.6 if c["align"] == "end" else 2.5) * SR)
        fin, fout = min(fin, len(y) // 2), min(fout, len(y) // 2)
        y[:fin] *= (np.sin(np.linspace(0, np.pi / 2, fin)) ** 2)[:, None]
        y[len(y) - fout:] *= (np.cos(np.linspace(0, np.pi / 2, fout)) ** 2)[:, None]
        bed[s:s + len(y)] += y
    bed *= (10 ** (env / 20))[:, None]
    full = nat + bed
    pre = WORK / "mix_prenorm.wav"
    sf.write(pre, full, SR, subtype="FLOAT")
    write_json(REPORTS / "audio_mix_report.json", {"groups": report, "peak_prenorm": float(np.abs(full).max())})

    # Two-pass EBU R128 loudness normalisation behind a peak limiter.
    lim = f"alimiter=limit={db(R['true_peak'] - 0.5):.3f}:attack=5:release=60:level=disabled"
    res = run(["ffmpeg", "-hide_banner", "-i", str(pre), "-af",
               f"{lim},loudnorm=I={R['target_lufs']}:TP={R['true_peak']}:LRA=11:print_format=json",
               "-f", "null", "-"])
    m = json.loads(re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", res.stderr.decode(), re.S)[-1])
    run(["ffmpeg", "-v", "error", "-y", "-i", str(pre), "-af",
         f"{lim},loudnorm=I={R['target_lufs']}:TP={R['true_peak']}:LRA=11:measured_I={m['input_i']}"
         f":measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
         f":offset={m['target_offset']}:linear=true,aresample={SR}", "-c:a", "pcm_f32le", str(out)])
    write_json(REPORTS / "loudness_pass1.json", m)
    return out


# ---------------------------------------------------------------- subtitles
def srt_time(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    hh, ms = divmod(ms, 3600000)
    mm, ms = divmod(ms, 60000)
    ss, ms = divmod(ms, 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def write_srts(edl: dict) -> tuple[Path, Path]:
    base = CFG["episode"].get("srt_basename", "subtitles")
    en = PROJECT_DIR / f"{base}_English.srt"
    blocks = []
    for i, s in enumerate(edl["subtitles_en"], 1):
        txt = f"<i>{s['text']}</i>" if s["translated"] else s["text"]
        blocks.append(f"{i}\n{srt_time(s['timeline_start'])} --> {srt_time(s['timeline_end'])}\n{txt}\n")
    en.write_text("\n".join(blocks), encoding="utf-8")
    hi = PROJECT_DIR / f"{base}_Hindi_DRAFT_unreviewed.srt"
    blocks = []
    for s in edl["subtitles_hi_draft"]:
        if s["avg_logprob"] < -0.75:
            continue
        blocks.append(f"{len(blocks) + 1}\n{srt_time(s['timeline_start'])} --> {srt_time(s['timeline_end'])}\n"
                      f"{s['text']}\n")
    hi.write_text("\n".join(blocks), encoding="utf-8")
    return en, hi


# ---------------------------------------------------------------- assemble
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated entry ids to render (preview)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    edl = read_json(REPORTS / "edit_decision_list.json")
    entries = edl["entries"]
    todo = [e for e in entries if not args.only or e["entry_id"] in args.only.split(",")]
    t0 = time.time()
    seg_paths: dict[str, Path] = {}
    tot_s = float(sum(e["frames"] for e in todo) / FPS)
    done_s = 0.0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(render_segment, e): e for e in todo}
        for f in as_completed(futs):
            e = futs[f]
            try:
                seg_paths[e["entry_id"]] = f.result()
            except Exception as exc:
                log.error("segment %s failed: %s", e["entry_id"], exc)
                raise
            done_s += float(e["frames"] / FPS)
            el = time.time() - t0
            log.info("segment %s ok (%.0f/%.0f s timeline, %.0fs elapsed, ~%.0fs left)", e["entry_id"],
                     done_s, tot_s, el, el / max(done_s, 1e-6) * (tot_s - done_s))
    if args.only:
        return
    for e in entries:
        nb = run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                  "stream=nb_read_packets", "-of", "csv=p=0", str(seg_paths[e["entry_id"]])]).stdout.decode().strip()
        if int(nb) != e["frames"]:
            raise RuntimeError(f"{e['entry_id']}: {nb} frames, expected {e['frames']}")
    log.info("all %d segments verified frame-exact; mixing audio", len(entries))
    mixed = mix(edl)
    en_srt, _ = write_srts(edl)
    lst = WORK / "concat_list.txt"
    lst.write_text("".join("file '%s'\n" % str(seg_paths[e["entry_id"]]).replace("'", "'\\''") for e in entries),
                   encoding="utf-8")
    tmp = WORK / "final.part.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(mixed),
         "-i", str(en_srt), "-map", "0:v", "-map", "1:a", "-map", "2:s", "-c:v", "copy",
         "-c:a", "aac", "-b:a", "256k", "-ar", str(SR), "-ac", "2", "-c:s", "mov_text",
         "-metadata:s:s:0", "language=eng", "-metadata:s:s:0", "title=English (italic = translated from Hindi)",
         "-metadata:s:a:0", "language=hin",
         "-metadata", "title=" + CFG["episode"].get("video_title", CFG["episode"]["title"]),
         "-metadata", "comment=Music by Kevin MacLeod (incompetech.com), CC BY 4.0",
         "-movflags", "+faststart", str(tmp)])
    assert_not_source(FINAL)
    tmp.replace(FINAL)
    shutil.copy(REPORTS / "edit_decision_list.json", PROJECT_DIR / "edit_decision_list.json")
    log.info("final video written: %s (%.0fs)", FINAL, time.time() - t0)


if __name__ == "__main__":
    main()
