"""Stage B — visual indexing: 1 fps frames, CV metrics, segments, contact sheets.

Frames are streamed from ffmpeg one clip at a time; nothing is loaded whole.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import CFG, REPORTS, REVIEW, WORK, fmt_tc, get_logger, load_inventory, read_json, run, write_json

log = get_logger("stage_b")
A = CFG["analysis"]
FRAMES = WORK / "frames"
SHEETS = REVIEW / "contact_sheets"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def extract_frames(clip: dict) -> list[Path]:
    out = FRAMES / clip["clip_id"]
    done = out / ".done"
    if done.exists():
        return sorted(out.glob("f_*.jpg"))
    out.mkdir(parents=True, exist_ok=True)
    w = 480
    # 4:3 sources are center-cropped to 16:9 so frames match the final framing.
    crop = ""
    if clip["width"] / clip["height"] < 1.6:
        crop = f"crop=iw:iw*9/16,"
    vf = f"fps=1/{A['frame_interval_s']},{crop}scale={w}:-2"
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if clip["video_codec"] == "hevc":
        cmd += ["-hwaccel", "videotoolbox"]
    cmd += ["-i", clip["source_path"], "-vf", vf, "-q:v", "4", str(out / "f_%05d.jpg")]
    run(cmd)
    done.touch()
    return sorted(out.glob("f_*.jpg"))


def frame_metrics(img: np.ndarray, prev_gray: np.ndarray | None) -> dict:
    small = cv2.resize(img, (A["analysis_width"], int(A["analysis_width"] * img.shape[0] / img.shape[1])))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    b, g, r = [small[..., i].astype(np.float32) for i in range(3)]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorful = math.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * math.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    blue = ((h >= 85) & (h <= 125) & (s > 40) & (v > 60)).mean()           # sea / sky
    green = ((h >= 35) & (h < 85) & (s > 50) & (v > 40)).mean()             # vegetation
    sand = ((h >= 8) & (h < 30) & (s > 25) & (s < 140) & (v > 120)).mean()  # sand / laterite
    top_blue = ((h[: h.shape[0] // 3] >= 85) & (h[: h.shape[0] // 3] <= 125)).mean()
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 8], [0, 180, 0, 256]).flatten()
    hist /= hist.sum() + 1e-9
    motion = float(np.abs(gray.astype(np.int16) - prev_gray).mean()) if prev_gray is not None else 0.0
    return {"luma": float(gray.mean()), "contrast": float(gray.std()),
            "sharp": float(cv2.Laplacian(gray, cv2.CV_64F).var()), "colorful": colorful,
            "blue": float(blue), "green": float(green), "sand": float(sand), "top_blue": float(top_blue),
            "motion": motion, "_hist": hist, "_gray": gray.astype(np.int16)}


def score_frame(m: dict) -> float:
    if m["luma"] < A["black_luma"] or m["contrast"] < 12:
        return 0.0
    expo = 1.0 - min(1.0, abs(m["luma"] - 125) / 125)
    sharp = min(1.0, math.log10(m["sharp"] + 1) / 3.2)
    col = min(1.0, m["colorful"] / 70)
    scenic = min(1.0, 1.6 * m["blue"] + 0.6 * m["green"] + 0.8 * m["sand"])
    shake = max(0.0, 1.0 - max(0.0, m["motion"] - 18) / 30)
    return round(100 * (0.22 * expo + 0.22 * sharp + 0.2 * col + 0.26 * scenic + 0.1 * shake) * (0.6 + 0.4 * shake), 2)


def analyse_clip(clip: dict) -> dict:
    cid = clip["clip_id"]
    cache = WORK / "scene" / f"{cid}.json"
    if cache.exists():
        return read_json(cache)
    frames = extract_frames(clip)
    rows, prev, prev_hist = [], None, None
    for i, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        m = frame_metrics(img, prev)
        cut = False
        if prev_hist is not None:
            dist = cv2.compareHist(prev_hist.astype(np.float32), m["_hist"].astype(np.float32),
                                   cv2.HISTCMP_BHATTACHARYYA)
            cut = dist > A["scene_threshold"]
        else:
            dist = 0.0
        prev, prev_hist = m.pop("_gray"), m.pop("_hist")
        t = i * A["frame_interval_s"] + A["frame_interval_s"] / 2
        rows.append({"t": round(min(t, clip["duration_s"]), 2), "frame": fp.name, "hist_dist": round(float(dist), 3),
                     "cut": bool(cut), "black": m["luma"] < A["black_luma"],
                     "score": score_frame(m), **{k: round(v, 4) for k, v in m.items()}})

    segments, cur = [], None
    for r in rows:
        bad = r["black"] or r["score"] < 8
        if cur and (r["cut"] or bad):
            segments.append(cur)
            cur = None
        if bad:
            continue
        if cur is None:
            cur = {"start": max(0.0, r["t"] - A["frame_interval_s"] / 2), "rows": []}
        cur["rows"].append(r)
    if cur:
        segments.append(cur)
    segs = []
    for s in segments:
        end = min(clip["duration_s"], s["rows"][-1]["t"] + A["frame_interval_s"] / 2)
        if end - s["start"] < A["min_segment_s"] and len(segments) > 1:
            continue
        sc = [r["score"] for r in s["rows"]]
        segs.append({"start": round(s["start"], 2), "end": round(end, 2),
                     "mean_score": round(float(np.mean(sc)), 2), "peak_score": round(float(np.max(sc)), 2),
                     "peak_t": s["rows"][int(np.argmax(sc))]["t"],
                     "blue": round(float(np.mean([r["blue"] for r in s["rows"]])), 3),
                     "green": round(float(np.mean([r["green"] for r in s["rows"]])), 3),
                     "motion": round(float(np.mean([r["motion"] for r in s["rows"]])), 2)})
    result = {"clip_id": cid, "source_kind": clip["source_kind"], "duration_s": clip["duration_s"],
              "n_frames": len(rows), "frames": rows, "segments": segs,
              "black_or_obstructed_s": round(sum(A["frame_interval_s"] for r in rows if r["black"]), 1),
              "clip_score": round(float(np.percentile([r["score"] for r in rows], 75)), 2) if rows else 0}
    cache.parent.mkdir(exist_ok=True)
    write_json(cache, result)
    return result


def contact_sheet(clips: list[dict], scenes: dict, path: Path, per_row: int = 6) -> None:
    tw, th = 240, 135
    lab = 18
    font = ImageFont.truetype(FONT, 13)
    big = ImageFont.truetype(FONT, 15)
    sheet = Image.new("RGB", (per_row * tw + 190, len(clips) * (th + lab)), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    for row, c in enumerate(clips):
        sc = scenes[c["clip_id"]]
        y = row * (th + lab)
        d.text((6, y + 6), c["clip_id"], font=big, fill=(255, 220, 90))
        d.text((6, y + 26), c["capture_local"][11:19], font=font, fill=(220, 220, 220))
        d.text((6, y + 44), f"{c['duration_s']:.0f}s {c['filename']}", font=font, fill=(180, 180, 180))
        d.text((6, y + 62), f"{'PROXY' if c['source_kind'] == 'proxy_only' else 'master'} "
                            f"{'' if c['has_audio'] else 'NO AUDIO'}", font=font, fill=(150, 200, 255))
        d.text((6, y + 80), f"score {sc['clip_score']}", font=font, fill=(180, 255, 180))
        fr = sc["frames"]
        if not fr:
            continue
        picks = [fr[min(len(fr) - 1, int((k + 0.5) * len(fr) / per_row))] for k in range(per_row)]
        for k, r in enumerate(picks):
            im = Image.open(FRAMES / c["clip_id"] / r["frame"]).convert("RGB")
            im = im.resize((tw, int(tw * im.height / im.width)))
            if im.height > th:
                im = im.crop((0, (im.height - th) // 2, tw, (im.height - th) // 2 + th))
            x = 190 + k * tw
            sheet.paste(im, (x, y))
            d.text((x + 4, y + th + 1), f"{fmt_tc(r['t'])}  s{r['score']:.0f}", font=font, fill=(230, 230, 230))
    sheet.save(path, quality=85)


def main() -> None:
    clips = [c for c in load_inventory() if c["source_kind"] != "unusable" and c.get("episode_date_ok", True)]
    t0 = time.time()
    scenes: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(analyse_clip, c): c for c in clips}
        for n, f in enumerate(as_completed(futs), 1):
            c = futs[f]
            try:
                scenes[c["clip_id"]] = f.result()
            except Exception as exc:
                log.error("clip %s failed: %s", c["clip_id"], exc)
                continue
            el = time.time() - t0
            log.info("[%d/%d] %s done (%.0fs elapsed, ~%.0fs left)", n, len(clips), c["clip_id"], el,
                     el / n * (len(clips) - n))
    ok = [c for c in clips if c["clip_id"] in scenes]
    SHEETS.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(ok), 10):
        contact_sheet(ok[i:i + 10], scenes, SHEETS / f"overview_{i // 10 + 1:02d}.jpg")

    # Redundancy: segments whose mean colour histogram nearly matches an earlier, higher-scoring segment.
    index = []
    for c in ok:
        for s in scenes[c["clip_id"]]["segments"]:
            index.append({"clip_id": c["clip_id"], "capture_local": c["capture_local"],
                          "source_kind": c["source_kind"], **s})
    for c in ok:
        fr = scenes[c["clip_id"]]["frames"]
        for s in [x for x in index if x["clip_id"] == c["clip_id"]]:
            sel = [r for r in fr if s["start"] <= r["t"] <= s["end"]]
            s["_vec"] = np.array([[r["blue"], r["green"], r["sand"], r["luma"] / 255, r["colorful"] / 100]
                                  for r in sel]).mean(axis=0)
    for i, s in enumerate(index):
        s["redundant_with"] = None
        for j in range(max(0, i - 6), i):
            o = index[j]
            if o["clip_id"] != s["clip_id"] and np.abs(o["_vec"] - s["_vec"]).sum() < 0.05 \
                    and o["mean_score"] >= s["mean_score"]:
                s["redundant_with"] = f"{o['clip_id']}@{o['start']}"
                break
    for s in index:
        s.pop("_vec")
    write_json(REPORTS / "scene_index.json", {"segments": index,
                                              "clips": {k: {kk: v[kk] for kk in ("clip_score", "black_or_obstructed_s",
                                                                                   "n_frames")}
                                                        for k, v in scenes.items()}})
    log.info("Indexed %d segments across %d clips; sheets in %s", len(index), len(ok), SHEETS)


if __name__ == "__main__":
    main()
