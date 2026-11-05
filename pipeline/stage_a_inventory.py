"""Stage A — media inventory of the Beach Trek folder (read-only on sources)."""
from __future__ import annotations

import csv
import re
import statistics
from datetime import datetime
from pathlib import Path

from common import CFG, REPORTS, SOURCE_DIR, WORK, episode_dates, ffprobe_json, get_logger, read_json, write_json
from gpmf import read_gps

log = get_logger("stage_a")
NAME_RE = re.compile(r"^G([HXL]|OPR)(\d{2})?(\d{4})\.(MP4|LRV|THM|JPG)$", re.I)
GPS_DIR = WORK / "gps"
GPS_DIR.mkdir(exist_ok=True)


def probe_video(path: Path) -> dict:
    try:
        meta = ffprobe_json(path)
    except Exception as exc:  # corrupted / unsupported
        return {"readable": False, "error": str(exc)[:300]}
    v = next((s for s in meta["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in meta["streams"] if s["codec_type"] == "audio"), None)
    fmt = meta.get("format", {})
    tags = fmt.get("tags", {})
    info = {"readable": v is not None, "duration_s": float(fmt.get("duration", 0) or 0),
            "creation_time_tag": tags.get("creation_time"), "firmware": tags.get("firmware"),
            "has_audio": a is not None}
    if v:
        num, den = (v.get("avg_frame_rate") or "0/1").split("/")
        rfr = v.get("r_frame_rate", "0/1")
        rotation = 0
        for sd in v.get("side_data_list", []) or []:
            if "rotation" in sd:
                rotation = int(sd["rotation"])
        rotation = int(v.get("tags", {}).get("rotate", rotation))
        info.update({"width": v["width"], "height": v["height"], "video_codec": v["codec_name"],
                     "r_frame_rate": rfr,
                     "avg_fps": round(float(num) / float(den), 3) if float(den) else None,
                     "pix_fmt": v.get("pix_fmt"), "color_transfer": v.get("color_transfer"),
                     "bit_rate": int(v.get("bit_rate", 0) or 0), "rotation": rotation,
                     "nb_frames": int(v.get("nb_frames", 0) or 0)})
    if a:
        info.update({"audio_codec": a["codec_name"], "audio_rate": int(a.get("sample_rate", 0)),
                     "audio_channels": a.get("channels")})
    return info


def main() -> None:
    if CFG["paths"].get("top_level_only"):
        files = sorted(p for p in SOURCE_DIR.iterdir() if p.is_file() and not p.name.startswith("."))
    else:
        files = sorted(p for p in SOURCE_DIR.rglob("*") if p.is_file() and not p.name.startswith("."))
    log.info("Scanning %d files under %s", len(files), SOURCE_DIR)
    groups: dict[str, dict] = {}
    photos, others = [], []
    for p in files:
        m = NAME_RE.match(p.name)
        if not m:
            others.append(str(p))
            continue
        kind, chapter, num, ext = m.group(1).upper(), m.group(2) or "01", m.group(3), m.group(4).upper()
        if ext == "JPG":
            photos.append({"path": str(p), "file": p.name, "size_bytes": p.stat().st_size})
            continue
        cid = f"C{num}_{chapter}"
        g = groups.setdefault(cid, {"clip_id": cid, "file_number": num, "chapter": chapter})
        if ext == "MP4":
            g["master_path"] = str(p)
            g["encoding"] = {"H": "H.264 (GH)", "X": "HEVC (GX)"}.get(kind, kind)
        elif ext == "LRV":
            g["proxy_path"] = str(p)
        elif ext == "THM":
            g["thumb_path"] = str(p)

    cache = read_json(REPORTS / "clip_inventory.json", {}) or {}
    cached = {c["clip_id"]: c for c in cache.get("clips", [])}
    clips = []
    for i, (cid, g) in enumerate(sorted(groups.items()), 1):
        c = cached.get(cid)
        if c and c.get("master_path") == g.get("master_path") and c.get("proxy_path") == g.get("proxy_path"):
            clips.append(c)
            continue
        log.info("[%d/%d] probing %s", i, len(groups), cid)
        g["master"] = probe_video(Path(g["master_path"])) if "master_path" in g else None
        g["proxy"] = probe_video(Path(g["proxy_path"])) if "proxy_path" in g else None
        if g["master"] and g["master"].get("readable"):
            g["source_kind"], g["source_path"], src = "master", g["master_path"], g["master"]
        elif g["proxy"] and g["proxy"].get("readable"):
            g["source_kind"], g["source_path"], src = "proxy_only", g["proxy_path"], g["proxy"]
        else:
            g["source_kind"], g["source_path"], src = "unusable", None, (g["master"] or g["proxy"] or {})
        for k in ("duration_s", "width", "height", "avg_fps", "r_frame_rate", "video_codec",
                  "has_audio", "audio_codec", "rotation", "creation_time_tag", "firmware",
                  "color_transfer"):
            g[k] = src.get(k)
        p = Path(g["source_path"] or g.get("master_path") or g.get("proxy_path"))
        g["file_size_bytes"] = p.stat().st_size
        g["filename"] = p.name

        gps = read_gps(p) if g["source_path"] else {"has_gpmf": False, "points": []}
        write_json(GPS_DIR / f"{cid}.json", gps)
        good = [pt for pt in gps["points"] if (pt["fix"] or 0) >= 2 and (pt["dop"] or 9999) < 1000
                and abs(pt["lat"]) > 0.1]
        g["gpmf_device"] = gps.get("device")
        g["gps_fix_points"] = len(good)
        if good:
            g["gps_lat_median"] = round(statistics.median(pt["lat"] for pt in good), 6)
            g["gps_lon_median"] = round(statistics.median(pt["lon"] for pt in good), 6)
            g["gps_utc_first"] = good[0]["utc"]
        notes = []
        if g["source_kind"] == "proxy_only":
            notes.append("full-resolution master missing; only 848x480/640x480 LRV proxy available")
        if g["source_kind"] != "unusable" and (g["duration_s"] or 0) < 1.0:
            notes.append("too short to use (<1 s)")
            g["source_kind"] = "unusable"
        if g.get("has_audio") is False:
            notes.append("no audio stream (TimeLapse/TimeWarp mode likely)")
        if g.get("width") and g.get("height") and abs(g["width"] / g["height"] - 16 / 9) > 0.05:
            notes.append(f"non-16:9 frame ({g['width']}x{g['height']}); needs crop")
        g["notes"] = notes
        clips.append(g)

    # Capture time: creation_time tag is camera-local clock written as 'Z' (GoPro has no TZ).
    for c in clips:
        t = c.get("creation_time_tag")
        c["capture_local"] = t[:19].replace("T", " ") if t else None
        allowed = episode_dates()
        c["episode_date_ok"] = bool(t and any(t.startswith(d) for d in allowed))
    clips.sort(key=lambda c: (c["capture_local"] or "9999", c["clip_id"]))
    for order, c in enumerate(clips):
        c["capture_order"] = order

    write_json(REPORTS / "clip_inventory.json", {"generated": datetime.now().isoformat(timespec="seconds"),
                                                 "source_dir": str(SOURCE_DIR), "clips": clips,
                                                 "photos": photos, "other_files": others})
    cols = ["clip_id", "capture_order", "capture_local", "source_kind", "filename", "source_path",
            "file_size_bytes", "duration_s", "width", "height", "avg_fps", "video_codec", "has_audio",
            "audio_codec", "rotation", "color_transfer", "firmware", "gpmf_device", "gps_fix_points",
            "gps_lat_median", "gps_lon_median", "master_path", "proxy_path", "notes"]
    with open(REPORTS / "clip_inventory.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for c in clips:
            w.writerow({**c, "notes": "; ".join(c["notes"])})

    usable = [c for c in clips if c["source_kind"] != "unusable"]
    tot = sum(c["duration_s"] or 0 for c in usable)
    lines = [f"Media summary — {SOURCE_DIR}", "",
             f"Files scanned: {len(files)}",
             f"Clip groups: {len(clips)}  (usable: {len(usable)})",
             f"  masters available: {sum(c['source_kind'] == 'master' for c in clips)}",
             f"  proxy-only (master missing): {sum(c['source_kind'] == 'proxy_only' for c in clips)}",
             f"  unusable: {sum(c['source_kind'] == 'unusable' for c in clips)}",
             f"Photos (JPG, not used as video): {len(photos)}",
             f"Other files: {len(others)}",
             f"Total usable footage: {tot / 60:.1f} min",
             f"Capture window: {usable[0]['capture_local']} -> {usable[-1]['capture_local']}" if usable else "",
             f"Clips on {', '.join(episode_dates())}: {sum(c['episode_date_ok'] for c in clips)}",
             f"Clips with GPS fix: {sum(1 for c in clips if c.get('gps_fix_points'))}", "",
             "Formats:"]
    fmts: dict[str, list] = {}
    for c in usable:
        fmts.setdefault(f"{c['width']}x{c['height']} @ {c['avg_fps']} {c['video_codec']} ({c['source_kind']})",
                        []).append(c["duration_s"])
    for k, v in sorted(fmts.items()):
        lines.append(f"  {k}: {len(v)} clips, {sum(v) / 60:.1f} min")
    lines += ["", "Notes per clip:"]
    lines += [f"  {c['clip_id']} {c['capture_local']} {c['filename']}: {'; '.join(c['notes'])}"
              for c in clips if c["notes"]]
    (REPORTS / "media_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("\n".join(lines[:14]))


if __name__ == "__main__":
    main()
