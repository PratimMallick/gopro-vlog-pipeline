"""Minimal GoPro GPMF (telemetry) reader for GPS5 / GPSF / GPSU / GPSP.

Spec: https://github.com/gopro/gpmf-parser (KLV: FourCC, type, size, repeat).
"""
from __future__ import annotations

import struct
import subprocess
from pathlib import Path

from common import ffprobe_json

_TYPES = {"b": "b", "B": "B", "s": "h", "S": "H", "l": "i", "L": "I",
          "f": "f", "d": "d", "j": "q", "J": "Q"}


def extract_gpmd(path: Path) -> bytes | None:
    meta = ffprobe_json(path)
    idx = next((s["index"] for s in meta["streams"]
                if s.get("codec_tag_string") == "gpmd"), None)
    if idx is None:
        return None
    res = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", f"0:{idx}",
                          "-c", "copy", "-f", "rawvideo", "-"], capture_output=True)
    return res.stdout if res.returncode == 0 and res.stdout else None


def _walk(buf: bytes, start: int, end: int, out: list, ctx: dict) -> None:
    i = start
    while i + 8 <= end:
        key = buf[i:i + 4].decode("latin-1")
        typ = chr(buf[i + 4])
        size = buf[i + 5]
        rep = struct.unpack(">H", buf[i + 6:i + 8])[0]
        length = size * rep
        data_start = i + 8
        data_end = data_start + length
        if data_end > end or not key.isprintable():
            return
        if buf[i + 4] == 0:
            _walk(buf, data_start, data_end, out, ctx if key != "STRM" else {})
        else:
            raw = buf[data_start:data_end]
            if key == "SCAL" and typ in _TYPES:
                n = size // struct.calcsize(_TYPES[typ])
                vals = struct.unpack(">" + _TYPES[typ] * (n * rep), raw)
                ctx["SCAL"] = list(vals)
            elif key == "GPSF" and typ in _TYPES:
                ctx["GPSF"] = struct.unpack(">" + _TYPES[typ], raw[:4])[0]
            elif key == "GPSP" and typ in _TYPES:
                ctx["GPSP"] = struct.unpack(">" + _TYPES[typ], raw[:2])[0]
            elif key == "GPSU":
                ctx["GPSU"] = raw.decode("latin-1").strip("\x00 ")
            elif key in ("DVNM", "MINF") and typ == "c":
                ctx.setdefault("_dev", raw.decode("latin-1").strip("\x00 "))
                out.append({"device": ctx["_dev"]})
            elif key == "GPS5" and typ == "l":
                scal = ctx.get("SCAL") or [1] * 5
                if len(scal) == 1:
                    scal = scal * 5
                for r in range(rep):
                    vals = struct.unpack(">5i", raw[r * 20:(r + 1) * 20])
                    lat, lon, alt, s2, s3 = (v / sc for v, sc in zip(vals, scal))
                    out.append({"lat": lat, "lon": lon, "alt": alt, "speed2d": s2,
                                "fix": ctx.get("GPSF"), "dop": ctx.get("GPSP"),
                                "utc": ctx.get("GPSU"), "rep": r, "n": rep})
        i = data_start + ((length + 3) & ~3)


def read_gps(path: Path) -> dict:
    raw = extract_gpmd(path)
    if not raw:
        return {"has_gpmf": False, "points": [], "device": None}
    out: list = []
    _walk(raw, 0, len(raw), out, {})
    device = next((o["device"] for o in out if "device" in o), None)
    pts = [o for o in out if "lat" in o]
    # One sample per GPMF payload (~1 Hz) is enough for location work.
    per_sec = [p for p in pts if p["rep"] == 0]
    good = [p for p in per_sec if (p["fix"] or 0) >= 2 and (p["dop"] or 9999) < 1000
            and abs(p["lat"]) > 0.1]
    return {"has_gpmf": True, "device": device, "n_samples": len(pts),
            "points": [{k: p[k] for k in ("lat", "lon", "alt", "speed2d", "fix", "dop", "utc")}
                       for p in per_sec],
            "good_fix_points": len(good)}
