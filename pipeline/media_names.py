"""Clip ids for GoPro chapter files and for any other video file."""
from __future__ import annotations

import re
from pathlib import Path

GOPRO_RE = re.compile(r"^G([HXL]|OPR)(\d{2})?(\d{4})\.(MP4|LRV|THM|JPG)$", re.I)
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv"}


def classify_name(name: str) -> dict | None:
    """Return a photo marker, a clip role, or None when the file is not media."""
    match = GOPRO_RE.match(name)
    if match:
        kind, chapter, num, ext = (match.group(1).upper(), match.group(2) or "01",
                                   match.group(3), match.group(4).upper())
        if ext == "JPG":
            return {"kind": "photo"}
        role = {"MP4": "master", "LRV": "proxy", "THM": "thumb"}[ext]
        out = {"kind": "clip", "clip_id": f"C{num}_{chapter}", "role": role,
               "file_number": num, "chapter": chapter}
        if role == "master":
            out["encoding"] = {"H": "H.264 (GH)", "X": "HEVC (GX)"}.get(kind, kind)
        return out
    ext = Path(name).suffix.lower()
    if ext in VIDEO_EXT:
        stem = re.sub(r"[^A-Za-z0-9]+", "_", Path(name).stem).strip("_")[:48] or "clip"
        return {"kind": "clip", "clip_id": f"V_{stem}", "role": "master",
                "file_number": None, "chapter": None, "encoding": ext[1:].upper()}
    if ext in {".jpg", ".jpeg"}:
        return {"kind": "photo"}
    return None
