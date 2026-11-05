"""Stage D/H — build the machine-readable edit decision list from editorial/edit_plan.json.

Resolves source clips, validates in/out points, quantises to the output frame grid,
snaps montage cuts to music beats, expands B-roll cutaways (video changes, dialogue
audio continues), places music cues and collects subtitles from the transcripts.
"""
from __future__ import annotations

import re
from fractions import Fraction

from common import CFG, PROJECT_DIR, REPORTS, get_logger, load_inventory, read_json, write_json

log = get_logger("stage_d")
FPS = Fraction(CFG["render"]["fps"])
PLAN = PROJECT_DIR / "editorial" / "edit_plan.json"
TRANSCRIPTS = PROJECT_DIR / "work" / "transcripts"
XFADE_MUSIC = 2.5
FROZEN = read_json(REPORTS / "frozen_spans.json", {})  # from pipeline/scan_frozen.py
# Location-card title → (stop id, phase) for the route-map overlay. Order is south to north.
ROUTE_MAPS = {
    "Gokarna Main Beach": ("main", "depart"),
    "Belekan Beach": ("belekan", "trek"),
    "Paradise Beach": ("paradise", "trek"),
    "Half Moon Beach": ("halfmoon", "trek"),
    "Om Beach": ("om", "trek"),
    "Kudle Beach": ("kudle", "trek"),
    "Back at Gokarna Main Beach": ("main", "done"),
}

# Proper-noun fixes for Whisper's English translation (place names it mis-hears).
TRANSLATION_FIXES = [
    (r"\b(Home|Oum|Ohm)\s+[Bb]each", "Om Beach"), (r"\bhome beach", "Om Beach"),
    (r"\b(Kutle|Hoodle|Good Day|Kudley)\s+[Bb]each", "Kudle Beach"), (r"\bKutle\b", "Kudle"),
    (r"\b(Balkan|Belkan|Belkin)\b(\s+[Bb]each)?", r"Belekan Beach"), (r"\bKokanna\b", "Gokarna"),
    (r"\b(\w)\s*paradise beach", r"\1 Paradise Beach"), (r"\bparadise beach", "Paradise Beach"),
    (r"\bhalf moon beach", "Half Moon Beach"), (r"\bom beach", "Om Beach"),
    (r"\bPasina\b", "sweat"), (r"\s+", " "),
]


def frames(seconds: float) -> int:
    return max(1, round(Fraction(seconds).limit_denominator(100000) * FPS))


def secs(n_frames: int) -> float:
    return float(n_frames / FPS)


def fix_text(text: str) -> str:
    for pat, rep in TRANSLATION_FIXES:
        text = re.sub(pat, rep, text)
    return text.strip()


def load_speech(clip_id: str) -> dict:
    return read_json(TRANSCRIPTS / f"{clip_id}.json", {}) or {}


def speech_lines(clip_id: str, a_in: float, a_out: float, overrides: dict, drops: set) -> list[dict]:
    """Subtitle candidates (source time) for an audio range: English shown on screen."""
    tr = load_speech(clip_id)
    lang = tr.get("language")
    out = []
    if lang == "en":
        src = [(s, False) for s in tr.get("segments", []) if s["keep"]]
    else:
        src = [(s, True) for s in tr.get("translation", []) if s["keep"]]
    for s, translated in src:
        key = f"{clip_id}@{s['start']:.2f}"
        if key in drops:
            continue
        if s["end"] <= a_in + 0.15 or s["start"] >= a_out - 0.15:
            continue
        text = overrides.get(key, fix_text(s["text"]))
        out.append({"key": key, "start": max(s["start"], a_in), "end": min(s["end"], a_out), "text": text,
                    "translated": translated, "source_language": lang})
    hindi = []
    if lang == "hi":
        for s in tr.get("segments", []):
            if s["keep"] and s["end"] > a_in + 0.15 and s["start"] < a_out - 0.15:
                hindi.append({"key": f"{clip_id}@{s['start']:.2f}", "start": max(s["start"], a_in),
                              "end": min(s["end"], a_out), "text": s["text"].strip(),
                              "avg_logprob": s["avg_logprob"]})
    return out, hindi


def main() -> None:
    global TRANSLATION_FIXES, ROUTE_MAPS
    plan = read_json(PLAN)
    # An episode can replace the built-in place-name fixes and the beach route map.
    if "translation_fixes" in plan:
        TRANSLATION_FIXES = [(a, b) for a, b in plan["translation_fixes"]]
    if "route_maps" in plan:
        ROUTE_MAPS = {k: tuple(v) for k, v in plan["route_maps"].items()}
    inv = {c["clip_id"]: c for c in load_inventory()}
    music = read_json(REPORTS / "music_analysis.json")
    overrides = plan.get("subtitle_overrides", {})
    drops = set(plan.get("subtitle_drop", []))

    entries, cues, subs, hindi_subs, problems = [], [], [], [], []
    t = 0  # timeline position in frames
    cur_cue = None
    for ch in plan["chapters"]:
        m = ch.get("music")
        if m:
            start_s = max(0.0, secs(t) - (XFADE_MUSIC if cues else 0))
            if cur_cue:
                cur_cue["timeline_end"] = secs(t) + (XFADE_MUSIC / 2 if m != "none" else 1.5)
            if m == "none":
                cur_cue = None
            else:
                track = music[m["track"]]
                offset = m.get("offset", 0.0)
                if offset > 0 and m.get("snap_offset_to_phrase", True) and track["phrases"]:
                    offset = min(track["phrases"], key=lambda p: abs(p - offset))
                cur_cue = {"cue_id": f"M{len(cues) + 1}", "track": m["track"],
                           "path": str(PROJECT_DIR / "music" / m["track"]), "track_offset": round(offset, 3),
                           "timeline_start": round(start_s if cues else 0.0, 3), "timeline_end": None,
                           "align": m.get("align", "start"), "track_end": m.get("track_end"),
                           "chapter": ch["id"], "tempo_bpm": track["tempo_bpm"]}
                cues.append(cur_cue)
        for e in ch["entries"]:
            clip = inv.get(e["clip"])
            if not clip or clip["source_kind"] == "unusable":
                problems.append(f"{e['clip']}: missing/unusable clip")
                continue
            speed = float(e.get("speed", 1.0))
            src_in, src_out = float(e["in"]), float(e["out"])
            max_out = clip["duration_s"] - 0.05
            if src_out > max_out:
                problems.append(f"{e['clip']}: out {src_out} > duration {clip['duration_s']:.2f}; clamped")
                src_out = max_out
            if src_in < 0 or src_out - src_in < 0.3:
                problems.append(f"{e['clip']}: invalid range {src_in}-{src_out}; skipped")
                continue
            n = frames((src_out - src_in) / speed)
            snapped = None
            if e.get("snap") and cur_cue and cur_cue["align"] == "start":
                tr = music[cur_cue["track"]]
                grid = tr["bars"] if e["snap"] == "bar" else tr["beats"]
                t_end = secs(t + n)
                cands = [cur_cue["timeline_start"] + (b - cur_cue["track_offset"]) for b in grid]
                cands = [c for c in cands if abs(c - t_end) <= (1.3 if e["snap"] == "bar" else 0.45)]
                for c in sorted(cands, key=lambda c: abs(c - t_end)):
                    n2 = frames(c - secs(t))
                    if n2 >= frames(0.8) and src_in + secs(n2) * speed <= max_out:
                        snapped = round(c, 3)
                        n = n2
                        break
            src_out = src_in + secs(n) * speed
            base = {"chapter": ch["id"], "speed": speed, "mode": e.get("mode", "ambient"),
                    "reason": e.get("why", ""), "grade": e.get("grade", {}), "stabilize": e.get("stabilize", False),
                    "fade_in": e.get("fade_in", 0), "fade_out": e.get("fade_out", 0),
                    "nat_gain_db": e.get("nat_gain_db", 0.0), "beat_snap": snapped,
                    "music_cue": cur_cue["cue_id"] if cur_cue else None}
            # Split into pieces when B-roll covers part of a dialogue shot.
            pieces = []
            cursor = 0.0
            for b in sorted(e.get("broll", []), key=lambda b: b["at"]):
                bclip = inv[b["clip"]]
                at = float(b["at"])
                dur = min(float(b["dur"]), float(secs(n) * speed) - at)   # frame quantisation can shave the shot
                if dur < 0.5 or b["in"] + dur > bclip["duration_s"] - 0.05:
                    problems.append(f"B-roll {b['clip']}@{b['in']} exceeds bounds; skipped")
                    continue
                if at > cursor:
                    pieces.append((e["clip"], src_in + cursor, at - cursor, "main", None))
                pieces.append((b["clip"], b["in"], dur, "broll", b.get("why")))
                cursor = at + dur
            total = secs(n) * speed
            if cursor < total - 1.5 / FPS:
                pieces.append((e["clip"], src_in + cursor, total - cursor, "main", None))
            a_pos = src_in
            used = 0
            for k, (vclip, vin, vdur, role, why) in enumerate(pieces):
                pn = n - used if k == len(pieces) - 1 else frames(vdur / speed)
                used += pn
                vc = inv[vclip]
                ent = {**base, "entry_id": f"E{len(entries) + 1:03d}", "role": role,
                       "video": {"clip_id": vclip, "path": vc["source_path"], "source_kind": vc["source_kind"],
                                 "in": round(vin, 4), "out": round(vin + secs(pn) * speed, 4),
                                 "width": vc["width"], "height": vc["height"], "codec": vc["video_codec"]},
                       "audio": {"clip_id": e["clip"], "path": clip["source_path"], "has_audio": clip["has_audio"],
                                 "in": round(a_pos, 4), "out": round(a_pos + secs(pn) * speed, 4)},
                       "timeline_start": round(secs(t), 4), "timeline_end": round(secs(t + pn), 4),
                       "frames": pn, "overlays": []}
                if role == "broll" and why:
                    ent["reason"] = f"Cutaway: {why}"
                if k > 0:
                    ent["fade_in"] = 0
                if k < len(pieces) - 1:
                    ent["fade_out"] = 0
                if ent["video"]["out"] > vc["duration_s"] + 0.01:
                    problems.append(f"{ent['entry_id']}: video out exceeds source")
                v = ent["video"]
                frozen = sum(max(0.0, min(b, v["out"]) - max(a, v["in"])) for a, b in FROZEN.get(v["clip_id"], []))
                if frozen > 0.2:
                    problems.append(f"{ent['entry_id']} {v['clip_id']}: {frozen:.1f}s of frozen picture "
                                    f"in {v['in']:.1f}-{v['out']:.1f}; cover with B-roll or retrim")
                entries.append(ent)
                a_pos += secs(pn) * speed
                t += pn
            e_start = entries[-len(pieces)]["timeline_start"]
            e_end = secs(t)
            for kind in ("title", "location", "info"):
                if kind in e:
                    o = e[kind]
                    st = e_start + o.get("at", 0.4)
                    en = min(e_end, st + o.get("dur", 4.5))
                    subs_kind = {"kind": kind, "text": o["text"], "detail": o.get("detail"),
                                 "timeline_start": round(st, 3), "timeline_end": round(en, 3),
                                 "source_ref": o.get("source")}
                    pending_here = entries[-1].setdefault("_overlays_pending", [])
                    pending_here.append(subs_kind)
                    place_maps = plan.get("place_maps") or {}
                    if kind == "location" and o["text"] in place_maps:
                        mend = min(e_end, st + float(place_maps[o["text"]].get("dur", 6.5)))
                        if mend - st >= 2.5:
                            pending_here.append({"kind": "map", "style": "places", **place_maps[o["text"]],
                                                 "timeline_start": round(st, 3), "timeline_end": round(mend, 3)})
                    elif kind == "location" and o["text"] in ROUTE_MAPS:
                        here, phase = ROUTE_MAPS[o["text"]]
                        mend = min(e_end, st + 6.5)
                        if mend - st >= 2.5:
                            pending_here.append({"kind": "map", "here": here, "phase": phase, "text": o["text"],
                                                 "basemap": "osm-z16",
                                                 "timeline_start": round(st, 3), "timeline_end": round(mend, 3)})
            if e.get("subs", e.get("mode") == "dialogue") and speed == 1.0:
                lines, hlines = speech_lines(e["clip"], src_in, src_in + secs(n), overrides, drops)
                for s in lines:
                    subs.append({**s, "timeline_start": round(e_start + s["start"] - src_in, 3),
                                 "timeline_end": round(e_start + s["end"] - src_in, 3), "clip_id": e["clip"]})
                for s in hlines:
                    hindi_subs.append({**s, "timeline_start": round(e_start + s["start"] - src_in, 3),
                                       "timeline_end": round(e_start + s["end"] - src_in, 3), "clip_id": e["clip"]})
    total_s = secs(t)
    if cur_cue:
        cur_cue["timeline_end"] = round(total_s, 3)
    for c in cues:
        tr = music[c["track"]]
        if c["align"] == "end":
            track_end = c["track_end"] or tr["duration_s"]
            c["track_offset"] = round(max(0.0, track_end - (c["timeline_end"] - c["timeline_start"])), 3)
            c["note"] = f"offset chosen so the track's ending ({track_end:.1f}s) lands on the final frame"
        avail = tr["duration_s"] - c["track_offset"]
        if c["timeline_end"] - c["timeline_start"] > avail:
            problems.append(f"{c['cue_id']} {c['track']}: needs {c['timeline_end'] - c['timeline_start']:.1f}s "
                            f"but only {avail:.1f}s available; will fade early")
            c["timeline_end"] = round(c["timeline_start"] + avail, 3)
    # Subtitle hygiene: no overlaps, minimum 1 s, maximum 6 s, split long lines.
    subs.sort(key=lambda s: s["timeline_start"])
    merged = []
    for s in subs:
        p = merged[-1] if merged else None
        short = (s["timeline_end"] - s["timeline_start"] < 1.8 or len(s["text"]) < 20 or
                 (p and (p["timeline_end"] - p["timeline_start"] < 1.8 or len(p["text"]) < 20)))
        if (p and p["clip_id"] == s["clip_id"] and p["translated"] == s["translated"] and short
                and s["timeline_start"] - p["timeline_end"] <= 0.6 and len(p["text"]) + len(s["text"]) <= 84):
            a, b = p["text"].rstrip(), s["text"].lstrip()
            sep = ". " if b[:1].isupper() and not b.startswith("I ") and a[-1:] not in ".,!?:-" else " "
            p["text"] = f"{a}{sep}{b}"
            p["timeline_end"] = s["timeline_end"]
            p["key"] += "+" + s["key"].split("@")[1]
        else:
            merged.append(dict(s))
    subs = merged
    clean = []
    for s in subs:
        if clean and s["timeline_start"] < clean[-1]["timeline_end"]:
            clean[-1]["timeline_end"] = max(clean[-1]["timeline_start"] + 0.6, s["timeline_start"] - 0.04)
        dur = s["timeline_end"] - s["timeline_start"]
        if len(s["text"]) > 84 and dur > 3:
            words = s["text"].split()
            half = len(words) // 2
            mid = s["timeline_start"] + dur * len(" ".join(words[:half])) / len(s["text"])
            clean.append({**s, "text": " ".join(words[:half]), "timeline_end": round(mid, 3)})
            clean.append({**s, "text": " ".join(words[half:]), "timeline_start": round(mid, 3)})
        else:
            clean.append(s)
    for i, s in enumerate(clean):
        s["sub_id"] = f"S{i + 1:03d}"
        if s["timeline_end"] - s["timeline_start"] < 1.0:
            nxt = clean[i + 1]["timeline_start"] - 0.04 if i + 1 < len(clean) else total_s
            s["timeline_end"] = round(min(nxt, s["timeline_start"] + 1.0), 3)
    # Attach overlays (with timeline-absolute windows) to every entry they intersect.
    pending = []
    for ent in entries:
        pending += ent.pop("_overlays_pending", [])
    for s in clean:
        pending.append({"kind": "subtitle", "text": s["text"], "translated": s["translated"],
                        "timeline_start": s["timeline_start"], "timeline_end": s["timeline_end"], "sub_id": s["sub_id"]})
    for c in plan.get("end_credits", []):
        pending.append({"kind": "credit", "text": c["text"], "timeline_start": round(total_s - c["from_end"], 3),
                        "timeline_end": round(total_s - c.get("until_end", 0.3), 3)})
    for o in pending:
        for ent in entries:
            if o["timeline_start"] < ent["timeline_end"] and o["timeline_end"] > ent["timeline_start"]:
                ent["overlays"].append(o)
                if o["kind"] == "subtitle":
                    ent.setdefault("subtitle_refs", []).append(o["sub_id"])
    edl = {"version": 1, "fps": str(FPS), "width": CFG["render"]["width"], "height": CFG["render"]["height"],
           "duration_s": round(total_s, 3), "entries": entries, "music_cues": cues, "subtitles_en": clean,
           "subtitles_hi_draft": hindi_subs, "problems": problems}
    write_json(REPORTS / "edit_decision_list.json", edl)
    for p in problems:
        log.warning(p)
    log.info("EDL: %d entries, %.1f s (%d:%02d), %d music cues, %d EN subtitles, %d HI draft lines",
             len(entries), total_s, int(total_s // 60), int(total_s % 60), len(cues), len(clean), len(hindi_subs))
    by_ch: dict[str, float] = {}
    for e in entries:
        by_ch[e["chapter"]] = by_ch.get(e["chapter"], 0) + e["timeline_end"] - e["timeline_start"]
    for k, v in by_ch.items():
        log.info("  chapter %-14s %6.1f s", k, v)


if __name__ == "__main__":
    main()
