"""Build and check an edit plan from local transcripts.

No network call lives here. stage_e_plan.py is what runs the local model.
"""
from __future__ import annotations

import json
import re
from typing import Any

SYSTEM_PROMPT = """You cut a travel vlog from the clips and speech lines in the user message.
Reply with one JSON object and no other text.

The film can only show what is in that message.
- Every clip id and every music filename must appear in the message.
- in and out are seconds. Stay inside the clip. Each shot is at least 0.8 seconds.
- The sum of (out - in) / speed across entries must fall between the minimum and maximum length.
- mode is dialogue only for a range that contains speech lines you are keeping. Otherwise use music or ambient.
- Do not write words the person did not say. subtitle_overrides may only use keys printed in the message, and only to fix a misheard name. subtitle_drop removes a printed line.
- A location or info card must repeat a fact that is in a speech line of that same clip. Leave the card out when the line does not say it. Do not add history, prices, heights, or distances.
- The title card uses the film title and date from the message.
- route_maps is {}. Do not add place_maps.
- A chapter with no music keeps the previous track. Use a track only for as long as that file lasts.
- end_credits names the tracks you used and says Kevin MacLeod, incompetech.com, CC BY 4.0.
- why is a private note. It is not shown on screen.
"""


def parse_json_object(text: str) -> dict:
    """Take the first JSON object out of a model reply."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        blob = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("The model did not return a JSON object.")
        blob = text[start:end + 1]
    blob = re.sub(r",\s*([}\]])", r"\1", blob)
    data = json.loads(blob)
    if not isinstance(data, dict):
        raise ValueError("The model returned JSON that is not an object.")
    return data


def speech_lines(clip_id: str, transcript: dict | None) -> list[dict]:
    """Lines that can appear on screen, with the key stage D expects."""
    transcript = transcript or {}
    lang = transcript.get("language")
    rows = transcript.get("segments") if lang == "en" else transcript.get("translation") or []
    lines = []
    for row in rows:
        if not row.get("keep", True):
            continue
        start = float(row["start"])
        lines.append({
            "key": f"{clip_id}@{start:.2f}",
            "start": start,
            "end": float(row["end"]),
            "text": str(row.get("text", "")).strip(),
            "avg_logprob": float(row.get("avg_logprob", 0)),
        })
    return lines


def build_brief(cfg: dict, clips: list[dict], transcripts: dict[str, dict],
                music: dict[str, dict], max_chars: int = 24000) -> str:
    """Compact text the local model can cut from. Video files are not included."""
    episode = cfg["episode"]
    usable = [c for c in clips if c.get("source_kind") != "unusable" and c.get("episode_date_ok", True)]
    usable.sort(key=lambda c: (c.get("capture_local") or "", c["clip_id"]))
    speech: dict[str, list[dict]] = {}
    for clip in usable:
        if clip.get("has_audio"):
            speech[clip["clip_id"]] = speech_lines(clip["clip_id"], transcripts.get(clip["clip_id"]))

    ranked = sorted(
        (line for lines in speech.values() for line in lines),
        key=lambda line: line["avg_logprob"],
        reverse=True,
    )
    kept_keys = {line["key"] for line in ranked[:180]}
    for clip_id, lines in speech.items():
        speech[clip_id] = [line for line in lines if line["key"] in kept_keys][:8]

    tracks = []
    for name, info in sorted(music.items()):
        if name.endswith(".mp3"):
            tracks.append(f"- {name} ({info.get('duration_s', 0):.0f}s)")
    if not tracks:
        tracks.append("- none. Leave music off.")

    header = [
        f"Film title: {episode['title']}",
        f"Date label: {episode['date_label']}",
        f"Length: {episode['min_duration_s']} to {episode['max_duration_s']} seconds.",
        "Music files:",
        *tracks,
        "",
        "Clips. A line may be used on screen only if it is printed under that clip.",
    ]
    body = []
    omitted = 0
    for clip in usable:
        lines = speech.get(clip["clip_id"]) or []
        block = [f"{clip['clip_id']}  {float(clip['duration_s']):.1f}s"
                 + ("" if clip.get("has_audio") else "  no audio")]
        for line in lines:
            block.append(f"  {line['key']} [{line['start']:.2f}-{line['end']:.2f}] {line['text']}")
        body.append("\n".join(block))
    text = "\n".join(header + body)
    while len(text) > max_chars and body:
        body.pop()
        omitted += 1
        text = "\n".join(header + body)
    if omitted:
        text += f"\n\n{omitted} other clips from this day were left out of this brief. Do not use them."
    return text


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if len(w) > 3]


def _grounded(text: str, spoken: str) -> bool:
    words = _words(text)
    if not words:
        return False
    blob = spoken.lower()
    return sum(word in blob for word in words) / len(words) >= 0.6


def _credit_text(tracks: list[str]) -> str:
    titles = ", ".join(f'"{name[:-4].replace("_", " ")}"' for name in tracks)
    return (f"Music {titles}\nKevin MacLeod (incompetech.com)\n"
            "Licensed under Creative Commons: By Attribution 4.0 License\n"
            "https://creativecommons.org/licenses/by/4.0/")


def prepare_plan(plan: dict, clips: list[dict], transcripts: dict[str, dict],
                 music: dict[str, dict], cfg: dict) -> tuple[dict, list[str], list[str]]:
    """Return the plan, hard errors, and notes about fields that were removed."""
    errors: list[str] = []
    notes: list[str] = []
    by_id = {c["clip_id"]: c for c in clips if c.get("source_kind") != "unusable" and c.get("episode_date_ok", True)}
    spoken: dict[str, list[dict]] = {
        cid: speech_lines(cid, transcripts.get(cid)) for cid in by_id
    }
    keys = {line["key"]: line["text"] for lines in spoken.values() for line in lines}

    if plan.get("place_maps"):
        plan.pop("place_maps", None)
        notes.append("Removed place_maps. A map needs coordinates checked by hand.")
    if plan.get("route_maps"):
        notes.append("Cleared route_maps. The built-in beach map is only for a plan that names those beaches.")
    plan["route_maps"] = {}
    fixes = plan.get("translation_fixes", [])
    if fixes and not all(isinstance(pair, list) and len(pair) == 2 for pair in fixes):
        errors.append("translation_fixes must be a list of [pattern, replacement] pairs.")

    for key in plan.get("subtitle_overrides", {}):
        if key not in keys:
            errors.append(f"subtitle override {key} does not match a speech line.")
    for key in plan.get("subtitle_drop", []):
        if key not in keys:
            errors.append(f"subtitle drop {key} does not match a speech line.")

    chapters = plan.get("chapters")
    if not isinstance(chapters, list) or not chapters:
        return plan, ["The plan has no chapters."], notes

    total = 0.0
    used_tracks: list[str] = []
    for ch in chapters:
        if not isinstance(ch, dict) or "id" not in ch or not isinstance(ch.get("entries"), list):
            errors.append("Each chapter needs an id and a list of entries.")
            continue
        music_spec = ch.get("music")
        if isinstance(music_spec, dict):
            track = music_spec.get("track")
            if track not in music:
                errors.append(f"Chapter {ch['id']} uses music {track}, which is not in this episode.")
            elif track not in used_tracks:
                used_tracks.append(track)
        elif music_spec not in (None, "none"):
            errors.append(f"Chapter {ch['id']} has a music value this pipeline cannot read.")
        for entry in ch["entries"]:
            total += _check_entry(entry, ch["id"], by_id, spoken, cfg, errors)
    lo = float(cfg["episode"]["min_duration_s"])
    hi = float(cfg["episode"]["max_duration_s"])
    available = sum(float(c["duration_s"]) for c in by_id.values())
    if total > hi + 0.5:
        errors.append(f"The cut is {total:.0f}s, longer than the maximum {hi:.0f}s.")
    if total + 0.5 < lo and available >= lo:
        errors.append(f"The cut is {total:.0f}s, shorter than the minimum {lo:.0f}s.")
    if total < 0.3:
        errors.append("The cut has no usable shots.")

    credits = plan.setdefault("end_credits", [])
    joined = "\n".join(str(c.get("text", "")) for c in credits if isinstance(c, dict))
    if used_tracks and "Kevin MacLeod" not in joined:
        credits.append({"text": _credit_text(used_tracks), "from_end": 6.5})
        notes.append("Added the Kevin MacLeod credit for the tracks in the plan.")
    return plan, errors, notes


def _check_entry(entry: Any, chapter: str, by_id: dict, spoken: dict, cfg: dict, errors: list[str]) -> float:
    if not isinstance(entry, dict):
        errors.append(f"Chapter {chapter} has an entry that is not an object.")
        return 0.0
    clip = by_id.get(entry.get("clip"))
    label = f"{chapter}/{entry.get('clip')}"
    if clip is None:
        errors.append(f"{label} is not a usable clip from this day.")
        return 0.0
    try:
        src_in, src_out = float(entry["in"]), float(entry["out"])
        speed = float(entry.get("speed", 1.0))
    except (KeyError, TypeError, ValueError):
        errors.append(f"{label} needs numeric in and out points.")
        return 0.0
    if speed <= 0 or src_in < 0 or src_out - src_in < 0.3 or src_out > float(clip["duration_s"]) - 0.04:
        errors.append(f"{label} range {src_in}-{src_out} does not fit the {clip['duration_s']:.1f}s clip.")
        return 0.0
    mode = entry.get("mode", "ambient")
    if mode not in ("dialogue", "music", "ambient"):
        errors.append(f"{label} mode must be dialogue, music, or ambient.")
    lines = spoken.get(entry["clip"], [])
    heard = " ".join(line["text"] for line in lines if line["end"] > src_in + 0.15 and line["start"] < src_out - 0.15)
    if mode == "dialogue" and not heard.strip():
        errors.append(f"{label} is marked dialogue, and that range has no kept speech.")
    title = cfg["episode"]["title"]
    for kind in ("location", "info"):
        card = entry.get(kind)
        if isinstance(card, dict) and not _grounded(str(card.get("text", "")), heard):
            errors.append(f"{label} {kind} card is not supported by speech in that shot.")
    card = entry.get("title")
    if isinstance(card, dict) and card.get("text") not in (title, cfg["episode"].get("date_label")):
        if not _grounded(str(card.get("text", "")), heard):
            errors.append(f"{label} title card is not the film title and is not in the speech.")
    for broll in entry.get("broll", []):
        other = by_id.get(broll.get("clip"))
        if other is None:
            errors.append(f"{label} B-roll {broll.get('clip')} is not a usable clip from this day.")
            continue
        try:
            if float(broll["in"]) < 0 or float(broll["in"]) + float(broll["dur"]) > float(other["duration_s"]) - 0.04:
                errors.append(f"{label} B-roll {broll.get('clip')} does not fit that clip.")
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label} B-roll {broll.get('clip')} needs in and dur.")
    return (src_out - src_in) / speed
