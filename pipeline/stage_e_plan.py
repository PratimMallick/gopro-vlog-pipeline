"""Stage E — write editorial/edit_plan.json with a local model.

The model is an open-weight MLX file, the same kind of local run as Whisper.
The first run downloads that file from Hugging Face. There is no API key and no
paid service. A plan that is already on disk is left as it is.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from common import CFG, PROJECT_DIR, REPORTS, WORK, get_logger, load_inventory, read_json, write_json
from plan_local import SYSTEM_PROMPT, build_brief, parse_json_object, prepare_plan

log = get_logger("stage_e")
PLAN = PROJECT_DIR / "editorial" / "edit_plan.json"
DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"


def model_id() -> str:
    return str(CFG.get("plan", {}).get("model") or DEFAULT_MODEL)


def max_tokens() -> int:
    return int(CFG.get("plan", {}).get("max_tokens") or 4096)


def load_transcripts(clips: list[dict]) -> dict[str, dict]:
    out = {}
    folder = WORK / "transcripts"
    for clip in clips:
        path = folder / f"{clip['clip_id']}.json"
        if path.exists():
            out[clip["clip_id"]] = read_json(path)
    return out


def complete(messages: list[dict], model_name: str, limit: int) -> str:
    """Run one local completion. The weights stay on this machine after download."""
    slug = "models--" + model_name.replace("/", "--")
    cached = Path.home() / ".cache" / "huggingface" / "hub" / slug
    if cached.is_dir() and not os.environ.get("HF_HUB_OFFLINE"):
        os.environ["HF_HUB_OFFLINE"] = "1"
        log.info("Using the cached model. This run does not contact a model service.")
    else:
        log.info("Downloading %s once. The file is free. No key and no account are required.", model_name)
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_name)
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return generate(model, tokenizer, prompt=prompt, max_tokens=limit,
                    sampler=make_sampler(temp=0), verbose=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the edit plan with a local model.")
    parser.add_argument("--force", action="store_true", help="Replace an edit plan that is already there.")
    parser.add_argument("--dry-run", action="store_true", help="Write the brief and do not load the model.")
    args = parser.parse_args()

    if PLAN.exists() and not args.force and not args.dry_run:
        log.info("Edit plan already exists at %s. Leave it, or pass --force to write a new one.", PLAN)
        return

    clips = load_inventory()
    transcripts = load_transcripts(clips)
    music = read_json(REPORTS / "music_analysis.json", {}) or {}
    brief = build_brief(CFG, clips, transcripts, music)
    brief_path = REPORTS / "plan_brief.txt"
    brief_path.write_text(brief, encoding="utf-8")
    log.info("Brief written to %s (%d characters).", brief_path, len(brief))
    if args.dry_run:
        return

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": brief},
    ]
    raw = ""
    plan: dict = {}
    errors: list[str] = []
    notes: list[str] = []
    for attempt in (1, 2):
        raw = complete(messages, model_id(), max_tokens())
        try:
            plan = parse_json_object(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            errors = [str(exc)]
        else:
            plan, errors, notes = prepare_plan(plan, clips, transcripts, music, CFG)
        if not errors:
            break
        log.warning("Attempt %d was rejected: %s", attempt, "; ".join(errors[:8]))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": brief},
            {"role": "assistant", "content": raw[:12000]},
            {"role": "user", "content": "Rejected:\n- " + "\n- ".join(errors[:30])
             + "\nReturn the full JSON object only."},
        ]
    draft = REPORTS / "plan_draft.txt"
    draft.write_text(raw, encoding="utf-8")
    if errors:
        log.error("No plan was written. %s", " ".join(errors[:12]))
        sys.exit(1)
    for note in notes:
        log.info(note)
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    write_json(PLAN, plan)
    log.info("Wrote %s. Run stage_d_edl.py next. Edit this file by hand if you want a different cut.", PLAN)


if __name__ == "__main__":
    main()
