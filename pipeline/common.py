"""Shared helpers: config, logging, ffprobe/ffmpeg wrappers, JSON state."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

# VLOG_PROJECT points at an episode folder (its own config.toml). Without it, the
# episode is the folder that contains this pipeline, so the original film still runs.
PROJECT_DIR = (Path(os.environ["VLOG_PROJECT"]).resolve() if os.environ.get("VLOG_PROJECT")
               else Path(__file__).resolve().parent.parent)


def load_config() -> dict[str, Any]:
    with open(PROJECT_DIR / "config.toml", "rb") as fh:
        return tomllib.load(fh)


CFG = load_config()
SOURCE_DIR = Path(CFG["paths"]["source_dir"])


def episode_dates() -> list[str]:
    """Days this film may use. One date, or capture_dates when two days belong together."""
    extra = CFG["episode"].get("capture_dates")
    if extra:
        return [str(d) for d in extra]
    return [str(CFG["episode"]["capture_date"])]
REPORTS = PROJECT_DIR / "reports"
WORK = PROJECT_DIR / "work"
LOGS = PROJECT_DIR / "logs"
REVIEW = PROJECT_DIR / "review"
for _d in (REPORTS, WORK, LOGS, REVIEW):
    _d.mkdir(parents=True, exist_ok=True)


def get_logger(stage: str) -> logging.Logger:
    log = logging.getLogger(stage)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s")
    fh = logging.FileHandler(LOGS / f"{stage}.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def assert_not_source(path: Path) -> None:
    """Guard: refuse to write anywhere inside the original media folder."""
    p = Path(path).resolve()
    if p == SOURCE_DIR.resolve() or SOURCE_DIR.resolve() in p.parents:
        raise RuntimeError(f"Refusing to write inside source folder: {p}")


def run(cmd: list[str], log: logging.Logger | None = None, check: bool = True,
        capture: bool = True) -> subprocess.CompletedProcess:
    if log:
        log.debug("RUN %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=capture, text=False)
    if check and res.returncode != 0:
        err = (res.stderr or b"").decode("utf-8", "replace")[-3000:]
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd[:6])} ...\n{err}")
    return res


def ffprobe_json(path: Path) -> dict:
    res = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format",
               "-show_streams", str(path)])
    return json.loads(res.stdout)


def read_json(path: Path, default: Any = None) -> Any:
    if not Path(path).exists():
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    assert_not_source(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    tmp.replace(path)


def load_inventory() -> list[dict]:
    inv = read_json(REPORTS / "clip_inventory.json")
    if inv is None:
        raise RuntimeError("Run stage_a_inventory.py first")
    return inv["clips"]


def fmt_tc(seconds: float) -> str:
    m, s = divmod(max(0.0, seconds), 60)
    return f"{int(m):02d}:{s:05.2f}"
