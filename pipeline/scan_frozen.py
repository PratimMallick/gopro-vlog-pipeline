"""Find in-camera frozen-video spans (encoder emitting near-empty P-frames while audio continues).

Writes reports/frozen_spans.json: {clip_id: [[start_s, end_s], ...]} in source seconds.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from common import REPORTS, get_logger, load_inventory, run, write_json

log = get_logger("frozen")
MIN_RUN_S = 0.75


def scan(clip: dict) -> tuple[str, list[list[float]]]:
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,size",
               "-of", "csv=p=0", clip["source_path"]]).stdout.decode().split()
    pk = [(float(t), int(s)) for t, s in (ln.split(",")[:2] for ln in out if "," in ln and ln.split(",")[0] != "N/A")]
    pk.sort()
    if not pk:
        return clip["clip_id"], []
    sizes = sorted(s for _, s in pk)
    # A frozen P-frame is tiny both absolutely and relative to the clip's typical frame.
    thresh = min(1500, sizes[int(len(sizes) * 0.9)] * 0.01)
    # Keyframes recur about once a second even while frozen, so judge 0.5 s windows by their share of tiny frames.
    win = 0.5
    n = int(pk[-1][0] / win) + 1
    tiny, total = [0] * n, [0] * n
    for t, s in pk:
        i = int(t / win)
        total[i] += 1
        tiny[i] += s < thresh
    spans = []
    for i in range(n):
        if total[i] and tiny[i] / total[i] >= 0.8:
            a, b = i * win, min((i + 1) * win, pk[-1][0])
            if spans and abs(spans[-1][1] - a) < 1e-6:
                spans[-1][1] = b
            else:
                spans.append([a, b])
    return clip["clip_id"], [[round(a, 2), round(b, 2)] for a, b in spans if b - a >= MIN_RUN_S]


def main() -> None:
    clips = [c for c in load_inventory() if c["source_kind"] != "unusable"]
    with ThreadPoolExecutor(8) as ex:
        res = dict(ex.map(scan, clips))
    res = {k: v for k, v in res.items() if v}
    write_json(REPORTS / "frozen_spans.json", res)
    for k, v in sorted(res.items()):
        log.info("%s frozen %s", k, v)
    log.info("%d of %d clips contain frozen spans", len(res), len(clips))


if __name__ == "__main__":
    main()
