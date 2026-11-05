"""Pillow-rendered text overlays (FFmpeg build has no drawtext/libass).

Every overlay is a full-frame transparent PNG; all text stays inside the 90% title-safe area.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from common import CFG, WORK

W, H = CFG["render"]["width"], CFG["render"]["height"]
SAFE_X, SAFE_Y = int(W * 0.05), int(H * 0.05)
AVENIR = "/System/Library/Fonts/Avenir Next.ttc"
FUTURA = "/System/Library/Fonts/Supplemental/Futura.ttc"
OUT = WORK / "overlays"
OUT.mkdir(parents=True, exist_ok=True)
S = W / 1920  # scale factor if rendering at another size


def _font(path: str, size: float, index: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, int(size * S), index=index)


def _shadowed(canvas: Image.Image, draw_fn, blur: float = 6, alpha: int = 150) -> Image.Image:
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(shadow), (0, 0, 0, alpha), 2)
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur * S))
    canvas = Image.alpha_composite(canvas, shadow)
    draw_fn(ImageDraw.Draw(canvas), None, 0)
    return canvas


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = f"{cur} {w}".strip()
        if draw.textlength(t, font=font) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render_title(title: str, subtitle: str) -> Image.Image:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    f1, f2 = _font(FUTURA, 104, 2), _font(AVENIR, 40, 2)
    d = ImageDraw.Draw(img)
    tw = d.textlength(title, font=f1)
    if tw > W - 2 * SAFE_X:
        f1 = _font(FUTURA, 104 * (W - 2 * SAFE_X) / tw, 2)
        tw = d.textlength(title, font=f1)
    sw = d.textlength(subtitle, font=f2)
    spaced = " ".join(subtitle)
    sw = d.textlength(spaced, font=f2)
    y = H // 2 - int(80 * S)

    def draw(dr, col, off):
        dr.text(((W - tw) / 2 + off, y + off), title, font=f1, fill=col or (255, 255, 255, 255))
        lw = int(140 * S)
        dr.rectangle(((W - lw) / 2 + off, y + int(135 * S) + off, (W + lw) / 2 + off, y + int(139 * S) + off),
                     fill=col or (255, 196, 64, 255))
        dr.text(((W - sw) / 2 + off, y + int(160 * S) + off), spaced, font=f2, fill=col or (255, 255, 255, 235))
    return _shadowed(img, draw, blur=10, alpha=170)


def render_location(name: str, detail: str | None) -> Image.Image:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    f1, f2 = _font(FUTURA, 58, 2), _font(AVENIR, 30, 5)
    # Sits above the two-line subtitle zone so both can be on screen together.
    x, y = SAFE_X + int(20 * S), H - SAFE_Y - int(300 * S)

    def draw(dr, col, off):
        dr.rectangle((x + off, y + int(8 * S) + off, x + int(7 * S) + off, y + int(120 * S) + off),
                     fill=col or (255, 196, 64, 255))
        dr.text((x + int(26 * S) + off, y + off), name.upper(), font=f1, fill=col or (255, 255, 255, 255))
        if detail:
            dr.text((x + int(28 * S) + off, y + int(78 * S) + off), detail, font=f2, fill=col or (240, 240, 240, 240))
    return _shadowed(img, draw)


def render_info(text: str) -> Image.Image:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    f = _font(AVENIR, 34, 5)
    d = ImageDraw.Draw(img)
    lines = _wrap(d, text, f, int(W * 0.42))
    lh = int(46 * S)
    box_w = max(d.textlength(l, font=f) for l in lines) + int(56 * S)
    box_h = lh * len(lines) + int(36 * S)
    x, y = W - SAFE_X - box_w, SAFE_Y + int(10 * S)
    d.rounded_rectangle((x, y, x + box_w, y + box_h), radius=int(14 * S), fill=(10, 20, 30, 150))
    d.rectangle((x, y + int(14 * S), x + int(6 * S), y + box_h - int(14 * S)), fill=(255, 196, 64, 230))
    for i, l in enumerate(lines):
        d.text((x + int(30 * S), y + int(16 * S) + i * lh), l, font=f, fill=(255, 255, 255, 250))
    return img


def render_subtitle(text: str, translated: bool) -> Image.Image:
    """Bottom-centre subtitle. Translations are italic, verbatim English is upright."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    f = _font(AVENIR, 44, 6 if translated else 5)
    d = ImageDraw.Draw(img)
    lines = _wrap(d, text, f, int(W * 0.72))[:2]
    lh = int(58 * S)
    y0 = H - SAFE_Y - int(30 * S) - lh * len(lines)
    for i, l in enumerate(lines):
        tw = d.textlength(l, font=f)
        x, y = (W - tw) / 2, y0 + i * lh
        d.rounded_rectangle((x - int(16 * S), y - int(4 * S), x + tw + int(16 * S), y + lh - int(2 * S)),
                            radius=int(8 * S), fill=(0, 0, 0, 140))
        d.text((x, y), l, font=f, fill=(255, 255, 255, 255))
    return img


# Published OpenStreetMap beach centroids (north-up). The camera recorded no GPS, so the
# line is the beach-to-beach order, bent to stay on land, not a tracked trail.
MAP_STOPS = [
    ("belekan", "Belekan", 1788, 1751),
    ("paradise", "Paradise", 1472, 1872),
    ("halfmoon", "Half Moon", 1161, 1879),
    ("om", "Om", 904, 1826),
    ("kudle", "Kudle", 458, 1375),
    ("main", "Main Beach", 229, 248),
]
MAP_CAPTION = {
    ("main", "depart"): "Staying here  ·  auto to Belekan, then the trek back",
    ("belekan", "trek"): "From Main Beach by auto  ·  next: Paradise",
    ("paradise", "trek"): "From Belekan  ·  next: Half Moon",
    ("halfmoon", "trek"): "From Paradise  ·  next: Om",
    ("om", "trek"): "From Half Moon  ·  next: Kudle",
    ("kudle", "trek"): "From Om  ·  next: Main Beach",
    ("main", "done"): "From Kudle  ·  trek complete",
}
LABEL_ANCHOR = {
    "belekan": "left", "paradise": "above", "halfmoon": "below",
    "om": "above", "kudle": "right", "main": "right",
}
PIN = (214, 48, 32, 255)
AMBER = (255, 196, 64, 255)


def _is_water(rgb: tuple) -> bool:
    r, g, b = rgb
    return b > 175 and b > r + 15 and b + 8 >= g


def _on_land(im: Image.Image, x: float, y: float) -> bool:
    if not (1 < x < im.width - 2 and 1 < y < im.height - 2):
        return False
    return not _is_water(im.getpixel((int(x), int(y))))


def _bend(im: Image.Image, p: tuple, q: tuple, depth: int = 0) -> list[tuple]:
    """Push a segment inland until it no longer cuts across the sea."""
    if depth > 3 or all(_on_land(im, p[0] * (1 - t) + q[0] * t, p[1] * (1 - t) + q[1] * t) for t in (0.25, 0.5, 0.75)):
        return [p, q]
    mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
    dx, dy = q[0] - p[0], q[1] - p[1]
    length = math.hypot(dx, dy) or 1
    px, py = -dy / length, dx / length
    for dist in range(30, 520, 20):
        for sign in (1, -1):
            c = (mx + px * dist * sign, my + py * dist * sign)
            if _on_land(im, *c):
                left, right = _bend(im, p, c, depth + 1), _bend(im, c, q, depth + 1)
                return left[:-1] + right
    return [p, q]


def _route_segments(im: Image.Image) -> list[list[tuple]]:
    return [_bend(im, (a[2], a[3]), (b[2], b[3])) for a, b in zip(MAP_STOPS, MAP_STOPS[1:])]


def _dashed(dr: ImageDraw.ImageDraw, pts: list[tuple[float, float]], fill, width: int, dash: int = 16, gap: int = 10):
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        length = math.hypot(x2 - x1, y2 - y1) or 1
        pos = 0.0
        on = True
        while pos < length:
            nxt = min(length, pos + (dash if on else gap))
            if on:
                dr.line((x1 + (x2 - x1) * pos / length, y1 + (y2 - y1) * pos / length,
                         x1 + (x2 - x1) * nxt / length, y1 + (y2 - y1) * nxt / length), fill=fill, width=width)
            pos = nxt
            on = not on


def _arrow(dr: ImageDraw.ImageDraw, x: float, y: float, dx: float, dy: float, fill):
    h = math.hypot(dx, dy) or 1
    ux, uy = dx / h, dy / h
    s = 11 * S
    px, py = -uy, ux
    tip = (x + ux * s * 0.3, y + uy * s * 0.3)
    dr.polygon([tip, (x - ux * s + px * s * 0.55, y - uy * s + py * s * 0.55),
                (x - ux * s - px * s * 0.55, y - uy * s - py * s * 0.55)], fill=fill)


def _halo(dr: ImageDraw.ImageDraw, xy: tuple, text: str, font, fill) -> None:
    x, y = xy
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        dr.text((x + dx, y + dy), text, font=font, fill=(255, 255, 255, 235))
    dr.text((x, y), text, font=font, fill=fill)


def _map_card(here: str, phase: str) -> Image.Image:
    """North-up OpenStreetMap of the six beaches, with the stop we're at pinned."""
    base = Image.open(WORK / "maps" / "osm_z16.jpg").convert("RGB")
    idx = next(i for i, s in enumerate(MAP_STOPS) if s[0] == here)
    xs = [s[2] for s in MAP_STOPS]
    ys = [s[3] for s in MAP_STOPS]
    # Extra room on the right and top so the Belekan name and the Main Beach pin stay inside the card.
    box = (max(0, min(xs) - 90), max(0, min(ys) - 170), min(base.width, max(xs) + 210), min(base.height, max(ys) + 110))
    crop = base.crop(box)
    cw, map_h, bar = int(560 * S), int(500 * S), int(64 * S)
    scale = min(cw / crop.width, map_h / crop.height)
    mw, mh = int(crop.width * scale), int(crop.height * scale)
    card = Image.new("RGBA", (cw, mh + bar), (255, 255, 255, 255))
    ox, oy = (cw - mw) // 2, 0
    card.paste(crop.resize((mw, mh), Image.Resampling.LANCZOS), (ox, oy))
    dr = ImageDraw.Draw(card)

    def C(p):
        return ox + (p[0] - box[0]) * scale, oy + (p[1] - box[1]) * scale

    def leg(i: int) -> str:
        if phase == "done":
            return "done"
        if phase == "depart":
            return "future"
        if i < idx:
            return "done"
        if i == idx:
            return "next"
        return "future"

    for i, seg in enumerate(_route_segments(base)):
        pts = [C(p) for p in seg]
        style = leg(i)
        if style == "done":
            dr.line(pts, fill=(255, 255, 255, 255), width=max(4, int(7 * S)), joint="curve")
            dr.line(pts, fill=PIN, width=max(2, int(4 * S)), joint="curve")
        elif style == "next":
            dr.line(pts, fill=(255, 255, 255, 230), width=max(3, int(6 * S)), joint="curve")
            _dashed(dr, pts, PIN, max(2, int(3 * S)), dash=int(11 * S), gap=int(7 * S))
        else:
            dr.line(pts, fill=(255, 255, 255, 180), width=max(2, int(4 * S)), joint="curve")
            dr.line(pts, fill=(90, 90, 90, 140), width=max(1, int(2 * S)), joint="curve")

    font = _font(AVENIR, 18, 2)
    for i, (key, name, x, y) in enumerate(MAP_STOPS):
        cx, cy = C((x, y))
        current = i == idx
        nxt = (phase == "trek" and i == idx + 1) or (phase == "depart" and i == 0)
        if current:
            r = int(13 * S)
            tip = cy
            head = tip - r * 1.65
            dr.polygon([(cx, tip + r * 0.35), (cx - r * 0.7, head + r * 0.15), (cx + r * 0.7, head + r * 0.15)], fill=PIN)
            dr.ellipse((cx - r, head - r, cx + r, head + r), fill=PIN, outline=(255, 255, 255, 255), width=max(2, int(2 * S)))
            dr.ellipse((cx - r * 0.38, head - r * 0.38, cx + r * 0.38, head + r * 0.38), fill=(255, 255, 255, 255))
            cy = head - r
        else:
            r = int(6 * S)
            dr.ellipse((cx - r, cy - r, cx + r, cy + r), fill=PIN if nxt or phase == "done" else (255, 255, 255, 255),
                       outline=PIN, width=max(2, int(2 * S)))
        tw = dr.textlength(name, font=font)
        anchor = LABEL_ANCHOR[key]
        if anchor == "left":
            lx, ly = cx - tw - int(10 * S), cy - int(10 * S)
        elif anchor == "right":
            lx, ly = cx + int(10 * S), cy - int(10 * S)
        elif anchor == "above":
            lx, ly = cx - tw / 2, cy - int(28 * S)
        else:
            lx, ly = cx - tw / 2, cy + int(10 * S)
        lx = min(max(int(4 * S), lx), cw - tw - int(4 * S))
        ly = min(max(int(4 * S), ly), mh - int(22 * S))
        _halo(dr, (lx, ly), name, font, (25, 25, 25, 255) if current else (40, 40, 40, 230))

    credit = _font(AVENIR, 13, 5)
    dr.text((int(8 * S), mh - int(18 * S)), "© OpenStreetMap", font=credit, fill=(40, 40, 40, 210))
    cap = MAP_CAPTION[(here, phase)]
    fc = _font(AVENIR, 20, 5)
    lines = _wrap(dr, cap, fc, cw - int(28 * S))
    lh = int(26 * S)
    for i, line in enumerate(lines):
        tw = dr.textlength(line, font=fc)
        dr.text(((cw - tw) / 2, mh + int(8 * S) + i * lh), line, font=fc, fill=(25, 25, 25, 255))
    return card


def _places_card(spec: dict) -> Image.Image:
    """Two real places on one north-up map. The line between them is the boat crossing."""
    base = Image.open(WORK / "maps" / spec["basemap"]).convert("RGB")
    pts = spec["points"]
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    pad = spec.get("pad", [140, 120, 180, 100])
    box = (max(0, min(xs) - pad[0]), max(0, min(ys) - pad[1]),
           min(base.width, max(xs) + pad[2]), min(base.height, max(ys) + pad[3]))
    crop = base.crop(box)
    cw, map_h, bar = int(640 * S), int(460 * S), int(72 * S)
    scale = min(cw / crop.width, map_h / crop.height)
    mw, mh = max(1, int(crop.width * scale)), max(1, int(crop.height * scale))
    card = Image.new("RGBA", (cw, mh + bar), (255, 255, 255, 255))
    ox, oy = (cw - mw) // 2, 0
    card.paste(crop.resize((mw, mh), Image.Resampling.LANCZOS), (ox, oy))
    dr = ImageDraw.Draw(card)

    def C(p):
        return ox + (p["x"] - box[0]) * scale, oy + (p["y"] - box[1]) * scale

    placed = [C(p) for p in pts]
    if len(placed) >= 2:
        if spec.get("leg") == "done":
            dr.line(placed, fill=(255, 255, 255, 255), width=max(4, int(7 * S)))
            dr.line(placed, fill=PIN, width=max(2, int(4 * S)))
        else:
            dr.line(placed, fill=(255, 255, 255, 230), width=max(3, int(6 * S)))
            _dashed(dr, placed, PIN, max(2, int(3 * S)), dash=int(11 * S), gap=int(7 * S))
        # A small arrow at the middle of the crossing, pointing toward the island.
        (x1, y1), (x2, y2) = placed[0], placed[-1]
        _arrow(dr, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, PIN)

    font = _font(AVENIR, 20, 2)
    here = spec.get("here")
    for p, (cx, cy) in zip(pts, placed):
        current = p["id"] == here
        if current:
            r = int(13 * S)
            tip = cy
            head = tip - r * 1.65
            dr.polygon([(cx, tip + r * 0.35), (cx - r * 0.7, head + r * 0.15), (cx + r * 0.7, head + r * 0.15)], fill=PIN)
            dr.ellipse((cx - r, head - r, cx + r, head + r), fill=PIN, outline=(255, 255, 255, 255), width=max(2, int(2 * S)))
            dr.ellipse((cx - r * 0.38, head - r * 0.38, cx + r * 0.38, head + r * 0.38), fill=(255, 255, 255, 255))
            label_y = head - r
        else:
            r = int(7 * S)
            dr.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255, 255), outline=PIN, width=max(2, int(2 * S)))
            label_y = cy
        name = p["name"]
        tw = dr.textlength(name, font=font)
        anchor = p.get("anchor", "above")
        if anchor == "left":
            lx, ly = cx - tw - int(12 * S), label_y - int(12 * S)
        elif anchor == "right":
            lx, ly = cx + int(12 * S), label_y - int(12 * S)
        elif anchor == "below":
            lx, ly = cx - tw / 2, label_y + int(12 * S)
        else:
            lx, ly = cx - tw / 2, label_y - int(30 * S)
        lx = min(max(int(6 * S), lx), cw - tw - int(6 * S))
        ly = min(max(int(4 * S), ly), mh - int(24 * S))
        _halo(dr, (lx, ly), name, font, (25, 25, 25, 255) if current else (40, 40, 40, 230))

    nf = _font(AVENIR, 16, 2)
    _halo(dr, (int(10 * S), int(8 * S)), "N", nf, (25, 25, 25, 255))
    credit = _font(AVENIR, 13, 5)
    dr.text((int(8 * S), mh - int(18 * S)), "© OpenStreetMap", font=credit, fill=(40, 40, 40, 210))
    cap = spec.get("caption", "")
    fc = _font(AVENIR, 20, 5)
    lines = _wrap(dr, cap, fc, cw - int(28 * S))[:2]
    lh = int(26 * S)
    for i, line in enumerate(lines):
        tw = dr.textlength(line, font=fc)
        dr.text(((cw - tw) / 2, mh + int(8 * S) + i * lh), line, font=fc, fill=(25, 25, 25, 255))
    return card


def render_places_map(spec: dict) -> Image.Image:
    card = _places_card(spec)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    x0, y0 = W - SAFE_X - card.width, SAFE_Y
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        (x0 + int(5 * S), y0 + int(7 * S), x0 + card.width + int(5 * S), y0 + card.height + int(7 * S)),
        radius=int(16 * S), fill=(0, 0, 0, 110))
    img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(7 * S)))
    mask = Image.new("L", card.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, card.width - 1, card.height - 1), radius=int(16 * S), fill=255)
    img.paste(Image.composite(card, Image.new("RGBA", card.size, (0, 0, 0, 0)), mask), (x0, y0), mask)
    return img


def _places_card(spec: dict) -> Image.Image:
    """Two real places on one north-up map. The line between them is the boat crossing."""
    base = Image.open(WORK / "maps" / spec["basemap"]).convert("RGB")
    pts = spec["points"]
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    pad = spec.get("pad", [140, 120, 180, 100])
    box = (max(0, min(xs) - pad[0]), max(0, min(ys) - pad[1]),
           min(base.width, max(xs) + pad[2]), min(base.height, max(ys) + pad[3]))
    crop = base.crop(box)
    cw, map_h, bar = int(640 * S), int(460 * S), int(72 * S)
    scale = min(cw / crop.width, map_h / crop.height)
    mw, mh = max(1, int(crop.width * scale)), max(1, int(crop.height * scale))
    card = Image.new("RGBA", (cw, mh + bar), (255, 255, 255, 255))
    ox, oy = (cw - mw) // 2, 0
    card.paste(crop.resize((mw, mh), Image.Resampling.LANCZOS), (ox, oy))
    dr = ImageDraw.Draw(card)

    def C(p):
        return ox + (p["x"] - box[0]) * scale, oy + (p["y"] - box[1]) * scale

    placed = [C(p) for p in pts]
    if len(placed) >= 2:
        if spec.get("leg") == "done":
            dr.line(placed, fill=(255, 255, 255, 255), width=max(4, int(7 * S)))
            dr.line(placed, fill=PIN, width=max(2, int(4 * S)))
        else:
            dr.line(placed, fill=(255, 255, 255, 230), width=max(3, int(6 * S)))
            _dashed(dr, placed, PIN, max(2, int(3 * S)), dash=int(11 * S), gap=int(7 * S))
        (x1, y1), (x2, y2) = placed[0], placed[-1]
        _arrow(dr, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, PIN)

    font = _font(AVENIR, 20, 2)
    here = spec.get("here")
    for p, (cx, cy) in zip(pts, placed):
        current = p["id"] == here
        if current:
            r = int(13 * S)
            tip = cy
            head = tip - r * 1.65
            dr.polygon([(cx, tip + r * 0.35), (cx - r * 0.7, head + r * 0.15), (cx + r * 0.7, head + r * 0.15)], fill=PIN)
            dr.ellipse((cx - r, head - r, cx + r, head + r), fill=PIN, outline=(255, 255, 255, 255), width=max(2, int(2 * S)))
            dr.ellipse((cx - r * 0.38, head - r * 0.38, cx + r * 0.38, head + r * 0.38), fill=(255, 255, 255, 255))
            label_y = head - r
        else:
            r = int(7 * S)
            dr.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255, 255), outline=PIN, width=max(2, int(2 * S)))
            label_y = cy
        name = p["name"]
        tw = dr.textlength(name, font=font)
        anchor = p.get("anchor", "above")
        if anchor == "left":
            lx, ly = cx - tw - int(12 * S), label_y - int(12 * S)
        elif anchor == "right":
            lx, ly = cx + int(12 * S), label_y - int(12 * S)
        elif anchor == "below":
            lx, ly = cx - tw / 2, label_y + int(12 * S)
        else:
            lx, ly = cx - tw / 2, label_y - int(30 * S)
        lx = min(max(int(6 * S), lx), cw - tw - int(6 * S))
        ly = min(max(int(4 * S), ly), mh - int(24 * S))
        _halo(dr, (lx, ly), name, font, (25, 25, 25, 255) if current else (40, 40, 40, 230))

    nf = _font(AVENIR, 16, 2)
    _halo(dr, (int(10 * S), int(8 * S)), "N", nf, (25, 25, 25, 255))
    credit = _font(AVENIR, 13, 5)
    dr.text((int(8 * S), mh - int(18 * S)), "© OpenStreetMap", font=credit, fill=(40, 40, 40, 210))
    cap = spec.get("caption", "")
    fc = _font(AVENIR, 20, 5)
    lines = _wrap(dr, cap, fc, cw - int(28 * S))[:2]
    lh = int(26 * S)
    for i, line in enumerate(lines):
        tw = dr.textlength(line, font=fc)
        dr.text(((cw - tw) / 2, mh + int(8 * S) + i * lh), line, font=fc, fill=(25, 25, 25, 255))
    return card


def render_places_map(spec: dict) -> Image.Image:
    card = _places_card(spec)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    x0, y0 = W - SAFE_X - card.width, SAFE_Y
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        (x0 + int(5 * S), y0 + int(7 * S), x0 + card.width + int(5 * S), y0 + card.height + int(7 * S)),
        radius=int(16 * S), fill=(0, 0, 0, 110))
    img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(7 * S)))
    mask = Image.new("L", card.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, card.width - 1, card.height - 1), radius=int(16 * S), fill=255)
    img.paste(Image.composite(card, Image.new("RGBA", card.size, (0, 0, 0, 0)), mask), (x0, y0), mask)
    return img


def render_route_map(here: str, phase: str) -> Image.Image:
    """Real map inset, top-right, so it stays out of the centre of a selfie."""
    card = _map_card(here, phase)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    x0, y0 = W - SAFE_X - card.width, SAFE_Y
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        (x0 + int(5 * S), y0 + int(7 * S), x0 + card.width + int(5 * S), y0 + card.height + int(7 * S)),
        radius=int(16 * S), fill=(0, 0, 0, 110))
    img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(7 * S)))
    mask = Image.new("L", card.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, card.width - 1, card.height - 1), radius=int(16 * S), fill=255)
    img.paste(Image.composite(card, Image.new("RGBA", card.size, (0, 0, 0, 0)), mask), (x0, y0), mask)
    return img


def render_credit(lines: list[str]) -> Image.Image:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    f = _font(AVENIR, 30, 5)
    d = ImageDraw.Draw(img)
    lh = int(42 * S)
    y0 = H - SAFE_Y - lh * len(lines) - int(10 * S)
    for i, l in enumerate(lines):
        tw = d.textlength(l, font=f)
        d.text(((W - tw) / 2 + 2, y0 + i * lh + 2), l, font=f, fill=(0, 0, 0, 160))
        d.text(((W - tw) / 2, y0 + i * lh), l, font=f, fill=(255, 255, 255, 235))
    return img


def overlay_png(spec: dict) -> Path:
    key = hashlib.sha1(json.dumps({**spec, "W": W, "H": H, "v": 5}, sort_keys=True).encode()).hexdigest()[:16]
    path = OUT / f"{spec['kind']}_{key}.png"
    if path.exists():
        return path
    k = spec["kind"]
    if k == "title":
        img = render_title(spec["text"], spec.get("detail", ""))
    elif k == "location":
        img = render_location(spec["text"], spec.get("detail"))
    elif k == "map":
        img = render_places_map(spec) if spec.get("style") == "places" else render_route_map(spec["here"], spec["phase"])
    elif k == "info":
        img = render_info(spec["text"])
    elif k == "subtitle":
        img = render_subtitle(spec["text"], spec.get("translated", False))
    elif k == "credit":
        img = render_credit(spec["text"].split("\n"))
    else:
        raise ValueError(f"unknown overlay kind {k}")
    img.save(path)
    return path
