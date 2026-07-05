#!/usr/bin/env python3
"""
BenchLLAMA — Vision fixture generator (Battery V).

Renders deterministic test images with PIL so the ground truth is EXACT — no
labelling, no licensing, no ambiguity. Writes images + ground_truth.json into
suites/vision/.

Tasks:
  ocr     — transcribe a known text block        → fuzzy ratio vs exact string
  count   — count red circles among distractors   → exact integer
  chart   — read a labelled bar's value           → numeric within tolerance
  spatial — relative position of two shapes        → yes/no keyword
  describe— multi-element scene                     → signal-count (JPEG-style)

  python3 suites/vision/generate.py
"""

import json
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
random.seed(42)  # deterministic layouts


def _font(size):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ── OCR ──────────────────────────────────────────────────────────────────────────
OCR_TEXT = ("Invoice #4827\n"
            "Date: 2026-03-14\n"
            "Item: Mechanical Keyboard\n"
            "Quantity: 3\n"
            "Unit Price: $89.00\n"
            "Total Due: $267.00")

def gen_ocr():
    img = Image.new("RGB", (640, 360), "white")
    d = ImageDraw.Draw(img)
    f = _font(34)
    y = 30
    for line in OCR_TEXT.split("\n"):
        d.text((40, y), line, fill="black", font=f)
        y += 52
    img.save(HERE / "ocr_1.png")
    return {"id": "ocr_1", "type": "ocr", "image": "ocr_1.png",
            "prompt": "Transcribe ALL text in this image exactly, preserving line breaks.",
            "answer": OCR_TEXT, "max_tokens": 300}


# ── COUNT ─────────────────────────────────────────────────────────────────────────
def gen_count():
    n_red = 7
    img = Image.new("RGB", (640, 400), "white")
    d = ImageDraw.Draw(img)
    cells = [(c, r) for r in range(4) for c in range(6)]
    random.shuffle(cells)
    # 7 red circles (target) + distractors: 4 blue circles, 3 green squares
    plan = [("red", "circle")] * n_red + [("blue", "circle")] * 4 + [("green", "square")] * 3
    random.shuffle(plan)
    for (col, row), (color, shape) in zip(cells, plan):
        cx, cy = 60 + col * 100, 60 + row * 95
        box = [cx - 28, cy - 28, cx + 28, cy + 28]
        if shape == "circle":
            d.ellipse(box, fill=color)
        else:
            d.rectangle(box, fill=color)
    img.save(HERE / "count_1.png")
    return {"id": "count_1", "type": "count", "image": "count_1.png",
            "prompt": "How many RED circles are in this image? Reply with just the number.",
            "answer": n_red, "max_tokens": 40}


# ── CHART ─────────────────────────────────────────────────────────────────────────
def gen_chart():
    values = {"A": 25, "B": 60, "C": 40, "D": 80, "E": 15}
    target = "C"
    img = Image.new("RGB", (640, 420), "white")
    d = ImageDraw.Draw(img)
    f = _font(24); ft = _font(20)
    base_y, max_h, bw, gap, x0 = 360, 280, 70, 40, 70
    vmax = max(values.values())
    for i, (lab, val) in enumerate(values.items()):
        x = x0 + i * (bw + gap)
        h = int(val / vmax * max_h)
        d.rectangle([x, base_y - h, x + bw, base_y], fill="#3b6ea5")
        d.text((x + 24, base_y + 8), lab, fill="black", font=f)
        d.text((x + 14, base_y - h - 26), str(val), fill="black", font=ft)
    d.line([x0 - 10, base_y, 600, base_y], fill="black", width=2)
    img.save(HERE / "chart_1.png")
    return {"id": "chart_1", "type": "chart", "image": "chart_1.png",
            "prompt": f"This is a bar chart. What is the value of the bar labelled '{target}'? "
                      f"Reply with just the number.",
            "answer": values[target], "tolerance": 3, "max_tokens": 40}


# ── SPATIAL ───────────────────────────────────────────────────────────────────────
def gen_spatial():
    img = Image.new("RGB", (640, 320), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([80, 110, 200, 230], fill="blue")           # blue square, left
    d.ellipse([440, 110, 560, 230], fill="red")             # red circle, right
    img.save(HERE / "spatial_1.png")
    return {"id": "spatial_1", "type": "spatial", "image": "spatial_1.png",
            "prompt": "Is the blue square to the LEFT of the red circle? Answer yes or no.",
            "answer": "yes", "max_tokens": 40}

def gen_spatial_no():
    # Mirror of spatial_1 with the answer flipped to "no" — paired so a blind model
    # that always guesses "yes" scores 0.5 on spatial, not 1.0. Same question wording.
    img = Image.new("RGB", (640, 320), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([80, 110, 200, 230], fill="red")              # red circle, LEFT
    d.rectangle([440, 110, 560, 230], fill="blue")          # blue square, RIGHT
    img.save(HERE / "spatial_2.png")
    return {"id": "spatial_2", "type": "spatial", "image": "spatial_2.png",
            "prompt": "Is the blue square to the LEFT of the red circle? Answer yes or no.",
            "answer": "no", "max_tokens": 40}

# Six spatial rounds total across THREE relations (left/right, above/below,
# inside/outside) with DIFFERENT shapes+colours each, balanced 3 yes / 3 no.
# A blind guesser's P(all 6 correct) = 0.5^6 ≈ 1.6%; a yes-biased pattern-matcher
# (glm-ocr) can't fluke it, and varied shapes defeat memorised associations.

def _triangle(d, cx, top_y, fill, h=90, w=110):
    d.polygon([(cx, top_y), (cx - w // 2, top_y + h), (cx + w // 2, top_y + h)], fill=fill)

def gen_spatial_above():
    img = Image.new("RGB", (640, 360), "white"); d = ImageDraw.Draw(img)
    _triangle(d, 320, 40, "green")                          # green triangle, TOP
    d.ellipse([270, 230, 370, 330], fill="gold")           # yellow circle, BOTTOM
    img.save(HERE / "spatial_3.png")
    return {"id": "spatial_3", "type": "spatial", "image": "spatial_3.png",
            "prompt": "Is the green triangle ABOVE the yellow circle? Answer yes or no.",
            "answer": "yes", "max_tokens": 40}

def gen_spatial_above_no():
    img = Image.new("RGB", (640, 360), "white"); d = ImageDraw.Draw(img)
    d.ellipse([270, 30, 370, 130], fill="gold")            # yellow circle, TOP
    _triangle(d, 320, 230, "green")                        # green triangle, BOTTOM
    img.save(HERE / "spatial_4.png")
    return {"id": "spatial_4", "type": "spatial", "image": "spatial_4.png",
            "prompt": "Is the green triangle ABOVE the yellow circle? Answer yes or no.",
            "answer": "no", "max_tokens": 40}

def gen_spatial_inside():
    img = Image.new("RGB", (640, 360), "white"); d = ImageDraw.Draw(img)
    d.rectangle([200, 80, 440, 300], outline="black", width=5)   # black rectangle outline
    d.ellipse([290, 150, 350, 230], fill="red")                  # red circle INSIDE
    img.save(HERE / "spatial_5.png")
    return {"id": "spatial_5", "type": "spatial", "image": "spatial_5.png",
            "prompt": "Is the red circle inside the black rectangle? Answer yes or no.",
            "answer": "yes", "max_tokens": 40}

def gen_spatial_inside_no():
    img = Image.new("RGB", (640, 360), "white"); d = ImageDraw.Draw(img)
    d.rectangle([120, 110, 320, 280], outline="black", width=5)  # black rectangle outline
    d.ellipse([480, 160, 560, 240], fill="red")                  # red circle OUTSIDE (right)
    img.save(HERE / "spatial_6.png")
    return {"id": "spatial_6", "type": "spatial", "image": "spatial_6.png",
            "prompt": "Is the red circle inside the black rectangle? Answer yes or no.",
            "answer": "no", "max_tokens": 40}


# ── DESCRIBE ──────────────────────────────────────────────────────────────────────
def gen_describe():
    img = Image.new("RGB", (640, 400), "#f5f5f5")
    d = ImageDraw.Draw(img)
    d.polygon([(120, 250), (60, 350), (180, 350)], fill="green")      # green triangle
    d.ellipse([260, 60, 360, 160], fill="orange")                    # orange circle (sun)
    d.rectangle([420, 240, 560, 350], fill="purple")                 # purple rectangle
    # yellow star
    import math
    cx, cy, R, r = 480, 130, 50, 20
    pts = []
    for k in range(10):
        ang = math.pi / 2 + k * math.pi / 5
        rad = R if k % 2 == 0 else r
        pts.append((cx + rad * math.cos(ang), cy - rad * math.sin(ang)))
    d.polygon(pts, fill="gold")
    d.text((230, 320), "CLOUD", fill="black", font=_font(40))         # the word CLOUD
    img.save(HERE / "describe_1.png")
    return {"id": "describe_1", "type": "describe", "image": "describe_1.png",
            "prompt": "Describe everything you see in this image — shapes, colours, and any text.",
            "signals": {
                "green_triangle": ["green triangle", "triangle"],
                "orange_circle":  ["orange circle", "sun", "orange"],
                "purple_rect":    ["purple", "rectangle", "purple square"],
                "yellow_star":    ["star", "yellow star", "gold"],
                "word_cloud":     ["cloud"],
            },
            "max_tokens": 400}


# ══════════════════════════════════════════════════════════════════════════════════
# V-HARD BAND — the ranking discriminator (weight 0.25). Continuous 0..1 scoring so a
# genuinely-better VLM can rank ABOVE qwen2.5vl:3b (which saturates V-core at 1.00).
# 3 seeded rounds each; grading is proximity / fuzzy-ratio (never binary). See hard-spec.md.
# ══════════════════════════════════════════════════════════════════════════════════

_HSHAPES = ["circle", "triangle", "square"]
_HCOLORS = {"red": "#d33", "blue": "#2a5cd8", "green": "#2a9d2a", "yellow": "#e8b200"}
# Charset excludes visually-ambiguous glyphs (0/O, 1/l/I) so the ground truth stays fair to a human.
_CODE_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789abcdefghijkmnpqrstuvwxyz"


def _rotated_text(text, font, angle, fill="#333"):
    tmp = Image.new("RGBA", (640, 160), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((12, 40), text, font=font, fill=fill)
    tmp = tmp.crop(tmp.getbbox())
    return tmp.rotate(angle, expand=True, resample=Image.BICUBIC)


def gen_count_dense(rng, idx):
    """H1 — attribute-conjunction counting among distractors + overlap + varied size."""
    tshape, tcolor = rng.choice(_HSHAPES), rng.choice(list(_HCOLORS))
    tcount, total = rng.randint(4, 7), rng.randint(20, 26)
    plan = [(tshape, tcolor)] * tcount
    while len(plan) < total:                         # distractors: any combo EXCEPT the target pair
        s, c = rng.choice(_HSHAPES), rng.choice(list(_HCOLORS))
        if (s, c) != (tshape, tcolor):
            plan.append((s, c))
    rng.shuffle(plan)
    img = Image.new("RGB", (700, 470), "white"); d = ImageDraw.Draw(img)
    cells = [(cx, cy) for cy in range(5) for cx in range(7)]; rng.shuffle(cells)
    for (col, row), (s, c) in zip(cells, plan):
        r = rng.randint(20, 34)
        cx, cy = 60 + col * 90 + rng.randint(-15, 15), 58 + row * 88 + rng.randint(-15, 15)
        fill = _HCOLORS[c]
        if s == "circle":   d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
        elif s == "square": d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fill)
        else:               d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fill)
    name = f"count_dense_{idx}.png"; img.save(HERE / name)
    return {"id": f"count_dense_{idx}", "type": "count_dense", "band": "hard", "image": name,
            "prompt": f"How many {tcolor.upper()} {tshape.upper()}S are in this image? Reply with just the number.",
            "answer": tcount, "max_tokens": 40}


def gen_ocr_hard(rng, idx, size, angle, groups, fill="#666"):
    """H2 — small, rotated, LOW-CONTRAST, no-dictionary code (nothing to guess from)."""
    code = "-".join("".join(rng.choice(_CODE_CHARS) for _ in range(g)) for g in groups)
    img = Image.new("RGB", (560, 200), "white")
    layer = _rotated_text(code, _font(size), angle, fill=fill)
    img.paste(layer, (70, 70), layer)
    name = f"ocr_hard_{idx}.png"; img.save(HERE / name)
    return {"id": f"ocr_hard_{idx}", "type": "ocr_hard", "band": "hard", "image": name,
            "prompt": "Transcribe the exact code in this image. It is a random, case-sensitive "
                      "alphanumeric string (letters and digits). Reply with only the code.",
            "answer": code, "max_tokens": 40}


def gen_chart_hard(rng, idx):
    """H3 — grouped bars, NO value labels, SPARSE gridlines (every 20 → must interpolate),
    FOUR series (more disambiguation). Read a target series×category off the axis."""
    cats, series = ["Jan", "Feb", "Mar", "Apr"], ["A", "B", "C", "D"]   # digit-free labels
    scol = {"A": "#3b6ea5", "B": "#e08a3c", "C": "#5a9e5a", "D": "#9b59b6"}
    data = {s: {c: rng.randint(12, 96) for c in cats} for s in series}
    ts, tc = rng.choice(series), rng.choice(cats)
    img = Image.new("RGB", (740, 470), "white"); d = ImageDraw.Draw(img)
    base_y, max_h, x0, vmax = 390, 290, 82, 100
    for v in range(0, 101, 20):                      # SPARSE gridlines (every 20) → interpolation required
        y = base_y - int(v / vmax * max_h)
        d.line([x0, y, 690, y], fill="#dddddd", width=1)
        d.text((x0 - 36, y - 8), str(v), fill="black", font=_font(15))
    d.line([x0, base_y, 690, base_y], fill="black", width=2)
    lx = 120                                          # legend row (top), above the plot
    for s in series:
        d.rectangle([lx, 12, lx + 16, 28], fill=scol[s]); d.text((lx + 22, 12), f"Series {s}", fill="black", font=_font(14)); lx += 128
    for i, cat in enumerate(cats):
        gx = x0 + 20 + i * 150
        for j, s in enumerate(series):
            h = int(data[s][cat] / vmax * max_h)
            bx = gx + j * 30
            d.rectangle([bx, base_y - h, bx + 26, base_y], fill=scol[s])   # NO value label — read the axis
        d.text((gx + 34, base_y + 8), cat, fill="black", font=_font(17))
    name = f"chart_hard_{idx}.png"; img.save(HERE / name)
    return {"id": f"chart_hard_{idx}", "type": "chart_hard", "band": "hard", "image": name,
            "prompt": f"This grouped bar chart has four series (A, B, C, D) across four categories. "
                      f"What is the value of Series {ts} in category {tc}? Read it off the y-axis. "
                      f"Reply with just the number.",
            # proximity denominator 35 (not the full 100 axis) → a sloppy read actually costs score
            "answer": data[ts][tc], "tolerance": 4, "range": 35, "max_tokens": 40}


# ocr_hard / chart_hard above are PARKED — qwen2.5vl:3b aced them even hardened (they're its
# strengths, not weaknesses). Kept for a possible weaker-VLM band; not in the active hard set.
# The active band targets qwen's real soft spots: counting + arithmetic-over-perception.

def gen_count_region(rng, idx):
    """H2 — count shapes of a TARGET COLOUR that are INSIDE the box (colour filter × region filter).
    Same-colour decoys sit OUTSIDE, so the model must apply both filters, not just count a colour."""
    W, H = 700, 460
    box, m = [180, 104, 525, 356], 34
    tcolor = rng.choice(list(_HCOLORS))
    img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
    d.rectangle(box, outline="black", width=5)

    def shape_at(cx, cy, color):
        s = rng.choice(_HSHAPES); r = rng.randint(16, 25); fill = _HCOLORS[color]
        if s == "circle":   d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
        elif s == "square": d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fill)
        else:               d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fill)

    # inside: 3×3 jittered grid (no occlusion). `tin` of the n_in shapes are the target colour (= answer).
    n_in = rng.randint(6, 8)
    tin  = min(rng.randint(2, 5), n_in)
    others = [c for c in _HCOLORS if c != tcolor]
    inside = [tcolor] * tin + [rng.choice(others) for _ in range(n_in - tin)]
    rng.shuffle(inside)
    ix0, iy0 = box[0] + m, box[1] + m
    iw, ih = box[2] - m - ix0, box[3] - m - iy0
    cells = [(ix0 + (c + 0.5) * iw / 3, iy0 + (r + 0.5) * ih / 3) for r in range(3) for c in range(3)]
    rng.shuffle(cells)
    for (cx, cy), col in zip(cells[:n_in], inside):
        shape_at(int(cx + rng.randint(-7, 7)), int(cy + rng.randint(-7, 7)), col)
    # outside: any colour incl. the target → the region filter has to exclude them
    for _ in range(rng.randint(5, 8)):
        while True:
            cx, cy = rng.randint(m, W - m), rng.randint(m, H - m)
            if not (box[0] - m < cx < box[2] + m and box[1] - m < cy < box[3] + m):
                break
        shape_at(cx, cy, rng.choice(list(_HCOLORS)))
    name = f"count_region_{idx}.png"; img.save(HERE / name)
    return {"id": f"count_region_{idx}", "type": "count_region", "band": "hard", "image": name,
            "prompt": f"How many {tcolor.upper()} shapes are INSIDE the black rectangle? Reply with just the number.",
            "answer": tin, "max_tokens": 40}


def gen_table_sum(rng, idx):
    """H3 — read a numeric column off a grid and SUM it (arithmetic over perception)."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]; rng.shuffle(months)
    rows = months[:4]
    cols = ["Revenue", "Cost"]
    data = {c: {mo: rng.randint(20, 90) for mo in rows} for c in cols}
    target = rng.choice(cols)
    total = sum(data[target][mo] for mo in rows)
    img = Image.new("RGB", (560, 330), "white"); d = ImageDraw.Draw(img)
    x0, y0, cw, ch = 55, 35, 150, 52
    headers = ["Month"] + cols
    f, fb = _font(22), _font(23)
    for r in range(5):                                # header + 4 data rows
        for c in range(3):
            x, y = x0 + c * cw, y0 + r * ch
            d.rectangle([x, y, x + cw, y + ch], outline="black", width=2)
            if r == 0:   txt = headers[c]
            elif c == 0: txt = rows[r - 1]
            else:        txt = str(data[cols[c - 1]][rows[r - 1]])
            d.text((x + 16, y + 15), txt, fill="black", font=(fb if r == 0 else f))
    name = f"table_sum_{idx}.png"; img.save(HERE / name)
    return {"id": f"table_sum_{idx}", "type": "table_sum", "band": "hard", "image": name,
            "prompt": f"This table has columns Month, Revenue, Cost. What is the SUM of all four "
                      f"values in the {target} column? Reply with just the total number.",
            "answer": total, "tolerance": 0, "range": 60, "max_tokens": 60}


def main():
    tasks = [gen_ocr(), gen_count(), gen_chart(),
             gen_spatial(), gen_spatial_no(),
             gen_spatial_above(), gen_spatial_above_no(),
             gen_spatial_inside(), gen_spatial_inside_no(),
             gen_describe()]
    # ── V-hard band (3 tasks × 3 seeded rounds) — counting + arithmetic, qwen's soft spots ──
    for i in (1, 2, 3):
        tasks.append(gen_count_dense(random.Random(100 + i), i))
    for i in (1, 2, 3):
        tasks.append(gen_count_region(random.Random(200 + i), i))
    for i in (1, 2, 3):
        tasks.append(gen_table_sum(random.Random(300 + i), i))

    (HERE / "ground_truth.json").write_text(json.dumps({"tasks": tasks}, indent=2))
    core = [t for t in tasks if t.get("band") != "hard"]
    hard = [t for t in tasks if t.get("band") == "hard"]
    print(f"Wrote {len(tasks)} fixtures ({len(core)} core + {len(hard)} hard) + ground_truth.json → {HERE}/")
    for t in tasks:
        ans = t.get("answer", f"{len(t.get('signals', {}))} signals")
        band = " [hard]" if t.get("band") == "hard" else ""
        print(f"  {t['id']:<16} {t['type']:<12} answer={ans!r}{band}")


if __name__ == "__main__":
    main()
