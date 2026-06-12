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


def main():
    tasks = [gen_ocr(), gen_count(), gen_chart(),
             gen_spatial(), gen_spatial_no(),
             gen_spatial_above(), gen_spatial_above_no(),
             gen_spatial_inside(), gen_spatial_inside_no(),
             gen_describe()]
    (HERE / "ground_truth.json").write_text(json.dumps({"tasks": tasks}, indent=2))
    print(f"Wrote {len(tasks)} fixtures + ground_truth.json → {HERE}/")
    for t in tasks:
        ans = t.get("answer", f"{len(t.get('signals', {}))} signals")
        print(f"  {t['id']:<12} {t['type']:<9} answer={ans!r}")


if __name__ == "__main__":
    main()
