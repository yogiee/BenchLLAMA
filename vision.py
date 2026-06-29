#!/usr/bin/env python3
"""
BenchLLAMA — Battery V (Vision)

Capability-routed: runs every model whose `capabilities` include `vision`
(specialists like qwen2.5vl / glm-ocr AND vision-capable workers like qwen3.5,
gemma4). Answers the consolidation question: does a vision specialist earn its
keep over a generalist that also sees?

Five tasks, graded objectively against PIL-generated ground truth:
  ocr     — transcribe a text block   → fuzzy ratio (difflib) vs exact string
  count   — count red circles          → exact integer
  chart   — read a labelled bar value  → numeric within tolerance
  spatial — relative position          → yes/no keyword
  describe— multi-element scene         → signal-count (JPEG-style, judge-free)

Fixtures: suites/vision/*.png + ground_truth.json (run generate.py once).
Protocol: num_ctx=16384, think=False, thermal cool-down between models (GPU-heavy).

Usage:
  python3 vision.py                              # all vision-capable models
  python3 vision.py --models qwen2.5vl:3b         # specific
  python3 vision.py --fast                         # skip cool-down (informal)
  python3 vision.py --force                        # ignore 24h resume window
  python3 vision.py --ollama http://host:11434
"""

import base64
import difflib
import json
import re
import sys
import time
import requests
import numpy as np
from collections import defaultdict
from pathlib import Path
from datetime import date
from bench_utils import cooldown, latest_result, sort_registry

REPO        = Path(__file__).parent
RESULTS_DIR = REPO / "results"
DATA_DIR    = REPO / "suites" / "vision"
STATUS_FILE = RESULTS_DIR / "status.json"
RESULTS_DIR.mkdir(exist_ok=True)

# ── CLI ─────────────────────────────────────────────────────────────────────────

def _flag(name): return name in sys.argv
def _arg(name, default=None):
    if name in sys.argv:
        idx = sys.argv.index(name)
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--"):
            return sys.argv[idx + 1]
    return default

fast_mode   = _flag("--fast")
force       = _flag("--force")
ollama_host = _arg("--ollama", "http://localhost:11434")
model_args  = []
if "--models" in sys.argv:
    idx = sys.argv.index("--models")
    model_args = [a for a in sys.argv[idx + 1:] if not a.startswith("--")]

NUM_CTX  = 16384
TIMEOUT  = 480
COOLDOWN = 0 if fast_mode else 300
OCR_PASS = 0.80   # fuzzy-ratio threshold for OCR pass

SYS_PROMPT = "You are a precise visual analysis assistant. Look carefully at the image and answer."

# ── Model selection (by capability) ──────────────────────────────────────────────

def load_models_by_cap(cap):
    path = REPO / "models.json"
    if not path.exists():
        sys.exit(f"models.json not found at {path} — run update_registry.py first")
    return [(m["name"], m.get("disk_gb", 0.0))
            for m in sort_registry(json.load(path.open()))   # run order: env BENCH_SORT (default size)
            if cap in m.get("capabilities", [])]

# ── Fixtures ─────────────────────────────────────────────────────────────────────

def load_tasks():
    gt = DATA_DIR / "ground_truth.json"
    if not gt.exists():
        sys.exit(f"{gt} missing — run: python3 suites/vision/generate.py")
    return json.load(gt.open())["tasks"]

def _b64(image_name):
    return base64.b64encode((DATA_DIR / image_name).read_bytes()).decode()

# ── Ollama vision chat ───────────────────────────────────────────────────────────

def chat_image(model, prompt, image_b64, max_tokens):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt, "images": [image_b64]},
        ],
        "stream": False,
        "options": {"num_ctx": NUM_CTX, "num_predict": max_tokens},
        "think": False,
    }
    t0 = time.time()
    r = requests.post(f"{ollama_host}/api/chat", json=payload, timeout=TIMEOUT)
    if r.status_code == 400 and "think" in payload:
        payload.pop("think")
        t0 = time.time()
        r = requests.post(f"{ollama_host}/api/chat", json=payload, timeout=TIMEOUT)
    wall = time.time() - t0
    r.raise_for_status()
    return r.json(), wall

def tps(data):
    ec, ed = data.get("eval_count", 0), data.get("eval_duration", 1)
    return round(ec / (ed / 1e9), 1) if ec and ed else None

def load_s(data):
    ld = data.get("load_duration", 0)
    return round(ld / 1e9, 1) if ld else None

# ── Scorers (objective) ──────────────────────────────────────────────────────────

def _norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()

def _numbers(text):
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))]

def score_ocr(resp, task):
    ratio = difflib.SequenceMatcher(None, _norm(task["answer"]), _norm(resp)).ratio()
    return {"ratio": round(ratio, 3), "correct": ratio >= OCR_PASS}

def score_count(resp, task):
    nums = [int(n) for n in _numbers(resp)]
    return {"answer": task["answer"], "found": nums[:8],
            "correct": task["answer"] in nums}

def score_chart(resp, task):
    tol = task.get("tolerance", 0)
    nums = _numbers(resp)
    ok = any(abs(n - task["answer"]) <= tol for n in nums)
    return {"answer": task["answer"], "found": nums[:8], "correct": ok}

def score_spatial(resp, task):
    toks = set(re.findall(r"[a-z]+", resp.lower()))
    want = task["answer"].lower()   # "yes"
    other = "no" if want == "yes" else "yes"
    if want in toks and other not in toks:
        correct = True
    elif other in toks and want not in toks:
        correct = False
    else:
        correct = None  # ambiguous / no clear yes-no
    return {"answer": want, "correct": correct}

def score_describe(resp, task):
    lo = resp.lower()
    signals = {k: any(syn in lo for syn in syns) for k, syns in task["signals"].items()}
    score = sum(signals.values()); mx = len(signals)
    return {"signals": signals, "score": score, "max": mx,
            "correct": score >= int(round(mx * 0.6))}

SCORERS = {"ocr": score_ocr, "count": score_count, "chart": score_chart,
           "spatial": score_spatial, "describe": score_describe}

def _task_unit(tid, res):
    """Map a task result to [0,1] for the composite."""
    if tid == "ocr":      return res["ratio"]
    if tid == "describe": return res["score"] / res["max"] if res["max"] else 0.0
    c = res.get("correct")
    return 1.0 if c else (0.0 if c is False else 0.5)

# ── Per-model runner ─────────────────────────────────────────────────────────────

def run_model(model_name, disk_gb, tasks):
    print(f"\n{'='*60}\nMODEL: {model_name}  ({disk_gb} GB disk)  [vision]\n{'='*60}", flush=True)
    result = {"model": model_name, "disk_gb": disk_gb, "tests": {}, "errors": []}
    tps_samples, load = [], None
    dim_units = defaultdict(list)   # group task units by type → one dimension each
    for task in tasks:
        tid = task["id"]; ttype = task["type"]
        print(f"  [{tid}]", end=" ", flush=True)
        try:
            data, wall = chat_image(model_name, task["prompt"], _b64(task["image"]),
                                    task.get("max_tokens", 300))
            resp = data.get("message", {}).get("content", "")
            res  = SCORERS[ttype](resp, task)
            res["tps"] = tps(data); res["wall_s"] = round(wall, 1)
            res["response"] = resp[:500]
            result["tests"][tid] = res
            if res["tps"]: tps_samples.append(res["tps"])
            if load is None: load = load_s(data)
            dim_units[ttype].append(_task_unit(ttype, res))
            mark = res.get("correct")
            mk = "✓" if mark else ("✗" if mark is False else "·")
            extra = f"ratio={res['ratio']}" if ttype == "ocr" else \
                    (f"{res['score']}/{res['max']}" if ttype == "describe" else "")
            print(f"{mk} {extra}  tps={res['tps']} wall={wall:.1f}s", flush=True)
            print(f"    → {resp[:90].replace(chr(10),' ')}", flush=True)
        except Exception as e:
            print(f"FAILED: {e}", flush=True)
            result["tests"][tid] = {"error": str(e), "correct": False}
            dim_units[ttype].append(0.0)

    result["load_s"]  = load
    result["avg_tps"] = round(sum(tps_samples) / len(tps_samples), 1) if tps_samples else None
    # Dimension-weighted: each task TYPE is one dimension (spatial = mean of its
    # rounds), so adding spatial rounds raises reliability without inflating weight.
    dims = {d: round(float(np.mean(u)), 4) for d, u in dim_units.items()}
    result["dimensions"] = dims
    result["composite"]  = round(float(np.mean(list(dims.values()))), 4) if dims else 0.0
    print(f"\n  composite={result['composite']}  dims={dims}  avg_tps={result['avg_tps']}  load={result['load_s']}s", flush=True)
    return result

# ── Status + unload ──────────────────────────────────────────────────────────────

def _ws(model, phase):
    try:
        STATUS_FILE.write_text(json.dumps({"model": model, "phase": phase, "ts": time.time()}))
    except Exception:
        pass

def unload(model):
    try:
        requests.post(f"{ollama_host}/api/chat",
                      json={"model": model, "messages": [], "keep_alive": 0}, timeout=15)
    except Exception:
        pass

# ── Summary ──────────────────────────────────────────────────────────────────────

def _cell(t, tid):
    r = t.get(tid, {})
    if "error" in r: return "ERR"
    if tid == "ocr_1":      return f"{r.get('ratio','?')}"
    if tid == "describe_1": return f"{r.get('score','?')}/{r.get('max','?')}"
    c = r.get("correct")
    return "✓" if c else ("✗" if c is False else "?")

def _spatial_cell(t):
    sp = [v for k, v in t.items() if k.startswith("spatial")]
    if not sp: return "—"
    passed = sum(1 for v in sp if v.get("correct") is True)
    return f"{passed}/{len(sp)}"

def write_summary(results, out_md, fast_mode):
    flag = " ⚠ FAST (no cool-down)" if fast_mode else ""
    lines = [
        f"# Vision Battery (V) — {out_md.stem}{flag}", "",
        "Objective vision benchmark vs PIL ground truth. Specialists and "
        "vision-capable workers compared head-to-head.", "",
        "## Scoreboard", "",
        "| Model | Disk | avg tok/s | OCR (ratio) | Count | Chart | Spatial | Describe | **Composite** |",
        "|-------|-----:|----------:|------------:|:-----:|:-----:|:-------:|:--------:|------:|",
    ]
    for r in sorted(results, key=lambda x: x.get("composite", 0), reverse=True):
        t = r.get("tests", {})
        lines.append(
            f"| `{r['model']}` | {r['disk_gb']}GB | {r.get('avg_tps','?')} "
            f"| {_cell(t,'ocr_1')} | {_cell(t,'count_1')} | {_cell(t,'chart_1')} "
            f"| {_spatial_cell(t)} | {_cell(t,'describe_1')} | **{r.get('composite','?')}** |")
    lines += ["", "## Responses", ""]
    for r in results:
        lines += [f"### `{r['model']}` (composite {r.get('composite','?')})", ""]
        for tid, res in r.get("tests", {}).items():
            resp = res.get("response", res.get("error", "—"))
            lines += [f"**{tid}** → {resp}", ""]
        lines += ["---", ""]
    out_md.write_text("\n".join(lines))
    print(f"MD → {out_md}", flush=True)

# ── Entrypoint ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TODAY    = date.today().isoformat()
    suffix   = "_fast" if fast_mode else ""
    OUT_JSON = RESULTS_DIR / f"vision_{TODAY}{suffix}.json"
    OUT_MD   = RESULTS_DIR / f"vision_{TODAY}{suffix}.md"

    tasks = load_tasks()
    if model_args:
        reg = {m["name"]: m for m in json.load((REPO / "models.json").open())}
        MODELS = [(m, reg.get(m, {}).get("disk_gb", 0.0)) for m in model_args]
    else:
        MODELS = load_models_by_cap("vision")

    if not MODELS:
        sys.exit("No vision-capable models found. Run update_registry.py, or pass --models <name>.")

    # (No completion-suite preflight here — 'tools' capability is irrelevant to
    # vision; missing models surface as per-task errors during the run.)
    print(f"BenchLLAMA Battery V{' [FAST]' if fast_mode else ''} — {TODAY}", flush=True)
    print(f"ollama={ollama_host} | {len(tasks)} tasks | models: {[m[0] for m in MODELS]}", flush=True)
    print(f"Output: {OUT_JSON}", flush=True)

    # Resume SOURCE = today's file if present, else most recent vision within 24h
    # (cross-day). Writes today's file, carrying prior results forward.
    all_results, completed = [], set()
    source = OUT_JSON if OUT_JSON.exists() else (None if force else latest_result(RESULTS_DIR, "vision", fast_mode, 24))
    if source is not None and not force:
        try:
            loaded      = json.load(source.open())
            all_results = [r for r in loaded if not r.get("errors")]   # retry replaces, no dupes
            completed   = {r["model"] for r in all_results}
            if completed:
                via = "" if source == OUT_JSON else f" (carried from {source.name})"
                print(f"  Resuming — done{via}: {sorted(completed)}", flush=True)
        except Exception:
            all_results, completed = [], set()

    first = True
    for model_name, disk_gb in MODELS:
        if model_name in completed:
            print(f"  ↷ {model_name} — already done, skipping", flush=True)
            continue
        if not first:
            _ws(model_name, "cooldown")
            cooldown(COOLDOWN, label="after previous model")
        first = False
        _ws(model_name, "running")
        try:
            r = run_model(model_name, disk_gb, tasks)
        except Exception as e:
            print(f"  ✗ {model_name} FAILED: {e}", flush=True)
            r = {"model": model_name, "disk_gb": disk_gb, "errors": [str(e)], "tests": {}}
        all_results = [x for x in all_results if x["model"] != model_name] + [r]
        OUT_JSON.write_text(json.dumps(all_results, indent=2))
        try:
            import results_db; results_db.record_all("vision", all_results)
        except Exception:
            pass
        unload(model_name)
        time.sleep(3)

    _ws("", "done")
    write_summary(all_results, OUT_MD, fast_mode)
    print(f"\n{'='*60}\nDONE\nJSON → {OUT_JSON}\nMD   → {OUT_MD}", flush=True)
