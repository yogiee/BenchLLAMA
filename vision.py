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

# ── V-hard scorers (continuous 0..1 — proximity, so a better VLM ranks above the ceiling) ──

def score_count_dense(resp, task):
    """H1 — conjunction counting. Proximity: exact = 1.0, off-by-one on 12 ≈ 0.92."""
    nums = [int(n) for n in _numbers(resp)]
    pred = nums[0] if nums else None
    true = task["answer"]
    prox = max(0.0, 1 - abs(pred - true) / true) if (pred is not None and true) else 0.0
    return {"answer": true, "pred": pred, "prox": round(prox, 3), "correct": pred == true}

def score_chart_hard(resp, task):
    """H3 — read a bar off the gridlines. Proximity within the axis range; digit-free labels."""
    nums = _numbers(resp)
    true = task["answer"]; rng = task.get("range") or max(true, 1)
    pred = nums[0] if nums else None
    prox = max(0.0, 1 - abs(pred - true) / rng) if pred is not None else 0.0
    return {"answer": true, "pred": pred, "prox": round(prox, 3),
            "correct": pred is not None and abs(pred - true) <= task.get("tolerance", 0)}

def score_table_sum(resp, task):
    """H3 — read a column + sum it. The sum of positive cells is the LARGEST number in a correct
    response ("45 + 60 + 32 + 50 = 187"), so take max(); proximity credits near-misses."""
    nums = _numbers(resp)
    true = task["answer"]; rng = task.get("range") or max(true * 0.3, 1)
    pred = max(nums) if nums else None
    prox = max(0.0, 1 - abs(pred - true) / rng) if pred is not None else 0.0
    return {"answer": true, "pred": pred, "prox": round(prox, 3),
            "correct": pred is not None and abs(pred - true) <= task.get("tolerance", 0)}

# ocr_hard/chart_hard scorers stay registered (harmless) but no active task uses them.
SCORERS = {"ocr": score_ocr, "count": score_count, "chart": score_chart,
           "spatial": score_spatial, "describe": score_describe,
           "count_dense": score_count_dense, "count_region": score_count_dense,
           "table_sum": score_table_sum, "ocr_hard": score_ocr, "chart_hard": score_chart_hard}

def _task_unit(tid, res):
    """Map a task result to [0,1] for the composite."""
    if tid in ("ocr", "ocr_hard"):                                  return res["ratio"]
    if tid in ("count_dense", "count_region", "chart_hard", "table_sum"): return res["prox"]
    if tid == "describe":                                           return res["score"] / res["max"] if res["max"] else 0.0
    c = res.get("correct")
    return 1.0 if c else (0.0 if c is False else 0.5)

# ── Per-model runner ─────────────────────────────────────────────────────────────

def run_model(model_name, disk_gb, tasks):
    print(f"\n{'='*60}\nMODEL: {model_name}  ({disk_gb} GB disk)  [vision]\n{'='*60}", flush=True)
    result = {"model": model_name, "disk_gb": disk_gb, "tests": {}, "errors": []}
    tps_samples, load = [], None
    core_units, hard_units = defaultdict(list), defaultdict(list)   # group by type → one dimension each, split by band
    for task in tasks:
        tid = task["id"]; ttype = task["type"]
        units = hard_units if task.get("band") == "hard" else core_units
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
            units[ttype].append(_task_unit(ttype, res))
            mark = res.get("correct")
            mk = "✓" if mark else ("✗" if mark is False else "·")
            if ttype in ("ocr", "ocr_hard"):           extra = f"ratio={res['ratio']}"
            elif ttype in ("count_dense", "count_region", "chart_hard", "table_sum"):
                                                       extra = f"pred={res['pred']}/{res['answer']} prox={res['prox']}"
            elif ttype == "describe":                  extra = f"{res['score']}/{res['max']}"
            else:                                      extra = ""
            print(f"{mk} {extra}  tps={res['tps']} wall={wall:.1f}s", flush=True)
            print(f"    → {resp[:90].replace(chr(10),' ')}", flush=True)
        except Exception as e:
            print(f"FAILED: {e}", flush=True)
            result["tests"][tid] = {"error": str(e), "correct": False}
            units[ttype].append(0.0)

    result["load_s"]  = load
    result["avg_tps"] = round(sum(tps_samples) / len(tps_samples), 1) if tps_samples else None
    # Two-band, like Battery E: each task TYPE is one dimension. V-core = the `sees?` gate baseline;
    # V-hard = the ranking discriminator (weight 0.25). composite = 0.75·core + 0.25·hard.
    core_dims = {d: round(float(np.mean(u)), 4) for d, u in core_units.items()}
    hard_dims = {d: round(float(np.mean(u)), 4) for d, u in hard_units.items()}
    comp_core = round(float(np.mean(list(core_dims.values()))), 4) if core_dims else 0.0
    comp_hard = round(float(np.mean(list(hard_dims.values()))), 4) if hard_dims else None
    result["dimensions"]     = {**core_dims, **hard_dims}
    result["composite_core"] = comp_core
    result["composite_hard"] = comp_hard
    result["composite"]      = round(0.75 * comp_core + 0.25 * comp_hard, 4) if comp_hard is not None else comp_core
    hs = f"  hard={comp_hard} {hard_dims}" if comp_hard is not None else ""
    print(f"\n  composite={result['composite']}  core={comp_core} {core_dims}{hs}  avg_tps={result['avg_tps']}  load={result['load_s']}s", flush=True)
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
    # Eligible universe = all vision-capable models (independent of --models, so carry-forward stays
    # complete); --models handled by resume as explicit targets.
    reg = {m["name"]: m for m in json.load((REPO / "models.json").open())}
    MODELS = load_models_by_cap("vision")
    if model_args:
        have = {n for n, *_ in MODELS}
        for m in model_args:
            if m not in have:
                MODELS.append((m, reg.get(m, {}).get("disk_gb", 0.0)))

    if not MODELS:
        sys.exit("No vision-capable models found. Run update_registry.py, or pass --models <name>.")

    # (No completion-suite preflight here — 'tools' capability is irrelevant to
    # vision; missing models surface as per-task errors during the run.)
    print(f"BenchLLAMA Battery V{' [FAST]' if fast_mode else ''} — {TODAY}", flush=True)
    print(f"ollama={ollama_host} | {len(tasks)} tasks | models: {[m[0] for m in MODELS]}", flush=True)
    print(f"Output: {OUT_JSON}", flush=True)

    # ── Content-addressed resume (docs/resume-spec.md): skip unchanged models, carry their prior
    #    result forward from the DB (lossless). No time window. ──
    import resume
    eligible = [n for n, *_ in MODELS]
    cloud = {n for n, m in reg.items() if m.get("cloud")}
    run_names, all_results, why = resume.plan_single_pass(
        "vision", eligible, host=ollama_host, cloud=cloud, force=force,
        explicit_models=(model_args or None), check_runtime="--check-runtime" in sys.argv)
    all_results = list(all_results)
    completed = set(eligible) - set(run_names)
    print("  " + resume.format_report("vision", run_names, sorted(completed), why).replace("\n", "\n  "), flush=True)

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
