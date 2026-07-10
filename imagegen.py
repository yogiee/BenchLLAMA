#!/usr/bin/env python3
"""
BenchLLAMA — Battery I (Image-Gen Characterization)

Capability-routed by the `image` capability (update_registry.py admits image-gen
models to the 'utility' lane; imagegen.py is the one consumer that selects by `image`).

Characterises local image-generation models on PERFORMANCE + prompt-ADHERENCE —
explicitly NOT quality (image quality is subjective; even the best VLM misreads
detail a human catches, so we never grade "is it good?"). Reference-only: image
models never enter the primary ranking lanes.

Per model: 5 prompts × N images (default 3, seeds logged) via POST /api/generate.
  • Perf   — load_duration / total_duration (→ gen_s) + /api/ps size_vram. Median over images.
  • Adherence — a VLM (qwen2.5vl:3b) answers yes/no/unsure per ✓-core element (drives adherence%);
                ⚠-fine elements are advisory only → the human-spot-check list.
  • Text   — fidelity (P3/P4): OCR (minicpm-v4.6:1b + qwen2.5vl:3b cross-check), fuzzy-match to the
             exact target string. legibility (P5): is there readable Latin/CJK text, no target.
The grader is blind to which model produced the image (sees image + checklist only).

Usage:
  python3 imagegen.py                              # all image-cap models
  python3 imagegen.py --models x/z-image-turbo:latest
  python3 imagegen.py --images 1                   # fewer images/prompt (faster / informal)
  python3 imagegen.py --prompts P1,P3              # subset of prompts
  python3 imagegen.py --smoke                      # 1 image, prompts P1+P3, first model only
  python3 imagegen.py --retries 3                  # transient-failure retries per image gen (default 2)
  python3 imagegen.py --fast                       # skip cool-down
  python3 imagegen.py --force                      # ignore 24h resume window
  python3 imagegen.py --ollama http://host:11434

NEVER in `all` by default (opt-in, reference-only). Rides along a full run via
`./bench.sh all --with-imagegen`.
"""

import base64
import difflib
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import requests

from bench_utils import cooldown, latest_result, sort_registry

REPO        = Path(__file__).parent
RESULTS_DIR = REPO / "results"
DATA_DIR    = REPO / "suites" / "imagegen"
IMG_DIR     = RESULTS_DIR / "imagegen_images"
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
smoke       = _flag("--smoke")
ollama_host = _arg("--ollama", "http://localhost:11434")
N_IMAGES    = int(_arg("--images", "1" if smoke else "3"))
prompt_filter = None
if smoke:
    prompt_filter = {"P1", "P3"}
elif _arg("--prompts"):
    prompt_filter = {p.strip().upper() for p in _arg("--prompts").split(",")}
model_args  = []
if "--models" in sys.argv:
    idx = sys.argv.index("--models")
    model_args = [a for a in sys.argv[idx + 1:] if not a.startswith("--")]

GRADER      = "qwen2.5vl:3b"      # V-battery champion (1.00); blind adherence + legibility + OCR cross-check
OCR_MODEL   = "minicpm-v4.6:1b"   # fast primary OCR for text fidelity
NUM_CTX     = 16384
GEN_TIMEOUT = 600                 # image gen is slow (flux slower than z-image)
GRADE_TIMEOUT = 300
COOLDOWN    = 0 if fast_mode else 300
FIDELITY_PASS = 0.80              # difflib ratio → text-fidelity "hit"
GEN_RETRIES   = int(_arg("--retries", "2"))   # transient-failure retries per image gen (unattended-run safety)
GRADE_RETRIES = 1                              # light retry for the VLM/OCR grading calls

# ── Model selection (by `image` capability) ──────────────────────────────────────

def load_models_by_cap(cap):
    path = REPO / "models.json"
    if not path.exists():
        sys.exit(f"models.json not found at {path} — run update_registry.py first")
    return [(m["name"], m.get("disk_gb", 0.0))
            for m in sort_registry(json.load(path.open()))
            if cap in m.get("capabilities", [])]

def load_prompts():
    p = DATA_DIR / "prompts.json"
    if not p.exists():
        sys.exit(f"{p} missing — the imagegen dataset is required.")
    data = json.load(p.open())
    prompts = data["prompts"]
    if prompt_filter:
        prompts = [q for q in prompts if q["id"] in prompt_filter]
    return prompts

# ── Ollama calls ─────────────────────────────────────────────────────────────────

def _generate_once(model, prompt, seed):
    """A single POST /api/generate → (img_b64, load_s, total_s, gen_s). Raises on failure.
    Module-level (not a closure) so the retry/recovery path in generate_image is unit-testable."""
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"seed": seed}}
    r = requests.post(f"{ollama_host}/api/generate", json=payload, timeout=GEN_TIMEOUT)
    r.raise_for_status()
    d = r.json()
    img = d.get("image") or (d.get("images") or [None])[0]
    if not img:
        raise RuntimeError(f"no image in response (keys={sorted(d)})")
    load_s  = round(d.get("load_duration", 0) / 1e9, 2)
    total_s = round(d.get("total_duration", 0) / 1e9, 2)
    gen_s   = round(max(total_s - load_s, 0.0), 2)
    return img, load_s, total_s, gen_s

def generate_image(model, prompt, seed):
    """→ (img_b64, load_s, total_s, gen_s, attempts, recovered).

    Normal attempts (1 initial + GEN_RETRIES) use a linear backoff. If ALL of them fail, one
    SPECIAL recovery attempt fires: unload the model (`ollama stop` = keep_alive:0, clears a
    wedged / OOM VRAM state), wait for it to actually leave VRAM, then reload fresh (the cold
    generate reloads it) and try one final time. Raises the last error only if the recovery
    attempt ALSO fails → the caller logs the failure and moves on. `recovered` flags a success
    that only happened because of the unload+reload."""
    last = None
    for attempt in range(1, GEN_RETRIES + 2):   # 1 initial + GEN_RETRIES retries
        try:
            img, load_s, total_s, gen_s = _generate_once(model, prompt, seed)
            return img, load_s, total_s, gen_s, attempt, False
        except Exception as e:
            last = e
            if attempt <= GEN_RETRIES:
                print(f"      ⟳ gen attempt {attempt} failed ({e}); retrying…", flush=True)
                time.sleep(min(5 * attempt, 15))

    # ── special recovery attempt: unload → wait for VRAM to free → reload + one final try ──
    print(f"      ⚕ {GEN_RETRIES + 1} attempts failed ({last}); unloading + reloading {model} "
          f"for a final attempt…", flush=True)
    unload(model)
    _wait_unloaded(model)
    img, load_s, total_s, gen_s = _generate_once(model, prompt, seed)   # cold reload+generate; raises → caller logs + moves on
    print("      ✓ recovered after unload+reload", flush=True)
    return img, load_s, total_s, gen_s, GEN_RETRIES + 2, True

def ps_stats(model):
    """/api/ps → (vram_gb, ram_gb) for the loaded model, best-effort. (None, None) if not loaded.
    size_vram = GPU/Metal residency; size = total resident (equal on unified-memory Macs,
    distinct on a discrete-GPU box)."""
    try:
        d = requests.get(f"{ollama_host}/api/ps", timeout=10).json()
        for m in d.get("models", []):
            if m.get("name") == model or m.get("model") == model:
                sv, sz = m.get("size_vram") or 0, m.get("size") or 0
                return (round(sv / 1e9, 2) if sv else None,
                        round(sz / 1e9, 2) if sz else None)
    except Exception:
        pass
    return (None, None)

def _wait_unloaded(model, timeout=15):
    """Poll /api/ps until the model has left VRAM (bounded) so the reload is genuinely cold."""
    for _ in range(int(timeout)):
        if ps_stats(model)[0] is None:   # no longer resident
            return True
        time.sleep(1)
    return False

def show_spec(model):
    """/api/show → 'params · quant' string for the reference table."""
    try:
        d = requests.post(f"{ollama_host}/api/show", json={"model": model}, timeout=15).json()
        det = d.get("details", {})
        return f"{det.get('parameter_size','?')} · {det.get('quantization_level','?')}"
    except Exception:
        return "?"

def vlm(model, prompt, image_b64, num_predict=400):
    """Vision chat (same plumbing as vision.py). Returns the message text.
    Light retry (GRADE_RETRIES) so a transient grading blip doesn't shrink the sample."""
    payload = {"model": model,
               "messages": [{"role": "system", "content": "You are a precise visual analysis assistant. Look carefully and answer exactly as instructed."},
                            {"role": "user", "content": prompt, "images": [image_b64]}],
               "stream": False, "options": {"num_ctx": NUM_CTX, "num_predict": num_predict}, "think": False}
    last = None
    for attempt in range(1, GRADE_RETRIES + 2):
        try:
            r = requests.post(f"{ollama_host}/api/chat", json=payload, timeout=GRADE_TIMEOUT)
            if r.status_code == 400 and "think" in payload:
                payload.pop("think")
                r = requests.post(f"{ollama_host}/api/chat", json=payload, timeout=GRADE_TIMEOUT)
            r.raise_for_status()
            return r.json().get("message", {}).get("content", "")
        except Exception as e:
            last = e
            if attempt <= GRADE_RETRIES:
                time.sleep(3)
    raise last

# ── Graders ──────────────────────────────────────────────────────────────────────

def _checklist(items):
    lines = "\n".join(f"{i+1}. {c}" for i, c in enumerate(items))
    return ("Look at the image. For EACH numbered item below, decide whether it is clearly present.\n"
            "Answer with exactly one line per item in the form `N: yes`, `N: no`, or `N: unsure`.\n"
            "Say `yes` only if it is clearly there. Do not add any other text.\n\n" + lines)

def _parse_verdicts(text, n):
    v = {}
    for m in re.finditer(r"(\d+)\s*[:.\)]\s*(yes|no|unsure)", text, re.I):
        idx = int(m.group(1))
        if 1 <= idx <= n:
            v[idx] = m.group(2).lower()
    return v

def grade_checklist(image_b64, core, fine):
    """One VLM call grades core + fine together. Returns core%/verdicts + advisory fine verdicts."""
    items = core + fine
    out = vlm(GRADER, _checklist(items), image_b64, num_predict=40 + 20 * len(items))
    v = _parse_verdicts(out, len(items))
    core_v = {core[i]: v.get(i + 1, "unsure") for i in range(len(core))}
    fine_v = {fine[i]: v.get(len(core) + i + 1, "unsure") for i in range(len(fine))}
    yes = sum(1 for x in core_v.values() if x == "yes")
    return {"core_pct": round(yes / len(core), 3) if core else None,
            "core": core_v, "fine": fine_v, "raw": out[:400]}

def _norm(s): return re.sub(r"\s+", " ", s.lower()).strip()

def grade_text_fidelity(image_b64, target):
    """OCR with the fast model + VLM cross-check; best difflib ratio vs the exact target."""
    texts = {}
    for m in (OCR_MODEL, GRADER):
        try:
            texts[m] = vlm(m, "Transcribe EXACTLY any text that appears in this image. "
                              "Output only the transcribed text, nothing else.", image_b64, num_predict=120)
        except Exception as e:
            texts[m] = f"[error: {e}]"
    ratios = {m: round(difflib.SequenceMatcher(None, _norm(target), _norm(t)).ratio(), 3)
              for m, t in texts.items()}
    best = max(ratios.values()) if ratios else 0.0
    return {"target": target, "best_ratio": best, "ratios": ratios,
            "texts": {m: t[:120] for m, t in texts.items()}, "hit": best >= FIDELITY_PASS}

def grade_legibility(image_b64):
    """Is there human-readable text in a real script? yes/no + quote (no target)."""
    out = vlm(GRADER, "Is there any human-readable text — real words in Latin or CJK script, "
                      "NOT random gibberish — visible in this image (e.g. on signs or billboards)? "
                      "Answer 'yes' or 'no' on the first line, then quote any legible text.", image_b64, 160)
    first = out.strip().splitlines()[0].lower() if out.strip() else ""
    legible = "yes" in first and "no" not in first[:4]
    return {"legible": legible, "raw": out[:200]}

# ── Per-model runner ─────────────────────────────────────────────────────────────

def _slug(name): return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")

def run_model(model_name, disk_gb, prompts, today):
    print(f"\n{'='*60}\nMODEL: {model_name}  ({disk_gb} GB disk)  [image]\n{'='*60}", flush=True)
    spec = show_spec(model_name)
    out_dir = IMG_DIR / today / _slug(model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"model": model_name, "disk_gb": disk_gb, "spec": spec,
              "n_images": N_IMAGES, "prompts": {}, "errors": []}
    load_samples, gen_samples, total_samples = [], [], []
    vram_samples, ram_samples = [], []
    gen_ok = gen_fail = retries_total = grade_errors = recoveries = 0   # reliability counters
    spotcheck = []   # (prompt_id, image_file, reason)

    for q in prompts:
        pid = q["id"]
        print(f"  [{pid}] {q['name']}  ×{N_IMAGES}", flush=True)
        imgs = []
        core_pcts, fid_ratios, legible_votes = [], [], []
        for n in range(N_IMAGES):
            seed = n
            try:
                b64, load_s, total_s, gen_s, attempts, recovered = generate_image(model_name, q["prompt"], seed)
            except Exception as e:
                gen_fail += 1
                retries_total += GEN_RETRIES + 1      # all normal retries + the recovery attempt exhausted
                print(f"      img{n} GEN FAILED after {GEN_RETRIES+2} attempt(s): {e}", flush=True)
                result["errors"].append(f"{pid} img{n}: {e}")
                continue
            gen_ok += 1
            if recovered: recoveries += 1
            retries_total += attempts - 1
            fn = out_dir / f"{pid}_{n}.png"
            fn.write_bytes(base64.b64decode(b64))
            vram, ram = ps_stats(model_name)
            load_samples.append(load_s); gen_samples.append(gen_s); total_samples.append(total_s)
            if vram: vram_samples.append(vram)
            if ram:  ram_samples.append(ram)

            rec = {"seed": seed, "file": str(fn.relative_to(RESULTS_DIR)),
                   "attempts": attempts, "recovered": recovered,
                   "load_s": load_s, "gen_s": gen_s, "total_s": total_s,
                   "vram_gb": vram, "ram_gb": ram}
            # adherence (core + advisory fine)
            try:
                ck = grade_checklist(b64, q["core"], q["fine"])
                rec["adherence"] = ck
                if ck["core_pct"] is not None: core_pcts.append(ck["core_pct"])
                for el, verd in ck["fine"].items():
                    if verd != "yes":
                        spotcheck.append((pid, fn.name, f"fine unsure/no: {el} → {verd}"))
                for el, verd in ck["core"].items():
                    if verd == "unsure":
                        spotcheck.append((pid, fn.name, f"core UNSURE: {el}"))
            except Exception as e:
                rec["adherence"] = {"error": str(e)}; grade_errors += 1
            # text
            mode = q.get("text_mode")
            if mode == "fidelity":
                try:
                    tf = grade_text_fidelity(b64, q["text_target"])
                    rec["text"] = tf; fid_ratios.append(tf["best_ratio"])
                    if not tf["hit"]:
                        spotcheck.append((pid, fn.name, f"text fidelity {tf['best_ratio']} < {FIDELITY_PASS} (target {q['text_target']!r})"))
                except Exception as e:
                    rec["text"] = {"error": str(e)}; grade_errors += 1
            elif mode == "legibility":
                try:
                    lg = grade_legibility(b64)
                    rec["text"] = lg; legible_votes.append(1 if lg["legible"] else 0)
                    if not lg["legible"]:
                        spotcheck.append((pid, fn.name, "P5 text not legible (verify by eye)"))
                except Exception as e:
                    rec["text"] = {"error": str(e)}; grade_errors += 1
            imgs.append(rec)
            adh = rec.get("adherence", {}).get("core_pct")
            note = "  ⚕recovered" if recovered else (f"  (⟳{attempts-1})" if attempts > 1 else "")
            print(f"      img{n} seed={seed}  gen={gen_s}s  adherence={adh}  vram={vram}GB ram={ram}GB{note}", flush=True)

        pr = {"name": q["name"], "kind": q["kind"], "expect_lead": q.get("expect_lead"),
              "images": imgs,
              "adherence_pct": round(statistics.mean(core_pcts), 3) if core_pcts else None}
        if fid_ratios:
            pr["text_fidelity"] = round(statistics.mean(fid_ratios), 3)
        if legible_votes:
            pr["text_legibility"] = round(statistics.mean(legible_votes), 3)
        result["prompts"][pid] = pr

    # ── aggregate (median perf; adherence split by kind) ──
    def med(xs): return round(statistics.median(xs), 2) if xs else None
    result["load_s"]      = med(load_samples)
    result["gen_s"]       = med(gen_samples)
    result["s_per_image"] = med(total_samples)
    result["vram_gb"]     = med(vram_samples)
    result["ram_gb"]      = med(ram_samples)

    # ── reliability (speed already above) ──
    attempted = gen_ok + gen_fail
    result["images_ok"]     = gen_ok
    result["images_failed"] = gen_fail
    result["error_rate"]    = round(gen_fail / attempted, 3) if attempted else None
    result["gen_retries"]   = retries_total          # extra generation attempts beyond the first
    result["recoveries"]    = recoveries             # images that only succeeded via unload+reload
    result["grade_errors"]  = grade_errors           # VLM/OCR grading calls that failed after retry

    prs = result["prompts"]
    def mean_over(kinds, key="adherence_pct"):
        vals = [p[key] for p in prs.values() if p.get("kind") in kinds and p.get(key) is not None]
        return round(statistics.mean(vals), 3) if vals else None
    result["adherence_overall"]  = mean_over({"photoreal", "design", "neutral"})
    result["adherence_photoreal"] = mean_over({"photoreal"})
    result["adherence_design"]    = mean_over({"design"})
    fids = [p["text_fidelity"] for p in prs.values() if "text_fidelity" in p]
    result["text_fidelity"]  = round(statistics.mean(fids), 3) if fids else None
    legs = [p["text_legibility"] for p in prs.values() if "text_legibility" in p]
    result["text_legibility"] = round(statistics.mean(legs), 3) if legs else None
    result["spotcheck"] = [f"{p} · {f} · {r}" for p, f, r in spotcheck]

    print(f"\n  adherence: overall={result['adherence_overall']} "
          f"photoreal={result['adherence_photoreal']} design={result['adherence_design']} "
          f"| text_fid={result['text_fidelity']} legible={result['text_legibility']}", flush=True)
    print(f"  perf: load={result['load_s']}s  s/img={result['s_per_image']}s  "
          f"vram={result['vram_gb']}GB ram={result['ram_gb']}GB", flush=True)
    print(f"  reliability: {gen_ok} ok / {gen_fail} failed  error_rate={result['error_rate']}  "
          f"gen_retries={retries_total}  recoveries={recoveries}  grade_errors={grade_errors}  "
          f"| {len(result['spotcheck'])} spot-check item(s)", flush=True)
    return result

# ── Status + unload ──────────────────────────────────────────────────────────────

def _ws(model, phase):
    try:
        STATUS_FILE.write_text(json.dumps({"model": model, "phase": phase, "ts": time.time()}))
    except Exception:
        pass

def unload(model):
    # keep_alive:0 with NO prompt → unload only; never risk a real (~minute-long) generation.
    try:
        requests.post(f"{ollama_host}/api/generate",
                      json={"model": model, "keep_alive": 0}, timeout=15)
    except Exception:
        pass

# ── Summary (reference-only) ──────────────────────────────────────────────────────

def _pct(x): return f"{round(x*100)}%" if isinstance(x, (int, float)) else "—"

def write_summary(results, out_md, fast_mode):
    flag = " ⚠ FAST (no cool-down)" if fast_mode else ""
    L = [f"# Image-Gen Battery (I) — {out_md.stem}{flag}", "",
         "**Reference only — NOT ranked.** Image models are characterised on *performance* and "
         "*prompt-adherence*, never quality. Perf is **not apples-to-apples** (each model uses its own "
         f"default steps/resolution — turbo *being* faster is the point). {N_IMAGES} image(s)/prompt.", "",
         "## Performance", "",
         "| Model | Spec | Disk | load s | s/image | VRAM | RAM |",
         "|-------|------|-----:|-------:|--------:|-----:|----:|"]
    for r in results:
        L.append(f"| `{r['model']}` | {r.get('spec','?')} | {r.get('disk_gb','?')}GB "
                 f"| {r.get('load_s','—')} | {r.get('s_per_image','—')} "
                 f"| {r.get('vram_gb','—')}GB | {r.get('ram_gb','—')}GB |")

    L += ["", "## Reliability", "",
          "_Generation retries on transient failure; if all fail, one recovery attempt unloads+reloads the "
          "model before a final try. A still-failing image is dropped from the medians and counted here. "
          "`recoveries` = images that only succeeded via unload+reload; `grade_errors` = VLM/OCR grading "
          "calls that failed after retry._", "",
          "| Model | images ok | failed | error-rate | gen retries | recoveries | grade errors |",
          "|-------|:---------:|:------:|:----------:|:-----------:|:----------:|:------------:|"]
    for r in results:
        L.append(f"| `{r['model']}` | {r.get('images_ok','—')} | {r.get('images_failed','—')} "
                 f"| {_pct(r.get('error_rate'))} | {r.get('gen_retries','—')} "
                 f"| {r.get('recoveries','—')} | {r.get('grade_errors','—')} |")

    L += ["", "## Prompt adherence  (✓-core captured, VLM-graded — blind)", "",
          "| Model | Overall | Photoreal (P1/P2) | Design (P3/P4) | Text fidelity (P3/P4) | Legibility (P5) |",
          "|-------|:-------:|:-----------------:|:--------------:|:---------------------:|:---------------:|"]
    for r in results:
        L.append(f"| `{r['model']}` | {_pct(r.get('adherence_overall'))} "
                 f"| {_pct(r.get('adherence_photoreal'))} | {_pct(r.get('adherence_design'))} "
                 f"| {_pct(r.get('text_fidelity'))} | {_pct(r.get('text_legibility'))} |")

    # hypothesis verdict (cross-model)
    L += ["", "## Hypothesis", "",
          "Spec predicts photoreal (P1/P2) favours a photoreal model, design+text (P3/P4) favours a "
          "design/text model. Observed leaders:"]
    def leader(key):
        cand = [(r["model"], r.get(key)) for r in results if r.get(key) is not None]
        return max(cand, key=lambda x: x[1]) if cand else (None, None)
    for label, key in [("Photoreal (adherence)", "adherence_photoreal"),
                       ("Design (adherence)", "adherence_design"),
                       ("Text fidelity (P3/P4)", "text_fidelity")]:
        m, v = leader(key)
        L.append(f"- **{label}:** {('`'+m+'` ' + _pct(v)) if m else '—'}")

    # spot-check
    L += ["", "## Human spot-check list", "",
          "_VLM is a first-pass filter, never the verdict. Every ⚠-fine miss, core `unsure`, and "
          "text-fidelity/legibility flag is surfaced here for your eye._", ""]
    any_sc = False
    for r in results:
        sc = r.get("spotcheck", [])
        if sc:
            any_sc = True
            L.append(f"### `{r['model']}`  (images: `results/{IMG_DIR.name}/{TODAY}/{_slug(r['model'])}/`)")
            L += [f"- {s}" for s in sc] + [""]
    if not any_sc:
        L.append("_(nothing flagged)_")
    out_md.write_text("\n".join(L))
    print(f"MD → {out_md}", flush=True)

# ── Entrypoint ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TODAY    = date.today().isoformat()
    suffix   = "_fast" if fast_mode else ""
    OUT_JSON = RESULTS_DIR / f"imagegen_{TODAY}{suffix}.json"
    OUT_MD   = RESULTS_DIR / f"imagegen_{TODAY}{suffix}.md"

    prompts = load_prompts()
    if model_args:
        reg = {m["name"]: m for m in json.load((REPO / "models.json").open())}
        MODELS = [(m, reg.get(m, {}).get("disk_gb", 0.0)) for m in model_args]
    else:
        MODELS = load_models_by_cap("image")
    if smoke:
        MODELS = MODELS[:1]

    if not MODELS:
        sys.exit("No image-capable models found. Run update_registry.py (image lane), or pass --models <name>.")

    print(f"BenchLLAMA Battery I (image-gen){' [FAST]' if fast_mode else ''} — {TODAY}", flush=True)
    print(f"ollama={ollama_host} | grader={GRADER} ocr={OCR_MODEL} | {len(prompts)} prompt(s) × {N_IMAGES} img | "
          f"models: {[m[0] for m in MODELS]}", flush=True)
    print(f"Output: {OUT_JSON}", flush=True)

    all_results, completed = [], set()
    source = OUT_JSON if OUT_JSON.exists() else (None if force else latest_result(RESULTS_DIR, "imagegen", fast_mode, 24))
    if source is not None and not force:
        try:
            loaded      = json.load(source.open())
            all_results = [r for r in loaded if not r.get("errors")]
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
            r = run_model(model_name, disk_gb, prompts, TODAY)
        except Exception as e:
            print(f"  ✗ {model_name} FAILED: {e}", flush=True)
            r = {"model": model_name, "disk_gb": disk_gb, "errors": [str(e)], "prompts": {}}
        all_results = [x for x in all_results if x["model"] != model_name] + [r]
        OUT_JSON.write_text(json.dumps(all_results, indent=2))
        try:
            import results_db; results_db.record_all("image", all_results, only={n for n, _ in MODELS} - completed)
        except Exception:
            pass
        unload(model_name)
        time.sleep(3)

    _ws("", "done")
    write_summary(all_results, OUT_MD, fast_mode)
    print(f"\n{'='*60}\nDONE\nJSON → {OUT_JSON}\nMD   → {OUT_MD}", flush=True)
