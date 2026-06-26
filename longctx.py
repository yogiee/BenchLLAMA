#!/usr/bin/env python3
"""
BenchLLAMA — Battery G: Long-Context Retrieval & Degradation.

The GraphWalks analog (inspired by disler/live-bench), BenchLLAMA-style: fill the context to
controlled token depths with plausible distractor text, plant verifiable NEEDLES at known
positions, and measure both ACCURACY degradation and the prefill/decode SPEED collapse as the
window fills. Distinct from Battery C's C4 (which re-runs a SHORT prompt at bigger num_ctx
*allocations* and never fills the window) — here the window is genuinely loaded, so we measure
whether the model can still reason over N filled tokens, and how slow prefill gets there.

Per depth bucket (1 call): 3 single-needle retrievals at early/mid/late positions (the
"lost in the middle" probe) + 1 three-hop manage-chain walk. All objective, exact-match graded.

Capability/role: COMPLETION models (worker + router), like Battery E — selected by completion
capability, NOT a role gate. `utility` specialists (embedding/vision/OCR) are skipped.

  python3 longctx.py                          # all completion models (resumes within 24h)
  python3 longctx.py gemma4:12b llama3.2:3b   # specific models (merge into existing JSON)
  python3 longctx.py --role worker            # filter by role
  python3 longctx.py --capable-only           # skip models that failed `calculate` in the latest standard run
  python3 longctx.py --fast                    # skip inter-model cool-down
  python3 longctx.py --force                    # ignore the 24h resume window
  python3 longctx.py --ollama http://host:11434

Deep (32768) bucket: run `python3 suites/longctx/build.py --deep` first to add it to the dataset.
"""

import json
import sys
import time
import requests
from pathlib import Path
from datetime import date
from bench_utils import cooldown, preflight, latest_result

REPO        = Path(__file__).parent
RESULTS_DIR = REPO / "results"
PROMPTS_DIR = REPO / "prompts"
DATASET     = REPO / "suites" / "longctx" / "dataset.json"
STATUS_FILE = RESULTS_DIR / "status.json"

RESULTS_DIR.mkdir(exist_ok=True)

# ── CLI ───────────────────────────────────────────────────────────────────────

def _flag(name):
    return name in sys.argv

def _arg(name, default=None):
    if name in sys.argv:
        idx = sys.argv.index(name)
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--"):
            return sys.argv[idx + 1]
    return default

fast_mode     = _flag("--fast")
force         = _flag("--force")
capable_only  = _flag("--capable-only")
ollama_host   = _arg("--ollama", "http://localhost:11434")
role_filter   = _arg("--role")
model_args    = [a for a in sys.argv[1:] if not a.startswith("--")
                 and a not in (ollama_host, role_filter)]

TIMEOUT     = 600
COOLDOWN    = 0 if fast_mode else 300
NUM_PREDICT = 320
CTX_MARGIN  = 1536   # num_ctx = bucket + margin → holds our own prompt + the short answer
COMPLETION_ROLES = ("worker", "router")

# ── Grading (objective, exact-match) ──────────────────────────────────────────

def grade(response, key):
    lo = response.lower()
    hits = {
        "needle_early": key["needle_early"] in response,
        "needle_mid":   key["needle_mid"]   in response,
        "needle_late":  key["needle_late"]  in response,
        "multihop":     key["multihop"].lower() in lo,
    }
    score = sum(hits.values())
    return {"hits": hits, "found": score, "max": 4, "accuracy": round(score / 4, 3)}

# ── Ollama helpers ────────────────────────────────────────────────────────────

def chat(model, messages, num_ctx, max_tokens=NUM_PREDICT):
    payload = {
        "model":    model,
        "messages": messages,
        "stream":   False,
        "options":  {"num_ctx": num_ctx, "num_predict": max_tokens},
        "think":    False,
    }
    t0 = time.time()
    r  = requests.post(f"{ollama_host}/api/chat", json=payload, timeout=TIMEOUT)
    if r.status_code == 400 and "think" in payload:
        payload.pop("think")
        t0 = time.time()
        r  = requests.post(f"{ollama_host}/api/chat", json=payload, timeout=TIMEOUT)
    wall = time.time() - t0
    r.raise_for_status()
    return r.json(), wall

def tps(data):
    ec = data.get("eval_count", 0); ed = data.get("eval_duration", 1)
    return round(ec / (ed / 1e9), 1) if ec and ed else None

def prefill_tps(data):
    pc = data.get("prompt_eval_count", 0); pd = data.get("prompt_eval_duration", 0)
    return round(pc / (pd / 1e9), 1) if pc and pd else None

def unload(model_name):
    try:
        requests.post(f"{ollama_host}/api/chat",
                      json={"model": model_name, "messages": [], "keep_alive": 0}, timeout=15)
    except Exception:
        pass

def _ws(model, phase, bucket=None):
    try:
        payload = {"segment": "longctx", "model": model, "phase": phase, "ts": time.time()}
        if bucket is not None:
            payload["bucket"] = bucket
        STATUS_FILE.write_text(json.dumps(payload))
    except Exception:
        pass

# ── Per-model battery ─────────────────────────────────────────────────────────

def run_longctx(model_name, role, disk_gb, items):
    prompt_file = "router_default.md" if role == "router" else "worker_default.md"
    sys_prompt  = (PROMPTS_DIR / prompt_file).read_text()
    sys_msgs    = [{"role": "system", "content": sys_prompt}]

    print(f"\n{'='*60}", flush=True)
    print(f"MODEL: {model_name}  role={role}  buckets: {[it['bucket'] for it in items]}", flush=True)
    print("=" * 60, flush=True)

    depths = {}
    for it in items:
        bucket  = it["bucket"]
        num_ctx = bucket + CTX_MARGIN
        _ws(model_name, "running", bucket=bucket)
        print(f"\n  ── depth={bucket} (num_ctx={num_ctx}) ──", flush=True)
        try:
            data, wall = chat(model_name,
                              sys_msgs + [{"role": "user", "content": it["prompt"]}],
                              num_ctx=num_ctx)
            resp = data.get("message", {}).get("content", "")
            g    = grade(resp, it["answer_key"])
            entry = {
                "accuracy":     g["accuracy"],
                "found":        g["found"],
                "hits":         g["hits"],
                "prefill_tps":  prefill_tps(data),
                "decode_tps":   tps(data),
                "wall_s":       round(wall, 1),
                "prompt_tokens": data.get("prompt_eval_count"),
                "response":     resp,
            }
            depths[str(bucket)] = entry
            ok = [k.replace("needle_", "").replace("multihop", "hop") for k, v in g["hits"].items() if v]
            print(f"    acc={g['accuracy']}  found={g['found']}/4 [{', '.join(ok) or '—'}]"
                  f"  prefill={entry['prefill_tps']}  tps={entry['decode_tps']}"
                  f"  tok={entry['prompt_tokens']}  wall={wall:.1f}s", flush=True)
        except Exception as e:
            print(f"    FAILED: {e}", flush=True)
            depths[str(bucket)] = {"error": str(e)}
        unload(model_name)
        time.sleep(2)

    return {"model": model_name, "role": role, "disk_gb": disk_gb,
            "depths": depths, "summary": summarize(depths)}

# ── Summary (export-friendly) ─────────────────────────────────────────────────

def summarize(depths):
    graded = {int(b): e for b, e in depths.items() if "accuracy" in e}
    if not graded:
        return {"composite": None, "clean_depth": None, "n_depths": 0}
    buckets = sorted(graded)
    threshold = json.loads(DATASET.read_text())["meta"]["threshold"]

    acc_by   = {b: graded[b]["accuracy"]    for b in buckets}
    pre_by   = {b: graded[b]["prefill_tps"] for b in buckets}
    dec_by   = {b: graded[b]["decode_tps"]  for b in buckets}
    wall_by  = {b: graded[b]["wall_s"]      for b in buckets}
    tok_by   = {b: graded[b]["prompt_tokens"] for b in buckets}

    # deepest bucket still at/above the accuracy bar (contiguous from the shallow end)
    clean = None
    for b in buckets:
        if acc_by[b] >= threshold:
            clean = b
        else:
            break

    # position recall across depths — exposes "lost in the middle"
    pos = {"early": 0, "mid": 0, "late": 0, "hop": 0}
    for b in buckets:
        h = graded[b]["hits"]
        pos["early"] += int(h["needle_early"]); pos["mid"] += int(h["needle_mid"])
        pos["late"]  += int(h["needle_late"]);  pos["hop"] += int(h["multihop"])
    n = len(buckets)
    position_recall = {k: round(v / n, 3) for k, v in pos.items()}

    shallow, deep = buckets[0], buckets[-1]
    pre_collapse = (round(pre_by[deep] / pre_by[shallow], 3)
                    if pre_by.get(shallow) and pre_by.get(deep) else None)

    return {
        "composite":      round(sum(acc_by.values()) / n, 3),   # mean accuracy across depths
        "clean_depth":    clean,                                 # the headline practical number
        "threshold":      threshold,
        "n_depths":       n,
        "accuracy_by_depth": acc_by,
        "prefill_by_depth":  pre_by,
        "decode_by_depth":   dec_by,
        "wall_by_depth":     wall_by,
        "tokens_by_depth":   tok_by,
        "position_recall":   position_recall,
        "prefill_collapse":  pre_collapse,   # prefill tok/s at deepest ÷ shallowest (<1 = slowdown)
    }

# ── Markdown ──────────────────────────────────────────────────────────────────

def write_summary(results, out_md, fast_mode=False):
    flag = " ⚠ FAST MODE" if fast_mode else ""
    meta = json.loads(DATASET.read_text())["meta"]
    lines = [
        f"# Battery G — Long-Context Retrieval — {out_md.stem}{flag}", "",
        "Fill the window to each token depth, plant needles (early/mid/late) + a 3-hop manage-chain, "
        "grade exact-match. Measures accuracy degradation **and** prefill/decode speed collapse as the "
        "context fills — the window-fill the C4 ctx_depth probe never does.", "",
        f"Buckets: {meta['buckets']} · clean = deepest depth ≥ {meta['threshold']} accuracy "
        "(3/4 sub-tasks) · collapse = prefill t/s at deepest ÷ shallowest.", "",
        "| Model | Role | Disk | Composite | Clean depth | Prefill collapse | early | mid | late | hop |",
        "|-------|------|-----:|----------:|------------:|-----------------:|:----:|:---:|:----:|:---:|",
    ]
    for r in sorted(results, key=lambda r: (r["summary"].get("clean_depth") or 0,
                                            r["summary"].get("composite") or 0), reverse=True):
        s  = r["summary"]
        pr = s.get("position_recall", {})
        lines.append(
            f"| `{r['model']}` | {r.get('role','')} | {r.get('disk_gb','?')}GB"
            f" | {s.get('composite','?')} | {s.get('clean_depth','—') or '—'}"
            f" | {s.get('prefill_collapse','—') or '—'}"
            f" | {pr.get('early','?')} | {pr.get('mid','?')} | {pr.get('late','?')} | {pr.get('hop','?')} |"
        )

    lines += ["", "## Accuracy × depth", "",
              "| Model | " + " | ".join(str(b) for b in meta["buckets"]) + " |",
              "|-------|" + "|".join("----:" for _ in meta["buckets"]) + "|"]
    for r in results:
        acc = r["summary"].get("accuracy_by_depth", {})
        row = " | ".join(str(acc.get(b, "—")) for b in meta["buckets"])
        lines.append(f"| `{r['model']}` | {row} |")

    lines += ["", "## Prefill tok/s × depth (speed collapse)", "",
              "| Model | " + " | ".join(str(b) for b in meta["buckets"]) + " |",
              "|-------|" + "|".join("----:" for _ in meta["buckets"]) + "|"]
    for r in results:
        pre = r["summary"].get("prefill_by_depth", {})
        row = " | ".join(str(pre.get(b, "—")) for b in meta["buckets"])
        lines.append(f"| `{r['model']}` | {row} |")

    out_md.write_text("\n".join(lines))
    print(f"MD → {out_md}", flush=True)

# ── Capable-only gate (reuse the latest standard run's `calculate` result) ─────

def _capable_models():
    src = latest_result(RESULTS_DIR, "benchmark", False, 72)
    if not src:
        return None
    try:
        data = json.load(src.open())
        return {r["model"] for r in data
                if r.get("tests", {}).get("calculate", {}).get("correct")}
    except Exception:
        return None

# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TODAY  = date.today().isoformat()
    suffix = "_fast" if fast_mode else ""

    if not DATASET.exists():
        sys.exit(f"dataset not found — run: python3 {DATASET.relative_to(REPO)}".replace("dataset.json", "build.py"))
    items = json.loads(DATASET.read_text())["items"]

    reg_path = REPO / "models.json"
    if not reg_path.exists():
        sys.exit("models.json not found — run update_registry.py first")
    registry = json.load(reg_path.open())

    if model_args:
        reg_map = {m["name"]: m for m in registry}
        MODELS = [(m, reg_map.get(m, {}).get("disk_gb", 0.0), reg_map.get(m, {}).get("role", "worker"))
                  for m in model_args]
    else:
        MODELS = [(m["name"], m["disk_gb"], m["role"]) for m in registry
                  if m.get("role") in COMPLETION_ROLES]

    if role_filter:
        MODELS = [(n, d, r) for n, d, r in MODELS if r == role_filter]

    if capable_only and not model_args:
        cap = _capable_models()
        if cap is not None:
            before = len(MODELS)
            MODELS = [(n, d, r) for n, d, r in MODELS if n in cap]
            print(f"  --capable-only: {before} → {len(MODELS)} models (passed `calculate`)", flush=True)

    if not MODELS:
        sys.exit("No models to test. Check models.json / --role / --capable-only.")

    preflight(MODELS, ollama_host)

    flag = " [FAST MODE]" if fast_mode else ""
    print(f"BenchLLAMA Battery G — long-context retrieval{flag} — {TODAY}", flush=True)
    print(f"ollama={ollama_host} | think=False | {len(MODELS)} model(s) | "
          f"buckets={[it['bucket'] for it in items]}", flush=True)

    OUT_JSON = RESULTS_DIR / f"longctx_{TODAY}{suffix}.json"
    OUT_MD   = RESULTS_DIR / f"longctx_{TODAY}{suffix}.md"
    print(f"Output: {OUT_JSON}\n", flush=True)

    # ── Resume / merge (24h window, same semantics as the aptitude batteries) ──
    all_results, completed = [], set()
    source = None
    if not (force and not model_args):
        source = OUT_JSON if OUT_JSON.exists() else latest_result(RESULTS_DIR, "longctx", fast_mode, 24)
    if source is not None:
        try:
            existing = json.load(source.open())
            if model_args:
                targets = {m for m, *_ in MODELS}
                all_results = [r for r in existing if r["model"] not in targets]
            else:
                all_results = [r for r in existing if r.get("summary", {}).get("n_depths")]
                completed   = {r["model"] for r in all_results}
                if completed:
                    via = "" if source == OUT_JSON else f" (carried from {source.name})"
                    print(f"  Resuming — {len(completed)} model(s) already done{via}: {sorted(completed)}", flush=True)
        except Exception:
            all_results, completed = [], set()

    first_run = True
    for model_name, disk_gb, role in MODELS:
        if model_name in completed:
            print(f"  ↷ {model_name} — already done, skipping", flush=True)
            continue
        if not first_run:
            _ws(model_name, "cooldown")
            cooldown(COOLDOWN, label="after previous model")
        first_run = False
        r = run_longctx(model_name, role, disk_gb, items)
        all_results.append(r)
        OUT_JSON.write_text(json.dumps(all_results, indent=2))

    _ws("", "done")
    write_summary(all_results, OUT_MD, fast_mode)
    print(f"\n{'='*60}\nDONE\nJSON → {OUT_JSON}\nMD   → {OUT_MD}", flush=True)
