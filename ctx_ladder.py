#!/usr/bin/env python3
"""
BenchLLAMA — num_ctx Ladder
Characterizes how RAM, speed, and quality change across context window sizes.
Run before aptitude batteries to find the optimal num_ctx per model/role.

Results are used to set per-model ctx values in Battery A, C, and D runs
instead of using the shared 16384 invariant from the standard suite.

Anchor tests (fixed across all ctx levels):
  cylinder  — multi-step reasoning, objective pass/fail (all roles)
  jpeg      — research depth, 7-signal coverage score (worker role only)

Ladder ranges per role:
  router : 2048 / 4096 / 8192 / 16384
  worker : 4096 / 8192 / 16384 / 32768

Usage:
  python3 ctx_ladder.py                            # all models (resumes within 72h)
  python3 ctx_ladder.py qwen3.5:4b-mlx             # specific models (merges into existing JSON)
  python3 ctx_ladder.py --role router              # filter by role
  python3 ctx_ladder.py --fast                     # skip inter-model cool-down
  python3 ctx_ladder.py --force                    # ignore resume window, overwrite results
  python3 ctx_ladder.py --ollama http://host:11434
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

fast_mode   = _flag("--fast")
force       = _flag("--force")
ollama_host = _arg("--ollama", "http://localhost:11434")
role_filter = _arg("--role")
model_args  = [a for a in sys.argv[1:] if not a.startswith("--")
               and a not in (ollama_host, role_filter)]

TIMEOUT  = 480
COOLDOWN = 0 if fast_mode else 300

# ── Ladder config per role ────────────────────────────────────────────────────

CTX_LADDER = {
    "router": [2048, 4096, 8192, 16384],
    "worker": [4096, 8192, 16384, 32768],
}

# ── Anchor tests ──────────────────────────────────────────────────────────────

CYLINDER_PROMPT = (
    "A cylindrical water tank has a diameter of 3 metres and is 4 metres tall. "
    "The tank is currently 60% full. Water is being pumped out at 15 litres per minute. "
    "How long will it take to completely empty the tank? "
    "Give the answer in hours and minutes. Think step by step."
)

def _cylinder_check(r):
    return (
        any(x in r for x in ["1130", "1131", "18 hour", "18h", "18 hr"])
        and any(x in r for x in ["51", "50.9"])
    )

JPEG_PROMPT = (
    "Give a concise, not-too-technical but detailed comparison of "
    "JPEG, JPEG-2000, and JPEG-XL formats."
)

def _jpeg_coverage(r):
    lo = r.lower()
    signals = {
        "dct_block":    any(x in lo for x in ["dct", "discrete cosine", "8×8", "8x8", "block"]),
        "wavelet":      "wavelet" in lo,
        "lossless":     "lossless" in lo,
        "transparency": any(x in lo for x in ["transparency", "alpha", "transparent"]),
        "browser":      any(x in lo for x in ["browser", "chrome", "firefox", "safari"]),
        "niche_use":    any(x in lo for x in ["medical", "dicom", "cinema", "archiv"]),
        "recompress":   any(x in lo for x in ["recompress", "transcode", "re-encode",
                                               "existing jpeg", "losslessly transc",
                                               "backward compat"]),
    }
    score = sum(signals.values())
    return {"signals": signals, "score": score, "max": 7, "pass": score >= 4}

# ── Ollama helpers ────────────────────────────────────────────────────────────

def chat(model, messages, num_ctx, max_tokens=None):
    payload = {
        "model":    model,
        "messages": messages,
        "stream":   False,
        "options":  {"num_ctx": num_ctx},
        "think":    False,
    }
    if max_tokens:
        payload["options"]["num_predict"] = max_tokens
    t0 = time.time()
    r  = requests.post(f"{ollama_host}/api/chat", json=payload, timeout=TIMEOUT)
    if r.status_code == 400 and "think" in payload:
        print(f"\n  ⚠  {model}: think parameter rejected (400) — retrying without it", flush=True)
        payload.pop("think")
        t0 = time.time()
        r  = requests.post(f"{ollama_host}/api/chat", json=payload, timeout=TIMEOUT)
    wall = time.time() - t0
    r.raise_for_status()
    return r.json(), wall

def tps(data):
    """Decode (generation) tok/s."""
    ec = data.get("eval_count", 0)
    ed = data.get("eval_duration", 1)
    return round(ec / (ed / 1e9), 1) if ec and ed else None

def prefill_tps(data):
    """Prefill (prompt-processing) tok/s — input read speed before the first output token."""
    pc = data.get("prompt_eval_count", 0)
    pd = data.get("prompt_eval_duration", 0)
    return round(pc / (pd / 1e9), 1) if pc and pd else None

def load_s(data):
    ld = data.get("load_duration", 0)
    return round(ld / 1e9, 1) if ld else None

def get_ram_gb(model_name):
    try:
        r = requests.get(f"{ollama_host}/api/ps", timeout=10)
        for m in r.json().get("models", []):
            if m["name"] == model_name:
                return round(m["size_vram"] / 1e9, 1)
    except Exception:
        pass
    return None

def unload(model_name):
    try:
        requests.post(
            f"{ollama_host}/api/chat",
            json={"model": model_name, "messages": [], "keep_alive": 0},
            timeout=15,
        )
    except Exception:
        pass

def _ws(model, phase, ctx=None):
    try:
        payload = {"segment": "ctx_ladder", "model": model, "phase": phase, "ts": time.time()}
        if ctx is not None:
            payload["ctx"] = ctx
        STATUS_FILE.write_text(json.dumps(payload))
    except Exception:
        pass

# ── Per-model ladder ──────────────────────────────────────────────────────────

def run_ctx_ladder(model_name, role, disk_gb, ctx_levels):
    prompt_file = "router_default.md" if role == "router" else "worker_default.md"
    sys_prompt  = (PROMPTS_DIR / prompt_file).read_text()
    sys_msgs    = [{"role": "system", "content": sys_prompt}]

    print(f"\n{'='*60}", flush=True)
    print(f"MODEL: {model_name}  role={role}  ctx: {ctx_levels}", flush=True)
    print("=" * 60, flush=True)

    result = {
        "model":    model_name,
        "role":     role,
        "disk_gb":  disk_gb,
        "levels":   {},
    }

    for num_ctx in ctx_levels:
        _ws(model_name, "running", ctx=num_ctx)
        print(f"\n  ── ctx={num_ctx} ──", flush=True)
        entry = {}

        # Warmup — captures load time at this ctx
        print(f"  [warmup]", end=" ", flush=True)
        try:
            data, _ = chat(model_name,
                           sys_msgs + [{"role": "user", "content": "Ready."}],
                           num_ctx=num_ctx, max_tokens=50)
            entry["load_s"] = load_s(data)
            time.sleep(1)
            entry["ram_gb"] = get_ram_gb(model_name)
            print(f"load={entry['load_s']}s  ram={entry['ram_gb']}GB", flush=True)
        except Exception as e:
            print(f"FAILED: {e}", flush=True)
            entry["error"] = str(e)
            result["levels"][num_ctx] = entry
            unload(model_name)
            time.sleep(2)
            continue

        # Cylinder — reasoning anchor (all roles)
        print(f"  [cylinder]", end=" ", flush=True)
        try:
            data, wall = chat(model_name,
                              sys_msgs + [{"role": "user", "content": CYLINDER_PROMPT}],
                              num_ctx=num_ctx, max_tokens=800)
            resp    = data.get("message", {}).get("content", "")
            t       = tps(data)
            pf      = prefill_tps(data)
            correct = _cylinder_check(resp)
            entry["cylinder"] = {
                "correct":     correct,
                "tps":         t,
                "prefill_tps": pf,
                "wall_s":      round(wall, 1),
                "response":    resp,
            }
            print(f"tps={t}  prefill={pf}  wall={wall:.1f}s  {'✓' if correct else '✗'}", flush=True)
            print(f"    → {resp[:120].replace(chr(10), ' ')}", flush=True)
        except Exception as e:
            print(f"FAILED: {e}", flush=True)
            entry["cylinder"] = {"error": str(e), "correct": False}

        # JPEG — research depth anchor (worker only)
        if role == "worker":
            print(f"  [jpeg]", end=" ", flush=True)
            try:
                data, wall = chat(model_name,
                                  sys_msgs + [{"role": "user", "content": JPEG_PROMPT}],
                                  num_ctx=num_ctx, max_tokens=1500)
                resp = data.get("message", {}).get("content", "")
                cov  = _jpeg_coverage(resp)
                t    = tps(data)
                pf   = prefill_tps(data)
                entry["jpeg"] = {
                    "score":       cov["score"],
                    "pass":        cov["pass"],
                    "signals":     {k: v for k, v in cov["signals"].items() if v},
                    "tps":         t,
                    "prefill_tps": pf,
                    "wall_s":      round(wall, 1),
                    "response":    resp,
                }
                print(f"coverage={cov['score']}/7  {'✓' if cov['pass'] else '✗'}  tps={t}  wall={wall:.1f}s", flush=True)
            except Exception as e:
                print(f"FAILED: {e}", flush=True)
                entry["jpeg"] = {"error": str(e)}

        result["levels"][num_ctx] = entry

        # Unload between levels — forces fresh KV allocation at next ctx
        unload(model_name)
        time.sleep(2)

    return result

# ── Markdown summary ──────────────────────────────────────────────────────────

def write_summary(results, out_md, fast_mode=False):
    flag = " ⚠ FAST MODE" if fast_mode else ""
    lines = [
        f"# ctx Ladder Results — {out_md.stem}{flag}", "",
        "Anchors: cylinder (reasoning · objective) · jpeg (research depth · workers only)",
        "Unload between levels — fresh KV allocation at each ctx.", "",
    ]

    for r in results:
        model = r["model"]
        role  = r["role"]
        levels = sorted(r["levels"].keys())
        has_jpeg = role == "worker"

        lines += [f"## `{model}` ({role}, {r['disk_gb']} GB)", ""]

        # Ladder summary table
        hdr = "| num_ctx | RAM | Load (s) | Prefill t/s | Decode t/s | Cylinder |"
        sep = "|--------:|----:|:--------:|------------:|-----------:|:--------:|"
        if has_jpeg:
            hdr += " JPEG |"
            sep += ":----:|"
        lines += [hdr, sep]

        for ctx in levels:
            e = r["levels"][ctx]
            if "error" in e:
                row = f"| {ctx} | — | — | — | — | ERROR |"
                if has_jpeg:
                    row += " — |"
                lines.append(row)
                continue

            ram  = f"{e.get('ram_gb', '?')} GB"
            ld   = f"{e.get('load_s', '?')}s"
            cyl  = e.get("cylinder", {})
            t    = cyl.get("tps", "?")
            pf   = cyl.get("prefill_tps", "?")
            mark = "✓" if cyl.get("correct") else ("✗" if "correct" in cyl else "?")
            row  = f"| {ctx:>7} | {ram:>7} | {ld:>8} | {str(pf):>11} | {str(t):>10} | {mark:^8} |"
            if has_jpeg:
                j     = e.get("jpeg", {})
                j_str = f"{j['score']}/7" if "score" in j else ("ERR" if "error" in j else "?")
                row  += f" {j_str:^6}|"
            lines.append(row)

        lines.append("")

        # Observations: flag where quality first holds / first drops
        cyl_results = [(ctx, r["levels"][ctx].get("cylinder", {}).get("correct")) for ctx in levels]
        first_pass = next((ctx for ctx, ok in cyl_results if ok), None)
        first_fail = next((ctx for ctx, ok in reversed(cyl_results) if ok is False), None)

        if first_pass:
            lines.append(f"**Cylinder first passes at ctx={first_pass}.**")
        if first_fail and first_pass and first_fail > first_pass:
            lines.append(f"⚠ Regression at ctx={first_fail} — cylinder fails after passing at lower ctx.")
        lines.append("")

        # Full cylinder responses
        lines += ["### Cylinder responses", ""]
        for ctx in levels:
            e    = r["levels"][ctx]
            cyl  = e.get("cylinder", {})
            resp = cyl.get("response", cyl.get("error", "—"))
            mark = " ✓" if cyl.get("correct") else (" ✗" if "correct" in cyl else "")
            lines += [f"**ctx={ctx}**{mark}", "", resp, ""]

        # Full JPEG responses (workers)
        if has_jpeg:
            lines += ["### JPEG responses", ""]
            for ctx in levels:
                e    = r["levels"][ctx]
                j    = e.get("jpeg", {})
                resp = j.get("response", j.get("error", "—"))
                sigs = list(j.get("signals", {}).keys())
                score_str = f"{j['score']}/7  [{', '.join(sigs)}]" if "score" in j else "—"
                lines += [f"**ctx={ctx}** — coverage: {score_str}", "", resp, ""]

        lines += ["---", ""]

    out_md.write_text("\n".join(lines))
    print(f"MD → {out_md}", flush=True)


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TODAY  = date.today().isoformat()
    suffix = "_fast" if fast_mode else ""

    reg_path = REPO / "models.json"
    if not reg_path.exists():
        sys.exit(f"models.json not found — run update_registry.py first")

    registry = json.load(reg_path.open())

    if model_args:
        reg_map = {m["name"]: m for m in registry}
        MODELS = [
            (m, reg_map.get(m, {}).get("disk_gb", 0.0), reg_map.get(m, {}).get("role", "worker"))
            for m in model_args
        ]
    else:
        MODELS = [
            (m["name"], m["disk_gb"], m["role"])
            for m in registry if m.get("role") in CTX_LADDER
        ]

    if role_filter:
        MODELS = [(n, d, r) for n, d, r in MODELS if r == role_filter]

    if not MODELS:
        sys.exit("No models to test. Check models.json or --role filter.")

    preflight(MODELS, ollama_host)

    flag = " [FAST MODE]" if fast_mode else ""
    print(f"BenchLLAMA ctx ladder{flag} — {TODAY}", flush=True)
    print(f"ollama={ollama_host} | think=False | {len(MODELS)} model(s)", flush=True)

    OUT_JSON = RESULTS_DIR / f"ctx_ladder_{TODAY}{suffix}.json"
    OUT_MD   = RESULTS_DIR / f"ctx_ladder_{TODAY}{suffix}.md"
    print(f"Output: {OUT_JSON}\n", flush=True)

    # ── Resume / merge logic ──────────────────────────────────────────────────
    # Same semantics as runner.py: resume SOURCE = today's file if present, else
    # the most recent ctx_ladder within 72h (cross-day). Always writes today's file.
    all_results = []
    completed   = set()

    source = None
    if not (force and not model_args):
        source = OUT_JSON if OUT_JSON.exists() else latest_result(RESULTS_DIR, "ctx_ladder", fast_mode, 72)

    if source is not None:
        try:
            existing = json.load(source.open())
            if model_args:
                target_names = {m for m, *_ in MODELS}
                all_results  = [r for r in existing if r["model"] not in target_names]
            else:
                # Drop errored entries so a retry replaces (not duplicates) them.
                all_results = [r for r in existing if "error" not in r]
                completed   = {r["model"] for r in all_results}
                if completed:
                    via = "" if source == OUT_JSON else f" (carried from {source.name})"
                    print(f"  Resuming — {len(completed)} model(s) already done{via}: {sorted(completed)}", flush=True)
        except Exception:
            all_results, completed = [], set()

    # ── New-model notice (full run only) ──────────────────────────────────────
    # Not in the resume source → the loop below runs them now and merges them in.
    if not model_args and not force and completed:
        eligible_names = {m["name"] for m in registry if m.get("role") in CTX_LADDER}
        done_names     = {r["model"] for r in all_results}
        new_models     = eligible_names - done_names
        if new_models:
            print(f"  + {len(new_models)} new model(s) not in the resumed results — "
                  f"running now, existing models skipped: {sorted(new_models)}", flush=True)
            print(f"    (pass --force to re-baseline everything instead.)", flush=True)

    first_run = True
    for model_name, disk_gb, role in MODELS:
        if model_name in completed:
            print(f"  ↷ {model_name} — already done, skipping", flush=True)
            continue
        if not first_run:
            _ws(model_name, "cooldown")
            cooldown(COOLDOWN, label=f"after previous model")
        first_run = False
        ctx_levels = CTX_LADDER.get(role, CTX_LADDER["worker"])
        r = run_ctx_ladder(model_name, role, disk_gb, ctx_levels)
        all_results.append(r)
        OUT_JSON.write_text(json.dumps(all_results, indent=2))

    _ws("", "done")
    write_summary(all_results, OUT_MD, fast_mode)

    print(f"\n{'='*60}", flush=True)
    print("DONE", flush=True)
    print(f"JSON → {OUT_JSON}", flush=True)
    print(f"MD   → {OUT_MD}", flush=True)

    # Quick inline summary
    print("\n| Model | Role | ctx | RAM | prefill t/s | decode t/s | Cylinder | JPEG |")
    print("|-------|------|----:|-----|------------:|-----------:|:--------:|:----:|")
    for r in all_results:
        for ctx, e in sorted(r["levels"].items()):
            if "error" in e:
                continue
            cyl   = e.get("cylinder", {})
            jpeg  = e.get("jpeg", {})
            j_str = f"{jpeg['score']}/7" if "score" in jpeg else "—"
            print(
                f"| {r['model']} | {r['role']} | {ctx}"
                f" | {e.get('ram_gb', '?')}GB | {cyl.get('prefill_tps', '?')} | {cyl.get('tps', '?')}"
                f" | {'✓' if cyl.get('correct') else '✗'} | {j_str} |"
            )
