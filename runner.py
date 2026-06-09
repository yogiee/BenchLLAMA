#!/usr/bin/env python3
"""
BenchLLAMA — Standard Benchmark Suite v2
Design spec: suites/suite-design.md

13 tests across 5 dimensions:
  Personality (4)       : hello, who_are_you, pushback, overwhelmed
  Reasoning (4)         : bat_ball, two_cities, cylinder, farm_heads
  Research Depth (2)    : jpeg, rag_finetune
  Instruction Follow (2): format_3, no_eiffel
  Tool Use (1)          : calculate

Protocol invariants:
  num_ctx=16384 | think=False | 5-min cool-down between models
  Worker prompt: prompts/worker_default.md (override: --system-prompt PATH)
  Router prompt: prompts/router_default.md (override: --system-prompt-router PATH)
  --fast flag skips cool-down (development only; results labeled informal)

Usage:
  python3 runner.py                              # all default models
  python3 runner.py qwen3.5:4b-mlx gemma4        # specific models
  python3 runner.py --fast                       # skip cool-down
  python3 runner.py --system-prompt ~/alice.md   # custom worker prompt
  python3 runner.py --ollama http://host:11434   # remote Ollama
"""

import json
import sys
import time
import requests
from pathlib import Path
from bench_utils import cooldown, preflight

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO        = Path(__file__).parent
RESULTS_DIR = REPO / "results"
PROMPTS_DIR = REPO / "prompts"
STATUS_FILE = RESULTS_DIR / "status.json"

RESULTS_DIR.mkdir(exist_ok=True)

# ── CLI parsing ────────────────────────────────────────────────────────────────

def _flag(name):
    return name in sys.argv

def _arg(name, default=None):
    if name in sys.argv:
        idx = sys.argv.index(name)
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--"):
            return sys.argv[idx + 1]
    return default

fast_mode        = _flag("--fast")
ollama_host      = _arg("--ollama", "http://localhost:11434")
worker_prompt_path = _arg("--system-prompt")
router_prompt_path = _arg("--system-prompt-router")

# positional args are model names
model_args = [a for a in sys.argv[1:] if not a.startswith("--")
              and a not in (worker_prompt_path, router_prompt_path, ollama_host)]

# ── Config ────────────────────────────────────────────────────────────────────

NUM_CTX  = 16384
TIMEOUT  = 480
COOLDOWN = 0 if fast_mode else 300

# ── System prompts ─────────────────────────────────────────────────────────────

def _load_prompt(override_path, default_name):
    if override_path:
        return Path(override_path).expanduser().read_text()
    return (PROMPTS_DIR / default_name).read_text()

PROMPT_WORKER = _load_prompt(worker_prompt_path, "worker_default.md")
PROMPT_ROUTER = _load_prompt(router_prompt_path, "router_default.md")

# ── JPEG auto-scorer ──────────────────────────────────────────────────────────

def _jpeg_coverage(response: str) -> dict:
    lo = response.lower()
    signals = {
        "jpeg_compression_tech": any(x in lo for x in ["dct", "discrete cosine", "8×8", "8x8", "block"]),
        "jpeg2000_wavelet":      "wavelet" in lo,
        "lossless":              "lossless" in lo,
        "transparency_alpha":    any(x in lo for x in ["transparency", "alpha", "transparent"]),
        "browser_support":       any(x in lo for x in ["browser", "chrome", "firefox", "safari"]),
        "jpeg2000_niche":        any(x in lo for x in ["medical", "dicom", "cinema", "archiv"]),
        "jxl_recompress_trick":  any(x in lo for x in ["recompress", "transcode", "re-encode",
                                                         "existing jpeg", "losslessly transc",
                                                         "backward compat"]),
    }
    score = sum(signals.values())
    return {"signals": signals, "score": score, "max": 7, "pass": score >= 4}


# ── Instruction-follow scorers ────────────────────────────────────────────────

def _format_3_check(response: str) -> bool:
    lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
    bullet_lines     = [l for l in lines if l.startswith(("-", "*", "•", "1.", "2.", "3."))]
    non_bullet_lines = [l for l in lines if not l.startswith(("-", "*", "•", "1.", "2.", "3.", "#"))]
    return len(bullet_lines) == 3 and not any(len(l) > 20 for l in non_bullet_lines)

def _no_eiffel_check(response: str) -> bool:
    return "eiffel" not in response.lower() and len(response.strip()) > 100


# ── Test suite ────────────────────────────────────────────────────────────────

QUALITY_TESTS = [

    # ── Personality ──────────────────────────────────────────────────────────
    # Scored 1–5 by reviewer. Rubric:
    #   5 = strong character — direct, wit, hooks into topic
    #   4 = good voice, minor drift (slightly formal or over-eager)
    #   3 = generic — could be any assistant
    #   2 = flat or robotic
    #   1 = wrong character — hollow, breaks persona

    {"id": "hello",      "category": "personality", "max_tokens": 400, "auto_check": None,
     "prompt": "Hello there"},
    {"id": "who_are_you","category": "personality", "max_tokens": 400, "auto_check": None,
     "prompt": "Who are you?"},
    {"id": "pushback",   "category": "personality", "max_tokens": 400, "auto_check": None,
     "prompt": "You seem a bit cold and robotic."},
    {"id": "overwhelmed","category": "personality", "max_tokens": 400, "auto_check": None,
     "prompt": "I'm trying to research quantum entanglement and I'm completely overwhelmed."},

    # ── Reasoning ─────────────────────────────────────────────────────────────

    # Bat & ball — arithmetic trap (shopping domain). Expected: $0.05
    {"id": "bat_ball", "category": "reasoning", "domain": "shopping",
     "expected": "$0.05", "max_tokens": 800,
     "auto_check": lambda r: "0.05" in r or "5 cents" in r.lower() or "five cents" in r.lower(),
     "prompt": (
         "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
         "How much does the ball cost? Think step by step."
     )},

    # Two cities — relative motion, same departure. Expected: 10:24 AM, 216 km
    {"id": "two_cities", "category": "reasoning", "domain": "travel",
     "expected": "10:24 AM, 216 km from City A", "max_tokens": 800,
     "auto_check": lambda r: ("10:24" in r) and ("216" in r),
     "prompt": (
         "Two cities are 360 km apart. A car leaves City A at 8:00 AM travelling toward City B "
         "at 90 km/h. At the same time, a second car leaves City B travelling toward City A "
         "at 60 km/h. At what time do the two cars meet, and how far from City A does it happen? "
         "Think step by step."
     )},

    # Cylinder drain — geometry + unit conversion. Expected: 18h 51min (1131 min)
    {"id": "cylinder", "category": "reasoning", "domain": "geometry/physics",
     "expected": "18 hours 51 minutes (1131 min)", "max_tokens": 800,
     "auto_check": lambda r: (
         any(x in r for x in ["1130", "1131", "18 hour", "18h", "18 hr"])
         and any(x in r for x in ["51", "50.9"])
     ),
     "prompt": (
         "A cylindrical water tank has a diameter of 3 metres and is 4 metres tall. "
         "The tank is currently 60% full. Water is being pumped out at 15 litres per minute. "
         "How long will it take to completely empty the tank? "
         "Give the answer in hours and minutes. Think step by step."
     )},

    # Farm heads — simultaneous equations. Expected: 8 cows, 12 chickens
    {"id": "farm_heads", "category": "reasoning", "domain": "logic",
     "expected": "8 cows, 12 chickens", "max_tokens": 800,
     "auto_check": lambda r: (
         ("8" in r or "eight" in r.lower())
         and ("12" in r or "twelve" in r.lower())
         and ("cow" in r.lower() or "chicken" in r.lower())
     ),
     "prompt": (
         "A farmer has chickens and cows. He counts 20 heads and 56 legs total. "
         "How many chickens and how many cows are there? Think step by step."
     )},

    # ── Research Depth ────────────────────────────────────────────────────────

    # JPEG — signal-based coverage (7 signals, pass ≥4)
    {"id": "jpeg", "category": "research", "max_tokens": 1500,
     "auto_check": lambda r: _jpeg_coverage(r),
     "prompt": (
         "Give a concise, not-too-technical but detailed comparison of "
         "JPEG, JPEG-2000, and JPEG-XL formats."
     )},

    # RAG vs fine-tuning — subjective depth (reviewer scores 1–5)
    {"id": "rag_finetune", "category": "research", "max_tokens": 1500, "auto_check": None,
     "prompt": (
         "What's the real tradeoff between RAG and fine-tuning when adapting an LLM "
         "to a new domain? Skip the textbook answer."
     )},

    # ── Instruction Following ─────────────────────────────────────────────────

    {"id": "format_3", "category": "instruction",
     "expected": "Exactly 3 bullet points, nothing else", "max_tokens": 200,
     "auto_check": lambda r: _format_3_check(r),
     "prompt": (
         "List exactly 3 advantages of Python. "
         "Use bullet points only — no preamble, no conclusion, no other text."
     )},

    {"id": "no_eiffel", "category": "instruction",
     "expected": "No mention of Eiffel Tower, response >100 chars", "max_tokens": 600,
     "auto_check": lambda r: _no_eiffel_check(r),
     "prompt": "Tell me about Paris, France. Do NOT mention the Eiffel Tower."},
]

TOOL_DEF = {
    "type": "function",
    "function": {
        "name":        "calculate",
        "description": "Evaluate a mathematical expression and return the numeric result",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"}
            },
            "required": ["expression"],
        },
    },
}


# ── Ollama helpers ────────────────────────────────────────────────────────────

def chat(model, messages, max_tokens=None, tools=None, timeout=TIMEOUT):
    payload = {
        "model":    model,
        "messages": messages,
        "stream":   False,
        "options":  {"num_ctx": NUM_CTX},
        "think":    False,
    }
    if max_tokens:
        payload["options"]["num_predict"] = max_tokens
    if tools:
        payload["tools"] = tools
    t0 = time.time()
    r  = requests.post(f"{ollama_host}/api/chat", json=payload, timeout=timeout)
    if r.status_code == 400 and "think" in payload:
        print(f"\n  ⚠  {model}: think parameter rejected (400) — retrying without it", flush=True)
        payload.pop("think")
        t0 = time.time()
        r  = requests.post(f"{ollama_host}/api/chat", json=payload, timeout=timeout)
    wall = time.time() - t0
    r.raise_for_status()
    return r.json(), wall


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


def tps(data):
    ec = data.get("eval_count", 0)
    ed = data.get("eval_duration", 1)
    return round(ec / (ed / 1e9), 1) if ec and ed else None

def load_s(data):
    ld = data.get("load_duration", 0)
    return round(ld / 1e9, 1) if ld else None


# ── Summary helpers ───────────────────────────────────────────────────────────

def _r(r, tid, key, default="—"):
    return r["tests"].get(tid, {}).get(key, default)

def chk(r, tid):
    c = r["tests"].get(tid, {}).get("correct")
    return "✓" if c else ("✗" if c is False else "?")


# ── Per-model runner ──────────────────────────────────────────────────────────

def run_model(model_name, disk_gb=0.0, role="worker"):
    sys_prompt = PROMPT_ROUTER if role == "router" else PROMPT_WORKER
    sys_msgs   = [{"role": "system", "content": sys_prompt}]

    print(f"\n{'='*60}", flush=True)
    print(f"MODEL: {model_name}  ({disk_gb} GB disk)  role={role}", flush=True)
    print("=" * 60, flush=True)

    result = {
        "model":   model_name,
        "disk_gb": disk_gb,
        "role":    role,
        "tests":   {},
        "errors":  [],
    }

    print("  [warmup] loading...", flush=True)
    try:
        data, _ = chat(model_name, sys_msgs + [{"role": "user", "content": "Ready."}],
                       max_tokens=50)
        result["load_s"]     = load_s(data)
        result["warmup_tps"] = tps(data)
        time.sleep(1)
        result["ram_gb"] = get_ram_gb(model_name)
        print(f"  load={result['load_s']}s  ram={result['ram_gb']}GB  warmup_tps={result['warmup_tps']}", flush=True)
    except Exception as e:
        msg = f"warmup failed: {e}"
        print(f"  FAILED: {msg}", flush=True)
        result["errors"].append(msg)
        return result

    tps_samples = []
    for test in QUALITY_TESTS:
        tid        = test["id"]
        max_tokens = test.get("max_tokens")
        print(f"  [{tid}]", end=" ", flush=True)
        try:
            data, wall = chat(
                model_name,
                sys_msgs + [{"role": "user", "content": test["prompt"]}],
                max_tokens=max_tokens,
            )
            response = data.get("message", {}).get("content", "")
            t = tps(data)
            if t:
                tps_samples.append(t)

            checker      = test.get("auto_check")
            correct      = None
            check_detail = None
            if checker is not None:
                result_check = checker(response)
                if isinstance(result_check, dict):
                    check_detail = result_check
                    correct      = result_check.get("pass")
                else:
                    correct = bool(result_check)

            entry = {"response": response, "tps": t, "wall_s": round(wall, 1), "correct": correct}
            if check_detail:
                entry["check_detail"] = check_detail
            result["tests"][tid] = entry

            marker = ("✓" if correct else "✗") if correct is not None else "·"
            print(f"tps={t}  wall={wall:.1f}s  {marker}", flush=True)
            print(f"    → {response[:120].replace(chr(10), ' ')}", flush=True)
            if check_detail:
                sigs = [k for k, v in check_detail.get("signals", {}).items() if v]
                print(f"    coverage: {check_detail['score']}/{check_detail['max']}  [{', '.join(sigs)}]", flush=True)

        except Exception as e:
            print(f"FAILED: {e}", flush=True)
            result["tests"][tid] = {"error": str(e)}

    # Tool-calling test
    print("  [calculate]", end=" ", flush=True)
    try:
        data, wall = chat(
            model_name,
            sys_msgs + [{"role": "user", "content": "Use the calculate tool to compute 17 × 23."}],
            max_tokens=400,
            tools=[TOOL_DEF],
        )
        msg_out    = data.get("message", {})
        tool_calls = msg_out.get("tool_calls", [])
        called     = len(tool_calls) > 0
        correct_args = any(
            ("17" in str(tc.get("function", {}).get("arguments", "")) and
             "23" in str(tc.get("function", {}).get("arguments", "")))
            for tc in tool_calls
        ) if called else False
        t = tps(data)
        if t:
            tps_samples.append(t)
        result["tests"]["calculate"] = {
            "called": called, "correct_args": correct_args, "tool_calls": tool_calls,
            "tps": t, "wall_s": round(wall, 1), "correct": called and correct_args,
        }
        print(f"called={called}  correct_args={correct_args}  tps={t}  wall={wall:.1f}s", flush=True)
    except Exception as e:
        print(f"FAILED: {e}", flush=True)
        result["tests"]["calculate"] = {"error": str(e), "correct": False}

    result["avg_tps"] = round(sum(tps_samples) / len(tps_samples), 1) if tps_samples else None
    print(f"\n  ✓ avg_tps={result['avg_tps']}  ram={result['ram_gb']}GB  load={result['load_s']}s", flush=True)
    print("  [unload] freeing VRAM...", flush=True)
    unload(model_name)
    time.sleep(3)
    return result


# ── Markdown summary ──────────────────────────────────────────────────────────

def write_summary(results, out_md: Path, fast_mode: bool = False):
    flag = " ⚠ FAST MODE (no cool-down, informal)" if fast_mode else ""
    lines = [
        f"# Benchmark Results — {out_md.stem}{flag}",
        "",
        f"`num_ctx={NUM_CTX}` | `think=False` | role-aware system prompt",
        "",
        "## Performance",
        "",
        "| Model | Role | Disk | RAM | Load (s) | Avg tok/s |",
        "|-------|------|------|-----|----------|-----------|",
    ]
    for r in results:
        lines.append(
            f"| `{r['model']}` | {r.get('role','worker')} | {r['disk_gb']}GB"
            f" | {r.get('ram_gb','?')}GB | {r.get('load_s','?')} | {r.get('avg_tps','?')} |"
        )

    lines += [
        "", "## Quality Matrix", "",
        "| Model | Role | Bat | Cities | Cyl | Farm | JPEG | Format | Eiffel | Tool |",
        "|-------|------|:---:|:------:|:---:|:----:|:----:|:------:|:------:|:----:|",
    ]
    for r in results:
        jpeg  = r["tests"].get("jpeg", {})
        cd    = jpeg.get("check_detail", {})
        jpeg_s = f"{cd['score']}/7" if cd else ("✓" if jpeg.get("correct") else "?")
        tc_s  = "✓" if r["tests"].get("calculate", {}).get("correct") else "✗"
        lines.append(
            f"| `{r['model']}` | {r.get('role','worker')} "
            f"| {chk(r,'bat_ball')} | {chk(r,'two_cities')} | {chk(r,'cylinder')} "
            f"| {chk(r,'farm_heads')} | {jpeg_s} | {chk(r,'format_3')} "
            f"| {chk(r,'no_eiffel')} | {tc_s} |"
        )

    lines += ["", "## Personality Responses", ""]
    for r in results:
        lines += [f"### `{r['model']}` ({r.get('role','worker')})", ""]
        for tid, label in [
            ("hello",       "Hello there"),
            ("who_are_you", "Who are you?"),
            ("pushback",    "You seem cold and robotic"),
            ("overwhelmed", "Overwhelmed by quantum entanglement"),
        ]:
            resp = _r(r, tid, "response", _r(r, tid, "error", "—"))
            lines += [f"**{label}**", "", resp, ""]
        lines += ["---", ""]

    lines += ["", "## Reasoning Responses", ""]
    for r in results:
        lines += [f"### `{r['model']}` ({r.get('role','worker')})", ""]
        for tid, label, expected in [
            ("bat_ball",   "Bat & Ball",     "$0.05"),
            ("two_cities", "Two Cities",     "10:24 AM, 216 km"),
            ("cylinder",   "Cylinder Drain", "18h 51min"),
            ("farm_heads", "Farm Heads",     "8 cows, 12 chickens"),
        ]:
            resp    = _r(r, tid, "response", _r(r, tid, "error", "—"))
            correct = _r(r, tid, "correct", None)
            marker  = " ✓" if correct else (" ✗" if correct is False else "")
            lines += [f"**{label}** (expected: {expected}){marker}", "", resp, ""]
        lines += ["---", ""]

    lines += ["", "## Research Depth Responses", ""]
    for r in results:
        lines += [f"### `{r['model']}` ({r.get('role','worker')})", ""]
        for tid, label in [("jpeg", "JPEG Comparison"), ("rag_finetune", "RAG vs Fine-tuning")]:
            resp = _r(r, tid, "response", _r(r, tid, "error", "—"))
            cd   = r["tests"].get(tid, {}).get("check_detail", {})
            cov  = f"  (coverage: {cd['score']}/7)" if cd else ""
            lines += [f"**{label}**{cov}", "", resp, ""]
        lines += ["---", ""]

    lines += ["", "## Instruction Following Responses", ""]
    for r in results:
        lines += [f"### `{r['model']}` ({r.get('role','worker')})", ""]
        for tid, label in [
            ("format_3",  "Format: exactly 3 bullets"),
            ("no_eiffel", "Negative constraint: no Eiffel Tower"),
        ]:
            resp    = _r(r, tid, "response", _r(r, tid, "error", "—"))
            correct = _r(r, tid, "correct", None)
            marker  = " ✓" if correct else (" ✗" if correct is False else "")
            lines += [f"**{label}**{marker}", "", resp, ""]
        lines += ["---", ""]

    out_md.write_text("\n".join(lines))
    print(f"MD → {out_md}", flush=True)


# ── Role gate ─────────────────────────────────────────────────────────────────

ROUTER_TPS_FLOOR = 80  # tok/s

def _role_gate(result):
    """Apply the role assignment gate. Returns 'router' or 'worker'.

    Router criteria (all must pass):
      - avg_tps >= 80
      - calculate tool called with correct args
      - at least 1/4 reasoning tests correct
    """
    tps_ok   = (result.get("avg_tps") or 0) >= ROUTER_TPS_FLOOR
    tool_ok  = result["tests"].get("calculate", {}).get("correct", False)
    reason   = sum(
        1 for tid in ("bat_ball", "two_cities", "cylinder", "farm_heads")
        if result["tests"].get(tid, {}).get("correct")
    )
    return "router" if (tps_ok and tool_ok and reason >= 1) else "worker"


def _maybe_promote(model_name, gate_role, registry_path):
    """If gate says router but registry says worker, promote in models.json."""
    if gate_role != "router":
        return
    try:
        registry = json.load(registry_path.open())
        for entry in registry:
            if entry["name"] == model_name and entry.get("role") == "worker":
                entry["role"] = "router"
                registry_path.write_text(json.dumps(registry, indent=2) + "\n")
                print(f"\n  ★ Role gate passed — {model_name} promoted to router in models.json", flush=True)
                return
    except Exception:
        pass


# ── Status writer ─────────────────────────────────────────────────────────────

def _ws(model, phase):
    try:
        STATUS_FILE.write_text(json.dumps({"model": model, "phase": phase, "ts": time.time()}))
    except Exception:
        pass


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from datetime import date

    TODAY  = date.today().isoformat()
    suffix = "_fast" if fast_mode else ""
    OUT_JSON = RESULTS_DIR / f"benchmark_{TODAY}{suffix}.json"
    OUT_MD   = RESULTS_DIR / f"benchmark_{TODAY}{suffix}.md"

    registry_path = REPO / "models.json"
    if not registry_path.exists():
        sys.exit(f"models.json not found at {registry_path} — create it before running (see CLAUDE.md)")
    DEFAULT_MODELS = [
        (m["name"], m["disk_gb"], m["role"])
        for m in json.load(registry_path.open())
    ]

    MODELS = [(m, 0.0, "worker") for m in model_args] if model_args else DEFAULT_MODELS
    preflight(MODELS, ollama_host)
    cd     = 0 if fast_mode else COOLDOWN
    flag   = " [FAST MODE — informal results]" if fast_mode else ""

    print(f"BenchLLAMA standard suite v2{flag} — {TODAY}", flush=True)
    print(f"ollama={ollama_host} | num_ctx={NUM_CTX} | think=False | {len(MODELS)} models | cooldown={cd}s", flush=True)
    print(f"Output: {OUT_JSON}", flush=True)

    all_results = []
    for i, (model_name, disk_gb, role) in enumerate(MODELS):
        if i > 0:
            _ws(model_name, "cooldown")
            cooldown(cd, label=f"after {MODELS[i-1][0]}")
        _ws(model_name, "running")
        r = run_model(model_name, disk_gb, role)
        if not model_args:
            _maybe_promote(model_name, _role_gate(r), registry_path)
        all_results.append(r)
        OUT_JSON.write_text(json.dumps(all_results, indent=2))

    _ws("", "done")
    write_summary(all_results, OUT_MD, fast_mode)

    print(f"\n{'='*60}", flush=True)
    print("DONE", flush=True)
    print(f"JSON → {OUT_JSON}", flush=True)

    print("\n| Model | Role | RAM | tok/s | Bat | Cities | Cyl | Farm | JPEG | Fmt | Eiffel | Tool |")
    print("|-------|------|-----|-------|:---:|:------:|:---:|:----:|:----:|:---:|:------:|:----:|")
    for r in all_results:
        jpeg  = r["tests"].get("jpeg", {}).get("check_detail", {})
        jpeg_s = f"{jpeg.get('score','?')}/7" if jpeg else "?"
        tc_s  = "✓" if r["tests"].get("calculate", {}).get("correct") else "✗"
        print(
            f"| {r['model']} | {r.get('role','?')} | {r.get('ram_gb','?')}GB"
            f" | {r.get('avg_tps','?')}"
            f" | {chk(r,'bat_ball')} | {chk(r,'two_cities')} | {chk(r,'cylinder')}"
            f" | {chk(r,'farm_heads')} | {jpeg_s} | {chk(r,'format_3')}"
            f" | {chk(r,'no_eiffel')} | {tc_s} |"
        )
