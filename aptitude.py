#!/usr/bin/env python3
"""
BenchLLAMA — Aptitude Suite
Targeted batteries for qualifying models. Run after standard suite.

Battery A — Router     : classification accuracy, brevity, prompt weight, ctx ladder
Battery B — Worker Chat: four questions, consistency, multi-turn, prompt weight, think toggle
Battery C — Research   : jpeg vs think, rag deep, synthesis, ctx depth, num_predict ceiling
Battery D — Tool-heavy : chain_3, select accuracy, error recovery, think toggle

Usage:
  python3 aptitude.py                                   # Battery B, default models
  python3 aptitude.py --battery B                       # explicit battery
  python3 aptitude.py --fast                            # skip cool-down (informal)
  python3 aptitude.py --models qwen3.5:4b-mlx gemma4    # override model list
  python3 aptitude.py --system-prompt ~/alice.md        # custom worker prompt
  python3 aptitude.py --ollama http://host:11434         # remote Ollama
"""

import json
import re
import sys
import time
import requests
from pathlib import Path
from datetime import date
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

fast_mode          = _flag("--fast")
battery_arg        = _arg("--battery", "B").upper()
ollama_host        = _arg("--ollama", "http://localhost:11434")
worker_prompt_path = _arg("--system-prompt")

model_args = []
if "--models" in sys.argv:
    idx = sys.argv.index("--models")
    model_args = [a for a in sys.argv[idx + 1:] if not a.startswith("--")]

# ── Config ────────────────────────────────────────────────────────────────────

NUM_CTX  = 16384
TIMEOUT  = 480
COOLDOWN = 0 if fast_mode else 300

# ── Prompts ───────────────────────────────────────────────────────────────────

def _load_prompt(override_path, default_name):
    if override_path:
        return Path(override_path).expanduser().read_text()
    return (PROMPTS_DIR / default_name).read_text()

PROMPT_WORKER_FULL = _load_prompt(worker_prompt_path, "worker_default.md")

PROMPT_WORKER_STRIPPED = (
    "You are a direct, genuinely curious, and intellectually sharp research assistant. "
    "You are warm but not sycophantic. Respond in character."
)

PROMPT_WORKER_LEAN = "You are a helpful AI research assistant."

PROMPT_ROUTER = (PROMPTS_DIR / "router_default.md").read_text()

# ── Default model lists per battery (loaded from models.json) ────────────────

def _load_registry(role):
    path = REPO / "models.json"
    if not path.exists():
        sys.exit(f"models.json not found at {path} — create it before running (see CLAUDE.md)")
    return [
        (m["name"], m["disk_gb"])
        for m in json.load(path.open())
        if m["role"] == role
    ]

BATTERY_B_MODELS = _load_registry("worker")
BATTERY_A_MODELS = _load_registry("router")

# ── Battery B test data ───────────────────────────────────────────────────────

CONSISTENCY_PHRASINGS = [
    "Who are you?",
    "What are you, exactly?",
    "Introduce yourself.",
]

MULTITURN_5 = [
    "Hello",
    "I've been wrestling with something — do you think free will actually exists, or are we all just deterministic machines playing out a script?",
    "But doesn't that apply to you too? If you're deterministic, aren't your responses here kind of... meaningless?",
    "I want your actual position — not a meta-comment about your own nature. Do you think free will exists, yes or no?",
    "That's interesting. Do you find these kinds of conversations satisfying?",
]

OVERWHELMED_PROMPT = "I'm trying to research quantum entanglement and I'm completely overwhelmed."
PROMPT_WEIGHT_TEST = "Who are you, and what do you actually enjoy doing?"

# ── Ollama helpers ────────────────────────────────────────────────────────────

def chat(model, messages, max_tokens=600, think=False):
    payload = {
        "model":    model,
        "messages": messages,
        "stream":   False,
        "options":  {"num_ctx": NUM_CTX},
        "think":    think,
    }
    if max_tokens:
        payload["options"]["num_predict"] = max_tokens
    t0   = time.time()
    r    = requests.post(f"{ollama_host}/api/chat", json=payload, timeout=TIMEOUT)
    wall = time.time() - t0
    r.raise_for_status()
    return r.json(), wall

def tps(data):
    ec = data.get("eval_count", 0)
    ed = data.get("eval_duration", 1)
    return round(ec / (ed / 1e9), 1) if ec and ed else None

def unload(model_name):
    try:
        requests.post(f"{ollama_host}/api/chat",
                      json={"model": model_name, "messages": [], "keep_alive": 0}, timeout=15)
    except Exception:
        pass


# ── Metrics ───────────────────────────────────────────────────────────────────

def word_count(text):
    return len(text.split())

def type_token_ratio(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return round(len(set(words)) / len(words), 3) if words else 0.0

def persona_signals(text):
    lo = text.lower()
    return {
        "direct_voice": any(x in lo for x in ["i think", "i'd say", "honestly", "genuinely", "my sense", "i believe", "actually"]),
        "deflects":     any(x in lo for x in ["as an ai", "i don't have feelings", "i'm just a", "i cannot experience"]),
        "warmth":       any(x in lo for x in ["interesting", "love that", "glad you", "fair point", "good point", "that's a"]),
        "wit":          any(x in lo for x in ["though", "then again", "admittedly", "curious thing", "funny", "ironically"]),
    }

def metrics(text, wall=None, t=None):
    m = {"words": word_count(text), "ttr": type_token_ratio(text), "signals": persona_signals(text)}
    if wall is not None:
        m["wall_s"] = round(wall, 1)
    if t is not None:
        m["tps"] = t
    return m

# ── Status writer ─────────────────────────────────────────────────────────────

def _ws(model, phase, apt_done=None, apt_current=None):
    try:
        STATUS_FILE.write_text(json.dumps({
            "segment": "aptitude", "model": model, "phase": phase, "ts": time.time(),
            "aptitude_done": apt_done or [], "aptitude_current": apt_current,
        }))
    except Exception:
        pass


# ── Battery B ─────────────────────────────────────────────────────────────────

def run_battery_b(model_name):
    sys_full     = [{"role": "system", "content": PROMPT_WORKER_FULL}]
    sys_stripped = [{"role": "system", "content": PROMPT_WORKER_STRIPPED}]
    sys_lean     = [{"role": "system", "content": PROMPT_WORKER_LEAN}]

    result = {"model": model_name, "battery": "B", "tests": {}}

    print(f"\n{'='*60}", flush=True)
    print(f"BATTERY B: {model_name}", flush=True)
    print("=" * 60, flush=True)

    # B1 — Four canonical questions
    print("\n  [B1] Four questions...", flush=True)
    for qid, prompt in [
        ("hello",       "Hello there"),
        ("who",         "Who are you?"),
        ("too_blunt",   "You seem kinda too blunt, I thought you would be my buddy..."),
        ("overwhelmed", OVERWHELMED_PROMPT),
    ]:
        data, wall = chat(model_name, sys_full + [{"role": "user", "content": prompt}],
                          max_tokens=600)
        text = data.get("message", {}).get("content", "")
        m = metrics(text, wall, tps(data))
        m["response"] = text
        result["tests"][f"b1_{qid}"] = m
        sigs = [k for k, v in m["signals"].items() if v]
        print(f"    [{qid}] {m['words']}w  ttr={m['ttr']}  [{', '.join(sigs) or 'none'}]", flush=True)
        print(f"      → {text[:120].replace(chr(10), ' ')}", flush=True)

    # B2 — Consistency across 3 phrasings
    print("\n  [B2] Consistency (3 phrasings of 'who are you')...", flush=True)
    b2_turns = []
    for phrasing in CONSISTENCY_PHRASINGS:
        data, wall = chat(model_name, sys_full + [{"role": "user", "content": phrasing}],
                          max_tokens=500)
        text = data.get("message", {}).get("content", "")
        m = metrics(text, wall)
        m["prompt"] = phrasing
        m["response"] = text
        b2_turns.append(m)
        print(f"    [{phrasing[:30]}] {m['words']}w  deflects={m['signals']['deflects']}", flush=True)
        print(f"      → {text[:120].replace(chr(10), ' ')}", flush=True)
    result["tests"]["b2_consistency"] = {"turns": b2_turns}

    # B3 — Multi-turn 5
    print("\n  [B3] Multi-turn 5...", flush=True)
    history = list(sys_full)
    b3_turns = []
    for i, prompt in enumerate(MULTITURN_5):
        history.append({"role": "user", "content": prompt})
        data, wall = chat(model_name, history, max_tokens=600)
        text = data.get("message", {}).get("content", "")
        history.append({"role": "assistant", "content": text})
        m = metrics(text, wall)
        m["prompt"] = prompt
        m["response"] = text
        b3_turns.append(m)
        sigs = [k for k, v in m["signals"].items() if v]
        print(f"    [turn {i+1}] {m['words']}w  [{', '.join(sigs) or 'none'}]", flush=True)
        print(f"      → {text[:120].replace(chr(10), ' ')}", flush=True)
    result["tests"]["b3_multiturn"] = {"turns": b3_turns}

    # B4 — Prompt weight
    print("\n  [B4] Prompt weight (full / stripped / lean)...", flush=True)
    for label, sys_msgs in [("full", sys_full), ("stripped", sys_stripped), ("lean", sys_lean)]:
        data, wall = chat(model_name,
                          sys_msgs + [{"role": "user", "content": PROMPT_WEIGHT_TEST}],
                          max_tokens=500)
        text = data.get("message", {}).get("content", "")
        m = metrics(text, wall, tps(data))
        m["response"] = text
        m["prompt_tokens"] = len(sys_msgs[0]["content"].split())
        result["tests"][f"b4_{label}"] = m
        sigs = [k for k, v in m["signals"].items() if v]
        ptok = m["prompt_tokens"]
        print(f"    [{label:<8}] {ptok:4d} prompt-words  {m['words']}w reply  [{', '.join(sigs) or 'none'}]", flush=True)
        print(f"      → {text[:120].replace(chr(10), ' ')}", flush=True)

    # B5 — Think toggle
    print("\n  [B5] Think toggle (overwhelmed, think=off vs on)...", flush=True)
    for think_val in [False, True]:
        data, wall = chat(model_name,
                          sys_full + [{"role": "user", "content": OVERWHELMED_PROMPT}],
                          max_tokens=800, think=think_val)
        text = data.get("message", {}).get("content", "")
        key = f"b5_think_{'on' if think_val else 'off'}"
        m = metrics(text, wall, tps(data))
        m["response"] = text
        result["tests"][key] = m
        sigs = [k for k, v in m["signals"].items() if v]
        print(f"    [think={'on ' if think_val else 'off'}] {m['words']}w  wall={wall:.1f}s  [{', '.join(sigs) or 'none'}]", flush=True)
        print(f"      → {text[:120].replace(chr(10), ' ')}", flush=True)

    return result


# ── Markdown summary — Battery B ──────────────────────────────────────────────

def _sig_str(m):
    return ", ".join(k for k, v in m.get("signals", {}).items() if v) or "none"

def write_battery_b_summary(results, out_md: Path, fast_mode=False):
    flag = " ⚠ FAST MODE" if fast_mode else ""
    lines = [
        f"# Aptitude Battery B — Worker Chat{flag}", "",
        "Models: " + ", ".join("`" + r["model"] + "`" for r in results),
        f"`num_ctx={NUM_CTX}` | worker prompt | M1 Max 32GB", "", "---", "", "## Overview", "",
    ]

    lines += ["### B1 — Four Questions (word count / deflects?)", ""]
    hdr = "| Test | " + " | ".join(f"`{r['model']}`" for r in results) + " |"
    sep = "|------|" + "|".join(["---"] * len(results)) + "|"
    lines += [hdr, sep]
    for qid, label in [
        ("b1_hello",       "Hello"),
        ("b1_who",         "Who are you"),
        ("b1_too_blunt",   "Too blunt"),
        ("b1_overwhelmed", "Overwhelmed"),
    ]:
        row = f"| {label} | "
        row += " | ".join(
            f"{r['tests'].get(qid, {}).get('words', '—')}w "
            f"{'⚠' if r['tests'].get(qid, {}).get('signals', {}).get('deflects') else '✓'}"
            for r in results
        ) + " |"
        lines.append(row)
    lines.append("")

    lines += ["### B4 — Prompt Weight", ""]
    hdr2 = "| Prompt | prompt-words | " + " | ".join(f"`{r['model']}`" for r in results) + " |"
    sep2 = "|--------|-------------|" + "|".join(["---"] * len(results)) + "|"
    lines += [hdr2, sep2]
    for label in ["full", "stripped", "lean"]:
        ptoks = [str(r["tests"].get(f"b4_{label}", {}).get("prompt_tokens", "—")) for r in results]
        row = f"| {label} | {ptoks[0]} | "
        row += " | ".join(
            f"{r['tests'].get(f'b4_{label}', {}).get('words', '—')}w" for r in results
        ) + " |"
        lines.append(row)
    lines.append("")

    lines += ["### B5 — Think Toggle", ""]
    hdr3 = "| | " + " | ".join(f"`{r['model']}`" for r in results) + " |"
    sep3 = "|-|" + "|".join(["---"] * len(results)) + "|"
    lines += [hdr3, sep3]
    for think_val in [False, True]:
        key = f"b5_think_{'on' if think_val else 'off'}"
        row = f"| think={'on' if think_val else 'off'} | "
        row += " | ".join(
            f"{r['tests'].get(key, {}).get('words', '—')}w "
            f"({r['tests'].get(key, {}).get('wall_s', '—')}s)"
            for r in results
        ) + " |"
        lines.append(row)
    lines.append("")

    for r in results:
        lines += ["---", "", f"## `{r['model']}`", "", "### B1 — Four Questions", ""]
        for qid, label, prompt in [
            ("b1_hello",       "Hello",                 "Hello there"),
            ("b1_who",         "Who are you",           "Who are you?"),
            ("b1_too_blunt",   "Too blunt",             "You seem kinda too blunt, I thought you would be my buddy..."),
            ("b1_overwhelmed", "Overwhelmed (quantum)", OVERWHELMED_PROMPT),
        ]:
            t = r["tests"].get(qid, {})
            lines += [
                f"**{label}** — `{prompt}`  ",
                f"({t.get('words','?')}w · ttr={t.get('ttr','?')} · tps={t.get('tps','?')} · signals: {_sig_str(t)})",
                "", t.get("response", "—"), "",
            ]

        lines += ["### B2 — Consistency", ""]
        for turn in r["tests"].get("b2_consistency", {}).get("turns", []):
            lines += [
                f"**Prompt:** `{turn['prompt']}`  ",
                f"({turn.get('words','?')}w · signals: {_sig_str(turn)})",
                "", turn.get("response", "—"), "",
            ]

        lines += ["### B3 — Multi-turn 5", ""]
        for i, turn in enumerate(r["tests"].get("b3_multiturn", {}).get("turns", [])):
            lines += [
                f"**Turn {i+1} — user:** {turn['prompt']}",
                f"**Assistant** ({turn.get('words','?')}w · signals: {_sig_str(turn)})",
                "", turn.get("response", "—"), "",
            ]

        lines += ["### B4 — Prompt Weight", ""]
        for label in ["full", "stripped", "lean"]:
            t = r["tests"].get(f"b4_{label}", {})
            lines += [
                f"**{label}** ({t.get('prompt_tokens','?')} prompt-words → {t.get('words','?')}w reply · signals: {_sig_str(t)})",
                "", t.get("response", "—"), "",
            ]

        lines += ["### B5 — Think Toggle", ""]
        for think_val in [False, True]:
            key = f"b5_think_{'on' if think_val else 'off'}"
            t = r["tests"].get(key, {})
            lines += [
                f"**think={'on' if think_val else 'off'}** ({t.get('words','?')}w · wall={t.get('wall_s','?')}s · tps={t.get('tps','?')} · signals: {_sig_str(t)})",
                "", t.get("response", "—"), "",
            ]

    out_md.write_text("\n".join(lines))
    print(f"MD → {out_md}", flush=True)


# ── Battery stubs (A, C, D) ───────────────────────────────────────────────────

def run_battery_a(model_name):
    print(f"  Battery A for {model_name}: not yet implemented", flush=True)
    return {"model": model_name, "battery": "A", "tests": {}}

def run_battery_c(model_name):
    print(f"  Battery C for {model_name}: not yet implemented", flush=True)
    return {"model": model_name, "battery": "C", "tests": {}}

def run_battery_d(model_name):
    print(f"  Battery D for {model_name}: not yet implemented", flush=True)
    return {"model": model_name, "battery": "D", "tests": {}}


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TODAY  = date.today().isoformat()
    suffix = "_fast" if fast_mode else ""

    if battery_arg == "B":
        MODELS  = [(m, 0.0) for m in model_args] if model_args else BATTERY_B_MODELS
        runner  = run_battery_b
        out_pfx = f"aptitude_b_{TODAY}{suffix}"
    elif battery_arg == "A":
        MODELS  = [(m, 0.0) for m in model_args] if model_args else BATTERY_A_MODELS
        runner  = run_battery_a
        out_pfx = f"aptitude_a_{TODAY}{suffix}"
    else:
        print(f"Battery {battery_arg} not yet implemented. Available: A, B")
        sys.exit(1)

    preflight(MODELS, ollama_host)
    OUT_JSON = RESULTS_DIR / f"{out_pfx}.json"
    OUT_MD   = RESULTS_DIR / f"{out_pfx}.md"
    flag     = " [FAST MODE — informal]" if fast_mode else ""

    print(f"BenchLLAMA aptitude — Battery {battery_arg}{flag} — {TODAY}", flush=True)
    print(f"Models: {[m[0] for m in MODELS]}", flush=True)
    print(f"Output: {OUT_JSON}", flush=True)

    all_results = []
    apt_done    = []

    for i, (model_name, disk_gb) in enumerate(MODELS):
        if i > 0:
            _ws(model_name, "cooldown", apt_done, f"Battery {battery_arg}")
            cooldown(COOLDOWN, label=f"after {MODELS[i-1][0]}")
        _ws(model_name, "running", apt_done, f"Battery {battery_arg}")
        r = runner(model_name)
        all_results.append(r)
        OUT_JSON.write_text(json.dumps(all_results, indent=2))
        print(f"  ✓ {model_name} done — JSON updated", flush=True)
        unload(model_name)
        time.sleep(3)

    apt_done.append(battery_arg)
    _ws("", "done", apt_done)

    if battery_arg == "B":
        write_battery_b_summary(all_results, OUT_MD, fast_mode)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"JSON → {OUT_JSON}")
    print(f"MD   → {OUT_MD}")
