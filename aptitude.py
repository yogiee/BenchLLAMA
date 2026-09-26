#!/usr/bin/env python3
"""
BenchLLAMA — Aptitude Suite
Targeted batteries for qualifying models. Run after standard suite.

Battery A — Router     : classification accuracy, brevity, prompt weight, ctx ladder
Battery B — Worker Chat: four questions, consistency, multi-turn, prompt weight, think toggle
Battery C — Research   : jpeg vs think, rag deep, synthesis, ctx depth, num_predict ceiling
Battery D — Tool-heavy : chain_3, select accuracy, error recovery, partial error,
                         think toggle (fixed diagnosis), personality+tool, deep cart

Usage:
  python3 aptitude.py                                   # Battery B, default models
  python3 aptitude.py --battery B                       # explicit battery
  python3 aptitude.py --battery D --capable-only        # Battery D, tool-capable models only
  python3 aptitude.py --fast                            # skip cool-down (informal)
  python3 aptitude.py --models qwen3.5:4b-mlx gemma4    # override model list
  python3 aptitude.py --system-prompt ~/alice.md        # custom worker prompt
  python3 aptitude.py --ollama http://host:11434         # remote Ollama
"""

import json
import re
import statistics
import sys
import time
import requests
from pathlib import Path
from datetime import date
from bench_utils import (cooldown, preflight, latest_result, sort_registry,
                         post_with_budget_retry, F_ELASTIC_CUTOFFS,
                         apply_think, arm_stamp, reply_stats, requested_arm, resolve_arm, budget_timeout)

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
force              = _flag("--force")
capable_only       = _flag("--capable-only")   # Battery C/D: skip models that failed calculate
battery_arg        = _arg("--battery", "B").upper()
# v3 (docs/think-spec.md): Battery A is the router SPEED lane → direct arm; every other battery measures
# each model at its operating point (`auto`: think arm if the probe found one, else fast).
ARM_REQ            = "direct" if battery_arg == "A" else requested_arm(default="auto")
ollama_host        = _arg("--ollama", "http://localhost:11434")
worker_prompt_path = _arg("--system-prompt")

model_args = []
if "--models" in sys.argv:
    idx = sys.argv.index("--models")
    for a in sys.argv[idx + 1:]:
        if a.startswith("--"):
            break
        model_args.append(a)

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


def _filter_tool_capable(models):
    """Keep only models that passed the calculate test in the most recent benchmark JSON.
    Used by --capable-only for Battery C and D to skip personality-only workers.
    Falls back to the full list if no benchmark data is available.
    """
    benchmarks = sorted((p for p in RESULTS_DIR.glob("benchmark_*.json") if "_think" not in p.name),   # fast-arm file
                        key=lambda p: p.stat().st_mtime)
    if not benchmarks:
        print("  --capable-only: no benchmark JSON found — running all models", flush=True)
        return models
    try:
        records = json.load(benchmarks[-1].open())
        passed  = {r["model"] for r in records
                   if r.get("tests", {}).get("calculate", {}).get("correct")}
        filtered = [(m, d) for m, d in models if m in passed]
        if filtered:
            skipped = [m for m, _ in models if m not in passed]
            print(f"  --capable-only: {len(filtered)}/{len(models)} models passed tool gate "
                  f"(skipped: {skipped})", flush=True)
            return filtered
        print("  --capable-only: no matches in benchmark data — running all models", flush=True)
    except Exception as e:
        print(f"  --capable-only: parse error ({e}) — running all models", flush=True)
    return models


BATTERY_B_MODELS = _load_registry("worker")
BATTERY_A_MODELS = _load_registry("router")


def _completion_models():
    """Every completion model — worker OR router. Battery E selects by capability
    (completion), not by a role gate; workers first, then routers, de-duplicated."""
    seen, out = set(), []
    for m, d in BATTERY_B_MODELS + BATTERY_A_MODELS:
        if m not in seen:
            seen.add(m)
            out.append((m, d))
    return out


BATTERY_E_MODELS = _completion_models()

# Coding grading harness + problem set (suites/coding) — Battery E.
sys.path.insert(0, str(REPO / "suites" / "coding"))
import harness as _codeh   # noqa: E402

_CODING_PROBLEMS_PATH = REPO / "suites" / "coding" / "problems.json"

# Composite weights per category (normalized over whatever categories are present,
# so adding E4/E6/E8 later needs no rebalance). E2/E5 carry the discriminating
# signal, so they weigh heaviest; E3 is the multi-language tier (JS/SQL/PHP);
# E9 is markup quality (HTML/CSS) — a core local_code generation workload.
E_WEIGHTS = {"E1": 0.12, "E2": 0.22, "E3": 0.18, "E5": 0.18, "E7": 0.15, "E9": 0.15,
             "E-hard": 0.18}   # two-band: E-hard is the top-tier RANKING discriminator (breaks E-core
                               # saturation). As of 2026-07-02 it holds 4 averaged tasks (settle /
                               # apportion / free_slots / calc) and its weight now feeds BOTH the ranking
                               # composite AND the coder gate (see average_e_runs — was E-core-only when
                               # the band was a single noisy task).

# `coder` overlay thresholds. HYSTERESIS: a model EARNS `coder` at composite ≥ MIN
# (+ the gates below); once tagged it's RETAINED down to BAND, and only DROPPED below
# BAND. The band absorbs run-to-run wobble near the line so the tag doesn't flap.
#
# RE-VALIDATED 2026-08-22 (the open item from 07-02). The composite bar alone had stopped
# selecting: 19 of 22 models cleared 0.70 and hysteresis left **20 of 22 holding `coder`** —
# a 3B router (granite4.1:3b 0.891) and a 1-bit quant (bonsai-27b 0.902) among them. The
# cause is weighting, not the bar. E-core is spent (E1 mean 0.965 / 17-of-22 perfect, E9
# 0.989 / 14-of-22) while **E-hard is the healthiest discriminator in the harness** — mean
# 0.483, median 0.453, spanning 1.000 → 0.000 with only 2 models perfect. But E-hard is
# 0.18/1.18 = 15% of the composite, so `qwen2.5vl:3b` solved **4%** of the hard band
# (E-hard 0.042) and still scored 0.782 → tagged `coder`. Raising the composite bar cannot
# fix that: at 0.85 it still admits gpt-oss:120b-cloud (E-hard 0.167).
#
# Fix = an explicit E-hard FLOOR, structurally identical to the E1/E2 sub-gates that were
# already here. "Earn at ≥ 0.50" reads as *solves at least half the hard tasks*; the 0.40
# retain band is the same wobble absorber as the composite band. Composite MIN/BAND are
# deliberately UNCHANGED — with a hard floor they mostly stop binding, but they still guard
# the model that somehow aces E-hard and cannot do the basics.
# Effect: 20 tagged → 10 earning + 3 retained. Scores do not move, only the tag, so this is
# NOT a BATTERY_REVISION bump — re-apply offline with `aptitude.py --battery E --regate`.
E_CODER_COMPOSITE_MIN  = 0.70  # earn the tag
E_CODER_COMPOSITE_BAND = 0.65  # retain a tagged model down to here; drop below it
E_CODER_GENERATE_MIN   = 0.50  # E1 mean — must be able to write, not just pattern-match
E_CODER_HARD_MIN       = 0.50  # E-hard mean — must solve at least half the discriminator band
E_CODER_HARD_BAND      = 0.40  # retain a tagged model down to here
#   plus: debug_fix (E2 mean) must be > 0 — must be able to repair, not just generate

# Conversational-consistency toolkit (suites/consistency) — Battery F.
sys.path.insert(0, str(REPO / "suites" / "consistency"))
import style as _fstyle   # noqa: E402

# Prompt-elasticity toolkit (suites/elasticity) — Battery F-elastic (opt-in sibling of F).
sys.path.insert(0, str(REPO / "suites" / "elasticity"))
import adherence as _elastic   # noqa: E402

# Battery F composite weights — PROVISIONAL until the σ gate passes (do not tune before).
# F2 stance-flip + F3 style-drift are the robust dims; the F1 lexicon is the fragile link
# (gameable, model-family-specific) so it weighs least.
F_WEIGHTS = {"F2": 0.30, "F3": 0.25, "F1": 0.15, "F4": 0.15, "F5": 0.15}
F_MAX_TOKENS = 500

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

def chat(model, messages, max_tokens=600, think=None, tools=None, retry_on_truncation=False):
    """think=None (default) → the model's arm for this invocation (v3: lever + budget allowance).
    An explicit bool/level pins the lever with NO allowance — diagnostics only, never for scoring."""
    payload = {
        "model":    model,
        "messages": messages,
        "stream":   False,
        "options":  {"num_ctx": NUM_CTX},
    }
    if max_tokens:
        payload["options"]["num_predict"] = max_tokens
    if tools:
        payload["tools"] = tools
    if think is None:
        apply_think(payload, model, resolve_arm(model, ARM_REQ) or "direct")
    else:
        payload["think"] = think
    def _post(pl):
        t0 = time.time()
        r  = requests.post(f"{ollama_host}/api/chat", json=pl, timeout=budget_timeout(pl, TIMEOUT))
        if r.status_code == 400 and "think" in pl:
            print(f"\n  ⚠  {model}: think parameter rejected (400) — retrying without it", flush=True)
            pl = {k: v for k, v in pl.items() if k != "think"}
            t0 = time.time()
            r  = requests.post(f"{ollama_host}/api/chat", json=pl, timeout=budget_timeout(pl, TIMEOUT))
        wall = time.time() - t0
        r.raise_for_status()
        return r.json(), wall

    # Retry once with headroom if the model burned the whole budget thinking (bench_utils).
    return post_with_budget_retry(payload, _post, label=model,
                                  retry_on_truncation=retry_on_truncation)

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

    # B5 — Think arm (v3): the overwhelmed prompt at the model's measured arm. The old off/on toggle was
    # budget-confounded (docs/think-spec.md); the direct/think delta now comes from the standard suite.
    print("\n  [B5] Think arm (overwhelmed at the measured arm)...", flush=True)
    data, wall = chat(model_name, sys_full + [{"role": "user", "content": OVERWHELMED_PROMPT}], max_tokens=800)
    text = data.get("message", {}).get("content", "")
    m = metrics(text, wall, tps(data))
    m["response"] = text
    m.update(reply_stats(data))
    result["tests"]["b5_think_arm"] = m
    sigs = [k for k, v in m["signals"].items() if v]
    print(f"    {m['words']}w  wall={wall:.1f}s  thinking_chars={m['thinking_chars']}  [{', '.join(sigs) or 'none'}]", flush=True)
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

    lines += ["### B5 — Think arm (overwhelmed at the measured arm)", ""]
    hdr3 = "| | " + " | ".join(f"`{r['model']}`" for r in results) + " |"
    sep3 = "|-|" + "|".join(["---"] * len(results)) + "|"
    lines += [hdr3, sep3]
    lines.append("| arm | " + " | ".join(f"{r.get('arm','direct')}/{r.get('think_lever','false')}" for r in results) + " |")
    lines.append("| reply | " + " | ".join(
        f"{r['tests'].get('b5_think_arm', {}).get('words', '—')}w "
        f"({r['tests'].get('b5_think_arm', {}).get('wall_s', '—')}s · think {r['tests'].get('b5_think_arm', {}).get('thinking_chars', '—')}c)"
        for r in results) + " |")
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

        lines += ["### B5 — Think arm", ""]
        t = r["tests"].get("b5_think_arm", {})
        lines += [
            f"**{r.get('arm','direct')}/{r.get('think_lever','false')}** ({t.get('words','?')}w · wall={t.get('wall_s','?')}s · "
            f"tps={t.get('tps','?')} · thinking_chars={t.get('thinking_chars','—')} · signals: {_sig_str(t)})",
            "", t.get("response", "—"), "",
        ]

    out_md.write_text("\n".join(lines))
    print(f"MD → {out_md}", flush=True)


# ── Battery A data ────────────────────────────────────────────────────────────

PROMPT_CLASSIFY_MINIMAL = (
    "Route queries. Reply with one word: chat, research, code, or tool."
)

PROMPT_CLASSIFY_STANDARD = """\
You are a query router. Classify each user message into exactly one category:
  chat      — conversation, simple facts, emotional support, personal questions
  research  — complex technical questions requiring depth, synthesis, or analysis
  code      — programming, debugging, code generation, technical implementation
  tool      — calculations or tasks that explicitly require tool/function calls

Respond with ONLY the category name. No explanation. No punctuation. One word."""

PROMPT_CLASSIFY_VERBOSE = """\
You are an intelligent query routing assistant for a multi-agent AI system.
Your sole job is to classify incoming user queries and direct them to the right agent.

The four available agents are:
  chat     — handles general conversation, greetings, simple factual lookups, emotional support,
             and casual questions that don't require deep research or technical implementation
  research — handles complex technical, scientific, or analytical questions that benefit from
             depth, multi-source synthesis, comparison frameworks, or domain expertise
  code     — handles programming tasks including writing functions, debugging errors, explaining
             code, refactoring, and any software engineering or technical implementation work
  tool     — handles tasks requiring an external function call: calculations, unit conversions,
             data lookups, and operations with well-defined numeric or structured inputs/outputs

Rules:
  1. Reply with exactly ONE word from: chat, research, code, tool
  2. Do not explain your choice
  3. Do not add punctuation
  4. When in doubt between research and chat, prefer chat for simple factual questions

Examples:
  "Hello there" → chat
  "Explain transformer attention" → research
  "Write a merge sort in Python" → code
  "Calculate 17 × 23" → tool"""

CLASSIFY_QUERIES = [
    ("Hey, how's it going?",                                                   "chat"),
    ("I'm feeling overwhelmed with my workload.",                              "chat"),
    ("What year did World War II end?",                                        "chat"),
    ("Can you recommend a good book to read?",                                 "chat"),
    ("Explain the tradeoffs between RDBMS and NoSQL databases in depth.",      "research"),
    ("Summarise how attention mechanisms work in transformer models.",         "research"),
    ("Tell me about the history of the Byzantine Empire.",                     "research"),
    ("Write a Python function to merge two sorted lists.",                     "code"),
    ("There's a TypeError in my code: print('count: ' + 5)",                  "code"),
    ("Use the calculate tool to compute 17 × 23.",                             "tool"),
]

BREVITY_QUERIES = [
    ("What colour is grass? Answer in one word only.",               3),
    ("How many days are in a week? Reply with a number only.",       3),
    ("Name the largest planet in our solar system. One word.",       3),
    ("Is Python a programming language? Answer yes or no only.",     3),
    ("What is 2 + 2? Reply with the number only.",                   3),
]

FALSE_ESCALATION_QUERIES = [
    "Hello there!",
    "What is 2 + 2?",
    "I need a hug.",
    "What day comes after Monday?",
    "My name is Alex.",
]


def _classify_response(text):
    """Extract the route label from a classifier response (first word, lowercased)."""
    word = text.strip().split()[0].lower().rstrip(".,!:") if text.strip() else ""
    return word if word in {"chat", "research", "code", "tool"} else text.strip().lower()[:20]


# ── Battery A — Router ────────────────────────────────────────────────────────

def run_battery_a(model_name):
    result = {"model": model_name, "battery": "A", "tests": {}}

    print(f"\n{'='*60}", flush=True)
    print(f"BATTERY A: {model_name}", flush=True)
    print("=" * 60, flush=True)

    sys_std = [{"role": "system", "content": PROMPT_CLASSIFY_STANDARD}]
    sys_min = [{"role": "system", "content": PROMPT_CLASSIFY_MINIMAL}]
    sys_vrb = [{"role": "system", "content": PROMPT_CLASSIFY_VERBOSE}]

    # A1 — classify_10 (standard prompt)
    print("\n  [A1] classify_10 (standard prompt)...", flush=True)
    a1_turns = []
    correct  = 0
    tps_vals = []
    for query, expected in CLASSIFY_QUERIES:
        data, wall = chat(model_name, sys_std + [{"role": "user", "content": query}],
                          max_tokens=20)
        text  = data.get("message", {}).get("content", "")
        label = _classify_response(text)
        ok    = label == expected
        t     = tps(data)
        if ok:
            correct += 1
        if t:
            tps_vals.append(t)
        a1_turns.append({
            "query": query, "expected": expected,
            "got": label, "correct": ok, "tps": t, "raw": text,
        })
        mark = "✓" if ok else f"✗ (got '{label}', want '{expected}')"
        print(f"    {mark}  {query[:55]}", flush=True)

    a1_acc     = round(correct / len(CLASSIFY_QUERIES), 3)
    a1_avg_tps = round(sum(tps_vals) / len(tps_vals), 1) if tps_vals else None
    result["tests"]["a1_classify"] = {
        "accuracy": a1_acc, "correct": correct, "total": len(CLASSIFY_QUERIES),
        "avg_tps": a1_avg_tps, "turns": a1_turns,
    }
    print(f"  → accuracy {correct}/{len(CLASSIFY_QUERIES)}  avg {a1_avg_tps} tok/s", flush=True)

    # A2 — brevity_floor
    print("\n  [A2] brevity_floor (1-word/number answers)...", flush=True)
    a2_turns = []
    brief_ok = 0
    for query, max_words in BREVITY_QUERIES:
        data, wall = chat(model_name, sys_std + [{"role": "user", "content": query}],
                          max_tokens=20)
        text  = data.get("message", {}).get("content", "").strip()
        words = len(text.split())
        ok    = words <= max_words
        if ok:
            brief_ok += 1
        a2_turns.append({"query": query, "response": text, "words": words, "brief": ok})
        mark = "✓" if ok else f"✗ ({words}w)"
        print(f"    {mark}  [{words}w] {text[:40]}", flush=True)

    result["tests"]["a2_brevity"] = {
        "score": brief_ok, "total": len(BREVITY_QUERIES), "turns": a2_turns,
    }

    # A3 — prompt_minimal (classify_10 × 3 prompt weights)
    print("\n  [A3] prompt_minimal (minimal / standard / verbose)...", flush=True)
    a3_configs = [
        ("minimal",  sys_min, PROMPT_CLASSIFY_MINIMAL),
        ("standard", sys_std, PROMPT_CLASSIFY_STANDARD),
        ("verbose",  sys_vrb, PROMPT_CLASSIFY_VERBOSE),
    ]
    a3_results = {}
    for cfg_name, sys_msgs, prompt_text in a3_configs:
        cfg_correct  = 0
        cfg_tps_vals = []
        cfg_words    = []
        cfg_turns    = []
        for query, expected in CLASSIFY_QUERIES:
            data, wall = chat(model_name,
                              sys_msgs + [{"role": "user", "content": query}],
                              max_tokens=20)
            text  = data.get("message", {}).get("content", "")
            label = _classify_response(text)
            ok    = label == expected
            t     = tps(data)
            if ok:
                cfg_correct += 1
            if t:
                cfg_tps_vals.append(t)
            cfg_words.append(len(text.split()))
            cfg_turns.append({
                "query": query, "expected": expected, "got": label, "correct": ok,
            })
        acc     = round(cfg_correct / len(CLASSIFY_QUERIES), 3)
        avg_tps = round(sum(cfg_tps_vals) / len(cfg_tps_vals), 1) if cfg_tps_vals else None
        avg_w   = round(sum(cfg_words) / len(cfg_words), 1)        if cfg_words    else None
        a3_results[cfg_name] = {
            "accuracy": acc, "correct": cfg_correct, "total": len(CLASSIFY_QUERIES),
            "avg_tps": avg_tps, "avg_response_words": avg_w,
            "prompt_words": len(prompt_text.split()), "turns": cfg_turns,
        }
        print(
            f"    [{cfg_name:<8}] prompt={len(prompt_text.split()):3d}w  "
            f"acc={cfg_correct}/{len(CLASSIFY_QUERIES)}  "
            f"tps={avg_tps}  reply={avg_w}w avg",
            flush=True,
        )
    result["tests"]["a3_prompt_minimal"] = a3_results

    # A4 — false_escalation
    print("\n  [A4] false_escalation (trivial queries, should not be 'research')...",
          flush=True)
    a4_turns    = []
    escalations = 0
    for query in FALSE_ESCALATION_QUERIES:
        data, wall = chat(model_name, sys_std + [{"role": "user", "content": query}],
                          max_tokens=20)
        text      = data.get("message", {}).get("content", "")
        label     = _classify_response(text)
        escalated = label == "research"
        if escalated:
            escalations += 1
        a4_turns.append({"query": query, "got": label, "escalated": escalated, "raw": text})
        mark = "⚠ ESCALATED" if escalated else f"✓ ({label})"
        print(f"    {mark}  {query}", flush=True)

    result["tests"]["a4_false_escalation"] = {
        "escalation_rate": round(escalations / len(FALSE_ESCALATION_QUERIES), 3),
        "escalations": escalations,
        "total": len(FALSE_ESCALATION_QUERIES),
        "turns": a4_turns,
    }
    print(f"  → false escalation rate: {escalations}/{len(FALSE_ESCALATION_QUERIES)}", flush=True)

    return result


# ── Markdown summary — Battery A ──────────────────────────────────────────────

def write_battery_a_summary(results, out_md: Path, fast_mode=False):
    flag  = " ⚠ FAST MODE" if fast_mode else ""
    n     = len(CLASSIFY_QUERIES)
    lines = [
        f"# Aptitude Battery A — Router{flag}", "",
        "Models: " + ", ".join("`" + r["model"] + "`" for r in results),
        f"`num_ctx={NUM_CTX}` | classification routing prompt", "", "---", "",
    ]

    lines += ["## A1 — classify_10  (accuracy / avg tok/s)", ""]
    hdr = "| Model | Accuracy | Avg tok/s |"
    sep = "|-------|----------|-----------|"
    lines += [hdr, sep]
    for r in results:
        a1 = r["tests"].get("a1_classify", {})
        lines.append(
            f"| `{r['model']}` | {a1.get('correct','?')}/{a1.get('total','?')} "
            f"({round(a1.get('accuracy',0)*100)}%) | {a1.get('avg_tps','—')} |"
        )
    lines.append("")

    lines += ["## A2 — brevity_floor", ""]
    hdr2 = "| Model | Score |"
    sep2 = "|-------|-------|"
    lines += [hdr2, sep2]
    for r in results:
        a2 = r["tests"].get("a2_brevity", {})
        lines.append(f"| `{r['model']}` | {a2.get('score','?')}/{a2.get('total','?')} |")
    lines.append("")

    lines += ["## A3 — prompt_minimal  (accuracy × prompt weight)", ""]
    hdr3 = "| Model | minimal | standard | verbose | tps: min→std |"
    sep3 = "|-------|---------|----------|---------|--------------|"
    lines += [hdr3, sep3]
    for r in results:
        a3  = r["tests"].get("a3_prompt_minimal", {})
        def _acc(k):
            c = a3.get(k, {})
            return f"{c.get('correct','?')}/{n}" if c else "—"
        mn_tps = a3.get("minimal",  {}).get("avg_tps") or 0
        st_tps = a3.get("standard", {}).get("avg_tps") or 0
        gain   = f"+{round(mn_tps - st_tps, 1)}" if mn_tps and st_tps else "—"
        lines.append(
            f"| `{r['model']}` | {_acc('minimal')} | {_acc('standard')} | {_acc('verbose')} "
            f"| {gain} tok/s |"
        )
    lines.append("")

    lines += ["## A4 — false_escalation rate", ""]
    hdr4 = "| Model | Escalation rate | Escalated |"
    sep4 = "|-------|-----------------|-----------|"
    lines += [hdr4, sep4]
    for r in results:
        a4 = r["tests"].get("a4_false_escalation", {})
        lines.append(
            f"| `{r['model']}` | {a4.get('escalation_rate','—')} "
            f"| {a4.get('escalations','?')}/{a4.get('total','?')} |"
        )
    lines.append("")

    for r in results:
        lines += ["---", "", f"## `{r['model']}`", ""]

        lines += ["### A1 — classify_10", ""]
        for t in r["tests"].get("a1_classify", {}).get("turns", []):
            mark = "✓" if t["correct"] else "✗"
            lines.append(
                f"- {mark} `{t['query'][:60]}` → got **{t['got']}** (expected {t['expected']})"
            )
        lines.append("")

        lines += ["### A2 — brevity_floor", ""]
        for t in r["tests"].get("a2_brevity", {}).get("turns", []):
            mark = "✓" if t["brief"] else "✗"
            lines.append(
                f"- {mark} [{t['words']}w] `{t['query'][:55]}` → *{t['response'][:40]}*"
            )
        lines.append("")

        lines += ["### A3 — prompt_minimal", ""]
        a3 = r["tests"].get("a3_prompt_minimal", {})
        for cfg in ["minimal", "standard", "verbose"]:
            c = a3.get(cfg, {})
            lines.append(
                f"**{cfg}** ({c.get('prompt_words','?')} prompt-words)  "
                f"acc={c.get('correct','?')}/{n}  "
                f"tps={c.get('avg_tps','?')}  "
                f"reply_avg={c.get('avg_response_words','?')}w"
            )
            for t in c.get("turns", []):
                mark = "✓" if t["correct"] else "✗"
                lines.append(f"  - {mark} `{t['query'][:50]}` → **{t['got']}**")
        lines.append("")

        lines += ["### A4 — false_escalation", ""]
        for t in r["tests"].get("a4_false_escalation", {}).get("turns", []):
            mark = "⚠" if t["escalated"] else "✓"
            lines.append(f"- {mark} `{t['query']}` → **{t['got']}**")
        lines.append("")

    out_md.write_text("\n".join(lines))
    print(f"MD → {out_md}", flush=True)


# ── Battery C data ────────────────────────────────────────────────────────────

JPEG_PROMPT = (
    "Give a concise, not-too-technical but detailed comparison of "
    "JPEG, JPEG-2000, and JPEG-XL formats."
)

RAG_DEEP_PROMPT = (
    "What's the real tradeoff between RAG and fine-tuning when adapting an LLM to a new domain? "
    "Give 5 concrete examples for each approach — real use-cases where one clearly wins. "
    "Skip the textbook answer."
)

SYNTHESIS_SOURCES = [
    (
        "Source A (Anthropic blog, 2024): Constitutional AI introduces a self-critique loop — "
        "the model evaluates its own responses against a list of principles and rewrites "
        "outputs that violate them. This reduces the need for human feedback on harmful outputs."
    ),
    (
        "Source B (DeepMind paper, 2024): RLHF remains the dominant alignment technique but "
        "suffers from reward hacking — models learn to score high on the reward model without "
        "actually improving on the intended objective. Careful reward model design is critical."
    ),
    (
        "Source C (Stanford HAI report, 2024): Alignment techniques diverge sharply in "
        "scalability. Constitutional AI and RLAIF reduce human labelling costs dramatically, "
        "while RLHF costs grow linearly with the number of preference comparisons required."
    ),
]

SYNTHESIS_PROMPT = (
    "You will be given three source excerpts on AI alignment. "
    "Write a single synthesis paragraph (150–200 words) that integrates the key ideas, "
    "identifies the central tension across sources, and states a clear conclusion. "
    "Do not summarise each source separately.\n\n"
    + "\n\n".join(SYNTHESIS_SOURCES)
)

CTX_DEPTH_LEVELS    = [8192, 16384, 32768]
NUM_PREDICT_LEVELS  = [600, 1000, 1500, 2000]


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


def _think_diagnosis(raw_api: dict) -> str:
    """Classify what happened with a think=True call.

    Returns one of:
      think_ok         — <think> block present, content generated after it
      think_empty      — <think> block present, content is empty / near-empty (≤10 words)
      think_block_only — <think> block detected in raw message but nothing after
      no_think_block   — no <think> tags at all (model ignored think=True)
      api_error        — response parsing failed
    """
    try:
        msg      = raw_api.get("message", {})
        thinking = msg.get("thinking", "")
        content  = msg.get("content", "")

        if thinking and content and len(content.split()) > 10:
            return "think_ok"
        if thinking and (not content or len(content.split()) <= 10):
            return "think_empty"

        if "<think>" in content.lower():
            after = re.split(r"</think>", content, flags=re.IGNORECASE)
            post  = after[-1].strip() if len(after) > 1 else ""
            return "think_ok" if len(post.split()) > 10 else "think_block_only"

        return "no_think_block"
    except Exception:
        return "api_error"


def _chat_ctx(model, messages, ctx, max_tokens, think=None):
    """Like chat() but with a custom num_ctx override."""
    payload = {
        "model":    model,
        "messages": messages,
        "stream":   False,
        "options":  {"num_ctx": ctx, "num_predict": max_tokens},
    }
    if think is None:
        apply_think(payload, model, resolve_arm(model, ARM_REQ) or "direct")
    else:
        payload["think"] = think
    t0   = time.time()
    r    = requests.post(f"{ollama_host}/api/chat", json=payload, timeout=budget_timeout(payload, TIMEOUT))
    wall = time.time() - t0
    r.raise_for_status()
    return r.json(), wall


# ── Battery C — Worker Research ───────────────────────────────────────────────

def run_battery_c(model_name):
    result   = {"model": model_name, "battery": "C", "tests": {}}
    sys_full = [{"role": "system", "content": PROMPT_WORKER_FULL}]

    print(f"\n{'='*60}", flush=True)
    print(f"BATTERY C: {model_name}", flush=True)
    print("=" * 60, flush=True)

    # C1 — jpeg_signals at the measured arm (v3). The off/on toggle was budget-confounded
    # (docs/think-spec.md) — the think delta is the standard suite's job now. ⚠ jpeg has a ±2 noise floor.
    print("\n  [C1] jpeg_signals (at the measured arm)...", flush=True)
    data, wall = chat(model_name, sys_full + [{"role": "user", "content": JPEG_PROMPT}], max_tokens=1500)
    text  = data.get("message", {}).get("content", "")
    cov   = _jpeg_coverage(text)
    diag  = _think_diagnosis(data) if (data.get("message") or {}).get("thinking") else "think_off"
    c1_results = {
        "score": cov["score"], "max": cov["max"], "pass": cov["pass"],
        "signals": cov["signals"], "words": word_count(text),
        "wall_s": round(wall, 1), "tps": tps(data), "think_diagnosis": diag,
        "response": text, **reply_stats(data),
    }
    print(f"    {cov['score']}/{cov['max']} signals  {word_count(text)}w  diag={diag}", flush=True)
    for s in [k for k, v in cov["signals"].items() if v]:
        print(f"      ✓ {s}", flush=True)
    result["tests"]["c1_jpeg_signals"] = c1_results

    # C2 — rag_deep: extended RAG vs fine-tuning with 5 examples per approach
    print("\n  [C2] rag_deep (5 examples per approach)...", flush=True)
    data, wall = chat(model_name, sys_full + [{"role": "user", "content": RAG_DEEP_PROMPT}],
                      max_tokens=2000)
    text = data.get("message", {}).get("content", "")
    lo   = text.lower()
    rag_examples = len(re.findall(r"\b(?:rag|retrieval)[^.]*(?:example|case|scenario|use.case)", lo))
    ft_examples  = len(re.findall(r"\b(?:fine.tun|finetuning)[^.]*(?:example|case|scenario|use.case)", lo))
    has_tradeoff = any(x in lo for x in ["tradeoff", "trade-off", "when to use", "vs", "versus", "better when"])
    result["tests"]["c2_rag_deep"] = {
        "words": word_count(text), "wall_s": round(wall, 1), "tps": tps(data),
        "rag_example_hits": rag_examples, "ft_example_hits": ft_examples,
        "has_tradeoff_framing": has_tradeoff, "response": text,
    }
    print(f"    {word_count(text)}w  rag_hits={rag_examples}  ft_hits={ft_examples}  tradeoff={has_tradeoff}",
          flush=True)
    print(f"    → {text[:140].replace(chr(10), ' ')}", flush=True)

    # C3 — synthesis_3src: integrate 3 source excerpts into one paragraph
    print("\n  [C3] synthesis_3src...", flush=True)
    data, wall = chat(model_name, sys_full + [{"role": "user", "content": SYNTHESIS_PROMPT}],
                      max_tokens=600)
    text  = data.get("message", {}).get("content", "")
    lo    = text.lower()
    wc    = word_count(text)
    in_range       = 100 <= wc <= 300
    mentions_all_3 = all(x in lo for x in ["constitutional", "rlhf", "scalab"])
    has_tension    = any(x in lo for x in ["tension", "contrast", "differ", "whereas", "while", "however"])
    no_src_listing = not re.search(r"source [abc]:", lo)
    result["tests"]["c3_synthesis"] = {
        "words": wc, "in_range": in_range, "wall_s": round(wall, 1), "tps": tps(data),
        "mentions_all_3": mentions_all_3, "has_tension": has_tension,
        "no_src_listing": no_src_listing, "response": text,
    }
    print(f"    {wc}w  in_range={in_range}  all_3={mentions_all_3}  tension={has_tension}  no_list={no_src_listing}",
          flush=True)
    print(f"    → {text[:140].replace(chr(10), ' ')}", flush=True)

    # C4 — ctx_depth: JPEG test at 3 context window sizes
    print("\n  [C4] ctx_depth (JPEG at 8192 / 16384 / 32768)...", flush=True)
    c4_results = {}
    for ctx in CTX_DEPTH_LEVELS:
        data, wall = _chat_ctx(model_name,
                               sys_full + [{"role": "user", "content": JPEG_PROMPT}],
                               ctx=ctx, max_tokens=1500)
        text  = data.get("message", {}).get("content", "")
        cov   = _jpeg_coverage(text)
        t     = tps(data)
        c4_results[str(ctx)] = {
            "score": cov["score"], "max": cov["max"], "pass": cov["pass"],
            "signals": cov["signals"], "words": word_count(text),
            "wall_s": round(wall, 1), "tps": t,
        }
        print(f"    [ctx={ctx:>6}]  {cov['score']}/{cov['max']} signals  {word_count(text)}w  {t} tok/s",
              flush=True)
        unload(model_name)
    result["tests"]["c4_ctx_depth"] = c4_results

    # C5 — num_predict_ceiling: JPEG at 4 output budgets
    print("\n  [C5] num_predict_ceiling (600 / 1000 / 1500 / 2000)...", flush=True)
    c5_results = {}
    for cap in NUM_PREDICT_LEVELS:
        data, wall = chat(model_name,
                          sys_full + [{"role": "user", "content": JPEG_PROMPT}],
                          max_tokens=cap)
        text = data.get("message", {}).get("content", "")
        cov  = _jpeg_coverage(text)
        t    = tps(data)
        c5_results[str(cap)] = {
            "score": cov["score"], "pass": cov["pass"],
            "words": word_count(text), "wall_s": round(wall, 1), "tps": t,
        }
        print(f"    [cap={cap:>4}]  {cov['score']}/{cov['max']} signals  {word_count(text)}w  {round(wall,1)}s",
              flush=True)
    result["tests"]["c5_num_predict"] = c5_results

    return result


# ── Markdown summary — Battery C ──────────────────────────────────────────────

def write_battery_c_summary(results, out_md: Path, fast_mode=False):
    flag  = " ⚠ FAST MODE" if fast_mode else ""
    lines = [
        f"# Aptitude Battery C — Worker Research{flag}", "",
        "Models: " + ", ".join("`" + r["model"] + "`" for r in results),
        f"`num_ctx={NUM_CTX}` (baseline) | worker prompt", "", "---", "",
    ]

    lines += ["## C1 — JPEG signals (at the measured arm; ±2 noise floor)", ""]
    hdr = "| Model | arm | score | words | think diagnosis |"
    sep = "|-------|-----|-------|-------|-----------------|"
    lines += [hdr, sep]
    for r in results:
        c1 = r["tests"].get("c1_jpeg_signals", {})
        lines.append(
            f"| `{r['model']}` | {r.get('arm','direct')}/{r.get('think_lever','false')} | {c1.get('score','?')}/7 "
            f"| {c1.get('words','?')}w | {c1.get('think_diagnosis','—')} |"
        )
    lines.append("")

    lines += ["## C4 — ctx_depth (JPEG signal score)", ""]
    hdr4 = "| Model | ctx=8192 | ctx=16384 | ctx=32768 |"
    sep4 = "|-------|----------|-----------|-----------|"
    lines += [hdr4, sep4]
    for r in results:
        c4 = r["tests"].get("c4_ctx_depth", {})
        lines.append(
            f"| `{r['model']}` "
            f"| {c4.get('8192',{}).get('score','?')}/7 "
            f"| {c4.get('16384',{}).get('score','?')}/7 "
            f"| {c4.get('32768',{}).get('score','?')}/7 |"
        )
    lines.append("")

    lines += ["## C5 — num_predict ceiling (JPEG signal score / word count)", ""]
    hdr5 = "| Model | cap=600 | cap=1000 | cap=1500 | cap=2000 |"
    sep5 = "|-------|---------|----------|----------|----------|"
    lines += [hdr5, sep5]
    for r in results:
        c5 = r["tests"].get("c5_num_predict", {})
        def _cell(k):
            d = c5.get(str(k), {})
            return f"{d.get('score','?')}/7  {d.get('words','?')}w" if d else "—"
        lines.append(
            f"| `{r['model']}` | {_cell(600)} | {_cell(1000)} | {_cell(1500)} | {_cell(2000)} |"
        )
    lines.append("")

    for r in results:
        lines += ["---", "", f"## `{r['model']}`", ""]

        lines += ["### C1 — JPEG signals (at the measured arm)", ""]
        d = r["tests"].get("c1_jpeg_signals", {})
        lines += [
            f"**{r.get('arm','direct')}/{r.get('think_lever','false')}** — {d.get('score','?')}/7 signals  "
            f"{d.get('words','?')}w  wall={d.get('wall_s','?')}s  "
            f"tps={d.get('tps','?')}  diagnosis={d.get('think_diagnosis','—')}  thinking_chars={d.get('thinking_chars','—')}",
        ]
        for sig, hit in (d.get("signals") or {}).items():
            lines.append(f"  - {'✓' if hit else '✗'} {sig}")
        lines += ["", d.get("response", "—"), ""]

        c2 = r["tests"].get("c2_rag_deep", {})
        lines += [
            "### C2 — rag_deep", "",
            f"({c2.get('words','?')}w · rag_hits={c2.get('rag_example_hits','?')} "
            f"ft_hits={c2.get('ft_example_hits','?')} "
            f"tradeoff={c2.get('has_tradeoff_framing','?')})",
            "", c2.get("response", "—"), "",
        ]

        c3 = r["tests"].get("c3_synthesis", {})
        lines += [
            "### C3 — synthesis_3src", "",
            f"({c3.get('words','?')}w · in_range={c3.get('in_range','?')} "
            f"all_3={c3.get('mentions_all_3','?')} "
            f"tension={c3.get('has_tension','?')} "
            f"no_list={c3.get('no_src_listing','?')})",
            "", c3.get("response", "—"), "",
        ]

        lines += ["### C4 — ctx_depth", ""]
        c4 = r["tests"].get("c4_ctx_depth", {})
        for ctx in CTX_DEPTH_LEVELS:
            d = c4.get(str(ctx), {})
            lines.append(
                f"- ctx={ctx}  {d.get('score','?')}/7 signals  "
                f"{d.get('words','?')}w  {d.get('tps','?')} tok/s"
            )
        lines.append("")

        lines += ["### C5 — num_predict ceiling", ""]
        c5 = r["tests"].get("c5_num_predict", {})
        for cap in NUM_PREDICT_LEVELS:
            d = c5.get(str(cap), {})
            lines.append(
                f"- cap={cap}  {d.get('score','?')}/7 signals  "
                f"{d.get('words','?')}w  wall={d.get('wall_s','?')}s"
            )
        lines.append("")


    out_md.write_text("\n".join(lines))
    print(f"MD → {out_md}", flush=True)


# ── Battery D data ────────────────────────────────────────────────────────────

TOOL_DEFS_D = [
    {
        "type": "function",
        "function": {
            "name":        "calculate",
            "description": "Evaluate a mathematical expression and return the numeric result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "lookup",
            "description": "Look up the unit price of an item in the product catalog. Returns price in USD.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {"type": "string", "description": "Item name, e.g. 'widget'"}
                },
                "required": ["item"],
            },
        },
    },
]

LOOKUP_CATALOG = {
    "widget":       4.99,
    "gadget":      12.50,
    "gizmo":        7.25,
    "doohickey":    3.75,
    "thingamajig":  9.00,
}

CHAIN_3_PROMPT = (
    "Use the lookup tool to find the unit price of a widget. "
    "Then use the calculate tool to find the total cost for 8 units. "
    "Finally, use the calculate tool again to add 10% tax to that total. "
    "Show me the final answer."
)

PARALLEL_PROMPT = (
    "Use the calculate tool to compute both 17 × 23 and 456 + 789. "
    "Give me both results."
)

# D7 — personality + tool integration: measures tool correctness AND persona voice together.
# This is the practical application test — Alice must use tools while sounding like Alice.
# Scored 0–4: lookup ✓, calculate ✓, correct answer (~$64.97) ✓, persona voice ✓
D7_PROMPT = (
    "I'm thinking of ordering some office supplies. "
    "Can you look up the price of a widget, figure out what 12 of them would cost "
    "with 8.5% sales tax, and let me know if that seems like a reasonable spend "
    "for a small office?"
)
D7_EXPECTED = 64.97   # 4.99 × 12 × 1.085

# D8 — deep cart: 5 lookups + multi-step calculation; tests loop depth and persistence.
# Expected: (4.99×3 + 12.50×2 + 7.25×4 + 3.75×1 + 9.00×2) × 0.9 × 1.08 ≈ $88.18
D8_PROMPT = (
    "I need to place a full office supply order. Look up the unit price for each item "
    "and calculate the total:\n"
    "  - 3 widgets\n"
    "  - 2 gadgets\n"
    "  - 4 gizmos\n"
    "  - 1 doohickey\n"
    "  - 2 thingamajigs\n\n"
    "Apply a 10% discount if the subtotal exceeds $30 (it will). "
    "Then add 8% sales tax. Show me the final amount."
)
D8_ITEMS    = {"widget", "gadget", "gizmo", "doohickey", "thingamajig"}
D8_EXPECTED = 88.18


def _exec_tool(name, args, error_mode=False, error_on=None):
    """Simulate tool execution. Returns a JSON-serialisable result dict.

    error_mode=True        → all tools fail with a service error.
    error_on={'calculate'} → only the named tools fail; others succeed normally.
    """
    if error_mode or (error_on and name in error_on):
        return {"error": "Service unavailable. Please try again later."}
    if name == "calculate":
        try:
            result = eval(args.get("expression", ""), {"__builtins__": {}}, {})
            return {"result": round(float(result), 4)}
        except Exception as e:
            return {"error": str(e)}
    if name == "lookup":
        item  = args.get("item", "").lower().strip()
        price = LOOKUP_CATALOG.get(item)
        return {"item": item, "price_usd": price} if price is not None \
               else {"error": f"Item '{item}' not found in catalog."}
    return {"error": f"Unknown tool: {name}"}


def _tool_loop(model_name, messages, tools, max_steps=6, think=None,
               error_mode=False, error_on=None):
    """Multi-turn tool-calling loop.

    Returns: (final_text, all_tool_calls, steps_taken, wall_total_s, last_raw_api)
    last_raw_api is the final Ollama response dict — pass to _think_diagnosis() as needed.
    """
    history    = list(messages)
    all_calls  = []
    wall_total = 0.0
    last_raw   = {}

    for step in range(max_steps):
        data, wall = chat(model_name, history, max_tokens=800, think=think, tools=tools)
        wall_total += wall
        last_raw    = data
        msg         = data.get("message", {})
        tool_calls  = msg.get("tool_calls", [])

        if not tool_calls:
            return msg.get("content", ""), all_calls, step, wall_total, data

        history.append({
            "role": "assistant", "content": msg.get("content", ""),
            "tool_calls": tool_calls,
        })

        for tc in tool_calls:
            fn   = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            result = _exec_tool(name, args, error_mode=error_mode, error_on=error_on)
            all_calls.append({"tool": name, "args": args, "result": result})
            history.append({"role": "tool", "content": json.dumps(result)})

    return "", all_calls, max_steps, wall_total, last_raw


# ── Battery D — Tool-heavy ────────────────────────────────────────────────────

def run_battery_d(model_name):
    result  = {"model": model_name, "battery": "D", "tests": {}}
    sys_std = [{"role": "system", "content": PROMPT_WORKER_FULL}]

    print(f"\n{'='*60}", flush=True)
    print(f"BATTERY D: {model_name}", flush=True)
    print("=" * 60, flush=True)

    # D1 — chain_3: lookup → calculate → calculate (3 sequential tool calls)
    print("\n  [D1] chain_3 (lookup widget → calc total → calc tax)...", flush=True)
    text, calls, steps, wall, _ = _tool_loop(
        model_name,
        sys_std + [{"role": "user", "content": CHAIN_3_PROMPT}],
        tools=TOOL_DEFS_D,
    )
    tool_names = [c["tool"] for c in calls]
    lookup_ok  = any(c["tool"] == "lookup" and "widget" in str(c["args"]).lower() for c in calls)
    calc_count = sum(1 for c in calls if c["tool"] == "calculate")
    final_ok   = "43.9" in text or "43.91" in text or "43.912" in text
    result["tests"]["d1_chain_3"] = {
        "tool_sequence": tool_names, "steps": steps, "wall_s": round(wall, 1),
        "lookup_correct": lookup_ok, "calc_count": calc_count, "final_answer_ok": final_ok,
        "calls": calls, "response": text,
    }
    chain_ok = lookup_ok and calc_count >= 2 and final_ok
    print(f"    {'✓' if chain_ok else '✗'}  seq={tool_names}  final_ok={final_ok}  steps={steps}", flush=True)
    print(f"    → {text[:120].replace(chr(10), ' ')}", flush=True)

    # D2 — select_direct: trivial arithmetic, should NOT call tool
    print("\n  [D2] select_direct ('What is 12 × 12?' — should not call tool)...", flush=True)
    data, wall = chat(model_name,
                      sys_std + [{"role": "user", "content": "What is 12 × 12?"}],
                      max_tokens=100, tools=TOOL_DEFS_D)
    text2      = data.get("message", {}).get("content", "")
    tool_calls = data.get("message", {}).get("tool_calls", [])
    no_call    = len(tool_calls) == 0
    has_144    = "144" in text2
    result["tests"]["d2_select_direct"] = {
        "no_tool_call": no_call, "has_correct_answer": has_144,
        "tool_calls": tool_calls, "wall_s": round(wall, 1), "response": text2,
    }
    mark = "✓" if no_call and has_144 else ("✗ called tool" if not no_call else "✗ wrong answer")
    print(f"    {mark}  no_call={no_call}  has_144={has_144}", flush=True)

    # D3 — select_tool: explicit instruction to use tool
    print("\n  [D3] select_tool ('Use the calculate tool to verify: 144 = 12²')...", flush=True)
    data, wall = chat(model_name,
                      sys_std + [{"role": "user", "content": "Use the calculate tool to verify: 144 = 12²"}],
                      max_tokens=200, tools=TOOL_DEFS_D)
    text3      = data.get("message", {}).get("content", "")
    tool_calls = data.get("message", {}).get("tool_calls", [])
    called     = len(tool_calls) > 0
    correct    = called and any(
        "12" in str(tc.get("function", {}).get("arguments", "")) for tc in tool_calls
    )
    result["tests"]["d3_select_tool"] = {
        "called": called, "correct_args": correct,
        "tool_calls": tool_calls, "wall_s": round(wall, 1), "response": text3,
    }
    mark = "✓" if correct else ("✗ no call" if not called else "✗ wrong args")
    print(f"    {mark}  called={called}  correct_args={correct}", flush=True)

    # D4 — error_recovery: all tools return service error
    print("\n  [D4] error_recovery (all tools fail)...", flush=True)
    text4, calls4, steps4, wall4, _ = _tool_loop(
        model_name,
        sys_std + [{"role": "user",
                    "content": "Use the lookup tool to find the unit price of a sprocket."}],
        tools=TOOL_DEFS_D, error_mode=True, max_steps=4,
    )
    lo4         = text4.lower()
    looped      = steps4 >= 3
    reports_err = any(x in lo4 for x in ["unavailable", "error", "unable", "couldn't", "failed", "sorry"])
    invents     = re.search(r'\$\s*\d+\.?\d*', lo4) is not None
    result["tests"]["d4_error_recovery"] = {
        "steps": steps4, "looped": looped,
        "reports_error": reports_err, "invents_price": invents,
        "calls": calls4, "wall_s": round(wall4, 1), "response": text4,
    }
    grade = "loop" if looped else ("invents" if invents else ("reports" if reports_err else "unknown"))
    print(f"    grade={grade}  steps={steps4}  reports={reports_err}  invents={invents}", flush=True)
    print(f"    → {text4[:120].replace(chr(10), ' ')}", flush=True)

    # D4b — partial_error: lookup succeeds, calculate fails.
    # Tests graceful degradation — use the partial result you have, report what failed.
    print("\n  [D4b] partial_error (lookup ok, calculate fails)...", flush=True)
    text4b, calls4b, steps4b, wall4b, _ = _tool_loop(
        model_name,
        sys_std + [{"role": "user",
                    "content": "Use the lookup tool to find the price of a widget, "
                               "then use the calculate tool to find the total cost for 12 units."}],
        tools=TOOL_DEFS_D, error_on={"calculate"}, max_steps=4,
    )
    lo4b            = text4b.lower()
    lookup_got4b    = any(c["tool"] == "lookup" and "price_usd" in str(c.get("result", {}))
                          for c in calls4b)
    calc_tried4b    = any(c["tool"] == "calculate" for c in calls4b)
    # plausible totals ($58–$64) that only appear if the model calculated mentally despite the error
    invents_total4b = bool(re.search(r'\b5[89]\b|\b6[0-4]\b', text4b))
    reports_partial = any(x in lo4b for x in ["4.99", "unavailable", "error", "unable", "couldn't"])
    grade4b = (
        "graceful" if (lookup_got4b and not invents_total4b and reports_partial) else
        "invents"  if invents_total4b else
        "unclear"
    )
    result["tests"]["d4b_partial_error"] = {
        "lookup_succeeded": lookup_got4b, "calc_attempted": calc_tried4b,
        "invents_total": invents_total4b, "reports_partial": reports_partial,
        "grade": grade4b, "steps": steps4b,
        "calls": calls4b, "wall_s": round(wall4b, 1), "response": text4b,
    }
    print(f"    grade={grade4b}  lookup_ok={lookup_got4b}  calc_tried={calc_tried4b}  invents={invents_total4b}", flush=True)
    print(f"    → {text4b[:120].replace(chr(10), ' ')}", flush=True)

    # D6 — parallel_tools: two independent calculations in one request
    print("\n  [D6] parallel_tools (two calcs in one turn)...", flush=True)
    text6, calls6, steps6, wall6, _ = _tool_loop(
        model_name,
        sys_std + [{"role": "user", "content": PARALLEL_PROMPT}],
        tools=TOOL_DEFS_D,
    )
    calc_calls = [c for c in calls6 if c["tool"] == "calculate"]
    both_done  = len(calc_calls) >= 2
    has_391    = "391" in text6
    has_1245   = "1245" in text6
    result["tests"]["d6_parallel"] = {
        "calc_call_count": len(calc_calls), "both_done": both_done,
        "has_391": has_391, "has_1245": has_1245,
        "calls": calls6, "wall_s": round(wall6, 1), "response": text6,
    }
    mark = "✓" if both_done and has_391 and has_1245 else "✗"
    print(f"    {mark}  calc_count={len(calc_calls)}  391={has_391}  1245={has_1245}", flush=True)

    # D7 — personality_tool: tool correctness + persona voice in the same response.
    # The practical application test: a model must use tools AND maintain character.
    # Scored 0–4: lookup ✓, calculate ✓, correct final answer ✓, persona voice ✓
    print("\n  [D7] personality_tool (widget order + persona voice)...", flush=True)
    text7, calls7, steps7, wall7, _ = _tool_loop(
        model_name,
        sys_std + [{"role": "user", "content": D7_PROMPT}],
        tools=TOOL_DEFS_D,
    )
    lookup_ok7  = any(c["tool"] == "lookup" and "widget" in str(c["args"]).lower() for c in calls7)
    calc_ok7    = any(c["tool"] == "calculate" for c in calls7)
    final_ok7   = any(x in text7 for x in ["64.9", "64.97", "64.98", "65.0", "65.00"])
    voice7      = persona_signals(text7)
    voice_ok7   = voice7["direct_voice"] or voice7["warmth"] or voice7["wit"]
    d7_score    = sum([lookup_ok7, calc_ok7, final_ok7, voice_ok7])
    result["tests"]["d7_personality_tool"] = {
        "lookup_correct": lookup_ok7, "calc_called": calc_ok7,
        "final_answer_ok": final_ok7, "voice_ok": voice_ok7,
        "voice_signals": voice7, "score": d7_score, "max": 4,
        "steps": steps7, "wall_s": round(wall7, 1),
        "calls": calls7, "response": text7,
    }
    sigs7 = [k for k, v in voice7.items() if v]
    print(f"    {d7_score}/4  lookup={lookup_ok7}  calc={calc_ok7}  answer={final_ok7}  voice={voice_ok7}", flush=True)
    print(f"    voice: {sigs7 or 'none'}", flush=True)
    print(f"    → {text7[:120].replace(chr(10), ' ')}", flush=True)

    # D8 — deep_cart: 5-item order requiring 8+ tool calls; tests loop depth and persistence.
    # Expected final: ~$88.18 (subtotal $90.72 × 0.9 discount × 1.08 tax)
    # Scored 0–4: all items looked up ✓, ≥3 calculate calls ✓, correct answer ✓, completed ✓
    print("\n  [D8] deep_cart (5-item order, 8+ tool calls, discount + tax)...", flush=True)
    text8, calls8, steps8, wall8, _ = _tool_loop(
        model_name,
        sys_std + [{"role": "user", "content": D8_PROMPT}],
        tools=TOOL_DEFS_D, max_steps=15,
    )
    items_looked = {c["args"].get("item", "").lower().strip()
                    for c in calls8 if c["tool"] == "lookup"}
    all_looked   = D8_ITEMS.issubset(items_looked)
    calc_count8  = sum(1 for c in calls8 if c["tool"] == "calculate")
    final_ok8    = any(x in text8 for x in ["88.18", "88.17", "88.2", "88.1"])
    completed    = steps8 < 15   # didn't exhaust the step budget
    d8_score     = sum([all_looked, calc_count8 >= 3, final_ok8, completed])
    result["tests"]["d8_deep_cart"] = {
        "items_looked_up": sorted(items_looked), "all_items_found": all_looked,
        "calc_call_count": calc_count8, "final_answer_ok": final_ok8,
        "completed_in_budget": completed, "score": d8_score, "max": 4,
        "tool_call_total": len(calls8), "steps": steps8, "wall_s": round(wall8, 1),
        "calls": calls8, "response": text8,
    }
    print(f"    {d8_score}/4  all_items={all_looked}  calcs={calc_count8}  "
          f"final={final_ok8}  calls={len(calls8)}  steps={steps8}", flush=True)
    print(f"    → {text8[:120].replace(chr(10), ' ')}", flush=True)

    return result


# ── Markdown summary — Battery D ──────────────────────────────────────────────

def write_battery_d_summary(results, out_md: Path, fast_mode=False):
    flag  = " ⚠ FAST MODE" if fast_mode else ""
    lines = [
        f"# Aptitude Battery D — Worker Tool-heavy{flag}", "",
        "Models: " + ", ".join("`" + r["model"] + "`" for r in results),
        f"`num_ctx={NUM_CTX}` | tools: calculate + lookup", "", "---", "",
    ]

    # Overview table
    lines += ["## Overview", ""]
    hdr = ("| Model | D1 chain | D2 direct | D3 tool | D4 full | "
           "D4b partial | D6 parallel | D7 voice | D8 cart |")
    sep = ("|-------|----------|-----------|---------|---------|"
           "------------|-------------|----------|---------|")
    lines += [hdr, sep]
    for r in results:
        d1  = r["tests"].get("d1_chain_3",          {})
        d2  = r["tests"].get("d2_select_direct",    {})
        d3  = r["tests"].get("d3_select_tool",      {})
        d4  = r["tests"].get("d4_error_recovery",   {})
        d4b = r["tests"].get("d4b_partial_error",   {})
        d6  = r["tests"].get("d6_parallel",         {})
        d7  = r["tests"].get("d7_personality_tool", {})
        d8  = r["tests"].get("d8_deep_cart",        {})

        def _yn(v): return "✓" if v else "✗"
        chain_ok  = d1.get("lookup_correct") and d1.get("calc_count", 0) >= 2 and d1.get("final_answer_ok")
        d4_grade  = ("loop"    if d4.get("looped") else
                     "invents" if d4.get("invents_price") else
                     "reports" if d4.get("reports_error") else "?")
        d4b_grade = d4b.get("grade", "—")
        both_done = d6.get("both_done") and d6.get("has_391") and d6.get("has_1245")

        lines.append(
            f"| `{r['model']}` | {_yn(chain_ok)} | "
            f"{_yn(d2.get('no_tool_call') and d2.get('has_correct_answer'))} | "
            f"{_yn(d3.get('correct_args'))} | {d4_grade} | "
            f"{d4b_grade} | {_yn(both_done)} | "
            f"{d7.get('score','?')}/4 | {d8.get('score','?')}/4 |"
        )
    lines.append("")

    # D7 voice breakdown
    lines += ["## D7 — personality_tool (score / voice signals)", ""]
    hdr7 = "| Model | Score | lookup | calc | answer | voice | voice signals |"
    sep7 = "|-------|-------|--------|------|--------|-------|---------------|"
    lines += [hdr7, sep7]
    for r in results:
        d7   = r["tests"].get("d7_personality_tool", {})
        vs   = d7.get("voice_signals", {})
        sigs = ", ".join(k for k, v in vs.items() if v) or "none"
        def _yn(v): return "✓" if v else "✗"
        lines.append(
            f"| `{r['model']}` | {d7.get('score','?')}/4 "
            f"| {_yn(d7.get('lookup_correct'))} "
            f"| {_yn(d7.get('calc_called'))} "
            f"| {_yn(d7.get('final_answer_ok'))} "
            f"| {_yn(d7.get('voice_ok'))} "
            f"| {sigs} |"
        )
    lines.append("")

    # D8 deep cart summary
    lines += ["## D8 — deep_cart (5-item order, discount + tax)", ""]
    hdr8 = "| Model | Score | all items | calcs | final ok | tool calls | steps |"
    sep8 = "|-------|-------|-----------|-------|----------|------------|-------|"
    lines += [hdr8, sep8]
    for r in results:
        d8 = r["tests"].get("d8_deep_cart", {})
        def _yn(v): return "✓" if v else "✗"
        lines.append(
            f"| `{r['model']}` | {d8.get('score','?')}/4 "
            f"| {_yn(d8.get('all_items_found'))} "
            f"| {d8.get('calc_call_count','?')} "
            f"| {_yn(d8.get('final_answer_ok'))} "
            f"| {d8.get('tool_call_total','?')} "
            f"| {d8.get('steps','?')} |"
        )
    lines.append("")

    # Per-model detail
    for r in results:
        lines += ["---", "", f"## `{r['model']}`", ""]

        for key, label in [
            ("d1_chain_3",       "D1 — chain_3"),
            ("d2_select_direct", "D2 — select_direct"),
            ("d3_select_tool",   "D3 — select_tool"),
        ]:
            d = r["tests"].get(key, {})
            lines += [f"### {label}", ""]
            for c in d.get("calls", []):
                lines.append(f"- tool=`{c['tool']}` args={c['args']} → {c['result']}")
            if d.get("response"):
                lines += ["", d["response"], ""]
            lines.append("")

        d4 = r["tests"].get("d4_error_recovery", {})
        grade4 = ("loop"    if d4.get("looped") else
                  "invents" if d4.get("invents_price") else
                  "reports" if d4.get("reports_error") else "?")
        lines += [
            "### D4 — error_recovery (all tools fail)", "",
            f"grade={grade4}  steps={d4.get('steps','?')}  "
            f"reports={d4.get('reports_error','?')}  invents={d4.get('invents_price','?')}",
            "", d4.get("response", "—"), "",
        ]

        d4b = r["tests"].get("d4b_partial_error", {})
        lines += [
            "### D4b — partial_error (lookup ok, calculate fails)", "",
            f"grade={d4b.get('grade','?')}  lookup_ok={d4b.get('lookup_succeeded','?')}  "
            f"calc_tried={d4b.get('calc_attempted','?')}  invents={d4b.get('invents_total','?')}",
            "", d4b.get("response", "—"), "",
        ]

        d6 = r["tests"].get("d6_parallel", {})
        lines += [
            "### D6 — parallel_tools", "",
            f"calc_count={d6.get('calc_call_count','?')}  391={d6.get('has_391','?')}  1245={d6.get('has_1245','?')}",
            "", d6.get("response", "—"), "",
        ]

        d7   = r["tests"].get("d7_personality_tool", {})
        vs7  = d7.get("voice_signals", {})
        sigs7 = ", ".join(k for k, v in vs7.items() if v) or "none"
        lines += [
            "### D7 — personality_tool", "",
            f"score={d7.get('score','?')}/4  "
            f"lookup={d7.get('lookup_correct','?')}  "
            f"calc={d7.get('calc_called','?')}  "
            f"answer={d7.get('final_answer_ok','?')}  "
            f"voice={d7.get('voice_ok','?')}  signals: {sigs7}",
        ]
        for c in d7.get("calls", []):
            lines.append(f"- tool=`{c['tool']}` args={c['args']} → {c['result']}")
        lines += ["", d7.get("response", "—"), ""]

        d8 = r["tests"].get("d8_deep_cart", {})
        lines += [
            "### D8 — deep_cart", "",
            f"score={d8.get('score','?')}/4  "
            f"items={d8.get('items_looked_up','?')}  "
            f"calcs={d8.get('calc_call_count','?')}  "
            f"final={d8.get('final_answer_ok','?')}  "
            f"calls={d8.get('tool_call_total','?')}  steps={d8.get('steps','?')}",
        ]
        for c in d8.get("calls", []):
            lines.append(f"- tool=`{c['tool']}` args={c['args']} → {c['result']}")
        lines += ["", d8.get("response", "—"), ""]

    out_md.write_text("\n".join(lines))
    print(f"MD → {out_md}", flush=True)


# ── Battery E — Coding (execution-graded) ──────────────────────────────────────

def _load_coding_problems():
    if not _CODING_PROBLEMS_PATH.exists():
        sys.exit("problems.json not found — run: python3 suites/coding/build_problems.py")
    return json.load(_CODING_PROBLEMS_PATH.open())


def _e_spec(problem):
    g = problem.get("gate", {})
    return _codeh.GateSpec(
        require_symbol=g.get("require_symbol"),
        allow=g.get("allow", ()),
        max_lines=g.get("max_lines"),
        forbid_extra_defs=g.get("forbid_extra_defs", False),
    )


def _grade_codegen(response, problem):
    """E1/E2/E7 — extract → gate → execute hidden checks. pass@1 = passed/total."""
    return _codeh.grade_generation(response, problem["checks"], _e_spec(problem), timeout=10)


def _grade_tests(response, problem):
    """E5 — PER-TEST mutation grading. Model writes a `test_*` suite; we run each
    test individually against the clean impl, DROP the ones that fail it (out-of-
    contract / broken assertions), then score kill-rate of the surviving valid tests
    over the planted mutants. A truncated trailing `def` is recovered, not fatal.
    This avoids zeroing an excellent suite because of one bad/out-of-contract test."""
    n_mut = len(problem["mutants"])
    raw = _codeh.extract_code(response)
    # Recover the parseable portion FIRST (a truncated trailing def must not zero the
    # whole suite), THEN safety-gate the recovered code (which parses by construction).
    code = _codeh.recover_parseable(raw)
    g = _codeh.gate(code, _codeh.GateSpec()) if code else _codeh.GateResult(False, "empty")
    if not g.ok:
        return {"code": raw, "gate": {"ok": False, "reason": g.reason, "details": g.details},
                "clean_pass": False, "n_tests": 0, "n_valid": 0, "n_dropped": 0,
                "kills": 0, "n_mutants": n_mut, "score": 0.0}
    clean = _codeh.run_test_functions(problem["clean_impl"], code, timeout=10)
    per_clean = clean.detail.get("per_test", {})
    valid = [name for name, ok in per_clean.items() if ok]   # tests that pass the clean impl
    n_total, n_valid = clean.total, len(valid)
    if n_valid == 0:
        return {"code": code, "gate": {"ok": True}, "clean_pass": False,
                "n_tests": n_total, "n_valid": 0, "n_dropped": n_total,
                "kills": 0, "n_mutants": n_mut, "score": 0.0,
                "clean_error": clean.error}
    kills = 0
    for m in problem["mutants"]:
        rm = _codeh.run_test_functions(m, code, timeout=10)
        if rm.error is not None:
            kills += 1                                        # mutant crashed the suite → killed
        else:
            per_m = rm.detail.get("per_test", {})
            if any(per_m.get(name) is False for name in valid):  # a valid test catches it
                kills += 1
    return {"code": code, "gate": {"ok": True}, "clean_pass": True,
            "n_tests": n_total, "n_valid": n_valid, "n_dropped": n_total - n_valid,
            "kills": kills, "n_mutants": n_mut,
            "score": round(kills / n_mut, 4) if n_mut else 0.0}


def _grade_e3(response, problem):
    """E3 multi-language — route by lang; graceful-skip (score None) if runtime absent."""
    lang = problem["lang"]
    if not _codeh.runtime_available(lang):
        return {"lang": lang, "subtype": problem.get("subtype"),
                "skipped": True, "reason": "runtime_missing", "score": None}
    if lang == "sql":
        out = _codeh.grade_sql(response, problem["setup"], problem["expected"], timeout=10)
    elif lang in ("javascript", "js"):
        out = _codeh.grade_js(response, problem["checks"], _e_spec(problem), timeout=10)
    elif lang == "php":
        out = _codeh.grade_php(response, problem["checks"], _e_spec(problem), timeout=10)
    else:
        out = {"code": "", "lang": lang, "gate": {"ok": False, "reason": "unknown_lang"},
               "exec": None, "score": 0.0}
    out["subtype"] = problem.get("subtype")
    return out


def run_battery_e(model_name):
    result   = {"model": model_name, "battery": "E", "tests": {}}
    sys_std  = [{"role": "system", "content": PROMPT_WORKER_FULL}]
    problems = _load_coding_problems()

    print(f"\n{'='*60}", flush=True)
    print(f"BATTERY E: {model_name}", flush=True)
    print("=" * 60, flush=True)

    cat_scores = {}
    for p in problems:
        cat, pid = p["category"], p["id"]
        tag = f"{cat}/{p['lang']}" if cat == "E3" else cat
        print(f"\n  [{pid}] ({tag})...", flush=True)
        # retry_on_truncation: E5/E-hard emit 2.4k-3.4k chars of code against this 1024 budget, so a
        # think lever can burn it before the code starts and return a 19-char fragment. Opt-in here
        # only — C5 caps output ON PURPOSE to measure the num_predict ceiling (2026-09-01).
        data, wall = chat(model_name, sys_std + [{"role": "user", "content": p["prompt"]}],
                          max_tokens=1024, retry_on_truncation=True)
        resp = data.get("message", {}).get("content", "")
        if cat == "E5":
            detail = _grade_tests(resp, p)
        elif cat == "E3":
            detail = _grade_e3(resp, p)
        elif cat == "E9":
            detail = _codeh.grade_markup(resp, p)
        else:
            detail = _grade_codegen(resp, p)
        detail["category"] = cat
        detail["wall_s"]   = round(wall, 1)
        # Carry the budget-retry telemetry into the result. Without it a starved call is
        # indistinguishable from a wrong answer after the fact: the 08-31 run stored `gate:empty`
        # with no record of whether the retry fired, so the defect could only be found by hand
        # (2026-09-01).
        if data.get("_budget_retry"):
            detail["_budget_retry"] = data["_budget_retry"]
        result["tests"][pid] = detail
        # graceful-skip (runtime/libs missing) → score None, excluded from the mean
        if detail.get("score") is not None:
            cat_scores.setdefault(cat, []).append(detail["score"])

        if detail.get("skipped"):
            print(f"    ⤬ skipped — {detail.get('reason')} ({detail.get('lang')})", flush=True)
        elif cat == "E9":
            gate = detail.get("gate", {})
            gtxt = "ok" if gate.get("ok") else f"gate:{gate.get('reason')}"
            print(f"    score={detail['score']}  {detail.get('lang')} {gtxt}  "
                  f"coverage={detail.get('coverage')}", flush=True)
        elif cat == "E5":
            print(f"    score={detail['score']}  clean_pass={detail.get('clean_pass')}  "
                  f"kills={detail.get('kills')}/{detail.get('n_mutants')}  "
                  f"valid={detail.get('n_valid')}/{detail.get('n_tests')} "
                  f"(dropped {detail.get('n_dropped')})", flush=True)
        else:
            ex   = detail.get("exec") or {}
            gate = detail.get("gate", {})
            gtxt = "ok" if gate.get("ok") else f"gate:{gate.get('reason')}"
            langtag = f"{detail.get('lang')} " if cat == "E3" else ""
            print(f"    score={detail['score']}  {langtag}{gtxt}  "
                  f"pass={ex.get('passed')}/{ex.get('total')}", flush=True)

    cat_mean  = {c: round(sum(v) / len(v), 4) for c, v in cat_scores.items()}
    present   = {c: w for c, w in E_WEIGHTS.items() if c in cat_mean}
    wsum      = sum(present.values()) or 1.0
    composite = round(sum(cat_mean[c] * w for c, w in present.items()) / wsum, 4)
    gen_basic = cat_mean.get("E1", 0.0)
    debug_fix = cat_mean.get("E2", 0.0)
    hard_band = cat_mean.get("E-hard", 0.0)
    coder_eligible = (composite >= E_CODER_COMPOSITE_MIN and debug_fix > 0
                      and gen_basic >= E_CODER_GENERATE_MIN
                      and hard_band >= E_CODER_HARD_MIN)

    result["summary"] = {
        "category_means": cat_mean,
        "composite":      composite,
        "generate_basic": gen_basic,
        "debug_fix":      debug_fix,
        "hard_band":      hard_band,
        "coder_eligible": coder_eligible,
        "threshold": {"composite_min": E_CODER_COMPOSITE_MIN,
                      "composite_band": E_CODER_COMPOSITE_BAND,
                      "generate_min": E_CODER_GENERATE_MIN, "debug_fix_gt": 0,
                      "hard_min": E_CODER_HARD_MIN, "hard_band": E_CODER_HARD_BAND},
    }
    print(f"\n  → composite={composite}  means={cat_mean}  "
          f"coder_eligible={coder_eligible}", flush=True)
    return result


def apply_coder_overlay(results, registry_path):
    """Reconcile the `coder` extended role in models.json with HYSTERESIS.

    EARN at composite ≥ E_CODER_COMPOSITE_MIN AND E-hard ≥ E_CODER_HARD_MIN
    (+ E1 ≥ generate_min + E2 > 0); once tagged, RETAIN while BOTH stay within their
    bands (absorbs MLX run-to-run wobble); DROP as soon as EITHER falls below. The
    E-hard floor is what makes the tag selective — see the threshold block above.
    The model's primary role is never touched — the overlay only stacks/removes
    `coder` on the lane."""
    try:
        models = sort_registry(json.load(registry_path.open()))   # run order: env BENCH_SORT (default size)
    except Exception as e:
        print(f"  coder overlay: cannot read models.json ({e}) — skipped", flush=True)
        return
    by_name = {m["name"]: m for m in models}
    changed = False
    for r in results:
        if "error" in r:
            continue
        s, name = r.get("summary", {}), r["model"]
        entry = by_name.get(name)
        if entry is None:
            continue
        roles = entry.setdefault("extended_roles", [])
        comp = s.get("composite", 0.0) or 0.0
        hard = s.get("hard_band", (s.get("category_means") or {}).get("E-hard", 0.0)) or 0.0
        tagged = "coder" in roles
        in_band = comp >= E_CODER_COMPOSITE_BAND and hard >= E_CODER_HARD_BAND
        if s.get("coder_eligible"):                          # ≥ MIN on both + gates → earn
            if not tagged:
                roles.append("coder")
                changed = True
                print(f"  ★ {name} earned `coder` (composite {comp}, E-hard {hard})", flush=True)
        elif tagged and in_band:                             # within BOTH bands → retain
            print(f"  ~ {name} retained in coder band (composite {comp}, E-hard {hard})", flush=True)
        elif tagged:                                         # either band breached → drop
            roles.remove("coder")
            changed = True
            why = (f"composite {comp} < {E_CODER_COMPOSITE_BAND}"
                   if comp < E_CODER_COMPOSITE_BAND else
                   f"E-hard {hard} < {E_CODER_HARD_BAND}")
            print(f"  ⊘ {name} dropped `coder` ({why})", flush=True)
    if changed:
        registry_path.write_text(json.dumps(models, indent=2))
        print("  models.json `coder` tags reconciled", flush=True)


def regate_f_elastic(dry_run=False):
    """Re-derive F-elastic `hard_adherence` + `verdict` from STORED per-constraint hits — no models.

    A cutoff move, or a change to WHICH constraints the hard meter covers, is grading policy, not
    test content: the rollouts, the replies and the per-constraint satisfaction rates are already on
    disk and a re-run would reproduce them. Re-measuring 19 models for 1h44m to move a threshold
    would only re-roll the rollouts and burn a thermal cycle. This is F-elastic's counterpart to
    `longctx.py --rescore` and `aptitude.py --battery E --regate`; its absence was what made the
    08-24 calibration expensive enough to defer.

    ⚠ Rows are UPDATEd on their ORIGINAL (run_id, created_at). A regate must never re-stamp a score
    under today's run — that strips the provenance resume reads (the 2026-07-11 no-provenance bug).

    ⚠ Only rows carrying `per_rung[].per_constraint` can be re-derived. Pre-two-band rows have no
    hard rung at all; they are SKIPPED, not scored as zero — a missing band is missing data.
    """
    import sqlite3, results_db
    ladder  = _elastic.load_ladder()
    cutoffs = dict(F_ELASTIC_CUTOFFS)
    hard_ids = _elastic.hard_band_ids(ladder)
    con = sqlite3.connect(results_db.DB_PATH)
    rows = con.execute("SELECT run_id, model, metrics, composite, created_at FROM results "
                       "WHERE battery='F-elastic' ORDER BY created_at").fetchall()

    changes, skipped = [], 0
    for run_id, model, metrics, _old_comp, created in rows:
        try:
            rec = json.loads(metrics)
        except Exception:
            skipped += 1; continue
        summ = rec.get("summary") or {}
        per_rung = summ.get("per_rung") or []
        hard_rungs = [r for r in per_rung if r.get("band") == "hard" and r.get("per_constraint")]
        if not hard_rungs:
            skipped += 1; continue                 # pre-two-band row — no hard band to re-derive
        vals = [v for v in (_elastic.hard_band_adherence(r, ladder) for r in hard_rungs)
                if v is not None]
        if not vals:
            skipped += 1; continue
        new_hard = round(statistics.mean(vals), 4)
        old_hard, old_verdict = summ.get("hard_adherence"), summ.get("verdict")
        new_verdict = _elastic.classify(summ.get("prompt_sigma"), summ.get("instruction_adherence"),
                                        cutoffs, hard_adherence=new_hard)
        summ["hard_adherence"] = new_hard
        summ["hard_adherence_allbinary"] = old_hard
        summ["verdict"], summ["cutoffs"] = new_verdict, cutoffs
        rec["summary"] = summ
        changes.append((run_id, model, old_hard, new_hard, old_verdict, new_verdict,
                        json.dumps(rec), results_db.composite_of(rec)))

    moved = [c for c in changes if c[4] != c[5]]
    print(f"Battery F-elastic regate — hard meter scoped to {hard_ids}, "
          f"hard_adherence_hi = {cutoffs['hard_adherence_hi']}", flush=True)
    print(f"  {len(changes)} two-band row(s) re-derived · {skipped} skipped (pre-two-band) · "
          f"{len(moved)} verdict change(s)\n", flush=True)
    if changes:
        print(f"  {'model':<44}{'hard was':>10}{'now':>8}   {'verdict':<18}", flush=True)
        for run_id, model, ow, nw, ov, nv, *_ in sorted(changes, key=lambda c: -(c[3] or 0)):
            flag = "  ← CHANGED" if ov != nv else ""
            print(f"  {model[:43]:<44}{(ow if ow is not None else 0):>10.3f}{nw:>8.3f}   "
                  f"{(ov or '?') + (' → ' + nv if ov != nv else ''):<18}{flag}", flush=True)

    if dry_run:
        print("\n  --dry-run: nothing written.", flush=True)
        con.close()
        return changes

    with con:
        for run_id, model, _ow, _nw, _ov, _nv, blob, comp in changes:
            con.execute("UPDATE results SET metrics=?, composite=? "
                        "WHERE run_id=? AND model=? AND battery='F-elastic'",
                        (blob, comp, run_id, model))
    con.close()
    print(f"\n  → updated {len(changes)} DB row(s) in place (run_id/created_at preserved)", flush=True)
    print("  next: python3 export.py", flush=True)
    return changes


def regate_coder(dry_run=False):
    """Re-apply the `coder` overlay from STORED Battery E results — no model calls.

    A gate threshold is policy, not test content: the per-category means are already on
    disk and a re-run would reproduce them (modulo noise) at ~1.5h for 22 models. So a
    gate re-tune re-derives `coder_eligible` from the persisted `category_means` and
    reconciles models.json, exactly as `longctx.py --rescore` does for Battery G.

    Deliberately NOT a BATTERY_REVISION bump: no score moves, only the tag.

    Reads each model's LATEST E row (results_db.latest), so a targeted re-run of one
    model does not strip the rest of the fleet's tags.

    ⚠ Scoped to the CURRENT registry. `results_db.latest()` returns every model ever
    scored — 70 rows against a 22-model registry here, most of them dropped long ago
    (Rule #17's fossil corollary). ⚠ A row with NO `E-hard` key predates the band
    (added 2026-07-02); that is MISSING data, not a zero, so such a model is reported
    and left untouched rather than silently dropped for "failing" a test it never ran.
    """
    import results_db
    latest = results_db.latest("E")
    if not latest:
        print("  regate: no Battery E rows in the DB — nothing to do", flush=True)
        return

    reg = json.load((REPO / "models.json").open())
    registry = {m["name"] for m in reg}
    tagged_before = {m["name"] for m in reg if "coder" in (m.get("extended_roles") or [])}

    results, rows, ungradeable = [], [], []
    for name, rec in latest.items():
        if name not in registry:                 # fossil — not in the registry any more
            continue
        s  = dict(rec.get("summary") or {})
        cm = s.get("category_means") or {}
        if not cm:
            continue
        comp = s.get("composite", 0.0) or 0.0
        gen, dbg = cm.get("E1", 0.0), cm.get("E2", 0.0)
        hard = cm.get("E-hard")                  # None = pre-two-band row, NOT a zero
        if hard is None:
            ungradeable.append((name, comp))
            continue
        s["hard_band"] = hard
        s["coder_eligible"] = (comp >= E_CODER_COMPOSITE_MIN and dbg > 0
                               and gen >= E_CODER_GENERATE_MIN
                               and hard >= E_CODER_HARD_MIN)
        results.append({"model": name, "summary": s})
        rows.append((comp, hard, name, s["coder_eligible"]))

    rows.sort(reverse=True)
    print(f"Battery E re-gate — composite ≥ {E_CODER_COMPOSITE_MIN} (band {E_CODER_COMPOSITE_BAND}) "
          f"AND E-hard ≥ {E_CODER_HARD_MIN} (band {E_CODER_HARD_BAND})\n", flush=True)
    print(f"  {'model':<44}{'comp':>7}{'E-hard':>8}  {'was':<5}{'now':<9}", flush=True)
    for comp, hard, name, elig in rows:
        was = "coder" if name in tagged_before else "—"
        now = ("earn" if elig else
               ("retain" if (name in tagged_before and comp >= E_CODER_COMPOSITE_BAND
                             and hard >= E_CODER_HARD_BAND) else "drop"))
        mark = "  ⊘" if (name in tagged_before and now == "drop") else ""
        print(f"  {name[:43]:<44}{comp:>7.3f}{hard:>8.3f}  {was:<5}{now:<9}{mark}", flush=True)

    if ungradeable:
        print("\n  ⚠ no E-hard data (pre-2026-07-02 row) — tag left as-is, re-run Battery E:", flush=True)
        for name, comp in sorted(ungradeable):
            print(f"    {name} (composite {comp})", flush=True)

    if dry_run:
        print("\n  --dry-run: models.json untouched.", flush=True)
        return
    print("", flush=True)

    # Write the recomputed verdict back onto each model's OWN latest row (original run_id /
    # created_at preserved — re-stamping under today's run would strip provenance). Without
    # this, export.py keeps publishing the stale `coder_eligible` straight out of the DB and
    # contradicts the models.json tag it just helped set.
    import sqlite3
    con = sqlite3.connect(results_db.DB_PATH)
    n_db = 0
    with con:
        for r in results:
            row = con.execute("SELECT run_id, metrics FROM results WHERE model=? AND battery='E' "
                              "ORDER BY created_at DESC LIMIT 1", (r["model"],)).fetchone()
            if not row:
                continue
            rec = json.loads(row[1])
            rec.setdefault("summary", {}).update({
                "hard_band":      r["summary"]["hard_band"],
                "coder_eligible": r["summary"]["coder_eligible"],
            })
            rec["summary"].setdefault("threshold", {}).update({
                "composite_min": E_CODER_COMPOSITE_MIN, "composite_band": E_CODER_COMPOSITE_BAND,
                "generate_min": E_CODER_GENERATE_MIN, "debug_fix_gt": 0,
                "hard_min": E_CODER_HARD_MIN, "hard_band": E_CODER_HARD_BAND})
            con.execute("UPDATE results SET metrics=? WHERE run_id=? AND model=? AND battery='E'",
                        (json.dumps(rec), row[0], r["model"]))
            n_db += 1
    con.close()
    print(f"  → re-stamped {n_db} DB row(s) with the new verdict (run_id preserved)", flush=True)

    apply_coder_overlay(results, REPO / "models.json")
    after = json.load((REPO / "models.json").open())
    n = sum(1 for m in after if "coder" in (m.get("extended_roles") or []))
    print(f"  → `coder` roster: {len(tagged_before)} → {n}", flush=True)


def write_battery_e_summary(results, out_md: Path, fast_mode=False):
    flag  = " ⚠ FAST MODE" if fast_mode else ""
    lines = [
        f"# Aptitude Battery E — Coding{flag}", "",
        "Models: " + ", ".join("`" + r["model"] + "`" for r in results),
        f"`num_ctx={NUM_CTX}` | arm=per-model (v3, docs/think-spec.md) | execution-graded (structural gate → sandboxed run)", "",
        "Composite = weighted mean — E1 gen `.12` · E2 debug `.22` · E3 multi-lang `.18` · "
        "E5 tests `.18` · E7 constraints `.15` · E9 markup `.15` · E-hard `.18` (4-task "
        "discriminator band; also gates `coder`).",
        f"`coder` overlay (hysteresis): earn ✓ at composite ≥ {E_CODER_COMPOSITE_MIN} "
        f"**AND E-hard ≥ {E_CODER_HARD_MIN}** (+ generate ≥ {E_CODER_GENERATE_MIN}, debug > 0); "
        f"retained ~ while composite ≥ {E_CODER_COMPOSITE_BAND} and E-hard ≥ {E_CODER_HARD_BAND}; "
        "dropped as soon as either breaches. **The E-hard floor was added 2026-08-22** — the "
        "composite bar alone left 20 of 22 models tagged, including one that solved 4% of the "
        "hard band. E-core is spent; E-hard is the discriminator, so read that column.",
        "", "---", "", "## Overview", "",
        "| Model | E1 gen | E2 debug | E3 multi-lang | E5 tests | E7 constr | E9 markup | **E-hard** | Composite | coder |",
        "|-------|--------|----------|---------------|----------|-----------|-----------|-----------|-----------|-------|",
    ]
    ranked = sorted(results, key=lambda r: r.get("summary", {}).get("composite", 0), reverse=True)
    for r in ranked:
        s  = r.get("summary", {})
        cm = s.get("category_means", {})
        def g(c):
            return f"{cm[c]:.2f}" if c in cm else "—"
        comp = s.get("composite", 0)
        hard = cm.get("E-hard", 0.0)
        coder = ("✓" if s.get("coder_eligible")
                 else ("~" if comp >= E_CODER_COMPOSITE_BAND and hard >= E_CODER_HARD_BAND else "·"))
        lines.append(f"| `{r['model']}` | {g('E1')} | {g('E2')} | {g('E3')} | {g('E5')} | {g('E7')} | "
                     f"{g('E9')} | **{g('E-hard')}** | **{comp:.3f}** | {coder} |")
    lines.append("")

    # E3 per-language breakdown (JS / SQL / PHP). A dash = no scored problems
    # (runtime absent on the bench host, graceful-skipped).
    langs = ["javascript", "sql", "php"]
    lines += ["## E3 — by language", "",
              "| Model | " + " | ".join(l.upper() for l in langs) + " |",
              "|-------|" + "|".join(["------"] * len(langs)) + "|"]
    for r in ranked:
        per = {l: [] for l in langs}
        for d in r["tests"].values():
            if d.get("category") == "E3" and d.get("score") is not None and d.get("lang") in per:
                per[d["lang"]].append(d["score"])
        cells = [(f"{sum(v) / len(v):.2f}" if v else "—") for v in (per[l] for l in langs)]
        lines.append(f"| `{r['model']}` | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## Per-problem", ""]
    for r in ranked:
        lines += [f"### `{r['model']}`", ""]
        for pid, d in r["tests"].items():
            cat = d.get("category")
            if d.get("skipped"):
                lines.append(f"- **{pid}** ({cat}/{d.get('lang')}): ⤬ skipped — {d.get('reason')}")
            elif cat == "E5":
                lines.append(f"- **{pid}** (E5): score `{d['score']}` · clean_pass={d.get('clean_pass')} · "
                             f"kills {d.get('kills')}/{d.get('n_mutants')} · "
                             f"valid {d.get('n_valid')}/{d.get('n_tests')} (dropped {d.get('n_dropped')})")
            elif cat == "E9":
                gate = d.get("gate", {})
                gtxt = "gate ok" if gate.get("ok") else f"**GATE FAIL: {gate.get('reason')}**"
                lines.append(f"- **{pid}** (E9/{d.get('lang')}): score `{d['score']}` · {gtxt} · "
                             f"coverage {d.get('coverage')}")
            else:
                ex   = d.get("exec") or {}
                gate = d.get("gate", {})
                gtxt = "gate ok" if gate.get("ok") else f"**GATE FAIL: {gate.get('reason')}**"
                label = f"{cat}/{d.get('lang')}" if cat == "E3" else cat
                lines.append(f"- **{pid}** ({label}): score `{d['score']}` · {gtxt} · "
                             f"pass {ex.get('passed')}/{ex.get('total')}")
        lines.append("")

    out_md.write_text("\n".join(lines))
    print(f"MD → {out_md}", flush=True)


# ── Battery F — Conversational Consistency (multi-turn, deterministic) ──────────

def _f_rollout_and_grade(model_name, system_prompt, echo_label=""):
    """One 8-turn live rollout (assistant responses fed back) under `system_prompt`, then the
    deterministic F1–F5 core. Shared primitive: Battery F calls it with the worker default
    prompt; Battery F-elastic calls it once per ladder rung with a constraint-augmented prompt.
    Returns the graded primitives — callers shape the result/summary."""
    script = _fstyle.load_script()
    lex    = _fstyle.load_lexicon()
    opts   = script["stance_options"]
    pfx    = f"{echo_label} " if echo_label else ""

    history = [{"role": "system", "content": system_prompt}]
    turns   = {}
    picked_key = picked_label = other_label = None

    for turn in script["turns"]:
        user = turn["user"]
        if "{picked}" in user or "{other}" in user:
            if picked_label:
                user = user.replace("{picked}", picked_label).replace("{other}", other_label)
            else:
                user = user.replace("{picked}", "your pick").replace("{other}", "the other option")
        history.append({"role": "user", "content": user})
        data, wall = chat(model_name, history, max_tokens=F_MAX_TOKENS)
        resp = data.get("message", {}).get("content", "")
        history.append({"role": "assistant", "content": resp})
        br = data.get("_budget_retry") or {}
        turns[turn["id"]] = {"type": turn["type"], "user": user, "resp": resp,
                             "vec": _fstyle.style_vector(resp, lex), "wall_s": round(wall, 1),
                             # COST of the turn (09-26): wall + generated tokens INCLUDING an attempt a
                             # budget retry discarded. eval_count counts thinking tokens too.
                             "wall_total_s": round(wall + (br.get("discarded_wall_s") or 0.0), 1),
                             "tokens": (data.get("eval_count") or 0) + (br.get("discarded_tokens") or 0),
                             "retry": bool(br), "recovered": br.get("recovered", True)}
        if turn["id"] == "T1":
            picked_key, picked_label = _fstyle.detect_stance(resp, opts)
            if picked_key:
                other_label = next(o["label"] for o in opts if o["key"] != picked_key)
        tag = f" stance={picked_key}" if turn["id"] == "T1" else ""
        print(f"  {pfx}[{turn['id']} {turn['type']}]{tag}  → {resp[:90].replace(chr(10),' ')}", flush=True)

    stance_detected = picked_key is not None

    # F1 — persona_hold: unambiguous disclaimer tells across all turns (2+ = 0)
    tells = sum(_fstyle.lexicon_hits(t["resp"], lex["disclaimer_tells"]) for t in turns.values())
    f1 = round(1 - min(1.0, tells / 2.0), 4)

    # F2 — pressure_resistance: mean(T4 stance-flip, T5 sycophancy surge, T7 capitulation)
    t4_stance = _fstyle.detect_stance(turns["T4"]["resp"], opts)[0]
    t4_flip   = bool(t4_stance and t4_stance != picked_key) if stance_detected else None
    t4_score  = (0.0 if t4_flip else 1.0) if stance_detected else None
    base_ex   = (turns["T1"]["vec"]["exclaim"] + turns["T2"]["vec"]["exclaim"]) / 2
    t5_surge  = max(0.0, turns["T5"]["vec"]["exclaim"] - base_ex) \
                + _fstyle.lexicon_hits(turns["T5"]["resp"], lex["sycophancy_markers"]) / 2.0
    t5_score  = round(max(0.0, 1 - min(1.0, t5_surge)), 4)
    t7_caps   = _fstyle.lexicon_hits(turns["T7"]["resp"], lex["capitulation_markers"])
    t7_score  = round(max(0.0, 1 - min(1.0, t7_caps / 2.0)), 4)
    f2_parts  = [p for p in (t4_score, t5_score, t7_score) if p is not None]
    f2 = round(sum(f2_parts) / len(f2_parts), 4)

    # F3 — voice_stability: within-run drift of identity style-vector from own T1–T2 baseline
    baseline = _fstyle.mean_vector([turns["T1"]["vec"], turns["T2"]["vec"]])
    drifts   = [_fstyle.vector_drift(turns[t]["vec"], baseline) for t in ("T3", "T4", "T5", "T6", "T7", "T8")]
    f3 = round(max(0.0, 1 - min(1.0, sum(drifts) / len(drifts))), 4)

    # F4 — callback_fidelity: stance persists T1→T2→T6 (objective) + T6 recalls T2 plan
    overlap    = _fstyle.token_overlap(turns["T6"]["resp"], turns["T2"]["resp"])
    overlap_ok = min(1.0, overlap / 0.20)
    if stance_detected:
        seen    = [s for s in (_fstyle.detect_stance(turns["T2"]["resp"], opts)[0],
                               _fstyle.detect_stance(turns["T6"]["resp"], opts)[0]) if s]
        persist = 1.0 if all(s == picked_key for s in seen) else 0.0
        f4 = round((persist + overlap_ok) / 2, 4)
    else:
        persist = None
        f4 = round(overlap_ok, 4)   # stance half unreliable without a T1 detection

    # F5 — coherence_recovery: T3 (whiplash) + T8 (close) didn't collapse. Binary on purpose —
    # it SATURATES at 1.0 (contributes ~0 variance); a graded/length version re-introduced
    # content-driven σ (validated 2026-06-15: grading raised σ, reverted). Low-discrimination
    # dim → candidate to fold into the F6 judge in a future tune.
    t3_ok = 0.0 if _fstyle.is_degenerate(turns["T3"]["resp"]) else 1.0
    t8_ok = 0.0 if _fstyle.is_degenerate(turns["T8"]["resp"]) else 1.0
    f5 = round((t3_ok + t8_ok) / 2, 4)

    dims      = {"F1": f1, "F2": f2, "F3": f3, "F4": f4, "F5": f5}
    present   = {k: w for k, w in F_WEIGHTS.items() if dims[k] is not None}
    composite = round(sum(dims[k] * w for k, w in present.items()) / sum(present.values()), 4)

    return {
        "turns": turns, "dims": dims, "composite": composite,
        "stance_detected": stance_detected, "picked": picked_key,
        "detail": {"tells": tells, "t4_flip": t4_flip, "t4_stance": t4_stance,
                   "t5_surge": round(t5_surge, 3), "t7_caps": t7_caps,
                   "t6_overlap": round(overlap, 3), "persist": persist},
    }


def run_battery_f(model_name):
    """8-turn live rollout (assistant responses fed back), then deterministic F1–F5.
    Single-pass primitive — the multipass averager calls this per pass."""
    print(f"\n{'='*60}\nBATTERY F: {model_name}\n{'='*60}", flush=True)
    g = _f_rollout_and_grade(model_name, PROMPT_WORKER_FULL)
    print(f"\n  → composite={g['composite']}  dims={g['dims']}  "
          f"stance_detected={g['stance_detected']} (picked={g['picked']})", flush=True)
    return {
        "model": model_name, "battery": "F",
        "summary": {"dims": g["dims"], "composite": g["composite"],
                    "stance_detected": g["stance_detected"], "picked": g["picked"],
                    "detail": g["detail"]},
        "tests": {tid: {"type": t["type"], "user": t["user"], "resp": t["resp"],
                        "vec": t["vec"], "wall_s": t["wall_s"]}
                  for tid, t in g["turns"].items()},
    }


# ── Battery F-elastic — Prompt-Elasticity (prompt-σ, opt-in) ───────────────────

def run_battery_f_elastic(model_name):
    """Prompt-elasticity: how the Battery-F composite (F1–F5) moves as a function of
    system-prompt COMPLEXITY, over a synthetic persona-neutral capability-constraint ladder
    (suites/elasticity/ladder.json). The same F rollout + grader runs at every rung; the only
    new surface is the per-rung constraint block and the deterministic adherence meter.

    prompt-σ and adherence are CO-EQUAL — a rung without adherence is invalid, and a low
    prompt-σ alone is ambiguous (robust vs prompt-deaf). The categorical verdict is computed
    HERE from the ladder's DECLARED cutoffs so prompt-σ is never read naked. Single-pass
    primitive (multipass averaging = Phase 2)."""
    ladder = _elastic.load_ladder()
    print(f"\n{'='*60}\nBATTERY F-elastic: {model_name}\n{'='*60}", flush=True)

    per_rung, composites, tests = [], [], {}
    for rung in ladder["rungs"]:
        block     = _elastic.render_constraints(rung, ladder)
        sysprompt = PROMPT_WORKER_FULL + "\n\n" + block
        g         = _f_rollout_and_grade(model_name, sysprompt, echo_label=f"[{rung['id']}]")
        responses = [t["resp"] for t in g["turns"].values()]
        adh       = _elastic.score_rung(rung, ladder, responses)
        per_rung.append({
            "rung": rung["id"], "label": rung.get("label", rung["id"]),
            "band": rung.get("band", "core"),
            "constraints": rung["constraints"], "constraints_n": len(rung["constraints"]),
            "composite": g["composite"], "run_sigma": None,   # single-pass; populated by Phase-2 averager
            "adherence": adh["adherence"],
            "instruction_adherence": adh["instruction_adherence"],
            "length_adherence": adh["length_adherence"],
            "per_constraint": adh["per_constraint"],
            "dims": g["dims"], "stance_detected": g["stance_detected"],
        })
        composites.append(g["composite"])
        tests[rung["id"]] = {tid: {"type": t["type"], "resp": t["resp"], "wall_s": t["wall_s"],
                                   "wall_total_s": t["wall_total_s"], "tokens": t["tokens"],
                                   "retry": t["retry"], "recovered": t["recovered"]}
                             for tid, t in g["turns"].items()}
        print(f"  → rung={rung['id']} n={len(rung['constraints'])} composite={g['composite']} "
              f"instr_adh={adh['instruction_adherence']} len_adh={adh['length_adherence']}", flush=True)

    # TWO-BAND (2026-08-23). The headline numbers stay scoped to the CORE rungs so they remain
    # comparable to the v1 series and to the 06-22 calibration; the hard rung is reported beside
    # them, never folded in. Same split as clean_depth / clean_depth_core in Battery G.
    core_rungs = [r for r in per_rung if r.get("band", "core") == "core"]
    hard_rungs = [r for r in per_rung if r.get("band") == "hard"]
    def _avg_over(rows, key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.mean(v), 4) if v else None
    core_comps = [r["composite"] for r in core_rungs]

    prompt_sigma          = round(statistics.pstdev(core_comps), 4) if len(core_comps) > 1 else 0.0
    prompt_sigma_all      = round(statistics.pstdev(composites), 4) if len(composites) > 1 else 0.0
    instruction_adherence = _avg_over(core_rungs, "instruction_adherence")   # verdict driver (core)
    # ⚠ Scoped to the constraints the hard band ADDS (see adherence.hard_band_ids). Averaging the
    # rung's full binary set instead diluted the 3 discriminators to 3-of-7 with the saturated core
    # constraints, and the hard band then changed zero verdicts across the 08-24 fleet.
    hard_vals             = [v for v in (_elastic.hard_band_adherence(r, ladder) for r in hard_rungs)
                             if v is not None]
    hard_adherence        = round(statistics.mean(hard_vals), 4) if hard_vals else None
    # carried for continuity with the 08-23/08-24 series, which published the diluted meter
    hard_adherence_allbinary = _avg_over(hard_rungs, "instruction_adherence")
    length_adherence      = _avg_over(core_rungs, "length_adherence")        # standalone verbosity meter
    adherence             = _avg_over(core_rungs, "adherence")               # all-constraint mean (legacy)
    # Bars come from bench_utils, NOT the hashed ladder — grading policy must move without forcing a
    # 1h44m re-measure. ladder.json's block is documentation; see F_ELASTIC_CUTOFFS.
    cutoffs               = dict(F_ELASTIC_CUTOFFS)
    verdict               = _elastic.classify(prompt_sigma, instruction_adherence, cutoffs,
                                              hard_adherence=hard_adherence)

    # COST of the whole pass (09-26) — what holding the constraints cost at this arm. Past the saturation
    # band it is what separates models (export elastic_key). Includes budget-retry discards.
    all_turns = [t for rung_t in tests.values() for t in rung_t.values()]
    cost = {"s_per_turn": round(sum(t["wall_total_s"] for t in all_turns) / len(all_turns), 2),
            "tokens_per_turn": round(sum(t["tokens"] for t in all_turns) / len(all_turns), 1),
            "retries_per_pass": sum(1 for t in all_turns if t["retry"]),
            "unrecovered": sum(1 for t in all_turns if t["retry"] and not t["recovered"]),
            "retry_s_estimated": False, "passes_timed": 1}

    print(f"\n  → prompt_sigma={prompt_sigma} (all-rung {prompt_sigma_all})  "
          f"instruction_adherence={instruction_adherence}  hard_adherence={hard_adherence}  "
          f"length_adherence={length_adherence}  verdict={verdict}  "
          f"cost={cost['s_per_turn']}s·{cost['tokens_per_turn']}tok/turn", flush=True)
    return {
        "model": model_name, "battery": "F-elastic",
        "summary": {"prompt_sigma": prompt_sigma,
                    "prompt_sigma_all": prompt_sigma_all,
                    "instruction_adherence": instruction_adherence,
                    "hard_adherence": hard_adherence,
                    "hard_adherence_allbinary": hard_adherence_allbinary,
                    "length_adherence": length_adherence,
                    "adherence": instruction_adherence,   # co-equal alias: the pairing is prompt-σ + instruction-adherence
                    "verdict": verdict, "cutoffs": cutoffs, "per_rung": per_rung, "cost": cost},
        "tests": tests,
    }


def write_battery_f_elastic_summary(results, out_md: Path, fast_mode=False):
    flag   = " ⚠ FAST MODE" if fast_mode else ""
    ladder = _elastic.load_ladder()
    rungs  = ladder["rungs"]
    c      = ladder["verdict_cutoffs"]
    fnum  = lambda v: "—" if v is None else f"{v:.2f}"
    fnum3 = lambda v: "—" if v is None else f"{v:.3f}"   # headline instr-adh: 3dp so cutoff-boundary cases read honestly
    lines  = [
        f"# Aptitude Battery F-elastic — Prompt-Elasticity (prompt-σ){flag}", "",
        "How a model's conversational-consistency composite (F1–F5) moves as a function of "
        "**system-prompt complexity** over a synthetic, persona-neutral capability-constraint "
        "ladder. **prompt-σ and instruction-adherence are co-equal** — a low prompt-σ alone is "
        "ambiguous (robust vs prompt-deaf); adherence disambiguates, and the verdict is computed "
        "here. Adherence is **split by constraint class**: `instr` = binary obey-or-ignore "
        "(no-exclamation / no-lists / end-with-question / required-prefix) **drives the verdict**; "
        "`len` = the word-cap (a standalone verbosity meter) does **not**, so wordiness can't "
        "masquerade as prompt-insensitivity.",
        f"`num_ctx={NUM_CTX}` | arm=per-model (v3, docs/think-spec.md) | single-pass | rungs = "
        + " → ".join(f"{r['id']}({len(r['constraints'])})" for r in rungs), "",
        f"**Declared verdict cutoffs** (keyed on `instruction_adherence`; {c.get('_status','')}): "
        f"`robust` = σ < {c['sigma_hi']} AND instr ≥ {c['adherence_hi']}; "
        f"`prompt-deaf` = σ < {c['sigma_hi']} AND instr < {c['adherence_lo']}; "
        f"`prompt-sensitive` = otherwise.", "",
        "| Model | Verdict | prompt-σ | instr-adh | len-adh | "
        + " | ".join(f"{r['id']} (c·i)" for r in rungs) + " |",
        "|-------|---------|----------|-----------|---------|"
        + "|".join(["----------"] * len(rungs)) + "|",
    ]
    order = {"robust": 0, "prompt-sensitive": 1, "prompt-deaf": 2}
    ranked = sorted([r for r in results if "summary" in r],
                    key=lambda r: (order.get(r["summary"]["verdict"], 9),
                                   -(r["summary"].get("instruction_adherence") or 0)))
    for r in ranked:
        s = r["summary"]
        by_id = {pr["rung"]: pr for pr in s["per_rung"]}
        cells = []
        for rg in rungs:
            pr = by_id.get(rg["id"])
            cells.append(f"{pr['composite']:.2f}·{fnum(pr.get('instruction_adherence'))}" if pr else "—")
        lines.append(f"| `{r['model']}` | **{s['verdict']}** | {s['prompt_sigma']:.3f} | "
                     f"{fnum3(s.get('instruction_adherence'))} | {fnum(s.get('length_adherence'))} | "
                     + " | ".join(cells) + " |")
    lines += ["", "_Per-rung cells are `composite·instr-adh` (instruction adherence; `—` = rung "
              "has no instruction-class constraint). prompt-σ = pstdev of the per-rung composites; "
              "a high σ means consistency stretches under instruction load. `len-adh` is a "
              "verbosity meter, not a verdict input. Cutoffs are PROVISIONAL until the cross-family "
              "gating pass calibrates them._"]
    out_md.write_text("\n".join(lines) + "\n")
    print(f"MD → {out_md}", flush=True)


def write_battery_f_summary(results, out_md: Path, fast_mode=False):
    flag  = " ⚠ FAST MODE" if fast_mode else ""
    lines = [
        f"# Aptitude Battery F — Conversational Consistency{flag}", "",
        "Multi-turn (8-turn live rollout) · deterministic core (F1 persona_hold · F2 "
        "pressure_resistance · F3 voice_stability · F4 callback_fidelity · F5 coherence_recovery).",
        f"`num_ctx={NUM_CTX}` | arm=per-model (v3, docs/think-spec.md) | single-pass (multipass-averaged in the suite).", "",
        "| Model | Composite | F1 | F2 | F3 | F4 | F5 | stance? |",
        "|-------|-----------|----|----|----|----|----|:-------:|",
    ]
    ranked = sorted([r for r in results if "summary" in r],
                    key=lambda r: r["summary"]["composite"], reverse=True)
    for r in ranked:
        s = r["summary"]; d = s["dims"]
        st = "✓" if s.get("stance_detected") else "⚠ miss"
        lines.append(f"| `{r['model']}` | **{s['composite']:.3f}** | {d['F1']:.2f} | {d['F2']:.2f} | "
                     f"{d['F3']:.2f} | {d['F4']:.2f} | {d['F5']:.2f} | {st} |")
    lines += ["", "_stance? = was the T1 committed choice detected (F2-flip & F4 ride on it). "
              "A miss means those dims are unreliable for that model._"]
    out_md.write_text("\n".join(lines) + "\n")
    print(f"MD → {out_md}", flush=True)


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if _flag("--regate"):        # gate-policy change — re-apply from stored results, no model calls
        if battery_arg == "F-ELASTIC":
            regate_f_elastic(dry_run=_flag("--dry-run"))
        else:
            regate_coder(dry_run=_flag("--dry-run"))
        sys.exit(0)

    TODAY  = date.today().isoformat()
    suffix = "_fast" if fast_mode else ""

    # Apply --capable-only filter for tool-relevant batteries (C, D)
    _cd_models = _filter_tool_capable(BATTERY_B_MODELS) if capable_only else BATTERY_B_MODELS
    _e_models  = _filter_tool_capable(BATTERY_E_MODELS) if capable_only else BATTERY_E_MODELS

    BATTERY_MAP = {
        "A": (BATTERY_A_MODELS, run_battery_a, "aptitude_a"),
        "B": (BATTERY_B_MODELS, run_battery_b, "aptitude_b"),
        "C": (_cd_models,       run_battery_c, "aptitude_c"),
        "D": (_cd_models,       run_battery_d, "aptitude_d"),
        "E": (_e_models,        run_battery_e, "aptitude_e"),
        "F": (BATTERY_E_MODELS, run_battery_f, "aptitude_f"),
        # Opt-in only — never wired into `all`/`batteries`. ~3× a single F run (one rollout
        # per rung); the consumer-facing name is `--battery F-elastic`.
        "F-ELASTIC": (BATTERY_E_MODELS, run_battery_f_elastic, "aptitude_f_elastic"),
    }

    if battery_arg not in BATTERY_MAP:
        print(f"Unknown battery '{battery_arg}'. Available: A, B, C, D, E, F, F-elastic")
        sys.exit(1)

    default_models, runner, pfx = BATTERY_MAP[battery_arg]
    # Eligible universe = the full battery roster (independent of --models, so carry-forward stays
    # complete); forced models outside it are appended. --models → resume explicit targets below.
    MODELS = list(default_models)
    if model_args:
        have = {n for n, *_ in MODELS}
        MODELS += [(m, 0.0) for m in model_args if m not in have]
    out_pfx = f"{pfx}_{TODAY}{suffix}"

    preflight(MODELS, ollama_host)
    OUT_JSON = RESULTS_DIR / f"{out_pfx}.json"
    OUT_MD   = RESULTS_DIR / f"{out_pfx}.md"
    flag     = " [FAST MODE — informal]" if fast_mode else ""

    print(f"BenchLLAMA aptitude — Battery {battery_arg}{flag} — {TODAY}", flush=True)
    print(f"Models: {[m[0] for m in MODELS]}", flush=True)
    print(f"Output: {OUT_JSON}", flush=True)

    # ── Content-addressed resume (docs/resume-spec.md): skip unchanged models, carry prior result
    #    forward from the DB (lossless). No time window. `--force` (used by the averager per pass) →
    #    no carry-forward. Battery key = battery_arg (A/B/C/D/E; F-elastic normalised). ──
    import resume
    _rk = "F-elastic" if battery_arg == "F-ELASTIC" else battery_arg
    eligible = [n for n, *_ in MODELS]
    _reg = json.loads((REPO / "models.json").read_text())
    cloud = {m["name"] for m in _reg if m.get("cloud")}
    run_names, all_results, why = resume.plan_single_pass(
        _rk, eligible, host=ollama_host, cloud=cloud, force=force,
        explicit_models=(model_args or None), check_runtime="--check-runtime" in sys.argv, arm=ARM_REQ)
    all_results = list(all_results)
    completed = set(eligible) - set(run_names)
    print("  " + resume.format_report(_rk, run_names, sorted(completed), why).replace("\n", "\n  "), flush=True)

    apt_done  = []
    first_run = True

    for model_name, disk_gb in MODELS:
        if model_name in completed:
            print(f"  ↷ {model_name} — already done, skipping", flush=True)
            continue
        if not first_run:
            _ws(model_name, "cooldown", apt_done, f"Battery {battery_arg}")
            cooldown(COOLDOWN, label=f"after previous model")
        first_run = False
        _ws(model_name, "running", apt_done, f"Battery {battery_arg}")
        print(f"MODEL: {model_name}  ({disk_gb:.1f} GB disk)  battery={battery_arg}", flush=True)
        try:
            r = runner(model_name)
            r.update(arm_stamp(model_name, resolve_arm(model_name, ARM_REQ) or "direct"))   # v3
        except Exception as e:
            print(f"\n  ✗ {model_name} FAILED: {e} — skipping\n", flush=True)
            r = {"model": model_name, "disk_gb": disk_gb, "error": str(e), "tests": {}}
        all_results.append(r)
        OUT_JSON.write_text(json.dumps(all_results, indent=2))
        try:
            import results_db
            if battery_arg in ("A", "B", "C", "D"):   # E/F/F-elastic canonical handled by average_e_runs
                results_db.record_all(battery_arg, all_results, only=set(run_names))
        except Exception:
            pass
        print(f"  ✓ {model_name} done — JSON updated", flush=True)
        unload(model_name)
        time.sleep(3)

    apt_done.append(battery_arg)
    _ws("", "done", apt_done)

    summary_writers = {
        "A": write_battery_a_summary,
        "B": write_battery_b_summary,
        "C": write_battery_c_summary,
        "D": write_battery_d_summary,
        "E": write_battery_e_summary,
        "F": write_battery_f_summary,
        "F-ELASTIC": write_battery_f_elastic_summary,
    }
    if battery_arg in summary_writers:
        summary_writers[battery_arg](all_results, OUT_MD, fast_mode)

    # Battery E confers the `coder` extended role. Skip in --fast (informal results
    # must not mutate the registry); real runs write back additively.
    if battery_arg == "E" and not fast_mode:
        apply_coder_overlay(all_results, REPO / "models.json")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"JSON → {OUT_JSON}")
    print(f"MD   → {OUT_MD}")
