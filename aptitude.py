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
import sys
import time
import requests
from pathlib import Path
from datetime import date
from bench_utils import cooldown, preflight, latest_result

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


def _filter_tool_capable(models):
    """Keep only models that passed the calculate test in the most recent benchmark JSON.
    Used by --capable-only for Battery C and D to skip personality-only workers.
    Falls back to the full list if no benchmark data is available.
    """
    benchmarks = sorted(RESULTS_DIR.glob("benchmark_*.json"), key=lambda p: p.stat().st_mtime)
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

def chat(model, messages, max_tokens=600, think=False, tools=None):
    payload = {
        "model":    model,
        "messages": messages,
        "stream":   False,
        "options":  {"num_ctx": NUM_CTX},
        "think":    think,
    }
    if max_tokens:
        payload["options"]["num_predict"] = max_tokens
    if tools:
        payload["tools"] = tools
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


def _chat_ctx(model, messages, ctx, max_tokens, think=False):
    """Like chat() but with a custom num_ctx override."""
    payload = {
        "model":    model,
        "messages": messages,
        "stream":   False,
        "options":  {"num_ctx": ctx, "num_predict": max_tokens},
        "think":    think,
    }
    t0   = time.time()
    r    = requests.post(f"{ollama_host}/api/chat", json=payload, timeout=TIMEOUT)
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

    # C1 — jpeg_signals: think=False vs think=True, capture diagnosis
    print("\n  [C1] jpeg_signals (think=off vs think=on)...", flush=True)
    c1_results = {}
    for think_val in [False, True]:
        data, wall = chat(model_name, sys_full + [{"role": "user", "content": JPEG_PROMPT}],
                          max_tokens=1500, think=think_val)
        text  = data.get("message", {}).get("content", "")
        cov   = _jpeg_coverage(text)
        diag  = _think_diagnosis(data) if think_val else "think_off"
        t     = tps(data)
        key   = "think_on" if think_val else "think_off"
        c1_results[key] = {
            "score": cov["score"], "max": cov["max"], "pass": cov["pass"],
            "signals": cov["signals"], "words": word_count(text),
            "wall_s": round(wall, 1), "tps": t, "think_diagnosis": diag,
            "response": text,
        }
        sigs_hit = [k for k, v in cov["signals"].items() if v]
        print(
            f"    [think={key[-2:]}]  {cov['score']}/{cov['max']} signals  "
            f"{word_count(text)}w  diag={diag}",
            flush=True,
        )
        for s in sigs_hit:
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

    # C6 — think_coverage: JPEG think=True with the lean prompt (vs full prompt in C1)
    print("\n  [C6] think_coverage (JPEG, think=on, lean prompt)...", flush=True)
    sys_lean = [{"role": "system", "content": PROMPT_WORKER_LEAN}]
    data, wall = chat(model_name, sys_lean + [{"role": "user", "content": JPEG_PROMPT}],
                      max_tokens=1500, think=True)
    text = data.get("message", {}).get("content", "")
    cov  = _jpeg_coverage(text)
    diag = _think_diagnosis(data)
    result["tests"]["c6_think_coverage"] = {
        "score": cov["score"], "max": cov["max"], "pass": cov["pass"],
        "signals": cov["signals"], "words": word_count(text),
        "wall_s": round(wall, 1), "tps": tps(data), "think_diagnosis": diag,
        "response": text,
    }
    c1_think = result["tests"]["c1_jpeg_signals"].get("think_on", {})
    delta    = cov["score"] - c1_think.get("score", 0)
    print(
        f"    {cov['score']}/{cov['max']} signals  {word_count(text)}w  "
        f"diag={diag}  Δ vs full-prompt-think={delta:+d}",
        flush=True,
    )

    return result


# ── Markdown summary — Battery C ──────────────────────────────────────────────

def write_battery_c_summary(results, out_md: Path, fast_mode=False):
    flag  = " ⚠ FAST MODE" if fast_mode else ""
    lines = [
        f"# Aptitude Battery C — Worker Research{flag}", "",
        "Models: " + ", ".join("`" + r["model"] + "`" for r in results),
        f"`num_ctx={NUM_CTX}` (baseline) | worker prompt", "", "---", "",
    ]

    lines += ["## C1 — JPEG signals: think=off vs think=on", ""]
    hdr = "| Model | off score | on score | delta | think diagnosis |"
    sep = "|-------|-----------|----------|-------|-----------------|"
    lines += [hdr, sep]
    for r in results:
        c1    = r["tests"].get("c1_jpeg_signals", {})
        off   = c1.get("think_off", {})
        on    = c1.get("think_on",  {})
        delta = (on.get("score", 0) or 0) - (off.get("score", 0) or 0)
        lines.append(
            f"| `{r['model']}` | {off.get('score','?')}/7 | {on.get('score','?')}/7 "
            f"| {delta:+d} | {on.get('think_diagnosis','—')} |"
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

    lines += ["## C6 — think_coverage (JPEG, lean prompt)", ""]
    hdr6 = "| Model | score | diagnosis | Δ vs full-think |"
    sep6 = "|-------|-------|-----------|-----------------|"
    lines += [hdr6, sep6]
    for r in results:
        c6   = r["tests"].get("c6_think_coverage", {})
        c1on = r["tests"].get("c1_jpeg_signals", {}).get("think_on", {})
        delta = (c6.get("score", 0) or 0) - (c1on.get("score", 0) or 0)
        lines.append(
            f"| `{r['model']}` | {c6.get('score','?')}/7 "
            f"| {c6.get('think_diagnosis','—')} | {delta:+d} |"
        )
    lines.append("")

    for r in results:
        lines += ["---", "", f"## `{r['model']}`", ""]

        lines += ["### C1 — JPEG signals (think off / on)", ""]
        c1 = r["tests"].get("c1_jpeg_signals", {})
        for key in ["think_off", "think_on"]:
            d = c1.get(key, {})
            lines += [
                f"**{key}** — {d.get('score','?')}/7 signals  "
                f"{d.get('words','?')}w  wall={d.get('wall_s','?')}s  "
                f"tps={d.get('tps','?')}  diagnosis={d.get('think_diagnosis','—')}",
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

        c6   = r["tests"].get("c6_think_coverage", {})
        c1on = r["tests"].get("c1_jpeg_signals", {}).get("think_on", {})
        delta = (c6.get("score", 0) or 0) - (c1on.get("score", 0) or 0)
        lines += [
            "### C6 — think_coverage (lean prompt)", "",
            f"({c6.get('score','?')}/7 signals · {c6.get('words','?')}w · "
            f"diagnosis={c6.get('think_diagnosis','—')} · Δ={delta:+d} vs full-prompt-think)",
            "", c6.get("response", "—"), "",
        ]

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


def _tool_loop(model_name, messages, tools, max_steps=6, think=False,
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

    # D5 — think_tools: chain_3 with think=False vs think=True.
    # Uses _think_diagnosis() on the actual final API response (not a step-count approximation).
    print("\n  [D5] think_tools (chain_3 with think=off vs think=on)...", flush=True)
    d5_results = {}
    for think_val in [False, True]:
        text5, calls5, steps5, wall5, last_raw5 = _tool_loop(
            model_name,
            sys_std + [{"role": "user", "content": CHAIN_3_PROMPT}],
            tools=TOOL_DEFS_D, think=think_val,
        )
        tool_seq5 = [c["tool"] for c in calls5]
        final_ok5 = "43.9" in text5 or "43.91" in text5
        diag5     = _think_diagnosis(last_raw5) if think_val else "think_off"
        key = "think_on" if think_val else "think_off"
        d5_results[key] = {
            "tool_sequence": tool_seq5, "steps": steps5,
            "final_answer_ok": final_ok5, "wall_s": round(wall5, 1),
            "think_diagnosis": diag5,
            "calls": calls5, "response": text5,
        }
        mark = "✓" if final_ok5 else "✗"
        print(f"    [think={key[-2:]}] {mark}  seq={tool_seq5}  steps={steps5}  "
              f"wall={round(wall5,1)}s  diag={diag5}", flush=True)
    result["tests"]["d5_think_tools"] = d5_results

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

    # D5 think_tools table (with think_diagnosis column)
    lines += ["## D5 — think_tools (chain_3 with think on/off)", ""]
    hdr5 = "| Model | off final | on final | off steps | on steps | think diag (on) |"
    sep5 = "|-------|-----------|----------|-----------|----------|-----------------|"
    lines += [hdr5, sep5]
    for r in results:
        d5  = r["tests"].get("d5_think_tools", {})
        off = d5.get("think_off", {})
        on  = d5.get("think_on",  {})
        lines.append(
            f"| `{r['model']}` "
            f"| {'✓' if off.get('final_answer_ok') else '✗'} "
            f"| {'✓' if on.get('final_answer_ok') else '✗'} "
            f"| {off.get('steps','?')} | {on.get('steps','?')} "
            f"| {on.get('think_diagnosis','—')} |"
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

        lines += ["### D5 — think_tools", ""]
        d5 = r["tests"].get("d5_think_tools", {})
        for k in ["think_off", "think_on"]:
            d = d5.get(k, {})
            lines += [
                f"**{k}** — seq={d.get('tool_sequence','?')}  "
                f"steps={d.get('steps','?')}  final={d.get('final_answer_ok','?')}  "
                f"wall={d.get('wall_s','?')}s  diag={d.get('think_diagnosis','—')}",
                "", d.get("response", "—"), "",
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


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TODAY  = date.today().isoformat()
    suffix = "_fast" if fast_mode else ""

    # Apply --capable-only filter for tool-relevant batteries (C, D)
    _cd_models = _filter_tool_capable(BATTERY_B_MODELS) if capable_only else BATTERY_B_MODELS

    BATTERY_MAP = {
        "A": (BATTERY_A_MODELS, run_battery_a, "aptitude_a"),
        "B": (BATTERY_B_MODELS, run_battery_b, "aptitude_b"),
        "C": (_cd_models,       run_battery_c, "aptitude_c"),
        "D": (_cd_models,       run_battery_d, "aptitude_d"),
    }

    if battery_arg not in BATTERY_MAP:
        print(f"Unknown battery '{battery_arg}'. Available: A, B, C, D")
        sys.exit(1)

    default_models, runner, pfx = BATTERY_MAP[battery_arg]
    MODELS  = [(m, 0.0) for m in model_args] if model_args else default_models
    out_pfx = f"{pfx}_{TODAY}{suffix}"

    preflight(MODELS, ollama_host)
    OUT_JSON = RESULTS_DIR / f"{out_pfx}.json"
    OUT_MD   = RESULTS_DIR / f"{out_pfx}.md"
    flag     = " [FAST MODE — informal]" if fast_mode else ""

    print(f"BenchLLAMA aptitude — Battery {battery_arg}{flag} — {TODAY}", flush=True)
    print(f"Models: {[m[0] for m in MODELS]}", flush=True)
    print(f"Output: {OUT_JSON}", flush=True)

    # ── Resume logic ──────────────────────────────────────────────────────────
    # Resume SOURCE = today's battery file if present, else the most recent file
    # for THIS battery within 24h (cross-day — filenames embed the date). Skip
    # models already completed; --force disables resume. Always writes today's file.
    all_results = []
    completed   = set()
    source = OUT_JSON if OUT_JSON.exists() else (None if force else latest_result(RESULTS_DIR, pfx, fast_mode, 24))
    if source is not None and not force:
        try:
            loaded      = json.load(source.open())
            all_results = [r for r in loaded if "error" not in r]   # retry replaces, no dupes
            completed   = {r["model"] for r in all_results}
            if completed:
                via = "" if source == OUT_JSON else f" (carried from {source.name})"
                print(f"  Resuming — {len(completed)} model(s) already done{via}: {sorted(completed)}", flush=True)
        except Exception:
            all_results, completed = [], set()

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
        except Exception as e:
            print(f"\n  ✗ {model_name} FAILED: {e} — skipping\n", flush=True)
            r = {"model": model_name, "disk_gb": disk_gb, "error": str(e), "tests": {}}
        all_results.append(r)
        OUT_JSON.write_text(json.dumps(all_results, indent=2))
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
    }
    if battery_arg in summary_writers:
        summary_writers[battery_arg](all_results, OUT_MD, fast_mode)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"JSON → {OUT_JSON}")
    print(f"MD   → {OUT_MD}")
