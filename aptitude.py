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


# ── Battery stubs (C, D) ──────────────────────────────────────────────────────

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

    BATTERY_MAP = {
        "A": (BATTERY_A_MODELS, run_battery_a, "aptitude_a"),
        "B": (BATTERY_B_MODELS, run_battery_b, "aptitude_b"),
        "C": (BATTERY_B_MODELS, run_battery_c, "aptitude_c"),  # worker models
        "D": (BATTERY_B_MODELS, run_battery_d, "aptitude_d"),  # worker models
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

    summary_writers = {
        "A": write_battery_a_summary,
        "B": write_battery_b_summary,
    }
    if battery_arg in summary_writers:
        summary_writers[battery_arg](all_results, OUT_MD, fast_mode)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"JSON → {OUT_JSON}")
    print(f"MD   → {OUT_MD}")
