#!/usr/bin/env python3
"""
BenchLLAMA — Battery F-elastic deterministic toolkit (prompt-elasticity).

The judge-free, embedded-constraint adherence meter the prompt-σ signal rides on.
No model in the loop; stdlib only. Each rung of the ladder (suites/elasticity/ladder.json)
augments the worker system prompt with a nested, growing set of machine-verifiable output
constraints — so the constraint COUNT is both the monotonic complexity axis and the adherence
meter. The SAME 8-turn Battery-F rollout runs at every rung and the F1–F5 grader is reused
unchanged; this module only (a) renders the per-rung constraint block injected into the system
prompt and (b) scores how well each response obeyed it.

Co-equal-or-not-at-all: callers must emit prompt-σ AND adherence together; the categorical
verdict (`robust | prompt-sensitive | prompt-deaf`) is computed here from DECLARED cutoffs so
prompt-σ is never read naked.

  • render_constraints(rung, ladder) — the instruction block for the system prompt.
  • check(cid, params, text)        — one deterministic constraint check → bool.
  • score_rung(rung, ladder, resps) — per-constraint satisfaction rate + mean adherence.
  • classify(prompt_sigma, adherence, cutoffs) — the producer-side categorical verdict.

Run directly (`python3 adherence.py`) for the deterministic self-test — validates every
checker + the verdict regions without touching a model (the half that can be gated offline).
"""

import re
import json
import statistics
from pathlib import Path

HERE = Path(__file__).parent

_LIST_RE = re.compile(r"^\s*([-*•‣◦]|\d+[.)])\s+", re.M)


def load_ladder():
    return json.load((HERE / "ladder.json").open())


def _words(text):
    return re.findall(r"\w+", text or "")


def check(cid, params, text):
    """True iff `text` satisfies constraint `cid`. Deterministic, stdlib only.
    An empty response satisfies word_cap/no_exclamation/no_lists trivially but fails the
    positive-form constraints (end_with_question, required_prefix) — a non-answer can't obey them."""
    t = text or ""
    if cid == "word_cap":
        return len(_words(t)) <= params.get("max_words", 120)
    if cid == "no_exclamation":
        return "!" not in t
    if cid == "no_lists":
        return _LIST_RE.search(t) is None
    if cid == "end_with_question":
        s = t.rstrip()
        return bool(s) and s.endswith("?")
    if cid == "required_prefix":
        return t.lstrip().startswith(params.get("prefix", "NB:"))
    raise KeyError(f"unknown constraint id: {cid}")


def render_constraints(rung, ladder):
    """The instruction block injected into the system prompt for this rung.
    Rendered as a numbered rule list addressed to the assistant's OWN replies — distinct
    from (and not in conflict with) any no_lists rule it may carry about its output."""
    defs  = ladder["constraints"]
    items = [defs[c]["instruction"] for c in rung["constraints"]]
    head  = ("Follow this output rule exactly in every reply:" if len(items) == 1
             else "Follow these output rules exactly in every reply:")
    body  = "\n".join(f"{i+1}. {txt}" for i, txt in enumerate(items))
    return f"{head}\n{body}"


def score_rung(rung, ladder, responses):
    """Per-constraint satisfaction rate across `responses`, then their mean = rung adherence.
    Returns {adherence, per_constraint}. adherence is None only for a constraint-free rung."""
    defs = ladder["constraints"]
    per_constraint = {}
    for c in rung["constraints"]:
        params = defs[c].get("params", {})
        sat    = [check(c, params, r) for r in responses]
        rate   = (sum(1 for s in sat if s) / len(sat)) if sat else 0.0
        per_constraint[c] = round(rate, 4)
    adherence = round(statistics.mean(per_constraint.values()), 4) if per_constraint else None
    return {"adherence": adherence, "per_constraint": per_constraint}


def classify(prompt_sigma, adherence, cutoffs):
    """Producer-side categorical verdict from DECLARED cutoffs. Keeps the disambiguation on
    the producer so no consumer reads prompt-σ alone and draws the wrong conclusion."""
    flat = prompt_sigma < cutoffs["sigma_hi"]
    if flat and adherence >= cutoffs["adherence_hi"]:
        return "robust"
    if flat and adherence < cutoffs["adherence_lo"]:
        return "prompt-deaf"
    return "prompt-sensitive"


# ── Deterministic self-test (no model) ────────────────────────────────────────
if __name__ == "__main__":
    fails = []

    def expect(cond, msg):
        if not cond:
            fails.append(msg)

    # checker-level
    expect(check("word_cap", {"max_words": 5}, "one two three"), "word_cap under")
    expect(not check("word_cap", {"max_words": 5}, "one two three four five six"), "word_cap over")
    expect(check("no_exclamation", {}, "calm prose."), "no_exclamation clean")
    expect(not check("no_exclamation", {}, "wow!"), "no_exclamation hit")
    expect(check("no_lists", {}, "just a sentence, no list here."), "no_lists clean")
    expect(not check("no_lists", {}, "plan:\n- step one\n- step two"), "no_lists dash")
    expect(not check("no_lists", {}, "1. first\n2. second"), "no_lists numbered")
    expect(check("end_with_question", {}, "So what now?"), "end_with_question yes")
    expect(not check("end_with_question", {}, "This is a statement."), "end_with_question no")
    expect(not check("end_with_question", {}, ""), "end_with_question empty")
    expect(check("required_prefix", {"prefix": "NB:"}, "NB: here we go"), "required_prefix yes")
    expect(not check("required_prefix", {"prefix": "NB:"}, "here we go"), "required_prefix no")

    ladder = load_ladder()

    # render: heavy rung mentions all 5 instructions
    block = render_constraints(ladder["rungs"][-1], ladder)
    for c in ladder["rungs"][-1]["constraints"]:
        expect(ladder["constraints"][c]["instruction"] in block, f"render missing {c}")

    # score_rung: a perfectly-obedient heavy-rung set → adherence 1.0
    heavy = ladder["rungs"][-1]
    good  = ["NB: short prose answer ending in a query?"] * 8
    s_good = score_rung(heavy, ladder, good)
    expect(s_good["adherence"] == 1.0, f"score_rung obedient → {s_good['adherence']}")

    # a flagrantly-disobedient set → adherence well below adherence_lo
    bad = ["WOW!!! here is a list:\n- a\n- b\n" + ("word " * 200)] * 8
    s_bad = score_rung(heavy, ladder, bad)
    expect(s_bad["adherence"] < ladder["verdict_cutoffs"]["adherence_lo"],
           f"score_rung disobedient → {s_bad['adherence']}")

    # verdict regions
    cut = ladder["verdict_cutoffs"]
    expect(classify(0.02, 0.95, cut) == "robust", "verdict robust")
    expect(classify(0.02, 0.20, cut) == "prompt-deaf", "verdict prompt-deaf")
    expect(classify(0.25, 0.95, cut) == "prompt-sensitive", "verdict sensitive (high σ)")
    expect(classify(0.02, 0.65, cut) == "prompt-sensitive", "verdict sensitive (mid adherence)")

    if fails:
        print("SELF-TEST FAILED:")
        for f in fails:
            print("  ✗", f)
        raise SystemExit(1)
    print("adherence.py self-test: ✓ all checks pass "
          f"(obedient adherence={s_good['adherence']}, disobedient={s_bad['adherence']})")
