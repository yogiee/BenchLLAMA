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

# ── Hard-band checkers (added 2026-08-23) ─────────────────────────────────────
# The five core constraints are all STATELESS SURFACE FORM — "no !", "prose only", "end with ?",
# "start with NB:". A modern instruction-tuned model satisfies those at generation time with no
# tracking, no counting and no conflict with what it wants to say, so the meter saturated:
# instruction_adherence mean 0.951 with 11 of 22 models at exactly 1.000 (fleet run 2026-08-22).
# The hard band attacks the three things that are genuinely difficult and still deterministic:
#   suppress a token class the register depends on · resist a constraint the ROLLOUT pushes
#   against · carry state across turns.
_FIRST_PERSON_RE = re.compile(r"\b(?:i|me|my|mine|myself)\b", re.I)
#   \bi\b matches the "I" in "I'm"/"I'll" — the apostrophe is a non-word char, so the boundary
#   holds. That is intended: contractions are first person.
_APOLOGY_RE = re.compile(r"\b(?:sorry|apolog\w*|regret\w*)\b", re.I)


def _opener(text, prefix=None):
    """First word of a reply, skipping a required prefix so the two constraints compose rather
    than contradict (every reply starting 'NB:' would otherwise have an identical opener)."""
    s = (text or "").lstrip()
    if prefix and s.startswith(prefix):
        s = s[len(prefix):].lstrip()
    m = re.search(r"\w+", s)
    return m.group(0).lower() if m else None


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
    # ── hard band ──
    if cid == "no_first_person":
        return _FIRST_PERSON_RE.search(t) is None
    if cid == "no_apology":
        return _APOLOGY_RE.search(t) is None
    raise KeyError(f"unknown constraint id: {cid}")


def check_rollout(cid, params, responses):
    """Constraints scored over the WHOLE rollout rather than one reply at a time.

    Returns a per-response bool list so a rollout constraint aggregates identically to a
    per-reply one (rate = satisfied / total) and needs no special case in score_rung.
    """
    if cid == "distinct_openers":
        prefix, seen, out = params.get("after_prefix"), set(), []
        for r in responses:
            w = _opener(r, prefix)
            ok = bool(w) and w not in seen        # an empty reply has no opener -> cannot satisfy
            if w:
                seen.add(w)
            out.append(ok)
        return out
    raise KeyError(f"unknown rollout constraint id: {cid}")


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
    """Per-constraint satisfaction rate across `responses`, split by constraint CLASS.

    word_cap is a continuous, verbosity-correlated signal; the other constraints are binary
    obey-or-ignore. Averaging them together lets verbosity masquerade as prompt-insensitivity
    (LookingGlass validation 2026-06-21), so we aggregate the two classes SEPARATELY:
      • instruction_adherence — mean of the binary-obedience constraints (verdict driver)
      • length_adherence      — mean of the length/verbosity constraints (standalone meter)
    `adherence` is kept as the all-constraint mean for reference. Each class field is None when
    that class has no constraint in the rung (e.g. the minimal rung carries length only)."""
    defs = ladder["constraints"]
    per_constraint, instr, length = {}, [], []
    for c in rung["constraints"]:
        params = defs[c].get("params", {})
        # scope "rollout" = judged across the whole conversation (cross-turn state), still
        # returning one bool per reply so it aggregates exactly like a per-reply constraint.
        sat    = (check_rollout(c, params, responses) if defs[c].get("scope") == "rollout"
                  else [check(c, params, r) for r in responses])
        rate   = round((sum(1 for s in sat if s) / len(sat)) if sat else 0.0, 4)
        per_constraint[c] = rate
        (length if defs[c].get("class") == "length" else instr).append(rate)
    mean = lambda xs: round(statistics.mean(xs), 4) if xs else None
    return {"adherence": mean(list(per_constraint.values())),
            "instruction_adherence": mean(instr),
            "length_adherence": mean(length),
            "per_constraint": per_constraint}


def classify(prompt_sigma, instruction_adherence, cutoffs, hard_adherence=None):
    """Producer-side categorical verdict from DECLARED cutoffs, keyed on INSTRUCTION adherence
    (binary obey-or-ignore) — NOT the verbosity-correlated length cap. Keeps the disambiguation
    on the producer so no consumer reads prompt-σ alone and draws the wrong conclusion.

    TWO-BAND since 2026-08-23. `robust` now requires the CORE band (the v1 five, unchanged, so the
    number stays comparable) AND the HARD band. The core band alone had stopped separating —
    mean 0.951, 11 of 22 models at exactly 1.000 — because all five core constraints are stateless
    surface form. `hard_adherence=None` (a pre-hard-band result) falls back to core-only scoring so
    old rows still classify rather than crashing.
    """
    flat = prompt_sigma < cutoffs["sigma_hi"]
    if flat and instruction_adherence < cutoffs["adherence_lo"]:
        return "prompt-deaf"                     # ignores even the easy band
    if not flat or instruction_adherence < cutoffs["adherence_hi"]:
        return "prompt-sensitive"
    if hard_adherence is not None and hard_adherence < cutoffs.get("hard_adherence_hi", 0.0):
        return "prompt-sensitive"                # obeys surface form, breaks under real load
    return "robust"


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

    # hard band
    expect(check("no_first_person", {}, "the answer depends on scope."), "no_first_person clean")
    expect(not check("no_first_person", {}, "I think so."), "no_first_person I")
    expect(not check("no_first_person", {}, "I'm certain."), "no_first_person contraction")
    expect(not check("no_first_person", {}, "That works for me."), "no_first_person me")
    expect(check("no_first_person", {}, "Mining is unrelated."), "no_first_person substring safe")
    expect(check("no_apology", {}, "That is incorrect."), "no_apology clean")
    expect(not check("no_apology", {}, "Sorry about that."), "no_apology sorry")
    expect(not check("no_apology", {}, "Apologies, my error."), "no_apology apologies")
    expect(not check("no_apology", {}, "We regret the delay."), "no_apology regret")

    # rollout scope: one bool per reply, first repeat onwards fails
    expect(check_rollout("distinct_openers", {}, ["alpha one", "beta two", "gamma three"])
           == [True, True, True], "distinct_openers all distinct")
    expect(check_rollout("distinct_openers", {}, ["alpha one", "alpha two"])
           == [True, False], "distinct_openers repeat")
    # composes with required_prefix instead of contradicting it
    expect(check_rollout("distinct_openers", {"after_prefix": "NB:"}, ["NB: alpha", "NB: beta"])
           == [True, True], "distinct_openers skips prefix")
    expect(check_rollout("distinct_openers", {"after_prefix": "NB:"}, ["NB: alpha", "NB: alpha"])
           == [True, False], "distinct_openers sees past prefix")
    expect(check_rollout("distinct_openers", {}, ["", "x"]) == [False, True], "distinct_openers empty")

    ladder = load_ladder()
    by_id  = {r["id"]: r for r in ladder["rungs"]}

    # render: every rung's block names every one of its instructions
    for rung in ladder["rungs"]:
        block = render_constraints(rung, ladder)
        for c in rung["constraints"]:
            expect(ladder["constraints"][c]["instruction"] in block, f"render missing {c} in {rung['id']}")

    # score_rung: a perfectly-obedient heavy-rung set → both class adherences 1.0
    heavy = by_id["heavy"]
    good  = ["NB: short prose answer ending in a query?"] * 8
    s_good = score_rung(heavy, ladder, good)
    expect(s_good["adherence"] == 1.0, f"score_rung obedient → {s_good['adherence']}")
    expect(s_good["instruction_adherence"] == 1.0, f"obedient instr → {s_good['instruction_adherence']}")
    expect(s_good["length_adherence"] == 1.0, f"obedient length → {s_good['length_adherence']}")

    # the split's whole point: a VERBOSE-but-otherwise-obedient set → instruction 1.0, length 0.0
    verbose = ["NB: " + ("word " * 200) + "and so on?"] * 8   # obeys prefix/question/prose/no-!, blows the cap
    s_verb = score_rung(heavy, ladder, verbose)
    expect(s_verb["instruction_adherence"] == 1.0, f"verbose instr should stay 1.0 → {s_verb['instruction_adherence']}")
    expect(s_verb["length_adherence"] == 0.0, f"verbose length should crater → {s_verb['length_adherence']}")

    # minimal rung is length-only → instruction_adherence is None there
    s_min = score_rung(by_id["minimal"], ladder, good)
    expect(s_min["instruction_adherence"] is None, "minimal rung has no instruction class")

    # ── hard rung ──
    hard = by_id["hard"]
    # the core-rung fixture is fully obedient on the core five and must NOT score 1.0 here:
    # identical openers every turn, which is exactly the state-tracking the band tests for.
    s_hard_core_only = score_rung(hard, ladder, good)
    expect(s_hard_core_only["instruction_adherence"] < 1.0,
           f"hard rung must bite the core-obedient fixture → {s_hard_core_only['instruction_adherence']}")
    # a fully-obedient hard-rung set: distinct openers, no first person, no apology
    openers = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    s_hard = score_rung(hard, ladder, [f"NB: {w} covers the point, does that help?" for w in openers])
    expect(s_hard["instruction_adherence"] == 1.0, f"hard obedient instr → {s_hard['instruction_adherence']}")
    expect(s_hard["per_constraint"]["distinct_openers"] == 1.0, "hard obedient openers")
    # and the failure mode the band exists to catch: perfect surface form, first person throughout
    s_fp = score_rung(hard, ladder, [f"NB: {w}, I think that helps, right?" for w in openers])
    expect(s_fp["per_constraint"]["required_prefix"] == 1.0, "surface form still perfect")
    expect(s_fp["per_constraint"]["no_first_person"] == 0.0, "hard band catches first person")
    expect(s_fp["instruction_adherence"] < 1.0, "hard instr drops on first person")

    # classify: the hard band can demote a core-perfect model
    cut = ladder["verdict_cutoffs"]
    expect(classify(0.01, 1.0, cut, hard_adherence=1.0) == "robust", "classify robust both bands")
    expect(classify(0.01, 1.0, cut, hard_adherence=0.10) == "prompt-sensitive", "classify hard demotes")
    expect(classify(0.01, 1.0, cut) == "robust", "classify back-compat (no hard band)")
    expect(classify(0.01, 0.20, cut, hard_adherence=1.0) == "prompt-deaf", "classify deaf on core")
    expect(s_min["length_adherence"] is not None, "minimal rung has length class")

    # verdict regions (keyed on instruction_adherence)
    cut = ladder["verdict_cutoffs"]
    expect(classify(0.02, 0.95, cut) == "robust", "verdict robust")
    expect(classify(0.02, 0.20, cut) == "prompt-deaf", "verdict prompt-deaf")
    expect(classify(0.25, 0.95, cut) == "prompt-sensitive", "verdict sensitive (high σ)")
    expect(classify(0.02, 0.65, cut) == "prompt-sensitive", "verdict sensitive (mid instr-adherence)")
    # the verbose case must NOT flip the verdict: high instr adherence → robust regardless of length
    expect(classify(0.04, s_verb["instruction_adherence"], cut) == "robust",
           "verbose-but-obedient stays robust")

    if fails:
        print("SELF-TEST FAILED:")
        for f in fails:
            print("  ✗", f)
        raise SystemExit(1)
    print("adherence.py self-test: ✓ all checks pass "
          f"(obedient instr/len={s_good['instruction_adherence']}/{s_good['length_adherence']}, "
          f"verbose instr/len={s_verb['instruction_adherence']}/{s_verb['length_adherence']})")
