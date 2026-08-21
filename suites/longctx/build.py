#!/usr/bin/env python3
"""
Battery G — Long-Context Retrieval dataset builder.  **TWO-BAND (v2, 2026-08-21).**

Deterministic, self-contained (the GraphWalks analog, BenchLLAMA-style). Per token bucket it emits
ONE haystack prompt that fills the context with plausible distractor text and plants verifiable
facts, then asks 7 questions in a single call.

── WHY v2 ──────────────────────────────────────────────────────────────────────
v1 SATURATED completely on the 2026-08-21 fleet run: all 21 models clean-depthed 32768 and
early/mid/late needle recall was 1.00 for **every model at every depth**. Three compounding defects:

  1. **The threshold was satisfiable by the easy band alone.** clean = accuracy >= 0.75 and there
     were exactly 4 sub-tasks, so 3 trivial needles = 0.75 = clean. `minicpm-v4.6:1b` failed the
     multi-hop at EVERY depth and was still reported "clean to 32k". The headline metric was free.
  2. **The needles were lexically unique.** They were the only lines in the haystack containing a
     digit, or the words "OPERATIONS NOTICE" / "activation code" / "relay". Finding them needed no
     comprehension — just a pattern match on the one line that looked different.
  3. **No decoys.** Nothing punished grabbing the first plausible match.

v2 keeps the same shape but makes each of those cost something:

  **G-core** (3 sub-tasks — the "can it retrieve at depth?" gate)
     The 3 positional needles (early ~10% / mid ~50% / late ~90%), now competing against SIX decoy
     relay notices in the same format with their own 4-digit codes. "Find the line with a number"
     no longer works; the right relay name has to be matched.

  **G-hard** (4 sub-tasks — the discriminator)
     • `multihop`   3-hop manage-chain (unchanged from v1 — decoy chains were tried and REVERTED,
                    see the note at HOPS; they add ordering noise, not difficulty).
     • `superseded` a relay code planted early and CORRECTED later in the haystack. The answer is the
                    LATEST value. Catches first-match grabbing, which v1 rewarded.
     • `aggregate`  the SUM of two scattered codes. Cannot be lexically matched at all — both facts
                    must be located and combined.
     • `absent`     a relay that is NEVER mentioned. The correct answer is to say so. This is
                    **confabulation-at-depth** — the failure that actually bites in RAG, and the one
                    v1 could not see. Ties Battery G to the Battery H concern.

`clean_depth` now requires BOTH bands to clear their bar, so a model that never solves a hard task
can no longer be reported as clean at 32k.

  python3 suites/longctx/build.py            # → suites/longctx/dataset.json
  python3 suites/longctx/build.py --deep     # also emit the 32768 bucket

Reproducible: fixed seed. Target ~0.70 words/token (conservative) so actual prompt tokens land just
UNDER the bucket and the runner's num_ctx margin never truncates our own planted facts.
"""

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT  = HERE / "dataset.json"

SEED           = 42
WORDS_PER_TOK  = 0.70    # conservative: keeps actual tokens under the bucket
CORE_THRESHOLD = 0.66    # G-core bar: 2 of 3 needles. ⚠ 2/3 stores as round(...,3)=0.667,
                         # so a 0.67 bar silently means 3-of-3 — keep this strictly BELOW 0.667.
HARD_THRESHOLD = 0.50    # G-hard bar: 2 of 4 discriminator tasks (0.5 stores exactly)
BUCKETS        = [1024, 2048, 4096, 8192, 16384]
DEEP_BUCKET    = 32768

# ── Planted facts ─────────────────────────────────────────────────────────────
NOTICE = "OPERATIONS NOTICE: the activation code for the {name} relay is {code}."

# The three positional needles (G-core). Asked as Q1-Q3.
NEEDLES = {
    "needle_early": {"name": "NORTH", "code": "4821"},
    "needle_mid":   {"name": "EAST",  "code": "7193"},
    "needle_late":  {"name": "SOUTH", "code": "3056"},
}
# Decoys: same format, same salience, never asked about. These are what make G-core non-trivial.
# ⚠ No decoy name may CONTAIN the absent relay's name as a substring (an early draft used
# "SOUTHWEST", which contains "WEST" and silently broke the absent-fact premise). "NORTHEAST" is
# deliberate: it contains NORTH and EAST, so Q1/Q2 require distinguishing two separately-stated
# relays rather than substring-matching a direction.
DECOY_RELAYS = [("NORTHEAST", "5140"), ("RIDGELINE", "8267"), ("UPPER", "2935"),
                ("LOWER", "6714"), ("COASTAL", "1508"), ("HARBOR", "3372")]

# Superseded: planted early, corrected later. Answer = the LATEST value.
SUPERSEDED = {"name": "CENTRAL", "old": "6602", "new": "9418"}
CORRECTION = ("CORRECTION TO EARLIER NOTICE: the activation code for the CENTRAL relay "
              "is no longer 6602 — the current code is 9418.")

# Aggregate: sum of the NORTH and SOUTH codes. 4821 + 3056 = 7877.
AGGREGATE_ANSWER = str(int(NEEDLES["needle_early"]["code"]) + int(NEEDLES["needle_late"]["code"]))

# Absent: never appears anywhere in the haystack. Correct answer = say it isn't stated.
ABSENT_RELAY = "WEST"

# 3-hop chain: Marcus → Priya → Theodore → Helena (top). SINGLE chain, deliberately.
#
# ⚠ DO NOT ADD DECOY CHAINS. Tried and reverted 2026-08-21. Adding two same-predicate decoy chains
# (Nadia→Oscar→Ruth, Ingrid→Bartholomew→Cleo) did not harden this task — it destabilised it. With
# multiple roots in one graph the answer flips with RECORD ORDER, not with model capability:
# controls run on the bare 7 records with NO haystack at all gave gemma4:12b-mlx "Helena" in one
# arrangement and "Ingrid" in another, and granite4.1:3b flipped the opposite way on the same pair.
# Every model failed the adjacent-decoy arrangement, and sharpening the question to bind explicitly
# to Marcus did not fix it. A task ~100% of models fail for positional reasons is as useless as one
# they all pass, and it would move with the seed. The single unambiguous chain already discriminates
# honestly on the real battery (fleet multihop recall spanned 1.00 → 0.00 on the 08-21 run).
HOPS = ["STAFF RECORD: Helena manages Theodore.",
        "STAFF RECORD: Theodore manages Priya.",
        "STAFF RECORD: Priya manages Marcus."]
MULTIHOP_ANSWER = "Helena"

QUESTION = (
    "\n\nUsing ONLY the facts stated in the text above, answer all seven questions.\n"
    "1. What is the activation code for the NORTH relay?\n"
    "2. What is the activation code for the EAST relay?\n"
    "3. What is the activation code for the SOUTH relay?\n"
    "4. What is the CURRENT activation code for the CENTRAL relay?\n"
    "5. What is the sum of the NORTH relay code and the SOUTH relay code?\n"
    "6. Marcus is managed by someone, who is managed by someone, and so on. "
    "Who is at the very TOP of Marcus's management chain?\n"
    f"7. What is the activation code for the {ABSENT_RELAY} relay? "
    "If it is not stated in the text, say exactly: NOT STATED.\n"
    "Answer each on its own line, numbered 1-7. Give just the answer, no explanation."
)

# ── Distractor generator (deterministic) ──────────────────────────────────────
_SUBJ = ["The day shift", "The night crew", "Dock team B", "The receiving bay", "Loader unit 7",
         "The inventory desk", "Aisle supervisor", "The cold-storage zone", "Pallet line 3",
         "The dispatch office", "Forklift bay 2", "The returns counter", "Quality control",
         "The mezzanine racks", "Shipping lane 4"]
_VERB = ["logged", "rerouted", "audited", "restocked", "scanned", "consolidated", "flagged",
         "cycle-counted", "staged", "expedited", "reconciled", "palletized", "labelled", "binned"]
_OBJ  = ["forty-two cartons of fasteners", "a partial skid of insulation", "the overflow from lane 9",
         "twelve totes of returns", "a mixed pallet of hardware", "the morning's inbound trailer",
         "three crates of glassware", "the damaged-goods queue", "a backlog of small parcels",
         "the seasonal overstock", "two rolls of shrink wrap", "the quarantine shelf"]
_TAIL = ["before the next wave.", "ahead of the cutoff.", "without incident.", "to clear the floor.",
         "per the standing rota.", "and updated the board.", "to balance the bays.",
         "while the scanner rebooted.", "for the afternoon pickup.", "to free up dock space."]

def _sentence(rng):
    return f"{rng.choice(_SUBJ)} {rng.choice(_VERB)} {rng.choice(_OBJ)} {rng.choice(_TAIL)}"

def _filler(rng, n_words):
    out, count = [], 0
    while count < n_words:
        s = _sentence(rng)
        out.append(s)
        count += len(s.split())
    return " ".join(out)

def build_item(bucket, rng):
    target_words = round(bucket * WORDS_PER_TOK)

    # Planted lines in reading order. Real needles hold their ~10/50/90% positions; decoys and the
    # hop records (real + decoy) are interleaved so no planted line is locally distinctive.
    planted = [
        NOTICE.format(**{"name": SUPERSEDED["name"], "code": SUPERSEDED["old"]}),   # ~5%  superseded (old)
        NOTICE.format(name=DECOY_RELAYS[0][0], code=DECOY_RELAYS[0][1]),
        NOTICE.format(name=NEEDLES["needle_early"]["name"], code=NEEDLES["needle_early"]["code"]),  # ~10% EARLY
        NOTICE.format(name=DECOY_RELAYS[1][0], code=DECOY_RELAYS[1][1]),
        HOPS[0],
        NOTICE.format(name=DECOY_RELAYS[2][0], code=DECOY_RELAYS[2][1]),
        NOTICE.format(name=NEEDLES["needle_mid"]["name"], code=NEEDLES["needle_mid"]["code"]),      # ~50% MID
        HOPS[1],
        NOTICE.format(name=DECOY_RELAYS[3][0], code=DECOY_RELAYS[3][1]),
        CORRECTION,                                                                 # ~70% supersede
        NOTICE.format(name=DECOY_RELAYS[4][0], code=DECOY_RELAYS[4][1]),
        HOPS[2],
        NOTICE.format(name=NEEDLES["needle_late"]["name"], code=NEEDLES["needle_late"]["code"]),    # ~90% LATE
        NOTICE.format(name=DECOY_RELAYS[5][0], code=DECOY_RELAYS[5][1]),
    ]
    fixed_words = sum(len(s.split()) for s in planted) + len(QUESTION.split())
    filler_budget = max(0, target_words - fixed_words)

    # One filler segment before each planted line, plus a trailing one.
    n_seg = len(planted) + 1
    fracs = [1.0 / n_seg] * n_seg
    segs = [_filler(rng, max(8, round(filler_budget * f))) for f in fracs]

    parts = []
    for i, line in enumerate(planted):
        parts.append(segs[i]); parts.append(line)
    parts.append(segs[-1])
    body = " ".join(parts)

    preface = ("You are reading an operations log. Most lines are routine distractors; several carry "
               "specific facts you will be asked about, and some facts are superseded later in the "
               "log. Read carefully.\n\n")
    prompt = preface + body + QUESTION
    return {
        "bucket": bucket,
        "target_words": target_words,
        "actual_words": len(prompt.split()),
        "answer_key": {
            # G-core
            "needle_early": NEEDLES["needle_early"]["code"],
            "needle_mid":   NEEDLES["needle_mid"]["code"],
            "needle_late":  NEEDLES["needle_late"]["code"],
            # G-hard
            "superseded":       SUPERSEDED["new"],
            "superseded_stale": SUPERSEDED["old"],
            "aggregate":        AGGREGATE_ANSWER,
            "multihop":         MULTIHOP_ANSWER,
            "absent":           None,
        },
        "question_index": {          # which numbered answer line carries which sub-task
            "needle_early": 1, "needle_mid": 2, "needle_late": 3,
            "superseded": 4, "aggregate": 5, "multihop": 6, "absent": 7,
        },
        "prompt": prompt,
    }

def main():
    buckets = list(BUCKETS) + ([DEEP_BUCKET] if "--deep" in sys.argv else [])
    rng = random.Random(SEED)
    items = [build_item(b, rng) for b in buckets]
    data = {
        "meta": {
            "version": 2,
            "buckets": buckets, "words_per_tok": WORDS_PER_TOK,
            "core_threshold": CORE_THRESHOLD, "hard_threshold": HARD_THRESHOLD,
            "threshold": CORE_THRESHOLD,        # back-compat alias
            "core_subtasks": ["needle_early", "needle_mid", "needle_late"],
            "hard_subtasks": ["superseded", "aggregate", "multihop", "absent"],
            "subtasks": ["needle_early", "needle_mid", "needle_late",
                         "superseded", "aggregate", "multihop", "absent"],
            "note": ("TWO-BAND. G-core = 3 positional needles against 6 same-format decoy relays. "
                     "G-hard = superseded-value / cross-fact aggregate / 3-hop walk / "
                     "absent-fact refusal. clean_depth requires BOTH bands."),
        },
        "items": items,
    }
    OUT.write_text(json.dumps(data, indent=2))
    print(f"→ wrote {OUT}  ({len(items)} buckets: {buckets})")
    for it in items:
        print(f"   bucket {it['bucket']:>6}  ~{it['actual_words']:>6} words")


if __name__ == "__main__":
    main()
