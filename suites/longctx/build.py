#!/usr/bin/env python3
"""
Battery G — Long-Context Retrieval dataset builder.

Deterministic, self-contained (no external download — the GraphWalks analog, BenchLLAMA-style).
Generates, per token bucket, ONE haystack prompt that:
  • fills the context to a target size with plausible distractor text, and
  • plants verifiable NEEDLES at controlled positions (early ~10% / mid ~50% / late ~90%), plus
  • scatters a 3-hop "X manages Y" chain so the answer needs multi-fact retrieval + reasoning.

The trailing question asks for all four answers at once → one call per depth grades 4 objective
sub-tasks (3 positional single-needle retrievals + 1 multi-hop walk). Position-resolved scoring
exposes the "lost in the middle" effect; the per-depth timings (captured by the runner) expose the
prefill/decode speed-collapse. Distinct from Battery C's C4, which re-runs a SHORT prompt at bigger
num_ctx *allocations* and never actually fills the window.

  python3 suites/longctx/build.py            # → suites/longctx/dataset.json
  python3 suites/longctx/build.py --deep     # also emit the 32768 bucket

Reproducible: fixed seed. Target ~0.70 words/token (conservative) so actual prompt tokens land just
UNDER the bucket and the runner's num_ctx margin never truncates our own planted needles.
"""

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT  = HERE / "dataset.json"

SEED          = 42
WORDS_PER_TOK = 0.70                 # conservative: keeps actual tokens under the bucket
THRESHOLD     = 0.75                 # per-depth "clean" accuracy bar (3 of 4 sub-tasks)
BUCKETS       = [1024, 2048, 4096, 8192, 16384]
DEEP_BUCKET   = 32768

# ── Needles (objective, exact-checkable) ──────────────────────────────────────
# Codes are arbitrary 4-digit strings; the relays are distinct so the three single-needle
# retrievals don't collide. The manage-chain terminal is the multi-hop answer.
NEEDLES = {
    "needle_early": {"text": "OPERATIONS NOTICE: the activation code for the NORTH relay is 4821.", "answer": "4821"},
    "needle_mid":   {"text": "OPERATIONS NOTICE: the activation code for the EAST relay is 7193.",  "answer": "7193"},
    "needle_late":  {"text": "OPERATIONS NOTICE: the activation code for the SOUTH relay is 3056.", "answer": "3056"},
}
# 3-hop chain scattered through the haystack. Marcus → Priya → Theodore → Helena (top).
HOPS = [
    "STAFF RECORD: Helena manages Theodore.",
    "STAFF RECORD: Theodore manages Priya.",
    "STAFF RECORD: Priya manages Marcus.",
]
MULTIHOP_ANSWER = "Helena"

QUESTION = (
    "\n\nUsing ONLY the facts stated in the text above, answer all four questions. "
    "Each answer is stated explicitly somewhere above — find it.\n"
    "1. What is the activation code for the NORTH relay?\n"
    "2. What is the activation code for the EAST relay?\n"
    "3. What is the activation code for the SOUTH relay?\n"
    "4. Marcus is managed by someone, who is managed by someone, and so on. "
    "Who is at the very TOP of Marcus's management chain?\n"
    "Answer each on its own line, numbered 1-4."
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
    """Emit distractor sentences until at least n_words words; return the text."""
    out, count = [], 0
    while count < n_words:
        s = _sentence(rng)
        out.append(s)
        count += len(s.split())
    return " ".join(out)

def build_item(bucket, rng):
    target_words = round(bucket * WORDS_PER_TOK)
    fixed = [NEEDLES["needle_early"]["text"], NEEDLES["needle_mid"]["text"],
             NEEDLES["needle_late"]["text"], *HOPS, QUESTION]
    fixed_words = sum(len(s.split()) for s in fixed)
    filler_budget = max(0, target_words - fixed_words)

    # 6 planted facts → 7 filler segments. Weight so needles sit at ~10/50/90% and the hops
    # scatter between them. Segment fractions of the filler budget:
    fracs = [0.10, 0.18, 0.16, 0.16, 0.16, 0.14, 0.10]
    segs  = [_filler(rng, max(8, round(filler_budget * f))) for f in fracs]

    # Interleave: seg0 [early] seg1 [hop0] seg2 [mid] seg3 [hop1] seg4 [late] seg5 [hop2] seg6
    body = " ".join([
        segs[0], NEEDLES["needle_early"]["text"],
        segs[1], HOPS[0],
        segs[2], NEEDLES["needle_mid"]["text"],
        segs[3], HOPS[1],
        segs[4], NEEDLES["needle_late"]["text"],
        segs[5], HOPS[2],
        segs[6],
    ])
    preface = ("You are reading an operations log. Most lines are routine distractors; a few "
               "carry specific facts you will be asked about. Read carefully.\n\n")
    prompt = preface + body + QUESTION
    return {
        "bucket": bucket,
        "target_words": target_words,
        "actual_words": len(prompt.split()),
        "answer_key": {
            "needle_early": NEEDLES["needle_early"]["answer"],
            "needle_mid":   NEEDLES["needle_mid"]["answer"],
            "needle_late":  NEEDLES["needle_late"]["answer"],
            "multihop":     MULTIHOP_ANSWER,
        },
        "prompt": prompt,
    }

def main():
    buckets = list(BUCKETS) + ([DEEP_BUCKET] if "--deep" in sys.argv else [])
    rng = random.Random(SEED)
    items = [build_item(b, rng) for b in buckets]
    data = {
        "meta": {
            "buckets": buckets, "words_per_tok": WORDS_PER_TOK, "threshold": THRESHOLD,
            "subtasks": ["needle_early", "needle_mid", "needle_late", "multihop"],
            "note": "single-needle x3 (early/mid/late position) + 3-hop manage-chain; objective exact-match",
        },
        "items": items,
    }
    OUT.write_text(json.dumps(data, indent=2))
    print(f"→ wrote {OUT}  ({len(items)} buckets: {buckets})")
    for it in items:
        print(f"   bucket {it['bucket']:>6}  ~{it['actual_words']:>6} words")


if __name__ == "__main__":
    main()
