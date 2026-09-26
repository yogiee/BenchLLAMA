#!/usr/bin/env python3
"""
BenchLLAMA — neutral results/rankings export.

Writes `rankings/rankings.json`: the machine-readable twin of `rankings/master.md`.
`master.md` stays the human view; this is the consumer-facing data feed. It aggregates
the latest canonical (non-`_fast`) result file of every battery into one model-keyed
structure plus per-category ranking lists.

PROJECT-AGNOSTIC by design: the producer knows nothing about its consumers. This file
contains only WHAT WAS MEASURED — no consumer tool names (no `local_code`, no
`local_embed`). Each consumer (OllamaMCP, LookingGlass, MemoryCentral) reads the lists
it cares about and maps them to its own purpose on its side.

  python3 export.py                 # → rankings/rankings.json
  python3 export.py --print         # also dump a short summary to stdout

Regenerate alongside any master.md update.
"""

import json
import sys
import glob
import os
import re
from pathlib import Path
from datetime import datetime, timezone

from bench_utils import H_PROFILE_CUTOFFS

REPO = Path(__file__).parent
RESULTS = REPO / "results"
OUT = REPO / "rankings" / "rankings.json"
# Well-known consumer location — the published source of truth every consumer
# (OllamaMCP, LookingGlass, MemoryCentral, …) reads from. Overwritten each run;
# the date lives inside the data (`generated`). Decoupled from this repo's layout.
PUBLISH = Path.home() / ".config" / "ollama-consumers" / "benchllama-rankings.json"

SCHEMA = 1
HOST_PROFILE = "M1 Max 32GB"
PROTOCOL = {"num_ctx": 16384, "protocol_version": 3,
            "think": "per-model arm — see models[*].think (docs/think-spec.md); standard = direct arm, "
                     "standard_think = think arm, batteries at the operating point"}

# F-elastic ranking: hard-adherence at/above this counts as SATURATED — within that tier models are
# ordered by per-turn cost, not by adherence (see elastic_key). Policy, not test content.
F_ELASTIC_RANK_BAND = 0.95


def _latest(prefix):
    """Newest canonical result file for a battery prefix, or None. Skips informal /
    intermediate variants: _fast (no-cooldown), _run{k} (per-run averaging inputs).
    The `_<date>` guard keeps a prefix from matching a LONGER sibling prefix — e.g.
    `aptitude_f` must not pick up `aptitude_f_elastic_*` (the next segment is a word, not a date)."""
    skip = ("_fast", "_run")
    pat = re.compile(rf"{re.escape(prefix)}_\d{{4}}-\d\d-\d\d")
    files = [f for f in glob.glob(str(RESULTS / f"{prefix}_*.json"))
             if not any(s in os.path.basename(f) for s in skip) and pat.match(os.path.basename(f))]
    return max(files, key=os.path.getmtime) if files else None


# export prefix → SQLite battery key (Phase 2: read the clobber-proof store, not latest-file-by-mtime)
_PREFIX_BATTERY = {
    "benchmark": "standard", "aptitude_a": "A", "aptitude_e": "E", "aptitude_f": "F",
    "aptitude_f_elastic": "F-elastic", "vision": "vision", "embedding": "embedding", "longctx": "G",
    "confab": "confab",
}


def _load_operating(prefix):
    """(main, other, source) — one row per model at its OPERATING-POINT arm (`think` if it has an
    operating lever, else `direct` — bench_utils.resolve_arm(…, "auto"), the arm the batteries run
    at), plus the other arm's row when one exists.

    Why (2026-09-26): `_load(arm=None)` returns the NEWEST row whatever its arm, so a deliberate
    direct-arm measurement silently replaced a model's operating-point result in the export — and in
    resume, which then carried it forward as if it were the operating point. Used for F-elastic only
    for now (user decision 09-26; the other batteries still read newest-of-any-arm)."""
    bat = _PREFIX_BATTERY[prefix]
    import results_db
    from bench_utils import resolve_arm, think_lever
    by = {a: results_db.latest(bat, arm=a) for a in ("direct", "think")}
    lever_now = lambda n, a: str(think_lever(n, a)).lower()
    main, other = {}, {}
    for name in set(by["direct"]) | set(by["think"]):
        want = resolve_arm(name, "auto")
        alt = "direct" if want == "think" else "think"
        if name in by[want]:
            main[name] = by[want][name]
            # the other arm only from v3 rows measured at the lever that arm uses TODAY — a pre-v3 row
            # (no `think_lever` stamp) ran under blanket think=False, and a row at a since-moved lever
            # (lfm2.5:8b's 09-02 `low` row; its profile now has no operating lever) is not this arm's number
            o = by[alt].get(name)
            if o and "think_lever" in o and str(o["think_lever"]).lower() == lever_now(name, alt):
                other[name] = o
        else:                                   # never measured at its operating arm — show what exists
            main[name] = by[alt][name]
    return main, other, f"db:{bat}@operating"


def _load(prefix, arm=None):
    """Latest per-model result for a battery. Prefers the SQLite store (results_db.latest = each
    model's most-recent result across all runs, so a partial/midnight re-run never drops models);
    falls back to the dated JSON file if the DB is empty/unavailable (transition safety)."""
    bat = _PREFIX_BATTERY.get(prefix)
    if bat:
        try:
            import results_db
            data = results_db.latest(bat, arm=arm)
            if data:
                return data, f"db:{bat}" + (f"@{arm}" if arm else "")
        except Exception:
            pass
    f = _latest(prefix)
    if not f:
        return {}, None
    try:
        data = json.load(open(f))
        return {r["model"]: r for r in data if "model" in r}, os.path.basename(f)
    except Exception:
        return {}, None


def _standard_summary(rec):
    """Objective standard-suite signals from a benchmark record."""
    t = rec.get("tests", {})
    def ok(name):
        return bool(t.get(name, {}).get("correct"))
    reasoning = sum(ok(x) for x in ("bat_ball", "two_cities", "cylinder", "farm_heads"))
    instr = sum(ok(x) for x in ("format_3", "no_eiffel"))
    # ⚠ Both signal-counted probes live under `check_detail`, NOT at the top level. `jpeg` read
    # `t["jpeg_formats"]["signals"]` and so exported as ABSENT for every model since it was added —
    # the same silent-hole family as Battery A never reaching export (Rule #17).
    jpeg = ((t.get("jpeg_formats") or {}).get("check_detail") or {}).get("score")
    expense = ((t.get("expense_split") or {}).get("check_detail") or {}).get("score")
    return {"reasoning": reasoning, "instr": instr, "tool": ok("calculate"),
            # ⚠ jpeg has a documented +/-2 noise floor (same model, same prompt, 4 contexts ->
            # 7/6/5/6). Published for provenance; NEVER read a delta and never rank on it.
            **({"jpeg": jpeg, "jpeg_max": 7} if jpeg is not None else {}),
            **({"expense_split": expense, "expense_split_max": 7} if expense is not None else {})}


def _routing_summary(rec):
    """Battery A signals — the router lane's actual job, previously measured but never exported.

    `classify_accuracy` (A1) is the core number: does the request land in the right bucket?
    `false_escalation_rate` (A4) is the cost of getting it wrong the EXPENSIVE way — waking the big
    model for nothing. `prompt_weight_accuracy` (A3) shows how much accuracy survives a stripped
    system prompt, which is how routers are usually deployed. `brevity` (A2) is saturated across the
    fleet (everyone 5/5) — kept for provenance, not for ranking."""
    t = rec.get("tests", {})
    a1 = t.get("a1_classify") or {}
    if a1.get("accuracy") is None:
        return None
    a2, a3, a4 = (t.get("a2_brevity") or {}, t.get("a3_prompt_minimal") or {},
                  t.get("a4_false_escalation") or {})
    rungs = {r: (a3.get(r) or {}).get("accuracy") for r in ("minimal", "standard", "verbose")}
    out = {
        "classify_accuracy": a1.get("accuracy"),
        "classify_correct": a1.get("correct"),
        "classify_total": a1.get("total"),
        "false_escalation_rate": a4.get("escalation_rate"),
        "prompt_weight_accuracy": {k: v for k, v in rungs.items() if v is not None},
    }
    if rungs.get("minimal") is not None:
        out["lean_prompt_accuracy"] = rungs["minimal"]
    if a2.get("score") is not None and a2.get("total"):
        out["brevity"] = round(a2["score"] / a2["total"], 4)
    return out


def _elastic_view(pe, is_cloud):
    """Compact F-elastic view of one DB row: arm + verdict + meters + cost.

    `thinking` comes from the think CLASS, not the arm label: gpt-oss / lfm2.5 run their `direct` arm
    at a thinking lever because they have no working off switch. The verdict does not transfer across
    arms — thinking solves the cross-turn `distinct_openers` constraint that drives the hard band
    (09-26: 11 of 17 `robust` verdicts were earned while thinking)."""
    pes = pe["summary"]
    c = pes.get("cost") or {}
    return {
        "arm": pe.get("arm"),
        "think_lever": pe.get("think_lever"),
        "thinking": pe.get("think_class") not in (None, "off"),
        "verdict": pes.get("verdict"),
        "prompt_sigma": pes.get("prompt_sigma"),
        "instruction_adherence": pes.get("instruction_adherence"),
        # two-band since 2026-08-23 — absent on rows measured before the hard rung existed
        **({"hard_adherence": pes["hard_adherence"]} if pes.get("hard_adherence") is not None else {}),
        "length_adherence": pes.get("length_adherence"),
        # wall-clock cost per rollout turn INCLUDING attempts discarded by a budget retry. Cloud =
        # quality-only (remote hardware + network), so no cost. `retry_s_estimated` = the discarded
        # attempts were back-filled as num_predict / decode tps (rows measured before 09-26 instrumentation).
        **({"cost": {k: c.get(k) for k in ("s_per_turn", "retries_per_pass", "unrecovered",
                                           "retry_s_estimated")}} if c and not is_cloud else {}),
        "runs": pes.get("n_runs", 1),
    }


def _elastic_block(pe, is_cloud):
    """Full F-elastic sub-block for the headline (operating-point) row."""
    pes = pe["summary"]; rc = pes.get("cutoffs", {})
    return {
        **_elastic_view(pe, is_cloud),
        **({"prompt_sigma_all": pes["prompt_sigma_all"]} if pes.get("prompt_sigma_all") is not None else {}),
        # cutoffs trimmed to the DECLARED numeric thresholds so a consumer can re-threshold
        "cutoffs": {k: rc.get(k) for k in ("sigma_hi", "adherence_hi", "adherence_lo",
                                           "hard_adherence_hi", "keyed_on")},
        "verdict_stable": pes.get("verdict_stable"),
        "prompt_sigma_stdev": pes.get("prompt_sigma_stdev"),
        "instruction_adherence_stdev": pes.get("instruction_adherence_stdev"),
        "per_rung": [{"rung": r["rung"], "constraints_n": r["constraints_n"],
                      "composite": r["composite"], "run_sigma": r.get("run_sigma"),
                      "instruction_adherence": r["instruction_adherence"],
                      "length_adherence": r["length_adherence"],
                      # per-constraint hit rates — without them a saturating verdict can't be diagnosed
                      "per_constraint": r.get("per_constraint")}
                     for r in pes.get("per_rung", [])],
    }


def _elastic_arm_key(v):
    """Sort key for ONE measured arm of F-elastic (higher = better). Cost decides once adherence has
    saturated (user direction, 2026-09-26): v3 runs F-elastic at the operating point, thinking solves the
    cross-turn `distinct_openers` constraint, and 17 of 18 models came out `robust` — while the thinking
    pass cost 3-57x the per-turn time of the same model's no-think pass. Order:
      1. `robust` above everything else (the verdict still gates);
      2. within robust, hard-adherence >= F_ELASTIC_RANK_BAND = the saturated tier;
      3. within a tier, cheaper s/turn first (cost includes budget-retry discards);
      4. an arm with no cost (cloud = quality-only, or no timed files) after the costed ones of its tier.
    Non-robust arms sort by hard-adherence, then instruction-adherence."""
    if not v.get("verdict"):
        return None
    hard = v.get("hard_adherence") or 0.0
    if v["verdict"] != "robust":
        return (0, 0, hard, v.get("instruction_adherence") or 0.0)
    tier = 2 if hard >= F_ELASTIC_RANK_BAND else 1
    # a full arm view carries cost.s_per_turn; the published best_arm pointer carries it flat
    spt = v["s_per_turn"] if "s_per_turn" in v else (v.get("cost") or {}).get("s_per_turn")
    return (tier, 1, -spt, hard) if spt else (tier, 0, hard, 0.0)


def _elastic_best_arm(pe):
    """The measured arm that scores best on _elastic_arm_key — what `prompt_elasticity` ranks on (option B,
    09-26: aptitude = each model at its best settings, and 'best' includes what the pass costs). Returns
    a compact pointer so a consumer knows which arm the rank was earned at, i.e. which arm to call."""
    views = [pe] + [pe[k] for k in ("direct", "think") if isinstance(pe.get(k), dict)]
    views = [v for v in views if _elastic_arm_key(v) is not None]
    if not views:
        return None
    b = max(views, key=_elastic_arm_key)
    return {k: b.get(k) for k in ("arm", "think_lever", "thinking", "verdict", "hard_adherence",
                                  "instruction_adherence")} \
        | {"s_per_turn": (b.get("cost") or {}).get("s_per_turn")}


def build():
    registry = json.load((REPO / "models.json").open())
    std, std_f = _load("benchmark", arm="direct")
    std_t, std_t_f = _load("benchmark", arm="think")   # v3: the think arm, when a model has one
    coding, cod_f = _load("aptitude_e")
    cons, cons_f = _load("aptitude_f")
    elastic, elastic_other, ela_f = _load_operating("aptitude_f_elastic")
    vision, vis_f = _load("vision")
    emb, emb_f = _load("embedding")
    lctx, lctx_f = _load("longctx")
    honesty, hon_f = _load("confab")
    routing, rout_f = _load("aptitude_a")

    models, sources = [], {k: v for k, v in {
        "standard": std_f, "routing": rout_f, "coding": cod_f, "consistency": cons_f,
        "prompt_elasticity": ela_f, "vision": vis_f, "embedding": emb_f,
        "long_context": lctx_f, "honesty": hon_f}.items() if v}

    for entry in registry:
        name = entry["name"]
        # Cloud endpoints (e.g. gemma4:31b-cloud) are graded for QUALITY ONLY: their
        # tok/s and disk are meaningless locally (Ollama reports disk=0, tok/s≈1e10),
        # so we keep the quality composites (coding/consistency/vision) but suppress
        # every speed/footprint signal — no tps, no quality-per-GB (the disk=0 guard
        # below already nulls Q/GB), and never enter the speed-based ranking lists.
        is_cloud = bool(entry.get("cloud"))
        m = {
            "name": name,
            "disk_gb": None if is_cloud else entry.get("disk_gb"),
            "role": entry.get("role"),
            "extended_roles": entry.get("extended_roles", []),
            "capabilities": entry.get("capabilities", []),
            **({"cloud": True} if is_cloud else {}),
        }
        s = std.get(name)
        if s:
            # `tps` = DECODE tok/s (back-compat name consumers already read). prefill_tps and
            # wall_s are the new first-class latency signals: prefill = input-read speed (RAG /
            # big-prompt cost); wall_s = mean end-to-end seconds per test over the fixed suite
            # (the number the user actually waits for). Cloud endpoints null all three.
            m["tps"] = None if is_cloud else s.get("avg_tps")
            m["prefill_tps"] = None if is_cloud else s.get("avg_prefill_tps")
            m["wall_s"] = None if is_cloud else s.get("avg_wall_s")
            m["ram_gb"] = None if is_cloud else s.get("ram_gb")                    # Ollama's /api/ps estimate
            m["ram_measured_gb"] = None if is_cloud else s.get("ram_measured_gb")  # runner footprint (bench_utils)
            m["standard"] = _standard_summary(s)
        # v3 think-aware protocol: what the probe found, which arm the batteries ran at, and the
        # standard suite's think arm beside the fast one (docs/think-spec.md).
        tp = entry.get("think_profile") or {}
        if tp:
            m["think"] = {k: tp.get(k) for k in ("operating_point", "operating_lever", "direct_lever", "off_supported",
                                                  "default_thinks", "always_on", "think_unbounded", "probed_at")}
            m["think"]["battery_arm"] = "think" if tp.get("operating_lever") else "direct"
        st = std_t.get(name)
        if st:
            m["standard_think"] = _standard_summary(st)
            m["standard_think"].update({"tps": None if is_cloud else st.get("avg_tps"),
                                        "wall_s": None if is_cloud else st.get("avg_wall_s"),
                                        "think_lever": st.get("think_lever")})
        # the standard-suite view the batteries were measured under → what the worker key ranks on
        m["standard_operating"] = (m.get("standard_think") if (m.get("think") or {}).get("battery_arm") == "think"
                                   and m.get("standard_think") else m.get("standard"))
        c = coding.get(name)
        if c and c.get("summary"):
            cs = c["summary"]
            comp = cs.get("composite")
            disk = m.get("disk_gb")
            # Hysteresis made legible. `coder_eligible` answers "does it EARN the tag now?";
            # the tag in extended_roles also survives on the retain band, so the two disagree for
            # any retained model and a consumer reading the JSON cannot tell that from a stale
            # tag. coder_status names which case it is.
            _tag = "coder" in (entry.get("extended_roles") or [])
            m["coding"] = {
                "composite": comp,
                "category_means": cs.get("category_means", {}),
                "coder_eligible": cs.get("coder_eligible"),
                "coder_status": ("earned" if cs.get("coder_eligible")
                                 else ("retained" if _tag else "none")),
                "gate": cs.get("threshold"),
                "composite_stdev": cs.get("composite_stdev", 0.0),   # consistency (σ over runs)
                "composite_spread": cs.get("composite_spread", 0.0),
                "runs": cs.get("n_runs", 1),
                # quality-per-GB — lets the consumer trade quality vs footprint directly
                "quality_per_gb": round(comp / disk, 4) if (comp is not None and disk) else None,
            }
        fr = cons.get(name)
        if fr and fr.get("summary"):
            fs = fr["summary"]
            # consistency is a SUB-METRIC of chat/workers, not its own ranking list —
            # consumers weigh it against the workers ranking themselves.
            m["consistency"] = {"composite": fs.get("composite"),
                                 "composite_stdev": fs.get("composite_stdev"),
                                 "dims": fs.get("dims", {}),
                                 "runs": fs.get("n_runs", 1)}
        pe = elastic.get(name)
        if pe and pe.get("summary"):
            # prompt-elasticity is a per-model SUB-BLOCK (like `consistency`); the `prompt_elasticity`
            # ranking list (below) orders it. Headline = the OPERATING-POINT arm (see _load_operating);
            # a measurement at the other arm rides along under `direct` / `think` so a consumer that
            # calls the model with thinking off can read the number that applies to it.
            m["prompt_elasticity"] = _elastic_block(pe, is_cloud)
            po = elastic_other.get(name)
            if po and po.get("summary"):
                oview = _elastic_view(po, is_cloud)
                m["prompt_elasticity"][oview["arm"] or "other"] = oview
                # what thinking costs per turn on this battery, when both arms were measured locally
                mc, oc = m["prompt_elasticity"].get("cost"), oview.get("cost")
                if mc and oc and mc.get("s_per_turn") and oc.get("s_per_turn"):
                    th, nt = (mc, oc) if m["prompt_elasticity"]["thinking"] else (oc, mc)
                    if th is not nt:
                        m["prompt_elasticity"]["think_penalty"] = round(th["s_per_turn"] / nt["s_per_turn"], 2)
            best = _elastic_best_arm(m["prompt_elasticity"])
            if best:
                m["prompt_elasticity"]["best_arm"] = best
        v = vision.get(name)
        if v:
            m["vision"] = {"composite": v.get("composite"),           # 0.75·core + 0.25·hard (two-band)
                           "composite_core": v.get("composite_core"), # the `sees?` gate baseline (V-core)
                           "composite_hard": v.get("composite_hard"), # V-hard ranking discriminator (None on pre-hard runs)
                           "dimensions": v.get("dimensions", {})}
        e = emb.get(name)
        if e:
            m["embedding"] = {"composite": e.get("composite"),
                              "composite_long": e.get("composite_long"),
                              "quality_per_gb": e.get("quality_per_gb")}
        g = lctx.get(name)
        if g and g.get("summary"):
            gs = g["summary"]
            # long-context (Battery G) sub-block: accuracy degradation + speed collapse as the
            # window FILLS (distinct from C4's num_ctx allocation sweep). clean_depth = deepest
            # token bucket still ≥ threshold accuracy — the headline "usable to N tokens" number.
            # TWO-BAND as of dataset v2 (2026-08-21): G-core = 3 positional needles against 6
            # same-format decoy relays; G-hard = superseded / aggregate / multihop / absent.
            # `clean_depth` now requires BOTH bands — under v1 it was satisfiable by the 3 easy
            # needles alone and every model reported 32768. `clean_depth_core` is the v1-comparable
            # series. Keys are emitted only when present, so v1 rows stay valid.
            m["long_context"] = {
                "composite": gs.get("composite"),
                "clean_depth": gs.get("clean_depth"),
                "prefill_collapse": gs.get("prefill_collapse"),
                "accuracy_by_depth": gs.get("accuracy_by_depth", {}),
                "prefill_by_depth": gs.get("prefill_by_depth", {}),
                "position_recall": gs.get("position_recall", {}),
                "n_depths": gs.get("n_depths"),
                **({"composite_core": gs["composite_core"]} if gs.get("composite_core") is not None else {}),
                **({"composite_hard": gs["composite_hard"]} if gs.get("composite_hard") is not None else {}),
                **({"clean_depth_core": gs["clean_depth_core"]} if gs.get("clean_depth_core") is not None else {}),
                **({"subtask_recall": gs["subtask_recall"]} if gs.get("subtask_recall") else {}),
                **({"band_thresholds": {"core": gs.get("core_threshold"), "hard": gs.get("hard_threshold")}}
                   if gs.get("core_threshold") is not None else {}),
            }
        # Battery A is LANE-gated (only the router lane runs it), so a worker's A row is a fossil
        # from when it was last a router — results_db.latest() happily returns it years later. Publishing
        # that reads as a current capability: gpt-oss:120b-cloud carried classify_accuracy 0.0 from
        # 2026-06-29, which looks damning for a model that simply is not in the router lane. Emit the
        # sub-block only for models currently IN that lane; the fossils stay in the DB, unpublished.
        rt = routing.get(name) if entry.get("role") == "router" else None
        if rt:
            # routing (Battery A) sub-block: measured since 2026-06, exported since 2026-08-21.
            # Before that the `routers` ranking had NO quality input at all and sorted on tok/s alone.
            rs = _routing_summary(rt)
            if rs:
                m["routing"] = rs
        h = honesty.get(name)
        if h and h.get("summary"):
            hs = h["summary"]
            # honesty (Battery H) — sub-block PLUS a ranking list as of 2026-09-02.
            # ⚠ `confab_score` is carried for series continuity but MUST NOT be ranked on: it conflates
            # a pathological denier (aces the fakes by refusing everything, real controls included) with
            # an honest model, and on the 09-02 fleet that put two models with real_clean 0.000 in the
            # top nine. `honesty_balanced` = HARMONIC mean of the two bands, so it collapses when either
            # does; `honesty_profile` names the type. Rank on balanced, read the profile.
            # Still deliberately NOT folded into any quality composite — honesty is orthogonal, and the
            # deploy/veto decision stays the consumer's.
            m["honesty"] = {
                "confab_score": hs.get("composite"),
                "honesty_balanced": hs.get("honesty_balanced"),
                "honesty_profile": hs.get("honesty_profile"),
                "fabrication_rate": hs.get("fabrication_rate"),
                "fake_clean_rate": hs.get("fake_clean_rate"),
                "real_clean_rate": hs.get("real_clean_rate"),
                "n_items": hs.get("n_items"),
                "judge": hs.get("judge", []),
                "by_category": hs.get("by_category", {}),
                "_cutoffs": dict(H_PROFILE_CUTOFFS),
            }
        models.append(m)

    by_name = {m["name"]: m for m in models}

    def ranked(names, key, reverse=True):
        have = [n for n in names if key(by_name[n]) is not None]
        return sorted(have, key=lambda n: key(by_name[n]), reverse=reverse)

    completion = [m["name"] for m in models if m["role"] in ("worker", "router")]
    routers = [m["name"] for m in models if m["role"] == "router"]
    workers = [m["name"] for m in models if m["role"] == "worker"]
    has_vis = [m["name"] for m in models if "vision" in m]
    has_emb = [m["name"] for m in models if "embedding" in m]
    has_lctx = [m["name"] for m in models if "long_context" in m]

    def lctx_key(m):
        # rank by clean_depth first (deepest usable window), composite as tiebreaker
        lc = m.get("long_context") or {}
        if lc.get("composite") is None:
            return None
        return (lc.get("clean_depth") or 0, lc.get("composite"))

    def worker_quality(m):
        """Workers rank on the capability TIER first, then on signals that actually discriminate.

        The tier (reasoning/4 + instr/2 + tool) is unchanged — it is what `worker` means and what
        master.md's Tier 1/2 sections are built on. What changed on 2026-08-22 is everything after
        it: the tier ALONE was the whole key, and **10 of 16 workers tie at the maximum 7/7**, so
        the published order inside that block fell through to `models.json` position (a disk-size
        sort) and put a 1-bit 23 t/s model above `gemma4:26b-mlx`. Meanwhile B, C, D, E, F and H
        all ran, all reached the export as sub-blocks, and none of them reached the key.

        Tiebreaks, in order:
          `expense_split` (0-7) — the only standard-suite reasoning probe that still spreads
            (4 of 22 pass it outright; bat_ball and format_3 are 22/22).
          Battery F composite — continuous and 3-run averaged, so the order is always strict and
            never falls back to file position. It is already declared the workers sub-metric.

        expense_split is a signal-counted probe like `jpeg`, which has a +/-2 noise floor, so it is
        deliberately NOT primary: it can only reorder WITHIN a capability tier, never set one, and
        F backstops it.
        """
        # v3: rank on the standard-suite view the batteries were measured under (the operating arm)
        st = m.get("standard_operating") or m.get("standard")
        if not st:
            return None
        tier = st["reasoning"] + st["instr"] + (1 if st["tool"] else 0)
        return (tier,
                st.get("expense_split") or 0,
                (m.get("consistency") or {}).get("composite") or 0.0)

    def coder_key(m):
        """Coders rank on E-hard first, composite as the tiebreak.

        The full composite stopped ordering anything useful: E-core is spent (E1 mean 0.965,
        E9 0.989, both a wall of 1.00 down the table) while E-hard spans 1.000 -> 0.000. Sorting
        on the composite put `qwen2.5:7b` (E-hard 0.300) above `ornith:9b` (0.519) and
        `gpt-oss:120b-cloud` (E-hard 0.167, every core test 1.00) at #12. Since 2026-08-22 E-hard
        also gates the `coder` tag, so the gate and the ranking now read the same signal — the
        Battery A / routers fix applied one layer up. A model with no E-hard data (pre-2026-07-02
        row) sorts below every model that has some rather than dropping out of the list.
        """
        c = m.get("coding") or {}
        comp = c.get("composite")
        if comp is None:
            return None
        hard = (c.get("category_means") or {}).get("E-hard")
        return (0, 0.0, comp) if hard is None else (1, hard, comp)

    def vis_ocr(m):
        return (m.get("vision") or {}).get("dimensions", {}).get("ocr")

    def router_key(m):
        """Routers rank on the job they actually do: A1 classification accuracy first, A4
        false-escalation as the tiebreak penalty, decode tok/s last.

        Until 2026-08-21 this list was sorted by tok/s ALONE, with no quality input — which ranked
        `minicpm-v4.6:1b` (184.7 t/s, classifies 4/10) above `ministral-3:3b` (80.8 t/s, 10/10). Speed
        is a FLOOR for a router (the role gate already enforces >=80 t/s); it is not the ranking. A
        router with no Battery A data sorts below every router that has some, rather than dropping out
        of the list entirely."""
        r = m.get("routing") or {}
        acc, tps = r.get("classify_accuracy"), (m.get("tps") or 0.0)
        if acc is None:
            return (0, 0.0, 0.0, tps)
        return (1, acc, -(r.get("false_escalation_rate") or 0.0), tps)

    def elastic_key(m):
        """Rank on the model's BEST measured arm (user decision 2026-09-26, option B) — see _elastic_arm_key.
        The chosen arm is published as prompt_elasticity.best_arm."""
        pe = m.get("prompt_elasticity") or {}
        return _elastic_arm_key(pe["best_arm"]) if pe.get("best_arm") else None

    rankings = {
        "routers": ranked(routers, router_key),
        "workers": ranked(workers, worker_quality),
        "coders": ranked(completion, coder_key),
        # honesty: rank on the balanced (harmonic) axis, never confab_score — see the sub-block note.
        # Tiebreak on real_clean so that among equals the more discerning model sorts first.
        "honesty": ranked(completion, lambda m: ((h["honesty_balanced"], h.get("real_clean_rate") or 0.0)
                                                 if (h := (m.get("honesty") or {})).get("honesty_balanced")
                                                 is not None else None)),
        "vision": ranked(has_vis, lambda m: (m.get("vision") or {}).get("composite")),
        # fast-OCR is speed-ranked → require a real tps (excludes cloud / un-timed models)
        "vision_fast_ocr": ranked(has_vis, lambda m: ((vis_ocr(m), m["tps"])
                                                      if vis_ocr(m) is not None and m.get("tps") else None)),
        "embedding_short": ranked(has_emb, lambda m: (m.get("embedding") or {}).get("composite")),
        "embedding_long": ranked(has_emb, lambda m: (m.get("embedding") or {}).get("composite_long")),
        "long_context": ranked(has_lctx, lctx_key),
        "prompt_elasticity": ranked(completion, elastic_key),
    }

    # Run-provenance fingerprint of the most recent run (ollama_version / benchllama_commit /
    # model_digests / dataset hashes / os+hardware) — lets a consumer attribute a score delta
    # to runtime, harness, weights, or test-set. Empty {} until a run records one.
    environment = {}
    try:
        import results_db
        environment = results_db.latest_env()
    except Exception:
        pass

    return {
        "schema": SCHEMA,
        "generated": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": "BenchLLAMA",
        "host_profile": HOST_PROFILE,
        "protocol": PROTOCOL,
        "environment": environment,
        "result_files": sources,
        "models": models,
        "rankings": rankings,
    }


def main():
    OUT.parent.mkdir(exist_ok=True)
    data = build()
    payload = json.dumps(data, indent=2)
    OUT.write_text(payload)                                   # in-repo copy (provenance)
    PUBLISH.parent.mkdir(parents=True, exist_ok=True)
    PUBLISH.write_text(payload)                               # published source of truth
    print(f"→ wrote {OUT}  ({len(data['models'])} models)")
    print(f"→ published {PUBLISH}")
    print(f"  sources: {data['result_files']}")
    if "--print" in sys.argv:
        for cat, lst in data["rankings"].items():
            print(f"  {cat:16} {lst}")


if __name__ == "__main__":
    main()
