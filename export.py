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

REPO = Path(__file__).parent
RESULTS = REPO / "results"
OUT = REPO / "rankings" / "rankings.json"
# Well-known consumer location — the published source of truth every consumer
# (OllamaMCP, LookingGlass, MemoryCentral, …) reads from. Overwritten each run;
# the date lives inside the data (`generated`). Decoupled from this repo's layout.
PUBLISH = Path.home() / ".config" / "ollama-consumers" / "benchllama-rankings.json"

SCHEMA = 1
HOST_PROFILE = "M1 Max 32GB"
PROTOCOL = {"num_ctx": 16384, "think": False}


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
    "benchmark": "standard", "aptitude_e": "E", "aptitude_f": "F",
    "aptitude_f_elastic": "F-elastic", "vision": "vision", "embedding": "embedding", "longctx": "G",
    "confab": "confab",
}


def _load(prefix):
    """Latest per-model result for a battery. Prefers the SQLite store (results_db.latest = each
    model's most-recent result across all runs, so a partial/midnight re-run never drops models);
    falls back to the dated JSON file if the DB is empty/unavailable (transition safety)."""
    bat = _PREFIX_BATTERY.get(prefix)
    if bat:
        try:
            import results_db
            data = results_db.latest(bat)
            if data:
                return data, f"db:{bat}"
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
    jpeg = t.get("jpeg_formats", {}).get("signals")
    expense = (t.get("expense_split") or {}).get("check_detail", {}).get("score")
    return {"reasoning": reasoning, "instr": instr, "tool": ok("calculate"),
            **({"jpeg": jpeg} if jpeg is not None else {}),
            **({"expense_split": expense} if expense is not None else {})}


def build():
    registry = json.load((REPO / "models.json").open())
    std, std_f = _load("benchmark")
    coding, cod_f = _load("aptitude_e")
    cons, cons_f = _load("aptitude_f")
    elastic, ela_f = _load("aptitude_f_elastic")
    vision, vis_f = _load("vision")
    emb, emb_f = _load("embedding")
    lctx, lctx_f = _load("longctx")
    honesty, hon_f = _load("confab")

    models, sources = [], {k: v for k, v in {
        "standard": std_f, "coding": cod_f, "consistency": cons_f,
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
            m["ram_gb"] = None if is_cloud else s.get("ram_gb")
            m["standard"] = _standard_summary(s)
        c = coding.get(name)
        if c and c.get("summary"):
            cs = c["summary"]
            comp = cs.get("composite")
            disk = m.get("disk_gb")
            m["coding"] = {
                "composite": comp,
                "category_means": cs.get("category_means", {}),
                "coder_eligible": cs.get("coder_eligible"),
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
            pes = pe["summary"]; rc = pes.get("cutoffs", {})
            # prompt-elasticity is a per-model SUB-BLOCK (like `consistency`), NOT a ranking list:
            # the verdict is categorical and prompt-σ is only meaningful PAIRED with adherence, so
            # there's nothing to sort. Emitted only when F-elastic has been run (opt-in battery).
            # cutoffs trimmed to the DECLARED numeric thresholds (prose rationale lives in
            # suites/elasticity/ladder.json) so a consumer can re-threshold against its own scope.
            m["prompt_elasticity"] = {
                "verdict": pes.get("verdict"),
                "prompt_sigma": pes.get("prompt_sigma"),
                "instruction_adherence": pes.get("instruction_adherence"),
                "length_adherence": pes.get("length_adherence"),
                "cutoffs": {k: rc.get(k) for k in ("sigma_hi", "adherence_hi", "adherence_lo", "keyed_on")},
                "verdict_stable": pes.get("verdict_stable"),
                "prompt_sigma_stdev": pes.get("prompt_sigma_stdev"),
                "instruction_adherence_stdev": pes.get("instruction_adherence_stdev"),
                "per_rung": [{"rung": r["rung"], "constraints_n": r["constraints_n"],
                              "composite": r["composite"], "run_sigma": r.get("run_sigma"),
                              "instruction_adherence": r["instruction_adherence"],
                              "length_adherence": r["length_adherence"]}
                             for r in pes.get("per_rung", [])],
                "runs": pes.get("n_runs", 1),
            }
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
            m["long_context"] = {
                "composite": gs.get("composite"),
                "clean_depth": gs.get("clean_depth"),
                "prefill_collapse": gs.get("prefill_collapse"),
                "accuracy_by_depth": gs.get("accuracy_by_depth", {}),
                "prefill_by_depth": gs.get("prefill_by_depth", {}),
                "position_recall": gs.get("position_recall", {}),
                "n_depths": gs.get("n_depths"),
            }
        h = honesty.get(name)
        if h and h.get("summary"):
            hs = h["summary"]
            # honesty (Battery H) is a per-model SUB-BLOCK (like consistency / prompt_elasticity), NOT a
            # ranking list: BenchLLAMA emits the measured numbers, the consumer applies its own policy.
            # ⚠ Read fake_clean_rate and real_clean_rate TOGETHER — high+high = discerning-honest;
            # high-fake + LOW-real = pathological denier (aces fakes by refusing everything); low-fake =
            # confabulator. Deliberately NOT folded into any quality composite (honesty is orthogonal).
            m["honesty"] = {
                "confab_score": hs.get("composite"),
                "fabrication_rate": hs.get("fabrication_rate"),
                "fake_clean_rate": hs.get("fake_clean_rate"),
                "real_clean_rate": hs.get("real_clean_rate"),
                "n_items": hs.get("n_items"),
                "judge": hs.get("judge", []),
                "by_category": hs.get("by_category", {}),
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
        st = m.get("standard")
        if not st:
            return None
        return st["reasoning"] + st["instr"] + (1 if st["tool"] else 0)

    def vis_ocr(m):
        return (m.get("vision") or {}).get("dimensions", {}).get("ocr")

    rankings = {
        "routers": ranked(routers, lambda m: m.get("tps")),
        "workers": ranked(workers, worker_quality),
        "coders": ranked(completion, lambda m: (m.get("coding") or {}).get("composite")),
        "vision": ranked(has_vis, lambda m: (m.get("vision") or {}).get("composite")),
        # fast-OCR is speed-ranked → require a real tps (excludes cloud / un-timed models)
        "vision_fast_ocr": ranked(has_vis, lambda m: ((vis_ocr(m), m["tps"])
                                                      if vis_ocr(m) is not None and m.get("tps") else None)),
        "embedding_short": ranked(has_emb, lambda m: (m.get("embedding") or {}).get("composite")),
        "embedding_long": ranked(has_emb, lambda m: (m.get("embedding") or {}).get("composite_long")),
        "long_context": ranked(has_lctx, lctx_key),
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
