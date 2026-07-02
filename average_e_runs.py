#!/usr/bin/env python3
"""
BenchLLAMA — multi-run averaging for the noisy batteries (E coding, F consistency).

Some batteries are noisy run-to-run (E's generation-dependent categories under MLX temp=0;
F's live multi-turn rollout). This runs the single-pass primitive N times, averages at the
component level, and surfaces per-component VOLATILITY + composite σ so the noise is visible.

Per-battery differences:
  • E (coding): grades CORRECTNESS → passes run `--fast` (no cool-down); `coder` overlay applied
    once on the average. Averaged at per-test → per-category.
  • F (consistency): grades BEHAVIOR (voice is warm/cold sensitive) → passes keep cool-down;
    no overlay. Averaged at per-dimension (F1–F5).

  python3 average_e_runs.py --runs 3                  # Battery E (default)
  python3 average_e_runs.py --battery F --runs 3      # Battery F
  python3 average_e_runs.py --battery F --average-only

Outputs:  aptitude_<e|f>_<date>_run{k}.json (provenance) + aptitude_<e|f>_<date>.json/.md (canonical
averaged — export reads this). Then: python3 export.py
"""

import json
import sys
import subprocess
import glob
import os
import statistics
from pathlib import Path
from datetime import date

REPO = Path(__file__).parent
RESULTS = REPO / "results"


def _arg(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
            return sys.argv[i + 1]
    return default


def _models_arg():
    if "--models" in sys.argv:
        i = sys.argv.index("--models")
        return [a for a in sys.argv[i + 1:] if not a.startswith("--")]
    return []


# Reuse weights / thresholds / overlay from aptitude.py (no duplication).
_src = open(REPO / "aptitude.py").read().split("# ── Entrypoint")[0]
_G = {"__name__": "_avg", "__file__": str(REPO / "aptitude.py")}
exec(compile(_src, "aptitude.py", "exec"), _G)

BAT = _arg("--battery", "E").upper()
PREFIX = {"E": "aptitude_e", "F": "aptitude_f", "F-ELASTIC": "aptitude_f_elastic"}.get(BAT, "aptitude_f")
FAST = BAT in ("E", "F-ELASTIC")          # E + F-elastic grade CORRECTNESS (not tok/s) → no cool-down; F keeps it
WEIGHTS = _G["E_WEIGHTS"] if BAT == "E" else _G["F_WEIGHTS"]
CMIN, BAND, GMIN = _G["E_CODER_COMPOSITE_MIN"], _G["E_CODER_COMPOSITE_BAND"], _G["E_CODER_GENERATE_MIN"]


def run_passes(n, model_args):
    """Run the battery n times and tag each pass as <prefix>_<date>_run{k}.json."""
    tagged = []
    for k in range(1, n + 1):
        print(f"\n{'#' * 60}\n# Battery {BAT} — averaging pass {k}/{n}\n{'#' * 60}", flush=True)
        cmd = [sys.executable, str(REPO / "aptitude.py"), "--battery", BAT, "--force"]
        if FAST:
            cmd.append("--fast")
        if model_args:
            cmd += ["--models", *model_args]
        subprocess.run(cmd, check=True)

        d = date.today().isoformat()
        produced = RESULTS / f"{PREFIX}_{d}{'_fast' if FAST else ''}.json"
        if not produced.exists():          # date may have rolled mid-run
            cands = glob.glob(str(RESULTS / f"{PREFIX}_*{'_fast' if FAST else ''}.json"))
            cands = [c for c in cands if "_run" not in c]
            produced = Path(max(cands, key=os.path.getmtime))
        dst = RESULTS / f"{PREFIX}_{date.today().isoformat()}_run{k}.json"
        dst.write_text(produced.read_text())
        tagged.append(dst)
        print(f"  saved pass {k} → {dst.name}", flush=True)
    return tagged


def _today_run_files():
    return sorted(glob.glob(str(RESULTS / f"{PREFIX}_{date.today().isoformat()}_run*.json")))


def _load_runs(run_files):
    runs = []
    for f in run_files:
        runs.append({r["model"]: r for r in json.load(open(f))
                     if "error" not in r and "summary" in r})
    models = sorted(set().union(*[set(r) for r in runs])) if runs else []
    return runs, models


def _average_e(run_files):
    runs, models = _load_runs(run_files)
    out = []
    for name in models:
        recs = [r[name] for r in runs if name in r]
        test_ids = set().union(*[set(rec["tests"]) for rec in recs])
        tests = {}
        for pid in test_ids:
            scores = [rec["tests"][pid].get("score") for rec in recs
                      if pid in rec["tests"] and rec["tests"][pid].get("score") is not None]
            if not scores:
                continue
            cat = next(rec["tests"][pid].get("category") for rec in recs if pid in rec["tests"])
            tests[pid] = {"category": cat, "score": round(sum(scores) / len(scores), 4),
                          "scores": [round(s, 3) for s in scores], "n": len(scores)}
        cs = {}
        for t in tests.values():
            cs.setdefault(t["category"], []).append(t["score"])
        cm = {c: round(sum(v) / len(v), 4) for c, v in cs.items()}
        present = {c: w for c, w in WEIGHTS.items() if c in cm}
        comp = round(sum(cm[c] * w for c, w in present.items()) / (sum(present.values()) or 1), 4)
        # two-band: `comp` (incl E-hard) is BOTH the ranking composite AND the coder gate — as of
        # 2026-07-02 the E-hard band grew from 1 → 4 averaged tasks (variance smoothed), so failing
        # the hard tier now legitimately blocks the `coder` role instead of being excused. `comp_core`
        # (E-core only) is still emitted as a diagnostic (shows the E-hard delta), not the gate.
        core = {c: w for c, w in present.items() if c != "E-hard"}
        comp_core = round(sum(cm[c] * w for c, w in core.items()) / (sum(core.values()) or 1), 4)
        gen, dbg = cm.get("E1", 0.0), cm.get("E2", 0.0)
        spread, cat_std = {}, {}
        for c in cm:
            pr = [rec["summary"]["category_means"].get(c) for rec in recs
                  if c in rec["summary"].get("category_means", {})]
            spread[c] = round(max(pr) - min(pr), 3) if len(pr) > 1 else 0.0
            cat_std[c] = round(statistics.pstdev(pr), 3) if len(pr) > 1 else 0.0
        comp_runs = [rec["summary"]["composite"] for rec in recs]
        out.append({"model": name, "battery": "E", "runs": len(recs), "tests": tests, "summary": {
            "category_means": cm, "composite": comp, "composite_core": comp_core,
            "generate_basic": gen, "debug_fix": dbg,
            "coder_eligible": (comp >= CMIN and dbg > 0 and gen >= GMIN),
            "threshold": {"composite_min": CMIN, "composite_band": BAND, "generate_min": GMIN, "debug_fix_gt": 0},
            "n_runs": len(recs),
            "composite_stdev": round(statistics.pstdev(comp_runs), 3) if len(comp_runs) > 1 else 0.0,
            "composite_spread": round(max(comp_runs) - min(comp_runs), 3) if len(comp_runs) > 1 else 0.0,
            "category_spread": spread, "category_stdev": cat_std}})
    return out, len(runs)


def _average_f(run_files):
    """Average per-dimension (F1–F5) across runs; recompose composite + σ. No per-test scores."""
    runs, models = _load_runs(run_files)
    out = []
    for name in models:
        recs = [r[name] for r in runs if name in r]
        keys = list(WEIGHTS.keys())
        dims = {k: round(sum(rec["summary"]["dims"][k] for rec in recs) / len(recs), 4) for k in keys}
        present = {k: w for k, w in WEIGHTS.items() if k in dims}
        comp = round(sum(dims[k] * w for k, w in present.items()) / (sum(present.values()) or 1), 4)
        dim_spread = {k: round(max(rec["summary"]["dims"][k] for rec in recs)
                               - min(rec["summary"]["dims"][k] for rec in recs), 3) for k in keys}
        dim_std = {k: round(statistics.pstdev([rec["summary"]["dims"][k] for rec in recs]), 3)
                   if len(recs) > 1 else 0.0 for k in keys}
        comp_runs = [rec["summary"]["composite"] for rec in recs]
        stance_hits = sum(1 for rec in recs if rec["summary"].get("stance_detected"))
        out.append({"model": name, "battery": "F", "runs": len(recs), "summary": {
            "dims": dims, "composite": comp, "n_runs": len(recs),
            "composite_stdev": round(statistics.pstdev(comp_runs), 3) if len(comp_runs) > 1 else 0.0,
            "composite_spread": round(max(comp_runs) - min(comp_runs), 3) if len(comp_runs) > 1 else 0.0,
            "dim_spread": dim_spread, "dim_stdev": dim_std,
            "stance_detected_rate": round(stance_hits / len(recs), 3)}})
    return out, len(runs)


def _average_f_elastic(run_files):
    """Average per-rung (composite + per-constraint) across runs, recompose prompt_sigma +
    instruction/length adherence from the means (never mean-of-final), reclassify the verdict, and
    surface RUN-TO-RUN CONSISTENCY — the reason multipass matters here: borderline verdicts ride on
    run noise (e.g. instr 0.797 vs the 0.80 cutoff). per_rung.run_sigma = stdev of that rung's
    composite across runs; verdict_stable = did every run agree."""
    runs, models = _load_runs(run_files)
    elastic = _G["_elastic"]
    ladder  = elastic.load_ladder()
    cutoffs = ladder["verdict_cutoffs"]
    klass   = lambda c: ladder["constraints"][c].get("class")
    mean    = lambda xs: round(statistics.mean(xs), 4) if xs else None
    pstd    = lambda xs: round(statistics.pstdev(xs), 4) if len(xs) > 1 else 0.0
    out = []
    for name in models:
        recs = [r[name] for r in runs if name in r]
        rung_ids = [pr["rung"] for pr in recs[0]["summary"]["per_rung"]]
        per_rung, composites = [], []
        for rid in rung_ids:
            prs = [next(p for p in rec["summary"]["per_rung"] if p["rung"] == rid) for rec in recs]
            comp_runs = [p["composite"] for p in prs]
            cons = list(prs[0]["per_constraint"].keys())
            pc = {c: round(statistics.mean([p["per_constraint"][c] for p in prs]), 4) for c in cons}
            instr  = [v for c, v in pc.items() if klass(c) != "length"]
            length = [v for c, v in pc.items() if klass(c) == "length"]
            composites.append(round(statistics.mean(comp_runs), 4))
            per_rung.append({"rung": rid, "constraints_n": len(cons),
                             "composite": composites[-1], "run_sigma": pstd(comp_runs),
                             "instruction_adherence": mean(instr), "length_adherence": mean(length),
                             "per_constraint": pc})
        prompt_sigma = pstd(composites)
        iadh = mean([r["instruction_adherence"] for r in per_rung if r["instruction_adherence"] is not None])
        ladh = mean([r["length_adherence"] for r in per_rung if r["length_adherence"] is not None])
        verdict = elastic.classify(prompt_sigma, iadh, cutoffs)
        run_verdicts = [rec["summary"]["verdict"] for rec in recs]
        out.append({"model": name, "battery": "F-elastic", "runs": len(recs), "summary": {
            "prompt_sigma": prompt_sigma, "instruction_adherence": iadh, "length_adherence": ladh,
            "adherence": iadh, "verdict": verdict, "cutoffs": cutoffs, "per_rung": per_rung,
            "n_runs": len(recs),
            "prompt_sigma_stdev": pstd([rec["summary"]["prompt_sigma"] for rec in recs]),
            "instruction_adherence_stdev": pstd([rec["summary"]["instruction_adherence"] for rec in recs
                                                 if rec["summary"].get("instruction_adherence") is not None]),
            "per_run_verdicts": run_verdicts, "verdict_stable": len(set(run_verdicts)) == 1}})
    return out, len(runs)


def average(run_files):
    if BAT == "E":          return _average_e(run_files)
    if BAT == "F-ELASTIC":  return _average_f_elastic(run_files)
    return _average_f(run_files)


def write_md(rows, path, k):
    if BAT == "F-ELASTIC":
        order = {"robust": 0, "prompt-sensitive": 1, "prompt-deaf": 2}
        rows = sorted(rows, key=lambda r: (order.get(r["summary"]["verdict"], 9),
                                           -(r["summary"]["instruction_adherence"] or 0)))
        fn = lambda v: "—" if v is None else f"{v:.2f}"
        L = [f"# Aptitude Battery F-elastic — Prompt-Elasticity (averaged over {k} runs)", "",
             "prompt-σ + **instruction-adherence** co-equal; verdict keyed on instruction-adherence. "
             "`len-adh` = standalone verbosity meter (not a verdict input). **The σ(run) columns are "
             "the calibration payload** — run-to-run stdev of prompt-σ and instruction-adherence; "
             "`stable` = all runs returned the same verdict. Cutoffs PROVISIONAL.", "",
             "| # | Model | Verdict | stable | prompt-σ | σ(run) | instr-adh | σ(run) | len-adh |",
             "|---|-------|---------|:------:|----------|--------|-----------|--------|---------|"]
        for i, r in enumerate(rows, 1):
            s = r["summary"]
            L.append(f"| {i} | `{r['model']}` | **{s['verdict']}** | {'✓' if s['verdict_stable'] else '⚠'} | "
                     f"{s['prompt_sigma']:.3f} | {s.get('prompt_sigma_stdev',0):.3f} | "
                     f"{fn(s.get('instruction_adherence'))} | {s.get('instruction_adherence_stdev',0):.3f} | "
                     f"{fn(s.get('length_adherence'))} |")
        path.write_text("\n".join(L) + "\n")
        return
    rows = sorted(rows, key=lambda r: r["summary"]["composite"], reverse=True)
    if BAT == "E":
        L = [f"# Aptitude Battery E — Coding (averaged over {k} runs)", "",
             "Composite = per-test mean recomposed from per-category means (never mean-of-composites). "
             "**σ** = stdev of per-run composite (consistency).", "",
             "| # | Model | Composite | σ | E1 | E2 | E3 | E5 | E7 | E9 | coder |",
             "|---|-------|-----------|---|----|----|----|----|----|----|-------|"]
        for i, r in enumerate(rows, 1):
            s = r["summary"]; cm = s["category_means"]
            g = lambda c: f"{cm[c]:.2f}" if c in cm else "—"
            coder = "✓" if s["coder_eligible"] else ("~" if s["composite"] >= BAND else "·")
            L.append(f"| {i} | `{r['model']}` | **{s['composite']:.3f}** | {s.get('composite_stdev', 0):.3f} | "
                     f"{g('E1')} | {g('E2')} | {g('E3')} | {g('E5')} | {g('E7')} | {g('E9')} | {coder} |")
    else:
        L = [f"# Aptitude Battery F — Conversational Consistency (averaged over {k} runs)", "",
             "Composite = per-dimension mean recomposed across runs. **σ** = stdev of per-run "
             "composite. F1 persona · F2 pressure · F3 voice · F4 callback · F5 coherence. "
             "`stance` = T1-detection hit-rate (F2-flip/F4 footing).", "",
             "| # | Model | Composite | σ | F1 | F2 | F3 | F4 | F5 | stance |",
             "|---|-------|-----------|---|----|----|----|----|----|:------:|"]
        for i, r in enumerate(rows, 1):
            s = r["summary"]; d = s["dims"]
            L.append(f"| {i} | `{r['model']}` | **{s['composite']:.3f}** | {s.get('composite_stdev', 0):.3f} | "
                     f"{d['F1']:.2f} | {d['F2']:.2f} | {d['F3']:.2f} | {d['F4']:.2f} | {d['F5']:.2f} | "
                     f"{s.get('stance_detected_rate', 0):.0%} |")
    path.write_text("\n".join(L) + "\n")


def main():
    if "--average-only" in sys.argv:
        files = _today_run_files()
        if not files:
            sys.exit(f"no {PREFIX}_<today>_run*.json files to average — run without --average-only first")
    else:
        files = run_passes(int(_arg("--runs", "3")), _models_arg())

    print(f"\nAveraging {len(files)} run(s): {[Path(f).name for f in files]}", flush=True)
    rows, k = average(files)
    d = date.today().isoformat()
    canon = RESULTS / f"{PREFIX}_{d}.json"
    canon.write_text(json.dumps(rows, indent=2))
    try:
        import results_db
        results_db.record_all("F-elastic" if BAT == "F-ELASTIC" else BAT, rows)
    except Exception:
        pass
    write_md(rows, RESULTS / f"{PREFIX}_{d}.md", k)
    if BAT == "E":
        _G["apply_coder_overlay"](rows, REPO / "models.json")
    print(f"\n→ canonical averaged result: {canon.name}  ({len(rows)} models, {k} runs)")
    print("  next: python3 export.py")


if __name__ == "__main__":
    main()
