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
PREFIX = "aptitude_e" if BAT == "E" else "aptitude_f"
FAST = (BAT == "E")                       # E skips cool-down; F keeps it
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
        gen, dbg = cm.get("E1", 0.0), cm.get("E2", 0.0)
        spread, cat_std = {}, {}
        for c in cm:
            pr = [rec["summary"]["category_means"].get(c) for rec in recs
                  if c in rec["summary"].get("category_means", {})]
            spread[c] = round(max(pr) - min(pr), 3) if len(pr) > 1 else 0.0
            cat_std[c] = round(statistics.pstdev(pr), 3) if len(pr) > 1 else 0.0
        comp_runs = [rec["summary"]["composite"] for rec in recs]
        out.append({"model": name, "battery": "E", "runs": len(recs), "tests": tests, "summary": {
            "category_means": cm, "composite": comp, "generate_basic": gen, "debug_fix": dbg,
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


def average(run_files):
    return _average_e(run_files) if BAT == "E" else _average_f(run_files)


def write_md(rows, path, k):
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
    write_md(rows, RESULTS / f"{PREFIX}_{d}.md", k)
    if BAT == "E":
        _G["apply_coder_overlay"](rows, REPO / "models.json")
    print(f"\n→ canonical averaged result: {canon.name}  ({len(rows)} models, {k} runs)")
    print("  next: python3 export.py")


if __name__ == "__main__":
    main()
