#!/usr/bin/env python3
"""
BenchLLAMA — Battery F build gate (σ + detection-reliability).

Runs Battery F N times per model and reports the two things that decide whether F is
viable BEFORE any weight tuning:
  • composite σ across runs — is a live multi-turn rollout reproducible enough? (The
    within-run-relative core predicts small σ; this confirms or refutes it.)
  • T1 stance-detection hit-rate — F2-flip and F4 ride on it; if it misses often, the
    "objective core" is objective in name only.

Informal (no cool-down between gate passes — we're measuring variance, not tok/s).

  python3 suites/consistency/gate.py                       # default model set, 3 runs
  python3 suites/consistency/gate.py --runs 3 --models a b
"""

import sys
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import aptitude as A  # noqa: E402

DEFAULT_MODELS = ["gemma4:12b", "granite4.1:3b", "qwen3.5:4b-mlx", "gemma4:latest"]


def _arg(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
            return sys.argv[i + 1]
    return default


def main():
    runs = int(_arg("--runs", "3"))
    if "--models" in sys.argv:
        i = sys.argv.index("--models")
        models = [a for a in sys.argv[i + 1:] if not a.startswith("--")]
    else:
        models = DEFAULT_MODELS

    print(f"Battery F gate — {len(models)} models × {runs} runs (informal, no cool-down)\n")
    rows = []
    for m in models:
        comps, stance_hits, per_dim = [], 0, {k: [] for k in A.F_WEIGHTS}
        for k in range(runs):
            print(f"\n### {m} — run {k+1}/{runs}", flush=True)
            try:
                r = A.run_battery_f(m)
            except Exception as e:
                print(f"   ✗ {m} run {k+1} failed: {e}", flush=True)
                continue
            s = r["summary"]
            comps.append(s["composite"])
            stance_hits += 1 if s["stance_detected"] else 0
            for kk, v in s["dims"].items():
                per_dim[kk].append(v)
            A.unload(m)
        if not comps:
            rows.append((m, None, None, 0, runs)); continue
        sigma = round(statistics.pstdev(comps), 3) if len(comps) > 1 else 0.0
        rows.append((m, round(sum(comps) / len(comps), 3), sigma, stance_hits, runs))

    print(f"\n\n{'='*64}\nGATE RESULTS\n{'='*64}")
    print(f"{'model':<22}{'mean':>7}{'σ':>7}{'stance hit':>12}")
    for m, mean, sigma, hits, n in rows:
        mean_s = f"{mean:.3f}" if mean is not None else "ERR"
        sig_s = f"{sigma:.3f}" if sigma is not None else "—"
        print(f"{m:<22}{mean_s:>7}{sig_s:>7}{f'{hits}/{n}':>12}")
    sigs = [s for _, _, s, _, _ in rows if s is not None]
    hitrate = sum(h for _, _, _, h, _ in rows) / max(1, sum(n for *_, n in rows))
    print(f"\nmax σ = {max(sigs) if sigs else 'n/a'}   |   T1 stance-detection hit-rate = {hitrate:.0%}")
    print("Verdict guide: σ ≲ 0.05 across models → ship live + tune weights. "
          "σ large → §4 pinned fallback. Low stance hit-rate → fix detector before trusting F2/F4.")


if __name__ == "__main__":
    main()
