#!/usr/bin/env python3
"""
BenchLLAMA — content-addressed resume (the ONE resume policy; docs/resume-spec.md).

Replaces every time/date-window resume ("was it scored in the last 24/72h?" — a bad proxy) with a
determinant diff: **re-run a (model, battery) result iff something that changes the result changed.**
Otherwise the stored score IS the current score → skip + carry forward (via the DB; export reads latest()).

Triggers (run iff ANY fired):
  1. weights   — the model's /api/tags digest differs from when it was scored          (HARD, always on)
  2. test      — a battery-relevant dataset hash OR the battery REVISION changed        (HARD, always on)
  3. runtime   — Ollama major.minor bumped (patch ignored: 0.31.1→0.31.2 = no)         (OPT-IN via --check-runtime)

Runtime is OFF by default: most Ollama releases are bugfixes, so re-running the whole fleet (+ cloud
cost) on every version bump is the wrong default. Pass `--check-runtime` when a release note flags a real
perf change (e.g. a Gemma4/MLX speedup) — it then re-runs any model whose scoring ran under a different
Ollama major.minor.

Diff source = provenance we already store: env_fingerprint() at run start → runs.env, joined per model
via results_db.latest_env_by_model(battery). No new instrumentation.

Policies (see spec): cloud models (no local digest) → skip UNLESS test-changed or --force (protects
quota); pre-provenance / unknown / never-scored → re-run (conservative). --force + --models override all;
--check-runtime opts into trigger 3.

Every battery script calls `targets(...)`; `should_run(...)` is the single-model primitive. `--resume-report`
uses `targets()` in dry mode to print the per-model split + reason.
"""

from bench_utils import BATTERY_REVISION, BATTERY_DATASETS


def runtime_material(old: str | None, new: str | None) -> bool:
    """True if the Ollama version bump is material (major.minor changed). Patch is ignored.
    Unknown on either side → True (can't prove compatibility → re-run)."""
    if not old or not new:
        return True
    try:
        om = tuple(int(x) for x in str(old).split(".")[:2])
        nm = tuple(int(x) for x in str(new).split(".")[:2])
    except Exception:
        return old != new
    return om != nm


def _test_changed(battery: str, prev_env: dict, cur_env: dict) -> str | None:
    """Reason string if the battery's TEST identity changed (dataset hash or REVISION), else None."""
    pd, cd = prev_env.get("datasets") or {}, cur_env.get("datasets") or {}
    for key in BATTERY_DATASETS.get(battery, []):
        if key in cd and key in pd and cd[key] != pd[key]:
            return f"test-data-changed ({key})"
        if key in cd and key not in pd:            # newly-hashed input the prior run didn't have
            return f"test-data-changed ({key})"
    # Revision triggers ONLY when both sides are known and differ. A prior run that predates
    # battery_revisions tracking (prev_rev None) is NOT treated as a change: those results already
    # reflect the current scoring code, and any test-DATA change is still caught by the hash check
    # above. Going forward every run records revisions, so future code-rev bumps are caught cleanly.
    cur_rev = BATTERY_REVISION.get(battery)
    prev_rev = (prev_env.get("battery_revisions") or {}).get(battery)
    if cur_rev is not None and prev_rev is not None and cur_rev != prev_rev:
        return f"battery-rev {prev_rev}→{cur_rev}"
    return None


def _think_point_changed(model: str, prev_env: dict, cur_env: dict, arm: str) -> str | None:
    """v3 (docs/think-spec.md): the lever a model is measured at is part of the test's identity for
    that arm. direct arm → its direct_lever; think arm → operating_lever; auto (batteries) → either."""
    keys = {"direct": ("direct_lever",), "think": ("operating_lever",),
            "auto": ("operating_lever", "direct_lever")}.get(arm, ())
    cur = (cur_env.get("think_profiles") or {}).get(model) or {}
    prev = (prev_env.get("think_profiles") or {}).get(model) or {}
    for k in keys:
        if cur.get(k) != prev.get(k):
            return f"think-point {k} {prev.get(k) or '—'}→{cur.get(k) or '—'}"
    return None


def should_run(battery: str, model: str, cur_env: dict, prev_env: dict | None, *,
               is_cloud: bool = False, scored: bool = True,
               force: bool = False, check_runtime: bool = False, arm: str = "direct") -> tuple[bool, str]:
    """(run?, reason) for one (model, battery). `prev_env` = runs.env of the run that last scored this
    model for this battery (None/{} if none). `scored` = does the model have ANY prior result here.
    `arm` = direct | think | auto — which think arm this invocation measures (v3)."""
    if force:
        return True, "forced"
    if not scored:
        return True, "new-model"
    if not prev_env:
        return True, "no-provenance"

    # 1. weights (local only — cloud has no /api/tags digest; provider updates silently)
    if not is_cloud:
        cur_dig = (cur_env.get("model_digests") or {}).get(model)
        prev_dig = (prev_env.get("model_digests") or {}).get(model)
        if cur_dig and prev_dig and cur_dig != prev_dig:
            return True, "weights-changed"
        if cur_dig and not prev_dig:
            return True, "weights (no prior digest)"

    # 2. test
    tc = _test_changed(battery, prev_env, cur_env)
    if tc:
        return True, tc

    # 3. think arm (v3) — the model's lever for this arm moved (probe re-ran, or it gained a think arm)
    tp = _think_point_changed(model, prev_env, cur_env, arm)
    if tp:
        return True, tp

    # runtime — OPT-IN only (--check-runtime); major.minor gated (patch ignored)
    if check_runtime and runtime_material(prev_env.get("ollama_version"), cur_env.get("ollama_version")):
        return True, f"ollama {prev_env.get('ollama_version')}→{cur_env.get('ollama_version')}"

    return False, "up-to-date"


def targets(battery: str, universe: list[str], cur_env: dict, *,
            scored_env: dict | None = None, scored_models: set | None = None,
            cloud: set | None = None, force: bool = False,
            explicit_models: list[str] | None = None,
            check_runtime: bool = False, arm: str = "direct") -> tuple[list, list, dict]:
    """Split `universe` into (to_run, skipped, reasons) for a battery.

      scored_env     = {model: prev_env}  from results_db.latest_env_by_model(battery, arm=…)
      scored_models  = set of models with ANY prior result here (results_db.latest(battery, arm=…).keys())
      cloud          = set of cloud model names (digest trigger suppressed)
      explicit_models= --models: run exactly these, ignore resume
      arm            = direct | think | auto (v3) — see should_run
    """
    if explicit_models:
        return list(explicit_models), [], {m: "explicit" for m in explicit_models}
    scored_env = scored_env or {}
    scored_models = scored_models if scored_models is not None else set(scored_env)
    cloud = cloud or set()
    to_run, skipped, reasons = [], [], {}
    for m in universe:
        run, why = should_run(battery, m, cur_env, scored_env.get(m),
                              is_cloud=(m in cloud), scored=(m in scored_models),
                              force=force, check_runtime=check_runtime, arm=arm)
        reasons[m] = why
        (to_run if run else skipped).append(m)
    return to_run, skipped, reasons


# Batteries whose prior result is looked up at each model's OPERATING-POINT arm, not newest-of-any-arm.
# 2026-09-26: a deliberate direct-arm F-elastic measurement became the newest row and was then carried
# forward as if it were the operating point. Mirrors export._load_operating (F-elastic only for now —
# user decision 09-26; the other batteries hold one arm per model in practice).
OPERATING_ARM_BATTERIES = {"F-elastic"}


def _operating_arm_prior(battery: str, universe: list[str]) -> tuple[dict, set]:
    """(scored_env, scored_models) from each model's row at bench_utils.resolve_arm(model, "auto").
    A model with no row at its operating arm counts as unscored → it runs ("new-model")."""
    import results_db
    from bench_utils import resolve_arm
    env = {a: results_db.latest_env_by_model(battery, arm=a) for a in ("direct", "think")}
    have = {a: set(results_db.latest(battery, arm=a)) for a in ("direct", "think")}
    scored_env, scored = {}, set()
    for m in universe:
        a = resolve_arm(m, "auto")
        if m in have[a]:
            scored.add(m)
            scored_env[m] = env[a].get(m) or {}
    return scored_env, scored


def resolve(battery: str, universe: list[str], *, host: str = "http://localhost:11434",
            cur_env: dict | None = None, cloud: set | None = None, force: bool = False,
            explicit_models: list[str] | None = None, check_runtime: bool = False,
            arm: str = "direct") -> tuple[list, list, dict]:
    """Convenience wrapper: pulls the per-model prior env + scored set from the DB and computes targets.
    `cur_env` may be passed to avoid re-fingerprinting (the orchestrator captures one per run).
    `arm` (v3): 'direct'/'think' read only that arm's rows (the standard suite holds both); 'auto'
    reads whichever arm the battery last stored per model."""
    from bench_utils import env_fingerprint
    import results_db
    cur_env = cur_env or env_fingerprint(host=host, models=universe)
    db_arm = None if arm == "auto" else arm
    if arm == "auto" and battery in OPERATING_ARM_BATTERIES:
        scored_env, scored_models = _operating_arm_prior(battery, universe)
    else:
        scored_env = results_db.latest_env_by_model(battery, arm=db_arm)
        scored_models = set(results_db.latest(battery, arm=db_arm).keys())
    return targets(battery, universe, cur_env, scored_env=scored_env, scored_models=scored_models,
                   cloud=cloud or set(), force=force, explicit_models=explicit_models,
                   check_runtime=check_runtime, arm=arm)


def plan_single_pass(battery: str, eligible: list[str], *, host: str = "http://localhost:11434",
                     cur_env: dict | None = None, cloud: set | None = None, force: bool = False,
                     explicit_models: list[str] | None = None, check_runtime: bool = False,
                     arm: str = "direct"):
    """For the single-pass batteries (standard/ladder/A–D/G/V/EMB). `eligible` = ALL registry models this
    battery runs (NOT filtered by --models). Returns (run_names, carry_forward, reasons):
      run_names     — models to actually benchmark this invocation
      carry_forward — prior full result dicts (from the DB, lossless) for every eligible model NOT being
                      run, so the output JSON/MD stays complete without a time-windowed file merge
      reasons       — {model: why} for the report
    The caller seeds `all_results = list(carry_forward)`, then runs `run_names` and appends."""
    import results_db
    run_names, skipped, reasons = resolve(battery, eligible, host=host, cur_env=cur_env, cloud=cloud or set(),
                                          force=force, explicit_models=explicit_models,
                                          check_runtime=check_runtime, arm=arm)
    # --force = "fresh overwrite" → NO carry-forward (matches legacy --force; also keeps the averager's
    # per-pass output — it calls with --force --models — to exactly its targets, not the whole fleet).
    if force:
        return run_names, [], reasons
    db_prev = results_db.latest(battery, arm=(None if arm == "auto" else arm))
    run_set = set(run_names)
    carry = [db_prev[m] for m in eligible if m in db_prev and m not in run_set]
    return run_names, carry, reasons


def format_report(battery: str, to_run: list, skipped: list, reasons: dict) -> str:
    """Human-readable dry-run report for --resume-report."""
    L = [f"Battery {battery}: {len(to_run)} to run, {len(skipped)} skipped (carried forward)"]
    for m in to_run:
        L.append(f"  ▶  {m:40} {reasons.get(m, '')}")
    for m in skipped:
        L.append(f"  ↷  {m:40} {reasons.get(m, '')}")
    return "\n".join(L)


# ── Dry report CLI: `python3 resume.py [--battery X] [--check-runtime]` ──────────────
# Shows, per battery, exactly which (model) would re-run and WHY — the transparency that makes a
# content-addressed system trustworthy. Runs NO benchmark. Also wired as `bench.sh <cmd> --resume-report`.

_UNIVERSE = {  # battery → (registry capability/role filter)
    "standard": ("role", {"worker", "router"}), "ladder": ("role", {"worker", "router"}),
    "A": ("role", {"router"}), "B": ("role", {"worker"}), "C": ("role", {"worker"}), "D": ("role", {"worker"}),
    "E": ("cap", "completion"), "F": ("cap", "completion"), "F-elastic": ("cap", "completion"),
    "G": ("role", {"worker", "router"}), "confab": ("cap", "completion"),
    "vision": ("cap", "vision"), "embedding": ("cap", "embedding"), "image": ("cap", "image"),
}


def _eligible(battery, reg):
    kind, val = _UNIVERSE.get(battery, ("role", {"worker", "router"}))
    if kind == "role":
        names = [m["name"] for m in reg if m.get("role") in val]
    else:
        names = [m["name"] for m in reg if val in (m.get("capabilities") or [])]
    # C/D run with --capable-only in the pipeline → mirror that here (else the report over-lists the
    # tool-less models the real run excludes). Capable = passed `calculate` in the latest standard result.
    if battery in ("C", "D"):
        try:
            import results_db
            std = results_db.latest("standard")
            capable = {m for m, r in std.items()
                       if ((r.get("tests") or {}).get("calculate") or {}).get("correct")}
            if capable:
                names = [n for n in names if n in capable]
        except Exception:
            pass
    return names


# The arm each battery's RUNNER actually resolves with — the report must mirror the runner or it
# lies. aptitude.py:61 uses `auto` for B/C/D/E/F/F-elastic and `direct` only for A;
# average_e_runs.py passes arm="auto". Reporting those at "direct" reads think-arm-only rows as
# absent and prints `new-model` for a model measured hours earlier (seen 2026-09-01 on
# granite4.2:3b and ornith-1.5:9b, both of which had fresh think-arm E rows).
# MUST track the runners. Source of truth, per runner's `arm=` into resume:
#   aptitude.py:61   ARM_REQ = "direct" if battery=="A" else requested_arm(default="auto")
#   average_e_runs   arm="auto"                    longctx.py:71   requested_arm(default="auto")
#   confab.py        requested_arm(default="auto") runner.py       requested_arm(default="direct")
#   ctx_ladder / vision / embedding — no arm, so the "direct" default below is correct.
# Anything absent here is reported at "direct"; add a battery the moment its runner moves to auto.
_REPORT_ARM = {"B": "auto", "C": "auto", "D": "auto", "E": "auto", "F": "auto",
               "F-elastic": "auto", "G": "auto", "confab": "auto"}


def report(batteries=None, host="http://localhost:11434", check_runtime=False) -> str:
    import json as _json
    from pathlib import Path as _Path
    from bench_utils import env_fingerprint
    reg = _json.loads((_Path(__file__).parent / "models.json").read_text())
    cloud = {m["name"] for m in reg if m.get("cloud")}
    order = ["standard", "ladder", "A", "B", "C", "D", "E", "F", "F-elastic", "G", "confab", "vision", "embedding", "image"]
    bats = batteries or order
    cur = env_fingerprint(host=host)
    out = [f"Resume report (ollama {cur.get('ollama_version')} · benchllama {cur.get('benchllama_commit')}"
           + ("  · --check-runtime" if check_runtime else "") + ")\n"]
    for b in bats:
        elig = _eligible(b, reg)
        if not elig:
            continue
        tr, sk, why = resolve(b, elig, cur_env=cur, cloud=cloud, check_runtime=check_runtime,
                              arm=_REPORT_ARM.get(b, "direct"))
        out.append(format_report(b, tr, sk, why))
    return "\n".join(out)


if __name__ == "__main__":
    import sys as _sys
    bats = None
    if "--battery" in _sys.argv:
        i = _sys.argv.index("--battery")
        bats = [_sys.argv[i + 1]] if i + 1 < len(_sys.argv) else None
    print(report(batteries=bats, check_runtime="--check-runtime" in _sys.argv))
