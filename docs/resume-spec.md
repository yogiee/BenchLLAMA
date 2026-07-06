# Content-Addressed Resume — SPEC

**Status:** designed 2026-07-06; **IMPLEMENTED 2026-07-06** (`resume.py` + all six callers converted).
Replaces all time/date-window resume with a single provenance-driven policy. The `average_e_runs`
DB-based resume-skip (d78b14f) is now a caller of the shared helper.

**Decisions taken (were "open" below):** runtime = **OPT-IN via `--check-runtime`** (major.minor gated,
patch ignored) — NOT a default trigger (most Ollama releases are bugfixes; re-running the fleet + cloud on
every bump is wrong; opt in when a release note flags a real perf change like the Gemma4/MLX speedup);
cloud = skip-unless-test/force; pre-provenance = re-run (conservative); REVISION = per-battery int in
`bench_utils.BATTERY_REVISION` → `runs.env.battery_revisions`. One refinement made during build: a prior
run that predates `battery_revisions` tracking does NOT trigger a revision re-run (its dataset hashes still
catch test-data changes, and its results already reflect current scoring) — avoids a one-time full-fleet
re-run on deploy. Dry report: `./bench.sh --resume-report` (or `python3 resume.py [--battery X]
[--check-runtime]`). **Default triggers are just weights + test;** runtime is the opt-in third.

## Problem

Resume is currently reimplemented **six different ways** and they've drifted:

| Script | Current resume |
|--------|----------------|
| `runner.py` (standard) | 72h file window + carry-forward |
| `ctx_ladder.py` | 72h file window |
| `aptitude.py` (A–D) | 24h file window |
| `average_e_runs.py` (E/F/F-elastic) | **was** hardcoded `--force` (whole fleet ×3); now DB-based skip (d78b14f) |
| `longctx.py` (G) | 24h file window |
| `vision.py` / `embedding.py` | 24h file window |

Time windows answer the wrong question. "Scored in the last 24h?" is a bad proxy for "would it score
differently now?" — it re-runs things that didn't change (wasting hours + **cloud API quota**) and would
skip things that did. This directly caused the 2026-07-05→06 mess: standard/ctx resumed, but E/F re-ran the
fleet (hardcoded force) and G re-ran the fleet (24h window expired after 2 days) → ~10h wasted + cloud
usage nearly capped. Root cause = **no single resume policy**.

## Principle

**Re-run a (model, battery) result iff a determinant of that result changed since it was last scored.**
Otherwise the old score IS the current score (modulo run-noise, which averaging/σ already handle) → skip
and carry it forward. This is content-addressed, not time-addressed.

## The triggers (skip iff NONE fired)

Per `(model, battery)`, compare the CURRENT environment to the one stored with the model's latest DB result:

1. **Model weights changed** — `/api/tags` digest differs. **Hard trigger, always on.** The biggest signal.
2. **Test/harness changed** — the battery's inputs changed. **Hard trigger, always on.**
   - **test data** — the battery's dataset/prompt sha256 (already in provenance `datasets`) differs; and
   - **grader/composition** — a per-battery **`REVISION`** integer constant (see below) differs.
   (Weights identical still means stale if the *test* changed — e.g. adding V-hard made every vision score stale.)
3. **Ollama runtime bumped** — `major.minor` of `ollama_version` differs (**ignore patch**: `0.31.1→0.31.2`
   = no, `0.31.1→0.32.0` = yes). **OPT-IN via `--check-runtime` — OFF by default.** Most Ollama releases are
   bugfixes (0.30.8→0.30.12 were minor), so re-running the fleet + cloud on every bump is the wrong default;
   opt in when a release note flags a real perf change (e.g. the 0.31 Gemma4/MLX speedup).

**Decision:** `run = weights_changed OR test_changed OR (check_runtime AND runtime_material)`. Else skip + carry forward.

## Data source — already captured

`bench_utils.env_fingerprint()` (per run, stored in `runs.env`) gives everything:
`ollama_version`, per-model `model_digests` (weight digests — verified: 38 models), `datasets`
(sha256 per dataset/prompt), `benchllama_commit`, `os`/`hardware`. Each DB result row carries a `run_id`
→ join to `runs.env` → the model's digest + version + dataset-hashes **at the time it was scored**.
So the resume decision is: `env_at_last_result(model, battery)` vs `env_fingerprint(now)`. No new
instrumentation — just wiring existing provenance into the decision.

## Per-battery REVISION

Grader-CODE changes with unchanged data aren't caught by dataset hashes (e.g. changing a scoring formula
in `vision.py` without touching `ground_truth.json`). Handle with an explicit **`REVISION = N`** constant
per battery module, bumped **only** on a material change to scoring/composition. Stored in the result (or
derived + stored in `runs.env` as `battery_revisions`). Chosen over hashing the whole module because a
module hash over-triggers on cosmetic edits (a comment change would re-run the fleet). Explicit + precise.
Seed each at its current logical version (e.g. E after E-hard = rev 2, V after V-hard = rev 2).

## Edge-case policies (confirm before building)

- **Cloud models** (`cloud:true`) — no local `/api/tags` digest, provider updates silently → weights
  trigger can't fire. **Default: skip unless test changed or `--force`.** This protects Ollama quota (the
  original pain), at the cost of cloud scores possibly going stale under the hood. Recommended for this repo.
- **Pre-provenance results** (older rows with no `env` link, or `env == {}`) — treat as **unknown → re-run**
  (conservative). Optional one-time backfill later. A model with no prior result at all → always run.
- **Failed digest probe** at run time (Ollama unreachable for a model) → unknown → **re-run** (conservative).
- **Noisy batteries** (E generation, F rollout) — content-addressed is correct: identical inputs ⇒ skip;
  run-noise is captured in σ over the 3 passes, NOT a reason to re-run. (These already 3-run-average.)

## Ollama version comparison

```
def runtime_material(old, new):        # "0.31.1", "0.32.0"
    if not old or not new: return True # unknown → re-run
    om = tuple(int(x) for x in old.split(".")[:2])
    nm = tuple(int(x) for x in new.split(".")[:2])
    return om != nm                    # ignore patch (3rd component)
```
Tunable later (e.g. major-only, or a min-bump threshold). Soft trigger → `--ignore-runtime` skips it.

## Shared API

New module `resume.py` (or in `results_db`), called by ALL battery scripts:

```
def should_run(battery, model, cur_env, *, is_cloud=False, force=False,
               explicit_models=None, ignore_runtime=False) -> (bool, str):
    """Returns (run?, reason). reason ∈ {'forced','explicit','new-model','weights-changed',
       'ollama 0.31→0.32','test-data-changed','battery-rev N→M','no-provenance','up-to-date'}."""

def targets(battery, universe, cur_env, **flags) -> (to_run: list, skipped: list, reasons: dict):
    """The full model split for a battery, driving the run + a report."""
```

Each script replaces its bespoke resume block with `resume.targets(...)`; skipped models are carried
forward **via the DB** (export reads `results_db.latest`), so dated JSON files may be thin — no merge
needed (same as the `average_e_runs` fix). `--force` re-runs all; `--models` = explicit; `--ignore-runtime`
mutes trigger #2; `--resume-report` prints the per-model split + reason and exits (dry run, no benchmarking).

## Rollout (tomorrow)

1. `resume.py` — `env_at_last_result()`, the 3 triggers, `should_run()`, `targets()` + unit tests
   (weights-diff → run; patch-bump → skip; minor-bump → run; rev-bump → run; cloud → skip-unless-test/force;
   no-provenance → run).
2. Add `REVISION` to each battery module (E, F, F-elastic, G, V, EMB, standard, ladder, A–D) at its current
   logical rev; thread into `env_fingerprint` (`battery_revisions`) so it lands in `runs.env`.
3. Swap the six resume blocks (table above) to call `resume.targets()`. Delete the 24h/72h `latest_result`
   windows. `average_e_runs._resume_targets` → thin wrapper over the shared helper.
4. Add `--resume-report` (dry) to `bench.sh`/orchestrator + each script.
5. Validate: dry-report against the current DB should show **nothing re-runs** when weights+runtime+tests
   are unchanged; flip one model's digest (or bump a REVISION) → only that (model|battery) re-runs.

## Open decisions to confirm

- Runtime threshold: `major.minor` (proposed) vs major-only.
- Cloud default: skip-unless-test/force (proposed) vs treat like local.
- Pre-provenance default: re-run (proposed) vs backfill-then-decide.
- REVISION home: per-module constant → `runs.env.battery_revisions` (proposed) vs stored per-result.

## Non-goals

Not changing WHAT the batteries measure or HOW they score — only *whether a given (model, battery) needs
re-running*. `--force` preserves today's "re-benchmark everything" behaviour verbatim.
