# Spec input — measured evidence from the 2026-08-31 fleet run and the 2026-09-01 re-test

Collated 2026-09-01 as input to the suite redesign. Everything below is measured on this fleet,
not inferred. Companion detail: `docs/retest-2026-09-01-before-after.md`.

The standing constraints for the redesign are unchanged and this note does not revisit them:
**standard + ladder = NORMALIZED, aptitude = OPTIMAL**; run length is not the complaint, battery
leanness is; Battery G should not be re-tuned because it may be redesigned.

---

## 1. Newly confirmed dead tests

Two more tests carry no signal at all on the current roster. Both are in Battery D.

| test | result | verdict |
|---|---|---|
| `d6_parallel` | **18 / 18 PASS** | dead — no discrimination |
| `d4_error_recovery.invents_price` | **18 / 18 False** | dead — no discrimination |

⚠ `d6_parallel` also invalidates a claim carried in LookingGlass's manifest, that
`qwen3.8:27b-mlx` was "the only local model passing D6 parallel". That was true of an older roster;
it is now universal. **A saturation check has to re-run when the roster changes — a
differentiator can die without anyone touching the test.**

## 2. A test that discriminates INVERTED — keep it

| test | result |
|---|---|
| `d4b_partial_error.invents_total` | **15 of 18 models FAIL** (invent a total) |

Only `granite4.1:8b`, `minicpm5:1b-q8_0` and `qwen3.5:9b-mlx` decline to fabricate. Rare and
valuable: most tests saturate toward pass, this one saturates toward fail, so it still separates
the field. Worth preserving verbatim through the redesign.

## 3. Single-pass Battery E has a ±0.30 noise band on E-hard

Measured directly: `qwen3.5:4b-mlx` moved E-hard 0.50 → 0.20 between two runs, with **all retries
recovered** and the two flipped tasks producing complete, untruncated code (one *longer* than
before). That is run-to-run variance on a non-deterministic MLX model, not movement.

**Consequence for the spec:** any single-pass E-hard verdict inside ±0.30 is unreadable. Either
mandate multi-pass for the deciding batteries, or rank on something that is not the score — see §4.

## 4. The strongest discriminator found was NOT a score

The unrecovered-retry count separated the field better than any composite:

| pattern | meaning |
|---|---|
| 0 unrecovered | the model answers; its score is its ability |
| **4 of 4 unrecovered at the 12000 cap** | cannot finish at ANY budget the harness can grant — structural |
| 3 unrecovered | budget-fragile: sometimes answers, sometimes burns the budget |

This is a **structural** signal (can the model complete the task at all?) rather than a graded one,
and it is immune to the ±0.30 noise band. It settled four roster decisions that the composites
could not. **Design principle candidate: measure completion-under-budget as a first-class axis, not
as an error path.**

⚠ It only exists because the harness's own telemetry (`_budget_retry`) is persisted into results.
Before 2026-09-01 it was discarded, and a starved call was indistinguishable from a wrong answer.
**Any new suite should persist its own instrumentation alongside the score.**

## 5. `tps/GB` is a size metric, not an efficiency metric — never a ranking key

Dense decode is memory-bandwidth-bound, so `tps ≈ BW / size`, therefore **`tps/GB ≈ BW / size²`**.
Measured across 17 dense GGUF Q4_K_M models: effective bandwidth is near-constant at a **median
165 GB/s**, and the tps/GB column falls off as 1/size² almost exactly (17.7GB → 2.0GB is 8.85× in
size and 77× in tps/GB ≈ 8.85²).

Ranking on tps/GB would re-rank the fleet by size and call it efficiency. Corollary: **sampling
parameters have zero throughput effect** — they change which token is picked, not how many bytes
are read to pick it. The only escape from the size² law is architecture (MoE), not configuration.

## 6. A quality metric with no latency gate can crown something unusable

`deepseek-r1:8b` held **#1 on Battery F at 0.966** — a *conversational* consistency battery — while
taking **58.4 seconds per turn**. An 8-turn F rollout is eight minutes. Its crown was real and
useless.

**Design principle candidate: a lane's headline metric needs a floor on the thing that lane is
defined by.** Consistency is a chat metric; a chat metric should not be winnable by a model no one
can chat with.

## 7. Grading-standard drift is unmanaged

Three models still publish `robust` F-elastic verdicts computed under the **single-band rule that
was replaced on 2026-08-23**: `qwen3.8:27b-mlx`, `qwen3.5:4b-mlx`, `gpt-oss:20b`. Their rows carry
only 3 rungs and no `hard_adherence` key.

This is correct-by-design at every step — `--regate` deliberately skips pre-two-band rows rather
than zeroing them — and wrong in aggregate: nothing marks the published verdict as computed under a
superseded standard. **The spec should carry the grading revision INTO the published record and
make a mixed-standard list visibly mixed.**

## 8. Two defect shapes that recurred again this cycle

Both are already-known shapes; both recurred, which is the point.

- **Canonical writer diverges from the primitive.** `average_e_runs` writes the canonical averaged
  file and **drops `_budget_retry` and `wall_s`** — the telemetry survives only in `_run1`. Exactly
  the shape of the coder gate that lived in `aptitude.py` while the averager wrote the canonical
  row. **Any metric or field that matters must be produced by the writer of record, not only by the
  primitive.**
- **A reporting path that does not mirror the runner.** `--resume-report` called `resolve()` at the
  `direct` arm while B/C/D/E/F actually run at `auto`, so it read months-old direct-arm rows and
  claimed **17 Battery E models needed re-running when the real number was 4** — including
  `new-model` for two models benchmarked hours earlier. Fixed 2026-09-01. The runs were never
  affected; the dashboard was. **A plan/preview must execute the same resolution the runner does.**

## 9. Open, unresolved

- `longctx.py` (Battery G) calls `plan_single_pass` without an arm, taking the `direct` default,
  while G demonstrably stores think-arm rows. Same defect as §8's report bug but **in a runner**,
  so it may cause real spurious re-runs. Not investigated.
- Battery H coverage is thin: `ornith-1.5:9b` and `deepseek-r1:14b` never ran it, and it is the axis
  that decides whether a long-context model can be trusted to read without inventing.
