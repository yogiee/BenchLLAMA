# Re-test 2026-09-01 — before / after

Historical record for the Battery E starvation fix. **BEFORE** is the 08-31 run
(`aptitude_e_2026-08-31_run1.json`, run_id `2026-08-30T12-45-50`); **AFTER** is the
2026-09-01 re-run (`aptitude_e_2026-09-01.json`).

## Why this run happened

Battery E granted every problem `max_tokens=1024`, but E5 (test-writing) and E-hard emit
2.4k–3.4k chars of code. On a think arm the lever spends the budget reasoning and the code is
cut mid-generation. `starved_on_length` only retried EMPTY replies, so a 19-char fragment
(`granite4.2:8b` / `e5_clamp` → `"test_normal_less_lo"`) was graded as a wrong answer.

Six models were affected. Their E and E-hard scores below are **understated, not earned**.

## Conditions

| | before (08-31) | after (09-01) |
|---|---|---|
| truncation retry | absent | `retry_on_truncation=True` (Battery E only) |
| `_budget_retry` telemetry | discarded by Battery E | persisted per test |
| cool-down | `--fast`, none | `--cooldown`, 5 min between models |
| passes | 1 | 1 |
| base `max_tokens` | 1024 | 1024 (unchanged — the retry is the fix) |

⚠ TWO conditions changed (retry + cool-down), so an improvement cannot be attributed purely to
the retry. Cool-down was added because this cohort has the fleet's longest calls
(`deepcoder:14b` had an 811 s call) and the 8× retry stacks on top — under sustained throttling a
long call can hit the HTTP read timeout and be scored as a failure, i.e. the same artifact class
the re-run exists to remove.

## BEFORE

| model | arm / allowance | E | E-core | E5 | E-hard | coder | starved calls |
|---|---|---|---|---|---|---|---|
| `granite4.2:3b` | think/low · 1024 | 0.739 | – | 0.33 | 0.18 | · | 2 — e5_fib, e5_merge_sorted |
| `granite4.2:8b` | think/low · 1024 | 0.726 | – | 0.20 | 0.35 | · | 2 — e5_clamp, e5_merge_sorted |
| `minicpm5:1b-q8_0` | think/low · 2952 | 0.698 | – | 0.40 | 0.09 | · | 2 — e5_clamp, ehard_apportion |
| `qwen3.5:4b-mlx` | think/low · 4736 | 0.863 | – | 0.80 | 0.50 | Y | 1 — e5_merge_sorted |
| `ornith-1.5:9b` | think/low · 1024 | 0.956 | – | 1.00 | 0.71 | Y | 1 — ehard_settle |
| `deepcoder:14b` | direct/false · 7268 | 0.886 | – | 1.00 | 0.25 | · | 3 — ehard_apportion, ehard_free_slots, ehard_calc |
| `deepseek-r1:8b` | direct/false · 8192 | 0.856 | – | 1.00 | 0.21 | · | 3 — ehard_settle, ehard_apportion, ehard_calc |

**Per-task E-hard** (⚠ = starved: <200 chars of code emitted)

| model | settle | apportion | free_slots | calc |
|---|---|---|---|---|
| `granite4.2:3b` | 0.40 | 0.00 | 0.33 | 0.00 |
| `granite4.2:8b` | 0.00 | 0.40 | 1.00 | 0.00 |
| `minicpm5:1b-q8_0` | 0.20 | 0.00 ⚠ | 0.17 | 0.00 |
| `qwen3.5:4b-mlx` | 0.00 | 1.00 | 1.00 | 0.00 |
| `ornith-1.5:9b` | 0.00 ⚠ | 1.00 | 0.83 | 1.00 |
| `deepcoder:14b` | 1.00 | 0.00 ⚠ | 0.00 ⚠ | 0.00 ⚠ |
| `deepseek-r1:8b` | 0.00 ⚠ | 0.00 ⚠ | 0.83 | 0.00 ⚠ |

## AFTER — Battery E complete (7/7), 311 min compute, wall 02:22 → 09:32

| model | E | E-hard | E5 | retries (unrecovered) | wall |
|---|---|---|---|---|---|
| `granite4.2:3b` | 0.74 → **0.83** | 0.18 → **0.75** | 0.33 → 0.40 | 0 (0) | 4m |
| `granite4.2:8b` | 0.73 → 0.66 | 0.35 → 0.50 | 0.20 → **0.00** | 0 (0) | 5m |
| `minicpm5:1b-q8_0` | 0.70 → 0.73 | 0.09 → 0.25 | 0.40 → 0.40 | 8 (3) | 13m |
| `qwen3.5:4b-mlx` | 0.86 → 0.85 | 0.50 → 0.20 | 0.80 → **1.00** | 8 (0) | 28m |
| `ornith-1.5:9b` | 0.96 → 0.82 | 0.71 → 0.46 | 1.00 → 0.80 | 12 (3) | 61m |
| `deepcoder:14b` | 0.89 → 0.85 | 0.25 → **0.00** | 1.00 → 1.00 | 5 (**4**) | 130m |
| `deepseek-r1:8b` | 0.86 → 0.85 | 0.21 → **0.00** | 1.00 → 1.00 | 4 (**4**) | 70m |

### The robust signal is the UNRECOVERED-RETRY COUNT, not the score delta

Most E-hard deltas sit inside the ±0.30 single-pass noise band and should not be read as movement.
What is *not* noisy is whether a retry to the 12000 cap produced an answer:

- **0 unrecovered** (`granite4.2:3b`, `granite4.2:8b`, `qwen3.5:4b-mlx`) — the model answers; its
  score is its ability.
- **4 of 4 / 4 of 5 unrecovered** (`deepseek-r1:8b`, `deepcoder:14b`) — the model cannot finish the
  hard band at any budget the harness can grant. Structural, not variance.
- **3 unrecovered** (`ornith-1.5:9b`, `minicpm5:1b-q8_0`) — budget-fragile: sometimes answers,
  sometimes burns the budget. Unreliable rather than incapable.

Use this column, not the E-hard delta, when a single-pass verdict has to be trusted.

⚠ **The telemetry only survives in `aptitude_e_<date>_run1.json`.** `average_e_runs` rewrites
`aptitude_e_<date>.json` from recomposed per-test entries and drops `_budget_retry` and `wall_s` —
the averaged file reports 0 retries for every model. Same defect shape as the coder gate that lived
only in `aptitude.py` while the averager wrote the canonical row. **Follow-up: carry `_budget_retry`
through `_average_e`.**
| `deepcoder:14b` | 0.89 → 0.85 | 0.25 → **0.00** | 1.00 → 1.00 | 5 (**none** recovered) |

### ⚠ Read these with a run-variance band, NOT as clean before/after

`qwen3.5:4b-mlx` E-hard fell 0.30 — but the retries all **recovered**, and the two tasks that
flipped did so with COMPLETE, untruncated code:

| task | before | after |
|---|---|---|
| `ehard_apportion` | 1.0 (1345 ch) | 0.0 (842 ch) |
| `ehard_free_slots` | 1.0 (1613 ch) | 0.0 (3674 ch) |
| `ehard_settle` | 0.0 (2680 ch) | **0.8** (7564 ch — recovered by retry) |
| `ehard_calc` | 0.0 (5662 ch) | 0.0 (1417 ch) |

That is **run-to-run variance on a non-deterministic MLX model**, not a harness defect — and it is
exactly why Battery E is normally **3-run averaged** ("consistency across runs matters far more for
coding than for chat/research"). This re-test is SINGLE PASS, chosen to match the 08-31 baseline,
which was also single pass. Consequence: **an E-hard delta smaller than roughly ±0.30 cannot be
distinguished from noise here.** Only deltas well outside that band are signal.

`granite4.2:3b` +0.57 IS outside it. Its E-hard 0.75 now exceeds `granite4.1:3b`'s 0.55 on a clean
direct-arm run — so **granite4.2 did not regress at 3b; it was starved.** Note it recorded 0 retries
this pass, so the recovery came from the run completing inside budget rather than from a retry
firing; cool-down may be doing some of that work (the two-condition caveat above).

### `deepcoder:14b` — zeros are REAL; the withdrawn drop is restored, with proof

All four E-hard tasks empty, every one retried to **12000 = the cap**, `recovered: false` on all of
them. E-hard 0.25 → 0.00. Same signature as `deepseek-r1:8b`.

This was pulled from the drop list on 09-01 with the reasoning "I can't drop a coding specialist on
coding scores I'd already shown were artifacts." That withdrawal was correct **at the time** — the
data could not distinguish starvation from incapacity. It now can, and they were not artifacts.

| 9.0 GB, same lineage | E | E-hard | tool | verdict |
|---|---|---|---|---|
| `deepseek-r1:14b` | 0.91 | **0.75** (clean, no retries) | ✗ | the family's only working hard-band coder |
| `deepcoder:14b` | 0.85 | **0.00** (4/4 empty at cap) | ✗ | strictly dominated |

**Drop candidate — restored.** Identical size to `deepseek-r1:14b`, both fail `calculate`, both
~59 s/turn, but 0.00 vs 0.75 on the hard band and the comparison is no longer confounded.

**R1/deepcoder family, settled:** `deepcoder:1.5b` (dropped, G wall), `deepseek-r1:8b` (E-hard 0.00
at cap), `deepcoder:14b` (E-hard 0.00 at cap) all fail; `deepseek-r1:14b` alone works. The common
cause is `off_supported: False` — thinking cannot be disabled, so the budget goes to reasoning and
no code is emitted. Raising budgets does not help; 12000 is already the cap.

### ⚠ REVERSAL — do NOT retire `ornith:9b`

The pre-run call ("the comparison is already settled; ornith-1.5:9b's only E-hard failure is the
starved `ehard_settle`, so it should reach ~0.96") was **wrong**. `ehard_settle` did not recover —
it was retried and came back empty again — and `ehard_calc` went the other way, 1.0 → empty.

| E-hard task | `ornith:9b` (08-31, clean, 0 retries) | `ornith-1.5:9b` 08-31 | `ornith-1.5:9b` 09-01 |
|---|---|---|---|
| `ehard_settle` | **1.00** | 0.00 (starved) | 0.00 (retried, NOT recovered) |
| `ehard_apportion` | 1.00 | 1.00 | 1.00 (retry recovered) |
| `ehard_free_slots` | 0.83 | 0.83 | 0.83 (retry recovered) |
| `ehard_calc` | 0.50 | 1.00 | **0.00** (retried, NOT recovered) |
| **E-hard** | **0.83** | 0.71 | **0.46** |

`ornith-1.5:9b` also **lost its `coder` tag** (eligible True → False) and needed **12 budget retries
against ornith:9b's zero**. The pattern is budget fragility: it spends the thinking budget and
returns empty on hard tasks, repeatedly and unpredictably.

**Decision: keep `ornith:9b`, and not merely because it is pinned.** It is the more reliable coder
(E-hard 0.83 clean vs 0.46 with 12 retries). `ornith-1.5:9b` still wins Battery F (0.88 vs 0.71),
decode (39.5 vs 35.8 tps) and carries a vision cap — so this is now a genuine split, not a
supersession. The queued F-elastic run on `ornith-1.5:9b` is still worth having: it is missing data
either way, and rePrompt selects on it.

⚠ Lesson: `ehard_settle` returning empty was read as "starved, will recover once the budget is
fixed". It was retried to the cap and stayed empty — the same shape as `deepseek-r1:8b`. **An empty
reply is evidence of a budget problem OR of a model that cannot finish the task; the retry is what
distinguishes them, and until it has run you cannot tell which.**

### ⚠ CORRECTION — granite4.2 was starved at 3b, but genuinely weaker at 8b

The pre-run diagnosis ("all three granite4.2 models collapse on E5, correlated with the think arm →
measurement artifact") holds at **3b** and is **wrong at 8b**.

`granite4.2:8b` re-ran with **0 retries**. `e5_clamp` emitted the same 19-char fragment
(`"test_normal_less_lo"`) as on 08-31 — but the truncation retry did not fire, which means
`done_reason != "length"`: **the model chose to stop, it was not cut off.** (The mechanism is
demonstrably live — `qwen3.5:4b-mlx` logged a `truncated_content_on_length` retry in the same pass.)

So its E5 is a real failure, not starvation, and it reproduces: **0.20 → 0.00 across two independent
runs against `granite4.1:8b`'s 1.00** — far outside the ±0.30 noise band.

| | E | E-hard | E5 | disk |
|---|---|---|---|---|
| `granite4.1:8b` (clean direct) | **0.90** | **0.60** | **1.00** | 5.3 GB |
| `granite4.2:8b` (re-run) | 0.66 | 0.50 | 0.00 | 5.3 GB |

`granite4.2:8b` is dominated by its own 4.1 sibling at identical size on coding. It keeps two merits:
Battery F 0.87 vs 0.79, and a working `thinking` capability 4.1 lacks. **Prune candidate — decide
against those two, not against the coding numbers.**

Note E-hard moved the other way (0.35 → 0.50) with two tasks swapping (`apportion` 0.4→0.0,
`settle` 0.0→1.0): inside the noise band, not signal.

### `deepseek-r1:8b` — E-hard 0.00 is REAL, and now proven

All four E-hard tasks returned **zero characters, before and after**. Each was retried to
`num_predict` **12000 = `BUDGET_RETRY_CAP`** with `"recovered": false`. Its `think_profile` shows
`off_supported: False` and every lever collapsing to one `think1` class — thinking cannot be
switched off, so it burns the entire budget reasoning and never emits code. **Not a budget problem
and not fixable by raising one:** `THINK_ALLOWANCE_MAX` is already 8192 and the retry cap is 12000.

This is precisely what fix #2 (persisting `_budget_retry`) bought — on 08-31 the same zeros were
indistinguishable from a starved call. Now the record shows the harness gave it every chance.

**Roster consequence:** `deepseek-r1:8b` moves from "your call, leaning drop" to a solid drop
candidate — E-hard 0.00 with full retry coverage, fails `calculate` despite a `tools` cap, 70 min
for a single E pass, 58 s/turn in chat, and the fleet's worst RAM blow-up (10.3 GB resident on a
5.2 GB model, 1.98×). Its only merit is Battery F 0.97, the fleet's highest.

**If a model's verdict lands inside the noise band, run 2 more passes for that model alone rather
than trusting this number:** `python3 average_e_runs.py --runs 3 --models <model> --cooldown`.

## Decisions waiting on this

| question | resolves how |
|---|---|
| `granite4.2:3b` / `:8b` keep-or-drop | E-hard on honest data vs granite4.1 siblings (4.1 scored 0.55 / 0.60 on clean direct-arm runs) |
| `ornith:9b` retire? | `ornith-1.5:9b` E-hard should go 0.71 → ~0.96 if `ehard_settle` recovers; it already ties or beats ornith:9b on every task it completed, and wins F 0.88 vs 0.71 |
| `ornith:9b` re-point | needs `ornith-1.5:9b` F-elastic = `robust` + `verdict_stable` before rePrompt's `.assignments.reprompt_rewrite.model` can move |
| `minicpm5:1b-q8_0`, `qwen3.5:4b-mlx`, `deepcoder:14b`, `deepseek-r1:8b` | true E-hard, currently unknown |


---

## F-elastic — `ornith-1.5:9b` (3-run avg, first ever measurement)

| | verdict | instr_adh | hard_adh | prompt-σ | **length_adh** | stable |
|---|---|---|---|---|---|---|
| `ornith-1.5:9b` | **robust** | 0.984 | **0.934** | **0.0127** | **0.167** | True |
| `ornith:9b` (08-23) | robust | 1.000 | 0.847 | 0.0318 | **0.847** | True |

It **passes the gate** — `robust` + `verdict_stable`, and it beats ornith:9b on both
verdict-driving metrics (hard-band adherence 0.934 vs 0.847, prompt-σ 0.0127 vs 0.0318).

### But do NOT re-point rePrompt to it

`length_adherence` is **0.167 vs 0.847 — five times worse.** That meter is deliberately excluded
from the verdict (so wordiness cannot masquerade as prompt-insensitivity), which is correct in
general and misleading here: **rePrompt's `text` subcommand is compress/expand.** Following a word
cap is not a side metric for that job, it is the job. A model at 0.167 ignores length instructions.

Combined with Battery E — `ornith-1.5:9b` budget-fragile (12 retries, 3 unrecovered), E-hard 0.46
vs 0.83, and it **lost its `coder` tag** — the conclusion is the opposite of the pre-run prediction:

**KEEP `ornith:9b` as rePrompt's `reprompt_rewrite.model`. No re-point, no drop.**

`ornith-1.5:9b` still earns its roster slot on Battery F (0.88 vs 0.71), decode (39.5 vs 35.8 tps),
a vision cap, and better prompt-elasticity on the verdict axes. The two are a genuine split — a
chat/vision lane vs a coding/length-discipline lane — not a supersession in either direction.

⚠ Generalises: **a gate can pass while the metric that matters for the consumer's actual task
fails.** Check the consumer's job against the excluded meters, not just the verdict.
