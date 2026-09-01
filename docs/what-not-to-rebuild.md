# What Not To Rebuild

**Post-mortem of run `2026-08-30T12-45-50`** — 32 models · 5 phases · 31.8 h · ollama 0.33.2
Compiled 2026-08-31. Input for the suite rewrite.

Eight design flaws the run exposed, with the evidence that exposed them, written as constraints
rather than as a bug list. **The infrastructure is not implicated anywhere in this document** —
see [What held up](#what-held-up).

---

## 1 · What we measured

Flaws in the tests themselves: what they ask of a model, and whether the answer still separates
anything.

### 1.1 Saturation is the end state of every test, not an accident

Across four batteries, the same pattern — tests every model now passes. These aren't merely easy;
they contribute **zero information** while still consuming weight, wall-clock and attention. The
suite's consistent response was to add a harder sibling rather than retire the dead one.

| test | at ceiling | fleet mean | verdict |
| :--- | ---: | ---: | :--- |
| F5 coherence_recovery | 32/32 | 1.000 | spent |
| V ocr / chart / describe | 12/12 | 1.000 | spent |
| F1 persona_hold | 29/32 | 0.979 | spent |
| E1 generate | 27/32 | 0.960 | spent |
| E9 markup | 26/32 | 0.973 | spent |
| G superseded | 26/32 | 0.948 | spent |
| G absent | 25/32 | 0.937 | spent |
| G aggregate | 16/32 | 0.755 | live |
| V count | 5/12 | 0.417 | live |
| E-hard | 5/32 | 0.542 | live |

**For the rewrite:** make retirement a first-class operation. A test whose fleet-wide ceiling rate
crosses a threshold has stopped measuring and should be *replaced*, not supplemented. Audit it every
run — saturation is the destination, so the check has to be permanent.

### 1.2 The live signal ended up outweighed by the dead ones

The same defect surfaced **four separate times in one run**, in four batteries written months apart.
That rules out carelessness: it is what happens when a hard band is bolted onto a composite whose
core has already saturated, instead of the scoring being redesigned around what still discriminates.

| battery | the discriminator | its weight | diluted by |
| :--- | ---: | :--- | :--- |
| E | E-hard (mean 0.542) | 15% | 6 mostly-spent categories hold the other 85% |
| F | F2, F4 | 45% | F1 + F5 both dead, holding 30% |
| F-elastic | 3 hard constraints | 3-of-7 | 4 saturated core constraints |
| G | aggregate, multihop | 2-of-4 | superseded + absent, both spent |

> E-hard's raw weight is `0.18` of a `1.18` total, so its **effective share is 15.3%**.

**For the rewrite:** never average a live signal with dead ones and call the mean a score. Scope each
band to the items that still separate models — the fix that worked on F-elastic, where re-scoping
moved 19/19 `robust` to 14/5, while raising the bar alone had changed **nothing**.

### 1.3 Cost went where the target user never goes

Battery E ran **15.6 h**, of which 12.7 h was model time. Five of thirty-two models — **16% of the
roster — consumed 51% of it**.

| model | measured min | why |
| :--- | ---: | :--- |
| deepcoder:14b | 116 | R1 lineage — no `off` think class exists |
| deepseek-r1:8b | 70 | same; reasons on every call, every battery |
| deepseek-r1:14b | 70 | same lineage |
| qwen3.5:9b-mlx | 67 | 7420-token think allowance |
| bonsai-27b:1bit | 67 | 23.5 tok/s — 27B squeezed into 1 bit |

Nobody who has just installed Ollama and OpenWebUI is running these.

**For the rewrite:** let the persona pick the roster. If a model wouldn't plausibly be installed by a
developer trying local LLMs for the first time, it doesn't earn fleet-wide wall-clock. Cost should
track relevance; right now it tracks pathology.

---

## 2 · How we scored and published it

Flaws between a correct measurement and a useful answer — the stretch where this run did most of its
damage.

### 2.1 Two rankings could not rank

The lists exist to answer *"which model should I use"*. Two of them can't: their sort keys produce a
plateau, so the published order at the top is **file order**.

| list | entries | distinct keys | tied at #1 | verdict |
| :--- | ---: | ---: | ---: | :--- |
| long_context | 32 | 21 | 8 | top quarter unordered |
| coders | 21 | 17 | 4 | top unordered |
| vision | 12 | 12 | 1 | clean |

Vision is the counterexample and proves the architecture isn't at fault — its hard band still
discriminates, so all twelve entries separate.

**For the rewrite:** treat the tie distribution of a sort key as an **acceptance test** for a
battery, not a post-hoc audit. A key that can't separate its top quarter has failed, however sound
the measurements underneath it are.

### 2.2 A gate lived in the wrong code path and was silently reverted — twice

The coder gate's E-hard floor was implemented in the per-run scorer (`aptitude.py`), but the
**canonical published row is written by the averager** (`average_e_runs._average_e`), which never had
it. One clean offline re-gate was destroyed by the very next run. The DB records the whole story:

| E run | E-hard gate in the stored row |
| :--- | :--- |
| 06-14 → 07-29 | absent — predates the re-tune, correct |
| 08-20, 08-21 | **present** — re-gated offline on 08-22 |
| 08-23 | absent again — overwritten by the next run |
| 08-30 | absent — overwritten again |

At the point it was caught: **28 of 32 models tagged** against 20 under the documented gate.
`qwen2.5vl:3b` — a vision model — held the `coder` role on an E-hard score of **0.042**. That is
verbatim the failure the 08-22 re-tune was written to eliminate.

**For the rewrite:** one scorer, one place a gate can live. If a battery has both a per-run and an
aggregate path, the aggregate is the only one allowed to decide anything — and an offline
re-derivation must be verified against a **freshly written** row, not just the rows it rewrote.

### 2.3 Unmeasurable was recorded as measured-and-failed

Three distinct cases in one run, each a harness limit published as a property of the model. The suite
has no consistent notion of *could not measure*, so it collapses into *measured badly*.

| case | what was published | what was true |
| :--- | :--- | :--- |
| granite4.2:30b, G | `clean_depth 16384` | 7/7 at every measured depth; 32k needs ~650 s prefill against a 600 s timeout |
| qwen3.5:4b-mlx, G | `prefill_collapse 1131.4` | a 645,690 tok/s reading; fleet range 0.39–0.69 |
| deepseek-r1:8b, E | three E-hard zeros | genuine — but only provable after the retry cap was fixed |

**For the rewrite:** every metric needs a validity predicate and **three** outcomes rather than two —
measured-good, measured-bad, and not-measurable. The third must never silently become the second.
Battery G already gets this right for a timed-out depth (it excludes rather than zeroes), so make
that the rule everywhere instead of one battery's local good behaviour.

---

## 3 · How we ran and reported it

Flaws in limits and in the reading of results — cheap to prevent, and both capable of invalidating
hours of correct measurement.

### 3.1 Limits that don't scale with the thing the test varies

| limit | scaled against | what the test actually varies |
| :--- | :--- | :--- |
| `longctx.TIMEOUT = 600` | `num_predict` | fill depth — 1k to 32k |
| `THINK_ALLOWANCE_MAX = 16384` | nothing — it exceeded `BUDGET_RETRY_CAP = 12000` | the safety net sat *below* the thing it protects |

The second was worse than a wrong number. When the allowance exceeded the retry cap, the retry
computed a smaller budget, bailed, and **printed nothing** — so an unretried starvation looked
identical to a model declining to answer. That same retry was recovering **82% of the calls it
caught** everywhere else.

**For the rewrite:** any limit must scale with the axis its battery sweeps, and relationships between
limits must be **asserted**, not assumed. A guard that can silently do nothing is worse than no
guard, because it looks like a result.

### 3.2 The reading of a result is part of the result

Three reporting defects, none of which touched a stored number, and one of which caused a real
misjudgement of where a 9.6 h phase stood.

| defect | effect |
| :--- | :--- |
| model cards never reset between passes | pass 3 read "31 of 32 done" at 16 of 32 actual |
| phase label hardcoded `· 3-run avg` | written to `phase_timings` under `--runs 1` — a permanent false record |
| progress regex `[\w.]+` | missed F's spaced markers (`[T1 establish]`), so F never had a progress bar at all |

**For the rewrite:** anything persisted alongside a score — a label, a phase record, a count — is a
claim about how the score was produced and must be as true as the score. Progress reporting for a
multipass battery has to know which pass it is in.

---

## What held up

Load-bearing and proven under stress this run. **Carry these into the rewrite unchanged** — the
redesign is of the batteries, not of what runs them.

- **Content-addressed resume.** Aborting mid-run cost nothing for the six completed phases — every
  model carried forward correctly, no re-measurement. Exercised for real, not in theory.
- **Honest exclusion.** The one place the suite already distinguishes *could not measure* from
  *failed*: a timed-out depth is dropped from the denominator (`n_depths: 5`) rather than zeroed.
- **Offline re-derivation.** `--regate` and `--rescore` re-apply a policy change against stored data
  on original run IDs. The prefill guard went across 58 rows with **zero grading changes** and no
  re-measurement.
- **Provenance.** `env_fingerprint` made every question in this post-mortem answerable — which
  runtime, which digests, which dataset, which revision.
- **Per-model checkpointing** — except in the multipass batteries, where its absence discarded
  3 h 25 m of Battery E when the run was stopped.
- **The two-band idea**, but only ever with the §1.1 audit attached. Vision demonstrates it working;
  G demonstrates what it becomes without one.

---

## Fixed during this run

| fix | where |
| :--- | :--- |
| retry cap inversion — `THINK_ALLOWANCE_MAX` 16384 → 8192 | `bench_utils.py` |
| coder gate's missing E-hard floor (28 → 21 tags) | `average_e_runs.py` + `--regate` |
| `prefill_collapse` sanity ceiling (58 rows re-derived) | `longctx.py` + `--rescore` |
| model cards reset between multipass passes | `orchestrator.py` |
| phase label tracks the real pass count | `orchestrator.py` |
| per-unit `params.runs` (single-pass E beside 3-pass F) | `webserver.py` |
| Battery F progress bar | `web/index.html` |

## Deliberately not fixed

- **Battery G's hard-band re-scope.** Calibration work on a battery the rewrite may replace.
- **`rankings/master.md`.** 133 KB of narrative describing batteries that may not survive.
