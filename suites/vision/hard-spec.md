# Vision Hard Band (V-hard) — SPEC

**Status:** designed 2026-07-05; **BUILT + calibrated 2026-07-05.** Active band = **count_dense · count_region · table_sum**
(weight 0.25). `ocr_hard` / `chart_hard` were built but PARKED after calibration — qwen2.5vl:3b aced them even
hardened (OCR + chart-reading are its *strengths*; kept in `generate.py` for a possible weaker-VLM band).

**Calibration (qwen2.5vl:3b, the ceiling check):** `composite_core = 1.0` (unchanged) → **`composite_hard = 0.839`**
→ `composite = 0.960`. **The 1.00 ceiling is broken** — a better VLM can now rank above qwen. Per-task on qwen:
`count_dense 0.756` (undercounts dense conjunctions — the strongest probe), `table_sum 0.828` (real arithmetic
slips: 256 vs 235, 197 vs 207), `count_region 0.933` (weakest — qwen counts a small filtered region fine even
after hardening to a colour×region conjunction). **Finding: qwen's exploitable weaknesses are DENSE counting +
ARITHMETIC, not perception (OCR/charts) or small-set counting.** Getting a *fair* band below ~0.8 isn't realistic
— qwen is genuinely strong — and weight 0.25 keeps the discrimination intentionally modest (per user).

## Why

Battery V **saturates**: `qwen2.5vl:3b` scores a perfect **1.00** (aces ocr/count/chart/spatial/describe).
Like Battery E before E-hard, the ceiling means no model can rank *above* the champion — a genuinely
better VLM can only tie at 1.00 or lose. And the obvious "more params" bet is already spent: the archive
shows `qwen2.5vl:7b`, `qwen3-vl:{2b,4b,8b}`, `llava:7b`, `mistral-small3.2:24b` were all benchmarked and
**lost to the 3B** on this fixture (`llama3.2-vision:11b` broke). **V-hard adds objective, PIL-ground-truth
tasks that break the 1.00 ceiling so VLMs can finally be ranked above `qwen2.5vl:3b`.**

## Design principles (inherited from the harness)

- **Objective + judge-free.** PIL renders every fixture, so ground truth is exact — no VLM/LLM judge (same
  as V-core and Battery G). This is the battery's core strength; keep it.
- **Two-band, like Battery E.** V-core (the current 5 tasks) = the *sees?* gate + baseline; **V-hard = the
  ranking discriminator** (weight ≈ 0.25). Report `composite_core`, `composite_hard`, and the weighted
  `composite`. The *sees?* gate stays on core (basic vision works ⇒ still admitted to the lane).
- **Continuous sub-scores over binary — this is the key to breaking the ceiling.** Binary pass/fail tasks
  saturate (qwen gets them or it doesn't). Graded 0..1 tasks (fuzzy-ratio, proximity, sequence-similarity)
  give the fine-grained separation that ranks two strong VLMs apart. That's *why* OCR (ratio) and describe
  (signal-fraction) already give non-binary signal — the hard band leans into it everywhere.
- **Seeded, multi-round.** 3 seeded rounds per task, averaged (kills single-scene luck); report σ.
- **Target real VLM failure modes.** Dense conjunction counting, small/rotated no-dictionary OCR,
  multi-series / interpolated charts, multi-object spatial + ordering, structured-table reads.

## The tasks (5 + 1 stretch)

### H1 · `count_dense` — attribute-conjunction counting
- **Render:** 15–30 shapes, {circle · triangle · square} × {red · blue · green · yellow}, varied size,
  slight overlap, on a light patterned background.
- **Ask:** "How many **blue triangles**?" (a shape × colour conjunction, not a single attribute).
- **Grade (continuous):** proximity `max(0, 1 − |pred − true| / true)` — off-by-one on 12 ≈ 0.92; far
  smoother than binary exact-match.
- **Why hard:** VLMs reliably fail counting past ~6, and conjunction-filtering + occlusion compounds it.
- **qwen2.5vl:3b target: 0.5–0.8.**

### H2 · `ocr_hard` — small / rotated / no-dictionary text
- **Render:** a random alphanumeric code (mixed case + digits, e.g. `7Xk-9Q4-ZM2p`) at small font, ±10°
  rotation, moderate contrast; plus a dense 6-line small-font block variant.
- **Ask:** transcribe exactly.
- **Grade:** difflib ratio (0..1 straight into the composite).
- **Why hard:** no language prior to fall back on (random code), and small+rotated defeats casual OCR — the
  current OCR task is large clean text that everyone aces.
- **qwen target: 0.6–0.85.**

### H3 · `chart_hard` — multi-series + between-gridline interpolation
- **Render:** a grouped bar chart (3 series × 4 categories) and/or a line chart with an unlabelled point
  *between* gridlines.
- **Ask:** "Value of **Series B** in category **Q3**?" / "y-value at x = 5?"
- **Grade:** proximity `1 − |err| / axis_range` (tight tolerance).
- **Why hard:** series disambiguation + reading between gridlines; the current chart task is a single
  labelled bar.
- **qwen target: 0.5–0.8.**

### H4 · `spatial_hard` — compound relations + ordering
- **Render:** 4–6 labelled shapes at seeded positions.
- **Ask (2 kinds):** (a) compound relation — "Is the green circle **above AND left of** the red square?"
  → yes/no; (b) ordering — "List the shapes **left → right**." → an ordered sequence.
- **Grade:** (a) binary; (b) **sequence similarity** (normalized LCS / rank-correlation, 0..1).
- **Why hard:** multiple simultaneous relations + serialization; the current spatial task is one yes/no.
- **qwen target: 0.5–0.8.**

### H5 · `table_read` — structured grid
- **Render:** a 4-row × 3-col table with row + column headers, numeric cells.
- **Ask:** "Value at row **March**, column **Revenue**?" (cell lookup) **and** "Sum of the **Revenue**
  column?" (read-all + arithmetic).
- **Grade:** cell = exact / ratio; column-sum = proximity.
- **Why hard:** grid parsing + cell indexing + arithmetic over read values — a genuine document / agent
  use case, untested by V-core.
- **qwen target: 0.6–0.85.**

### H6 (stretch, deferred) · `gauge_clock` — analog read
- Read an analog clock to the nearest 5 min, or a gauge needle to the nearest tick; grade ±1 tick. A
  classic VLM failure. Deferred — clean rendering is more work and the tolerance is fuzzier.

## Scoring & composite

- Each hard task = one **dimension**, averaged over its 3 seeded rounds (continuous 0..1).
- `composite_hard` = mean of the 5 hard dimensions.
- `composite` = **0.75 · composite_core + 0.25 · composite_hard** _(weight to calibrate; start 0.25)._
- `composite_core` (the current 5 tasks) is unchanged → drives the *sees?* gate + back-compat.
- Report per-task detail + σ across rounds (so a noisy dimension is visible, not hidden in the mean).

## Integration points

- `suites/vision/generate.py` — add `gen_hard_*()`; tag hard tasks `"band": "hard"`; +~15 PNGs (5 × 3 rounds).
- `suites/vision/ground_truth.json` — hard tasks carry `"band": "hard"`.
- `vision.py` — new scorers (`proximity_count`, `sequence`, `table`); group units by band; emit
  `composite_core` / `composite_hard` / `composite` + per-band; keep the *sees?* gate on core.
- `export.py` — the `vision` sub-block gains `composite_hard` / `composite_core` (mirrors coding's core/full).
- `rankings/master.md` — the Vision table gains a **Hard** column; V finally becomes the ranking
  discriminator it can't be today.

## Validation / calibration — do this BEFORE trusting any rank

1. Generate the fixtures; run `qwen2.5vl:3b` + `minicpm-v4.6:1b` + one gemma vision worker.
2. **PASS criterion:** `qwen2.5vl:3b` must drop to roughly **0.4–0.7 on `composite_hard`** (NOT ~1.0). If it
   still aces the band, the tasks aren't hard enough → more objects / smaller text / tighter tolerance,
   and re-check. This is the whole point — headroom above the champion.
3. Calibrate the 0.25 band weight so hard reshapes ranking without collapsing the battery (E-hard landed
   on 0.18 after the same exercise).
4. Sanity: every ground truth must be **unambiguous to a human**. Reject any task whose "right answer" is
   debatable — objectivity is the battery's contract.

## Build sequence

1. `generate.py` hard fixtures + ground truth (near-free — produces the 15 PNGs to eyeball first).
2. `vision.py` scorers + two-band composite.
3. **Calibration run on `qwen2.5vl:3b`** (the ceiling check) → tune difficulty + weight.
4. export + master.md wiring.
