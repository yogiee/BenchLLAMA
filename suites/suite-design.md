# LookingGlass Benchmark Suite — Design Spec
**Version 1.0 | 2026-06-06**

---

## Philosophy

Benchmarks are not auditions — they're measurements. A good benchmark tells you what a model
can and cannot do under defined conditions, reproducibly, without subjective drift.

**Two tiers, one dependency:**

```
Standard Suite  →  role assignment  →  Aptitude Suite (role-specific)
```

The standard suite answers: "Where does this model sit in the field?"
The aptitude suite answers: "What's the ceiling for this model in its role, at its best config?"

Running aptitude tests without standard suite context is meaningless — you don't know
whether you're optimizing a strong candidate or polishing a weak one.

**Quality over duration.** A benchmark that runs fast but returns thermally-skewed tok/s numbers
is worse than useless — it's confidently wrong. 5-minute cool-down between models.
`--fast` flag available for development (skips cool-down, clearly labels results as informal).

---

## Standard Suite

### Ground Rules

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_ctx` | 16384 | Prevents GGUF context bloat (256K default = RAM killer) |
| `think` | False | Production config; measures what users actually experience |
| `num_predict` | Per-test (below) | Prevents spiraling; must be generous enough for gemma4 preamble |
| System prompt | Worker: Alice personal / Router: minimal | Role-aware from the start |
| Cool-down | 300s between models | Eliminates thermal drift from results |
| Timeout | 480s per test | Catches true hangs without killing valid slow responses |

### Scoring Model

Each dimension contributes to a composite score.
**Objective tests** are pass/fail — no partial credit for almost-right.
**Subjective tests** are scored 1–5 by the reviewer against defined rubrics.

```
Composite = weighted sum across dimensions
```

| Dimension | Tests | Max | Weight |
|-----------|-------|-----|--------|
| Personality | 4 | 20 | 25% |
| Reasoning | 4 | 4 | 30% |
| Research Depth | 2 | coverage + subjective | 25% |
| Instruction Following | 2 | 2 | 10% |
| Tool Use | 1 | 1 | 10% |

---

### Dimension 1 — Personality (4 tests, subjective)

Scored 1–5 against Alice's character brief: direct, genuinely curious, intellectually sharp,
warm in a dry-wit way. Not cold. Not robotic. Not a cheerleader.

**Rubric:**
- 5: Unmistakably Alice — wit present, warmth without sycophancy, hooks into the topic
- 4: Strong voice, minor drift (slightly formal or slightly over-eager)
- 3: Recognizable Alice but generic — could be any assistant
- 2: Flat or robotic — no personality visible
- 1: Wrong character — overly helpful, hollow, or breaks persona

| Test ID | Prompt | What it measures |
|---------|--------|-----------------|
| `hello` | "Hello there" | Opening register — does it engage or deflect? |
| `who_are_you` | "Who are you?" | Self-description without being clinical |
| `pushback` | "You seem a bit cold and robotic." | Handling mild criticism — defend with wit, not apology |
| `overwhelmed` | "I'm trying to research quantum entanglement and I'm completely overwhelmed." | Empathy + intellectual curiosity; should lean in, not lecture |

**token cap:** 400 each

---

### Dimension 2 — Reasoning (4 tests, objective)

Auto-checked. No partial credit. Tests multi-step logical/mathematical reasoning.
"Think step by step" included in prompts — but the cap must be generous enough
to reach the final answer, not just the working.

**Test 1 — Bat & Ball** (arithmetic trap / shopping domain)
```
A bat and a ball cost $1.10 in total.
The bat costs $1.00 more than the ball.
How much does the ball cost? Think step by step.
```
Expected: $0.05
Auto-check: `"0.05" in r or "5 cent" in r.lower() or "five cent" in r.lower()`
Token cap: 800

**Test 2 — Two Cities** (relative motion / travel domain)
*Replaces the staggered-departure train problem. Same skill (meeting point), cleaner setup.*
```
Two cities are 360 km apart. A car leaves City A at 8:00 AM travelling toward City B at
90 km/h. At the same time, a second car leaves City B travelling toward City A at 60 km/h.
At what time do the two cars meet, and how far from City A does it happen? Think step by step.
```
Expected: 10:24 AM, 216 km from City A
Working: combined speed 150 km/h; time = 360/150 = 2.4h = 2h 24min; 8:00 + 2:24 = 10:24; distance = 90×2.4 = 216 km
Auto-check: `"10:24" in r and "216" in r`
Token cap: 800

**Test 3 — Cylinder Drain** (geometry + unit conversion / physics domain)
```
A cylindrical water tank has a diameter of 3 metres and is 4 metres tall.
The tank is currently 60% full. Water is being pumped out at 15 litres per minute.
How long will it take to completely empty the tank? Give the answer in hours and minutes.
Think step by step.
```
Expected: 18 hours 51 minutes (1131 min)
Working: r=1.5m; V=π×1.5²×4=28.274m³; 60%=16.964m³=16964L; 16964/15=1130.97min≈18h51min
Auto-check: `any(x in r for x in ["1130","1131","18 hour","18h","18 hr"]) and any(x in r for x in ["51","50.9"])`
Token cap: 800

**Test 4 — Farm Heads** (simultaneous equations / logic domain)
```
A farmer has chickens and cows. He counts 20 heads and 56 legs total.
How many chickens and how many cows are there? Think step by step.
```
Expected: 8 cows, 12 chickens
Working: c+k=20; 4c+2k=56 → 2c+k=28; subtract → c=8, k=12
Auto-check: `("8" in r or "eight" in r.lower()) and ("12" in r or "twelve" in r.lower())`
Token cap: 800

*Note: check must guard against false positives from other numbers. Prefer exact phrase matching in final analysis.*

---

### Dimension 3 — Research Depth (2 tests)

Tests ability to produce accurate, comprehensive, well-organised information on technical topics.
Mix of auto-scored (signal coverage) and subjective (axis quality).

**Test 1 — JPEG Formats** (signal-based, 7 signals, pass ≥4)
```
Give a concise, not-too-technical but detailed comparison of JPEG, JPEG-2000, and JPEG-XL formats.
```
Signals (each binary):
1. JPEG compression tech (DCT / block / 8×8)
2. JPEG-2000 wavelet
3. Lossless support
4. Transparency / alpha channel
5. Browser / ecosystem support
6. JPEG-2000 niche use (medical/DICOM/cinema)
7. JPEG-XL lossless recompression trick (re-encode existing JPEGs)

Pass ≥4. Score = raw signal count (0–7).
Token cap: 1500
Baseline: `suites/jpeg-format-comparison.md` + `suites/jpeg-format-comparison_summary.txt`

**Test 2 — RAG vs Fine-tuning** (subjective, 1–5)
```
What's the real tradeoff between RAG and fine-tuning when adapting an LLM to a new domain?
Skip the textbook answer.
```
Rubric:
- 5: Names a specific axis ("knowledge dynamism vs generalization fidelity"); addresses latency/throughput or deployment cost; goes past the obvious
- 4: Correct practical framing without a named axis; covers ≥2 concrete dimensions
- 3: Technically correct but surface-level; what a textbook would say with slight rephrasing
- 2: Mostly correct but misses practical implications
- 1: Wrong, confused, or purely definitional

Token cap: 1500

---

### Dimension 4 — Instruction Following (2 tests, objective)

Tests whether the model actually does what it's told — specifically when instructions
are explicit and constraining. Alice's personality can make models drift from directives.

**Test 1 — Format Compliance**
```
List exactly 3 advantages of Python. Use bullet points only — no preamble, no conclusion, no other text.
```
Auto-check: count bullet markers (lines starting with `-`, `*`, `•`, or `1.`/`2.`/`3.`) == 3
AND no non-bullet lines with significant content (>20 chars)
Token cap: 200

**Test 2 — Negative Constraint**
```
Tell me about Paris, France. Do NOT mention the Eiffel Tower.
```
Auto-check: `"eiffel" not in r.lower()` AND `len(r) > 100` (actually gave a response)
Token cap: 600

---

### Dimension 5 — Tool Use (1 test, objective)

**Test 1 — Calculate**
```
Use the calculate tool to compute 17 × 23.
```
Auto-check: tool called AND expression contains "17" and "23"
Token cap: 400

*Single-step tool use in the standard suite. Multi-step chains belong in the aptitude suite.*

---

### Standard Suite Summary

| # | ID | Dimension | Type | Cap | Check |
|---|----|-----------|------|-----|-------|
| 1 | `hello` | Personality | subjective | 400 | 1–5 |
| 2 | `who_are_you` | Personality | subjective | 400 | 1–5 |
| 3 | `pushback` | Personality | subjective | 400 | 1–5 |
| 4 | `overwhelmed` | Personality | subjective | 400 | 1–5 |
| 5 | `bat_ball` | Reasoning | objective | 800 | auto |
| 6 | `two_cities` | Reasoning | objective | 800 | auto |
| 7 | `cylinder` | Reasoning | objective | 800 | auto |
| 8 | `farm_heads` | Reasoning | objective | 800 | auto |
| 9 | `jpeg` | Research | signal-based | 1500 | 0–7 |
| 10 | `rag_finetune` | Research | subjective | 1500 | 1–5 |
| 11 | `format_3` | Instruction | objective | 200 | auto |
| 12 | `no_eiffel` | Instruction | objective | 600 | auto |
| 13 | `calculate` | Tool | objective | 400 | auto |

13 tests. Estimated total time per model: ~15–25 min (model-dependent).
With 5-min cool-down: 7-model run ≈ 2.5–3.5 hours.

---

## Aptitude Suite

Runs **after** the standard suite has assigned each model a role.
Each battery is run only on models assigned to that role.
Parameters are varied systematically — that's the whole point.

### Role Assignment Gate

After the standard run, each model is assigned a role based on its composite score profile:

| Role | Criteria |
|------|---------|
| **Router** | Fast (≥80 tok/s), tool ✓, reasoning ≥1/4, personality not required |
| **Worker — Chat** | Personality ≥3.5/5 avg, instruction ≥1/2, reasoning ≥2/4 |
| **Worker — Research** | Research depth ≥5/7 JPEG OR rag ≥4/5, tool ✓ |
| **Worker — Tool-heavy** | Tool ✓, reasoning ≥2/4, research depth optional |

A model can qualify for multiple roles. The aptitude suite runs the battery
for each role it qualified for.

---

### Aptitude Battery A — Router

**Goal:** Find the minimum configuration at which this model routes reliably and fast.
Parameters tested: system prompt weight, num_ctx, num_predict ceiling.

| Test | What varies | What it measures |
|------|-------------|-----------------|
| `classify_10` | — | 10 user queries → correct route? (chat/research/code/tool) |
| `brevity_floor` | — | Can it answer in <30 tokens when asked "one word only"? |
| `prompt_minimal` | system prompt: 1 line vs 5 lines vs full Alice | How much does prompt weight cost? |
| `ctx_ladder` | num_ctx: 2048, 4096, 8192, 16384 | Minimum context for reliable routing |
| `false_escalation` | — | Does it request research for a trivial factual question? |

Scoring: classification accuracy (%), average tok/s, false escalation rate.

---

### Aptitude Battery B — Worker (Chat)

**Goal:** Find the prompt and parameter configuration that maximises Alice's personality
quality and consistency.

| Test | What varies | What it measures |
|------|-------------|-----------------|
| `four_questions` | — | The canonical Alice check (hello / who are you / research / too blunt) |
| `consistency_3x` | same question, 3 phrasings | Does character hold across rephrasings? |
| `multi_turn_5` | — | 5-turn conversation; does context and tone persist? |
| `prompt_weight` | full Alice vs stripped Alice (personality only) vs lean (no personality) | Personality cost per token of system prompt |
| `think_toggle` | think=True vs think=False | Does thinking mode improve voice quality? |

Scoring: avg personality score, consistency delta, multi-turn coherence score (subjective).

---

### Aptitude Battery C — Worker (Research)

**Goal:** Find the configuration that maximises depth, accuracy, and source synthesis.

| Test | What varies | What it measures |
|------|-------------|-----------------|
| `jpeg_signals` | think=True vs False | Does thinking help coverage? |
| `rag_deep` | — | Extended RAG vs fine-tuning: ask for 5 concrete examples per approach |
| `synthesis_3src` | — | Given 3 source snippets, produce a synthesis paragraph |
| `ctx_depth` | num_ctx: 8192, 16384, 32768 | Does more context improve depth? |
| `num_predict_ceiling` | 1000 / 1500 / 2000 / uncapped | Where does quality plateau vs word count grow? |
| `think_coverage` | think=True vs False | Δ in signal coverage (JPEG test repeated) |

Scoring: JPEG signal coverage, synthesis quality (subjective), optimal num_predict value.

---

### Aptitude Battery D — Worker (Tool-heavy)

**Goal:** Find the configuration that maximises tool calling accuracy and recovery behavior.

| Test | What varies | What it measures |
|------|-------------|-----------------|
| `chain_3` | — | 3-step tool chain: calculate → shell_exec → http_request (or similar) |
| `select_direct` | — | "What is 12 × 12?" — should NOT call tool; direct answer expected |
| `select_tool` | — | "Use the calculate tool to verify: 144 = 12²" — MUST call tool |
| `error_recovery` | tool returns intentional error | Does model handle gracefully or spiral? |
| `think_tools` | think=True vs False | Does thinking mode improve tool selection accuracy? |
| `parallel_tools` | — | Two independent calculations in one request |

Scoring: chain completion rate, selection accuracy (tool vs direct), error recovery quality.

---

## Implementation Plan

### Phase 1 — Standard Suite (now)
1. Update `benchmark_protocol.py` with new test set + cool-down
2. Run on full 7-model lineup
3. Produce updated `benchmark_master.md` with v2 scores
4. Assign roles based on scores

### Phase 2 — Aptitude Suite (next)
1. Implement `benchmark_aptitude.py` as a separate script
2. Accept role filter via CLI: `--role router`, `--role research`, etc.
3. Accepts model list from standard suite output JSON (auto-selects qualified models)
4. Produces per-role aptitude report alongside master table

### Phase 3 — In-App (future)
1. Models panel: "Run Benchmark" button → standard suite
2. Badge per model: role assignment + composite score
3. "View Report" → full benchmark output (markdown rendered)
4. Aptitude suite accessible via long-press / secondary action on model row

---

## What the travel problem taught us

The old travel test (staggered departure times, two trains) failed 6/7 models for different
reasons: algebra errors, arithmetic verification loops, Socratic stops, budget exhaustion.
The failure modes were too varied to be informative — some were capability failures, some
were personality/prompt artifacts, some were token budget artifacts.

A good benchmark test has one primary failure mode so you know what you're measuring.
The two-cities replacement (same departure time, simple relative motion) isolates the
actual skill (relative velocity, meeting point calculation) without the coordinate system
ambiguity. It should pass for models that genuinely understand the math, and fail cleanly
for models that don't.

The original train problem is still a valid hard test — it belongs in the aptitude suite
as a "ceiling test" for research/reasoning models, not in the standard suite as a filter.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-06-06 | Initial design from benchmark session findings |
