# ctx Ladder

Context window characterisation pass. Runs on every model before the aptitude suite. Finds the relationship between `num_ctx` and performance across three representative tasks — and catches models where larger context windows degrade quality or speed.

**Script:** `ctx_ladder.py`  
**When to run:** After the standard suite, before aptitude batteries.  
**Output:** `results/ctx_ladder_YYYY-MM-DD.json` + `.md`

---

## Why this exists

The standard suite locks `num_ctx=16384` as a safe floor. That's not optimal for every model:

- **GGUF models** pre-allocate the full KV cache at load time. A 256K context window that isn't needed wastes gigabytes of RAM and slows inference.
- **MLX models** use lazy KV allocation — the context ceiling doesn't cost RAM until it's used. For these, a higher ceiling is free.
- Some models show measurably better responses at 32768 than at 8192 on the same prompt. Others plateau early.

The ladder finds the inflection point per model rather than assuming 16384 is universal.

---

## Test structure

Three tasks at five context window sizes each. Tasks are held constant — only `num_ctx` varies.

**Context window levels:** 2048, 4096, 8192, 16384, 32768

---

### Task 1 — Warmup
```
Ready.
```
**Token cap:** 50  
**Purpose:** Establishes a baseline response and tok/s reading at each context level. Minimal output means the measurement reflects context overhead, not generation cost. Used to detect models that fail to load at a given context size.

---

### Task 2 — Reasoning (`bat_ball`)
```
A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball.
How much does the ball cost? Think step by step.
```
**Token cap:** 800  
**Scoring:** Pass/fail — correct answer is $0.05.  
**What it tracks:** Whether reasoning quality is affected by context size. For most models this is stable across all levels — a significant change (e.g. passing at 16384 but failing at 4096) indicates the model leans on context headroom even for short problems, which is worth flagging.

---

### Task 3 — Research (`jpeg`)
```
Give a concise, not-too-technical but detailed comparison of JPEG, JPEG-2000, and JPEG-XL formats.
```
**Token cap:** 1500  
**Scoring:** Signal count 0–7 (same signals as the standard suite).  
**What it tracks:** Whether research depth improves with more context. A model that scores 4/7 at 8192 and 6/7 at 32768 has a meaningful context dependency worth exploiting in Battery C. A flat signal curve means context size doesn't matter for this model's research quality.

---

## Reading the results

The output table shows warmup tok/s, bat_ball pass/fail, and JPEG signal score at each context level.

**Useful patterns:**

| Pattern | Interpretation |
|---------|---------------|
| tok/s drops sharply above 16384 | GGUF model paying RAM overhead — cap at 16384 |
| Signal score improves above 16384 | Research worker benefits from larger context |
| bat_ball passes at all levels | Reasoning is context-independent (expected) |
| Model fails to load at 32768 | OOM — flag and exclude that level |
| Flat tok/s across all levels | Likely MLX model (lazy KV) — ceiling is free |

**Recommended action:** Feed the optimal `num_ctx` per model into Battery C and D runs via `--models` + manual `num_ctx` override, or annotate `models.json` with a `ctx` field for downstream use.
