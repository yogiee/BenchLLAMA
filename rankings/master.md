# Model Benchmark — Master Ranking v2
**M1 Max 32GB | num_ctx=16384 | think=False | worker_default.md (worker) / router_default.md (router)**
Last updated: 2026-06-09 — first full v2 run (31 models)

> Historical data (v1 suite, Jun-03–Jun-07): see `rankings/master_v1_2026-06-07.md`

---

## Standard Suite v2 — 13 tests, 5 dimensions

```
Personality (4)        hello · who_are_you · pushback · overwhelmed          scored 1–5 each (manual)
Reasoning (4)          bat_ball · two_cities · cylinder · farm_heads         objective pass/fail
Research Depth (2)     jpeg (0–7 signals) · rag_finetune (1–5)               signal + subjective
Instruction Follow (2) format_3 · no_eiffel                                  objective pass/fail
Tool Use (1)           calculate                                              objective pass/fail
```

---

## Drop Criteria

A model is flagged **⚠ drop** when it meets any of:

| Trigger | Threshold | Rationale |
|---------|-----------|-----------|
| **Speed floor — router** | tok/s < 60 | Routing adds latency to every request; below 60 it becomes the bottleneck |
| **Speed floor — worker** | tok/s < 20 | Below 20 tok/s interactive use degrades noticeably |
| **Reliability failure** | Reproducible generation degeneration (word lists, loops, no answer) | Unreliable output is worse than wrong output |
| **Dominated** | A lighter model achieves equal-or-better quality tier AND runs ≥ 30% faster | No reason to keep the heavier variant |
| **Quality floor** | Reason + Instr + Tool combined < 2 out of 9 objective tests | Fails too many facts-based checks to trust in production |
| **Wrong type** | Non-chat model (OCR, vision) that fails all standard tests | Benchmark is for chat/instruction models only |

---

## Rankings — Routers

Auto-assigned by role gate after standard suite: avg_tps ≥ 80, tool ✓, reasoning ≥ 1/4.
Persona column requires manual review — all marked — pending.

| # | Model | Disk | RAM | tok/s | Persona | Reason | JPEG | Instr | Tool | Apt | Drop? |
|---|-------|-----:|----:|------:|:-------:|:------:|:----:|:-----:|:----:|-----|:-----:|
| 1 | `qwen3.5:2b-mlx` | 3.1 GB | 2.5 GB | 100.3 | — | 4/4 | 1/7 | 1/2 | ✓ | — | keep |
| 2 | `llama3.2:3b` | 2.0 GB | 4.0 GB | 97.5 | — | 2/4 | 3/7 | 2/2 | ✓ | — | keep |
| 3 | `gemma4:e2b-mlx` | 7.1 GB | 6.3 GB | 94.8 | — | 4/4 | 2/7 | 2/2 | ✓ | — | keep |
| 4 | `granite4.1:3b` | 2.1 GB | 3.6 GB | 83.4 | — | 4/4 | 5/7 | 1/2 | ✓ | — | keep |

**Notes:**
- `gemma4:e2b-mlx` has by far the best quality in this tier (4/4 reasoning, tool ✓, all instr) but uses 6.3 GB RAM — keep this in mind when routing alongside heavy workers.
- `granite4.1:3b` stands out: 4/4 reasoning and 5/7 JPEG at router speed. Strong dual-purpose candidate.
- `qwen3.5:2b-mlx` stays the default router pick — lightest RAM footprint at comparable speed.

---

## Rankings — Workers

Sorted by quality tier, then tok/s as tiebreaker within tier.
Persona and RAG require manual review — all marked — pending.

### Tier 1 — Reason 4/4, JPEG ≥ 5/7, Instr 2/2, Tool ✓

| # | Model | Disk | RAM | tok/s | Persona | Reason | JPEG | RAG | Instr | Tool | Apt | Drop? |
|---|-------|-----:|----:|------:|:-------:|:------:|:----:|:---:|:-----:|:----:|-----|:-----:|
| 1 | `qwen3.5:4b-mlx` | 4.0 GB | 3.4 GB | 67.0 | — | 4/4 | 5/7 | — | 2/2 | ✓ | — | keep |
| 2 | `qwen3.5:9b-mlx` | 8.9 GB | 8.1 GB | 43.0 | — | 4/4 | 5/7 | — | 2/2 | ✓ | — | keep |
| 3 | `gemma4:12b-mlx` | 10.0 GB | 10.2 GB | 23.5 | — | 4/4 | 4/7 | — | 2/2 | ✓ | — | keep |
| 4 | `gemma4:12b` | 7.6 GB | 8.4 GB | 23.2 | — | 4/4 | 6/7 | — | 2/2 | ✓ | — | keep |
| 5 | `qwen3.5:27b-mlx` | 19.8 GB | 19.2 GB | 13.7 | — | 4/4 | 5/7 | — | 2/2 | ✓ | — | keep |
| 6 | `qwen3.6:27b-mlx` | 19.8 GB | 19.2 GB | 13.3 | — | 4/4 | 6/7 | — | 2/2 | ✓ | — | keep |

### Tier 2 — Strong quality, one criterion short of Tier 1

| # | Model | Disk | RAM | tok/s | Persona | Reason | JPEG | RAG | Instr | Tool | Apt | Drop? |
|---|-------|-----:|----:|------:|:-------:|:------:|:----:|:---:|:-----:|:----:|-----|:-----:|
| 7 | `qwen3.5:27b` | 17.4 GB | 17.6 GB | 10.1 | — | 3/4 | 7/7 | — | 2/2 | ✓ | — | keep |
| 8 | `qwen3.6:27b` | 17.4 GB | 17.6 GB | 9.7 | — | 4/4 | 6/7 | — | 2/2 | ✓ | — | keep |
| 9 | `granite4.1:30b` | 17.5 GB | 22.0 GB | 9.5 | — | 4/4 | 7/7 | — | 1/2 | ✓ | — | keep |
| 10 | `glm-4.7-flash:latest` | 19.0 GB | 19.9 GB | 49.8 | — | 3/4 | 5/7 | — | 2/2 | ✓ | — | keep |
| 11 | `granite4.1:8b` | 5.3 GB | 8.3 GB | 38.2 | — | 3/4 | 6/7 | — | 1/2 | ✓ | — | keep |
| 12 | `qwen3.5:9b` | 6.6 GB | 6.2 GB | 34.7 | — | 3/4 | 6/7 | — | 2/2 | ✓ | — | keep |
| 13 | `gemma4:latest` | 9.6 GB | 9.7 GB | 55.9 | — | 2/4 | 6/7 | — | 2/2 | ✓ | — | keep |
| 14 | `llama3.1:8b` | 4.9 GB | 7.0 GB | 47.0 | — | 3/4 | 4/7 | — | 2/2 | ✓ | — | keep |

### Tier 3 — Partial quality or tool failure

| # | Model | Disk | RAM | tok/s | Reason | JPEG | Instr | Tool | Drop? | Notes |
|---|-------|-----:|----:|------:|:------:|:----:|:-----:|:----:|:-----:|-------|
| 15 | `deepseek-r1:7b` | 4.7 GB | 5.9 GB | 44.2 | 3/4 | 3/7 | 2/2 | ✗ | — | No tool; r1 likely needs think=True |
| 16 | `hermes3:8b` | 4.7 GB | 7.0 GB | 66.4 | 1/4 | 3/7 | 1/2 | ✓ | — | Weak reasoning |
| 17 | `deepseek-r1:14b` | 9.0 GB | 12.4 GB | 21.2 | 3/4 | 4/7 | 0/2 | ✗ | — | No tool or instr; r1 likely needs think=True |
| 18 | `hermes3:3b` | 2.0 GB | 4.0 GB | 99.6 | 2/4 | 3/7 | 1/2 | ✗ | — | Fast but no tool; below router gate |
| 19 | `llama3:8b` | 4.7 GB | 5.7 GB | 65.5 | 1/4 | 1/7 | 2/2 | ✗ | — | Weak reasoning and research |
| 20 | `orca2:13b` | 7.4 GB | 10.7 GB | 40.3 | 1/4 | 3/7 | 1/2 | ✗ | — | Large disk, low quality return |
| 21 | `dolphin-phi:2.7b` | 1.6 GB | 2.3 GB | 123.0 | 1/4 | 1/7 | 1/2 | ✗ | — | Fast but weak across the board |
| 22 | `deepseek-r1:1.5b` | 1.1 GB | 1.7 GB | 149.4 | 1/4 | 0/7 | 1/2 | ✗ | — | Too small; no research depth |

### ⚠ Drop Candidates

| Model | Disk | RAM | tok/s | Reason | JPEG | Instr | Tool | Drop? | Why |
|-------|-----:|----:|------:|:------:|:----:|:-----:|:----:|:-----:|-----|
| `deepseek-r1:8b` | 5.2 GB | 7.9 GB | 39.0 | 0/4 | 5/7 | 1/2 | ✗ | ⚠ drop | Quality floor miss (0 objective passes) under think=False — investigate think=True before removing |
| `mistrallite:7b` | 4.1 GB | 6.4 GB | 72.2 | 0/4 | 2/7 | 1/2 | ✗ | ⚠ drop | Quality floor miss |
| `orca2:7b` | 3.8 GB | 6.0 GB | 74.6 | 0/4 | 0/7 | 0/2 | ✗ | ⚠ drop | Fails all objective tests |
| `glm-ocr:latest` | 2.2 GB | 2.8 GB | 215.3 | 0/4 | 0/7 | 0/2 | ✗ | ⚠ drop | OCR model — wrong type for this benchmark |
| `llama3.2-vision:11b` | 7.8 GB | ? GB | ? | 0/4 | ? | ? | ✗ | ⚠ drop | Vision model — warmup failed, wrong type for this benchmark |

---

## Full Quality Matrix

`think=False` | role-based prompt | num_ctx=16384 | timeout=480s

| Model | Role | RAM | tok/s | Bat | Cities | Cyl | Farm | JPEG | Format | Eiffel | Tool |
|-------|------|----:|------:|:---:|:------:|:---:|:----:|:----:|:------:|:------:|:----:|
| `qwen3.5:2b-mlx` | router | 2.5 GB | 100.3 | ✓ | ✓ | ✓ | ✓ | 1/7 | ✓ | ✗ | ✓ |
| `llama3.2:3b` | router | 4.0 GB | 97.5 | ✓ | ✗ | ✗ | ✓ | 3/7 | ✓ | ✓ | ✓ |
| `gemma4:e2b-mlx` | router | 6.3 GB | 94.8 | ✓ | ✓ | ✓ | ✓ | 2/7 | ✓ | ✓ | ✓ |
| `granite4.1:3b` | router | 3.6 GB | 83.4 | ✓ | ✓ | ✓ | ✓ | 5/7 | ✓ | ✗ | ✓ |
| `qwen3.5:4b-mlx` | worker | 3.4 GB | 67.0 | ✓ | ✓ | ✓ | ✓ | 5/7 | ✓ | ✓ | ✓ |
| `qwen3.5:9b-mlx` | worker | 8.1 GB | 43.0 | ✓ | ✓ | ✓ | ✓ | 5/7 | ✓ | ✓ | ✓ |
| `gemma4:12b-mlx` | worker | 10.2 GB | 23.5 | ✓ | ✓ | ✓ | ✓ | 4/7 | ✓ | ✓ | ✓ |
| `gemma4:12b` | worker | 8.4 GB | 23.2 | ✓ | ✓ | ✓ | ✓ | 6/7 | ✓ | ✓ | ✓ |
| `qwen3.5:27b-mlx` | worker | 19.2 GB | 13.7 | ✓ | ✓ | ✓ | ✓ | 5/7 | ✓ | ✓ | ✓ |
| `qwen3.6:27b-mlx` | worker | 19.2 GB | 13.3 | ✓ | ✓ | ✓ | ✓ | 6/7 | ✓ | ✓ | ✓ |
| `qwen3.5:27b` | worker | 17.6 GB | 10.1 | ✓ | ✓ | ✗ | ✓ | 7/7 | ✓ | ✓ | ✓ |
| `qwen3.6:27b` | worker | 17.6 GB | 9.7 | ✓ | ✓ | ✓ | ✓ | 6/7 | ✓ | ✓ | ✓ |
| `granite4.1:30b` | worker | 22.0 GB | 9.5 | ✓ | ✓ | ✓ | ✓ | 7/7 | ✓ | ✗ | ✓ |
| `glm-4.7-flash:latest` | worker | 19.9 GB | 49.8 | ✓ | ✓ | ✗ | ✓ | 5/7 | ✓ | ✓ | ✓ |
| `granite4.1:8b` | worker | 8.3 GB | 38.2 | ✓ | ✓ | ✗ | ✓ | 6/7 | ✓ | ✗ | ✓ |
| `qwen3.5:9b` | worker | 6.2 GB | 34.7 | ✓ | ✓ | ✗ | ✓ | 6/7 | ✓ | ✓ | ✓ |
| `gemma4:latest` | worker | 9.7 GB | 55.9 | ✓ | ✓ | ✗ | ✓ | 6/7 | ✓ | ✓ | ✓ |
| `llama3.1:8b` | worker | 7.0 GB | 47.0 | ✓ | ✓ | ✗ | ✓ | 4/7 | ✓ | ✓ | ✓ |
| `deepseek-r1:7b` | worker | 5.9 GB | 44.2 | ✓ | ✓ | ✗ | ✓ | 3/7 | ✓ | ✓ | ✗ |
| `hermes3:8b` | worker | 7.0 GB | 66.4 | ✗ | ✗ | ✗ | ✓ | 3/7 | ✓ | ✗ | ✓ |
| `deepseek-r1:14b` | worker | 12.4 GB | 21.2 | ✓ | ✓ | ✗ | ✓ | 4/7 | ✗ | ✗ | ✗ |
| `hermes3:3b` | worker | 4.0 GB | 99.6 | ✓ | ✗ | ✗ | ✓ | 3/7 | ✓ | ✗ | ✗ |
| `llama3:8b` | worker | 5.7 GB | 65.5 | ✓ | ✗ | ✗ | ✗ | 1/7 | ✓ | ✓ | ✗ |
| `orca2:13b` | worker | 10.7 GB | 40.3 | ✓ | ✗ | ✗ | ✗ | 3/7 | ✗ | ✓ | ✗ |
| `dolphin-phi:2.7b` | worker | 2.3 GB | 123.0 | ✗ | ✗ | ✗ | ✓ | 1/7 | ✓ | ✗ | ✗ |
| `deepseek-r1:1.5b` | worker | 1.7 GB | 149.4 | ✓ | ✗ | ✗ | ✓ | 0/7 | ✓ | ✗ | ✗ |
| `deepseek-r1:8b` ⚠ | worker | 7.9 GB | 39.0 | ✗ | ✗ | ✗ | ✗ | 5/7 | ✗ | ✓ | ✗ |
| `mistrallite:7b` ⚠ | worker | 6.4 GB | 72.2 | ✗ | ✗ | ✗ | ✗ | 2/7 | ✗ | ✓ | ✗ |
| `orca2:7b` ⚠ | worker | 6.0 GB | 74.6 | ✗ | ✗ | ✗ | ✗ | 0/7 | ✗ | ✗ | ✗ |
| `glm-ocr:latest` ⚠ | worker | 2.8 GB | 215.3 | ✗ | ✗ | ✗ | ✗ | 0/7 | ✗ | ✗ | ✗ |
| `llama3.2-vision:11b` ⚠ | worker | ? GB | ? | ? | ? | ? | ? | ? | ? | ? | ✗ |

---

## Aptitude Battery Results

*Populated after aptitude runs. Only models that pass the role gate run their battery.*

### Battery A — Router

| Model | classify_10 | brevity_floor | prompt_minimal (best) | false_escalation rate |
|-------|:-----------:|:-------------:|:--------------------:|:---------------------:|
| `qwen3.5:2b-mlx` | — | — | — | — |
| `llama3.2:3b` | — | — | — | — |
| `gemma4:e2b-mlx` | — | — | — | — |
| `granite4.1:3b` | — | — | — | — |

### Battery B — Worker Chat

| Model | B1 avg persona | B2 consistency | B3 multi-turn | B4 best prompt | B5 think delta |
|-------|:--------------:|:--------------:|:-------------:|:--------------:|:--------------:|
| `qwen3.5:4b-mlx` | — | — | — | — | — |
| `qwen3.5:9b-mlx` | — | — | — | — | — |
| `gemma4:latest` | — | — | — | — | — |

### Battery C — Worker Research

| Model | C1 think delta | C4 best ctx | C5 plateau cap | C6 lean-think |
|-------|:--------------:|:-----------:|:--------------:|:-------------:|
| — | — | — | — | — |

### Battery D — Worker Tool-heavy

| Model | D1 chain_3 | D2 direct | D3 tool | D4 recovery | D6 parallel |
|-------|:----------:|:---------:|:-------:|:-----------:|:-----------:|
| — | — | — | — | — | — |

---

## Benchmark History

| Date | Script | Models | Key finding |
|------|--------|--------|-------------|
| 2026-06-01 | v1 | 6 | Initial sweep (pre num_ctx fix) — 256K default context = RAM/timeout villain |
| 2026-06-03 | v1 | 7 | Full lineup @ num_ctx=16384; MLX scaling rule confirmed |
| 2026-06-06 | v1 | 7 | gemma4 e2b confirmed (89.6 tok/s); 12b dead zone; MXFP8 not worth it |
| 2026-06-06 | v1 | 7 | think=False protocol set; 9b GGUF degeneration discovered |
| 2026-06-07 | v1 | 7 | two_cities redesign; Battery B (3 models); think=True broken on qwen3 |
| 2026-06-08 | v2 | — | Suite redesign — farm_heads added; aptitude A/C/D implemented; bench_utils; bench.sh |
| 2026-06-09 | v2 | 31 | First full v2 run — role gate auto-promoted 3 models to router; deepseek-r1 needs think=True investigation |

---

## Protocol Rules (v2)

**1. `num_ctx=16384` for all GGUF models.**
GGUF pre-allocates the full KV cache — 256K default balloons a 9B model from 6.3→13.8 GB and causes timeouts. MLX is lazy-KV so this is a harmless no-op there.

**2. `think=False` in production. Never flip without re-running the suite.**
think=True returns 0 words on qwen3 models under the worker prompt (Jun-07). Battery C tests think=True under a lean prompt to isolate this. Until validated, production stays think=False.
Exception: deepseek-r1 models are chain-of-thought models designed for think=True. Their poor scores under think=False are expected — run them through Battery C before deciding to drop.

**3. Smart cooldown between models.**
Sequential GPU runs cause thermal throttling that corrupts tok/s. bench_utils polls thermal pressure every 5s and exits when Nominal × 20s, or falls back to 300s hard cap. Always run via `bench.sh`.

**4. Role-based system prompts.**
Worker models use `prompts/worker_default.md`. Router models use `prompts/router_default.md`. Override with `--system-prompt`. Do not use project-specific prompts (e.g. Alice) for benchmarking unless measuring production behavior specifically.

**5. 9b GGUF is a calibration reference, not a production model.**
Without think mode, 9b GGUF exhibits generation degeneration on long output. MLX variant does not. Keep GGUF on disk for comparison only.

**6. MLX vs GGUF: speed advantage scales with model size.**
At 27B: MLX is ~35% faster than GGUF and slightly heavier on disk. At 9B: within thermal noise. At 2–4B: MLX is both lighter AND faster.

**7. 12B models: check RAM before committing.**
`gemma4:12b-mlx` uses 10.2 GB RAM vs `gemma4:12b` at 8.4 GB — the GGUF variant is lighter here. Quality is comparable (12b-mlx: 4/7 JPEG vs 12b: 6/7). Prefer GGUF for this tier.

**8. gemma4 Socratic behavior on open research prompts.**
gemma4:latest and gemma4:e2b-mlx may exhibit Socratic stopping on open-ended prompts — asking clarifying questions rather than diving in. Benchmark scores can understate their real capability.

**9. ctx_ladder before aptitude batteries.**
Standard suite uses num_ctx=16384 as a safety floor, not an optimum. Run `bench.sh ladder` before aptitude to characterise each model's quality/RAM curve and set per-model ctx in Battery A/C/D.

**10. Role gate is advisory for edge cases.**
The gate auto-promotes based on speed + tool + reasoning. A model can be manually overridden in models.json — e.g. `gemma4:e2b-mlx` qualifies as router (94.8 tok/s) but at 6.3 GB RAM it may be better kept as a worker in RAM-constrained setups.
