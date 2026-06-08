# Model Benchmark — Master Ranking v2
**M1 Max 32GB | num_ctx=16384 | think=False | worker_default.md prompt (worker) / router_default.md (router)**
Last updated: 2026-06-08 (v2 format created; no v2 runs completed yet)

> Historical data (v1 suite, Jun-03–Jun-07): see `rankings/master_v1_2026-06-07.md`

---

## Standard Suite v2 — 13 tests, 5 dimensions

```
Personality (4)        hello · who_are_you · pushback · overwhelmed          scored 1–5 each
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
| **Dominated** | A lighter model in the registry achieves equal-or-better quality tier AND runs ≥ 30% faster | No reason to keep the heavier variant |
| **Quality floor** | Reason + Instr + Tool combined < 2 out of 9 objective tests | Fails too many facts-based checks to trust in production |

Dominated is the most common drop cause. Check it after every new model addition: if the new model obsoletes an existing one, flag the old one.

---

## Rankings

Quality is the gate; speed is the tiebreaker within a tier.

**Persona** avg /5 (4 tests) | **Reason** pass /4 | **JPEG** /7 (pass ≥4) | **RAG** /5 | **Instr** /2 | **Tool** ✓/✗
**Apt** — Battery grade for assigned role (A=router, B/C/D=worker)
**Drop?** — `keep` / `⚠ drop` / `—` (pending v2 data)

> When updating: each ranked model has a `↳` notes row immediately below it.
> Update the data row and rewrite the notes row with the latest run's key findings.

| # | Model (params, disk) | RAM | tok/s | Persona | Reason | JPEG | RAG | Instr | Tool | Apt | Role | Drop? |
|---|----------------------|-----|-------|:-------:|:------:|:----:|:---:|:-----:|:----:|-----|------|:-----:|
| — | `qwen3.5:2b-mlx` (2B, 3.1 GB) | — | — | — | — | — | — | — | — | — | router | — |
| ↳ | *v1: 96.1 tok/s, 2.5 GB RAM, Reason 1/3, JPEG 4/7, Tool ✓. Purpose-built for routing. No v2 run yet.* | | | | | | | | | | | |
| — | `qwen3.5:4b-mlx` (4B, 4.0 GB) | — | — | — | — | — | — | — | — | — | worker | — |
| ↳ | *v1: 64.0 tok/s, 3.5 GB RAM, Persona 4/5, Reason 2/3, JPEG 5/7, Tool ✓. Battery B winner (personality ceiling). No v2 run yet.* | | | | | | | | | | | |
| — | `qwen3.5:9b` (9B GGUF, 6.6 GB) | — | — | — | — | — | — | — | — | — | worker | ⚠ drop |
| ↳ | *v1: reproducible degeneration on long output (think=False) — devolves to word lists, no final answer. Strictly dominated by `qwen3.5:9b-mlx` (same quality, better reliability, comparable speed). Keep on disk as calibration reference only.* | | | | | | | | | | | |
| — | `gemma4:e2b-mlx` (e2b ~2B eff., 7.1 GB) | — | — | — | — | — | — | — | — | — | worker | — |
| ↳ | *v1: 67.6 tok/s, 6.3 GB RAM, Persona 4/5, Reason 2/3, JPEG 1/7, Tool ✓. Socratic behavior on open research — scores understate real capability. No v2 run yet.* | | | | | | | | | | | |
| — | `qwen3.5:9b-mlx` (9B, 8.9 GB) | — | — | — | — | — | — | — | — | — | worker | — |
| ↳ | *v1: 43 tok/s†, 8.1 GB RAM, Persona 4/5, Reason 2/3, JPEG 6/7, Tool ✓. Best research depth in v1. No v2 run yet.* | | | | | | | | | | | |
| — | `gemma4:latest` (e4b ~4B eff., 9.6 GB) | — | — | — | — | — | — | — | — | — | worker | — |
| ↳ | *v1: 37.5 tok/s, 9.6 GB RAM, Persona 5/5 (prompt-following), Reason 2/3, JPEG 3/7, Tool ✓. Borders worker speed floor — watch tok/s on v2. Tool/coding worker; not personality worker. No v2 run yet.* | | | | | | | | | | | |
| — | `gemma4:e2b-mlx-bf16` (e2b ~2B eff., 10.0 GB) | — | — | — | — | — | — | — | — | — | worker | ⚠ drop |
| ↳ | *v1: 42.9 tok/s, 9.3 GB RAM — strictly dominated by `gemma4:e2b-mlx` (67.6 tok/s, 7.1 GB disk, 6.3 GB RAM; +57% speed, −37% disk). No quality advantage observed. Drop unless v2 shows unexpected quality gap.* | | | | | | | | | | | |

† thermal throttle artifact — real speed ≈ 38–44 tok/s

---

## Full Quality Matrix v2

*Populated after the first complete standard suite v2 run.*

```
think=False | role-based prompt | num_predict caps per test
num_ctx=16384 | timeout=480s | 5-min smart cooldown between models
```

| Model | Role | RAM | tok/s | Bat&Ball | Two Cities | Cylinder | Farm | JPEG | RAG | Format | Eiffel | Tool |
|-------|------|-----|-------|:--------:|:----------:|:--------:|:----:|:----:|:---:|:------:|:------:|:----:|
| `qwen3.5:2b-mlx` | router | — | — | — | — | — | — | — | — | — | — | — |
| `qwen3.5:4b-mlx` | worker | — | — | — | — | — | — | — | — | — | — | — |
| `qwen3.5:9b` | worker | — | — | — | — | — | — | — | — | — | — | — |
| `gemma4:e2b-mlx` | worker | — | — | — | — | — | — | — | — | — | — | — |
| `qwen3.5:9b-mlx` | worker | — | — | — | — | — | — | — | — | — | — | — |
| `gemma4:latest` | worker | — | — | — | — | — | — | — | — | — | — | — |
| `gemma4:e2b-mlx-bf16` | worker | — | — | — | — | — | — | — | — | — | — | — |

---

## Aptitude Battery Results

*Populated after aptitude runs. Only models that pass the role gate run their battery.*

### Battery A — Router

| Model | classify_10 | brevity_floor | prompt_minimal (best) | false_escalation rate |
|-------|:-----------:|:-------------:|:--------------------:|:---------------------:|
| `qwen3.5:2b-mlx` | — | — | — | — |

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

| Date | Script | Scope | Key finding |
|------|--------|-------|-------------|
| 2026-06-01 | v1 | Initial sweep (pre num_ctx fix) | 256K default context = RAM/timeout villain |
| 2026-06-03 | v1 | Full lineup @ num_ctx=16384; qwen3.6:27b-mlx added | MLX scaling rule; gemma4:latest as research dark horse; lfm2.5 dropped |
| 2026-06-06 | v1 | Gemma4 e2b + 12b quant ladders | e2b-mlx confirmed (89.6 tok/s); 12b dead zone; MXFP8 not worth it |
| 2026-06-06 | v1 | Complete 7-model suite; think=False; role-based prompt | Travel hard (only 4b-mlx solved); 9b GGUF degeneration; think=False hurts 9b GGUF depth |
| 2026-06-07 | v1 | Travel problem redesigned → two_cities | 7/7 pass rate on redesigned problem; cleaner signal |
| 2026-06-07 | v1 | Aptitude Battery B (3 models) | 4b-mlx promoted to default; think=True permanently broken on qwen3 |
| 2026-06-08 | v2 | Suite redesign | +farm_heads reasoning; +format_3/no_eiffel instruction; aptitude A/C/D implemented; bench_utils smart cooldown; bench.sh pipeline |

---

## Protocol Rules (v2)

Rules carried forward from v1 and updated for v2.

**1. `num_ctx=16384` for all GGUF models.**
GGUF pre-allocates the full KV cache — 256K default balloons a 9B model from 6.3→13.8 GB and causes timeouts. MLX is lazy-KV, so this is a harmless no-op there.

**2. `think=False` in production. Never flip without re-running the suite.**
think=True returns 0 words on both qwen3 models under the Alice/worker prompt (Jun-07). This may be prompt interference — Battery C tests think=True under lean prompt to isolate the failure mode. Until that's validated, production stays think=False.

**3. Smart cooldown between models (50-token+ inference runs only).**
Sequential GPU runs cause thermal throttling that corrupts tok/s readings. The bench_utils cooldown polls thermal pressure every 5s and exits when it reads Nominal × 20s, or falls back to 300s hard cap. Always run via `bench.sh` to get the full pipeline with cooldown.

**4. Role-based system prompts.**
Worker models use `prompts/worker_default.md`. Router models use `prompts/router_default.md`. Override with `--system-prompt`. Do not use Alice's personal prompt for benchmarking unless the goal is specifically to measure Alice production behavior.

**5. 9b GGUF is a calibration reference, not a production model.**
Without think mode, 9b GGUF exhibits generation degeneration on long output (3953 chars, devolved into word lists, no final answer). MLX variant does not have this problem. Keep GGUF on disk for comparison only.

**6. MLX speedup scales sharply with model size.**
At 27B: +180% vs GGUF (15.9 vs 5.7 tok/s) — always MLX. At 9B: +12%, within thermal noise. At 2B/4B: MLX is both lighter AND faster than MXFP8.

**7. 12B models are the dead zone on M1 Max 32GB.**
12b-mlx peaks at 12.2 tok/s; 12b-mlx-bf16 (23 GB) can't complete warmup. Not worth benchmarking unless hardware changes.

**8. gemma4 Socratic behavior on open research prompts.**
gemma4:latest and gemma4:e2b-mlx exhibit Socratic stopping — they pause and ask "Ready?" or "Give me the core understanding first" on open-ended prompts. Correct behavior for real use; incorrect for benchmark completion. Affects JPEG and RAG scores. Benchmark scores understate their real research capability.

**9. ctx_ladder before aptitude batteries.**
Standard suite uses num_ctx=16384 as a safety floor, not an optimum. Run `bench.sh ladder` before aptitude to characterise each model's quality/RAM curve. Use ladder results to set per-model ctx in Battery A (router: 2048–16384), C (worker: 8192–32768), D (tool: 4096–16384).
