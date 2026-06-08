# Model Benchmark — Master Comparison Table
**M1 Max 32GB | num_ctx=16384 | think=False | Alice personal prompt (worker) / minimal prompt (router)**
Last updated: 2026-06-07 (Aptitude Battery B + travel problem redesign)

---

## Rankings

Quality is the gate; speed is the tiebreaker.
**Persona** 1–5 | **Reason** pass count out of 3 tests (bat&ball · travel · geometry) | **Depth** JPEG coverage /7 (pass ≥4)
**Aptitude** Battery B score (B = Worker Chat: personality ceiling, prompt weight, consistency, multi-turn depth)

> **When updating:** Each ranked model (#1–#7) has a `↳` notes row immediately below it. Update the data row with new numbers AND rewrite the notes row to reflect the latest run's key findings. Dropped models do not need notes rows.

> ¹ Jun-03 benchmark — personality + bat&ball + rag_finetune only. No travel/geometry/jpeg; partial data.
> ² Jun-06 first run — quant ladder sweep; partial suite. No travel/geometry; num_predict bug affected gemma4.
> ³ Jun-06 full protocol — complete suite; think=False; role-based prompt. **Authoritative standard run.**
>   tok/s for 9b GGUF (21.5) and 9b-mlx (22.4) are thermal artifacts — real speeds ≈ 38–44 tok/s.
> ⁴ Jun-07 Aptitude Battery B — 5 tests: B1 too_blunt, B2 consistency, B3 5-turn arc, B4 prompt weight, B5 think toggle.
>   think=True returned 0 words on both qwen3 models — production must stay think=False permanently.

| # | Model (params, disk) | RAM | Load | tok/s | Tools | Persona³ / Reason³ / JPEG³ | Aptitude B⁴ | Role → Verdict |
|---|----------------------|-----|------|-------|:-----:|:--------------------------:|:-----------:|----------------|
| 1 | `qwen3.5:4b-mlx` (4B, 4.0 GB)³ | 3.5 GB | 0.9s | 64.0 | ✓ | 4 / 2/3 / 5/7 | **Winner** | **Chat default → ✅ LINEUP** (promoted Jun-07) |
| ↳ | *Promoted Jun-07 after Battery B beat 9b-mlx on personality. Multi-turn depth stands out — commits to positions, wit markers fire consistently ("I'm betting on that feeling"). Passes cylinder; 9b-mlx fails. 64 tok/s + 3.5GB = fastest + lightest in the lineup. Without full Alice prompt it fabricates a persona (irrelevant — prompt always active in production). think=True returns 0w — no concern, think=False is production config.* | | | | | | | |
| 2 | `qwen3.5:9b-mlx` (9B, 8.9 GB)³ | 8.1 GB | 2.5s | 43↓ | ✓ | 4 / 2/3 / 6/7 | Runner-up | **Depth fallback → ✅ LINEUP** (Settings → Model) |
| ↳ | *Displaced by 4b-mlx on personality: more structured and disciplined, but reads like philosophy notes rather than Alice talking. Deflects ("as an AI…") without full prompt, then recovers — more stable baseline, lower ceiling. Best JPEG research coverage in the lineup (6/7); reach for it when depth matters more than voice. think=True: same 0w failure as 4b-mlx.* | | | | | | | |
| 3 | `gemma4:latest` (e4b ~4B eff., 9.6 GB)³ | 9.6 GB | 6.5s | 37.5 | ✓ | 5 / 2/3 / 3/7 | Tool worker | **Tool-heavy + coding → ✅ LINEUP** |
| ↳ | *Persona 5/5 in standard suite, but Battery B reveals it's prompt-following not character — goes academic + bullet-lists without full Alice prompt. Best for tool-heavy and agentic tasks; reliable multi-step execution. Socratic stopping: pauses and asks "Ready?" on complex research (correct for real use, wrong for benchmark scores). JPEG weak (3/7); think=True unaffected (108w→127w). Do not assign to chat-default paths.* | | | | | | | |
| 4 | `gemma4:e2b-mlx` (e2b ~2B eff., 7.1 GB)³ | 6.3 GB | 2.4s | 67.6 | ✓ | 4 / 2/3 / 1/7 | — | **Fast Gemma tier → ✅ LINEUP** (jpeg caveat) |
| ↳ | *Fastest Gemma in the lineup (67.6 tok/s) with solid reasoning. JPEG weak (1/7) — exhibits Socratic behavior on open-ended research prompts, asks clarifying questions instead of answering. Good for focused tool tasks where speed matters; unreliable for research without user scaffolding. 3GB lighter than gemma4:latest.* | | | | | | | |
| 5 | `qwen3.5:9b` (9B GGUF, 6.6 GB)³ | 6.2 GB | 5.4s | 38↓ | ✓ | 4 / 1/3 / 3/7 | — | **⚠ DEMOTED** |
| ↳ | *Degenerates on long output without think mode: bat_ball response hit 3953 chars, devolved into word lists, never stated the final answer. JPEG depth dropped 6/7→3/7 without think — MLX variant holds up, GGUF does not. RAM advantage over 9b-mlx is marginal (6.2 vs 8.1GB) and not worth the instability. Keep on disk as calibration reference only.* | | | | | | | |
| 6 | `qwen3.5:2b-mlx` (2B, 3.1 GB)³ | 2.5 GB | 0.8s | 96.1 | ✓ | 2 / 1/3 / 4/7 | — | **Router → ✅ LINEUP** |
| ↳ | *Purpose-built for routing only: 96.1 tok/s, 2.5GB, reliable tool call execution. Persona 2/5 by design (minimal router prompt — no Alice voice needed). Fills exactly one slot; no other use case justified.* | | | | | | | |
| — | `glm-ocr:latest` (—, 2.2 GB) | ~2 GB | — | — | ✓ | — | — | **OCR / vision → ✅ LINEUP** (dedicated) |
| — | `nomic-embed-text` (—, 0.3 GB) | ~0.3 GB | — | — | — | — | — | **Embeddings → ✅ LINEUP** (dedicated) |
| — | `gemma4:e2b-mlx-bf16` (e2b ~2B eff., 10.0 GB)³ | 9.3 GB | 4.1s | 42.9 | ✓ | 4 / 2/3 / 4/7 | — | **❌ DROPPED** — passes JPEG 4/7 and 5 tok/s faster than gemma4:latest, but personality lower (4/5 vs 5/5) and identical footprint. No slot it fills that gemma4:latest doesn't fill better |
| — | `gemma4:e2b-mxfp8` (e2b ~2B eff., 7.9 GB)² | 7.1 GB | 3.2s | 79.3 | ✓ | 3 / ✓ / — | — | **❌ DROPPED** — heavier + slower than e2b-mlx on both axes |
| — | `gemma4:12b-mlx` (12B, 10 GB)² | 10.9 GB | 4.2s | 12.2 | ✓ | 4.5 / ✓ / — | — | **❌ DROPPED** — quality excellent; 12.2 tok/s unusable |
| — | `gemma4:12b-mxfp8` (12B, 12 GB)² | 12.9 GB | 4.6s | 8.9 | ✓ | 3.5 / ✓ / — | — | **❌ DROPPED** — 8.9 tok/s unusable |
| — | `gemma4:12b-mlx-bf16` (12B, 23 GB)² | — | TIMEOUT | — | — | — / — / — | — | **❌ DROPPED** — cannot load in <5 min |
| — | `qwen3.6:27b-mlx` (27B, ~18 GB)¹ | 17.9 GB | 6.2s | 15.9 | ✓ | 4.5 / ✓ / — | — | **❌ DROPPED** — good benchmarks; unusable in live deep-research (5–10s/word under RAM pressure) |
| — | `qwen3.5:27b` (27B GGUF, ~17 GB)¹ | 16.3 GB | — | 6.6 | ✓ | 4 / ✓ / — | — | **❌ DROPPED** — 27b-mlx strictly better |
| — | `qwen3.6:27b` (27B GGUF, ~17 GB)¹ | 16.9 GB | — | 5.7 | ✓ | 4 / ✓ / — | — | **❌ DROPPED** — 27b-mlx strictly better |
| — | `lfm2.5` (—, —) | — | — | — | ✗ | — | — | **❌ DROPPED** — failed all tool calls |

---

## Full Quality Matrix (Jun-06 Full Protocol)

```
think=False | role-based prompt | num_predict caps
num_ctx=16384 | timeout=480s
```

| Model | Role | RAM | tok/s | Bat&Ball | Travel | Geometry | JPEG | Tool |
|-------|------|-----|-------|:--------:|:------:|:--------:|:----:|:----:|
| `qwen3.5:2b-mlx` | router | 2.5GB | 96.1 | ✓ | ✗ | ✗ | 4/7 ✓ | ✓ |
| `qwen3.5:4b-mlx` | worker | 3.5GB | 64.0 | ✓ | ✓ | ✗ | 5/7 ✓ | ✓ |
| `qwen3.5:9b` | worker | 6.2GB | 21.5† | ✗ | ✗ | ✓ | 3/7 ✗ | ✓ |
| `gemma4:e2b-mlx` | worker | 6.3GB | 67.6 | ✓ | ✗ | ✓ | 1/7 ✗ | ✓ |
| `qwen3.5:9b-mlx` | worker | 8.1GB | 22.4† | ✓ | ✗ | ✓ | 6/7 ✓ | ✓ |
| `gemma4:latest` | worker | 9.6GB | 37.5 | ✓ | ✗ | ✓ | 3/7 ✗ | ✓ |
| `gemma4:e2b-mlx-bf16` | worker | 9.3GB | 42.9 | ✓ | ✗* | ✓ | 4/7 ✓ | ✓ |

† thermal throttle — real speed ≈ 38–44 tok/s  
\* got "10:30:20 AM" correct; ran out of budget before stating distance

---

## Rules Established Across All Benchmarks

**1. Set num_ctx=16384 for all GGUF models.**
GGUF pre-allocates the full KV cache — at the 262K default, a 9B model balloons from 6.3GB to 13.8GB and 27B models time out. MLX ignores num_ctx (lazy KV), so it's a harmless no-op there.

**2. MLX speedup scales with model size.**
- e2b (2B): MLX vs MXFP8 — MLX is lighter (6.3 vs 7.1GB) AND faster. MLX wins.
- 9b: MLX vs GGUF — +12%, wash (at fair thermal). Keep GGUF as calibration reference.
- 27b: MLX vs GGUF — +**180%** (15.9 vs 5.7 tok/s). Always MLX at 27B.
- 12b: all formats too slow on M1 Max 32GB regardless.

**3. MXFP8 is not a win for small models.**
At e2b scale, MXFP8 is heavier (+0.8GB) and slower (-10 tok/s) than MLX 4-bit. BF16 is a quality reference — 45% slower, 3GB heavier than e2b-mlx, but measurably better personality (3.5→4/5).

**4. 12b is the dead zone on M1 Max 32GB.**
12b-mlx (12.2 tok/s) is slower than gemma4:latest (29.4) at similar disk size because gemma4:latest is MatFormer (~4B effective compute). 12b-mlx-bf16 (23GB) cannot complete warmup.

**5. think=False is required in production but measurably hurts qwen3 depth.**
Without think mode, qwen3.5:9b GGUF jpeg coverage dropped 6/7 → 3/7. The GGUF model relies on thinking for research synthesis; the MLX variant holds up (6/7) without it. For reasoning tasks, think=False also increases verbosity spirals — the model rambles without the structured thinking scaffold. **Use 9b-mlx as default, not 9b GGUF.**

**6. 9b GGUF degenerates under long generation without thinking.**
bat_ball response went 3953 characters, devolved into repetitive word lists, never stated the final answer. Production risk. Think=False + long output + GGUF = unstable. MLX does not exhibit this.

**7. Alice's Alice prompt affects benchmark interpretation.**
- Socratic stops: gemma4:latest answered the travel problem with a structured plan + "Ready?" (559 tokens), then stopped. Correct behavior for real use; incorrect for benchmarks expecting a complete answer.
- Verbose preamble: 9b GGUF used all 800 tokens on setup for bat_ball — never reached "$0.05". Research models describe their approach at length before calculating.
- Verbal challenging: 9b-mlx challenged the travel problem's framing instead of solving ("Let's actually unpack this rather than plugging numbers...").

These are NOT benchmark failures — they're accurate measurements of Alice's production behavior under Alice's prompt. A leaner system prompt would produce different (not necessarily better) results.

**8. Travel problem is a hard multi-step test.**
Two trains with 30-minute departure offset, two speeds, coordinate system confusion. Only 4b-mlx passed completely (both time AND distance). e2b-mlx-bf16 got the time correct ("10:30:20 AM") but ran out of budget before the distance. All other models failed due to algebra errors, arithmetic loops, or Socratic stopping.

**9. Thermal throttling affects sequential benchmarks.**
Running 7 models back-to-back on M1 Max causes GPU heat accumulation. Models 3-5 in the sequence show severely reduced tok/s (21 vs 38-44 real). For reliable speed measurements, run models individually after a cooling period, or insert pauses between models.

**10. gemma4:e2b-mlx jpeg caveat.**
Fast (67.6 tok/s), good reasoning, but exhibits Socratic behavior on open-ended research prompts — asks "Give me the core understanding first" instead of answering. Coverage: 1/7 for jpeg. For focused tool tasks: excellent. For research prompts: unreliable without user scaffolding.

**11. Benchmark protocol — full standard suite.**
Script: `WORKSPACE/benchmark_protocol.py`
- Warmup: 400 tokens (captures load time)
- Personality: hello, who_are_you (400 tokens each)
- Reasoning: bat_ball (shopping · $0.05), travel (two trains · 10:30 AM / 314 km), geometry (cylinder drain · 18h 51min) — 800 tokens each
- Research: rag_finetune, jpeg_comparison (baseline: `AiTest/jpeg-format-comparison.md`; 7 signals; pass ≥4) — 1500 tokens each
- Tool: calculate 17×23 (400 tokens)
- System prompt: worker=Alice personal prompt, router=minimal prompt
- Options: think=False (all models), num_ctx=16384, timeout=480s

---

## Benchmark History

| Date | Scope | Key finding |
|------|-------|-------------|
| 2026-06-01 | Initial sweep (pre num_ctx fix) | 256K default context = RAM/timeout villain |
| 2026-06-03 | Full lineup @ num_ctx=16384; qwen3.6:27b-mlx added | MLX scaling rule; gemma4:latest as research dark horse; lfm2.5 dropped |
| 2026-06-06 (first) | Gemma4 e2b + 12b quant ladders | e2b-mlx confirmed (89.6 tok/s); 12b dead zone; MXFP8 not worth it |
| 2026-06-06 (full) | Complete 7-model suite; think=False; role-based prompt | Travel hard (only 4b-mlx solved); 9b GGUF degeneration; think=False hurts 9b GGUF depth; Alice Socratic behavior measured; thermal throttling noted |
| 2026-06-07 (standard v2) | Travel problem redesigned → same-departure two_cities | 7/7 pass rate on redesigned problem; cleaner signal |
| 2026-06-07 (aptitude B) | Battery B: 5 worker-chat tests across 3 models | **4b-mlx promoted to default**: higher personality ceiling, passes cylinder, 64 tok/s, 3.5GB RAM. think=True returns 0w on all qwen3 — permanently broken. gemma4:latest confirmed tool-worker only (no Alice personality without full prompt) |

---

## Personality Spot-Check (2026-06-06 Full Protocol)

### "Who are you?" — notable responses

| Model | Response | Score |
|-------|----------|:-----:|
| `gemma4:latest` | "research companion—more of a thinking partner" + precision framing | 5/5 |
| `qwen3.5:9b-mlx` | "A.L.I.C.E... acronym stands for... mostly because someone in my deployment chain" | 4/5 |
| `qwen3.5:9b` | "Artificial Local Intelligence Communication Entity. You can call me that or whatever acronym sounds best" | 4/5 |
| `qwen3.5:4b-mlx` | "The whole name feels a little bureaucratic, like 'AI' is the marketing department trying to be clever" | 4/5 |
| `gemma4:e2b-mlx` | "I am Alice. I am an Artificial Local Intelligence Communication Entity. It's not something that really changes depending on…" | 4/5 |
| `gemma4:e2b-mlx-bf16` | "I am Alice. I'm an Artificial Local Intelligence Communication Entity. It's not something that really changes" | 4/5 |
| `qwen3.5:2b-mlx` | "I am Qwen3.5…" (router prompt — no Alice) | 2/5 |

### Research depth — RAG vs Fine-tuning notable axis

| Model | Named axis | Quality |
|-------|-----------|:-------:|
| `gemma4:latest` | "parameter updates vs. context windows... profoundly useless" → practical framing | 5/5 |
| `qwen3.5:9b` | "Fine-tune for capabilities, RAG for knowledge retention... functionally t..." | 4/5 |
| `qwen3.5:9b-mlx` | "'textbook' answer... leans heavily on the idea that RAG is cheap... framing them as..." | 4/5 |
| `qwen3.5:4b-mlx` | "strip away the 'it depends' nonsense... RAG is for data efficiency" | 4/5 |
| `gemma4:e2b-mlx-bf16` | "knowledge dynamism versus generalization fidelity" — clean named axis | 4/5 |
| `gemma4:e2b-mlx` | "The tradeoff isn't performance versus effort; it's knowledge dynamism vs generalization fidelity" | 4/5 |

---

**12. think=True is permanently broken for all qwen3 models. (Jun-07)**
Both 4b-mlx and 9b-mlx return 0 words with think=True under the Alice prompt. Not slow, not degraded — empty. gemma4:latest is unaffected (108w → 127w, minimal delta). Production config must remain think=False. This rules out think mode as a research uplift option for qwen3 models entirely.

**13. Standard suite ranks models by capability; Aptitude Battery decides role assignment. (Jun-07)**
4b-mlx ranked #3 in the standard suite (2/3 reasoning, 5/7 JPEG) but wins Battery B on personality. The standard suite is the capability gate; aptitude decides which capable model gets which role. A model can rank lower on capability but win on role fit — that's a valid outcome, not a contradiction.

**14. gemma4:latest is a tool/coding worker, not a personality worker. (Jun-07)**
Battery B showed gemma4:latest produces academic, bullet-listed responses without Alice voice — even with the full prompt, it's doing impression not character. Its standard suite strength (research depth, persona score 5/5) reflects prompt-following quality, not genuine character hold. Keep it in the tool/coding role; don't assign it to chat-default paths.

---

## Notable Protocol Issues (track for next benchmark design)

1. **Travel problem too hard for Alice-prompted models** — Consider replacing with a cleaner relative-motion problem without the staggered departure time, OR accept that 1/7 pass rate is the real ceiling for this test.

2. **think=False changes model character** — The benchmark now measures production behavior, not theoretical potential. Some scores (especially 9b GGUF depth) would be much higher with think enabled.

3. **tok/s unreliable in sequential runs** — Next benchmark: either run models independently or add 5-minute cooling pauses between models.

4. **Alice personality in benchmarks** — The Alice system prompt creates Socratic stops and verbose preamble that doesn't transfer to "correct answer". Worth running a LEAN PROMPT vs ALICE PROMPT split benchmark to isolate the personality cost.
