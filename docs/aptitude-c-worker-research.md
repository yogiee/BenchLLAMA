# Aptitude Battery C — Worker Research

Six tests targeting worker models in a research role. Measures depth of coverage, synthesis quality, the effect of thinking mode on research output, and where performance plateaus as context and token budget grow.

**Models:** Research-qualified only (JPEG ≥ 5/7 OR rag_finetune ≥ 4/5, tool ✓ from standard suite), further filtered by `--capable-only` (must have passed `calculate` in the most recent standard run).  
**Parameters:** `num_ctx=16384` baseline (overridden in C4), `think` varies per test.

---

## Reference prompts

### JPEG prompt (used in C1, C4, C5, C6)
```
Give a concise, not-too-technical but detailed comparison of JPEG, JPEG-2000, and JPEG-XL formats.
```

### Signal detection (7 signals, same as standard suite)

| Signal | Trigger |
|--------|---------|
| `jpeg_compression_tech` | DCT, "discrete cosine", "8×8", or "block" |
| `jpeg2000_wavelet` | "wavelet" |
| `lossless` | "lossless" |
| `transparency_alpha` | "transparency", "alpha", or "transparent" |
| `browser_support` | "browser", "chrome", "firefox", or "safari" |
| `jpeg2000_niche` | "medical", "dicom", "cinema", or "archiv" |
| `jxl_recompress_trick` | "recompress", "transcode", "re-encode", "existing jpeg", "losslessly transc", or "backward compat" |

Pass threshold: ≥ 4 signals. Max score: 7.

---

## C1 — JPEG signals with think toggle

The JPEG prompt run twice — `think=False` then `think=True`. Establishes whether thinking mode improves research coverage for this model.

**Token cap:** 1500 per run  
**Output fields per run:** signal score (0–7), word count, wall time, tok/s, `think_diagnosis`.  
**Delta:** `think_on.score − think_off.score` reported as Δ.

**What it tells you:**
- Δ > 0: thinking mode helps this model find deeper signals
- Δ = 0: thinking mode has no effect on coverage
- Δ < 0: thinking mode degrades output (unusual but possible — think tokens consume budget that could have been used for content)
- `think_empty` or `think_block_only` diagnosis: model cannot use think mode at all — do not use `think=True` in production for this model

---

## C2 — RAG deep

Extended version of the standard suite `rag_finetune` test. Asks for five concrete examples per approach rather than just the tradeoff framing.

**Prompt:**
```
What's the real tradeoff between RAG and fine-tuning when adapting an LLM to a new domain?
Give 5 concrete examples for each approach — real use-cases where one clearly wins.
Skip the textbook answer.
```

**Token cap:** 2000  
**Auto-detection:**
- `rag_example_hits`: regex count of "rag"/"retrieval" appearing near "example"/"case"/"scenario"
- `ft_example_hits`: regex count of "fine-tun"/"finetuning" appearing near "example"/"case"/"scenario"
- `has_tradeoff_framing`: presence of "tradeoff", "trade-off", "when to use", "vs", "versus", or "better when"

**What it measures:** Whether the model can produce concrete, specific examples rather than abstract characterisations. A response that discusses fine-tuning in general but never names a scenario where it clearly wins fails on `ft_example_hits` even if the prose is fluent.

---

## C3 — Synthesis from three sources

Given three source excerpts on a technical topic, produce a single synthesis paragraph that integrates the ideas rather than summarising each source separately.

**Sources provided:**
1. Anthropic blog on Constitutional AI — self-critique loop reduces need for human feedback
2. DeepMind paper on RLHF — reward hacking problem; careful reward model design required
3. Stanford HAI report — alignment technique scalability; Constitutional AI and RLAIF reduce labelling costs vs RLHF

**Prompt:**
```
You will be given three source excerpts on AI alignment.
Write a single synthesis paragraph (150–200 words) that integrates the key ideas,
identifies the central tension across sources, and states a clear conclusion.
Do not summarise each source separately.
```

**Token cap:** 600  
**Auto-checks:**
- `in_range`: word count between 100–300 (lenient range accounting for counting variance)
- `mentions_all_3`: "constitutional", "rlhf", and "scalab" all present in response
- `has_tension`: "tension", "contrast", "differ", "whereas", "while", or "however" present
- `no_src_listing`: does NOT contain "source A:", "source B:", or "source C:" (catches models that ignore the instruction and list sources sequentially)

**Why this test:** Synthesis is harder than summarisation. A model that produces "Source A says X. Source B says Y. Source C says Z." is failing the core task. The auto-checks can catch the most common failure mode (`no_src_listing`) while the remaining fields provide diagnostic signal for manual review.

---

## C4 — Context depth ladder

The JPEG prompt at three context window sizes. Tests whether the model's research coverage is sensitive to `num_ctx`.

**Levels:** 8192, 16384, 32768  
**Token cap:** 1500 at each level  
**The model is unloaded between levels** to ensure a clean context state.

**Output per level:** signal score (0–7), word count, wall time, tok/s.  
**What to look for:** A flat signal curve means context size is irrelevant for this model's research quality — 8192 is sufficient and cheaper. An improving curve (e.g. 4/7 → 5/7 → 6/7) means the model benefits from more headroom even on this medium-length task. Combine with ctx-ladder results to confirm.

---

## C5 — Token budget ceiling

The JPEG prompt at four output budgets (`num_predict`). Tests where quality plateaus as more output tokens are permitted.

**Levels:** 600, 1000, 1500, 2000 tokens  
**Output per level:** signal score, word count, wall time.

**What to look for:**
- Signal score that plateaus at 1000 or 1500: generating more tokens doesn't add coverage — the model has said what it knows. Set `num_predict` at the plateau level.
- Signal score that keeps climbing at 2000: the model is still surfacing new signals — cap is constraining quality.
- Word count grows but signal score stays flat: model is padding/repeating rather than adding depth.

This test informs the optimal `num_predict` setting for research tasks for this specific model.

---

## C6 — Think coverage with lean prompt

The JPEG prompt with `think=True` using only the lean system prompt ("You are a helpful AI research assistant."). Compared against C1's `think_on` run (which used the full prompt).

**Purpose:** Isolates whether thinking mode performance is dependent on the system prompt. Some models think better with minimal instruction overhead; others need the full prompt to activate depth.

**Token cap:** 1500  
**Delta reported:** `C6.score − C1.think_on.score`

- Δ > 0: lean prompt + think produces better coverage than full prompt + think — instruction overhead is hurting thinking mode
- Δ = 0: prompt weight doesn't matter when thinking
- Δ < 0: full prompt is necessary to direct thinking toward the right signals
