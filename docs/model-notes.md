# Model notes — vendor guidance vs. what the harness sends

**Compiled 2026-08-29** from four sourced research briefs (official model cards, Ollama docs/source/release
notes, GitHub issues, Unsloth guides, community reports) plus local verification on **Ollama 0.33.1**
(`ollama show --modelfile/--parameters`, `/api/chat` probes). Claims marked *unverified* could not be
confirmed from a primary source. Companion to `docs/think-spec.md` (v3 think-aware protocol).

Roster covered: gemma4 (12b, 12b-mlx, 26b-mlx, e4b-mlx, 31b-cloud) · gpt-oss (20b, 120b-cloud) ·
granite4.1 (3b, 8b) · granite4.2 (3b, 8b, 30b) · ministral-3:3b · devstral:24b · falcon3:10b ·
llama3.2:3b · fluxassistant · minicpm5:1b · minicpm-v4.6:1b · ornith:9b · ornith-1.5:9b ·
qwen2.5vl:3b · qwen3-vl:4b · qwen3.5:{4b,9b}-mlx · qwen3.8:27b-mlx · deepseek-r1:{8b,14b} ·
deepcoder:{1.5b,14b} · bonsai-27b (two uploads).

**Added 2026-09-02** (candidates, not yet on the roster): **lfm2.5:8b** and **lfm2:24b** — researched ahead of onboarding
after its 2026-06-03 v1 drop was found to predate Battery D, the budget-retry and v3. Local verification
still pending; every Ollama-specific claim below is marked *unverified*.

---

## 1. Cross-cutting facts (Ollama mechanics, verified)

| Fact | Consequence for BenchLLAMA |
|---|---|
| **The harness sets NO sampling parameters** — only `num_ctx`, `num_predict`, `think`. Sampling comes from each tag's Modelfile, else Ollama's defaults (temp 0.8, top_p 0.9, top_k 40, min_p 0). | Cross-model comparisons mix `qwen2.5vl` at temp **0.0001**, `ministral` **0.15**, ornith/deepseek/deepcoder **0.6**, gemma4/granite4.2/qwen3.5 **1.0**, and five tags with *nothing* baked (0.8). See §3. |
| **`repeat_penalty` default 1.1 → 1.0 in Ollama 0.32.10** (2026-08-12). | A silent sampling change for every tag without a baked value (granite4.1, devstral, falcon3, llama3.2, deepseek-r1:14b, gpt-oss, minicpm5, ornith-1.5, minicpm-v4.6, qwen3-vl). Runs before/after 0.32.10 are not sampling-identical → a legitimate `--check-runtime` trigger, unlike the 0.33 cache work. |
| **`think` omitted = thinking ON** for every thinking-capable model (docs; qwen3.5 parser `thinkingEnabled=true` when nil). Thinking tokens count against `num_predict`. | Consumers that omit `think` get a full trace and, at a tight budget, empty content. The v3 probe's `default_thinks` flag exists for exactly this; manifests need a `think` value per assignment. |
| **Level semantics live in the renderer, not the API.** `api.ThinkValue.Bool()` is true for any of low/medium/high/max; a renderer that never reads the string treats every level as `true`. | qwen3.5 → levels ≡ true (measured). qwen3.8 → `low`/`medium`/`xhigh` are real and **omitted = xhigh**. granite4.2 → only `low` is special (template `reasoning_effort == "low"`). gemma4 → booleans only. gpt-oss → only levels, cannot disable. DeepSeek/DeepCoder Jinja → `enable_thinking` ignored, always thinks. Exactly the heterogeneity `think_probe.py` measures. |
| **Runtime split by tag**: only `-mlx` tags (gemma4, qwen3.5, qwen3.8) use the MLX runner; everything else is llama.cpp GGUF — including gpt-oss (no MLX tag), all Granite/Mistral/Falcon/Llama/MiniCPM/ornith/bonsai. | MLX release-note items (MTP, NVFP4 fixes, cache leak) only reach the three MLX families. Rule #12's MLX-vs-GGUF prefill finding is a runtime comparison, not a weights comparison. |
| **Determinism**: seed + temp 0 is not reproducible in general on Ollama; MLX float paths diverge with matrix shape/reduction order. | Treat MLX σ as partly runtime noise; the probe groups levers by *behaviour* for this reason. |
| Ollama's default `num_ctx` is VRAM-tiered (4k < 24 GiB, 32k 24–48, 256k ≥ 48). | Rule #1's explicit 16384 stays correct policy; agents/coding guidance is ≥ 64k. |
| Greedy decoding is explicitly discouraged for gpt-oss (81% "reasoning blackholes" at temp 0, arXiv 2509.23882) and Qwen3.x ("endless repetitions"). | Confirmed locally 2026-08-29: at temp 0 qwen3.5:4b-mlx and granite4.2:3b looped past 16k tokens on bat_ball. `think_probe.py` v2 therefore sends **no temperature** (Modelfile sampling + seed) — the batteries' own conditions. |

## 2. Per-model reference

`Baked` = what the Ollama Modelfile sets (local `ollama show`); `Vendor` = the model card's recommendation
(direct = non-thinking, think = thinking). `Lever` = how `think` actually behaves. Budgets are vendor or
measured output floors so the answer is not starved.

| Tag | Arch · runtime | Baked | Vendor direct / think | Lever behaviour | Tools | Floors | Traps |
|---|---|---|---|---|---|---|---|
| gemma4:12b / 12b-mlx / e4b-mlx | gemma4 · GGUF / MLX nvfp4 | 1.0 / 0.95 / k64 | same for both modes (Google); low temp *hurts* | bool only; omitted = on; ~300 tok on simple items, ×3.7 reply length in think mode | native; 0.32.1 fixed system+think:false+tools parse (#15539) | direct ≥1024, think ≥4096 | thinking **lowers strict instruction-following** (Multi-IF, arXiv 2603.23160) — matches our e4b direct 4/4 vs think 2/4 smoke |
| gemma4:26b-mlx | 26B-A4B MoE · MLX | 1.0 / 0.95 / k64 | same | as above | as above | as above | MoE quantizes badly (Q8 KL 0.544 vs dense 31B 0.163); false-negative refusals; open MLX QAT loader bug #16740 |
| gemma4:31b-cloud | cloud BF16 | — | same | as above | — | — | no local digest → runtime fixes never re-trigger |
| gpt-oss:20b | gptoss MXFP4 · **GGUF** | temp 1 only → top_p 0.9/k40 by default | OpenAI: 1.0 / top_p **1.0** / top_k **0** | only low/medium/high; bools ignored; cannot disable; omitted ≈ medium (*unverified*) | harmony `commentary` channel; several open 2025 parse issues | low ≥2048, medium ≥8192, high ≥16–30k | never greedy (loops); avoid `format`; empty final channel = starvation |
| gpt-oss:120b-cloud | cloud | — | as 20b | as 20b | | | as 20b |
| granite4.1:3b / 8b | granite · GGUF | **nothing** (0.8/0.9/40) | IBM none; Unsloth temp 0 / top_p 1 / k0 for deterministic IF | none — `think:true` → 400 | JSON `<tool_call>` | ≥2048 | repeat_penalty flipped at 0.32.10 |
| granite4.2:3b / 8b / 30b | granite · GGUF | 1.0 / 0.95 | IBM: 1.0 / 0.95 for ALL modes | template: omitted = on; **only `"low"` distinct**; medium/high/true identical; low appends `{reasoning effort: low}` | XML `<function=…>`; agentic RL only 8B/30B | IBM: think 8192, low 4096, off 2048, tools 4096 | 3B full-think runs past 4096 (measured); direct arm gets bat_ball WRONG on 3b/8b |
| ministral-3:3b | mistral3 · GGUF | temp 0.15 | Mistral: < 0.1 production; keep tool set minimal | none | `[AVAILABLE_TOOLS]`; **`format` overrides tools** (#13750); empty output with >2 tools at large num_ctx (#13328) | ≥2048 chat, 4096 tools | **injects a ~300-word Le Chat system prompt when none is sent** (harness always sends one — safe) |
| devstral:24b | llama · GGUF | **nothing**; OpenHands prompt baked as SYSTEM | Mistral: 0.15 | none | `[TOOL_CALLS]`; open parse issues #11470, #11296 (template already uses $lastUserIndex) | ≥4096; agents want num_ctx ≥32k | **this tag is Devstral 1 (2505)**; Small 2 is `devstral-small-2:24b` (temp 0.15 baked, vision) |
| falcon3:10b | GGUF | nothing | none (empty generation_config) | none | **no `.Tools` template** → no tools cap | ≥2048; num_ctx ≤32768 | random empty responses — ollama #8157 open since Dec 2024 → budget-retry is the mitigation |
| llama3.2:3b | llama · GGUF | stops only | Meta: 0.6 / 0.9 | none | template **instructs** a call whenever tools are attached (#9947/#6127) | ≥256 | put the one-word rule in system AND user; never attach tools on classify turns |
| fluxassistant (llama3.2 ft) | llama Q8 · GGUF | FLUX SYSTEM baked; no `.Tools` branch | 0.6 / 0.9 | none | reports tools, passed calculate (*mechanism unverified*) | ≥512 | harness system prompt replaces the persona → plain Llama 3.2 3B |
| minicpm5:1b-q8_0 | llama · GGUF (local HF import) | stops only | OpenBMB: direct 0.7 / 0.95 · think 0.9 / 0.95 | omitted = on; toggle works despite legacy template | XML; SGLang parser recommended — Ollama best-effort | ≥256 fast, ≥2048 think | our classify 0.2 — fast, not a classifier |
| minicpm-v4.6:1b | qwen35 + SigLIP2 · GGUF | nothing | — | omitted = on; toggle works | tools/thinking/vision | ≥512 | `max_slice_nums`/downsample not exposed in Ollama; OCR temp 0–0.2 |
| ornith:9b | **qwen35** · GGUF | 0.6 / 0.95 / k20 + SYSTEM ("think step by step… be concise") | card: coding 0.6/0.95/20; general 1.0 + presence 1.5 | opens with `<think>` by default; `think:false` clean | `<tool_call>` XML | ≥1024 code, ≥4096 think | DeepReinforce agentic coder (not obscure); harness prompt overrides the baked persona (intended) |
| ornith-1.5:9b | qwen35 + vision · GGUF | **nothing** (0.8/0.9/40) | same as 1.0 | same | same | same | card recipe not baked → set 0.6/0.95/20 explicitly |
| qwen2.5vl:3b | qwen25vl · GGUF (llama.cpp compat) | temp **0.0001** + SYSTEM | upstream 1e-6, repetition 1.05 | none | none | ≥256 | pre-scale docs to ~1000–1200 px; counting degrades past ~5 objects; grounding wants ≥1024 image tokens |
| qwen3-vl:4b (bare) | GGUF | 1.0 / 0.95 / k20 (= thinking preset) | instruct 0.7/0.8/20/presence 1.5 · thinking 1.0/0.95 | **thinking renderer discards `think`** (#13353) → cannot disable | | ≥512 | **pull `qwen3-vl:4b-instruct`** for a vision worker |
| qwen3.5:4b-mlx / 9b-mlx | qwen3_5 · MLX nvfp4 | 1.0 / 0.95 / k20 / min_p 0 / presence 1.5 (= thinking/general preset) | direct 0.7 / 0.8 / 20 / presence 1.5 · think 1.0 / 0.95 / 20 / presence 1.5 (coding 0.6, presence 0) | `false` prefills empty think block; **levels ≡ true** (renderer never reads the level) | XML `<function=…>`; parser fixed 0.17.x; qwen3.6 drift #16383 pending | direct ≥400, think ≥4096 (1857 measured on a riddle); Qwen says 32k | `/api/generate` ignores think:false (#14793); images push the reply into `thinking` even at think:false (#14716 open); `format` ignored with think off (#14645) |
| qwen3.8:27b-mlx | MLX nvfp4 | 1.0 / presence 0 / repeat 1 / min_p 0 | think 1.0/0.95/20/presence 0 · instruct 0.7/0.8/20/presence 1.5 | renderer: omitted → **xhigh**; `low` → low; `medium` → no instruction; `high`/`max` → xhigh; `false` → empty block; `preserve_thinking` always on | OK | low ≥2048, xhigh ≥8192 (*unverified*) | users report xhigh over-thinks trivial tasks; medium ≈33% faster; MTP GGUF tags separate |
| deepseek-r1:8b | **qwen3** (R1-0528-Qwen3-8B) · GGUF | 0.6 / 0.95 + stops | 0.6 / 0.95; system prompt supported (0528); Unsloth min_p 0.05 | Jinja ignores `enable_thinking` → **always thinks** | template has tool syntax; emission *unverified* | ≥4096 | brevity probes fail by design |
| deepseek-r1:14b | qwen2 (R1-Distill-Qwen-14B) · GGUF | **stops only → temp 0.8** | 0.6 / 0.95; **no system prompt** | always thinks; bare `<｜Assistant｜>` prompt so it may occasionally skip thinking | *unverified* | ≥6000 | set 0.6 explicitly |
| deepcoder:1.5b / 14b | qwen2 (R1-distil ft) · GGUF | 0.6 / 0.95 | 0.6 / 0.95, max_tokens ≥64k, no system prompt | always thinks (byte-identical across levers — measured) | no tool training; cap is template-derived; never emits tool_calls (measured) | ≥4096 (1.5b answers ~1.1k), ≥8192 E-hard | exclude from tool batteries |
| bonsai-27b (codecraftersllc) | **Qwen3.6-27B** Q1_0 · GGUF | 0.6 / 0.95 / k20 / num_ctx 32768 | PrismML 0.7/0.95/20/min_p 0 | Go qwen3.5 renderer → levels ≡ true | vision+tools+thinking | think ≥6144 (token-hungry, up to 14× more tokens) | 89.5% of FP16 avg; tool-calling 80→66, IF 78.5→65.7, vision 72.6→59.6; 25 t/s (1-bit cuts memory, not compute) |
| bonsai-27b (oamazonasgabriel 1bit-8gbGPU) | same · Jinja template | 0.7 / 0.8 / repeat 1.05 | same | Jinja `enable_thinking` — honoured? *unverified* | no vision cap | same | different upload, different sampling |
| **lfm2.5:8b** *(MEASURED 2026-09-02)* | **LFM2.5-8B-A1B** — 8B / ~1.5B-active MoE on the LFM2 **hybrid conv+attention** backbone · GGUF, 5.2 GB, 128K | **temp 0.2 · top_k 80 · repeat_penalty 1.05** — verified via `ollama show --parameters`; **matches Liquid's recommendation exactly** (this tag needs no sampling correction) | Liquid: 0.2 / top_k 80 / rp 1.05, same for both modes; no top_p or min_p given | `false` = genuinely off · `absent`/`low`/`medium`/`high`/`true` all collapse to one thinking class (≤262 think-tok). `direct_lever: false`, `operating_lever: low`. ⚠ **thinking text lands in `content`, NOT the `thinking` field, even at `think:false`** — so `thinking_chars` under-reports and a tight budget is spent on prose | **Pythonic** list between `<\|tool_call_start\|>`/`<\|tool_call_end\|>` — **Ollama HAS an lfm2 renderer and parses it into proper `tool_calls` (verified)**. ⚠ **But emission is ~20% reliable**: 2/12 (harness defaults) and 3/12 (explicit vendor sampling) with a system prompt + tools attached | think ≥8192; vendor says leave `num_predict` unset | **Tool emission — not parsing — is the blocker.** Battery D `chain_3` gets 1 step where a working model gets 3; `calculate` ✗ on both arms. Sampling is NOT the cause (2/12 vs 3/12 is noise). Also: G composite **0.438**, clean-**1024**, early-needle recall 0.000 — on a 128K-advertised model |
| **lfm2:24b** *(candidate)* | **LFM2-24B-A2B** — 24B / **2B-active** MoE, same hybrid backbone · GGUF, 14 GB, **32K only** | *unknown* | Liquid (family): 0.2 / top_k 80 / repeat 1.05 *(unconfirmed for this size)* | **no `thinking` cap** — Ollama lists text only | **no `tools` cap** — Ollama lists text only | ≥2048 | **NOT a fleet-anchor candidate: no tools, no thinking, and 32K vs lfm2.5's 128K.** The higher version number is the *less* capable tag for our purposes — do not conflate |

## 3. What the harness sends vs. what is recommended — the gaps

1. **Sampling is uncontrolled.** Every battery compares models at their own Modelfile temperature
   (0.0001 → 1.0), and five roster tags run at Ollama's 0.8 default because the uploader baked
   nothing (`ornith-1.5`, `deepseek-r1:14b`, `devstral`, `granite4.1`, `llama3.2`). Vendor
   recommendations also differ **by mode** (Qwen: direct 0.7/0.8 vs think 1.0/0.95; gpt-oss: top_p 1.0 /
   top_k 0, which Ollama does not bake). Options:
   - (a) *Respect Modelfile defaults* (today): measures what a consumer gets out of the box — but the box
     is wrong for `ornith-1.5`, `deepseek-r1:14b`, `devstral`, `gpt-oss` and for Qwen's direct arm.
   - (b) *Vendor presets per arm*: a `sampling` block in `think_profile` (direct/think) sourced from this
     table, sent by `apply_think`. Consumers then read the same block from the export. Recommended.
   - (c) *Harness-wide pin* (e.g. temp 0.6): comparable, but explicitly against Gemma/Granite/Qwen guidance
     ("low temp hurts"; greedy loops).
   Not a `BATTERY_REVISION` matter until chosen; if (b), it is a full re-measure trigger — fold it into
   the pending v3 run rather than after it.
2. **Registry corrections** (cheap, do before the run): `deepseek-r1:8b` is the Qwen3-8B 0528 distil;
   `deepseek-r1:14b` has no temperature; `devstral:24b` is Devstral **1** — decide whether
   `devstral-small-2:24b` was intended; `qwen3-vl:4b` should be `qwen3-vl:4b-instruct` for a vision
   worker; `ornith-1.5:9b` bakes no params; both `bonsai` uploads are Qwen3.**6** with different sampling
   and templates (only one belongs in a ranking).
3. **`--check-runtime` should fire for 0.32.10** (repeat_penalty) — the current major.minor gate
   ignores it. Either pin `repeat_penalty 1.0` in every call (makes runs comparable regardless of Ollama
   version) or record it in the fingerprint. Pinning is simpler and matches every Qwen/Gemma card.
4. **Think probe temperature — RESOLVED 2026-08-29.** Probe v1 pinned temp 0 and qwen3.5:4b-mlx /
   granite4.2:3b looped past 16k tokens on bat_ball (the documented greedy-decoding repetition), reading as
   `think_unbounded`. Probe v2 sends no temperature (Modelfile sampling + seed 42) — the batteries' own
   conditions; the ±25%/behaviour grouping absorbs the sampling noise.
5. **Thinking vs instruction-following is a real trade-off**, not a harness artifact: Gemma-4-31B's
   thinking mode lowered Multi-IF while lifting math (arXiv 2603.23160); our e4b smoke (direct 4/4 vs
   think 2/4 on standard reasoning) is the same shape. The two standard-suite arms will show this
   fleet-wide; the operating-point rule may want a "think arm must not lose to direct on the standard
   suite" clause (open decision, `docs/think-spec.md` §8).
6. **Budget floors** per family (§2) are all ≥ the harness's base `max_tokens` only after the v3
   allowance; the ×8 retry remains the backstop. DeepSeek/DeepCoder "brevity" tests (20–50 tokens)
   fail by design — that is a real property, not a bug.
7. Already safe by construction: every harness call sends a system prompt (Ministral's hidden Le Chat
   prompt never triggers); the harness never sends `format` (Ministral/Qwen `format`-vs-tools bugs).

8. **Self-gated thinking breaks the probe's grouping assumption** (new 2026-09-02, LFM2.5).
   `think_probe.py` classes levers by *identical output* over 4 tiny items. A model that decides
   **per prompt** whether to think can return non-thinking output at every lever on easy items,
   collapsing all six classes into one and producing a `direct_lever` / `operating_lever` pair that
   describes nothing. LFM2.5 is the first roster candidate to document this explicitly ("simple prompts
   may skip reasoning entirely — this is expected behavior, not an error"), but it is a general shape,
   not an LFM quirk. The probe set needs at least one item hard enough to *force* the reasoning block
   before lever classes are meaningful for such a model. **Tell:** every class identical with
   `thinking_chars` 0 throughout — read that as "probe inconclusive", not "model has no think lever".
   ⚠ **CORRECTION 2026-09-02 — this did NOT happen on LFM2.5, and the prediction was wrong.** The probe
   separated its levers cleanly: `false` genuinely disables thinking and is the ONLY class that fails
   `classify_1word`, while every other lever thinks (≤262 tokens) and passes 4/4. Self-gated thinking is
   still a real hazard for a probe that groups by output identity, but LFM2.5 is not an instance of it —
   the tiny items were hard enough. Keep the watch, drop the worked example.
9. **Vendors that say "do not cap output" cannot be honoured.** Liquid tells operators to leave
   `max_tokens` unset so the reasoning block is not truncated mid-stream; the harness always sends
   `num_predict`. The v3 allowance plus the ×8 budget retry is the mitigation, and this is precisely the
   family where the retry earns its keep.

## 4. Family notes (condensed)

**Gemma 4** — Google: 1.0/0.95/64 for all sizes; community found sub-1.0 temperature *degrades* coding
and reasoning. `<|think|>` at the start of the system turn; Google added an empty-thinking token to the
12B/26B/31B templates to suppress ghost thoughts when off; strip prior-turn thoughts from history except
between tool calls. 5:1 local:global attention; int8 KV at 32k costs ≤1.1 GB. MLX: nvfp4 halves 4-bit loss;
MTP +≈90% decode on 12B without changing output. The 26B-A4B MoE quantizes badly and its Q4_K_M varies
2× between uploaders. Failure mode is reluctance (false-negative refusal), not confabulation; a viral
"MoE vs dense" claim was retracted as a `max_tokens` budget bug. Ollama 0.32.1 improved Gemma 4 tool
calling / multi-turn reasoning and fixed the MLX recurrent cache leak; 0.32.5 fixed an NVFP4 Metal quality
bug; a gemma-4 MLX QAT/KV-shared loader bug is open (#16740).

**gpt-oss** — harmony format, three channels; default medium reasoning; OpenAI recommends temp 1.0 /
top_p 1.0 / no top_k — Ollama bakes only temperature, so send top_p/top_k yourself. Ollama: only levels,
bools ignored, cannot disable, no `minimal`. Together AI advises ~30k `max_tokens` at high. Greedy
decoding → 81% reasoning blackholes. SimpleQA hallucination 0.914. Drop prior CoT after a `final`
message; pass CoT back between tool calls. All known Ollama tool-parse issues are 0.11-era; no 0.32/0.33
note mentions gpt-oss.

**Granite 4.2** — IBM: 1.0/0.95 everywhere; budgets 8192 think / 4096 low / 2048 off / 4096 tools;
RLHF reasoning-length penalty. Template: `enable_thinking` default True; `low_effort = reasoning_effort ==
"low"` appends `{reasoning effort: low}` to the last user turn; `truncate_history_thinking=True`. XML tool
format; agentic RL only on 8B/30B; 128K native, 512K extension. **Granite 4.1** has no thinking mode and
no baked sampling.

**Mistral** — Ministral-3: temp 0.15 baked (Mistral says < 0.1 in production); hidden Le Chat system
prompt when none is sent; `format` overrides tools; keep ≤ 2 tools per call. Devstral: `devstral:24b` =
Devstral 1 with the OpenHands prompt baked as SYSTEM and no temperature (Mistral recommends 0.15);
Devstral Small 2 (`devstral-small-2:24b`) generalizes beyond OpenHands, adds vision, bakes 0.15.

**Falcon3** — no tools template (hence no `tools` cap), no sampling, random empty responses (#8157,
open since 2024), 32K, TII license.

**Llama 3.2** — Meta 0.6/0.9; the Ollama tool template *instructs* a call whenever tools are attached
(the over-call mechanism); Meta's own 1B/3B guidance uses a different `[func(a=1)]` syntax and no built-in
tools. Strongest router in our fleet (classify 1.0 / escalation 0.0 / 89 t/s). fluxassistant = a
Llama-3.2-3B FLUX prompt-writer fine-tune (Unsloth/TRL; the "uncensored" delta is undocumented).

**MiniCPM** — minicpm5 (1.08B, Llama arch): OpenBMB direct 0.7/0.95 · think 0.9/0.95, num_ctx 8192,
XML tools (SGLang parser recommended); our local import got a legacy template yet thinks by default.
minicpm-v4.6: SigLIP2-400M + Qwen3.5-0.8B; slice/downsample knobs not exposed in Ollama.

**Ornith** — DeepReinforce's agentic-coding family (MIT, 256K); the 9B is post-trained on **Qwen 3.5**
(arch `qwen35`), opens with `<think>` by default; 1.5 adds vision + YaRN to ~1M. Card: coding
0.6/0.95/20; general temp 1.0 + presence 1.5. `ornith:9b` bakes 0.6/20/0.95 + a system persona;
`ornith-1.5:9b` bakes nothing.

**Qwen3.5 / 3.8** — Qwen3.5 card: think/general 1.0/0.95/20/presence 1.5, think/coding 0.6/presence 0,
non-think 0.7/0.8/20/presence 1.5 (the "reasoning tasks" non-think preset contradicts itself, discussion
#51 unresolved); greedy decoding → endless repetition; presence > 1.5 → language mixing; 32k output
recommended. Ollama's qwen3.5 renderer prefills an empty think block for `false` and never reads the
level. Qwen3.8: `reasoning_effort` xhigh (default) / medium / low, `preserve_thinking` always on in
Ollama, non-leading system messages normalized 0.32.15; users report medium ≈33% faster than xhigh.
MTP: 0.32.6 auto speculative decoding on MLX for Qwen3.5; our matched-quant test: MTP +30–40% decode,
quality a wash. Static YaRN taxes short prompts — don't enable it for a 16k harness.

**Qwen-VL** — qwen2.5vl:3b: near-greedy baked; llama.cpp compat path (clip limits 8–4096 image tokens);
pre-scale documents to ~1000–1200 px; ask for a bbox list rather than "how many". qwen3-vl bare tag uses
the *thinking* renderer that discards `think` — use `:4b-instruct`.

**DeepSeek-R1 distils / DeepCoder** — 0.6/0.95 (R1: no system prompt; 0528 supports one); Jinja templates
without an `enable_thinking` branch → always think; bare `<｜Assistant｜>` generation prompt; DeepCoder:
max_tokens ≥ 64k, no tool training. 14b bakes no temperature.

**Bonsai-27B** — PrismML Q1_0_g128 of Qwen3.6-27B, 1.125 bpw, 3.9 GB; retains 89.5% of FP16 average but
tool-calling/IF/vision drop 14–16 points and it needs up to 14× more tokens with reasoning on;
recommended 0.7/0.95/20/min_p 0; needs Ollama ≥ 0.32.5. Our two uploads differ in template and sampling.

**⚠ Liquid: two distinct Ollama tags, and the bigger number is the weaker one for our lanes.**
`lfm2:24b` = **LFM2-24B-A2B** (24B / 2B active, 14 GB, **32K**, Ollama lists **text only — no tools, no
thinking**). `lfm2.5:8b` = **LFM2.5-8B-A1B** (8B / ~1.5B active, 5.2 GB, **128K**, **tools + thinking**).
`lfm2.5` is *not* a point release of `lfm2` in the way the names imply — it is the smaller, more capable
tag, and the only one of the two that can enter the tool batteries at all. Same disambiguation shape as
`devstral:24b` (Devstral 1) vs `devstral-small-2:24b` in §3.2. Both are MoE, so both are useful data
points for the MoE-vs-dense throughput question, but only 2.5 is an anchor candidate.

**Liquid AI / LFM2.5** *(MEASURED 2026-09-02, full suite + F-elastic + H, 46m32s)* — a **hybrid
conv + attention** backbone, not a transformer stack. Ollama's `lfm2.5:8b` is **LFM2.5-8B-A1B**: 8B total,
~1.5B active MoE, 128K context, 5.2 GB. **The tag bakes Liquid's exact recommendation** (temp 0.2 /
top_k 80 / repeat_penalty 1.05, verified) — one of the few roster tags needing no sampling correction,
so it is NOT an instance of the §3.1 uncontrolled-sampling gap. All experts stay resident (Liquid sizes
BF16 at ~17 GB): the MoE buys decode cost, not footprint — and it delivers, at **145.3 t/s direct**, the
fastest local model on the roster.

**What it is good at (measured):** honesty and instruction-following. Battery **H 0.667, profile
`discerning`** (fake 0.667 / real 0.667) — the best *local* honesty on the roster and 4th overall;
F-elastic **robust and stable** (instruction-adherence 0.99, prompt-σ 0.036); Battery F 0.871.

**What kills it: tool-call EMISSION, not parsing.** ⚠ The pre-measurement entry flagged the Pythonic
format as the #1 risk. **That was wrong** — Ollama ships an lfm2 renderer and turns
`<|tool_call_start|>`-wrapped Pythonic calls into proper `tool_calls`; verified directly. The real defect
is that the model usually does not emit one at all: **2/12 at harness defaults, 3/12 with explicit vendor
sampling** (system prompt + tools attached, identical prompt). Indistinguishable rates, so sampling is not
the cause. `calculate` ✗ on BOTH arms; Battery D `chain_3` returns 1 step where a working model returns 3.
Not tunable — treat "native function calling for agentic workflows" as unsupported at these odds.

**Other measured weaknesses:** Battery G composite **0.438**, clean depth **1024**, early-needle recall
**0.000**, multihop 0.167 — on a model advertising 128K context, the second-worst long-context result on
the roster. Battery E 0.812 but **E-hard 0.253** → no `coder`. Thinking text lands in `content` rather
than the `thinking` field even at `think:false`, so `thinking_chars` under-reports for this family.

**Think levers separate cleanly** (contra the §3.8 worry): `false` is genuinely off and is the only class
that fails `classify_1word`; `absent`/`low`/`medium`/`high`/`true` collapse into one class at ≤262 think
tokens. `direct_lever: false`, `operating_lever: low`.

**VERDICT: the 2026-06-03 "failed all tool calls" drop was CORRECT, and re-testing it was still worth
doing.** The old verdict was reached under a protocol now known to be broken, so it was untrustworthy
evidence for a true conclusion — the re-test converted a guess into a measurement, and produced a usable
honesty datapoint besides. Both hypotheses in the pre-measurement entry (budget starvation; unparsed
Pythonic calls) were falsified. Keep the model only if a use case wants a fast, honest, prompt-adherent
local chat model with **no tool use**; it is not a fleet-anchor candidate.

⚠ A dedicated **LFM2.5-1.2B-Thinking** (<1 GB) exists and is unmeasured. Community uploads
(`hadad/LFM2.5-1.2B`, `maternion/lfm2.5`, `oamazonasgabriel/lfm2-1.2b-tool`) ship their own templates and
sampling — benchmark the official library tag, not a mirror. ⚠ `lfm2:24b` remains a different, weaker tag
for our lanes (no tools, no thinking, 32K) — see the disambiguation above.
## 5. Ollama 0.32.x–0.33.x entries that matter here

0.32.1 Gemma 4 tool calling + multi-turn reasoning; MLX recurrent cache leak fixed · 0.32.3 GLM tool
calls dropped at end fixed · 0.32.4 Qwen3 MoE decode fix · 0.32.5 NVFP4 Metal quality fix · 0.32.6
Qwen3.5 MTP speculative decoding on MLX; `/v1` `finish_reason: length` on truncation; image-gen removed ·
**0.32.10 `repeat_penalty` default 1.1 → 1.0**; NVFP4 prefill +7–8% · 0.32.12 Qwen3.8 27B (+mlx) ·
0.32.13 Qwen3.8 developer instructions · 0.32.14 qwen renderer tolerates non-leading system · 0.32.15
Qwen3.8 system-message normalization; metadata cache halves TTFT · 0.33.0 prefill restore points; MLX
update · 0.33.1 Qwen3.8 Flash Next on MLX; mlxrunner structured output; Metal load-timeout fix. No entry
mentions gpt-oss, deepseek-r1, deepcoder, Llama 3.2, MiniCPM, ornith, Granite, Mistral, Falcon or the VL
models.

## 6. Sources (primary)

Ollama: docs.ollama.com/{capabilities/thinking, context-length, modelfile}; github.com/ollama/ollama
releases v0.32.0–v0.33.2; source `model/renderers/qwen35.go`, `api/types.go`, `llm/server.go`,
`llm/llama_server.go`, `x/mlxrunner/server.go`; issues #586 #5321 #6127 #8157 #9947 #10899 #11296 #11470
#11725 #11751 #11781 #11867 #11991 #12004 #12187 #12203 #12589 #12692 #12906 #13328 #13334 #13353 #13519
#13750 #14493 #14645 #14716 #14745 #14793 #14798 #14820 #15260 #15539 #15831 #16383 #16456 #16740.
Vendors: ai.google.dev/gemma (model_card_4, capabilities/thinking); arXiv 2607.02770 (Gemma 4 TR),
2603.23160 (UniDial), 2509.23882, 2510.01266, 2508.10925, 2510.04401; huggingface.co openai/gpt-oss-20b
(+120b discussion #21); developers.openai.com harmony guide; ibm-granite/granite-4.2-3b (+ blog, GitHub),
granite-4.1-8b; mistralai/Ministral-3-3B-Instruct-2512, Devstral-Small-2505, Devstral-Small-2-24B-2512;
tiiuae/Falcon3-10B-Instruct; meta-llama/Llama-3.2-3B-Instruct + Meta prompt-format doc;
openbmb/MiniCPM5-1B (+ ollama.md), MiniCPM-V-4_6; deepreinforce-ai/Ornith-1.0-9B, ornith-ai/Ornith-1.5-9B;
Qwen/Qwen3.5-9B (+ #51), Qwen3.8-27B (+ #113), Qwen3-VL-4B-Instruct, Qwen2.5-VL-3B-Instruct;
deepseek-ai/DeepSeek-R1, DeepSeek-R1-0528; agentica-org/DeepCoder-14B-Preview, DeepCoder-1.5B-Preview;
prismml.com Bonsai-27B, prism-ml/Bonsai-27B-gguf. Guides/community: unsloth.ai docs (gemma-4, gpt-oss,
granite-4.1, ministral-3, deepseek-r1, qwen3.5); localbench.substack.com; kaitchup.substack.com;
adityakarnam.com (MLX determinism); antekapetanovic.com; famstack.dev; zolotukhin.ai; docs.together.ai.
Liquid AI (added 2026-09-02): docs.liquid.ai/lfm/models/lfm25-8b-a1b; liquid.ai/blog/lfm2-5-1-2b-thinking-on-device-reasoning-under-1gb;
recipes.vllm.ai/LiquidAI/LFM2.5-8B-A1B; docs.vllm.ai/en/latest/api/vllm/tool_parsers/lfm2_tool_parser;
ollama.com/library/lfm2.5; ollama.com/library/lfm2. ⚠ huggingface.co/LiquidAI/LFM2.5-8B returns 401 to unauthenticated fetches —
the sampling/thinking/tool facts above come from Liquid's own docs and the vLLM recipe, NOT the HF card;
re-check the card directly when onboarding.
