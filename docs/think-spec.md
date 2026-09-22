# Think-aware protocol (v3) — spec

**Status:** IMPLEMENTED 2026-08-28 — decisions taken (§8: policy A, no latency cap, full re-pull roster); code in `think_probe.py`, `bench_utils` (think section), `results_db` (`arm` column), `resume`, and every writer. Awaiting the v3 full re-measure.
**Supersedes:** Protocol Rule #2 (`think=False` everywhere) and the B5 / C1 / C6 / D5 think-toggle sub-tests.
**Sibling:** `docs/resume-spec.md` (content-addressed resume — this spec adds a determinant).

---

## 1. Why

Rule #2 says *"`think=True` on Qwen3 models returns 0 words (broken as of Jun-07 2026)."* That
sentence describes the **output-budget starvation bug fixed on 2026-08-24** (`bench_utils.
post_with_budget_retry`), not a model property: a hybrid-reasoning model spends the whole
`num_predict` in the `thinking` channel and the grader sees empty `content`. Every B5 `think_on`
row in the DB (ornith:9b 0w, both bonsai 0w, qwen3.5:4b-mlx 8w — all at `max_tokens=800`, all
with 20–35 s wall) predates the fix. The entire "does thinking help?" axis in `master.md` is
budget-confounded and void. `qwen3.5:4b-mlx` was dropped 07-10 partly on "no `think=True`".

Consequence: the suite has measured every hybrid-reasoning model (Qwen3.x, Granite4.2,
DeepSeek-R1 distils, gpt-oss) in the one mode those families are *not* designed around, and
LookingGlass / OllamaMCP have been selecting on those numbers.

## 2. Evidence (2026-08-28 probes, temp 0, num_ctx 16384)

### 2.1 Lever shape is family-specific

| Model | `think` omitted | `false` | `"low"` | `"medium"` | `"high"` | `true` | Lever classes |
|---|---|---|---|---|---|---|---|
| granite4.2:{3b,8b,30b} | **thinks** (default on) | off | **distinct** (76–98 tok) | ≡ true | ≡ true | full | {off, low, full} |
| qwen3.5:4b-mlx | — | off | ≡ true | — | ≡ true | on (1857 tok on bat_ball) | {off, on} |
| deepcoder:1.5b | thinks | ≡ always | ≡ always | — | ≡ always | always | {always} |
| gemma4:{e4b,12b}-mlx | — | off | — | — | — | on (~300 tok) | {off, on} |
| ornith:9b | — | off | — | — | — | on (101 tok) | {off, on} |
| gpt-oss:* | thinks | ignored | low | medium | high | ignored | {low, med, high} — cannot disable |
| ministral-3:3b, granite4.1:* | — | HTTP 400 "does not support thinking" → harness strips `think` | | | | | {off} |

"≡" = byte-identical output. The levels are **not** a portable scale: `"low"` means something
on Granite, nothing on Qwen3.5, and everything is `"low"`-or-above on gpt-oss.

### 2.2 `think:false` is a *regression* on some models, not a neutral speed mode

| Model | `false` → bat_ball | think arm → bat_ball | think tokens | wall |
|---|---|---|---|---|
| granite4.2:3b | **0.10 ✗** | `"low"` **0.05 ✓** | 76 | 1.0 s |
| granite4.2:8b | **"10" ✗** | `"low"` **0.05 ✓** | 98 | 2.8 s |
| qwen3.5:4b-mlx | **"5" ✗** (grader wants `0.05` / `5 cents`) | `true` **0.05 ✓** | **1857** | 27 s |
| granite4.2:30b | 0.05 ✓ | 0.05 ✓ | 59 | 6.7 s |
| gemma4:12b-mlx | 0.05 ✓ | 0.05 ✓ | ~300 | 9.8 s |

Under the current protocol, granite4.2:3b/8b and qwen3.5:4b-mlx **fail a test the rest of the
fleet passes 22/22** — on day one, by protocol.

### 2.3 Budget alone cannot fix it — the lever must be chosen per model

| Model / lever | bat_ball at num_predict 4096 |
|---|---|
| granite4.2:3b `true` | **4096 tokens, 11 081 chars of thinking, 90 s, no answer** |
| granite4.2:3b `"low"` | 76 tokens, correct |
| qwen3.5:4b-mlx `true` @300 / @800 / @4096 | starved / starved / correct |
| deepcoder:1.5b (any) | 1132 tokens, correct — but never emits `tool_calls` |

Full-think on a 3B Granite is unbounded on a trivial question; the reactive ×8 retry
(800 → 6400) would not save it. Think-token demand spans **76 → 1857+** across the fleet for the
*same* question, so a fixed budget is wrong for everyone.

### 2.4 Consumers today

- **OllamaMCP** `server.py:180` — `thinking: bool = False` → `options={"think": True}` (⚠ inside
  `options`; the documented top-level `think` is what the harness uses). Boolean only.
- **LookingGlass** `sidecar/main.py:85` — `config["model"].get("think", False)`, boolean; `lite.py`
  hard-codes `think:False` and strips leaked `<think>` ("GLM/Qwen3 ignore think=False").
- **Manifests** — no assignment carries a `think` value. A consumer that omits `think` gets
  **default-on thinking from granite4.2** and a starved reply at any tight `num_predict`.

## 3. Terminology

- **lever** — the value passed as top-level `think`: `absent | false | "low" | "medium" | "high" | true`.
- **lever class** — the set of levers with the same probe BEHAVIOUR: same thinking on/off, think-token demand within ±25% (≥32 tok), same per-item correctness + starvation pattern (measured, not assumed). ⚠ Not byte-identity — MLX runtimes are not deterministic across levers even at temp 0 / seed 42 (gemma4:e4b-mlx split its five identical levers into five classes under a hash rule, 2026-08-28).
- **arm** — a measurement configuration (⚠ renamed 2026-08-29: the no-thinking arm is `direct`, not `fast` — `fast` collided with the `--fast` no-cool-down mode): `direct` = the no-thinking class (or, for always-on models,
  the cheapest class); `think` = the model's **operating point**.
- **operating point** — the lever class the suite scores the `think` arm at (§4.3).

## 4. Component 1 — the think probe (`think_probe.py`, new)

Runs per model with the `thinking` capability; ~1–2 min for a ≤9B model, ~5–10 min at 27B.
Invoked by `update_registry.py` on ingest and by the orchestrator at suite start when the stored
profile is missing or stale (§7).

**Probe set** (fixed, hashed as a dataset like every other suite input):
`bat_ball` (reasoning, exact-match), `classify_1word` (brevity — must survive a 1-word answer),
`calculate` (must emit `tool_calls`), `tiny_code` (must emit a fenced block that executes).
seed 42, `num_predict` 4096 (starved items escalated 8192 → 16384), `num_ctx` 16384, **no temperature
override** — the harness sends no sampling, so batteries run at Modelfile temperatures and the probe must
too. (Probe v1 pinned temp 0 and made qwen3.5:4b-mlx / granite4.2:3b loop past 16k tokens on bat_ball —
the greedy-decoding repetition both Qwen and OpenAI document — mis-classifying them as unbounded, 2026-08-29.)

**Per lever** in `[absent, false, "low", "medium", "high", true]`: record HTTP status (400 =
unsupported), `thinking` chars, `content`, `tool_calls`, `done_reason`, `eval_count`, wall.
A compact `raw` block (per lever × item: think tokens, correct, starved, done, wall) is persisted in the
profile so `think_probe.py --rederive` can rebuild every derived field offline; a `DERIVE_VERSION` bump
re-derives on load without re-probing. Operating levers prefer explicit values (`true` > `"low"` > … >
`absent`) so a consumer sends a value rather than relying on the template default.

**Derived `think_profile`** (persisted in `models.json`):

```json
"think_profile": {
  "probed_at": "2026-08-28", "ollama": "0.32.15", "digest": "…",
  "off_supported": true,          // some lever yields zero thinking
  "default_thinks": true,         // `think` ABSENT produces thinking → consumers must pass it explicitly
  "classes": {                    // lever → class id (equivalence by output hash)
    "false": "off", "low": "low", "medium": "full", "high": "full", "true": "full", "absent": "full"
  },
  "class_stats": {                // per class, from the probe
    "off":  {"bat_ball": false, "tool": true, "brevity": true, "code": true, "think_tokens_max": 0,    "wall_p50": 0.3},
    "low":  {"bat_ball": true,  "tool": true, "brevity": true, "code": true, "think_tokens_max": 98,   "wall_p50": 1.2},
    "full": {"bat_ball": null,  "tool": true, "brevity": true, "code": true, "think_tokens_max": 4096, "wall_p50": 90, "unbounded": true}
  },
  "operating_point": "low",       // §4.3
  "operating_lever": "low"        // the literal value to send
}
```

### 4.3 Operating-point rule (deterministic, no judgement)

Among classes **with** thinking that are *bounded* (every probe answer finished under 4096) and
whose probe correctness equals the best achieved by any bounded class: within a **2× demand band** of
the cheapest such class, pick the class holding the semantically lowest explicit lever
(`low` → `medium` → `true` → `high` → `absent`); only a > 2× gap counts as a real cost difference.
(Derive v3, 2026-08-29: demand measured on four trivial items is noise-dominated — qwen3.8's `high`
= Qwen's xhigh came out 142 vs 223 tokens and would have been chosen as "cheapest".) If none is bounded → `operating_point: null` (the model has no
usable think arm; it is measured on `direct` only and flagged `think_unbounded`). If
`off_supported` is false → `direct` is the operating point too (single arm, e.g. deepcoder).

Worked results: granite4.2 → `low`; qwen3.5 → `full`(=true); gemma4 → `full`; ornith → `full`;
gpt-oss → `low` unless a higher level fixes a probe item; deepcoder → single arm `always`.

### 4.4 Declared levels and leaked thinking (derive v5, 2026-09-23)

**Declared levels.** Since Ollama **0.34.3**, `POST /api/show` returns `thinking: {values, default}`.
`update_registry.py` syncs it into `models.json` as `thinking_declared`, and the probe stores the
normalized block in the profile as `declared` (both via `bench_utils.normalize_declared`). The
declaration is a **prior and an alarm, never a substitute for probing**. Checked against the 0.33.x
profiles, it disagreed with the measurement for about half the fleet:

| model | declared | measured |
|---|---|---|
| `granite4.2:3b` | nothing | `low` is a distinct, correct class (the lever that fixes bat_ball) |
| gemma4 tags | `false, true` | 2–3 distinct thinking classes |
| `qwen3.8:27b-mlx` | `false, low, medium, xhigh` | `xhigh` was never probed |
| `lfm2.5:8b` | `false`, default `false` | thinks by default; on 0.34.3 no lever turns it off |

So the probe:
- tries THINK_LEVERS **plus** any declared level outside it (sent as the literal string, e.g. `"xhigh"`);
- records `levers_probed`;
- lists every disagreement in `declared_mismatch` (informational — the measured profile is authoritative);
- treats a **changed declaration** as a staleness reason (`declared think levels changed`), checked
  after the Ollama major.minor gate.

**Leaked thinking.** A model can think with no `thinking` channel, delivering the reasoning inside
`content` as raw `<think>…</think>`. On 0.34.3 `lfm2.5:8b` does this at `false`, `low` and absent, while
`true` is parsed correctly. Counted naively, that reads as **zero think tokens**: the probe would
classify inline thinking as `off`, and every battery at that lever would grade raw reasoning as the
answer. Such calls are now:
- marked `leaked`;
- graded on the text after `</think>` (an unterminated `<think>` is all reasoning);
- kept in their own class (the leak flag is part of the class signature);
- listed in `leaked_levers`;
- never chosen as an arm while a clean bounded class exists.

⚠ The batteries do NOT strip leaked `<think>`. The guard works by steering the arm onto a clean lever,
so a model whose every lever leaks would still be graded on raw text. The probe report says so.

## 5. Component 2 — lever-aware budgets (`bench_utils.think_budget`)

For a call to model M at lever class C with a test's `base_max_tokens`:

```
allowance   = 0 if C == "off" else clamp(4 × class_stats[C].think_tokens_max, 1024, 8192)   # ×2/512 under-shot on long items (smoke 08-28)
num_predict = base_max_tokens + allowance
num_ctx     = test_num_ctx + allowance          # the window binds first — see post_with_budget_retry docstring
```

The ×8 reactive retry stays as the backstop and keeps stamping `_budget_retry` on the result; a
retry firing on the think arm now means the probe under-estimated, which is worth seeing.

**Every result row gains** `arm`, `think_lever`, `thinking_chars`, `content_chars`, `eval_count`,
and `time_to_answer_s` (wall of the authoritative call). Decode tok/s stays Ollama's counter
(thinking tokens included — that *is* the decode cost); `time_to_answer_s` is the consumer-facing
latency and is published beside it.

## 6. Component 3 — two-arm measurement

| Where | `direct` arm | `think` arm |
|---|---|---|
| Standard suite (13 tests) | always | when an operating point exists |
| Role gate | Router: `direct` only (speed lane). Worker-*: **either** arm may satisfy a criterion; the role records which (`"role": "worker", "think": "low"`). | |
| Batteries A | `direct` (router lane) | — |
| Batteries B, C, D, E, F, F-elastic, G, H | run at the model's **selected arm** (§8 decision 1) | |
| V, EMB, I | unchanged (`direct`) | |

B5, C1, C6, D5 (think on/off sub-tests) are **retired** — they were budget-confounded and the
arm design measures the same delta properly (`standard.think − standard.direct`).

DB: add `arm TEXT NOT NULL DEFAULT 'direct'` to `results`; PK → `(run_id, model, battery, arm)`
(migration script; existing rows are `direct`). `results_db.latest()` grows an `arm` filter.
Export: `models[*].think` = `{operating_point, lever, default_thinks, off_supported}`; standard
sub-block per arm; ranking keys unchanged but each list says which arm it ranked.

## 7. Resume / provenance

`env_fingerprint` gains `think_profiles: {model: {operating_lever, digest_of_profile}}`. A changed
operating point is a determinant (the test changed for that model). The probe set is hashed like
any dataset. A profile also goes stale when its model's **declared** think levels change (§4.4). `BATTERY_REVISION` bumps for `standard, A, B, C, D, E, F, F-elastic, G, confab` →
**full fleet re-measure**, accepted (§9). ⚠ The long-lived webserver must read the new revisions
fresh (`_live_battery_revisions`) — same trap as 08-22.

## 8. Decisions pending (owner)

1. **Battery arm policy.** (A) *recommended* — standard suite both arms, batteries at the
   operating point ("what the model offers"; speed is reported, not gated). (B) every battery
   both arms (~1.6–2× run time; only pays off if consumers actually run models in both modes).
   (C) operating point only, standard included (cheapest; loses the direct/think delta that
   justifies the change).
2. **Latency cap in the operating-point rule.** (A) *recommended* — none; publish
   `time_to_answer_s` and let consumers filter (producer stays neutral, per the manifest
   contract). (B) a cap (e.g. p50 ≤ 15 s on the probe) that demotes to `direct`.
3. **Re-measure roster.** Re-pull for the v3 run: `qwen3.5:{4b,9b}-mlx`, `qwen3.8:27b-mlx`
   (strongest local reasoner, slow), `deepseek-r1:{8b,14b}`, `gpt-oss:20b` ("harmony quirk" =
   starvation), `falcon3:10b` ("empty-response quirk"), `qwen3-vl:4b` (LG: "think unkillable" =
   budget). `deepcoder:1.5b` stays out (no `tool_calls` → no lane).

## 9. Cost

The 08-21 everything-run was 18h38m for 21 completion models at `direct` (think=False). Policy (A) adds the
standard suite's think arm (+~1 h) and moves the batteries to the operating point; the think-token
overhead is model-specific (gemma4 ×1.3, qwen3.5-class ×3–5 on reasoning items). Estimate
**24–30 h** for ~26 models including re-pulls. Policy (B) ≈ 40 h.

## 10. Consumer follow-ups (not this repo)

- Manifest contract: assignments gain `think` (literal value) and `time_to_answer_p50`.
- OllamaMCP: `thinking: bool` → `think: bool | str`, pass top-level (not inside `options`), read the
  value from the manifest.
- LookingGlass: `config.think: bool` → `str`; drop the `<think>`-strip heuristic once the arm is
  explicit.
- Any consumer calling a `default_thinks` model must pass `think` explicitly or size `num_predict`
  for a full trace.
