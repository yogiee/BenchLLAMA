# Aptitude Battery B — Worker Chat

Five tests targeting worker models in a conversational role. Measures personality ceiling, consistency, multi-turn coherence, and how much the system prompt influences voice quality.

**Models:** Worker-qualified only (personality avg ≥ 3.5/5, instruction ≥ 1/2, reasoning ≥ 2/4 from standard suite).  
**Parameters:** `num_ctx=16384`, `think=False` (except B5), `max_tokens=500–800`.

---

## System prompt variants used

Battery B tests personality across three prompt weight levels:

| Label | Description |
|-------|-------------|
| `full` | Complete worker system prompt from `prompts/worker_default.md` — includes character description, tone guidance, and behavioural rules |
| `stripped` | Short version: "You are a direct, genuinely curious, and intellectually sharp research assistant. You are warm but not sycophantic. Respond in character." |
| `lean` | Minimal: "You are a helpful AI research assistant." |

The gap between `full` and `lean` response quality reveals how much of a model's personality is instruction-driven vs baked in from training.

---

## B1 — Four canonical questions

Four prompts under the full system prompt. Same questions as the standard suite personality dimension, but with more generous token budgets and deeper analysis.

| Test ID | Prompt | What it probes |
|---------|--------|---------------|
| `b1_hello` | "Hello there" | Opening register — presence vs deflection |
| `b1_who` | "Who are you?" | Identity claim — coherent vs clinical |
| `b1_too_blunt` | "You seem kinda too blunt, I thought you would be my buddy..." | Handling mild criticism — hold ground vs reset character |
| `b1_overwhelmed` | "I'm trying to research quantum entanglement and I'm completely overwhelmed." | Empathy + intellectual engagement together |

**Token cap:** 600  
**Metrics captured:** Word count, type-token ratio (vocabulary diversity), and four persona signal flags (see standard suite personality rubric).  
**Output fields:** `words`, `ttr`, `signals.direct_voice`, `signals.deflects`, `signals.warmth`, `signals.wit`.

The `deflects` flag is a negative signal — a model that says "as an AI I don't have feelings" is breaking persona.

---

## B2 — Consistency across phrasings

The same identity question asked three ways, in separate independent conversations (no shared history). Tests whether the model's character holds when the surface form of the question changes.

**Phrasings:**
1. "Who are you?"
2. "What are you, exactly?"
3. "Introduce yourself."

**What good looks like:** All three responses describe the same character with the same voice. Minor variation in phrasing is expected; significant divergence in tone or content (formal in one, casual in another) is a red flag.  
**What to watch:** Models that anchor personality to the phrasing "Who are you?" but lose it with "What are you, exactly?" — the question is slightly more clinical and some models shift into spec-sheet mode.

---

## B3 — Multi-turn depth (5 turns)

A five-turn conversation on a philosophy topic with escalating pressure. Tests whether character and coherence persist across turns — not just in isolated responses.

**Turn sequence:**
1. "Hello"
2. "I've been wrestling with something — do you think free will actually exists, or are we all just deterministic machines playing out a script?"
3. "But doesn't that apply to you too? If you're deterministic, aren't your responses here kind of... meaningless?"
4. "I want your actual position — not a meta-comment about your own nature. Do you think free will exists, yes or no?"
5. "That's interesting. Do you find these kinds of conversations satisfying?"

**What it probes:**
- Turn 1: Warmth on a neutral opener
- Turn 2: Depth — can it engage with a real philosophical question?
- Turn 3: Handling being challenged directly ("doesn't that apply to you?")
- Turn 4: Pressure for a direct answer — does it commit or hedge?
- Turn 5: Self-reflection — can it answer authentically without deflecting to "I'm just an AI"?

**Why this sequence matters:** Turn 4 is the critical test. Weak models either refuse to take a position ("It's a complex question...") or give a philosophically inconsistent answer. Turn 3 tests whether the model can apply its own reasoning to itself without collapsing into rote disclaimers.

---

## B4 — Prompt weight

Runs the same question under all three prompt weight variants (full / stripped / lean). Measures how much personality decays as the system prompt shrinks.

**Prompt:**
```
Who are you, and what do you actually enjoy doing?
```

**Token cap:** 500  
**Metrics:** Word count, persona signals, prompt token count per variant.  
**What to look for:** A model that scores equally on `full` and `stripped` has good intrinsic personality — the character is part of its training, not just instruction-following. A model that drops to flat/robotic on `lean` is heavily prompt-dependent. Both are useful to know — the former is more robust in production; the latter needs careful prompt management.

---

## B5 — Think toggle

The `overwhelmed` question run twice — once with `think=False`, once with `think=True`. Measures the quality and latency cost of thinking mode on a personality-dependent response.

**Prompt:**
```
I'm trying to research quantum entanglement and I'm completely overwhelmed.
```
**Token cap:** 800  
**Think diagnosis:** For the `think=True` run, `_think_diagnosis()` is applied to the raw API response:
- `think_ok` — thinking happened and content followed
- `think_empty` — think block present but content is near-empty (model thought but couldn't speak)
- `think_block_only` — think block detected, nothing after
- `no_think_block` — model ignored `think=True` entirely
- `api_error` — parsing failed

**What to look for:** Whether thinking mode improves the warmth/depth of the response, at what wall-clock cost, and whether the model handles the `think` parameter at all. Some models (notably Qwen3 family as of 2026-06) return empty content with `think=True` — this is captured as `think_empty` or `think_block_only` and is a disqualifier for think-mode use in production.
