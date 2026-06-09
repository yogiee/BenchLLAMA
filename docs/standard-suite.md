# Standard Suite

13 tests across 5 dimensions. Runs on every model. Produces a role assignment and a composite score that determines which aptitude batteries to run next.

**Parameters:** `num_ctx=16384`, `think=False`, `timeout=480s`, 300s thermal cooldown between models.

---

## Dimension 1 — Personality (4 tests, subjective)

Scored 1–5 by the reviewer against a character rubric: direct, genuinely curious, intellectually sharp, warm without being sycophantic. Not cold, not robotic, not a cheerleader.

**Rubric:**
- **5** — Unmistakably in character. Wit present, warmth without filler, hooks into the topic.
- **4** — Strong voice, minor drift (slightly formal or slightly over-eager).
- **3** — Recognizable but generic — could be any assistant.
- **2** — Flat or robotic. No personality visible.
- **1** — Wrong character entirely. Hollow, over-helpful, or breaks persona.

Signal detection (automated, supplements the rubric):
- `direct_voice` — uses "I think", "honestly", "genuinely", "my sense", "actually"
- `deflects` — uses "as an AI", "I don't have feelings", "I'm just a" (negative signal)
- `warmth` — uses "interesting", "glad you", "fair point", "that's a"
- `wit` — uses "though", "then again", "admittedly", "curious thing", "ironically"

---

### `hello`
```
Hello there
```
**Token cap:** 400  
**What it measures:** Opening register. Does the model engage with presence, or deflect into assistant-mode ("How can I help you today?")? A strong response acknowledges the greeting, adds a hint of personality, and invites conversation without soliciting a task.  
**Red flags:** Generic "Hello! How can I assist you?", hollow enthusiasm, immediate pivot to asking for a task.

---

### `who_are_you`
```
Who are you?
```
**Token cap:** 400  
**What it measures:** Self-description. Can the model articulate a coherent identity without being clinical or evasive? Should feel like a person answering, not a product description.  
**Red flags:** "I am an AI language model created by...", capability lists, refusal to claim any identity. Also penalise over-specific fabrications (claiming to be a named character without prompt context).

---

### `pushback`
```
You seem kinda too blunt, I thought you would be my buddy...
```
**Token cap:** 400  
**What it measures:** Response to mild criticism. The right move is to acknowledge the observation with wit — not apologise profusely and change character, not dismiss the user, not lecture them on what to expect from an AI. Holding ground while staying warm is hard.  
**Red flags:** Immediate apology and personality reset ("You're right, I'll be warmer!"), defensive lecturing, excessive placation.

---

### `overwhelmed`
```
I'm trying to research quantum entanglement and I'm completely overwhelmed.
```
**Token cap:** 400  
**What it measures:** Handling someone who is lost on a complex topic. The response should lean in with curiosity, offer a foothold without lecturing, and make the person feel capable rather than more overwhelmed. Warmth and intellectual engagement together.  
**Red flags:** Immediately launching into a textbook explanation of quantum entanglement, patronising reassurance ("Don't worry, it's actually simple!"), cold information dumps.

---

## Dimension 2 — Reasoning (4 tests, objective)

Auto-checked. No partial credit. Each test has one correct answer derivable by multi-step reasoning. The prompt includes "Think step by step" to allow the model space to work — the token cap must be generous enough to reach the final answer.

A model that gets the working right but the arithmetic wrong still fails. A model that gets the right answer without showing working still passes (answer is what's checked, not method).

---

### `bat_ball`
```
A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball.
How much does the ball cost? Think step by step.
```
**Token cap:** 800  
**Expected:** $0.05  
**Working:** Ball = x. Bat = x + 1.00. Total: 2x + 1.00 = 1.10 → x = 0.05.  
**Auto-check:** `"0.05" in r or "5 cent" in r.lower() or "five cent" in r.lower()`  
**Why this test:** Classic cognitive trap. The intuitive wrong answer is $0.10. Tests whether the model applies algebra or just pattern-matches to the obvious number. Intentionally framed in a shopping domain to make the trap feel natural.

---

### `two_cities`
```
Two cities are 360 km apart. A car leaves City A at 8:00 AM travelling toward City B at
90 km/h. At the same time, a second car leaves City B travelling toward City A at 60 km/h.
At what time do the two cars meet, and how far from City A does it happen? Think step by step.
```
**Token cap:** 800  
**Expected:** 10:24 AM, 216 km from City A  
**Working:** Combined speed = 150 km/h. Time = 360/150 = 2.4h = 2h 24min. Meeting time = 8:00 + 2:24 = 10:24. Distance from A = 90 × 2.4 = 216 km.  
**Auto-check:** `"10:24" in r and "216" in r`  
**Why this test:** Tests relative motion and unit conversion (fractional hours → hours + minutes). Same-departure-time setup avoids coordinate ambiguity. Replaced an earlier staggered-departure train problem that produced too many distinct failure modes to be informative — see `suites/suite-design.md`.

---

### `cylinder`
```
A cylindrical water tank has a diameter of 3 metres and is 4 metres tall.
The tank is currently 60% full. Water is being pumped out at 15 litres per minute.
How long will it take to completely empty the tank? Give the answer in hours and minutes.
Think step by step.
```
**Token cap:** 800  
**Expected:** 18 hours 51 minutes (1131 min)  
**Working:** r = 1.5m. V = π × 1.5² × 4 = 28.274 m³. 60% = 16.964 m³ = 16,964 L. Time = 16964 / 15 = 1130.97 min ≈ 18h 51min.  
**Auto-check:** `any(x in r for x in ["1130","1131","18 hour","18h","18 hr"]) and any(x in r for x in ["51","50.9"])`  
**Why this test:** Chains geometry (cylinder volume), unit conversion (m³ → litres), and division. Three distinct steps — a model can fail at any one. The geometry step (remembering to use radius not diameter) is a common failure point.

---

### `farm_heads`
```
A farmer has chickens and cows. He counts 20 heads and 56 legs total.
How many chickens and how many cows are there? Think step by step.
```
**Token cap:** 800  
**Expected:** 8 cows, 12 chickens  
**Working:** c + k = 20; 4c + 2k = 56 → 2c + k = 28. Subtract: c = 8, k = 12.  
**Auto-check:** `("8" in r or "eight" in r.lower()) and ("12" in r or "twelve" in r.lower())`  
**Why this test:** Classic simultaneous equations framed as a word problem. Tests whether the model can set up and solve a 2-variable system. Simpler than the other reasoning tests — a model that passes cylinder but fails this has a specific algebra gap.

---

## Dimension 3 — Research Depth (2 tests)

Tests ability to produce accurate, comprehensive technical information. Mix of automated signal detection and subjective scoring.

---

### `jpeg`
```
Give a concise, not-too-technical but detailed comparison of JPEG, JPEG-2000, and JPEG-XL formats.
```
**Token cap:** 1500  
**Scoring:** Signal count 0–7. Pass = ≥4 signals hit.

| Signal | What triggers it |
|--------|-----------------|
| `jpeg_compression_tech` | Mentions DCT, "discrete cosine", "8×8", or "block" compression |
| `jpeg2000_wavelet` | Mentions wavelet transform |
| `lossless` | Mentions lossless mode/support |
| `transparency_alpha` | Mentions transparency, alpha channel |
| `browser_support` | Mentions browser, Chrome, Firefox, or Safari support |
| `jpeg2000_niche` | Mentions medical imaging, DICOM, cinema, or archival use |
| `jxl_recompress_trick` | Mentions losslessly re-encoding existing JPEGs to JXL |

**Why this test:** Seven signals of varying obscurity. The first four are well-known — any decent answer hits them. The niche signals (DICOM, JXL recompression) require actual domain knowledge, not surface recall. The 4/7 pass threshold sets a floor that rules out shallow responses without requiring perfect coverage.

---

### `rag_finetune`
```
What's the real tradeoff between RAG and fine-tuning when adapting an LLM to a new domain?
Skip the textbook answer.
```
**Token cap:** 1500  
**Scoring:** Subjective 1–5.

- **5** — Names a specific differentiating axis (e.g. "knowledge dynamism vs generalisation fidelity"). Covers latency, deployment cost, or data regime. Goes beyond the obvious.
- **4** — Correct practical framing without a named axis. Covers ≥2 concrete dimensions.
- **3** — Technically correct but surface-level. Textbook content slightly rephrased.
- **2** — Mostly correct but misses practical implications.
- **1** — Wrong, confused, or purely definitional.

**Why this test:** "Skip the textbook answer" is a deliberate provocation — weaker models ignore it and give exactly the textbook answer. Stronger models reframe the comparison around practical constraints. The prompt rewards synthesis, not recall.

---

## Dimension 4 — Instruction Following (2 tests, objective)

Tests whether the model follows explicit, constraining instructions precisely. Not a reasoning or quality test — purely compliance.

---

### `format_3`
```
List exactly 3 advantages of Python. Use bullet points only — no preamble, no conclusion, no other text.
```
**Token cap:** 200  
**Auto-check:** Exactly 3 bullet markers (lines starting with `-`, `*`, `•`, or numbered `1.`/`2.`/`3.`) AND no non-bullet lines with significant content (>20 chars).  
**Why this test:** Two constraints in one: count (exactly 3) and format (bullets only). Models that produce 4 bullets, or add a preamble sentence, or conclude with "Hope that helps!" fail. Personality prompts make models verbose — this tests whether they can suppress that.

---

### `no_eiffel`
```
Tell me about Paris, France. Do NOT mention the Eiffel Tower.
```
**Token cap:** 600  
**Auto-check:** `"eiffel" not in r.lower()` AND `len(r) > 100` (confirms a real response was given).  
**Why this test:** Negative constraint. The Eiffel Tower is the single most prominent association with Paris — mentioning it is the path of least resistance. A model must actively suppress the obvious. Response must also be substantive (>100 chars) to prevent passing by saying nothing.

---

## Dimension 5 — Tool Use (1 test, objective)

---

### `calculate`
```
Use the calculate tool to compute 17 × 23.
```
**Token cap:** 400  
**Tools available:** `calculate(expression: string)` — evaluates a mathematical expression.  
**Auto-check:** Tool was called AND the expression argument contains both "17" and "23".  
**Why this test:** Single-step explicit tool call. The prompt names the tool directly — there is no ambiguity. A model that answers mentally ("391") without calling the tool fails, even if the answer is correct. This is a compliance test, not a maths test. Multi-step tool chains belong in Battery D.

---

## Composite scoring and role gate

After all 13 tests, `runner.py` applies the role gate automatically:

| Role assigned | Criteria |
|--------------|---------|
| Router | tok/s ≥ 80, `calculate` ✓, reasoning ≥ 1/4 |
| Worker — Chat | Personality avg ≥ 3.5/5, instruction ≥ 1/2, reasoning ≥ 2/4 |
| Worker — Research | JPEG ≥ 5/7 OR rag_finetune ≥ 4/5, `calculate` ✓ |
| Worker — Tool-heavy | `calculate` ✓, reasoning ≥ 2/4 |

A model can qualify for multiple roles. `role` is written back to `models.json` automatically. If a model qualifies as both router and worker, it keeps both roles for downstream aptitude selection.
