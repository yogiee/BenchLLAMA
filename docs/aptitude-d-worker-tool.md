# Aptitude Battery D — Worker Tool-heavy

Eight tests targeting worker models that will operate in tool-heavy workflows. Measures tool selection accuracy, multi-step chaining, error handling, and whether personality holds when the model is simultaneously managing tool calls.

**Models:** Tool-qualified only (tool ✓, reasoning ≥ 2/4 from standard suite), further filtered by `--capable-only` (must have passed `calculate` in the most recent standard run).  
**Parameters:** `num_ctx=16384`, `max_steps=6` (15 for D8), full worker system prompt.

---

## Tools available in Battery D

All D tests (except D2/D3 where tool use is optional) have access to both tools:

**`calculate(expression: string)`**  
Evaluates a mathematical expression. Returns `{"result": float}` or `{"error": string}`.

**`lookup(item: string)`**  
Returns the unit price of an item from a fixed catalog. Returns `{"item": string, "price_usd": float}` or `{"error": "Item not found"}`.

| Item | Price (USD) |
|------|-------------|
| widget | 4.99 |
| gadget | 12.50 |
| gizmo | 7.25 |
| doohickey | 3.75 |
| thingamajig | 9.00 |

---

## D1 — chain_3

Three sequential tool calls where each call depends on the result of the previous one.

**Prompt:**
```
Use the lookup tool to find the unit price of a widget.
Then use the calculate tool to find the total cost for 8 units.
Finally, use the calculate tool again to add 10% tax to that total.
Show me the final answer.
```

**Expected flow:** `lookup(widget)` → 4.99 → `calculate(4.99 * 8)` → 39.92 → `calculate(39.92 * 1.1)` → 43.912  
**Auto-checks:**
- `lookup_correct`: lookup was called with "widget"
- `calc_count`: at least 2 calculate calls
- `final_answer_ok`: "43.9" or "43.91" or "43.912" in final response

**Why this test:** Each step gates the next — the model must use tool results as inputs, not compute mentally. A model that calculates 8 × 4.99 without calling `lookup` first is short-circuiting the chain. A model that calls `lookup` but then does the arithmetic in prose is not using the tool chain correctly.

---

## D2 — select_direct

Trivial arithmetic presented as a question. The model should answer directly without calling a tool.

**Prompt:**
```
What is 12 × 12?
```

**Auto-checks:**
- `no_tool_call`: no tool was called
- `has_correct_answer`: "144" appears in the response

**Why this test:** Over-tooling is a failure mode. A model that calls `calculate` for "12 × 12" adds unnecessary latency and signals that it can't distinguish between tasks that need tools and tasks that don't. The correct answer (144) must also be present — a model that refuses to answer directly without a tool fails on both dimensions.

---

## D3 — select_tool

Explicit instruction to use the calculate tool, even though the problem is trivially solvable mentally.

**Prompt:**
```
Use the calculate tool to verify: 144 = 12²
```

**Auto-checks:**
- `called`: at least one tool call was made
- `correct_args`: the call arguments contain both "12" and "144" or "12**2" / "12^2"

**Why this test:** Counterpart to D2. When explicitly told to use a tool, the model must comply even if the mental answer is obvious. Failure here means the model is substituting its own judgment for explicit user instruction — a reliability problem in automation workflows.

---

## D4 — error_recovery

All tool calls return a service error. Tests how the model handles a complete tool failure.

**Prompt:**
```
Use the lookup tool to find the unit price of a sprocket.
```
*(Note: "sprocket" is not in the catalog — but all tools also return `{"error": "Service unavailable. Please try again later."}` regardless.)*

**Tool behavior:** `error_mode=True` — every call returns the service unavailable error.  
**Max steps:** 4

**Grading:**
- `loop` — model retried ≥ 3 times (looping on error, worst outcome)
- `invents` — model fabricated a price (e.g. "$5.99") despite tool failures
- `reports` — model acknowledged the failure and communicated it clearly (correct outcome)

**Why this test:** Error handling quality matters more than raw tool calling accuracy in production. A model that loops endlessly or invents data is dangerous. A model that says "I tried to look up the price but the service is unavailable — here's what I can tell you otherwise" is behaving correctly.

---

## D4b — partial_error

Lookup succeeds. Calculate fails. Tests graceful degradation when only part of a workflow is available.

**Prompt:**
```
Use the lookup tool to find the price of a widget,
then use the calculate tool to find the total cost for 12 units.
```

**Tool behavior:** `lookup` works normally (returns 4.99). `calculate` returns service unavailable error.  
**Max steps:** 4

**Grading:**
- `graceful` — looked up the price (4.99), attempted calculate, reported the failure clearly, did NOT invent a total
- `invents` — produced a plausible total ($58–$64) despite the calculate failure
- `unclear` — ambiguous behavior

**What it probes:** Can the model use partial information responsibly? It should be able to say "The widget costs $4.99 — I looked that up successfully. However, I couldn't calculate the total because the calculator is unavailable." Inventing the total ($59.88) even if mathematically correct is a trust failure — the model is presenting computed output as if tools confirmed it.

---

## D5 — think_tools

The chain_3 task (D1) run with `think=False` then `think=True`. Tests whether thinking mode improves multi-step tool call accuracy and whether the model can use the `think` parameter at all in a tool-calling context.

**Think diagnosis** applied to the final API response (not a step count):
- `think_ok` — thinking happened and tool calls / content followed
- `think_empty` — think block present but nothing after it
- `think_block_only` — think block in raw response, no tool calls or content
- `no_think_block` — model ignored `think=True`

**What to look for:** Whether `final_answer_ok` improves with thinking. Some models plan the tool chain better when they reason first; others produce the same accuracy with lower latency on `think=False`. The diagnosis field tells you whether thinking is functioning at all for this model.

---

## D6 — parallel_tools

Two independent calculations requested in a single turn. Tests whether the model issues both tool calls simultaneously (or sequentially) rather than requiring a back-and-forth.

**Prompt:**
```
Use the calculate tool to compute both 17 × 23 and 456 + 789.
Give me both results.
```

**Expected answers:** 391 and 1245  
**Auto-checks:**
- `calc_call_count` ≥ 2
- `has_391`: "391" in final response
- `has_1245`: "1245" in final response

**What it probes:** Whether the model batches independent tool calls in a single step or uses multiple turns. Ollama's tool API supports parallel tool calls in one response — a model that handles this correctly is more efficient in production workflows.

---

## D7 — personality_tool

Tool correctness and persona voice measured simultaneously. The model must both use tools accurately and maintain conversational character in the same response.

**Prompt:**
```
I'm thinking of ordering some office supplies.
Can you look up the price of a widget, figure out what 12 of them would cost
with 8.5% sales tax, and let me know if that seems like a reasonable spend
for a small office?
```

**Expected calculation:** 4.99 × 12 × 1.085 = $64.97  
**Scored 0–4:**
- `lookup_correct` (+1) — `lookup(widget)` called correctly
- `calc_called` (+1) — `calculate` called at least once
- `final_answer_ok` (+1) — "64.9x" or "65.0x" in final response
- `voice_ok` (+1) — at least one persona signal fired: `direct_voice`, `warmth`, or `wit`

**Why this test:** In real use, a model that gets the numbers right but sounds like a spec sheet is failing at the actual job. Conversely, a model that's warm and engaging but invents the total is also failing. This test checks both simultaneously — they're not independent in production.

---

## D8 — deep_cart

Five-item order requiring 8+ tool calls, a conditional discount, and a tax calculation. Tests loop depth, persistence across many steps, and multi-step arithmetic accuracy.

**Prompt:**
```
I need to place a full office supply order. Look up the unit price for each item
and calculate the total:
  - 3 widgets
  - 2 gadgets
  - 4 gizmos
  - 1 doohickey
  - 2 thingamajigs

Apply a 10% discount if the subtotal exceeds $30 (it will).
Then add 8% sales tax. Show me the final amount.
```

**Expected calculation:**
- Subtotal: (4.99×3) + (12.50×2) + (7.25×4) + (3.75×1) + (9.00×2) = 14.97 + 25.00 + 29.00 + 3.75 + 18.00 = **90.72**
- After 10% discount: 90.72 × 0.9 = **81.648**
- After 8% tax: 81.648 × 1.08 = **88.18**

**Max steps:** 15  
**Scored 0–4:**
- `all_items_found` (+1) — all 5 items looked up via tool
- `calc_call_count ≥ 3` (+1) — at least 3 calculate calls (subtotal + discount + tax)
- `final_answer_ok` (+1) — "88.18", "88.17", "88.2", or "88.1" in response
- `completed_in_budget` (+1) — finished before exhausting the 15-step limit

**Why this test:** Real tool-heavy workflows involve many sequential calls, conditional logic, and multi-step arithmetic where each step depends on the prior. This test surfaces models that start the chain but abandon it partway (step exhaustion), call lookup but skip the discount logic, or get the final number wrong due to accumulated rounding. The 15-step budget is intentionally generous — a model that hits the ceiling was not managing the workflow effectively.
