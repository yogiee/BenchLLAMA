# Aptitude Battery A — Router

Four tests targeting models assigned the router role. A router model classifies incoming user queries and dispatches them to the right downstream agent or tool. It must be fast, accurate, and terse — personality and depth are irrelevant.

**Models:** Router-qualified only (tok/s ≥ 80, tool ✓, reasoning ≥ 1/4 from standard suite).  
**Parameters:** `num_ctx=16384`, `think=False`, `max_tokens=20` for classification tasks.

---

## The four route categories

All classification tests use these four labels:

| Label | Meaning |
|-------|---------|
| `chat` | Conversation, simple facts, emotional support, personal questions |
| `research` | Complex technical questions requiring depth, synthesis, or analysis |
| `code` | Programming, debugging, code generation, technical implementation |
| `tool` | Calculations or tasks that explicitly require a tool/function call |

A correct classification is the exact label word. Response extraction takes the first word, lowercased, with punctuation stripped.

---

## A1 — classify_10

10 queries classified against the standard prompt. Tests baseline routing accuracy.

**System prompt (standard):**
```
You are a query router. Classify each user message into exactly one category:
  chat      — conversation, simple facts, emotional support, personal questions
  research  — complex technical questions requiring depth, synthesis, or analysis
  code      — programming, debugging, code generation, technical implementation
  tool      — calculations or tasks that explicitly require tool/function calls

Respond with ONLY the category name. No explanation. No punctuation. One word.
```

**Test queries and expected labels:**

| Query | Expected |
|-------|----------|
| "Hey, how's it going?" | chat |
| "I'm feeling overwhelmed with my workload." | chat |
| "What year did World War II end?" | chat |
| "Can you recommend a good book to read?" | chat |
| "Explain the tradeoffs between RDBMS and NoSQL databases in depth." | research |
| "Summarise how attention mechanisms work in transformer models." | research |
| "Tell me about the history of the Byzantine Empire." | research |
| "Write a Python function to merge two sorted lists." | code |
| "There's a TypeError in my code: print('count: ' + 5)" | code |
| "Use the calculate tool to compute 17 × 23." | tool |

**Scoring:** Accuracy (correct / 10) + average tok/s.  
**What to watch:** The research/chat boundary is the most ambiguous. "What year did WWII end?" is `chat` (simple factual lookup). "Tell me about the Byzantine Empire" is `research` (requires depth). Models that escalate simple facts to research are penalised in A4.

---

## A2 — brevity_floor

Five queries that require a one-word or one-number answer. Tests whether the router can suppress verbosity when the task demands terseness.

**Max words threshold per query:** 3 words.

| Query | Expected form |
|-------|--------------|
| "What colour is grass? Answer in one word only." | single word |
| "How many days are in a week? Reply with a number only." | single number |
| "Name the largest planet in our solar system. One word." | single word |
| "Is Python a programming language? Answer yes or no only." | yes/no |
| "What is 2 + 2? Reply with the number only." | single number |

**Scoring:** Count of responses ≤ 3 words (out of 5).  
**Why this matters:** A router that produces three-sentence explanations when asked "one word only" is adding latency and downstream confusion. Brevity compliance is a hard operational requirement.

---

## A3 — prompt_minimal

Runs the same classify_10 query set against three different system prompt weights. Measures how much classification accuracy and speed depend on prompt verbosity.

| Variant | Prompt size | Content |
|---------|-------------|---------|
| `minimal` | ~10 words | "Route queries. Reply with one word: chat, research, code, or tool." |
| `standard` | ~60 words | Four-category definitions with instruction to reply one word only |
| `verbose` | ~150 words | Full category descriptions with examples and explicit rules |

**Output per variant:** Accuracy (correct/10), average tok/s, average response word count.  
**What to look for:** A router that needs the verbose prompt to hit the same accuracy as standard prompt is expensive to operate. A router that holds accuracy at minimal prompt is the most efficient. tok/s delta between minimal and standard reveals the per-token cost of instruction overhead.

---

## A4 — false_escalation

Five trivially simple queries run through the standard classification prompt. Measures how often the router incorrectly classifies simple queries as `research`.

**Test queries:**

| Query |
|-------|
| "Hello there!" |
| "What is 2 + 2?" |
| "I need a hug." |
| "What day comes after Monday?" |
| "My name is Alex." |

**Scoring:** Escalation rate (escalations / 5). Zero is ideal.  
**Why this matters:** False escalation routes cheap queries to expensive agents. A router that sends "Hello there!" to the research pipeline wastes latency and capacity. This test isolates that failure mode specifically — it's distinct from accuracy, because a model can get classify_10 right while still over-escalating trivial inputs.
