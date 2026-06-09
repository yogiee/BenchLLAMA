# BenchLLAMA — Test Documentation

Reference documentation for every test in the benchmark harness. Each document covers the exact prompts used, scoring and evaluation logic, what different outcomes mean, and design rationale.

For high-level philosophy and suite architecture, see [`suites/suite-design.md`](../suites/suite-design.md).

---

## Origin and the "Alice" references

BenchLLAMA was built to support a specific AI assistant project (LookingGlass) that needed to select and rank local Ollama models for production use. That project's assistant character — named Alice — has a defined personality: direct, genuinely curious, intellectually sharp, warm without being sycophantic.

The personality dimension of the standard suite and Battery B were originally designed to measure how well a model could embody that character. The rubric and signal detection (direct voice, warmth, wit, deflection) reflect Alice's character brief specifically.

BenchLLAMA has since been extracted as a standalone, model-agnostic tool. The personality tests remain — they measure real and useful qualities in conversational models regardless of the application. But you'll see "Alice" referenced in `prompts/worker_default.md` and `suites/suite-design.md`. When running BenchLLAMA for your own project, substitute your own system prompt via `--system-prompt` and score the personality dimension against your own character rubric. The signal detection (direct voice, wit, warmth, deflection avoidance) is character-neutral and applies broadly.

---

## Documents

| File | What it covers |
|------|---------------|
| [standard-suite.md](standard-suite.md) | All 13 tests across 5 dimensions — personality, reasoning, research, instruction following, tool use |
| [ctx-ladder.md](ctx-ladder.md) | Context window characterisation pass — finds optimal `num_ctx` per model |
| [aptitude-a-router.md](aptitude-a-router.md) | Battery A — 4 tests targeting router-role models |
| [aptitude-b-worker-chat.md](aptitude-b-worker-chat.md) | Battery B — 5 tests targeting worker chat-role models |
| [aptitude-c-worker-research.md](aptitude-c-worker-research.md) | Battery C — 6 tests targeting worker research-role models |
| [aptitude-d-worker-tool.md](aptitude-d-worker-tool.md) | Battery D — 8 tests targeting worker tool-heavy-role models |

---

## Pipeline overview

```
./bench.sh all
    │
    ├── Standard Suite (runner.py)
    │       13 tests × all models
    │       → role gate applied automatically
    │
    ├── ctx Ladder (ctx_ladder.py)
    │       num_ctx sweep × all models
    │
    ├── Battery A  (aptitude.py --battery A)   router models only
    ├── Battery B  (aptitude.py --battery B)   worker models
    ├── Battery C  (aptitude.py --battery C --capable-only)   worker models, tool-capable
    └── Battery D  (aptitude.py --battery D --capable-only)   worker models, tool-capable
```

## Role gate criteria

| Role | Criteria |
|------|---------|
| Router | ≥80 tok/s, tool ✓, reasoning ≥1/4 |
| Worker — Chat | Personality avg ≥3.5/5, instruction ≥1/2, reasoning ≥2/4 |
| Worker — Research | JPEG ≥5/7 OR rag_finetune ≥4/5, tool ✓ |
| Worker — Tool-heavy | Tool ✓, reasoning ≥2/4 |

A model can qualify for multiple roles. `--capable-only` (auto-applied to Battery C and D in pipelines) further filters to models that passed the `calculate` test in the most recent standard run.
