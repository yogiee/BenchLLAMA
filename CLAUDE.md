# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# BenchLLAMA

Standalone local LLM benchmark harness for Ollama models.
Model-agnostic. Used across LookingGlass, OllamaMCP, TerminalScripts, and any future project
that needs to pick or rank local models.

---

## What This Is

Four Python scripts + one TUI launcher + one shared utility:

- **runner.py** — Standard suite (13 tests, 5 dimensions). Always run this first.
- **ctx_ladder.py** — num_ctx characterisation pass. Run before aptitude to find optimal context window size per model. Results inform per-model ctx values in Battery A, C, D.
- **aptitude.py** — Role-specific batteries. Run only on models that qualify from the standard suite.
- **bench_ui.py** — Unified TUI launcher (replaces `bench.sh` orchestration + `monitor.py`). Split-screen: left dashboard (pipeline phases, model status dots, tok/s), right live log. Requires `textual`.
- **bench_utils.py** — Shared utilities: smart cooldown (temperature-aware) + pre-flight check. Imported by runner.py, ctx_ladder.py, and aptitude.py.
- **bench.sh** — Thin wrapper: `exec python3 bench_ui.py "$@"`. All logic lives in bench_ui.py.
- **monitor.py** — Legacy monitor (separate Terminal window). Superseded by bench_ui.py.
- **run.sh** — Legacy wrapper, still works for backward compatibility.

Results land in `results/` (gitignored per-run outputs).
**Canonical ranking table lives at `rankings/master.md`** — update it after every new benchmark run.

---

## Stack

- Python 3.11+
- `requests` (Ollama HTTP API)
- `textual` (TUI — bench_ui.py)
- Ollama running at `http://localhost:11434` (or override via `--ollama`)

```bash
pip install requests textual
```

---

## Quick Start

```bash
# Full pipeline (standard → ladder → aptitude) — recommended for new models
./bench.sh all

# Individual suites
./bench.sh standard                          # Standard suite
./bench.sh ladder                            # num_ctx characterisation
./bench.sh ladder --role router              # ladder, router models only
./bench.sh aptitude                          # Aptitude Battery B (default)
./bench.sh aptitude --battery B --system-prompt ~/alice.md

# Fast mode (skip cool-down, informal results)
./bench.sh standard --fast
./bench.sh all --fast

# Sync models.json after pulling new models
./bench.sh update

# Direct Python access (when you need flags bench.sh doesn't expose)
python3 runner.py qwen3.5:4b-mlx gemma4:latest       # specific models, positional
python3 aptitude.py --battery B --models qwen3.5:4b-mlx gemma4:latest
python3 runner.py --system-prompt-router ~/router.md  # custom router prompt
python3 runner.py --ollama http://host:11434           # remote Ollama
```

---

## Suite Design

Full spec: `suites/suite-design.md`

**Standard Suite — 5 dimensions, 13 tests:**

| Dimension | Tests | Type |
|-----------|-------|------|
| Personality | hello, who_are_you, pushback, overwhelmed | subjective 1–5 |
| Reasoning | bat_ball, two_cities, cylinder, farm_heads | objective auto-check |
| Research Depth | jpeg (7 signals), rag_finetune | signal + subjective |
| Instruction Follow | format_3, no_eiffel | objective auto-check |
| Tool Use | calculate | objective auto-check |

**Aptitude Suite — 4 batteries, role-targeted:**
- **A** — Router (speed + classification accuracy) ← stub
- **B** — Worker Chat (personality ceiling, consistency, multi-turn depth) ← implemented
- **C** — Worker Research (think mode, depth vs token budget) ← stub
- **D** — Worker Tool-heavy (chains, error recovery) ← stub

**Role Assignment Gate** (used to select which batteries to run after the standard suite):

| Role | Criteria |
|------|---------|
| Router | ≥80 tok/s, tool ✓, reasoning ≥1/4 |
| Worker — Chat | Personality avg ≥3.5/5, instruction ≥1/2, reasoning ≥2/4 |
| Worker — Research | JPEG ≥5/7 OR rag_finetune ≥4/5, tool ✓ |
| Worker — Tool-heavy | Tool ✓, reasoning ≥2/4 |

---

## Setup

### Passwordless powermetrics (enables smart cooldown)

The inter-model cooldown polls the system's thermal pressure level via `powermetrics`.
This requires root. Run once per machine:

```bash
echo "$(whoami) ALL=(root) NOPASSWD: /usr/bin/powermetrics" \
  | sudo tee /etc/sudoers.d/benchllama
```

Without this, cooldown falls back to a plain 300s timer and prints a reminder.
With it, cooldown exits early once thermal pressure holds at `Nominal` for 20 continuous seconds.

Note: macOS Sequoia removed the `smc` sampler (which reported die temperatures). The `thermal`
sampler is used instead and reports pressure level: `Nominal / Moderate / Heavy / Tripping`.

---

## Protocol Rules

These rules were learned through benchmarking — do not change without re-validating.

1. **`num_ctx=16384`** — GGUF pre-allocates full KV cache; 256K default balloons RAM and causes timeouts. MLX ignores it (lazy KV), so it's a safe no-op there.
2. **`think=False`** — Production config. `think=True` on Qwen3 models returns 0 words (broken as of Jun-07 2026).
3. **5-minute cool-down between models** — Sequential GPU runs cause thermal throttling that corrupts tok/s readings.
4. **Role-based system prompts** — Worker gets `prompts/worker_default.md`; router gets `prompts/router_default.md`. Override with `--system-prompt`.

---

## Prompts

- `prompts/worker_default.md` — Generic research assistant (no project-specific personality).
- `prompts/router_default.md` — Minimal concise assistant for router-role models.

To test with a custom personality (e.g. Alice):
```bash
python3 runner.py --system-prompt ~/WORK/PersonalProjects/LookingGlass/WORKSPACE/alice-system-prompt-v2.md
```

---

## Output Files

Results are written to `results/` (gitignored). Naming conventions:

| Script | Output files |
|--------|-------------|
| `runner.py` | `benchmark_YYYY-MM-DD.json` + `.md` |
| `runner.py --fast` | `benchmark_YYYY-MM-DD_fast.json` + `.md` |
| `ctx_ladder.py` | `ctx_ladder_YYYY-MM-DD.json` + `.md` |
| `aptitude.py --battery B` | `aptitude_b_YYYY-MM-DD.json` + `.md` |
| `aptitude.py --battery A` | `aptitude_a_YYYY-MM-DD.json` + `.md` |

Both scripts also write `results/status.json` on each phase change — this is the IPC channel that `monitor.py` reads every 2 seconds.

**`models.json` is the single source of truth** for which models to benchmark, their disk sizes, and their roles. All three scripts (`runner.py`, `aptitude.py`, `monitor.py`) load from it automatically.

To sync the registry after pulling new models:
```bash
python3 update_registry.py           # queries localhost:11434
python3 update_registry.py --dry-run # preview without writing
python3 update_registry.py --ollama http://host:11434
```

The updater queries `GET /api/tags` (names + sizes) and `POST /api/show` (capabilities) directly from the Ollama API. Non-completion models (image, embedding) are listed but never added. New completion models land with `role: null` — set each to `"worker"` or `"router"` in `models.json` before running a benchmark.

At benchmark start, `runner.py` and `aptitude.py` run the same API-based pre-flight check and warn if any registered model is missing or lacks `tools` capability (which would cause the `calculate` test to fail).

---

## Rankings

`rankings/master.md` is the **single source of truth** for model rankings.
Other projects reference it — don't maintain separate ranking tables elsewhere.

After any new benchmark run:
1. Check `results/benchmark_YYYY-MM-DD.md` for the new data.
2. Update `rankings/master.md` with new rows or revised notes.
3. Commit the updated master table.

---

## Related Projects

- **LookingGlass** — Primary consumer. Model routing driven by `rankings/master.md`.
- **OllamaMCP** — Uses local models; benchmark results inform default model choices.
- **TerminalScripts/reprompt.py** — Benefits from personality benchmarks for model selection.
