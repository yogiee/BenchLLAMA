<p align="center">
  <img src="assets/logo.png" width="360" alt="BenchLLAMA"/>
</p>

<h1 align="center">BenchLLAMA</h1>
<p align="center">Standalone benchmark harness for local Ollama models</p>

---

BenchLLAMA runs structured, repeatable benchmarks against any model served by [Ollama](https://ollama.com). It measures personality, reasoning, research depth, instruction following, and tool use — then produces ranked results you can act on.

## What's inside

| Script | Purpose |
|--------|---------|
| `runner.py` | Standard suite — 13 tests across 5 dimensions |
| `ctx_ladder.py` | Context window characterisation — finds optimal `num_ctx` per model |
| `aptitude.py` | Role-specific batteries — deep evaluation for router and worker models |
| `bench_ui.py` | Unified TUI — split-screen dashboard + live log |
| `bench_utils.py` | Shared utilities — smart thermal cooldown, pre-flight checks |

## Quick start

```bash
pip install requests textual

# Full pipeline: standard suite → ctx ladder → all aptitude batteries
./bench.sh all

# Individual suites
./bench.sh standard
./bench.sh ladder
./bench.sh aptitude --battery B
./bench.sh aptitude --battery D --capable-only

# Fast mode (skip cooldown — informal results)
./bench.sh standard --fast
```

## Standard suite — 5 dimensions, 13 tests

| Dimension | Tests | Scoring |
|-----------|-------|---------|
| Personality | hello, who_are_you, pushback, overwhelmed | Subjective 1–5 |
| Reasoning | bat_ball, two_cities, cylinder, farm_heads | Auto-checked |
| Research Depth | jpeg (7 signals), rag_finetune | Signal count + subjective |
| Instruction Follow | format_3, no_eiffel | Auto-checked |
| Tool Use | calculate | Auto-checked |

## Aptitude batteries

Run after the standard suite on models that qualify.

| Battery | Role | What it measures |
|---------|------|-----------------|
| A | Router | Classification accuracy, brevity, prompt weight, false-escalation rate |
| B | Worker — Chat | Personality ceiling, consistency, multi-turn depth, think toggle |
| C | Worker — Research | JPEG signal coverage (think on/off), RAG depth, synthesis, ctx ladder, token ceiling |
| D | Worker — Tool-heavy | Tool chains, error recovery, partial failure handling, personality + tool integration |

Use `--capable-only` with Battery C and D to automatically skip models that failed the tool-use test in the most recent standard run.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running at `http://localhost:11434`
- `pip install requests textual`

### Optional: passwordless powermetrics (smarter cooldown — macOS only)

```bash
echo "$(whoami) ALL=(root) NOPASSWD: /usr/bin/powermetrics" \
  | sudo tee /etc/sudoers.d/benchllama
```

Without this, cooldown falls back to a flat 300s timer. With it, the harness polls thermal pressure and exits early once the chip holds `Nominal` for 20 seconds.

## Output

Results land in `results/` (gitignored). Each run produces a JSON + Markdown report, and a timestamped `run_YYYY-MM-DD_HH-MM.log` capturing the full session.

| Script | Output |
|--------|--------|
| `runner.py` | `benchmark_YYYY-MM-DD.json` + `.md` |
| `ctx_ladder.py` | `ctx_ladder_YYYY-MM-DD.json` + `.md` |
| `aptitude.py --battery B` | `aptitude_b_YYYY-MM-DD.json` + `.md` |

## Syncing the model registry

```bash
python3 update_registry.py        # query localhost:11434 and update models.json
python3 update_registry.py --dry-run
```

New models are added as `role: "worker"` automatically. After running the standard suite, `runner.py` applies the role gate and promotes qualifying models to `router` — no manual editing required.
