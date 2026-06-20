<p align="center">
  <img src="assets/logo.png" width="360" alt="BenchLLAMA"/>
</p>

<h1 align="center">BenchLLAMA</h1>
<p align="center">Standalone benchmark harness for local Ollama models</p>

---

BenchLLAMA runs structured, repeatable benchmarks against any model served by [Ollama](https://ollama.com). It measures personality, reasoning, research depth, instruction following, tool use, coding, consistency, vision, and embedding — then produces ranked results you can act on, served live in a browser dashboard.

## What's inside

| Script | Purpose |
|--------|---------|
| `runner.py` | Standard suite — 13 tests across 5 dimensions |
| `ctx_ladder.py` | Context window characterisation — finds optimal `num_ctx` per model |
| `aptitude.py` | Role-specific batteries (A–F) — deep evaluation for router and worker models |
| `vision.py` | Battery V — capability-routed vision/OCR evaluation |
| `embedding.py` | Battery EMB — capability-routed embedding evaluation |
| `orchestrator.py` | Headless orchestration core + plain-text console runner |
| `webserver.py` | Web UI server (aiohttp) — drives the orchestrator, serves the live dashboard over WebSocket |
| `web/index.html` | The browser dashboard — phase tree, model cards, streamed log, Rankings + Files viewers |
| `bench_utils.py` | Shared utilities — smart thermal cooldown, pre-flight checks |

## Quick start

```bash
pip install requests aiohttp beautifulsoup4 html5lib tinycss2

# Web dashboard (default) — opens browser model selection, then Start/Stop from the UI
./bench.sh                 # no command → browser selection
./bench.sh all             # full pipeline in the dashboard

# Plain terminal instead of the browser (headless / SSH / quick glance)
./bench.sh all --console

# Full pipeline: standard → ctx ladder → A–D → E → F → Vision → Embedding
./bench.sh all

# Individual suites
./bench.sh standard
./bench.sh ladder
./bench.sh aptitude --battery B
./bench.sh aptitude --battery D --capable-only
./bench.sh aptitude --battery E       # coding — 3-run averaged by default
./bench.sh vision                     # every vision-capable model
./bench.sh embedding                  # every embedding model

# Fast mode (skip cooldown — informal results)
./bench.sh standard --fast
```

### Live dashboard

`./bench.sh` launches the aiohttp web UI (`webserver.py`) and serves `web/index.html` —
a single-file dashboard (orange-cream theme, light/dark/system) showing the phase tree,
per-model cards (role + extended-role badges, capability tags, tok/s), a streamed log,
and built-in **Rankings** + **Files** viewers. Selection and Start/Stop are browser-driven.
Bind to the LAN with `--host 0.0.0.0` (read-only unless you add `--allow-control`).

> The Textual TUI (`bench_ui.py` / `monitor.py`) was retired 2026-06-15 in favour of the
> web UI; `textual` is no longer a dependency.

## Standard suite — 5 dimensions, 13 tests

| Dimension | Tests | Scoring |
|-----------|-------|---------|
| Personality | hello, who_are_you, pushback, overwhelmed | Subjective 1–5 |
| Reasoning | bat_ball, two_cities, cylinder, farm_heads | Auto-checked |
| Research Depth | jpeg (7 signals), rag_finetune | Signal count + subjective |
| Instruction Follow | format_3, no_eiffel | Auto-checked |
| Tool Use | calculate | Auto-checked |

## Aptitude batteries

Batteries A–F run after the standard suite on models that qualify; Vision and Embedding are **capability-routed** (selected by a model's `capabilities` array in `models.json`, not gated by the standard suite).

| Battery | Role / routing | What it measures |
|---------|----------------|-----------------|
| A | Router | Classification accuracy, brevity, prompt weight, false-escalation rate |
| B | Worker — Chat | Personality ceiling, consistency, multi-turn depth, think toggle |
| C | Worker — Research | JPEG signal coverage (think on/off), RAG depth, synthesis, ctx ladder, token ceiling |
| D | Worker — Tool-heavy | Tool chains, error recovery, partial failure handling, personality + tool integration |
| E | Completion (worker + router) | Coding, execution-graded — generate, debug, multi-language (JS/SQL/PHP), test-writing, constraints, HTML/CSS quality. 3-run averaged; clears threshold → earns the `coder` extended role |
| F | Worker — Chat | Conversational consistency across multi-turn runs (within-run-relative; reports σ) |
| V | `vision` capability | Vision/OCR vs PIL ground truth — ocr, count, chart, spatial, describe |
| EMB | `embedding` capability | Embedding quality — STS, triplet, retrieval (length-stratified), clustering; quality-per-GB |

Use `--capable-only` with Batteries C, D, and E to automatically skip models that failed the tool-use test in the most recent standard run.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running at `http://localhost:11434`
- `pip install requests aiohttp beautifulsoup4 html5lib tinycss2`
- Optional Battery E runtimes (graceful-skip if absent): `node` (JS), `php` (PHP), `tidy` (HTML validity); SQL/Python are built in

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
| `aptitude.py --battery E` | `aptitude_e_YYYY-MM-DD.json` + `.md` |
| `vision.py` | `vision_YYYY-MM-DD.json` + `.md` |
| `embedding.py` | `embedding_YYYY-MM-DD.json` + `.md` |

The canonical ranking table lives at `rankings/master.md`; `export.py` produces the machine-readable, already-pruned `rankings/rankings.json` (and publishes it for consumers).

## Syncing the model registry

```bash
python3 update_registry.py        # query localhost:11434 and update models.json
python3 update_registry.py --keep-missing   # retain entries not installed locally
python3 update_registry.py --dry-run
```

New models are added as `role: "worker"` automatically. After running the standard suite, `runner.py` applies the role gate and promotes qualifying models to `router` — no manual editing required. Models no longer installed are pruned by default (pass `--keep-missing` to retain them, e.g. when syncing against a remote Ollama host).
