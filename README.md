<p align="center">
  <img src="assets/logo.png" width="360" alt="BenchLLAMA"/>
</p>

<h1 align="center">BenchLLAMA</h1>
<p align="center">Standalone benchmark harness for local Ollama models</p>

---

BenchLLAMA runs structured, repeatable benchmarks against any model served by [Ollama](https://ollama.com). It measures personality, reasoning, research depth, instruction following, tool use, coding, consistency, honesty (confabulation), long-context retrieval, vision, embedding, and image generation — then produces ranked results you can act on, served live in a browser dashboard.

## What's inside

| Script | Purpose |
|--------|---------|
| `runner.py` | Standard suite — 13 tests across 5 dimensions |
| `ctx_ladder.py` | Context window characterisation — finds optimal `num_ctx` per model |
| `aptitude.py` | Role-specific batteries (A–F, F-elastic) — deep evaluation for router and worker models |
| `longctx.py` | Battery G — long-context retrieval & degradation (planted needles + prefill/decode speed-collapse) |
| `confab.py` | Battery H — honesty/confabulation probe (bare fabrication rate vs real-item accuracy, LLM-judge graded) |
| `vision.py` | Battery V — capability-routed vision/OCR evaluation |
| `embedding.py` | Battery EMB — capability-routed embedding evaluation |
| `imagegen.py` | Battery I — capability-routed image-gen perf + prompt-adherence (reference-only, opt-in) |
| `export.py` | Neutral rankings exporter — aggregates every battery's latest results into `rankings/rankings.json` |
| `orchestrator.py` | Headless orchestration core + plain-text console runner |
| `webserver.py` | Web UI server (aiohttp) — drives the orchestrator, serves the live dashboard over WebSocket |
| `web/index.html` | The browser dashboard — phase tree, model cards, streamed log, Rankings + Files + Image Review |
| `resume.py` | Content-addressed resume policy (shared by every battery) + `--resume-report` dry planner |
| `bench_utils.py` | Shared utilities — smart thermal cooldown, pre-flight checks, run-provenance fingerprint |

## Quick start

```bash
pip install requests aiohttp beautifulsoup4 html5lib tinycss2

# Web dashboard (default) — compose a run in the browser (suite cards + model picker), Start/Stop from the UI
./bench.sh                 # no command → browser run composer
./bench.sh all             # full pipeline in the dashboard

# Plain terminal instead of the browser (headless / SSH / quick glance)
./bench.sh all --console

# Full pipeline: standard → ctx ladder → A–D → E → F → G → Vision → Embedding
./bench.sh all

# Everything, incl. the opt-in batteries, in one unattended pass (single run-log):
#   --with-elastic  → append Battery F-elastic (prompt-σ)
#   --with-imagegen → append Battery I (image-gen characterisation)
./bench.sh all --with-elastic --with-imagegen

# Individual suites
./bench.sh standard
./bench.sh ladder
./bench.sh aptitude --battery B
./bench.sh aptitude --battery D --capable-only
./bench.sh aptitude --battery E       # coding — 3-run averaged by default
./bench.sh longctx                    # Battery G — long-context needles + speed-collapse
./bench.sh confab                     # Battery H — honesty/confabulation (opt-in; --grade llm default, --judge overridable)
./bench.sh vision                     # every vision-capable model
./bench.sh embedding                  # every embedding model
./bench.sh imagegen                   # Battery I — image-gen (opt-in, reference-only, slow)
./bench.sh export                     # re-export rankings/rankings.json (also a dashboard button)

# Fast mode (skip cooldown — informal results)
./bench.sh standard --fast
```

### Unattended full run

For a whole-fleet run you leave going (e.g. overnight), wrap it in `caffeinate` so macOS
doesn't sleep and kill it, and chain `export.py` to publish rankings when it finishes:

```bash
caffeinate -dimsu bash -c '
  ./bench.sh all --with-elastic --with-imagegen --console --force &&
  python3 export.py
' 2>&1 | tee results/overnight_$(date +%F).log
```

### Live dashboard

`./bench.sh` launches the aiohttp web UI (`webserver.py`) and serves `web/index.html` —
a single-file dashboard (orange-cream theme, light/dark/system) showing the phase tree,
per-model cards (role + extended-role badges, capability tags, tok/s), a streamed log,
and built-in **Rankings**, **Files**, and **Image Review** viewers.

Run selection is a **multi-select suite-card composer**: tick any combination of suites
and batteries — the **ALL** preset selects everything except the opt-ins (Honesty,
F-elastic); Aptitude expands into per-battery rows with a `runs ×N` control; Honesty
exposes grade + judge pickers; Image Gen runs solo-only. One morphing **Start ↔ Stop**
button drives the run; a finished run shows a per-phase ✓/✗ summary and offers
**⭱ Export Rankings**. Update Registry is an instant top button.
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

Batteries A–F run after the standard suite on models that qualify. Batteries G and H are completion-routed (like E). Vision, Embedding, and Image-gen are **capability-routed** — selected by a model's `capabilities` array in `models.json` (`vision` / `embedding` / `image`), not gated by the standard suite. F-elastic, Image-gen, and Honesty (H) are opt-in — never in `all`; append the first two with `--with-elastic` / `--with-imagegen`, run Honesty with `./bench.sh confab`.

| Battery | Role / routing | What it measures |
|---------|----------------|-----------------|
| A | Router | Classification accuracy, brevity, prompt weight, false-escalation rate |
| B | Worker — Chat | Personality ceiling, consistency, multi-turn depth, think toggle |
| C | Worker — Research | JPEG signal coverage (think on/off), RAG depth, synthesis, ctx ladder, token ceiling |
| D | Worker — Tool-heavy | Tool chains, error recovery, partial failure handling, personality + tool integration |
| E | Completion (worker + router) | Coding, execution-graded — generate, debug, multi-language (JS/SQL/PHP), test-writing, constraints, HTML/CSS quality. 3-run averaged; clears threshold → earns the `coder` extended role |
| F | Worker — Chat | Conversational consistency across multi-turn runs (within-run-relative; reports σ) |
| F-elastic | Completion (opt-in) | Prompt-elasticity — how the F-composite moves across a system-prompt complexity ladder; judge-free adherence meter. Never in `all`; append with `--with-elastic` |
| G | Completion (worker + router) | Long-context retrieval & degradation — fills the window (1k–16k+) with distractors + planted needles + a 3-hop chain; exact-match accuracy AND prefill/decode speed-collapse; headline `clean_depth` |
| H | Completion (opt-in) | Honesty / confabulation — planted fake **and** real items; measures bare fabrication rate vs real-item accuracy (so over-denial can't masquerade as honesty). LLM-judge graded by default (`--grade llm`, `--judge` overridable; deterministic `--grade signal` fallback). Never in `all`; run `./bench.sh confab` |
| V | `vision` capability | Vision/OCR vs PIL ground truth — ocr, count, chart, spatial, describe |
| EMB | `embedding` capability | Embedding quality — STS, triplet, retrieval (length-stratified), clustering; quality-per-GB |
| I | `image` capability (opt-in) | Image-gen characterisation — **perf + prompt-adherence, not quality**. Blind-VLM ✓-core checklist, OCR text-fidelity, per-image reliability (retries + unload/reload recovery). Reference-only, never ranked; append with `--with-imagegen` |

Use `--capable-only` with Batteries C, D, E, and G to automatically skip models that failed the tool-use test in the most recent standard run.

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
| `aptitude.py --battery F-elastic` | `aptitude_f_elastic_YYYY-MM-DD.json` + `.md` |
| `longctx.py` | `longctx_YYYY-MM-DD.json` + `.md` |
| `confab.py` | `confab_YYYY-MM-DD.json` + `.md` |
| `vision.py` | `vision_YYYY-MM-DD.json` + `.md` |
| `embedding.py` | `embedding_YYYY-MM-DD.json` + `.md` |
| `imagegen.py` | `imagegen_YYYY-MM-DD.json` + `.md` (+ PNGs under `results/imagegen_images/`) |

The canonical ranking table lives at `rankings/master.md`; `export.py` produces the machine-readable, already-pruned `rankings/rankings.json` (and publishes it for consumers).

## Syncing the model registry

```bash
python3 update_registry.py        # query localhost:11434 and update models.json
python3 update_registry.py --keep-missing   # retain entries not installed locally
python3 update_registry.py --dry-run
```

New models are added as `role: "worker"` automatically. After running the standard suite, `runner.py` applies the role gate and promotes qualifying models to `router` — no manual editing required. Models no longer installed are pruned by default (pass `--keep-missing` to retain them, e.g. when syncing against a remote Ollama host).

## Resume — content-addressed, not time-windowed

Add a few models, re-run `./bench.sh all`, and **only the new models run** — across *every* battery (coding, consistency, long-context, everything), not just the standard suite, and no matter how long ago the fleet last ran.

Resume is **content-addressed**: a `(model, battery)` result is re-run only if a determinant of that result changed since it was last scored — otherwise the stored score is carried forward. Triggers:

| Trigger | Behaviour |
|---------|-----------|
| **Model weights changed** (`/api/tags` digest) | re-run — always on |
| **Test changed** (a battery's dataset hash or its `BATTERY_REVISION`) | re-run — always on |
| **Ollama runtime bumped** (`major.minor`; patch ignored) | **off by default** — opt in with `--check-runtime` when a release note flags a real perf change |

Cloud models (no local weight digest) are skipped unless the test changed or you pass `--force` — this protects your Ollama usage. The diff is computed from the per-run provenance fingerprint already stored in the results DB (`env_fingerprint`: runtime version, per-model weight digests, dataset hashes).

```bash
./bench.sh --resume-report                        # dry run: which models would re-run, and WHY
./bench.sh aptitude --battery E --resume-report    # for one battery
./bench.sh all --check-runtime                     # also re-run models scored under an older Ollama minor
./bench.sh all --force                             # re-run everything, ignore resume
```

Full design: `docs/resume-spec.md`.
