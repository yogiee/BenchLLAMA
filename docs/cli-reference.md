# BenchLLAMA — CLI Reference

All commands are invoked via `./bench.sh <command> [flags]`.

---

## Commands

### `standard`
Runs the standard benchmark suite: 13 tests across 5 dimensions (personality, reasoning, research depth, instruction following, tool use).

```bash
./bench.sh standard
./bench.sh standard new-model:tag          # single model, merges into existing JSON
./bench.sh standard --fast                 # skip cool-down (informal results)
./bench.sh standard --force                # re-run all models, ignore 72h resume window
```

Output: `results/benchmark_YYYY-MM-DD.json` + `.md`

**Resume behaviour:** Resumes within 72 hours by default — models already present in today's output file are skipped. Pass `--force` to re-run everything. If you pass specific model names as positional arguments, those models are merged into the existing file (existing results for other models are untouched).

---

### `ladder`
Runs the `num_ctx` characterisation pass. Sweeps each model across four context window sizes and measures RAM, tok/s, and quality at each level. Run this before aptitude batteries to find the optimal `num_ctx` per model.

```bash
./bench.sh ladder
./bench.sh ladder new-model:tag            # single model, merges into existing JSON
./bench.sh ladder --role router            # filter to router models only
./bench.sh ladder --fast
./bench.sh ladder --force
```

Output: `results/ctx_ladder_YYYY-MM-DD.json` + `.md`

**Resume behaviour:** Same 72h window as `standard`. Merge mode on targeted runs.

---

### `aptitude`
Runs a single aptitude battery against worker or router models. Battery B is the default.

```bash
./bench.sh aptitude                                         # Battery B (default)
./bench.sh aptitude --battery A                             # Battery A (router)
./bench.sh aptitude --battery C --capable-only              # Battery C, tool-capable models only
./bench.sh aptitude --battery D --capable-only
./bench.sh aptitude --battery B --system-prompt ~/my.md     # custom system prompt
./bench.sh aptitude --force                                 # re-run, disable 24h resume
```

Output: `results/aptitude_<battery>_YYYY-MM-DD.json` + `.md`

**Resume behaviour:** Resumes within 24 hours. `--force` disables resume and overwrites.

---

### `batteries`
Runs all four aptitude batteries in sequence: A → B → C → D. Batteries C and D automatically apply `--capable-only`.

```bash
./bench.sh batteries
./bench.sh batteries --force               # re-run all batteries
./bench.sh batteries --fast
```

**Resume behaviour:** Each battery independently resumes within 24h. `--force` disables all four.

---

### `all`
Full pipeline: standard → ladder → Battery A → B → C → D.

```bash
./bench.sh all
./bench.sh all --fast                      # skip cool-downs throughout
./bench.sh all --force                     # re-run everything, ignore all resume windows
```

**Resume behaviour:** Each script applies its own resume logic independently. `--force` propagates to all scripts.

---

### `update`
Syncs `models.json` from the local Ollama instance. Queries `/api/tags` for model names and sizes, and `/api/show` for capabilities (tools support). New completion models are added with `role: "worker"`. Non-completion models (image, embedding) are listed but never benchmarked.

```bash
./bench.sh update
./bench.sh update --dry-run                # preview without writing
./bench.sh update --ollama http://host:11434
```

---

## Flags

| Flag | Applies to | Description |
|------|-----------|-------------|
| `--fast` | all | Skip inter-model cool-down. Results are labeled informal. |
| `--force` | standard, ladder, aptitude, batteries, all | Disable resume logic. Re-runs all models and overwrites output. |
| `--capable-only` | aptitude (C, D) | Filter to models that passed `calculate` in the most recent standard run. Auto-applied by `batteries` and `all` for batteries C and D. |
| `--battery A\|B\|C\|D` | aptitude | Select which battery to run (default: B). |
| `--role router\|worker` | ladder, aptitude | Filter models by role. |
| `--system-prompt <path>` | standard, aptitude | Override the worker system prompt. |
| `--system-prompt-router <path>` | standard | Override the router system prompt. |
| `--ollama <url>` | all | Target a remote Ollama instance (default: `http://localhost:11434`). |

---

## Resume behaviour summary

| Script | Default resume window | Override |
|--------|-----------------------|---------|
| `runner.py` (standard) | 72 hours | `--force` |
| `ctx_ladder.py` (ladder) | 72 hours | `--force` |
| `aptitude.py` (any battery) | 24 hours | `--force` |

When the resume window is active and no `--force` is set, models already present in the output JSON are skipped. A model with `"error"` in its result is not counted as completed — it will be re-run automatically.

**Roster change warning:** When `standard` or `ladder` resumes a full run, it compares `models.json` against the existing results. If new models are found that aren't in the last run, a warning is printed with a targeted run suggestion:

```
⚠ 2 new model(s) in models.json not in last run: ['phi4:latest', 'qwen3:8b']
  → './bench.sh standard phi4:latest qwen3:8b' to add them, or '--force' to rebaseline.
```

**Merge mode:** When positional model name arguments are passed to `standard` or `ladder`, those models are run and their results are merged into the existing output file. Other models' results are untouched. This is the recommended way to add a new model without re-running the full suite.

```bash
# Pull a new model, register it, then add it to existing results
ollama pull phi4:latest
./bench.sh update
./bench.sh standard phi4:latest
./bench.sh ladder phi4:latest
```

---

## Batteries quick reference

| Battery | Role | Tests | Key measurements |
|---------|------|-------|-----------------|
| A | Router | 4 | Classification accuracy, brevity, prompt weight sensitivity, false escalation rate |
| B | Worker — Chat | 5 | Personality ceiling, consistency across phrasings, multi-turn depth, prompt weight, think toggle |
| C | Worker — Research | 6 | JPEG signal coverage with/without think, RAG depth, synthesis, ctx depth ladder, token budget ceiling |
| D | Worker — Tool-heavy | 8 | Tool chaining, selection accuracy, error recovery, partial failure handling, think+tools, parallel calls, personality+tools, deep cart order |

See [docs/](.) for full per-test documentation.
