"""
BenchLLAMA — shared utilities
Imported by runner.py, ctx_ladder.py, and aptitude.py.
"""

import hashlib
import os
import platform
import re
import subprocess
import time
from pathlib import Path

import requests

_REPO = Path(__file__).resolve().parent


# ── Terminal styling (TTY/env-gated; safe to import anywhere) ──────────────────
# ANSI is emitted ONLY for an interactive terminal. When stdout is a pipe (the
# orchestrator captures every subprocess over a PIPE → the run-log file + the web
# dashboard), COLOR is False, so those surfaces stay plain — no escape-code litter.
# Overridable: BENCH_COLOR=1 forces on, BENCH_COLOR=0 / NO_COLOR forces off.

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_CODES = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
    "blue": "\033[34m", "cyan": "\033[36m",
    "orange": "\033[38;5;208m", "grey": "\033[38;5;245m",
}


def _color_enabled() -> bool:
    force = os.environ.get("BENCH_COLOR")
    if force is not None:
        return force not in ("0", "", "false", "no")
    if os.environ.get("NO_COLOR") is not None:
        return False
    try:
        import sys
        return sys.stdout.isatty()
    except Exception:
        return False


COLOR = _color_enabled()


def paint(text: str, *styles: str) -> str:
    """Wrap text in ANSI styles when COLOR is on; a plain no-op otherwise."""
    if not COLOR or not styles:
        return text
    pre = "".join(_CODES.get(s, "") for s in styles)
    return f"{pre}{text}{_CODES['reset']}"


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# ── Run provenance / environment fingerprint ──────────────────────────────────
# Captured ONCE at run start so a score delta can be attributed to the right cause:
# runtime (ollama_version) · harness (benchllama_commit) · model weights
# (model_digests) · test set (datasets) · OS/Metal (os/hardware). Best-effort, per
# field — NEVER raises into the caller (a probe failure must not break a benchmark).

# Scoring-relevant inputs whose content silently changes results — hashed so a
# test-set/prompt edit is visible in provenance. Missing files are skipped.
_DATASET_FILES = {
    "coding_problems":   "suites/coding/problems.json",
    "elasticity_ladder": "suites/elasticity/ladder.json",
    "longctx":           "suites/longctx/dataset.json",
    "vision_gt":         "suites/vision/ground_truth.json",
    "confab_items":      "suites/confab/items.json",
    "prompt_worker":     "prompts/worker_default.md",
    "prompt_router":     "prompts/router_default.md",
}

# ── Content-addressed resume: test-identity per battery (see docs/resume-spec.md) ──
# BATTERY_REVISION — bump the int ONLY when you materially change a battery's scoring/composition
# in CODE (a change dataset hashes can't see, e.g. a new weight, a two-band split, a scorer rewrite).
# A bump re-runs that battery for every model next run. Do NOT bump for cosmetic edits.
BATTERY_REVISION = {
    "standard": 1, "ladder": 1,
    "A": 1, "B": 1, "C": 1, "D": 1,
    "E": 2,           # two-band E-hard (2026-07-02)
    "F": 1,
    "F-elastic": 1,
    "G": 2,           # two-band G-hard: decoys + superseded/aggregate/absent; gated clean_depth (2026-08-21)
    "vision": 2,      # two-band V-hard (2026-07-05)
    "embedding": 2,   # length-stratified re-tune (2026-06-13)
    "image": 1,
    "confab": 1,      # Battery H — honesty/confabulation (2026-07-07)
}

# ── Battery G — grading bars (see longctx.py) ─────────────────────────────────
# Deliberately NOT stored in suites/longctx/dataset.json: the dataset hash is a resume trigger, so
# putting a grading POLICY there would force a 2h re-measure of 22 models every time a bar moves —
# even though the model responses are already on disk and would be byte-identical. Bars live here;
# `longctx.py --rescore` re-derives every stored summary from the persisted per-depth hits instead.
# bench_utils.py is not a hashed dataset file, so editing these does NOT re-run the battery.
#
# G_CORE: 2 of 3 positional needles. ⚠ keep strictly BELOW 0.667 — per-depth accuracies are stored
#         ROUNDED, so 2/3 lands on 0.667 and a bar written as the "obvious" 0.67 silently means 3-of-3.
# G_HARD: 3 of 4 discriminators (raised from 0.50 on 2026-08-22). At 2-of-4 the band leaked: the
#         2026-08-22 fleet run measured `absent` at 0.984 mean recall (20/21 perfect) and
#         `superseded` at 0.960 (18/21) — those two alone satisfied a 2-of-4 bar, so a model could
#         fail BOTH real discriminators (`aggregate` 0.611, `multihop` 0.667) and still report
#         clean-32k. 15/21 did. At 3-of-4 that falls to 11/21 across 6 tiers, and clearing the bar
#         now REQUIRES at least one of aggregate/multihop. Raised together with the one-dip
#         tolerance in summarize() — a tighter bar makes the old break-on-first-failure walk
#         brittle, and the two changes are only correct as a pair.
G_CORE_THRESHOLD = 0.66
G_HARD_THRESHOLD = 0.75

# Which dataset/prompt hashes (keys of _DATASET_FILES) actually feed each battery's result.
# A change to one of these = a test-data change → re-run. Batteries not listed / with [] rely on
# BATTERY_REVISION alone (their test data isn't a hashed file — e.g. F rollout, EMB seed sets).
BATTERY_DATASETS = {
    "standard": ["prompt_worker", "prompt_router"],
    "ladder":   ["prompt_worker", "prompt_router"],
    "A": ["prompt_router"], "B": ["prompt_worker"], "C": ["prompt_worker"], "D": ["prompt_worker"],
    "E": ["coding_problems"],
    "F": [],
    "F-elastic": ["elasticity_ladder"],
    "G": ["longctx"],
    "vision": ["vision_gt"],
    "embedding": [],
    "image": [],
    "confab": ["confab_items"],
}


def _sh(*argv) -> str | None:
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _benchllama_commit() -> str | None:
    sha = _sh("git", "-C", str(_REPO), "rev-parse", "--short", "HEAD")
    if not sha:
        return None
    dirty = _sh("git", "-C", str(_REPO), "status", "--porcelain")
    return sha + ("-dirty" if dirty else "")


def _dataset_hashes() -> dict:
    out = {}
    for key, rel in _DATASET_FILES.items():
        p = _REPO / rel
        try:
            out[key] = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
        except Exception:
            pass  # absent (optional dataset) — omit rather than error
    return out


def _os_hardware() -> tuple[dict, dict]:
    osd = {"name": platform.system(), "kernel": platform.release()}
    try:
        mac = platform.mac_ver()[0]
        if mac:
            osd["version"] = mac
    except Exception:
        pass
    hw = {"cores": os.cpu_count()}
    chip = _sh("sysctl", "-n", "machdep.cpu.brand_string") or _sh("sysctl", "-n", "hw.model")
    if chip:
        hw["chip"] = chip
    mem = _sh("sysctl", "-n", "hw.memsize")
    if mem and mem.isdigit():
        hw["ram_gb"] = round(int(mem) / (1024 ** 3))
    return osd, hw


def _model_digests(host: str, only: set | None = None) -> dict:
    try:
        tags = requests.get(f"{host}/api/tags", timeout=10).json().get("models", [])
    except Exception:
        return {}
    out = {}
    for m in tags:
        name = m.get("name") or m.get("model")
        dig = m.get("digest")
        if name and dig and (only is None or name in only):
            out[name] = dig[:16]
    return out


def _live_battery_revisions() -> dict:
    """BATTERY_REVISION read FRESH from this file's source, not from the caller's import-time copy.

    webserver.py is long-lived and captures the run-start fingerprint in-process (orchestrator.py),
    so it snapshots whatever bench_utils looked like when the server booted. Bump a revision during
    a session and the run records the OLD number — which means the bump is never consumed and that
    battery re-arms on every subsequent run. Battery G hit exactly this on 2026-08-21: the run that
    produced the v2 two-band results stamped `G: 1`, so a full 22-model re-run stayed pending
    against results that were already v2. The scoring subprocesses import fresh code, so only this
    one capture was stale.

    Parsed with ast (literal only, no import, no side effects); falls back to the in-memory dict.
    """
    try:
        import ast
        tree = ast.parse(Path(__file__).read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(tgt, ast.Name) and tgt.id == "BATTERY_REVISION" for tgt in node.targets):
                val = ast.literal_eval(node.value)
                if isinstance(val, dict) and val:
                    return val
    except Exception:
        pass
    return dict(BATTERY_REVISION)


def env_fingerprint(host: str = "http://localhost:11434", models=None) -> dict:
    """Run-provenance snapshot: ollama runtime, harness commit, model weight digests,
    dataset/prompt hashes, and structured OS/hardware. Best-effort — a failed probe
    yields a missing key, never an exception. `models` (names) filters the digest map."""
    try:
        ver = requests.get(f"{host}/api/version", timeout=10).json().get("version")
    except Exception:
        ver = None
    osd, hw = _os_hardware()
    return {
        "ollama_version":    ver,
        "benchllama_commit": _benchllama_commit(),
        "datasets":          _dataset_hashes(),
        "battery_revisions": _live_battery_revisions(),  # content-addressed resume: test-code identity
        "model_digests":     _model_digests(host, set(models) if models else None),
        "os":                osd,
        "hardware":          hw,
        "captured_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ── Model sort order (shared run + dashboard sort) ────────────────────────────
def sort_key(default: str = "size") -> str:
    """Sort key for the run/display order, from env BENCH_SORT (the orchestrator sets it from the
    dashboard; or export BENCH_SORT=name for a CLI run). One of: size | name | fresh.
    Defaults to size (smallest disk first)."""
    k = (os.environ.get("BENCH_SORT") or default).strip().lower()
    return k if k in ("size", "name", "fresh") else default


def sort_registry(models: list, key: str | None = None) -> list:
    """Return a NEW list of model dicts sorted by `key` (or env BENCH_SORT):
      size  → disk_gb ascending, cloud/null disk last  (default)
      name  → name A-Z (case-insensitive)
      fresh → install order, newest first (added_idx asc, stamped when models.json was size-sorted).
    Stable + non-mutating."""
    key = key or sort_key()
    if key == "name":
        return sorted(models, key=lambda m: (m.get("name") or "").lower())
    if key == "fresh":
        return sorted(models, key=lambda m: (m.get("added_idx", -1), (m.get("name") or "").lower()))
    return sorted(models, key=lambda m: (m.get("disk_gb") if m.get("disk_gb") else float("inf"),
                                         (m.get("name") or "").lower()))

# ── Thermal monitoring (Apple Silicon / macOS Sequoia+) ───────────────────────
#
# Uses: sudo powermetrics -n 1 -i 200 --samplers thermal
#
# The legacy 'smc' sampler (which gave die temperatures) was removed in
# macOS Sequoia. The 'thermal' sampler exposes pressure levels instead:
#   Nominal → no throttling (target)
#   Moderate → some throttling
#   Heavy → significant throttling
#   Tripping → emergency (rare)
#
# Requires passwordless sudo for powermetrics. One-time setup:
#   echo "$(whoami) ALL=(root) NOPASSWD: /usr/bin/powermetrics" \
#     | sudo tee /etc/sudoers.d/benchllama
#
# Without that, falls back to timer-only cooldown.

PRESSURE_TARGET = "Nominal"   # thermal pressure level to declare cool
TEMP_SUSTAIN    = 20          # s — must hold target pressure continuously before proceeding
TEMP_POLL       = 5           # s — interval between thermal checks


def _read_thermal_pressure():
    """Returns thermal pressure level string (Nominal/Moderate/Heavy/Tripping),
    or None if unavailable (sudo not cached, powermetrics error, etc.)."""
    try:
        r = subprocess.run(
            ["sudo", "-n", "powermetrics", "-n", "1", "-i", "200", "--samplers", "thermal"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0:
            return None
        for line in r.stdout.splitlines():
            m = re.search(r"Current pressure level:\s+(\w+)", line)
            if m:
                return m.group(1)
        return None
    except Exception:
        return None


def cooldown(max_seconds, label=""):
    """Cool-down between benchmark models.

    Smart path (powermetrics available):
      Polls every TEMP_POLL seconds. Once thermal pressure is PRESSURE_TARGET
      (Nominal) continuously for TEMP_SUSTAIN seconds, exits early. If pressure
      rises again during the sustain window, the sustain clock resets.
      If max_seconds elapses first, proceeds regardless.

    Fallback (powermetrics unavailable):
      Plain countdown timer — same behaviour as the original implementation.
    """
    if max_seconds <= 0:
        return

    tag = f" [{label}]" if label else ""

    # Probe: is thermal monitoring available?
    pressure = _read_thermal_pressure()
    smart = pressure is not None

    if not smart:
        print(
            f"\n  ⏱  Cool-down{tag}: {max_seconds}s  "
            f"(thermal monitoring unavailable — "
            f"see CLAUDE.md → Setup for passwordless powermetrics)",
            flush=True,
        )
        step, remaining = 30, max_seconds
        while remaining > 0:
            wait = min(step, remaining)
            time.sleep(wait)
            remaining -= wait
            if remaining > 0:
                print(f"     {remaining}s remaining", flush=True)
        print("  Cool-down complete.\n", flush=True)
        return

    print(
        f"\n  ⏱  Cool-down{tag}  "
        f"target: {PRESSURE_TARGET} thermal pressure × {TEMP_SUSTAIN}s  "
        f"(hard limit {max_seconds}s)",
        flush=True,
    )

    t_start     = time.time()
    t_on_target = None  # timestamp when pressure first reached target

    while True:
        elapsed   = time.time() - t_start
        remaining = max(0.0, max_seconds - elapsed)

        if elapsed >= max_seconds:
            pressure = _read_thermal_pressure()
            lvl = pressure or "unknown"
            print(f"\n  Timer expired ({max_seconds}s)  pressure: {lvl} — proceeding.\n", flush=True)
            break

        pressure = _read_thermal_pressure()
        lvl      = pressure or "unknown"

        if pressure == PRESSURE_TARGET:
            if t_on_target is None:
                t_on_target = time.time()
            sustained = time.time() - t_on_target
            print(
                f"  {lvl}  ✓ sustained {int(sustained)}/{TEMP_SUSTAIN}s  [{int(remaining)}s left]",
                flush=True,
            )
            if sustained >= TEMP_SUSTAIN:
                print(
                    f"\n  Cool-down complete — {PRESSURE_TARGET} pressure "
                    f"held for {TEMP_SUSTAIN}s.\n",
                    flush=True,
                )
                break
        else:
            if t_on_target is not None:
                t_on_target = None  # pressure rose again — reset sustain clock
            print(f"  {lvl}  [{int(remaining)}s left]", flush=True)

        time.sleep(TEMP_POLL)


# ── Pre-flight check ──────────────────────────────────────────────────────────

def preflight(models, host):
    """Warn about models not installed or lacking required capabilities on host.

    models: iterable of (name, ...) tuples — only the first element (name) is used.
    """
    try:
        tags = requests.get(f"{host}/api/tags", timeout=10).json()
    except Exception:
        return  # Ollama unreachable — let the benchmark surface the error

    installed = {}
    for m in tags.get("models", []):
        name = m["name"]
        try:
            show = requests.post(f"{host}/api/show", json={"name": name}, timeout=10).json()
            installed[name] = set(show.get("capabilities", []))
        except Exception:
            installed[name] = set()

    warn = []
    for name, *_ in models:
        if name not in installed:
            warn.append(f"  ⚠  {name}: not installed on {host}")
        elif "tools" not in installed[name]:
            warn.append(f"  ⚠  {name}: no 'tools' capability — calculate test will fail")

    if warn:
        print("\nPre-flight check:")
        for w in warn:
            print(w, flush=True)
        print()


# ── Cross-day resume source ───────────────────────────────────────────────────

def latest_result(results_dir, prefix, fast, hours):
    """Most recent results/<prefix>_<date>[_fast].json within `hours` (by mtime), or None.

    Enables cross-day resume. Output filenames embed the date, so checking only
    today's file can never see yesterday's run — the 'resume within N hours'
    promise would be dead across midnight. This scans every matching file and
    returns the newest one inside the window. `fast` selects the _fast variant
    (True → only *_fast.json; False → only the non-fast files).
    """
    best, best_m = None, -1.0
    for f in results_dir.glob(f"{prefix}_*.json"):
        name = f.name
        if "status" in name:
            continue
        if name.endswith("_fast.json") != fast:
            continue
        m = f.stat().st_mtime
        if (time.time() - m) / 3600 < hours and m > best_m:
            best, best_m = f, m
    return best


# ── Output-budget starvation guard (2026-08-24) ───────────────────────────────
#
# A model that emits a reasoning channel spends the SAME `num_predict` budget on its thinking as on
# its answer. When the budget runs out mid-thought the reply comes back with `done_reason: "length"`
# and an EMPTY `content` — which every grader in this repo scores as a wrong answer rather than as a
# failed call.
#
# This is not hypothetical and it is not cosmetic. `gpt-oss:120b-cloud` **ignores `think: False`**
# (verified 08-24: identical content with think false/true/omitted), so it always reasons. Battery G
# allowed 320 tokens; the model needs 413. It returned `content: ""` at all six depths, scored 0/7
# six times, and published as `0.000` — LAST of 19 in `rankings.long_context`. Re-measured at 1024 it
# scores **7/7**, sweeping both bands. The worst-published long-context model in the fleet was
# actually one of the best. Battery H lost 5 of 9 items the same way; `expense_split` lost its answer
# to a 1800-token cap against an 8644-token need.
#
# Retrying at a larger budget is preferred over simply raising every cap: the caps exist to stop
# runaway generation from wrecking wall-clock, and only the starved calls should pay for headroom.
# The retry is RECORDED (never silent) so a reader can tell a re-budgeted score from a first-try one.

# Factor 8, not 4. Measured need on 2026-08-24: Battery G 413 tokens against a 320 cap (4x is
# plenty), but `expense_split` took 8644 against an 1800 cap — 4x reaches only 7200 and still fails,
# burning a retry to record the same empty answer. num_predict is a CEILING, not a target: a model
# that finishes early still stops early, so the factor costs nothing on calls that recover quickly.
#
# ⚠ The ceiling matters anyway because the retry grows `num_ctx` by the same amount, and a GGUF
# model pre-allocates its full KV cache (Protocol Rule #1). BUDGET_RETRY_CAP bounds that blast
# radius: worst case a local model is asked for ~12k extra context on the exception path only.
BUDGET_RETRY_FACTOR = 8        # multiply num_predict by this on a starved call
BUDGET_RETRY_CAP    = 12000    # absolute ceiling; expense_split needed 8644


def starved_on_length(data: dict) -> bool:
    """True when a reply terminated on the output cap having emitted no `content`.

    The tell is `done_reason == "length"` together with empty content: the model was still going
    when the budget ran out, and everything it produced went to a channel the grader never sees.
    A short-but-present answer is NOT starvation — the model chose to stop.
    """
    if data.get("done_reason") != "length":
        return False
    return not (data.get("message", {}).get("content") or "").strip()


def post_with_budget_retry(payload: dict, post, *, label: str = "", quiet: bool = False):
    """POST `payload` via `post(payload) -> (data, wall)`, retrying ONCE if the reply was starved.

    Returns `(data, wall)` from whichever attempt is authoritative. When a retry happens the
    returned `data` carries `_budget_retry` describing it, so result writers can surface the fact:

        {"reason": "empty_content_on_length", "num_predict": [320, 1280],
         "num_ctx": [2560, 3520], "recovered": True}

    ⚠ The retry grows `num_ctx` BY THE SAME AMOUNT as `num_predict`, and that is not optional.
    A battery sizes its window for the prompt plus a SHORT answer — Battery G uses
    `num_ctx = bucket + 1536`, which at bucket 1024 leaves ~1284 tokens of generation room once the
    haystack is in. Raising `num_predict` past that is a no-op: the context window binds first, the
    reply is truncated anyway, and `done_reason` still reads "length". Measured directly on
    2026-08-24 — a retry at num_predict=1280 inside an unchanged num_ctx=2560 failed exactly as the
    original 320 did. Growing both is what makes the retry mean anything.

    `wall` is the AUTHORITATIVE attempt's own wall time, not the sum — it is used as a performance
    number, and the honest cost of an answer from this model is the call that actually produced one.
    """
    data, wall = post(payload)
    if not starved_on_length(data):
        return data, wall

    opts = payload.get("options") or {}
    old_p = opts.get("num_predict")
    if not old_p:                                # uncapped already — a retry cannot help
        return data, wall
    new_p = min(int(old_p) * BUDGET_RETRY_FACTOR, BUDGET_RETRY_CAP)
    if new_p <= old_p:
        data["_budget_retry"] = {"reason": "empty_content_on_length",
                                 "num_predict": [old_p, old_p], "recovered": False,
                                 "note": "already at BUDGET_RETRY_CAP"}
        return data, wall

    new_opts = {**opts, "num_predict": new_p}
    old_c = opts.get("num_ctx")
    if old_c:                                    # give the window the extra output room too
        new_opts["num_ctx"] = int(old_c) + (new_p - int(old_p))

    if not quiet:
        print(f"    ⚠  {label or payload.get('model','?')}: empty content at num_predict={old_p} "
              f"(done_reason=length) — retrying at {new_p}"
              + (f", num_ctx {old_c}→{new_opts['num_ctx']}" if old_c else ""), flush=True)

    data2, wall2 = post({**payload, "options": new_opts})
    recovered = bool((data2.get("message", {}).get("content") or "").strip())
    data2["_budget_retry"] = {"reason": "empty_content_on_length",
                              "num_predict": [old_p, new_p],
                              "num_ctx": [old_c, new_opts.get("num_ctx")] if old_c else None,
                              "recovered": recovered}
    if not quiet and not recovered:
        print(f"    ⚠  {label or payload.get('model','?')}: STILL empty at {new_p} — recording as failed call",
              flush=True)
    return data2, wall2


# ── Battery F-elastic verdict cutoffs (2026-08-24) ────────────────────────────
#
# These live HERE, not in suites/elasticity/ladder.json, for the same reason G_CORE_THRESHOLD does:
# ladder.json is HASHED as a resume trigger, so a bar stored there forces a pointless 1h44m fleet
# re-measure every time a threshold is re-tuned. A grading bar is policy, not test content — the
# rollouts and the per-constraint hits are already on disk and would come back identical.
# ladder.json still carries a `verdict_cutoffs` block; it is DOCUMENTATION, not authority.
# Re-apply a change offline with `python3 aptitude.py --battery F-elastic --regate`.
#
# CORE cutoffs — UNCHANGED since the 2026-06-22 calibration (20 models x 3-run, cross-family).
#   sigma_hi 0.10 / adherence_hi 0.80 / adherence_lo 0.35, keyed on instruction_adherence.
#
# HARD cutoff — 0.75, calibrated 2026-08-24 on 19 models x 3-run average, replacing the PROVISIONAL
# 0.60 placeholder. At 0.60 the hard band changed ZERO verdicts: 18 of 19 models cleared it and the
# one that did not (qwen2.5vl:3b) already failed the core band, so requiring both bands reproduced
# the core-only verdict list exactly.
#
# ⚠ The bar was not the problem — the SCOPE was. `hard_adherence` was the mean of every binary
# constraint on the hard rung, four of which are the saturated core ones (`no_exclamation` 1.000
# field-wide, `no_lists` 0.963, `end_with_question` 0.873, `required_prefix` 0.754). The three real
# discriminators carried 3 of 7. Scoping the meter to the constraints the hard band ADDS is the fix;
# raising the bar over a diluted meter could not be. Same defect shape as the coder gate, where
# E-hard was 15% of the composite.
#
# 0.75 sits in the natural gap between gpt-oss:120b-cloud 0.708 (run-sigma 0.000) and
# minicpm-v4.6:1b 0.778 (run-sigma 0.020) — both ends of the gap are stable across three runs, which
# is what a cutoff needs. Splits the fleet 14 pass / 5 fail.
#
# ⚠ Calibrated on 19 models, one short of the 20+ bar the core cutoffs met, and gemma-heavy (7 of
# 19). Treat the hard verdict as provisional-but-anchored until the roster is back over 20.
F_ELASTIC_CUTOFFS = {
    "sigma_hi":          0.10,
    "adherence_hi":      0.80,
    "adherence_lo":      0.35,
    "hard_adherence_hi": 0.75,
    "keyed_on":          "instruction_adherence",
    "_hard_scope":       "hard-band-only (constraints the hard rung ADDS over the core rungs)",
    "_status":           "core CALIBRATED 2026-06-22 (20 models x 3-run); hard CALIBRATED 2026-08-24 "
                         "(19 models x 3-run) — was PROVISIONAL 0.60 over a diluted meter that moved "
                         "no verdicts. REQUIRES 3-run averaging.",
}
