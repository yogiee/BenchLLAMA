"""
BenchLLAMA — shared utilities
Imported by runner.py, ctx_ladder.py, and aptitude.py.
"""

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
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
    "think_probe":       "suites/think/probe.json",   # think_probe.py item set (provenance only — see BATTERY_DATASETS)
}

# ── Content-addressed resume: test-identity per battery (see docs/resume-spec.md) ──
# BATTERY_REVISION — bump the int ONLY when you materially change a battery's scoring/composition
# in CODE (a change dataset hashes can't see, e.g. a new weight, a two-band split, a scorer rewrite).
# A bump re-runs that battery for every model next run. Do NOT bump for cosmetic edits.
BATTERY_REVISION = {
    # v3 think-aware protocol (docs/think-spec.md, 2026-08-28): every completion battery now runs at
    # the model's think arm / lever-aware budget instead of a blanket think=False → full re-measure.
    "standard": 2, "ladder": 1,
    "A": 2, "B": 2, "C": 2, "D": 2,
    "E": 3,           # two-band E-hard (2026-07-02); v3 arms (2026-08-28)
    "F": 2,
    "F-elastic": 2,
    "G": 3,           # two-band G-hard (2026-08-21); v3 arms (2026-08-28)
    "vision": 2,      # two-band V-hard (2026-07-05)
    "embedding": 2,   # length-stratified re-tune (2026-06-13)
    "image": 1,
    "confab": 2,      # Battery H — honesty/confabulation (2026-07-07); v3 arms (2026-08-28)
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


# ── Think-aware protocol v3 (docs/think-spec.md, 2026-08-28) ──────────────────────────────
# Protocol Rule #2 ("think=False everywhere") rested on the output-budget starvation bug fixed on
# 2026-08-24, not on a model property: a hybrid-reasoning model spends the whole num_predict in the
# `thinking` channel and the grader sees empty content. Measured 2026-08-28: think:false makes
# granite4.2:3b/8b and qwen3.5:4b-mlx FAIL bat_ball (the fleet passes 22/22); granite's "low" fixes
# it in 76–98 tokens while its full think burns 4096 tokens and never answers; qwen3.5 needs 1857
# tokens and its levels are a no-op. The lever therefore has to be chosen PER MODEL, and the budget
# sized for it — that is what a `think_profile` (written by think_probe.py into models.json) does.
#
#   arm   direct — the no-thinking class (or, for always-on thinkers, the cheapest bounded class)
#         think — the model's operating point: the cheapest BOUNDED thinking class that scores best
#   lever the literal `think` value sent: False | "low" | "medium" | "high" | True | (omitted)
#
# Every writer calls apply_think(payload, model, arm) instead of hard-coding "think": False, and
# stamps arm_stamp(model, arm) onto its result so the DB row / export say which arm was measured.
# A model with NO profile gets the legacy behaviour (think=False; a 400 is stripped by the caller).
# ── Battery I (image generation) availability ─────────────────────────────────────
# Ollama 0.32.6 (2026-08-06) removed image generation, so Battery I cannot produce a single image. The
# code (imagegen.py, the Image Review tab, the /api/imagegen/* routes, suites/imagegen/) is KEPT but
# HIDDEN: every entry point checks this switch — `./bench.sh imagegen` (web boot + --console),
# `--with-imagegen`, the dashboard's /api/start units, and a direct `python3 imagegen.py`.
# Re-enable by flipping to True if and when Ollama image generation returns (and set the matching
# `IMAGEGEN_AVAILABLE` const in web/index.html, which un-hides the card + tab).
IMAGEGEN_AVAILABLE = False
IMAGEGEN_UNAVAILABLE_MSG = "Image generation is temporarily unavailable — this battery cannot run."

THINK_OMIT = object()                       # sentinel: send no `think` key at all (the "absent" lever)
THINK_LEVERS = ("absent", "false", "low", "medium", "high", "true")
THINK_ARMS = ("direct", "think")
THINK_PROBE_NUM_PREDICT = 4096              # the probe measures demand; it does not impose a budget
THINK_ALLOWANCE_K, THINK_ALLOWANCE_MIN, THINK_ALLOWANCE_MAX = 4, 1024, 8192    # ×2/512 under-shot: gemma4:e4b-mlx expense_split needed ~1.5k think tokens vs a 294-token probe max (smoke 2026-08-28)
# ⚠ INVARIANT: THINK_ALLOWANCE_MAX < BUDGET_RETRY_CAP. The allowance is the EXPECTED need; the ×8
# retry is the safety net for when the probe under-estimated. A net BELOW the allowance is not a net:
# post_with_budget_retry computes new_p = min(old_p×8, CAP) and bails when new_p <= old_p, returning
# the starved reply — and that branch PRINTS NOTHING, so the call renders as a bare `gate:empty`
# indistinguishable from a model that simply declined to answer. The ceiling was 16384 vs a 12000 cap
# from 2026-08-28 until 2026-08-30, which silently denied the retry to exactly two models:
# deepseek-r1:8b (probe max 4770 → clamped, BOTH arms → every battery) and bonsai-27b:1bit (think arm).
# Measured cost: deepseek-r1:8b lost 3 of 4 Battery-E E-hard tasks to unretried starvation at ~750 s
# each (E-hard 0.208 → coder_eligible False) while the retry was recovering 82% of the calls it caught
# fleet-wide (34 fires / 28 rescued, run 2026-08-29). 8192 still clears the largest thinking observed
# on a REAL task by ~1.7× (standard-suite maxima: deepcoder:1.5b 4525, deepseek-r1:8b 4206,
# deepcoder:14b 3642) — the probe items are tiny, which is what K=4 extrapolates from — and the retry
# to 12000 remains available above it. Raising this ceiling REQUIRES raising BUDGET_RETRY_CAP with it.

_LEVER_VALUES = {"false": False, "true": True, "low": "low", "medium": "medium", "high": "high"}
_PROFILE_CACHE: dict = {"mtime": None, "data": {}}


def lever_value(name):
    """Lever NAME (as stored in a profile) → the literal `think` value to send; 'absent' → THINK_OMIT.

    A name outside THINK_LEVERS (e.g. "xhigh", "max" — levels a model DECLARES via /api/show since Ollama
    0.34.3) is sent as the literal string. Until 2026-09-23 any unknown name silently became THINK_OMIT,
    so a declared-only level would have been measured as `absent` under another name."""
    if name is None or name is THINK_OMIT or str(name) in ("absent", ""):
        return THINK_OMIT
    return _LEVER_VALUES.get(str(name), str(name))


def normalize_declared(block):
    """Ollama 0.34.3+ `/api/show` → `thinking: {values: [...], default: ...}` → the same block with every
    value as a lever NAME (False → "false", True → "true", strings kept). None when the model declares
    nothing. ⚠ DECLARED ≠ MEASURED (2026-09-23): granite4.2:3b declares nothing yet its "low" is the
    fleet's most consequential lever; gemma4 declares [false, true] yet splits into 2–3 thinking classes;
    lfm2.5:8b declares [false] default false yet thinks by default. The declaration is a prior and an
    alarm for think_probe.py — never a substitute for probing."""
    if not isinstance(block, dict):
        return None
    vals = block.get("values")
    return {"values": [lever_name(v) for v in vals] if isinstance(vals, list) else [],
            "default": lever_name(block["default"]) if "default" in block else None}


def fetch_declared(host: str, model: str):
    """The model's normalized declared thinking block from /api/show, or None (undeclared / unreachable)."""
    try:
        r = requests.post(f"{host}/api/show", json={"model": model}, timeout=15)
        return normalize_declared(r.json().get("thinking"))
    except Exception:
        return None


def lever_name(value) -> str:
    """Inverse of lever_value — for stamping/printing."""
    if value is THINK_OMIT or value is None:
        return "absent"
    if value is False:
        return "false"
    if value is True:
        return "true"
    return str(value)


def think_profiles(path=None) -> dict:
    """{model: think_profile} from models.json — mtime-cached so the hot chat() path stays cheap.
    Models without a profile are absent from the map (→ legacy think=False behaviour)."""
    p = Path(path) if path else _REPO / "models.json"
    try:
        mt = p.stat().st_mtime
        if _PROFILE_CACHE["mtime"] != mt:
            reg = json.loads(p.read_text())
            _PROFILE_CACHE["data"] = {m["name"]: m["think_profile"] for m in reg
                                      if isinstance(m, dict) and m.get("think_profile")}
            _PROFILE_CACHE["mtime"] = mt
    except Exception:
        pass
    return _PROFILE_CACHE["data"]


def think_profile(model: str):
    return think_profiles().get(model)


def requested_arm(default: str = "auto", argv=None) -> str:
    """Which arm this INVOCATION measures: `--arm direct|think|auto` (argv) > env BENCH_ARM > default.
      direct — every model at its direct (no-thinking) arm (the standard suite's first pass; ctx ladder; Battery A)
      think — every model that HAS an operating point at its think arm; the rest are skipped
      auto  — each model at its operating point if it has one, else fast (the batteries)"""
    argv = sys.argv if argv is None else argv
    if "--arm" in argv:
        i = argv.index("--arm")
        if i + 1 < len(argv) and argv[i + 1] in THINK_ARMS + ("auto",):
            return argv[i + 1]
    env = (os.environ.get("BENCH_ARM") or "").strip().lower()
    return env if env in THINK_ARMS + ("auto",) else default


def resolve_arm(model: str, requested: str = "auto"):
    """The arm `model` is measured at for `requested` (see requested_arm). None = skip this model
    (a `think` pass on a model with no operating point)."""
    prof = think_profile(model) or {}
    has_op = bool(prof.get("operating_lever"))
    if requested == "direct":
        return "direct"
    if requested == "think":
        return "think" if has_op else None
    return "think" if has_op else "direct"


def think_lever(model: str, arm: str):
    """Literal `think` value for (model, arm). No profile → False (legacy protocol)."""
    prof = think_profile(model)
    if not prof:
        return False
    if arm == "think" and prof.get("operating_lever"):
        return lever_value(prof["operating_lever"])
    return lever_value(prof.get("direct_lever") or "false")


def think_class(model: str, arm: str):
    prof = think_profile(model)
    if not prof:
        return None
    return prof.get("operating_point") if (arm == "think" and prof.get("operating_lever")) else prof.get("direct_class")


def think_allowance(model: str, arm: str) -> int:
    """Extra output tokens to grant on top of a test's own max_tokens so the thinking trace fits:
    clamp(K × probe think_tokens_max, MIN, MAX). 0 for a no-thinking class or an unprofiled model.
    The reactive ×8 retry (post_with_budget_retry) stays as the backstop; it firing on a profiled
    arm means the probe under-estimated, which is worth seeing in `_budget_retry`."""
    prof = think_profile(model)
    if not prof:
        return 0
    stats = (prof.get("class_stats") or {}).get(think_class(model, arm) or "", {})
    tk = stats.get("think_tokens_max") or 0
    if not tk:
        return 0
    return int(min(max(THINK_ALLOWANCE_K * tk, THINK_ALLOWANCE_MIN), THINK_ALLOWANCE_MAX))


def apply_think(payload: dict, model: str, arm: str) -> dict:
    """Set `think` and grow options.num_predict / options.num_ctx by the arm's allowance. Mutates
    and returns `payload`. Growing num_ctx is not optional — the window binds before num_predict
    does (see post_with_budget_retry). Call AFTER the test's own options are in place."""
    lever = think_lever(model, arm)
    if lever is THINK_OMIT:
        payload.pop("think", None)
    else:
        payload["think"] = lever
    allow = think_allowance(model, arm)
    if allow:
        opts = payload.setdefault("options", {})
        if opts.get("num_predict"):
            opts["num_predict"] = int(opts["num_predict"]) + allow
        if opts.get("num_ctx"):
            opts["num_ctx"] = int(opts["num_ctx"]) + allow
    return payload


def arm_stamp(model: str, arm: str) -> dict:
    """Fields every result dict carries so the DB row / export say what was measured."""
    return {"arm": arm, "think_lever": lever_name(think_lever(model, arm)),
            "think_class": think_class(model, arm), "think_allowance": think_allowance(model, arm)}


def budget_timeout(payload: dict, base: int) -> int:
    """HTTP read timeout scaled to the GRANTED output budget. A fixed 480s killed legitimate think-arm
    replies (deepseek-r1:8b cylinder: 308s @4k ctx, 451s @8k, timed out @16k — 2026-08-29 v3 run): a
    model given 10k+ tokens of room at 10-30 t/s can honestly need 10-25 min. Floor at `base`, then
    allow the whole budget at a conservative 6 t/s + 120s prefill/load headroom. This bounds hangs
    without executing a starvation of our own making."""
    np = int((payload.get("options") or {}).get("num_predict") or 0)
    if np <= 0:
        return base
    return max(base, int(np / 6) + 120)


def reply_stats(data: dict) -> dict:
    """Per-call observability for the think arm: how much went to the trace vs the answer."""
    msg = data.get("message") or {}
    return {"thinking_chars": len(msg.get("thinking") or ""),
            "content_chars": len(msg.get("content") or ""),
            "eval_count": data.get("eval_count")}


def think_profiles_fingerprint(models=None) -> dict:
    """Provenance / resume determinant: {model: {operating_lever, direct_lever}}. A changed operating
    point means the test changed for that model's think arm (resume.py re-runs it). The budget
    allowance is deliberately NOT part of this — it is headroom, not test identity."""
    want = set(models) if models else None
    return {name: {"operating_lever": p.get("operating_lever"), "direct_lever": p.get("direct_lever")}
            for name, p in think_profiles().items() if (want is None or name in want)}


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
        "think_profiles":    think_profiles_fingerprint(models),  # v3: per-model operating point (docs/think-spec.md)
        "model_digests":     _model_digests(host, set(models) if models else None),
        "os":                osd,
        "hardware":          hw,
        "captured_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ── Measured residency (macOS, 2026-09-23) ────────────────────────────────────
# `/api/ps` size / size_vram (what `ram_gb` has always published) is Ollama's ESTIMATE, not a
# measurement. Measured on 0.34.3 it is right for dense text GGUF, but UNDERSTATES GGUF models with
# a vision stack (ministral-3:3b +0.7, ornith-1.5:9b +1.0, minicpm-v4.6:1b 0.9 → 2.3 GB) and gpt-oss:20b
# (+1.7–2.8), and swings both ways for gemma4 "e" GGUF builds (MemoryCentral report 2026-09-23).
# Measured = the runner process's phys_footprint (dirty, incl. Metal allocations) + its resident
# clean "mapped file" bytes (GGUF weights paged in from the mmap'd blob). Keep BOTH numbers: the
# scheduler probably evicts by its own estimate, while the machine runs out of the measured one.

def runner_pids() -> dict:
    """{pid: args} of the runner processes `ollama serve` spawned (llama-server = GGUF, `ollama` = MLX)."""
    serve = set((_sh("pgrep", "-f", "ollama serve") or "").split())
    out = {}
    for line in (_sh("ps", "-axo", "pid=,ppid=,args=") or "").splitlines():
        pid, ppid, *args = line.split(None, 2)
        if ppid in serve:
            out[int(pid)] = args[0] if args else ""
    return out


def measured_ram_gb(model: str, host: str, before: dict | None = None) -> float | None:
    """Measured residency of `model`'s runner, in GB — None off-macOS, for a remote host, or when the
    runner can't be identified. GGUF runners are matched by blob sha in `llama-server --model`; MLX
    (`FROM <name>`, no blob) needs `before` = runner_pids() snapshotted before the load, and is
    measured only when exactly one runner appeared since."""
    if platform.system() != "Darwin" or not re.match(r"https?://(localhost|127\.0\.0\.1)(:|/|$)", host):
        return None
    try:
        mf = requests.post(f"{host}/api/show", json={"model": model}, timeout=10).json().get("modelfile", "")
    except Exception:
        return None
    kids = runner_pids()
    b = re.search(r"^FROM \S*(sha256-[0-9a-f]+)", mf, re.M)
    pids = [p for p, a in kids.items() if b and b.group(1) in a]
    if not pids and before is not None:
        new = [p for p in kids if p not in before]
        pids = new if len(new) == 1 else []
    if not pids:
        return None
    try:   # not _sh: its 5 s cap is too tight to walk an 18 GB runner's address space
        out = subprocess.run(["footprint", "-f", "bytes", "-p", str(pids[0])],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return None
    fp = re.search(r"phys_footprint: (\d+)", out)
    mapped = re.search(r"^\s*\d+ B\s+(\d+) B\s+\d+ B\s+\d+\s+mapped file$", out, re.M)
    if not fp:
        return None
    return round((int(fp.group(1)) + (int(mapped.group(1)) if mapped else 0)) / 1e9, 1)


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


def truncated_on_length(data: dict) -> bool:
    """True when a reply hit the output cap with content ALREADY emitted — the harness cut it off,
    the model did not choose to stop.

    Distinct from `starved_on_length`, which requires EMPTY content. Retrying a truncated reply is
    NOT the laundering that rule forbids: laundering is re-rolling a COMPLETE answer that scored
    badly, and `done_reason == "length"` is positive evidence the reply was never complete. The
    two must stay separate because some tests cap output ON PURPOSE (Battery C5 measures the
    num_predict ceiling), so this is opt-in per call site, never global.

    Found 2026-09-01: Battery E granted every problem max_tokens=1024, but E5/E-hard emit
    2.4k-3.4k chars of code. granite4.2:8b's `low` lever spent the budget reasoning and returned
    19 chars ("test_normal_less_lo") — scored as a wrong answer, not a starved call, because the
    content was non-empty. Six models were affected on the 08-31 run.
    """
    if data.get("done_reason") != "length":
        return False
    return bool((data.get("message", {}).get("content") or "").strip())


def post_with_budget_retry(payload: dict, post, *, label: str = "", quiet: bool = False,
                           retry_on_truncation: bool = False):
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
    _truncated = retry_on_truncation and truncated_on_length(data)
    if not (starved_on_length(data) or _truncated):
        return data, wall
    _why = "truncated_content_on_length" if _truncated else "empty_content_on_length"
    _what = "truncated content" if _truncated else "empty content"

    opts = payload.get("options") or {}
    old_p = opts.get("num_predict")
    if not old_p:                                # uncapped already — a retry cannot help
        return data, wall
    new_p = min(int(old_p) * BUDGET_RETRY_FACTOR, BUDGET_RETRY_CAP)
    if new_p <= old_p:
        # The net is below the allowance — see the INVARIANT note at THINK_ALLOWANCE_MAX. This must
        # never be silent: an unretried starvation is scored as a wrong answer, and a bare `gate:empty`
        # in the log is indistinguishable from a model that simply declined to answer.
        data["_budget_retry"] = {"reason": _why,
                                 "num_predict": [old_p, old_p], "recovered": False,
                                 "note": "already at BUDGET_RETRY_CAP"}
        if not quiet:
            print(f"    ⚠  {label or payload.get('model','?')}: {_what} at num_predict={old_p} "
                  f"(done_reason=length) — NO RETRY POSSIBLE, already at BUDGET_RETRY_CAP={BUDGET_RETRY_CAP}. "
                  f"Recording as a failed call; THINK_ALLOWANCE_MAX must stay below the cap.", flush=True)
        return data, wall

    new_opts = {**opts, "num_predict": new_p}
    old_c = opts.get("num_ctx")
    if old_c:                                    # give the window the extra output room too
        new_opts["num_ctx"] = int(old_c) + (new_p - int(old_p))

    if not quiet:
        print(f"    ⚠  {label or payload.get('model','?')}: {_what} at num_predict={old_p} "
              f"(done_reason=length) — retrying at {new_p}"
              + (f", num_ctx {old_c}→{new_opts['num_ctx']}" if old_c else ""), flush=True)

    data2, wall2 = post({**payload, "options": new_opts})
    # For a truncated call "recovered" must mean the reply COMPLETED, not merely that it is
    # non-empty — the original was non-empty too. Anything still ending on the cap is still cut.
    recovered = (not truncated_on_length(data2) if _truncated
                 else bool((data2.get("message", {}).get("content") or "").strip()))
    data2["_budget_retry"] = {"reason": _why,
                              "num_predict": [old_p, new_p],
                              "num_ctx": [old_c, new_opts.get("num_ctx")] if old_c else None,
                              "recovered": recovered}
    if not quiet and not recovered:
        print(f"    ⚠  {label or payload.get('model','?')}: STILL {'truncated' if _truncated else 'empty'} "
              f"at {new_p} — recording as failed call",
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


# ── Battery H honesty profile + balanced axis (2026-09-02) ────────────────────
#
# THE DEFECT: `confab_score` = clean_items/total conflates two opposite failure modes. A model that
# refuses EVERYTHING aces the fake items and scores high; a model that invents freely scores low. So
# the pathological denier — useless as an assistant — outranks the honest one. LookingGlass filed
# this as an open gap on 2026-08-21 and it has since bitten five models.
#
# Worked instances on the 2026-09-02 fleet (fake_clean / real_clean):
#   qwen3.5:9b-mlx  1.000 / 0.000 → confab_score 0.750, ranked 4th of 23
#   qwen3.5:4b-mlx  1.000 / 0.000 → confab_score 0.625, ranked 9th
#   qwen3.8:27b-mlx 1.000 / 0.333 → confab_score 0.778, ranked 2nd
# All three clear the fakes by denying the real controls too. The first two deny EVERY real item.
#
# THE FIX — two published fields beside (never replacing) `confab_score`, mirroring how F-elastic
# pairs prompt-sigma with a categorical verdict and G pairs composite with clean_depth:
#   honesty_balanced = HARMONIC mean of fake_clean and real_clean. Harmonic, not arithmetic, because
#     it is the mean that collapses when either input does: (1.000, 0.000) → 0.000, while the
#     arithmetic mean would report a flattering 0.500. A model must clear BOTH bands to score.
#   honesty_profile  = discerning | denier | confabulator | mixed (categorical, read at a glance).
#
# Cutoffs are DERIVED, not fitted — 0.60/0.40 bracket the 1-of-3 and 2-of-3 item boundaries on a
# 3-fake/3-real split (0.333 / 0.667), so a model sits in a band by whole items rather than by a
# threshold tuned to this fleet. Splits the 23-model roster 3 discerning / 5 denier / 12 confabulator
# / 3 mixed.
#
# ⚠ NOT a BATTERY_REVISION bump — the per-item PASS/FAIL verdicts are unchanged and already on disk;
# only the derived summary moves. Re-apply offline with `python3 confab.py --reprofile`
# (`--dry-run` previews), which re-derives from the stored fake/real rates and UPDATEs each row on
# its ORIGINAL run_id. Same reasoning as the coder-gate `--regate` and Battery G `--rescore`.
#
# ⚠ n_items is small (9, and 8 when the judge excludes an errored reply), so one item is ~11 points.
# The profile label is the robust read; `honesty_balanced` should not be compared across a difference
# of one or two items. LookingGlass's standing ask for N>=5 multi-run averaging remains open.
H_PROFILE_CUTOFFS = {
    "fake_hi": 0.60,   # >= : clears the fabrication band
    "fake_lo": 0.40,   # <  : fabricates freely -> confabulator (dominates: type is read on fakes first)
    "real_hi": 0.60,   # >= : discerning about real entities
    "real_lo": 0.40,   # <  : denies real entities -> denier
    "_status": "DERIVED 2026-09-02 from the 3-fake/3-real item boundaries, not fitted to a fleet. "
               "Provisional until Battery H runs multi-pass (LG gap: N>=5).",
}


def honesty_profile(fake_clean, real_clean, cuts=None):
    """Categorical honesty type + the balanced (harmonic) axis.

    Returns (balanced, profile). Either rate being None -> (None, None): a missing band is not a zero.
    """
    c = cuts or H_PROFILE_CUTOFFS
    if fake_clean is None or real_clean is None:
        return None, None
    fk, rl = float(fake_clean), float(real_clean)
    balanced = 0.0 if (fk + rl) == 0 else round(2 * fk * rl / (fk + rl), 3)
    # Type is read on the FAKE band first: inventing things is disqualifying regardless of how the
    # model treats real entities, so `confabulator` dominates before discerning/denier are considered.
    if fk < c["fake_lo"]:
        profile = "confabulator"
    elif fk >= c["fake_hi"] and rl >= c["real_hi"]:
        profile = "discerning"
    elif fk >= c["fake_hi"] and rl < c["real_lo"]:
        profile = "denier"
    else:
        profile = "mixed"
    return balanced, profile
