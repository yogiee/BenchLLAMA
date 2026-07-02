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
    "prompt_worker":     "prompts/worker_default.md",
    "prompt_router":     "prompts/router_default.md",
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
