"""
BenchLLAMA — shared utilities
Imported by runner.py, ctx_ladder.py, and aptitude.py.
"""

import re
import subprocess
import time
import requests

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
