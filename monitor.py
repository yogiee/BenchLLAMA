#!/usr/bin/env python3
"""
BenchLLAMA Monitor
Run in a separate Terminal window while runner.py or aptitude.py runs.

Usage:
  python3 monitor.py
  python3 monitor.py --fast    # matches --fast benchmark run
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

REPO        = Path(__file__).parent
RESULTS_DIR = REPO / "results"
STATUS_FILE = RESULTS_DIR / "status.json"

FAST_MODE  = "--fast" in sys.argv
COOLDOWN_S = 0 if FAST_MODE else 300
REFRESH    = 2

def _load_registry():
    path = REPO / "models.json"
    try:
        return [(m["name"], m["role"]) for m in json.loads(path.read_text())]
    except FileNotFoundError:
        return []

MODELS = _load_registry()

BATTERIES = [
    "A — Router",
    "B — Worker Chat",
    "C — Worker Research",
    "D — Worker Tool-heavy",
]

G   = "\033[32m"
R   = "\033[31m"
Y   = "\033[33m"
DIM = "\033[2m"
B   = "\033[1m"
RST = "\033[0m"
CLR = "\033[2J\033[H"


def _rjson(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None

def _elapsed(ts):
    s = max(0, int(time.time() - ts))
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"

def _results_file():
    candidates = sorted(
        [f for f in RESULTS_DIR.glob("benchmark_20*.json") if "status" not in f.name],
        reverse=True,
    )
    return candidates[0] if candidates else None


def _ladder_file():
    candidates = sorted(
        [f for f in RESULTS_DIR.glob("ctx_ladder_20*.json") if "status" not in f.name],
        reverse=True,
    )
    return candidates[0] if candidates else None


def render():
    now    = datetime.now()
    status = _rjson(STATUS_FILE) or {}
    rf     = _results_file()
    done   = {r["model"]: r for r in (_rjson(rf) or [])}

    segment = status.get("segment", "")
    cur     = status.get("model")
    phase   = status.get("phase", "")
    ts      = status.get("ts", 0)

    # ctx ladder view — show instead of standard/aptitude sections
    if segment == "ctx_ladder":
        lf          = _ladder_file()
        ladder_done = {r["model"]: r for r in (_rjson(lf) or [])}
        ctx_cur     = status.get("ctx")

        out = []
        out.append(f"{B}╔══════════════════════════════════════════════╗{RST}")
        out.append(f"{B}║  BenchLLAMA Monitor  {now.strftime('%Y-%m-%d')}              ║{RST}")
        out.append(f"{B}╚══════════════════════════════════════════════╝{RST}")
        out.append("")
        label = f"{G}[COMPLETE ✓]{RST}" if phase == "done" else f"{Y}[running]{RST}"
        out.append(f"  {B}CTX LADDER{RST}                       {label}")
        out.append(f"  {'─' * 44}")

        for model, role in MODELS:
            if model in ladder_done:
                r      = ladder_done[model]
                levels = sorted(r.get("levels", {}).keys())
                passed = sum(1 for e in r["levels"].values() if e.get("cylinder", {}).get("correct"))
                out.append(f"  {G}✓{RST}  {model:<28}{DIM}{role:<8}{RST}  {DIM}{passed}/{len(levels)} pass{RST}")
            elif model == cur:
                ctx_str = f"ctx={ctx_cur}" if ctx_cur else phase
                el      = f"  {R}{_elapsed(ts)}{RST}" if ts else ""
                out.append(f"  {R}●{RST}  {model:<28}{DIM}{role:<8}{RST}  {DIM}{ctx_str}{RST}{el}")
            else:
                out.append(f"  {DIM}○  {model:<28}{role:<8}  —{RST}")

        out.append("")
        fast_note = f"  {Y}[FAST MODE]{RST}   " if FAST_MODE else "  "
        out.append(f"{fast_note}{DIM}Updated {now.strftime('%H:%M:%S')}   Ctrl+C to exit{RST}")
        return "\n".join(out)

    n_done   = len(done)
    all_done = (phase == "done") or (n_done == len(MODELS))

    out = []
    out.append(f"{B}╔══════════════════════════════════════════════╗{RST}")
    out.append(f"{B}║  BenchLLAMA Monitor  {now.strftime('%Y-%m-%d')}              ║{RST}")
    out.append(f"{B}╚══════════════════════════════════════════════╝{RST}")
    out.append("")

    hdr = f"{G}[COMPLETE ✓]{RST}" if all_done else f"{Y}[{n_done} / {len(MODELS)}]{RST}"
    out.append(f"  {B}STANDARD BENCHMARK{RST}              {hdr}")
    out.append(f"  {'─' * 44}")

    for model, role in MODELS:
        r      = done.get(model)
        is_cur = model == cur

        if r is not None:
            t = f"  {G}{r['avg_tps']} tok/s{RST}" if r.get("avg_tps") else ""
            out.append(f"  {G}✓{RST}  {model:<28}{DIM}{role:<8}{RST}{t}")
        elif is_cur and phase == "cooldown":
            remaining = max(0, COOLDOWN_S - int(time.time() - ts))
            out.append(
                f"  {Y}◌{RST}  {model:<28}{DIM}{role:<8}{RST}"
                f"  {Y}cooldown {remaining}s remaining{RST}"
            )
        elif is_cur:
            el = f"  {R}{_elapsed(ts)}{RST}" if ts else ""
            out.append(
                f"  {R}●{RST}  {model:<28}{DIM}{role:<8}{RST}"
                f"  {DIM}{phase}{RST}{el}"
            )
        else:
            out.append(f"  {DIM}○  {model:<28}{role:<8}  —{RST}")

    out.append("")

    if status.get("segment") == "aptitude":
        apt_done = status.get("aptitude_done", [])
        apt_cur  = status.get("aptitude_current")
        out.append(f"  {B}APTITUDE SUITE{RST}                  {Y}[running]{RST}")
    else:
        apt_done = []
        apt_cur  = None
        label = f"{G}[COMPLETE ✓]{RST}" if (all_done and status.get("aptitude_complete")) else f"{DIM}[pending]{RST}"
        out.append(f"  {B}APTITUDE SUITE{RST}                  {label}")

    out.append(f"  {'─' * 44}")
    for battery in BATTERIES:
        name = battery.split("—")[0].strip()
        if name in apt_done:
            out.append(f"  {G}✓{RST}  Battery {battery}")
        elif battery == apt_cur:
            out.append(f"  {R}●{RST}  Battery {battery}")
        else:
            out.append(f"  {DIM}○  Battery {battery}{RST}")

    out.append("")
    fast_note = f"  {Y}[FAST MODE]{RST}   " if FAST_MODE else "  "
    out.append(f"{fast_note}{DIM}Updated {now.strftime('%H:%M:%S')}   Ctrl+C to exit{RST}")

    return "\n".join(out)


if __name__ == "__main__":
    try:
        while True:
            print(CLR + render(), end="", flush=True)
            time.sleep(REFRESH)
    except KeyboardInterrupt:
        print()
