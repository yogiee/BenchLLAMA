#!/usr/bin/env python3
"""
BenchLLAMA — headless orchestration core.

UI-agnostic. Spawns the worker subprocesses for a pipeline, aggregates phase / model /
tok-s state + a capped log buffer, writes a run-log, and emits events via callbacks.
Consumed by the web server (webserver.py) and the `--console` streamer here; no UI imports.

Run directly for the plain-text console (the headless / quick-glance / SSH path):
    python3 orchestrator.py <command> [flags]
"""

import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

REPO        = Path(__file__).parent
MODELS_FILE = REPO / "models.json"
PAUSE_SECS  = 10        # between pipeline phases
MAX_LOG     = 4000      # capped in-memory log buffer (for late-joining web clients)

COMMANDS = {"standard", "ladder", "aptitude", "batteries", "all", "update", "vision", "embedding"}

# ── State ─────────────────────────────────────────────────────────────────────

@dataclass
class PhaseState:
    label:       str
    role_filter: Optional[str] = None
    status:      str           = "pending"   # pending | running | done | error

@dataclass
class ModelState:
    name:   str
    role:   Optional[str]   = None
    tps:    Optional[float] = None
    status: str             = "pending"   # pending | running | done | error | skip
    active: bool            = True
    caps:   list            = field(default_factory=list)
    extended_roles: list    = field(default_factory=list)   # earned (e.g. coder)
    cloud:  bool            = False                          # quality-only cloud model

class BenchState:
    def __init__(self):
        self.phases:          list[PhaseState] = []
        self.models:          list[ModelState] = []
        self.current_phase:   int              = 0
        self.last_tps:        Optional[float]  = None
        self.start_time:      float            = time.time()
        self.finished:        bool             = False
        self.pause_remaining: int              = 0
        self.aborted:         bool             = False
        self.log:             list[str]        = []
        self.log_total:       int              = 0   # monotonic count (trim-safe streaming)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_all_models() -> list[ModelState]:
    try:
        return [ModelState(name=m["name"], role=m.get("role") or None,
                           caps=m.get("capabilities", []),
                           extended_roles=m.get("extended_roles", []),
                           cloud=bool(m.get("cloud", False)))
                for m in json.loads(MODELS_FILE.read_text())]
    except Exception:
        return []

def _arg(args: list[str], flag: str) -> Optional[str]:
    try:
        i = args.index(flag)
        return args[i + 1] if i + 1 < len(args) else None
    except ValueError:
        return None

def _cmd(*parts) -> list[str]:
    return [sys.executable] + [str(p) for p in parts]

# ── Phase builder ─────────────────────────────────────────────────────────────

def build_phases(cmd: str, extra: list[str]) -> list[tuple]:
    """Returns [(label, argv, role_filter), ...]"""
    apt = REPO / "aptitude.py"
    x   = extra
    role_in_extra = _arg(extra, "--role")

    if cmd == "standard":
        return [("Standard Suite", _cmd(REPO/"runner.py", *x), role_in_extra)]
    if cmd == "ladder":
        return [("ctx Ladder", _cmd(REPO/"ctx_ladder.py", *x), role_in_extra)]
    if cmd == "aptitude":
        # Batteries E (coding), F (consistency) and F-elastic (prompt-σ) default to MULTIPASS
        # averaging — all grade correctness and are noisy at the boundary on a single run
        # (F-elastic calibration 2026-06-22 showed single runs flip 5 borderline verdicts).
        bat = (_arg(extra, "--battery") or "B").upper()
        AVG = {"E": [], "F": ["--battery", "F"], "F-ELASTIC": ["--battery", "F-elastic"]}
        if bat in AVG:
            drop = ("--battery", "E", "e", "F", "f", "F-elastic", "f-elastic", "F-ELASTIC")
            clean = [a for a in x if a not in drop]
            return [(f"Battery {bat} (3-run avg)", _cmd(REPO/"average_e_runs.py", *AVG[bat], *clean), "cap:completion")]
        return [("Aptitude", _cmd(apt, *x), role_in_extra)]
    if cmd == "update":
        return [("Update Registry", _cmd(REPO/"update_registry.py", *x), None)]
    if cmd == "vision":
        return [("Vision (Battery V)", _cmd(REPO/"vision.py", *x), "cap:vision")]
    if cmd == "embedding":
        return [("Embedding (Battery EMB)", _cmd(REPO/"embedding.py", *x), "cap:embedding")]
    if cmd == "batteries":
        return [
            ("Battery A", _cmd(apt, "--battery", "A", "--role", "router", *x),                    "router"),
            ("Battery B", _cmd(apt, "--battery", "B", "--role", "worker", *x),                    "worker"),
            ("Battery C", _cmd(apt, "--battery", "C", "--role", "worker", "--capable-only", *x),  "worker"),
            ("Battery D", _cmd(apt, "--battery", "D", "--role", "worker", "--capable-only", *x),  "worker"),
            ("Battery E (3-run avg)", _cmd(REPO/"average_e_runs.py", *x),                         "cap:completion"),
            ("Battery F (3-run avg)", _cmd(REPO/"average_e_runs.py", "--battery", "F", *x),       "cap:completion"),
        ]
    if cmd == "all":
        return [
            ("Standard Suite", _cmd(REPO/"runner.py", *x),                                              None),
            ("ctx Ladder",     _cmd(REPO/"ctx_ladder.py", *x),                                          None),
            ("Battery A",      _cmd(apt, "--battery", "A", "--role", "router", *x),              "router"),
            ("Battery B",      _cmd(apt, "--battery", "B", "--role", "worker", *x),              "worker"),
            ("Battery C",      _cmd(apt, "--battery", "C", "--role", "worker", "--capable-only", *x), "worker"),
            ("Battery D",      _cmd(apt, "--battery", "D", "--role", "worker", "--capable-only", *x), "worker"),
            ("Battery E (3-run avg)", _cmd(REPO/"average_e_runs.py", *x),                        "cap:completion"),
            ("Battery F (3-run avg)", _cmd(REPO/"average_e_runs.py", "--battery", "F", *x),      "cap:completion"),
            ("Vision (Battery V)",      _cmd(REPO/"vision.py", *x),                              "cap:vision"),
            ("Embedding (Battery EMB)", _cmd(REPO/"embedding.py", *x),                           "cap:embedding"),
        ]
    return []

# ── Orchestrator ────────────────────────────────────────────────────────────────

class Orchestrator:
    """Runs a pipeline (list of phases) as sequential subprocesses, maintaining live
    state. `on_log(line)` fires per log line; `on_event()` fires after state changes
    (so a view can refresh/push). Both optional."""

    def __init__(self, phases_spec: list[tuple],
                 on_log: Optional[Callable[[str], None]] = None,
                 on_event: Optional[Callable[[], None]] = None):
        self._phases_spec = phases_spec
        self.state = BenchState()
        self.state.phases = [PhaseState(ph[0], role_filter=ph[2]) for ph in phases_spec]
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._stopped = False
        self._run_log_fh = None
        self._on_log = on_log or (lambda s: None)
        self._on_event = on_event or (lambda: None)

    def snapshot(self) -> dict:
        """JSON-able view of current state (for the web client)."""
        s = self.state
        cur = s.phases[s.current_phase].label if s.current_phase < len(s.phases) else ""
        return {
            "phases": [{"label": p.label, "status": p.status} for p in s.phases],
            "current_phase": s.current_phase, "phase_label": cur,
            "models": [{"name": m.name, "role": m.role, "tps": m.tps,
                        "status": m.status, "active": m.active, "cloud": m.cloud,
                        "caps": m.caps, "extended_roles": m.extended_roles} for m in s.models],
            "last_tps": s.last_tps, "elapsed": int(time.time() - s.start_time),
            "finished": s.finished, "pause_remaining": s.pause_remaining,
            "aborted": s.aborted,
        }

    def _emit(self, line: str) -> None:
        self.state.log.append(line)
        self.state.log_total += 1
        if len(self.state.log) > MAX_LOG:
            del self.state.log[0:len(self.state.log) - MAX_LOG]
        self._on_log(line)
        if self._run_log_fh:
            self._run_log_fh.write(line + "\n")
            self._run_log_fh.flush()

    def stop(self) -> None:
        """Abort the ENTIRE run — terminate the current subprocess AND break the phase
        loop so the pipeline does not advance to the next phase (stop ≠ skip)."""
        self._stopped = True
        self.state.aborted = True
        self._emit("\n■  Stop requested — aborting run (remaining phases will not start)")
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass

    async def run(self) -> None:
        run_log = REPO / "results" / f"run_{time.strftime('%Y-%m-%d_%H-%M')}.log"
        run_log.parent.mkdir(exist_ok=True)
        self._run_log_fh = run_log.open("w")
        self._emit(f"Run log → {run_log}")
        self.state.models = load_all_models()
        self._on_event()

        for i, (label, argv, role_filter) in enumerate(self._phases_spec):
            if self._stopped:
                break
            if i > 0:
                for remaining in range(PAUSE_SECS, 0, -1):
                    if self._stopped:
                        break
                    self.state.pause_remaining = remaining
                    self._on_event()
                    await asyncio.sleep(1)
                self.state.pause_remaining = 0
            if self._stopped:
                break

            self.state.current_phase = i
            self.state.phases[i].status = "running"
            self.state.last_tps = None
            self._set_active_for_phase(role_filter)
            self._emit(f"\n{'━' * 58}\n  ▶  {label}\n{'━' * 58}")
            self._on_event()

            ok = await self._run_subprocess(argv)
            if self._stopped:
                # aborted mid-phase: mark this phase + its running model as stopped, do not advance
                self.state.phases[i].status = "error"
                for ms in self.state.models:
                    if ms.active and ms.status == "running":
                        ms.status = "error"
                self._on_event()
                break
            self.state.phases[i].status = "done" if ok else "error"
            for ms in self.state.models:
                if ms.active and ms.status == "running":
                    ms.status = "done" if ok else "error"
            self._sync_roles()
            self._on_event()

        self.state.finished = True
        self._on_event()
        if self._run_log_fh:
            self._run_log_fh.close()
            self._run_log_fh = None

    def _set_active_for_phase(self, filt: Optional[str]) -> None:
        for ms in self.state.models:
            if filt is None:
                ms.active = True
            elif filt.startswith("cap:"):
                ms.active = filt[4:] in (ms.caps or [])
            else:
                ms.active = (ms.role == filt)
            if ms.active:
                ms.status = "pending"

    def _sync_roles(self) -> None:
        """Re-read models.json — runner promotes to router; Battery E adds the coder
        extended role. Both should reflect live on the model cards."""
        try:
            reg = {m["name"]: m for m in json.loads(MODELS_FILE.read_text())}
            for ms in self.state.models:
                if ms.name in reg:
                    ms.role = reg[ms.name].get("role")
                    ms.extended_roles = reg[ms.name].get("extended_roles", [])
        except Exception:
            pass

    async def _run_subprocess(self, argv: list[str]) -> bool:
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT, cwd=str(REPO))
            self._proc = proc
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").rstrip()
                self._emit(line)
                self._parse_line(line)
                self._on_event()
            await proc.wait()
            self._proc = None
            return proc.returncode == 0
        except Exception as exc:
            self._emit(f"  [subprocess error] {exc}")
            self._proc = None
            return False

    # ── log parsing → model state ──────────────────────────────────────────────
    def _parse_line(self, line: str) -> None:
        m = re.search(r"MODEL:\s+(\S+)", line)
        if m:
            name = m.group(1)
            for ms in self.state.models:
                if ms.status == "running":
                    ms.status = "done"
                if ms.name == name and ms.active and ms.status == "pending":
                    ms.status = "running"
            self.state.last_tps = None
            return
        m = re.search(r"✓\s+avg_tps=([\d.]+)", line)
        if m:
            try:
                self._mark_running("done", tps=float(m.group(1)))
            except ValueError:
                self._mark_running("done")
            return
        if "done — JSON updated" in line:
            self._mark_running("done"); return
        if re.search(r"✗.+FAILED", line):
            self._mark_running("error"); return
        m = re.search(r"↷\s+(\S+)", line)
        if m:
            for ms in self.state.models:
                if ms.name == m.group(1):
                    ms.status = "skip"
            return
        m = re.search(r"★ Role gate passed — (\S+) promoted to (\w+)", line)
        if m:
            for ms in self.state.models:
                if ms.name == m.group(1):
                    ms.role = m.group(2)
            return
        m = re.search(r"skipped: \[(.+?)\]", line)
        if m:
            names = re.findall(r"'([^']+)'", m.group(1))
            for ms in self.state.models:
                if ms.name in names:
                    ms.status = "skip"
            return
        m = re.search(r"\btps=([\d.]+)", line)
        if m:
            try:
                self.state.last_tps = float(m.group(1))
            except ValueError:
                pass

    def _mark_running(self, new_status: str, tps: Optional[float] = None) -> None:
        for ms in self.state.models:
            if ms.status == "running":
                ms.status = new_status
                if tps is not None:
                    ms.tps = tps

# ── Console mode (the headless / quick-glance path) ─────────────────────────────

def run_console(phases_spec: list[tuple]) -> None:
    orch = Orchestrator(phases_spec, on_log=lambda l: print(l, flush=True))
    try:
        asyncio.run(orch.run())
    except KeyboardInterrupt:
        orch.stop()
        print("\n[interrupted]", flush=True)


if __name__ == "__main__":
    raw = sys.argv[1:]
    cmd = next((a for a in raw if a in COMMANDS), None)
    if not cmd or "--help" in raw or "-h" in raw:
        print(f"Usage: python3 orchestrator.py <{' | '.join(sorted(COMMANDS))}> [flags]")
        sys.exit(0 if not cmd else 1)
    phases = build_phases(cmd, [a for a in raw if a != cmd])
    if not phases:
        sys.exit(f"Unknown command: {cmd}")
    run_console(phases)
