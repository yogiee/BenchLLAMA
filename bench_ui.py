#!/usr/bin/env python3
"""
BenchLLAMA — Terminal UI
Split-screen: left dashboard (pipeline phases, model status dots, tok/s)
              right live log (full subprocess stdout)

Replaces bench.sh orchestration + monitor.py.

Install:  pip install textual
Usage:    python3 bench_ui.py <command> [flags]
          (same commands and flags as bench.sh)
"""

import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from textual.app import App, ComposeResult
    from textual.widgets import Header, Footer, RichLog, Static
    from rich.text import Text
except ImportError:
    sys.exit("textual not installed — run: pip install textual")

REPO        = Path(__file__).parent
MODELS_FILE = REPO / "models.json"
PAUSE_SECS  = 10   # between pipeline phases

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
    active: bool            = True        # False = greyed out, not part of current phase
    caps:   list            = field(default_factory=list)  # capabilities (for cap-routed phases)

class BenchState:
    def __init__(self):
        self.phases:          list[PhaseState]  = []
        self.models:          list[ModelState]  = []
        self.current_phase:   int               = 0
        self.last_tps:        Optional[float]   = None   # live tps while a model is running
        self.start_time:      float             = time.time()
        self.finished:        bool              = False
        self.pause_remaining: int               = 0

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_all_models() -> list[ModelState]:
    """Load full model list from registry, preserving any roles already set."""
    try:
        return [
            ModelState(name=m["name"], role=m.get("role") or None,
                       caps=m.get("capabilities", []))
            for m in json.loads(MODELS_FILE.read_text())
        ]
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
        return [("Aptitude", _cmd(apt, *x), role_in_extra)]
    if cmd == "update":
        return [("Update Registry", _cmd(REPO/"update_registry.py", *x), None)]
    if cmd == "vision":
        # Capability-routed: highlight only models with the vision cap (utility + workers).
        return [("Vision (Battery V)", _cmd(REPO/"vision.py", *x), "cap:vision")]
    if cmd == "embedding":
        return [("Embedding (Battery EMB)", _cmd(REPO/"embedding.py", *x), "cap:embedding")]
    if cmd == "batteries":
        return [
            ("Battery A", _cmd(apt, "--battery", "A", "--role", "router", *x),                    "router"),
            ("Battery B", _cmd(apt, "--battery", "B", "--role", "worker", *x),                    "worker"),
            ("Battery C", _cmd(apt, "--battery", "C", "--role", "worker", "--capable-only", *x),  "worker"),
            ("Battery D", _cmd(apt, "--battery", "D", "--role", "worker", "--capable-only", *x),  "worker"),
        ]
    if cmd == "all":
        return [
            ("Standard Suite", _cmd(REPO/"runner.py", *x),                                                    None),
            ("ctx Ladder",     _cmd(REPO/"ctx_ladder.py", *x),                                                None),
            ("Battery A",      _cmd(apt, "--battery", "A", "--role", "router", *x),              "router"),
            ("Battery B",      _cmd(apt, "--battery", "B", "--role", "worker", *x),              "worker"),
            ("Battery C",      _cmd(apt, "--battery", "C", "--role", "worker", "--capable-only", *x), "worker"),
            ("Battery D",      _cmd(apt, "--battery", "D", "--role", "worker", "--capable-only", *x), "worker"),
        ]
    return []

# ── Dashboard widget ──────────────────────────────────────────────────────────

ICONS = {
    "pending": ("○", "dim"),
    "running": ("●", "bold green"),
    "done":    ("✓", "green"),
    "error":   ("✗", "red"),
    "skip":    ("↷", "dim"),
}

class Dashboard(Static):
    """Fixed-width left panel showing phase + model status."""

    def on_mount(self) -> None:
        self.set_interval(0.5, self.refresh)

    def render(self) -> Text:
        state: BenchState = self.app._state  # type: ignore
        t = Text()

        elapsed = int(time.time() - state.start_time)
        h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60

        total_models  = len(state.models)
        active_models = sum(1 for ms in state.models if ms.active)

        # ── Pipeline phases (only when >1) ────────────────────────────────────
        if len(state.phases) > 1:
            t.append("PIPELINE\n", style="bold")
            for i, ph in enumerate(state.phases):
                icon, style = ICONS[ph.status]
                arrow = "▶ " if (i == state.current_phase and ph.status == "running") else "  "
                t.append(f"{arrow}")
                t.append(f"{icon} ", style=style)
                t.append(f"{ph.label}\n")
            t.append("\n")

        # ── Current phase label with model count ──────────────────────────────
        if state.current_phase < len(state.phases):
            ph = state.phases[state.current_phase]
            prefix = "" if len(state.phases) == 1 else "▶ "
            count  = f" ({active_models}/{total_models})" if total_models else ""
            t.append(f"{prefix}{ph.label}{count}\n", style="bold cyan")

        # ── Pause countdown ───────────────────────────────────────────────────
        if state.pause_remaining > 0:
            t.append(f"  next phase in {state.pause_remaining}s…\n", style="yellow")

        # ── Model list ────────────────────────────────────────────────────────
        if state.models:
            t.append("\n")
            for ms in state.models:
                icon, icon_style = ICONS[ms.status]
                is_running = ms.status == "running"
                arrow = "▶ " if is_running else "  "

                role_str = f" {ms.role.upper()}" if ms.role else ""
                # Show live tps for running model; settled tps otherwise
                if is_running and state.last_tps:
                    tps_str = f" ({state.last_tps:.0f}t/s)"
                elif ms.tps is not None:
                    tps_str = f" ({ms.tps:.0f}t/s)"
                else:
                    tps_str = ""

                suffix = f" —{role_str}{tps_str}" if (role_str or tps_str) else ""

                if ms.active:
                    name_style = "bold" if is_running else "default"
                    t.append(f"  {arrow}")
                    t.append(f"{icon} ", style=icon_style)
                    t.append(f"{ms.name}", style=name_style)
                    t.append(f"{suffix}\n", style="dim")
                else:
                    t.append(f"  {icon} ", style="dim")
                    t.append(f"{ms.name}{suffix}\n", style="dim")

            done  = sum(1 for ms in state.models if ms.status in ("done", "skip", "error"))
            total = len(state.models)
            t.append(f"\n  {done}/{total} models\n", style="dim")

        # ── Footer ────────────────────────────────────────────────────────────
        t.append("\n")
        if state.finished:
            t.append("  ✓ DONE\n", style="bold green")
        t.append(f"  {h:02d}:{m:02d}:{s:02d}", style="dim")

        return t

# ── App ───────────────────────────────────────────────────────────────────────

class BenchUI(App):
    TITLE = "BenchLLAMA"
    CSS = """
    Screen { layout: horizontal; }
    Dashboard {
        width: 50;
        padding: 1 1;
        border-right: solid $panel-lighten-2;
        background: $surface-darken-1;
    }
    RichLog {
        width: 1fr;
        padding: 0 1;
    }
    """
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, phases_spec: list[tuple], **kwargs):
        super().__init__(**kwargs)
        self._phases_spec   = phases_spec
        self._state         = BenchState()
        self._current_proc: Optional[asyncio.subprocess.Process] = None
        self._run_log_fh    = None
        self._state.phases  = [PhaseState(ph[0], role_filter=ph[2]) for ph in phases_spec]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Dashboard()
        yield RichLog(highlight=True, markup=False, wrap=True)
        yield Footer()

    async def on_mount(self) -> None:
        self._log = self.query_one(RichLog)
        asyncio.create_task(self._orchestrate())

    def action_quit(self) -> None:
        if self._current_proc:
            try:
                self._current_proc.terminate()
            except Exception:
                pass
        self.exit()

    # ── Orchestration ─────────────────────────────────────────────────────────

    async def _orchestrate(self) -> None:
        log_name = f"run_{time.strftime('%Y-%m-%d_%H-%M')}.log"
        run_log  = REPO / "results" / log_name
        self._run_log_fh = run_log.open("w")
        self._log.write(f"Run log → {run_log}")

        # Model list is persistent across all phases
        self._state.models = load_all_models()

        for i, (label, argv, role_filter) in enumerate(self._phases_spec):
            if i > 0:
                for remaining in range(PAUSE_SECS, 0, -1):
                    self._state.pause_remaining = remaining
                    await asyncio.sleep(1)
                self._state.pause_remaining = 0

            self._state.current_phase = i
            self._state.phases[i].status = "running"
            self._state.last_tps = None

            # Mark which models are active for this phase; reset their status
            self._set_active_for_phase(role_filter)

            self._log.write(f"\n{'━' * 58}")
            self._log.write(f"  ▶  {label}")
            self._log.write("━" * 58)
            if self._run_log_fh:
                self._run_log_fh.write(f"\n{'━' * 58}\n  ▶  {label}\n{'━' * 58}\n")
                self._run_log_fh.flush()

            ok = await self._run_subprocess(argv)
            self._state.phases[i].status = "done" if ok else "error"

            # Settle any active model still showing as running
            for ms in self._state.models:
                if ms.active and ms.status == "running":
                    ms.status = "done" if ok else "error"

            # Re-sync roles — runner.py may have promoted models to router
            self._sync_roles()

        self._state.finished = True
        if self._run_log_fh:
            self._run_log_fh.close()
            self._run_log_fh = None

    def _set_active_for_phase(self, filt: Optional[str]) -> None:
        """Mark models active/inactive for the current phase and reset active models to pending.

        filt:  None         → all models active (standard / ladder)
               "cap:<name>"  → active if <name> in the model's capabilities (vision / embedding)
               "<role>"      → active if model role matches (router / worker batteries)
        """
        for ms in self._state.models:
            if filt is None:
                ms.active = True
            elif filt.startswith("cap:"):
                ms.active = filt[4:] in (ms.caps or [])
            else:
                ms.active = (ms.role == filt)
            if ms.active:
                ms.status = "pending"

    def _sync_roles(self) -> None:
        """Re-read models.json and update roles on existing ModelState objects in place."""
        try:
            role_map = {m["name"]: m.get("role") for m in json.loads(MODELS_FILE.read_text())}
            for ms in self._state.models:
                if ms.name in role_map:
                    ms.role = role_map[ms.name]
        except Exception:
            pass

    async def _run_subprocess(self, argv: list[str]) -> bool:
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(REPO),
            )
            self._current_proc = proc
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").rstrip()
                self._log.write(line)
                self._parse_line(line)
                if self._run_log_fh:
                    self._run_log_fh.write(line + "\n")
                    self._run_log_fh.flush()
            await proc.wait()
            self._current_proc = None
            return proc.returncode == 0
        except Exception as exc:
            msg = f"  [subprocess error] {exc}"
            self._log.write(msg)
            if self._run_log_fh:
                self._run_log_fh.write(msg + "\n")
                self._run_log_fh.flush()
            self._current_proc = None
            return False

    # ── Log parsing ───────────────────────────────────────────────────────────

    def _parse_line(self, line: str) -> None:
        # New model starting — "MODEL: name  (...)"
        m = re.search(r"MODEL:\s+(\S+)", line)
        if m:
            name = m.group(1)
            for ms in self._state.models:
                if ms.status == "running":
                    ms.status = "done"
                if ms.name == name and ms.active and ms.status == "pending":
                    ms.status = "running"
            self._state.last_tps = None
            return

        # Standard suite model done — "  ✓ avg_tps=XX.X ..."
        m = re.search(r"✓\s+avg_tps=([\d.]+)", line)
        if m:
            try:
                self._mark_running("done", tps=float(m.group(1)))
            except ValueError:
                self._mark_running("done")
            return

        # Aptitude model done — "  ✓ name done — JSON updated"
        if "done — JSON updated" in line:
            self._mark_running("done")
            return

        # Model error — "  ✗ name FAILED: ..."
        if re.search(r"✗.+FAILED", line):
            self._mark_running("error")
            return

        # Resume skip — "  ↷ name — already done"
        m = re.search(r"↷\s+(\S+)", line)
        if m:
            name = m.group(1)
            for ms in self._state.models:
                if ms.name == name:
                    ms.status = "skip"
            return

        # Role promotion — "★ Role gate passed — name promoted to router"
        m = re.search(r"★ Role gate passed — (\S+) promoted to (\w+)", line)
        if m:
            name, role = m.group(1), m.group(2)
            for ms in self._state.models:
                if ms.name == name:
                    ms.role = role
            return

        # capable-only skips — "N tool-capable models(skipped: ['m1', 'm2'])"
        m = re.search(r"skipped: \[(.+?)\]", line)
        if m:
            raw_names = re.findall(r"'([^']+)'", m.group(1))
            for ms in self._state.models:
                if ms.name in raw_names:
                    ms.status = "skip"
            return

        # Live tok/s — "tps=67.0" (intermediate, shown on currently running model)
        m = re.search(r"\btps=([\d.]+)", line)
        if m:
            try:
                self._state.last_tps = float(m.group(1))
            except ValueError:
                pass

    def _mark_running(self, new_status: str, tps: Optional[float] = None) -> None:
        for ms in self._state.models:
            if ms.status == "running":
                ms.status = new_status
                if tps is not None:
                    ms.tps = tps

# ── CLI ───────────────────────────────────────────────────────────────────────

COMMANDS = {"standard", "ladder", "aptitude", "batteries", "all", "update", "vision", "embedding"}

def _usage() -> None:
    print("""
Usage:  bench_ui.py <command> [flags]

Commands:
  standard    Standard suite (13 tests, 5 dimensions)
  ladder      num_ctx characterisation
  aptitude    Single aptitude battery (default: B)
  batteries   All aptitude batteries A → B → C → D
  all         Full pipeline: standard → ladder → A → B → C → D
  vision      Battery V — vision models (capability-routed, head-to-head)
  embedding   Battery EMB — embedding models (capability-routed)
  update      Sync models.json from Ollama

Flags (passed through to the Python scripts):
  --fast                  Skip cool-down (informal results)
  --force                 Ignore resume window, re-run and overwrite
  --battery A|B|C|D       Aptitude battery (default B)
  --role router|worker    Filter models by role
  --system-prompt <path>  Custom worker system prompt
  --ollama <url>          Remote Ollama (default: http://localhost:11434)
  --models m1 [m2 ...]    Specific models
""")

if __name__ == "__main__":
    raw = sys.argv[1:]
    cmd = next((a for a in raw if a in COMMANDS), None)

    if "--help" in raw or "-h" in raw or not cmd:
        _usage()
        sys.exit(0 if not cmd else 1)

    extra  = [a for a in raw if a != cmd]
    phases = build_phases(cmd, extra)

    if not phases:
        sys.exit(f"Unknown command: {cmd}")

    BenchUI(phases).run()
