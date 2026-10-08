#!/usr/bin/env python3
"""
BenchLLAMA — sleep-guard check (CLAUDE.md Working Rule #4). App-agnostic.

Keep-awake apps (Amphetamine, Vorssaint, `caffeinate`, …) don't change the OS sleep
settings. Each holds an IOKit power assertion while it's armed. So the question "is
something keeping this Mac awake for an unattended run?" is answered from
`pmset -g assertions`, whichever app holds it.

A holder counts as a GUARD only if its assertion
  · is PreventUserIdleSystemSleep or PreventSystemSleep,
  · has NO timeout (a timed hold lapses mid-run — e.g. Claude Code's own
    `caffeinate -i -t 300`, renewed only while a session is active), and
  · is owned by a user-level process, not an Apple system daemon (powerd's
    "while display is on" ends when the display sleeps; bluetoothd / sharingd /
    useractivityd holds are transient).

Exit 0 = guarded · 1 = NOT guarded · 2 = can't tell (not macOS / pmset failed).

  python3 sleep_guard.py          # human-readable report
  python3 sleep_guard.py --quiet  # exit code only
"""

import platform
import re
import subprocess
import sys

TYPES = ("PreventUserIdleSystemSleep", "PreventSystemSleep")
SYSTEM_PREFIXES = ("/System/", "/usr/libexec/", "/usr/sbin/")
_HOLDER = re.compile(r"^\s+pid (\d+)\((.+?)\): \[[^\]]*\] (\S+) (\w+) named: \"(.*)\"")


def _exe(pid: str) -> str:
    try:
        return subprocess.run(["ps", "-o", "comm=", "-p", pid],
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def holders() -> list[dict] | None:
    """Every sleep-preventing assertion, classified. None if pmset can't be read."""
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.run(["pmset", "-g", "assertions"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    rows, cur = [], None
    for line in out.splitlines():
        m = _HOLDER.match(line)
        if m:
            pid, name, age, kind, label = m.groups()
            cur = {"pid": pid, "name": name, "age": age, "type": kind, "label": label, "timeout": None}
            if kind in TYPES:
                rows.append(cur)
        elif cur is not None and line.startswith("\t"):
            t = re.search(r"Timeout will fire in (\d+) secs", line)
            if t:
                cur["timeout"] = int(t.group(1))
        elif not line.startswith("\t"):
            cur = None
    for r in rows:
        exe = _exe(r["pid"])
        r["system"] = exe.startswith(SYSTEM_PREFIXES)
        r["guard"] = r["timeout"] is None and not r["system"]
    return rows


def guarded() -> bool | None:
    rows = holders()
    return None if rows is None else any(r["guard"] for r in rows)


if __name__ == "__main__":
    quiet = "--quiet" in sys.argv
    rows = holders()
    if rows is None:
        if not quiet:
            print("sleep-guard: can't tell (not macOS, or pmset failed)")
        sys.exit(2)
    ok = any(r["guard"] for r in rows)
    if not quiet:
        for r in rows:
            why = ("GUARD" if r["guard"] else
                   f"timed ({r['timeout']}s left)" if r["timeout"] is not None else "system daemon")
            print(f"  {'✓' if r['guard'] else '·'} {r['name']} (pid {r['pid']}, held {r['age']}): "
                  f"{r['type']} \"{r['label']}\" — {why}")
        print("sleep-guard: ARMED" if ok else
              "sleep-guard: NOT ARMED — nothing holds the Mac awake for the whole run")
    sys.exit(0 if ok else 1)
