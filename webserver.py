#!/usr/bin/env python3
"""
BenchLLAMA — web UI (aiohttp). Drives orchestrator.py, serves the live dashboard.

  python3 webserver.py                      # idle → browser selection screen → Start
  python3 webserver.py <command> [flags]    # pre-populated → auto-starts + monitors
  [--host H] [--port P]

localhost by default; --host 0.0.0.0 exposes on the LAN. NOTE: /api/start spawns processes,
so LAN exposure is a deliberate trust decision (see P5 for read-only/token gating).
"""

import asyncio
import json
import os
import signal
import socket
import sys
import webbrowser
from pathlib import Path

from aiohttp import web

import orchestrator as O
from bench_utils import paint

REPO = Path(__file__).parent
WEB = REPO / "web"
RESULTS = REPO / "results"
RANKINGS_JSON = REPO / "rankings" / "rankings.json"
MASTER_MD = REPO / "rankings" / "master.md"
DEFAULT_PORT = 8077


def _banner(open_url, lan, port, mode, control):
    """Compact, color-when-interactive launch card (TTY-gated via bench_utils.paint)."""
    rule = paint("─" * 46, "dim")
    row  = lambda k, v: print("  " + paint(f"{k:<8}", "dim") + v)
    print()
    print("  " + paint("BenchLLAMA", "bold", "orange") + paint("  ·  web dashboard", "dim"))
    print("  " + rule)
    row("Local",   paint(open_url, "cyan", "bold"))
    if lan:
        row("LAN", paint(f"http://{lan}:{port}", "cyan") + paint("   (phone / iPad · read-only)", "dim"))
    row("Mode",    mode)
    row("Control", paint(control, "dim"))
    print("  " + rule)
    print("  " + paint("Ctrl-C to stop", "dim"))
    print()


def _lan_ip():
    """Best-effort primary LAN IP (no traffic actually sent)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def _build_extra(payload: dict) -> list:
    extra = []
    if payload.get("battery") and payload.get("cmd") == "aptitude":
        extra += ["--battery", str(payload["battery"])]
    if payload.get("fast"):
        extra.append("--fast")
    if payload.get("force"):
        extra.append("--force")
    models = (payload.get("models") or "").split()
    if models:
        extra += ["--models", *models]
    return extra


def start_run(app, cmd: str, extra: list, sort: str = "size"):
    """Create + launch an Orchestrator run. Returns (ok, msg). One run at a time.

    Per-run state lives in the mutable `app["rt"]` holder (created pre-startup) — we
    mutate its CONTENTS, never the app mapping itself, so aiohttp doesn't warn about
    'changing state of a started application' once the server is live."""
    rt = app["rt"]
    orch = rt.get("orch")
    task = rt.get("task")
    if orch and task and not task.done() and not orch.state.finished:
        return False, "a run is already active"
    phases = O.build_phases(cmd, extra)
    if not phases:
        return False, f"unknown command: {cmd}"
    orch = O.Orchestrator(phases)
    orch.sort = sort if sort in ("size", "name", "fresh") else "size"   # → BENCH_SORT for subprocs
    rt["orch"] = orch
    rt["task"] = asyncio.create_task(orch.run())
    return True, "started"


async def _index(request):
    # no-store: the dashboard is a single file we iterate on; never serve a stale cached copy
    return web.FileResponse(WEB / "index.html",
                            headers={"Cache-Control": "no-store, must-revalidate"})


async def _models(request):
    """Registered models (for the selector multi-select). name + role + caps + extended."""
    try:
        reg = O.sort_registry(json.loads((REPO / "models.json").read_text()))
    except Exception:
        return web.json_response([])
    return web.json_response([
        {"name": m["name"], "role": m.get("role"), "disk_gb": m.get("disk_gb"), "added_idx": m.get("added_idx"),
         "caps": m.get("capabilities", []), "extended_roles": m.get("extended_roles", []),
         "cloud": bool(m.get("cloud", False))}
        for m in reg
    ])


def _readonly(request):
    """Per-connection: control allowed from localhost; LAN clients are read-only
    unless the server was started with --allow-control."""
    if request.app["allow_control"]:
        return False
    return (request.remote or "") not in ("127.0.0.1", "::1")


async def _ws(request):
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    cursor = 0
    ro = _readonly(request)
    try:
        while not ws.closed:
            orch = request.app["rt"].get("orch")
            if orch is None:
                await ws.send_json({"idle": True, "read_only": ro})
                cursor = 0
            else:
                total = orch.state.log_total
                base = total - len(orch.state.log)
                start = 0 if cursor < base else cursor - base
                new = orch.state.log[start:]
                cursor = total
                await ws.send_json({"snapshot": orch.snapshot(), "log": new, "read_only": ro})
            await asyncio.sleep(0.4)
    except (asyncio.CancelledError, ConnectionResetError):
        pass
    return ws


def _guard(request):
    if _readonly(request):
        return web.json_response({"ok": False, "msg": "read-only (LAN) — control from the host machine, or relaunch with --allow-control"}, status=403)
    return None


async def _start(request):
    blocked = _guard(request)
    if blocked:
        return blocked
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"ok": False, "msg": "bad json"}, status=400)
    if payload.get("cmd") not in O.COMMANDS:
        return web.json_response({"ok": False, "msg": "bad command"}, status=400)
    ok, msg = start_run(request.app, payload["cmd"], _build_extra(payload), payload.get("sort", "size"))
    return web.json_response({"ok": ok, "msg": msg}, status=200 if ok else 409)


async def _stop(request):
    blocked = _guard(request)
    if blocked:
        return blocked
    orch = request.app["rt"].get("orch")
    if orch:
        orch.stop()
    return web.json_response({"ok": True})


# ── results / rankings (read-only viewer) ──────────────────────────────────────

async def _rankings(request):
    if not RANKINGS_JSON.exists():
        return web.json_response({"error": "no rankings.json yet — run export.py"}, status=404)
    return web.json_response(json.loads(RANKINGS_JSON.read_text()))


async def _model_detail(request):
    """Drill-down evidence for one model: its per-test prompt+response from the newest
    canonical standard-suite file (skips _fast). Powers the dashboard's click-a-row modal —
    the structured per-battery numbers come from rankings.json client-side; this serves the
    'actual prompt + response that produced the score' for the standard suite."""
    name = request.match_info["name"]
    files = sorted(
        [p for p in RESULTS.glob("benchmark_*.json")
         if "_fast" not in p.name and p.name[10:11].isdigit()],
        key=lambda p: p.stat().st_mtime, reverse=True)
    fallback = None  # newest record found, even if it errored with no tests
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        rec = next((r for r in data if r.get("model") == name), None)
        if rec is None:
            continue
        if rec.get("tests"):                       # prefer a record with actual evidence
            return web.json_response({"source": f.name, "model": rec})
        if fallback is None:
            fallback = {"source": f.name, "model": rec}
    if fallback:
        return web.json_response(fallback)
    return web.json_response({"error": "no standard-suite record for this model"}, status=404)


async def _results_list(request):
    files = sorted(RESULTS.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    return web.json_response([{"name": f.name, "mtime": int(f.stat().st_mtime),
                               "size": f.stat().st_size} for f in files])


async def _result_file(request):
    name = request.match_info["name"]
    f = (RESULTS / name)
    # path-traversal guard: must resolve inside RESULTS and be a result file
    if "/" in name or ".." in name or f.suffix not in (".md", ".json") or not f.is_file():
        return web.Response(status=404, text="not found")
    return web.Response(text=f.read_text(), content_type="text/plain")


async def _master(request):
    return web.Response(text=MASTER_MD.read_text() if MASTER_MD.exists() else "# (no master.md yet)",
                        content_type="text/plain")


async def _on_startup(app):
    boot = app.get("_boot")
    if boot:
        start_run(app, *boot)


async def _on_cleanup(app):
    orch = app["rt"].get("orch")
    if orch:
        orch.stop()
    task = app["rt"].get("task")
    if task:
        task.cancel()


def main():
    raw = sys.argv[1:]
    cmd = next((a for a in raw if a in O.COMMANDS), None)
    host = O._arg(raw, "--host") or "0.0.0.0"   # default: reachable on the LAN (monitoring)
    port = int(O._arg(raw, "--port") or DEFAULT_PORT)

    allow_control = "--allow-control" in raw

    app = web.Application()
    app["rt"] = {"orch": None, "task": None}   # mutable run holder (set pre-startup; contents mutated live)
    app["allow_control"] = allow_control
    if cmd:
        boot_extra = [a for a in raw if a not in (cmd, "--host", host, "--port", str(port), "--allow-control")]
        app["_boot"] = (cmd, boot_extra)
    app.router.add_get("/", _index)
    app.router.add_get("/ws", _ws)
    app.router.add_post("/api/start", _start)
    app.router.add_post("/api/stop", _stop)
    app.router.add_get("/api/models", _models)
    app.router.add_get("/api/rankings", _rankings)
    app.router.add_get("/api/model/{name}", _model_detail)
    app.router.add_get("/api/results", _results_list)
    app.router.add_get("/api/results/{name}", _result_file)
    app.router.add_get("/api/master", _master)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)

    # 0.0.0.0 includes localhost, so open the local URL on the host machine
    open_url = f"http://localhost:{port}" if host in ("0.0.0.0", "127.0.0.1", "localhost") else f"http://{host}:{port}"
    lan = _lan_ip() if host == "0.0.0.0" else None
    mode = f"monitoring {cmd}" if cmd else "select a run in the browser"
    control = "all clients" if allow_control else "this machine only · LAN read-only"
    _banner(open_url, lan, port, mode, control)
    try:
        asyncio.run(_serve(app, host, port, open_url))
    except KeyboardInterrupt:
        pass   # belt-and-suspenders: a SIGINT racing before handlers are installed


async def _serve(app, host, port, open_url):
    """Run the app with our OWN signal handling so a single Ctrl-C exits cleanly.

    aiohttp's web.run_app installs handlers that often swallow the first SIGINT (you had to
    press Ctrl-C twice). Here the first Ctrl-C/SIGTERM sets the stop event → graceful cleanup
    (which stops any active orchestrator run); a second forces an immediate exit."""
    runner = web.AppRunner(app)
    await runner.setup()                    # fires on_startup (auto-run if launched with a command)
    site = web.TCPSite(runner, host, port)
    await site.start()
    try:
        webbrowser.open(open_url)           # open only after the server is actually accepting
    except Exception:
        pass

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    presses = {"n": 0}
    def _sig():
        presses["n"] += 1
        if presses["n"] >= 2:
            os._exit(130)                   # second Ctrl-C → hard exit, no waiting
        print(paint("\n  shutting down…", "dim"), flush=True)
        stop.set()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, _sig)
        except (NotImplementedError, RuntimeError):
            pass                            # non-Unix / no running loop → fall back to KeyboardInterrupt

    try:
        await stop.wait()
    finally:
        await runner.cleanup()              # fires on_cleanup → stops the orchestrator run


if __name__ == "__main__":
    main()
