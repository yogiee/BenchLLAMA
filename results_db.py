"""results_db.py — SQLite system-of-record for BenchLLAMA results (storage migration, Phase 0).

WHY: the dated `results/<battery>_<YYYY-MM-DD>.json` files key results by *calendar date* and each
writer merges differently (runner/ctx_ladder carry-forward; the aptitude averager overwrites). A run
that crosses midnight splits across two dates, and a separate single-/few-model run silently clobbers
the fleet's averaged file — then export.py (latest-file-by-mtime) publishes the truncated set. (Hit
2026-06-29; see memory feedback_aptitude_averager_clobber.)

FIX: identity is a `run_id` stamped ONCE at run start (NOT the date), and every result is a per-model
UPSERT keyed (run_id, model, battery). Re-running one model overwrites only its own row — it can never
drop another model. A midnight crossing is a non-event (same run_id throughout). `latest(battery)`
returns the most-recent result *per model* across all runs, so a partial re-run keeps every other
model's prior result automatically (the carry-forward, done right and uniformly).

This DB is the eventual source of truth; the dated JSON + master.md become generated VIEWS. stdlib
sqlite3 only — no new dependency (Battery E already uses sqlite3 for E3).

API (used by writers in Phase 1, by export/consumers in Phase 2):
    start_run(run_id, flags=...)                      register a run (idempotent)
    record(run_id, model, battery, composite, metrics, per_test=None)   UPSERT one result
    latest(battery)        -> {model: metrics_dict}   latest per-model (anti-clobber read)
    history(model, battery)-> [ {run_id, started_at, composite, metrics}, ... ]   trend
    runs()                 -> [ {run_id, started_at, host, flags}, ... ]
"""
import sqlite3, json, time, socket, os
from pathlib import Path
from contextlib import contextmanager

REPO    = Path(__file__).resolve().parent
DB_PATH = REPO / "results" / "benchllama.db"

# batteries are stored under these canonical keys
BATTERIES = ("standard", "ladder", "A", "B", "C", "D", "E", "F", "F-elastic", "G", "vision", "embedding")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,  -- run-start stamp, e.g. "2026-06-28T19:42" (NOT the calendar date)
    started_at  TEXT NOT NULL,     -- ISO8601
    host        TEXT,
    flags       TEXT,              -- JSON: {"fast":bool,"force":bool,...}
    finished_at TEXT,              -- ISO8601, stamped when the run completes/aborts (NULL while live)
    status      TEXT,              -- done | aborted | error | NULL(running)
    elapsed_s   REAL               -- total wall-clock seconds (orchestrated runs)
);
CREATE TABLE IF NOT EXISTS results (
    run_id     TEXT NOT NULL,
    model      TEXT NOT NULL,
    battery    TEXT NOT NULL,      -- one of BATTERIES
    composite  REAL,               -- nullable headline metric (for fast sort/trend)
    metrics    TEXT,               -- JSON: the model's full per-battery result dict
    per_test   TEXT,               -- JSON: optional per-test detail
    created_at TEXT NOT NULL,      -- when this row was written
    PRIMARY KEY (run_id, model, battery)
);
-- Per-phase wall-clock for an orchestrated pipeline (Standard, ctx Ladder, A-G, Vision, Embedding).
-- cooldowns live INSIDE a phase so they're counted in elapsed_s; the 10s inter-phase pause is not.
CREATE TABLE IF NOT EXISTS phase_timings (
    run_id      TEXT NOT NULL,
    idx         INTEGER NOT NULL,  -- phase position in the pipeline (0-based)
    label       TEXT NOT NULL,     -- e.g. "Research (Battery C)"
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    elapsed_s   REAL,
    status      TEXT,              -- done | error | aborted
    n_models    INTEGER,           -- active models this phase touched
    PRIMARY KEY (run_id, idx)
);
CREATE INDEX IF NOT EXISTS idx_results_model_battery ON results(model, battery);
CREATE INDEX IF NOT EXISTS idx_results_battery       ON results(battery);
"""

# Columns added to `runs` after the original schema shipped — applied to pre-existing DBs by _migrate.
_RUNS_ADDED = (("finished_at", "TEXT"), ("status", "TEXT"), ("elapsed_s", "REAL"),
               ("env", "TEXT"))  # JSON run-provenance fingerprint (bench_utils.env_fingerprint)


def _migrate(c) -> None:
    """Add columns introduced after the first schema to an already-created `runs` table.
    CREATE TABLE IF NOT EXISTS never alters an existing table, so new columns need ALTER."""
    have = {r["name"] for r in c.execute("PRAGMA table_info(runs)").fetchall()}
    for col, decl in _RUNS_ADDED:
        if col not in have:
            c.execute(f"ALTER TABLE runs ADD COLUMN {col} {decl}")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@contextmanager
def _conn(path: Path = DB_PATH):
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    c = sqlite3.connect(str(path))
    c.row_factory = sqlite3.Row
    try:
        c.executescript(SCHEMA)
        _migrate(c)
        yield c
        c.commit()
    finally:
        c.close()


def start_run(run_id: str, flags: dict | None = None, host: str | None = None, path: Path = DB_PATH) -> None:
    """Register a run (idempotent — safe to call from every phase of the same run)."""
    started = run_id if (run_id and run_id[0].isdigit() and "T" in run_id) else _now()
    with _conn(path) as c:
        c.execute("INSERT INTO runs(run_id, started_at, host, flags) VALUES(?,?,?,?) "
                  "ON CONFLICT(run_id) DO NOTHING",
                  (run_id, started, host or socket.gethostname(), json.dumps(flags or {})))


def record(run_id: str, model: str, battery: str, composite=None,
           metrics=None, per_test=None, path: Path = DB_PATH) -> None:
    """UPSERT one model's result for a battery. Overwrites only this (run_id, model, battery)
    row — the anti-clobber core. `metrics` is the model's full result dict (stored as JSON)."""
    start_run(run_id, path=path)  # ensure FK parent exists even if start_run wasn't called first
    with _conn(path) as c:
        c.execute(
            "INSERT INTO results(run_id,model,battery,composite,metrics,per_test,created_at) "
            "VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(run_id,model,battery) DO UPDATE SET "
            "  composite=excluded.composite, metrics=excluded.metrics, "
            "  per_test=excluded.per_test, created_at=excluded.created_at",
            (run_id, model, battery, composite,
             json.dumps(metrics) if metrics is not None else None,
             json.dumps(per_test) if per_test is not None else None, _now()))


def latest(battery: str, path: Path = DB_PATH) -> dict:
    """{model: metrics_dict} — the most recent result PER MODEL for this battery, across all runs.
    A partial re-run never drops other models: each model resolves to its own latest run."""
    with _conn(path) as c:
        rows = c.execute(
            "SELECT r.model, r.metrics, r.composite FROM results r "
            "JOIN runs ru ON ru.run_id=r.run_id "
            "WHERE r.battery=? ORDER BY ru.started_at ASC", (battery,)).fetchall()
    out: dict = {}
    for row in rows:  # ascending → the last write per model wins = latest (ISO timestamps sort right)
        out[row["model"]] = json.loads(row["metrics"]) if row["metrics"] else {"composite": row["composite"]}
    return out


def latest_env_by_model(battery: str, path: Path = DB_PATH) -> dict:
    """{model: run_env_dict} — the provenance fingerprint (runs.env) of the run that produced each
    model's LATEST result for this battery. Powers content-addressed resume (resume.py): the env a
    model was scored under, to diff against the current env. `{}` for a model whose run has no env
    (pre-provenance) — resume treats that as unknown → re-run."""
    with _conn(path) as c:
        rows = c.execute(
            "SELECT r.model, ru.env FROM results r "
            "JOIN runs ru ON ru.run_id=r.run_id "
            "WHERE r.battery=? ORDER BY ru.started_at ASC", (battery,)).fetchall()
    out: dict = {}
    for row in rows:  # ascending → last write per model wins = latest
        try:
            out[row["model"]] = json.loads(row["env"]) if row["env"] else {}
        except Exception:
            out[row["model"]] = {}
    return out


def history(model: str, battery: str, path: Path = DB_PATH) -> list:
    """Chronological trend for one (model, battery)."""
    with _conn(path) as c:
        rows = c.execute(
            "SELECT r.run_id, ru.started_at, r.composite, r.metrics FROM results r "
            "JOIN runs ru ON ru.run_id=r.run_id "
            "WHERE r.model=? AND r.battery=? ORDER BY ru.started_at ASC",
            (model, battery)).fetchall()
    return [{"run_id": x["run_id"], "started_at": x["started_at"], "composite": x["composite"],
             "metrics": json.loads(x["metrics"]) if x["metrics"] else None} for x in rows]


def runs(path: Path = DB_PATH) -> list:
    with _conn(path) as c:
        rows = c.execute("SELECT run_id, started_at, host, flags FROM runs ORDER BY started_at ASC").fetchall()
    return [{"run_id": x["run_id"], "started_at": x["started_at"], "host": x["host"],
             "flags": json.loads(x["flags"] or "{}")} for x in rows]


# ── run/phase timing (orchestrator-driven) ──────────────────────────────────────
# These NEVER raise into the caller: a DB hiccup must not break a live benchmark.

def finish_run(run_id: str, status: str = "done", elapsed_s: float | None = None,
               path: Path = DB_PATH) -> None:
    """Stamp a run complete — run-level wall-clock. Idempotent; safe to call once at pipeline end."""
    try:
        start_run(run_id, path=path)
        with _conn(path) as c:
            c.execute("UPDATE runs SET finished_at=?, status=?, elapsed_s=? WHERE run_id=?",
                      (_now(), status, elapsed_s, run_id))
    except Exception:
        pass


def set_env(run_id: str, env: dict, path: Path = DB_PATH) -> None:
    """Store the run-provenance fingerprint (bench_utils.env_fingerprint) on the run row.
    Idempotent; never raises into the caller (a DB hiccup must not break a live benchmark)."""
    try:
        start_run(run_id, path=path)
        with _conn(path) as c:
            c.execute("UPDATE runs SET env=? WHERE run_id=?", (json.dumps(env), run_id))
    except Exception:
        pass


def ensure_env(run_id: str, models=None, path: Path = DB_PATH) -> None:
    """Guarantee the run row carries a provenance fingerprint. Best-effort, idempotent, never raises.
    Captures bench_utils.env_fingerprint() — a network probe done OUTSIDE any write lock — ONLY when the
    run has none yet, so a standalone/bare script run (`python3 runner.py …`) is provenanced exactly like
    an orchestrated one. Without this, a bare run's carry-forward writes would re-stamp the fleet under an
    env-less run and strip its provenance (the resume `no-provenance` bug, 2026-07-11)."""
    try:
        with _conn(path) as c:
            row = c.execute("SELECT env FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row and row["env"]:
            return
        from bench_utils import env_fingerprint
        set_env(run_id, env_fingerprint(models=list(models) if models else None), path=path)
    except Exception:
        pass


def latest_env(path: Path = DB_PATH) -> dict:
    """Provenance fingerprint of the most recent run that recorded one ({} if none).
    Surfaced by export.py as the `environment` block."""
    try:
        with _conn(path) as c:
            row = c.execute("SELECT env FROM runs WHERE env IS NOT NULL "
                            "ORDER BY started_at DESC LIMIT 1").fetchone()
        return json.loads(row["env"]) if row and row["env"] else {}
    except Exception:
        return {}


def record_phase(run_id: str, idx: int, label: str, started_at: str,
                 finished_at: str | None = None, elapsed_s: float | None = None,
                 status: str = "done", n_models: int | None = None, path: Path = DB_PATH) -> None:
    """UPSERT one pipeline phase's timing. Keyed (run_id, idx) so a re-run of the same phase overwrites."""
    try:
        start_run(run_id, path=path)
        with _conn(path) as c:
            c.execute(
                "INSERT INTO phase_timings(run_id,idx,label,started_at,finished_at,elapsed_s,status,n_models) "
                "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(run_id,idx) DO UPDATE SET "
                "  label=excluded.label, started_at=excluded.started_at, finished_at=excluded.finished_at, "
                "  elapsed_s=excluded.elapsed_s, status=excluded.status, n_models=excluded.n_models",
                (run_id, idx, label, started_at, finished_at, elapsed_s, status, n_models))
    except Exception:
        pass


def run_timings(run_id: str | None = None, path: Path = DB_PATH) -> dict:
    """Timing for one run (default: latest by start). {run:{...}, phases:[{...}, ...]} or {} if none."""
    with _conn(path) as c:
        if run_id is None:
            row = c.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
        else:
            row = c.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            return {}
        ph = c.execute("SELECT * FROM phase_timings WHERE run_id=? ORDER BY idx ASC",
                       (row["run_id"],)).fetchall()
    return {"run": {k: row[k] for k in row.keys()},
            "phases": [{k: p[k] for k in p.keys()} for p in ph]}


def _fmt_dur(s) -> str:
    if s is None:
        return "  —  "
    s = int(round(s))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def format_timings(run_id: str | None = None, path: Path = DB_PATH) -> str:
    """Human-readable per-phase timing block for a run (default latest) — for the run-log + CLI."""
    t = run_timings(run_id, path)
    if not t:
        return "(no run timing recorded yet)"
    r, ph = t["run"], t["phases"]
    total = r.get("elapsed_s")
    if total is None and ph:                       # live/partial: sum what we have
        total = sum(p["elapsed_s"] or 0 for p in ph)
    lines = [f"run {r['run_id']}  status={r.get('status') or 'running'}  total={_fmt_dur(total)}"]
    for p in ph:
        nm = f"({p['n_models']} models)" if p.get("n_models") else ""
        flag = "" if (p.get("status") in (None, "done")) else f"  [{p['status']}]"
        lines.append(f"  {p['label']:<30} {_fmt_dur(p['elapsed_s']):>8}  {nm}{flag}")
    return "\n".join(lines)


# ── dual-write helpers (used by the result-writing scripts in Phase 1) ──────────
_RUN_ID = None

def current_run_id() -> str:
    """run_id for this process: env BENCH_RUN_ID (set by the orchestrator so every phase of one
    pipeline shares it) — else a per-process start timestamp generated once (standalone script run)."""
    global _RUN_ID
    rid = os.environ.get("BENCH_RUN_ID")
    if rid:
        return rid
    if _RUN_ID is None:
        _RUN_ID = time.strftime("%Y-%m-%dT%H-%M-%S")
    return _RUN_ID


def composite_of(rec: dict):
    """Best-effort headline number for a per-model result dict (varies by battery)."""
    s = rec.get("summary", {})
    if isinstance(s, dict):
        for k in ("composite", "composite_long", "clean_depth"):
            if k in s:
                return s[k]
    for k in ("composite", "clean_depth"):
        if k in rec:
            return rec[k]
    return None


def record_all(battery: str, results: list, run_id: str | None = None,
               only: set | None = None, path: Path = DB_PATH) -> int:
    """Dual-write hook called by each writer right after it writes its JSON: UPSERT every per-model
    record for this battery+run in ONE transaction (the write-points fire per-model, so a single
    connection per call keeps it cheap). NEVER raises into the caller — a DB hiccup must not break a
    live benchmark (the JSON write already succeeded). Returns the number of rows written.

    `only` (model-name set) restricts the write to the models actually scored THIS invocation — pass
    the caller's `run_names`. Carried models are then left on their ORIGINAL (run_id, env) row instead
    of being re-stamped under the current run, so a provenance-less/bare run can never strip the rest
    of the fleet's provenance (the `no-provenance` bug, 2026-07-11). only=None keeps every non-error
    row (the force-based averager already passes only its freshly-run rows)."""
    rid = run_id or current_run_id()
    n = 0
    try:
        recs = [r for r in results
                if isinstance(r, dict) and r.get("model") and "error" not in r
                and (only is None or r["model"] in only)]
        started = rid if (rid and rid[0].isdigit() and "T" in rid) else _now()
        now = _now()
        with _conn(path) as c:
            c.execute("INSERT INTO runs(run_id,started_at,host,flags) VALUES(?,?,?,?) "
                      "ON CONFLICT(run_id) DO NOTHING",
                      (rid, started, socket.gethostname(), "{}"))
            for rec in recs:
                c.execute(
                    "INSERT INTO results(run_id,model,battery,composite,metrics,per_test,created_at) "
                    "VALUES(?,?,?,?,?,?,?) ON CONFLICT(run_id,model,battery) DO UPDATE SET "
                    "  composite=excluded.composite, metrics=excluded.metrics, "
                    "  per_test=excluded.per_test, created_at=excluded.created_at",
                    (rid, rec["model"], battery, composite_of(rec), json.dumps(rec), None, now))
                n += 1
        ensure_env(rid, [r["model"] for r in recs], path=path)   # Guard 2: never leave a run env-less
    except Exception:
        pass
    return n
