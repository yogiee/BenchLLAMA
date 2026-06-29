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
    run_id     TEXT PRIMARY KEY,   -- run-start stamp, e.g. "2026-06-28T19:42" (NOT the calendar date)
    started_at TEXT NOT NULL,      -- ISO8601
    host       TEXT,
    flags      TEXT                -- JSON: {"fast":bool,"force":bool,...}
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
CREATE INDEX IF NOT EXISTS idx_results_model_battery ON results(model, battery);
CREATE INDEX IF NOT EXISTS idx_results_battery       ON results(battery);
"""


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


def record_all(battery: str, results: list, run_id: str | None = None, path: Path = DB_PATH) -> int:
    """Dual-write hook called by each writer right after it writes its JSON: UPSERT every per-model
    record for this battery+run in ONE transaction (the write-points fire per-model, so a single
    connection per call keeps it cheap). NEVER raises into the caller — a DB hiccup must not break a
    live benchmark (the JSON write already succeeded). Returns the number of rows written."""
    rid = run_id or current_run_id()
    n = 0
    try:
        started = rid if (rid and rid[0].isdigit() and "T" in rid) else _now()
        now = _now()
        with _conn(path) as c:
            c.execute("INSERT INTO runs(run_id,started_at,host,flags) VALUES(?,?,?,?) "
                      "ON CONFLICT(run_id) DO NOTHING",
                      (rid, started, socket.gethostname(), "{}"))
            for rec in results:
                if isinstance(rec, dict) and rec.get("model") and "error" not in rec:
                    c.execute(
                        "INSERT INTO results(run_id,model,battery,composite,metrics,per_test,created_at) "
                        "VALUES(?,?,?,?,?,?,?) ON CONFLICT(run_id,model,battery) DO UPDATE SET "
                        "  composite=excluded.composite, metrics=excluded.metrics, "
                        "  per_test=excluded.per_test, created_at=excluded.created_at",
                        (rid, rec["model"], battery, composite_of(rec), json.dumps(rec), None, now))
                    n += 1
    except Exception:
        pass
    return n
