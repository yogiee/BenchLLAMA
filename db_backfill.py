"""db_backfill.py — one-time import of existing results/*.json into the SQLite store (Phase 0b).

Maps each canonical dated result file → (battery, run_id = its date) and UPSERTs every model, so the
DB starts with full history. Idempotent (UPSERT). Skips per-pass shards (_run1/_run2/_run3/_fast) and
status.json — only the canonical averaged/final files are imported.

    python3 db_backfill.py            # import everything in results/
    python3 db_backfill.py --reset    # wipe the DB first, then import
"""
import json, re, glob, os, sys
from pathlib import Path
import results_db as db

REPO = Path(__file__).resolve().parent
RES  = REPO / "results"

# filename prefix → canonical battery key. ORDER MATTERS: aptitude_f_elastic before aptitude_f.
PREFIX_BATTERY = [
    ("benchmark", "standard"), ("ctx_ladder", "ladder"),
    ("aptitude_a", "A"), ("aptitude_b", "B"), ("aptitude_c", "C"), ("aptitude_d", "D"),
    ("aptitude_e", "E"), ("aptitude_f_elastic", "F-elastic"), ("aptitude_f", "F"),
    ("longctx", "G"), ("vision", "vision"), ("embedding", "embedding"),
]
DATE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})")


def battery_for(base: str):
    for pre, bat in PREFIX_BATTERY:          # f_elastic checked before f (prefix overlap)
        if base.startswith(pre + "_"):
            return bat
    return None


def composite_of(rec: dict):
    s = rec.get("summary", {})
    if isinstance(s, dict):
        for k in ("composite", "composite_long", "clean_depth"):
            if k in s:
                return s[k]
    for k in ("composite", "clean_depth"):
        if k in rec:
            return rec[k]
    return None


def backfill(verbose: bool = True):
    imported = []
    for f in sorted(glob.glob(str(RES / "*.json"))):
        base = os.path.basename(f)
        if base == "status.json" or re.search(r"_(run\d+|fast)\.json$", base):
            continue
        bat = battery_for(base)
        m = DATE_RE.search(base)
        if not bat or not m:
            continue
        run_id = m.group(1)                   # date-as-run_id for historical files
        try:
            data = json.load(open(f))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        db.start_run(run_id, flags={"backfilled": True})
        n = 0
        for rec in data:
            if isinstance(rec, dict) and rec.get("model"):
                db.record(run_id, rec["model"], bat, composite=composite_of(rec), metrics=rec)
                n += 1
        imported.append((base, bat, run_id, n))
        if verbose:
            print(f"  {base:44} → {bat:10} run={run_id} models={n}")
    return imported


if __name__ == "__main__":
    if "--reset" in sys.argv and db.DB_PATH.exists():
        db.DB_PATH.unlink(); print(f"(reset {db.DB_PATH.name})")
    rows = backfill()
    print(f"\n✓ backfilled {len(rows)} files → {db.DB_PATH}")
    # quick integrity read: per-battery latest model counts
    for bat in db.BATTERIES:
        n = len(db.latest(bat))
        if n:
            print(f"  latest({bat:10}) → {n} models")
