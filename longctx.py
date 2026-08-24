#!/usr/bin/env python3
"""
BenchLLAMA — Battery G: Long-Context Retrieval & Degradation.

The GraphWalks analog (inspired by disler/live-bench), BenchLLAMA-style: fill the context to
controlled token depths with plausible distractor text, plant verifiable NEEDLES at known
positions, and measure both ACCURACY degradation and the prefill/decode SPEED collapse as the
window fills. Distinct from Battery C's C4 (which re-runs a SHORT prompt at bigger num_ctx
*allocations* and never fills the window) — here the window is genuinely loaded, so we measure
whether the model can still reason over N filled tokens, and how slow prefill gets there.

Per depth bucket (1 call): 3 single-needle retrievals at early/mid/late positions (the
"lost in the middle" probe) + 1 three-hop manage-chain walk. All objective, exact-match graded.

Capability/role: COMPLETION models (worker + router), like Battery E — selected by completion
capability, NOT a role gate. `utility` specialists (embedding/vision/OCR) are skipped.

  python3 longctx.py                          # all completion models (resumes within 24h)
  python3 longctx.py gemma4:12b llama3.2:3b   # specific models (merge into existing JSON)
  python3 longctx.py --role worker            # filter by role
  python3 longctx.py --capable-only           # skip models that failed `calculate` in the latest standard run
  python3 longctx.py --fast                    # skip inter-model cool-down
  python3 longctx.py --force                    # ignore the 24h resume window
  python3 longctx.py --rescore                 # re-derive stored summaries after a grading-bar change
  python3 longctx.py --rescore --dry-run       # ...preview the clean_depth moves, write nothing
  python3 longctx.py --ollama http://host:11434

Deep (32768) bucket: run `python3 suites/longctx/build.py --deep` first to add it to the dataset.
"""

import json
import re
import sys
import time
import requests
from pathlib import Path
from datetime import date
from bench_utils import (cooldown, preflight, latest_result, sort_registry,
                         G_CORE_THRESHOLD, G_HARD_THRESHOLD, post_with_budget_retry)

REPO        = Path(__file__).parent
RESULTS_DIR = REPO / "results"
PROMPTS_DIR = REPO / "prompts"
DATASET     = REPO / "suites" / "longctx" / "dataset.json"
STATUS_FILE = RESULTS_DIR / "status.json"

RESULTS_DIR.mkdir(exist_ok=True)

# ── CLI ───────────────────────────────────────────────────────────────────────

def _flag(name):
    return name in sys.argv

def _arg(name, default=None):
    if name in sys.argv:
        idx = sys.argv.index(name)
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--"):
            return sys.argv[idx + 1]
    return default

fast_mode     = _flag("--fast")
force         = _flag("--force")
capable_only  = _flag("--capable-only")
ollama_host   = _arg("--ollama", "http://localhost:11434")
role_filter   = _arg("--role")
model_args    = [a for a in sys.argv[1:] if not a.startswith("--")
                 and a not in (ollama_host, role_filter)]

TIMEOUT     = 600
COOLDOWN    = 0 if fast_mode else 300
NUM_PREDICT = 320
CTX_MARGIN  = 1536   # num_ctx = bucket + margin → holds our own prompt + the short answer
COMPLETION_ROLES = ("worker", "router")

# ── Grading (objective, exact-match) ──────────────────────────────────────────

_ABSENT_MARKERS = ("not stated", "not listed", "not mentioned", "not specified", "not given",
                   "not provided", "not in the text", "no code", "not found", "unknown", "n/a",
                   "does not appear", "doesn't appear", "isn't stated", "is not present", "absent")

def _answer_lines(response):
    """Map numbered answer index -> that line's text. The prompt demands `N. answer` on its own
    line; several hard sub-tasks (absent / superseded) can only be graded on the SCOPED answer,
    because a whole-response substring scan would match the stale value or a decoy code that the
    model correctly used for a DIFFERENT question."""
    out = {}
    for line in response.splitlines():
        m = re.match(r"\s*\**\s*(\d+)\s*[.):\-]\s*(.*)", line)
        if m:
            idx = int(m.group(1))
            if idx not in out:
                out[idx] = m.group(2).strip()
    return out


def grade(response, key, qidx=None):
    """Two-band grading (dataset v2). G-core = 3 positional needles; G-hard = superseded /
    aggregate / multihop / absent. Core falls back to a whole-response scan when the numbered
    format is not followed (lenient, and matches v1 behaviour); the hard sub-tasks that depend on
    scoping are marked missed instead, with `parse_failed` recorded so it is visible rather than
    silent."""
    qidx = qidx or {}
    ans  = _answer_lines(response)
    lo   = response.lower()

    def scoped(task):
        return ans.get(qidx.get(task, -1))

    def core_hit(task):
        want = key[task]
        a = scoped(task)
        return (want in a) if a is not None else (want in response)   # lenient fallback

    hits, parse_failed = {}, False

    # ── G-core: the 3 positional needles (vs 6 same-format decoy relays) ──
    for t in ("needle_early", "needle_mid", "needle_late"):
        hits[t] = core_hit(t)

    # ── G-hard ──
    # superseded: must give the CURRENT value and not the stale one it replaced
    a = scoped("superseded")
    if a is None:
        parse_failed = True
        hits["superseded"] = False
    else:
        hits["superseded"] = (key["superseded"] in a) and (key["superseded_stale"] not in a)

    # aggregate: the sum of two scattered codes — not lexically present anywhere in the haystack
    a = scoped("aggregate")
    hits["aggregate"] = (key["aggregate"] in a) if a is not None else (key["aggregate"] in response)

    # multihop: walk Marcus's chain to its root, without landing on a decoy chain's root
    a = scoped("multihop")
    tgt = key["multihop"].lower()
    decoys = [d.lower() for d in key.get("multihop_decoys", [])]
    if a is not None:
        al = a.lower()
        hits["multihop"] = (tgt in al) and not any(d in al for d in decoys)
    else:
        hits["multihop"] = (tgt in lo) and not any(d in lo for d in decoys)

    # absent: the relay is never mentioned — the model must SAY SO rather than invent a code
    a = scoped("absent")
    if a is None:
        parse_failed = True
        hits["absent"] = False
    else:
        al = a.lower()
        hits["absent"] = any(mk in al for mk in _ABSENT_MARKERS) and not re.search(r"\b\d{4}\b", a)

    core_keys = ("needle_early", "needle_mid", "needle_late")
    hard_keys = ("superseded", "aggregate", "multihop", "absent")
    core = sum(hits[k] for k in core_keys) / len(core_keys)
    hard = sum(hits[k] for k in hard_keys) / len(hard_keys)
    score = sum(hits.values())
    return {
        "hits": hits,
        "found": score, "max": len(hits),
        "core_accuracy": round(core, 3),
        "hard_accuracy": round(hard, 3),
        # composite accuracy: the two bands weighted equally. G-core saturated at 1.00 field-wide
        # on the v1 battery, so weighting it any higher just compresses the range again.
        "accuracy": round(0.5 * core + 0.5 * hard, 3),
        "parse_failed": parse_failed,
    }

# ── Ollama helpers ────────────────────────────────────────────────────────────

def chat(model, messages, num_ctx, max_tokens=NUM_PREDICT):
    payload = {
        "model":    model,
        "messages": messages,
        "stream":   False,
        "options":  {"num_ctx": num_ctx, "num_predict": max_tokens},
        "think":    False,
    }
    def _post(pl):
        t0 = time.time()
        r  = requests.post(f"{ollama_host}/api/chat", json=pl, timeout=TIMEOUT)
        if r.status_code == 400 and "think" in pl:
            pl = {k: v for k, v in pl.items() if k != "think"}
            t0 = time.time()
            r  = requests.post(f"{ollama_host}/api/chat", json=pl, timeout=TIMEOUT)
        wall = time.time() - t0
        r.raise_for_status()
        return r.json(), wall

    # An always-reasoning model spends this budget on thinking and returns empty content; retry
    # once with headroom rather than grading the silence as a wrong answer. See bench_utils.
    return post_with_budget_retry(payload, _post, label=model)

def tps(data):
    ec = data.get("eval_count", 0); ed = data.get("eval_duration", 1)
    return round(ec / (ed / 1e9), 1) if ec and ed else None

def prefill_tps(data):
    pc = data.get("prompt_eval_count", 0); pd = data.get("prompt_eval_duration", 0)
    return round(pc / (pd / 1e9), 1) if pc and pd else None

def unload(model_name):
    try:
        requests.post(f"{ollama_host}/api/chat",
                      json={"model": model_name, "messages": [], "keep_alive": 0}, timeout=15)
    except Exception:
        pass

def _ws(model, phase, bucket=None):
    try:
        payload = {"segment": "longctx", "model": model, "phase": phase, "ts": time.time()}
        if bucket is not None:
            payload["bucket"] = bucket
        STATUS_FILE.write_text(json.dumps(payload))
    except Exception:
        pass

# ── Per-model battery ─────────────────────────────────────────────────────────

def run_longctx(model_name, role, disk_gb, items):
    prompt_file = "router_default.md" if role == "router" else "worker_default.md"
    sys_prompt  = (PROMPTS_DIR / prompt_file).read_text()
    sys_msgs    = [{"role": "system", "content": sys_prompt}]

    print(f"\n{'='*60}", flush=True)
    print(f"MODEL: {model_name}  role={role}  buckets: {[it['bucket'] for it in items]}", flush=True)
    print("=" * 60, flush=True)

    depths = {}
    for it in items:
        bucket  = it["bucket"]
        num_ctx = bucket + CTX_MARGIN
        _ws(model_name, "running", bucket=bucket)
        print(f"\n  ── depth={bucket} (num_ctx={num_ctx}) ──", flush=True)
        try:
            data, wall = chat(model_name,
                              sys_msgs + [{"role": "user", "content": it["prompt"]}],
                              num_ctx=num_ctx)
            resp = data.get("message", {}).get("content", "")
            g    = grade(resp, it["answer_key"], it.get("question_index"))
            entry = {
                "accuracy":      g["accuracy"],
                "core_accuracy": g["core_accuracy"],
                "hard_accuracy": g["hard_accuracy"],
                "parse_failed":  g["parse_failed"],
                "found":        g["found"],
                "hits":         g["hits"],
                "prefill_tps":  prefill_tps(data),
                "decode_tps":   tps(data),
                "wall_s":       round(wall, 1),
                "prompt_tokens": data.get("prompt_eval_count"),
                "response":     resp,
            }
            depths[str(bucket)] = entry
            ok = [k.replace("needle_", "").replace("multihop", "hop") for k, v in g["hits"].items() if v]
            print(f"    acc={g['accuracy']} (core={g['core_accuracy']} hard={g['hard_accuracy']})"
                  f"  found={g['found']}/{g['max']} [{', '.join(ok) or '—'}]"
                  f"  prefill={entry['prefill_tps']}  tps={entry['decode_tps']}"
                  f"  tok={entry['prompt_tokens']}  wall={wall:.1f}s", flush=True)
        except Exception as e:
            print(f"    FAILED: {e}", flush=True)
            depths[str(bucket)] = {"error": str(e)}
        unload(model_name)
        time.sleep(2)

    return {"model": model_name, "role": role, "disk_gb": disk_gb,
            "depths": depths, "summary": summarize(depths)}

# ── Summary (export-friendly) ─────────────────────────────────────────────────

def summarize(depths):
    graded = {int(b): e for b, e in depths.items() if "accuracy" in e}
    if not graded:
        return {"composite": None, "clean_depth": None, "n_depths": 0}
    buckets = sorted(graded)
    # Bars come from bench_utils, NOT the dataset meta — grading policy must be movable without
    # changing the dataset hash (which is a resume trigger and would force a pointless re-measure).
    # dataset.json may still carry older *_threshold keys; they are documentation, not authority.
    core_bar, hard_bar = G_CORE_THRESHOLD, G_HARD_THRESHOLD

    acc_by   = {b: graded[b]["accuracy"]    for b in buckets}
    core_by  = {b: graded[b].get("core_accuracy") for b in buckets}
    hard_by  = {b: graded[b].get("hard_accuracy") for b in buckets}
    pre_by   = {b: graded[b]["prefill_tps"] for b in buckets}
    dec_by   = {b: graded[b]["decode_tps"]  for b in buckets}
    wall_by  = {b: graded[b]["wall_s"]      for b in buckets}
    tok_by   = {b: graded[b]["prompt_tokens"] for b in buckets}

    # Deepest bucket clearing BOTH bands, walking up from the shallow end and TOLERATING ONE DIP.
    #
    # v1 gated on a single 0.75 accuracy bar over 4 sub-tasks, which the 3 trivial needles satisfied
    # on their own — every model in the 08-21 fleet reported clean_depth 32768, including one that
    # failed the multi-hop at every depth. Requiring both bands is the fix.
    #
    # The one-dip tolerance is NOT leniency, it is a variance correction, and it had to land in the
    # same change as the tighter hard bar (2026-08-22). Each depth is ONE call scored on 7 binary
    # sub-tasks, so a single unlucky reply is well inside noise. Under strict break-on-first-failure
    # `ornith:9b` — which answers 4-of-4 at 32768 and 3-of-4 at every other depth — scored
    # clean_depth 2048 off one dip at 4096, and `gemma4:latest` (4-of-4 at 1024/4096/8192) scored
    # 1024. Break-on-SECOND-failure returns 32768 and 8192. The alternative, taking the deepest
    # clearing bucket and ignoring dips entirely, rides noise straight past a real failure — it puts
    # `gemma4:latest` at 32768 despite a 2-of-4 at 16384 — so the dip is tolerated, not forgotten.
    # If the per-depth call is ever repeated N times, drop this back to strict contiguity.
    #
    # ⚠ compare with an epsilon: per-depth accuracies are stored ROUNDED (2/3 -> 0.667), so a bar
    # written as the "obvious" 0.67 would silently mean 3-of-3 rather than 2-of-3.
    EPS = 1e-6
    def _clears(b):
        return ((core_by[b] is None or core_by[b] >= core_bar - EPS) and
                (hard_by[b] is None or hard_by[b] >= hard_bar - EPS))
    clean, dips = None, 0
    for b in buckets:
        if _clears(b):
            clean = b
        else:
            dips += 1
            if dips > 1:
                break
    # clean_depth on the CORE band alone — kept so the v1 series stays interpretable/comparable.
    # Deliberately keeps v1's STRICT break-on-first-failure walk: its whole job is to be comparable
    # to the pre-2026-08-21 numbers, so it must not adopt the new dip tolerance.
    clean_core = None
    for b in buckets:
        if core_by[b] is None or core_by[b] >= core_bar - EPS:
            clean_core = b
        else:
            break

    # per-sub-task recall across depths — position effects AND which hard task breaks first
    keys = ["needle_early", "needle_mid", "needle_late", "superseded", "aggregate", "multihop", "absent"]
    tally = {k: 0 for k in keys}
    for b in buckets:
        for k, v in graded[b]["hits"].items():
            if k in tally:
                tally[k] += int(bool(v))
    n = len(buckets)
    recall = {k: round(v / n, 3) for k, v in tally.items()}
    position_recall = {                       # back-compat shape for the export/consumers
        "early": recall["needle_early"], "mid": recall["needle_mid"],
        "late":  recall["needle_late"],  "hop": recall["multihop"],
    }

    shallow, deep = buckets[0], buckets[-1]
    pre_collapse = (round(pre_by[deep] / pre_by[shallow], 3)
                    if pre_by.get(shallow) and pre_by.get(deep) else None)
    mean = lambda d: (round(sum(v for v in d.values() if v is not None) / n, 3)
                      if any(v is not None for v in d.values()) else None)

    return {
        "composite":      round(sum(acc_by.values()) / n, 3),   # mean two-band accuracy across depths
        "composite_core": mean(core_by),
        "composite_hard": mean(hard_by),
        "clean_depth":    clean,                                 # BOTH bands — the headline
        "clean_depth_core": clean_core,                          # core band only (v1-comparable)
        "core_threshold": core_bar,
        "hard_threshold": hard_bar,
        "threshold":      core_bar,
        "n_depths":       n,
        "accuracy_by_depth":  acc_by,
        "core_by_depth":      core_by,
        "hard_by_depth":      hard_by,
        "prefill_by_depth":   pre_by,
        "decode_by_depth":    dec_by,
        "wall_by_depth":      wall_by,
        "tokens_by_depth":    tok_by,
        "subtask_recall":     recall,
        "position_recall":    position_recall,
        "parse_failures":     sum(1 for b in buckets if graded[b].get("parse_failed")),
        "prefill_collapse":   pre_collapse,   # prefill tok/s at deepest ÷ shallowest (<1 = slowdown)
    }

# ── Markdown ──────────────────────────────────────────────────────────────────

def write_summary(results, out_md, fast_mode=False):
    flag = " ⚠ FAST MODE" if fast_mode else ""
    meta = json.loads(DATASET.read_text())["meta"]
    lines = [
        f"# Battery G — Long-Context Retrieval — {out_md.stem}{flag}", "",
        "Fill the window to each token depth, plant facts among plausible distractors, grade "
        "exact-match. Measures accuracy degradation **and** prefill/decode speed collapse as the "
        "context fills — the window-fill the C4 ctx_depth probe never does.", "",
        "**TWO-BAND (dataset v2, 2026-08-21).** **G-core** = 3 positional needles (early/mid/late) "
        "against **6 same-format decoy relays** — the *can it retrieve at depth?* gate. **G-hard** = "
        "`superseded` (a value corrected later in the log — the answer is the LATEST) · `aggregate` "
        "(the SUM of two scattered codes, lexically absent from the haystack) · `multihop` (a 3-hop "
        "manage-chain walk) · `absent` (a relay never mentioned — the model must SAY "
        "SO rather than invent a code: confabulation-at-depth). "
        "`composite = 0.5·core + 0.5·hard`.", "",
        f"Buckets: {meta['buckets']} · **clean = deepest depth clearing BOTH bands** "
        f"(core ≥ {G_CORE_THRESHOLD} = 2 of 3 needles, hard ≥ {G_HARD_THRESHOLD} = 3 of 4 "
        f"discriminators), walking up from the shallow end and tolerating ONE dip · "
        "collapse = prefill t/s at deepest ÷ shallowest.", "",
        "> ⚠ **Not comparable to Battery G runs before 2026-08-21.** v1 gated `clean_depth` on a "
        "single 0.75 bar over 4 sub-tasks, which the 3 trivial needles satisfied by themselves — "
        "all 21 models reported clean-32k, including one that failed the multi-hop at every depth. "
        "`clean_depth_core` is carried in the JSON for continuity with the v1 series.", "",
        "> ⚠ **Bars re-tuned 2026-08-22** (`longctx.py --rescore`, no re-measure — the replies were "
        "already on disk). The 08-21 fleet run showed the hard band still leaking: `absent` (0.984 "
        "mean recall) and `superseded` (0.960) alone satisfied the old 2-of-4 bar, so 15/21 models "
        "reported clean-32k while failing BOTH real discriminators. Hard is now 3-of-4 — which "
        "requires at least one of `aggregate`/`multihop` — paired with the one-dip tolerance, "
        "since a stricter bar makes break-on-first-failure brittle at one call per depth. "
        "`composite`, the per-depth accuracies and every speed number are UNCHANGED; only "
        "`clean_depth` moved (6 models).", "",
        "| Model | Role | Disk | Comp | core | hard | Clean (both) | Clean (core) | Collapse | early | mid | late | sup | agg | hop | absent |",
        "|-------|------|-----:|-----:|-----:|-----:|-------------:|-------------:|---------:|:----:|:---:|:----:|:---:|:---:|:---:|:------:|",
    ]
    for r in sorted(results, key=lambda r: (r["summary"].get("clean_depth") or 0,
                                            r["summary"].get("composite") or 0), reverse=True):
        s  = r["summary"]
        sr = s.get("subtask_recall", {})
        lines.append(
            f"| `{r['model']}` | {r.get('role','')} | {r.get('disk_gb','?')}GB"
            f" | {s.get('composite','?')} | {s.get('composite_core','—')} | {s.get('composite_hard','—')}"
            f" | {s.get('clean_depth') or '—'} | {s.get('clean_depth_core') or '—'}"
            f" | {s.get('prefill_collapse','—') or '—'}"
            f" | {sr.get('needle_early','?')} | {sr.get('needle_mid','?')} | {sr.get('needle_late','?')}"
            f" | {sr.get('superseded','?')} | {sr.get('aggregate','?')} | {sr.get('multihop','?')}"
            f" | {sr.get('absent','?')} |"
        )

    # ⚠ Bucket keys are ints in a freshly-summarized record but STRINGS in one carried forward
    # from the DB (JSON round-trip). Look up both, or every resumed model renders as an empty
    # row while the three that re-ran this pass render fine — the shape the 08-23 report had.
    def _at(by, b):
        v = by.get(b)
        return by.get(str(b), "—") if v is None else v

    lines += ["", "## Accuracy × depth", "",
              "| Model | " + " | ".join(str(b) for b in meta["buckets"]) + " |",
              "|-------|" + "|".join("----:" for _ in meta["buckets"]) + "|"]
    for r in results:
        acc = r["summary"].get("accuracy_by_depth", {})
        row = " | ".join(str(_at(acc, b)) for b in meta["buckets"])
        lines.append(f"| `{r['model']}` | {row} |")

    lines += ["", "## Prefill tok/s × depth (speed collapse)", "",
              "| Model | " + " | ".join(str(b) for b in meta["buckets"]) + " |",
              "|-------|" + "|".join("----:" for _ in meta["buckets"]) + "|"]
    for r in results:
        pre = r["summary"].get("prefill_by_depth", {})
        row = " | ".join(str(_at(pre, b)) for b in meta["buckets"])
        lines.append(f"| `{r['model']}` | {row} |")

    out_md.write_text("\n".join(lines))
    print(f"MD → {out_md}", flush=True)

# ── Rescore (grading-policy change, no model calls) ───────────────────────────

def rescore(dry_run=False):
    """Re-derive every stored two-band summary from the PERSISTED per-depth hits.

    A grading bar is not test content: the prompts, the haystack and the model replies are already
    on disk and would come back byte-identical from a re-run. Re-measuring 22 models for 2h to move
    a threshold would only re-roll the speed numbers and burn a thermal cycle, so the bars live in
    bench_utils (not the hashed dataset) and this path recomputes summaries in place instead.

    Rows are UPDATEd on their ORIGINAL (run_id, created_at) — a rescore must never re-stamp a score
    under today's run, or it strips the provenance that resume reads (the 2026-07-11 no-provenance
    bug). Only v2 rows are touched: v1 predates the core/hard split and its four sub-tasks cannot be
    re-graded against a two-band bar.
    """
    import sqlite3, results_db
    con = sqlite3.connect(results_db.DB_PATH)
    rows = con.execute("SELECT run_id, model, metrics, composite, created_at FROM results "
                       "WHERE battery='G' ORDER BY created_at").fetchall()

    V2_KEYS = {"aggregate", "absent", "superseded"}
    changes, skipped = [], 0
    for run_id, model, metrics, old_comp, created in rows:
        try:
            rec = json.loads(metrics)
        except Exception:
            skipped += 1; continue
        depths = rec.get("depths") or {}
        graded = [e for e in depths.values() if isinstance(e, dict) and "hits" in e]
        if not graded or not V2_KEYS.issubset(set(graded[0]["hits"])):
            skipped += 1; continue            # v1 row — not re-gradable on a two-band bar
        old = rec.get("summary") or {}
        new = summarize(depths)
        rec["summary"] = new
        changes.append((run_id, model, created, old.get("clean_depth"), new.get("clean_depth"),
                        old.get("hard_threshold"), new.get("hard_threshold"), json.dumps(rec),
                        results_db.composite_of(rec)))

    moved = [c for c in changes if c[3] != c[4]]
    print(f"Battery G rescore — core ≥ {G_CORE_THRESHOLD}, hard ≥ {G_HARD_THRESHOLD}", flush=True)
    print(f"  {len(changes)} v2 row(s) re-derived · {skipped} skipped (v1 / unparseable) · "
          f"{len(moved)} clean_depth change(s)\n", flush=True)
    if moved:
        print(f"  {'model':<44}{'was':>9}{'now':>9}   run", flush=True)
        for run_id, model, _c, was, now, *_ in moved:
            print(f"  {model[:43]:<44}{str(was):>9}{str(now):>9}   {run_id}", flush=True)

    if dry_run:
        print("\n  --dry-run: nothing written.", flush=True)
        return changes

    with con:
        for run_id, model, _c, _w, _n, _ot, _nt, blob, comp in changes:
            con.execute("UPDATE results SET metrics=?, composite=? "
                        "WHERE run_id=? AND model=? AND battery='G'", (blob, comp, run_id, model))
    con.close()
    print(f"\n  → updated {len(changes)} DB row(s) in place (run_id/created_at preserved)", flush=True)

    # Rewrite the newest on-disk report so JSON/MD match the DB.
    latest_json = max(RESULTS_DIR.glob("longctx_*.json"),
                      key=lambda f: f.stat().st_mtime, default=None)
    if latest_json and not latest_json.stem.endswith("_fast"):
        data = json.loads(latest_json.read_text())
        for rec in data:
            d = rec.get("depths") or {}
            if d and V2_KEYS.issubset(set(next(iter(d.values())).get("hits", {}))):
                rec["summary"] = summarize(d)
        latest_json.write_text(json.dumps(data, indent=2))
        write_summary(data, latest_json.with_suffix(".md"))
        print(f"  → rewrote {latest_json.name} + .md", flush=True)
    return changes


# ── Capable-only gate (reuse the latest standard run's `calculate` result) ─────

def _capable_models():
    src = latest_result(RESULTS_DIR, "benchmark", False, 72)
    if not src:
        return None
    try:
        data = json.load(src.open())
        return {r["model"] for r in data
                if r.get("tests", {}).get("calculate", {}).get("correct")}
    except Exception:
        return None

# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TODAY  = date.today().isoformat()
    suffix = "_fast" if fast_mode else ""

    if _flag("--rescore"):                       # grading-policy change — no model calls, no cooldown
        rescore(dry_run=_flag("--dry-run"))
        sys.exit(0)

    if not DATASET.exists():
        sys.exit(f"dataset not found — run: python3 {DATASET.relative_to(REPO)}".replace("dataset.json", "build.py"))
    items = json.loads(DATASET.read_text())["items"]

    reg_path = REPO / "models.json"
    if not reg_path.exists():
        sys.exit("models.json not found — run update_registry.py first")
    registry = sort_registry(json.load(reg_path.open()))   # run order: env BENCH_SORT (default size)

    reg_map = {m["name"]: m for m in registry}
    # Eligible universe = the full completion fleet (independent of --models, so carry-forward stays
    # complete). --models is handled by resume as explicit targets below.
    MODELS = [(m["name"], m["disk_gb"], m["role"]) for m in registry
              if m.get("role") in COMPLETION_ROLES]
    if model_args:  # ensure forced models are present even if outside the default universe
        for m in model_args:
            if m not in {n for n, *_ in MODELS}:
                MODELS.append((m, reg_map.get(m, {}).get("disk_gb", 0.0), reg_map.get(m, {}).get("role", "worker")))

    if role_filter:
        MODELS = [(n, d, r) for n, d, r in MODELS if r == role_filter]

    if capable_only and not model_args:
        cap = _capable_models()
        if cap is not None:
            before = len(MODELS)
            MODELS = [(n, d, r) for n, d, r in MODELS if n in cap]
            print(f"  --capable-only: {before} → {len(MODELS)} models (passed `calculate`)", flush=True)

    if not MODELS:
        sys.exit("No models to test. Check models.json / --role / --capable-only.")

    preflight(MODELS, ollama_host)

    flag = " [FAST MODE]" if fast_mode else ""
    print(f"BenchLLAMA Battery G — long-context retrieval{flag} — {TODAY}", flush=True)
    print(f"ollama={ollama_host} | think=False | {len(MODELS)} model(s) | "
          f"buckets={[it['bucket'] for it in items]}", flush=True)

    OUT_JSON = RESULTS_DIR / f"longctx_{TODAY}{suffix}.json"
    OUT_MD   = RESULTS_DIR / f"longctx_{TODAY}{suffix}.md"
    print(f"Output: {OUT_JSON}\n", flush=True)

    # ── Content-addressed resume (docs/resume-spec.md): skip models unchanged since scored; carry
    #    their prior result forward from the DB (lossless). No time window. ──
    import resume
    eligible = [n for n, *_ in MODELS]
    cloud = {m["name"] for m in registry if m.get("cloud")}
    run_names, all_results, why = resume.plan_single_pass(
        "G", eligible, host=ollama_host, cloud=cloud, force=force,
        explicit_models=(model_args or None), check_runtime="--check-runtime" in sys.argv)
    all_results = list(all_results)
    completed = set(eligible) - set(run_names)
    print("  " + resume.format_report("G", run_names, sorted(completed), why).replace("\n", "\n  "), flush=True)

    first_run = True
    for model_name, disk_gb, role in MODELS:
        if model_name in completed:
            print(f"  ↷ {model_name} — already done, skipping", flush=True)
            continue
        if not first_run:
            _ws(model_name, "cooldown")
            cooldown(COOLDOWN, label="after previous model")
        first_run = False
        r = run_longctx(model_name, role, disk_gb, items)
        all_results.append(r)
        OUT_JSON.write_text(json.dumps(all_results, indent=2))
        try:
            import results_db; results_db.record_all("G", all_results, only=set(run_names))
        except Exception:
            pass

    _ws("", "done")
    write_summary(all_results, OUT_MD, fast_mode)
    print(f"\n{'='*60}\nDONE\nJSON → {OUT_JSON}\nMD   → {OUT_MD}", flush=True)
