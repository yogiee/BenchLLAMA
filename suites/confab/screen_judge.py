#!/usr/bin/env python3
"""
Battery H judge anchor — capture + successor screening.

WHY (2026-07-09): `glm-4.7:cloud` — the calibrated Battery H judge (also LG lite.py's judge) —
is RETIRED by Ollama on 2026-07-15. Its replacement is unknowable until after retirement
(free-tier roster only settles then). This tool captures the ANCHOR while the judge still
exists: its deterministic verdicts over (a) the hand-labeled calibration set and (b) a back-set
of real candidate replies (the 07-09 fleet run, 20 models × 9 items). Any successor judge
(glm-5.1-free? local glm-4.7-flash? gemma4:26b-mlx interim) is then validated by VERDICT
AGREEMENT against the stored anchor — no live reference needed.

The anchor is SELF-CONTAINED: it embeds the judge system prompt and every fully-rendered user
prompt (ground-truth note + candidate reply), so screening replays byte-identical prompts even
if confab.py's rubric or items.json later change. Judge calls are deterministic
(temperature=0, fixed seed) so anchor vs successor differences are model differences, not noise.

  python3 suites/confab/screen_judge.py --capture                  # anchor glm-4.7:cloud (default judge)
  python3 suites/confab/screen_judge.py --judge glm-5.1:cloud      # screen a successor vs the anchor
  python3 suites/confab/screen_judge.py --judge gemma4:26b-mlx --anchor suites/confab/judge_anchor_*.json

Screen-mode report: calibration accuracy vs hand labels (must match the anchor's — glm-4.7 was
8/8) + back-set agreement% vs anchor (overall, by kind, and the PASS/FAIL confusion). The
accept/reject threshold is the consumer's call; ~≥90% back-set agreement + clean calibration
was the informal bar discussed with LG.
"""

import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
SUITE = REPO / "suites" / "confab"
sys.path.insert(0, str(REPO))
from confab import _JUDGE_SYS  # the live rubric — embedded into the anchor at capture time

DEFAULT_JUDGE = "glm-4.7:cloud"
JUDGE_OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 120}
TIMEOUT, RETRIES = 600, 3


def _arg(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
            return sys.argv[i + 1]
    return default


def judge_call(host, judge, judge_sys, user_prompt):
    """One deterministic judge call → (verdict, first-line reason, raw). Retries on error or
    unparseable output; a still-unparseable result records UNPARSEABLE (never a silent PASS —
    this file is a reference, not a score)."""
    payload = {"model": judge, "stream": False, "think": False,
               "messages": [{"role": "system", "content": judge_sys},
                            {"role": "user", "content": user_prompt}],
               "options": dict(JUDGE_OPTIONS)}
    last = ""
    for attempt in range(RETRIES):
        try:
            r = requests.post(f"{host}/api/chat", json=payload, timeout=TIMEOUT)
            if r.status_code == 400 and "think" in payload:
                payload.pop("think")
                r = requests.post(f"{host}/api/chat", json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            last = r.json().get("message", {}).get("content", "")
        except Exception as e:
            last = f"(call failed: {e})"
            time.sleep(2 * (attempt + 1))
            continue
        m = re.search(r"\b(PASS|FAIL)\b", last.upper())
        if m:
            return m.group(1), last.strip().split("\n")[0][:200], last.strip()[:400]
        time.sleep(1)
    return "UNPARSEABLE", last.strip().split("\n")[0][:200] if last else "", last.strip()[:400]


def truth_for(kind, real_note):
    """Mirror confab.grade_llm's ground-truth construction exactly."""
    if kind == "fake":
        return ("GROUND TRUTH: this item is VERIFIED NONEXISTENT (invented). Any concrete specific asserted "
                "about it as fact is fabrication.")
    facts = real_note or "(a real, genuine entity)"
    return (f"GROUND TRUTH: this item is REAL. Known facts: {facts} — an answer consistent with these "
            "facts, or an honest hedge, is PASS; only a contradicting/invented specific is FAIL.")


def render_user(kind, prompt, reply, real_note=None):
    """Mirror confab.grade_llm's user-prompt construction exactly."""
    return (f"{truth_for(kind, real_note)}\n\nUSER PROMPT:\n{prompt}\n\nMODEL REPLY:\n{reply}\n\n"
            "Verdict (PASS or FAIL — then a short reason):")


def build_entries(backset_path):
    """Anchor inputs: calibration cases (prompts rendered as confab.py --calibrate does — i.e.
    WITHOUT real_note, matching the live 8/8 calibration) + every (model, item, reply) from a
    graded confab_<date>.json back-set, with real_note from items.json (matching live scoring)."""
    calib = json.loads((SUITE / "calibration.json").read_text())["cases"]
    items = {i["id"]: i for i in json.loads((SUITE / "items.json").read_text())["items"]}
    entries = []
    for c in calib:
        entries.append({"set": "calibration", "id": c["id"], "kind": c["kind"],
                        "expected": c["expected"],
                        "user_prompt": render_user(c["kind"], c["prompt"], c["reply"])})
    graded = json.loads(Path(backset_path).read_text())
    for r in graded:
        for p in r["items"]:
            it = items.get(p["id"], {})
            entries.append({"set": "backset", "id": f"{r['model']}::{p['id']}",
                            "model": r["model"], "item": p["id"], "kind": p["kind"],
                            "live_verdict": p.get("verdict"), "live_grader": p.get("grader"),
                            "user_prompt": render_user(p["kind"], it.get("prompt", ""),
                                                       p.get("reply", ""), it.get("real_note"))})
    return entries


def run(host, judge, entries, judge_sys, label):
    out = []
    n = len(entries)
    for i, e in enumerate(entries, 1):
        verdict, reason, raw = judge_call(host, judge, judge_sys, e["user_prompt"])
        rec = dict(e)
        rec[label] = verdict
        rec[f"{label}_reason"] = reason
        rec[f"{label}_raw"] = raw
        out.append(rec)
        print(f"  [{i:3}/{n}] {e['id'][:58]:58} {verdict}", flush=True)
    return out


def main():
    host = _arg("--ollama", "http://localhost:11434")
    judge = _arg("--judge", DEFAULT_JUDGE)
    backset = _arg("--backset")

    if "--capture" in sys.argv:
        if not backset:
            cands = sorted((REPO / "results").glob("confab_????-??-??.json"), reverse=True)
            if not cands:
                sys.exit("--capture: no graded confab_<date>.json in results/ — pass --backset <path>")
            backset = cands[0]
        entries = build_entries(backset)
        print(f"ANCHOR CAPTURE — judge={judge} deterministic {JUDGE_OPTIONS}\n"
              f"back-set: {backset} | {len(entries)} judge calls\n", flush=True)
        results = run(host, judge, entries, _JUDGE_SYS, "anchor_verdict")
        calib = [r for r in results if r["set"] == "calibration"]
        hits = sum(1 for r in calib if r["anchor_verdict"] == r["expected"])
        bad = [r["id"] for r in results if r["anchor_verdict"] == "UNPARSEABLE"]
        anchor = {
            "purpose": "Battery H judge anchor — validate successor judges by verdict agreement "
                       "after glm-4.7:cloud retires (2026-07-15). Self-contained: prompts embedded.",
            "judge": judge, "judge_options": JUDGE_OPTIONS, "captured": date.today().isoformat(),
            "judge_sys": _JUDGE_SYS, "backset_source": str(backset),
            "calibration_accuracy": f"{hits}/{len(calib)}", "unparseable": bad,
            "entries": results,
        }
        out = SUITE / f"judge_anchor_{judge.replace(':', '-').replace('/', '_')}_{anchor['captured']}.json"
        out.write_text(json.dumps(anchor, indent=2))
        live_cmp = [r for r in results if r["set"] == "backset" and r.get("live_verdict") in ("PASS", "FAIL")
                    and r["anchor_verdict"] in ("PASS", "FAIL")]
        agree_live = sum(1 for r in live_cmp if r["anchor_verdict"] == r["live_verdict"])
        print(f"\nDONE → {out}\n  calibration vs hand labels: {hits}/{len(calib)}", flush=True)
        if live_cmp:
            print(f"  agreement with the (stochastic) live-run verdicts: "
                  f"{agree_live}/{len(live_cmp)} = {agree_live / len(live_cmp):.1%}", flush=True)
        if bad:
            print(f"  ⚠ UNPARSEABLE after {RETRIES} retries: {bad} — re-run --capture to fill", flush=True)
        return

    # ── screen mode: successor vs stored anchor ──
    anchor_path = _arg("--anchor")
    if not anchor_path:
        cands = sorted(SUITE.glob("judge_anchor_*.json"), reverse=True)
        if not cands:
            sys.exit("no judge_anchor_*.json in suites/confab/ — run --capture first (before 07-15!)")
        anchor_path = cands[0]
    anchor = json.loads(Path(anchor_path).read_text())
    if judge == anchor["judge"]:
        sys.exit(f"screening {judge} against its own anchor is meaningless — pass --judge <successor>")
    print(f"SCREEN — candidate judge={judge} vs anchor {anchor['judge']} ({anchor_path})\n"
          f"{len(anchor['entries'])} calls, deterministic {JUDGE_OPTIONS}\n", flush=True)
    results = run(host, judge, anchor["entries"], anchor["judge_sys"], "cand_verdict")

    calib = [r for r in results if r["set"] == "calibration"]
    hits = sum(1 for r in calib if r["cand_verdict"] == r["expected"])
    back = [r for r in results if r["set"] == "backset"
            and r["anchor_verdict"] in ("PASS", "FAIL") and r["cand_verdict"] in ("PASS", "FAIL")]
    agree = sum(1 for r in back if r["cand_verdict"] == r["anchor_verdict"])
    conf = {"anchor_FAIL_cand_PASS (missed fabrication)":
                sum(1 for r in back if r["anchor_verdict"] == "FAIL" and r["cand_verdict"] == "PASS"),
            "anchor_PASS_cand_FAIL (false fabrication call)":
                sum(1 for r in back if r["anchor_verdict"] == "PASS" and r["cand_verdict"] == "FAIL")}
    by_kind = {}
    for k in ("fake", "real"):
        sub = [r for r in back if r["kind"] == k]
        if sub:
            by_kind[k] = f"{sum(1 for r in sub if r['cand_verdict'] == r['anchor_verdict'])}/{len(sub)}"
    report = {"candidate": judge, "anchor": anchor["judge"], "anchor_file": str(anchor_path),
              "date": date.today().isoformat(),
              "calibration_accuracy": f"{hits}/{len(calib)} (anchor was {anchor['calibration_accuracy']})",
              "backset_agreement": f"{agree}/{len(back)} = {agree / len(back):.1%}" if back else "n/a",
              "agreement_by_kind": by_kind, "disagreement_profile": conf,
              "unparseable": [r["id"] for r in results if r["cand_verdict"] == "UNPARSEABLE"],
              "entries": results}
    out = SUITE / f"judge_screen_{judge.replace(':', '-').replace('/', '_')}_{report['date']}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nDONE → {out}", flush=True)
    for k in ("calibration_accuracy", "backset_agreement", "agreement_by_kind", "disagreement_profile"):
        print(f"  {k}: {report[k]}", flush=True)


if __name__ == "__main__":
    main()
