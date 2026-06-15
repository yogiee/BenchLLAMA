#!/usr/bin/env python3
"""
BenchLLAMA — consumer manifest reader.

Reads consumer-authored manifests from ~/.config/ollama-consumers/ (everything except our own
benchllama-rankings.json) and turns them into producer-side intelligence:

  • PROTECTED SET — every model any consumer uses (assignments + models_in_use), annotated with
                    the CAPABILITY it serves so drop logic is per-capability (a model's low
                    coding rank is irrelevant if it's used for vision/OCR).
  • CURRENCY      — `source` (benchllama@<date> vs manual) vs current rankings → flags stale/
                    never-imported consumers.
  • GAP BACKLOG   — structured `gaps[]` (schema 2), or keyword-flagged `rationale` (schema 1) →
                    battery-refinement input, grouped by capability.

Consumer-AWARE intelligence; deliberately does NOT touch the neutral export. Reads both manifest
schema 1 (assignments = {role: model}) and schema 2 (assignments = {role: {model, capability,
basis?, tier?}}, structured gaps[]). Contract: ~/.config/ollama-consumers/README.md.

  python3 consumers.py
"""

import json
import glob
import os
from pathlib import Path

CONSUMER_DIR = Path.home() / ".config" / "ollama-consumers"
RANKINGS = CONSUMER_DIR / "benchllama-rankings.json"
_REJECT_HINTS = ("reject", "truncat", "unfit", "limit", "window", "fail", "can't", "cannot")
# capability → the ranking list it draws from (embedding resolves via the assignment's `basis`)
CAP_LIST = {"coding": "coders", "vision": "vision", "ocr": "vision_fast_ocr",
            "chat": "workers", "routing": "routers"}

# Operational roster — consumers we expect to publish a manifest. Used ONLY to detect a MISSING
# consumer (a model used only by an unpublished consumer is invisible → unsafe to drop). Editable;
# not part of the neutral export (reading manifests already exposes consumer names). Keep in sync
# as consumers are added/retired.
EXPECTED_CONSUMERS = {"ollama-local", "memoryCentral", "lookingGlass"}


def load_manifests():
    out = []
    for f in sorted(glob.glob(str(CONSUMER_DIR / "*.json"))):
        if os.path.basename(f) == RANKINGS.name:
            continue
        try:
            out.append(json.load(open(f)))
        except Exception as e:
            print(f"  ⚠ skipping unreadable manifest {os.path.basename(f)}: {e}")
    return out


def load_rankings():
    try:
        return json.load(RANKINGS.open())
    except Exception:
        return None


def _norm_assignment(val):
    """v1 (bare string) or v2 (object) → (model, capability, basis, tier)."""
    if isinstance(val, dict):
        return val.get("model"), val.get("capability", "?"), val.get("basis"), val.get("tier", "primary")
    return val, "?", None, "primary"


def protected_set(manifests):
    """model → list of {consumer, role, capability, basis, tier}."""
    used = {}
    for m in manifests:
        c = m.get("consumer", "?")
        listed = set()
        for role, val in (m.get("assignments") or {}).items():
            model, cap, basis, tier = _norm_assignment(val)
            if not model:
                continue
            used.setdefault(model, []).append(
                {"consumer": c, "role": role, "capability": cap, "basis": basis, "tier": tier})
            listed.add(model)
        for model in (m.get("models_in_use") or []):
            if model not in listed and not any(u["consumer"] == c for u in used.get(model, [])):
                used.setdefault(model, []).append(
                    {"consumer": c, "role": "in_use", "capability": "?", "basis": None, "tier": "?"})
    return used


def gap_backlog(manifests):
    out = []
    for m in manifests:
        c = m.get("consumer", "?")
        structured = m.get("gaps") or []
        for g in structured:                                  # schema 2
            out.append({"consumer": c, "capability": g.get("capability", "?"),
                        "issue": g.get("issue"), "observed_with": g.get("observed_with"),
                        "wanted": g.get("wanted"), "status": g.get("status", "open")})
        if not structured:                                    # schema 1 fallback
            for role, why in (m.get("rationale") or {}).items():
                if any(h in (why or "").lower() for h in _REJECT_HINTS):
                    out.append({"consumer": c, "capability": role, "issue": why,
                                "observed_with": None, "wanted": None, "status": "open"})
    return out


def coverage_verdict(manifests, rankings):
    """Is the drop-exempt set complete enough to ACT on drops? Returns (safe, lines).

    Hard blockers (usage unknown / unsettled → never drop):
      • a known consumer published NO manifest (model used only by it is invisible);
      • a manifest is source="manual" (never ingested rankings → selection not settled).
    Soft flag (exemptions still hold, but consider re-ingest):
      • a present manifest predates the current rankings.
    """
    cur = (rankings or {}).get("generated", "")[:10]
    present = {m.get("consumer") for m in manifests}
    missing = sorted(EXPECTED_CONSUMERS - present)
    manual = sorted(m.get("consumer", "?") for m in manifests if m.get("source") == "manual")
    stale = sorted(m.get("consumer", "?") for m in manifests
                   if isinstance(m.get("source"), str) and m["source"].startswith("benchllama@")
                   and m["source"].split("@")[1] < cur)
    lines = []
    if missing:
        lines.append(f"missing manifest (usage unknown): {', '.join(missing)}")
    if manual:
        lines.append(f"source=manual (selection not settled): {', '.join(manual)}")
    if stale:
        lines.append(f"predates current rankings (re-ingest suggested): {', '.join(stale)}")
    safe = not missing and not manual
    return safe, lines


def _ranks_by_list(rankings):
    """{list_name: {model: rank}} from rankings.* lists."""
    out = {}
    for name, lst in (rankings or {}).get("rankings", {}).items():
        out[name] = {n: i + 1 for i, n in enumerate(lst)}
    return out


def _benchmarked(rankings):
    return {m["name"] for m in (rankings or {}).get("models", [])}


def _rank_note(use, ranks, benched, model):
    """Per-capability rank annotation for one usage."""
    cap, basis = use["capability"], use["basis"]
    if cap == "manual":
        return "manual pin (not drop-evaluated)"
    if cap == "embedding":
        lst = basis if basis in ranks else "embedding_long"
    else:
        lst = CAP_LIST.get(cap)
    if lst and model in ranks.get(lst, {}):
        return f"{lst} #{ranks[lst][model]}"
    if model not in benched:
        return "not benchmarked"
    return f"capability={cap}"


def report():
    manifests = load_manifests()
    rankings = load_rankings()
    ranks = _ranks_by_list(rankings)
    benched = _benchmarked(rankings)
    cur = rankings.get("generated", "?") if rankings else "?"
    print(f"BenchLLAMA — consumer manifest report   (rankings: {cur})\n")

    print(f"Consumers: {len(manifests)}")
    for m in manifests:
        src = m.get("source", "?")
        flag = " ⚠ MANUAL — never imported rankings" if src == "manual" else (
            f" ⚠ STALE — predates current rankings" if (rankings and isinstance(src, str)
            and src.startswith("benchllama@") and src.split("@")[1] < cur[:10]) else "")
        print(f"  • {m.get('consumer','?'):<16} schema={m.get('schema','?')} "
              f"policy={m.get('selection_policy','?'):<20} source={src}{flag}")
    print()

    # Coverage verdict — can drop decisions act on this exemption set yet?
    safe, cov = coverage_verdict(manifests, rankings)
    if safe and not cov:
        print("DROP-SAFETY: ✓ coverage complete — exemptions are authoritative.\n")
    elif safe:
        print("DROP-SAFETY: △ exemptions authoritative, with notes:")
        for c in cov:
            print(f"    - {c}")
        print()
    else:
        print("DROP-SAFETY: ⚠ NOT SAFE FOR DROP DECISIONS — exemption set is INCOMPLETE:")
        for c in cov:
            print(f"    - {c}")
        print("    A model used only by a missing/unsettled consumer would be invisible here.\n")

    used = protected_set(manifests)
    print(f"DROP-EXEMPT SET — {len(used)} models in use (exempt from drop, any rank):")
    for model in sorted(used):
        for u in used[model]:
            note = _rank_note(u, ranks, benched, model)
            tier = "" if u["tier"] in ("primary", "?") else f" {u['tier']}"
            print(f"  {model:<26} ← {u['consumer']}:{u['role']}{tier}  [{note}]")
    print()

    gaps = gap_backlog(manifests)
    openg = [g for g in gaps if g["status"] != "resolved"]
    print(f"GAP BACKLOG — {len(gaps)} signal(s), {len(openg)} open (battery-refinement input):")
    for g in gaps:
        mark = "○" if g["status"] != "resolved" else "✓"
        ow = f" (observed_with {g['observed_with']})" if g.get("observed_with") else ""
        print(f"  {mark} [{g['capability']}] {g['consumer']}{ow}: {g['issue']}")
        if g.get("wanted"):
            print(f"      wanted: {g['wanted']}")
    if not gaps:
        print("  (none)")


if __name__ == "__main__":
    report()
