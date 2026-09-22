#!/usr/bin/env python3
"""
think_probe.py — measure every think lever a model exposes and write its `think_profile` into
models.json. The producer half of the v3 think-aware protocol (docs/think-spec.md).

WHY: Ollama's `think` parameter is not a portable scale. Measured 2026-08-28 on the same question:
granite4.2 has a real "low" level (76 tokens, correct) while its full think burns 4096 tokens and never
answers; qwen3.5's "low"/"high"/true are byte-identical (1857 tokens); deepcoder always thinks
whatever you send; gpt-oss cannot stop. A blanket think=False made granite4.2:3b/8b and qwen3.5:4b
FAIL bat_ball — a protocol artifact. So the lever is chosen PER MODEL, from measurement:

  for each lever in absent/false/low/medium/high/true → 4 tiny items at a ROOMY budget (4096)
  → group levers into CLASSES by behaviour (Modelfile sampling + seed 42 — the batteries' own conditions)
  → per class: think-token demand, correctness, bounded? (never starved at 4096)
  → direct arm = the no-thinking class (or, always-on model, the cheapest bounded class)
  → think arm = operating point = cheapest BOUNDED thinking class with the best score

  python3 think_probe.py                       # --missing: thinking-capable models w/o a fresh profile
  python3 think_probe.py --all                 # re-probe every thinking-capable model
  python3 think_probe.py --models a:b c:d      # explicit
  python3 think_probe.py --models a:b --stale-only   # explicit set, but skip the fresh ones (orchestrator ride-along)
  python3 think_probe.py --dry-run             # report only, don't write models.json
  python3 think_probe.py --rederive            # rebuild every profile from its stored raw (no calls)
  python3 think_probe.py --starved             # re-probe models whose profile has a starved class

A profile is STALE when the model digest, the Ollama major.minor, the probe item-set hash, or the
model's DECLARED think levels (Ollama ≥ 0.34.3 `/api/show` → `thinking`) moved.
Models without the `thinking` capability get no profile (→ legacy think=False in every writer).

DECLARED LEVELS (2026-09-23): the declaration is a prior and an alarm, never a substitute for probing —
checked against the 0.33.x profiles it was wrong for about half the fleet (see bench_utils.normalize_declared).
So the probe (a) ALSO tries any declared level outside THINK_LEVERS (qwen3.8 declares "xhigh"), (b) stores
the declaration in the profile, (c) reports every declared-vs-measured disagreement, and (d) re-probes when
the declaration changes.

LEAKED THINKING: a model can think with no `thinking` channel at all — the reasoning arrives inside
`content` as raw `<think>…</think>` (lfm2.5:8b on 0.34.3 at think=false and "low"). Counting that as zero
think tokens classifies inline thinking as "off", and every battery at that lever would then grade raw
reasoning text as the answer. Such calls are marked `leaked`, graded on the text after `</think>`, kept
in their own class, and never chosen for an arm while a clean lever exists.
"""
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from bench_utils import (THINK_LEVERS, THINK_PROBE_NUM_PREDICT, lever_value, THINK_OMIT, fetch_declared)

PROBE_FILE = REPO / "suites" / "think" / "probe.json"
REGISTRY   = REPO / "models.json"
TIMEOUT    = 900
SLOW_ITEM_S = 30   # items slower than this print a progress line — a lever only reports once all 4 items land,
                   # and on the slowest local model (bonsai-27b, ~100 s/item) that is ~7 silent minutes per lever
PROBE_ESCALATION = (8192, 16384)   # num_predict retries for items starved at the base probe budget


def _flag(n): return n in sys.argv
def _arg(n, d=None):
    if n in sys.argv:
        i = sys.argv.index(n)
        if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
            return sys.argv[i + 1]
    return d


ollama_host = _arg("--ollama", "http://localhost:11434")
_VALUED = {"--ollama", "--arm"}
model_args = []
if "--models" in sys.argv:
    i = sys.argv.index("--models") + 1
    while i < len(sys.argv) and not sys.argv[i].startswith("--"):
        model_args.append(sys.argv[i]); i += 1


# ── item checks ───────────────────────────────────────────────────────────────

def _check_bat_ball(msg):
    c = (msg.get("content") or "").lower()
    return "0.05" in c or "5 cents" in c or "five cents" in c


def _check_one_word(msg, want):
    c = (msg.get("content") or "").strip().strip(".!").strip()
    return c.upper() == want.upper()


def _check_tool(msg, needles):
    for tc in msg.get("tool_calls") or []:
        args = json.dumps((tc.get("function") or {}).get("arguments") or {})
        if all(n in args for n in needles):
            return True
    return False


def _check_code(msg, fn):
    c = msg.get("content") or ""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", c, re.S)
    if not blocks:
        return False
    src = blocks[0]
    tests = [("A man, a plan, a canal: Panama", True), ("race a car", False), ("", True), ("No 'x' in Nixon", True)]
    prog = src + f"\nimport json\nprint(json.dumps([{fn}(s) == e for s, e in {tests!r}]))\n"
    try:
        out = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, timeout=10)
        return out.returncode == 0 and all(json.loads(out.stdout.strip().splitlines()[-1]))
    except Exception:
        return False


def check(item, msg) -> bool:
    spec = item["check"]
    if spec == "bat_ball":
        return _check_bat_ball(msg)
    if spec.startswith("one_word:"):
        return _check_one_word(msg, spec.split(":", 1)[1])
    if spec.startswith("tool_call:"):
        return _check_tool(msg, spec.split(":", 1)[1].split(","))
    if spec.startswith("code:"):
        return _check_code(msg, spec.split(":", 1)[1])
    return False


# ── one call ──────────────────────────────────────────────────────────────────

def call(model, item, lever, options):
    msgs = ([{"role": "system", "content": item["system"]}] if item.get("system") else []) + \
           [{"role": "user", "content": item["prompt"]}]
    payload = {"model": model, "messages": msgs, "stream": False, "options": dict(options)}
    if item.get("tools"):
        payload["tools"] = item["tools"]
    val = lever_value(lever)
    if val is not THINK_OMIT:
        payload["think"] = val
    t0 = time.time()
    r = requests.post(f"{ollama_host}/api/chat", json=payload, timeout=TIMEOUT)
    wall = time.time() - t0
    if r.status_code == 400:
        return {"unsupported": True, "error": r.text[:160], "wall": wall}
    r.raise_for_status()
    d = r.json(); msg = d.get("message") or {}
    thinking, content = msg.get("thinking") or "", msg.get("content") or ""
    thinking, content, leaked = split_leaked(thinking, content)
    ev = d.get("eval_count") or 0
    tc = thinking and (len(thinking) / max(1, len(thinking) + len(content)))
    return {
        "unsupported": False,
        "thinking": thinking, "content": content, "tool_calls": msg.get("tool_calls"),
        "done_reason": d.get("done_reason"), "eval_count": ev,
        "think_tokens_est": int(round(ev * tc)) if thinking else 0,
        "starved": d.get("done_reason") == "length" and not content.strip(),
        "correct": check(item, {**msg, "content": content}), "wall": round(wall, 1),
        "leaked": leaked,
    }


def split_leaked(thinking: str, content: str):
    """(thinking, content, leaked). With an empty thinking channel, reasoning that arrived inline as
    `<think>…</think>` is moved back out of `content`; an unterminated `<think>` is all reasoning (the
    reply never started — it reads as starved when the budget ran out)."""
    if thinking or not ("<think>" in content or "</think>" in content):
        return thinking, content, False
    pre, sep, post = content.partition("</think>")
    if sep:
        return pre.replace("<think>", "", 1).strip(), post.strip(), True
    return content.replace("<think>", "", 1).strip(), "", True


def unload(model):
    try:
        requests.post(f"{ollama_host}/api/chat", json={"model": model, "messages": [], "keep_alive": 0}, timeout=15)
    except Exception:
        pass


# ── profile derivation ────────────────────────────────────────────────────────

DERIVE_VERSION = 5   # bump when the class/operating-point derivation changes → profiles re-derive from `raw` on load
                     # v5 (2026-09-23): leaked-thinking classes + declared-level provenance/mismatches


def _lever_order(lv: str):
    """THINK_LEVERS first in their canonical order, then any declared-only level (xhigh, max…) by name."""
    return (THINK_LEVERS.index(lv), "") if lv in THINK_LEVERS else (len(THINK_LEVERS), lv)


def _sig(res: dict):
    """Behaviour signature of one lever: per-item correctness + starvation pattern (item-id order)."""
    ids = sorted(res)
    return (tuple(bool(res[i]["correct"]) for i in ids), tuple(bool(res[i]["starved"]) for i in ids),
            any(res[i].get("leaked") for i in ids))


def _same_demand(a: int, b: int) -> bool:
    """Two levers are the same class if both don't think, or both think within ±25% (≥32 tok) of each other.
    Byte-identity is NOT usable: MLX runtimes are not deterministic across levers even at temp 0 / seed 42."""
    if a == 0 or b == 0:
        return a == b
    return abs(a - b) <= max(0.25 * max(a, b), 32)


def derive(model, per_lever: dict, meta) -> dict:
    """per_lever = {lever: {item_id: call-dict}} → think_profile (deterministic; no model calls)."""
    ok = [lv for lv, res in per_lever.items() if res and not any(r.get("unsupported") for r in res.values())]
    info = {lv: (max(r["think_tokens_est"] for r in per_lever[lv].values()), _sig(per_lever[lv])) for lv in ok}
    groups: list[list[str]] = []
    for lv in sorted(ok, key=lambda x: (info[x][0], _lever_order(x))):
        tk, sig = info[lv]
        for g in groups:
            gtk, gsig = info[g[0]]
            if _same_demand(tk, gtk) and sig == gsig:
                g.append(lv)
                break
        else:
            groups.append([lv])

    def stats(levers):
        allres = [per_lever[lv] for lv in levers]
        rep = allres[0]
        return {
            "levers": levers,
            "think_tokens_max": max(r["think_tokens_est"] for res in allres for r in res.values()),
            "thinking_chars_max": max(len(r["thinking"]) for res in allres for r in res.values()),
            "starved": any(r["starved"] for res in allres for r in res.values()),
            "hit_cap": any(r["done_reason"] == "length" for res in allres for r in res.values()),
            "leaked": any(r.get("leaked") for res in allres for r in res.values()),
            "score": sum(1 for r in rep.values() if r["correct"]),
            "items": {iid: {"correct": r["correct"], "think_tokens": r["think_tokens_est"],
                            "done": r["done_reason"], "wall": r["wall"]} for iid, r in rep.items()},
            "wall_p50": round(sorted(r["wall"] for r in rep.values())[len(rep) // 2], 1),
        }

    classes_raw = [stats(g) for g in groups]
    off = [c for c in classes_raw if c["thinking_chars_max"] == 0]
    thinking = sorted((c for c in classes_raw if c["thinking_chars_max"] > 0), key=lambda c: c["think_tokens_max"])
    class_stats, lever_class = {}, {}
    if off:                                  # every zero-thinking group is the one "off" class
        merged = dict(off[0]); merged["levers"] = sum((c["levers"] for c in off), [])
        merged["score"] = max(c["score"] for c in off)
        class_stats["off"] = merged
        for lv in merged["levers"]:
            lever_class[lv] = "off"
    for n, c in enumerate(thinking, 1):
        cid = f"think{n}"
        c["bounded"] = not c["starved"]
        class_stats[cid] = c
        for lv in c["levers"]:
            lever_class[lv] = cid
    for lv in per_lever:
        if lv not in lever_class:
            lever_class[lv] = "unsupported"

    bounded = [cid for cid, c in class_stats.items() if cid != "off" and c.get("bounded")]
    # A leaked class answers only once `</think>` is stripped — no battery does that, so it is never an arm
    # while a clean bounded class exists (it stays in class_stats and is reported).
    if any(not class_stats[c].get("leaked") for c in bounded):
        bounded = [c for c in bounded if not class_stats[c].get("leaked")]
    op = None
    if bounded:
        best = max(class_stats[c]["score"] for c in bounded)
        cands = [c for c in bounded if class_stats[c]["score"] == best]
        cheapest = min(cands, key=lambda c: class_stats[c]["think_tokens_max"])
        # Demand measured on four trivial items is noise-dominated: qwen3.8's "high" (Qwen's xhigh, the
        # mode that over-thinks hard tasks) came out 142 vs 223 tokens for low/medium and would have been
        # chosen as "cheapest". Within a 2× demand band of the cheapest class, prefer the class holding the
        # semantically LOWEST explicit lever instead; only a >2× gap is a real cost difference.
        band = [c for c in cands
                if class_stats[c]["think_tokens_max"] <= 2 * max(1, class_stats[cheapest]["think_tokens_max"])]
        rank = {lv: i for i, lv in enumerate(("low", "medium", "true", "high", "absent"))}
        op = min(band, key=lambda c: (min(rank.get(lv, 99) for lv in class_stats[c]["levers"]),
                                      class_stats[c]["think_tokens_max"]))

    def pick_lever(cid, prefer):
        levers = class_stats[cid]["levers"]
        for p in prefer:
            if p in levers:
                return p
        return levers[0]

    # explicit levers beat `absent` (a consumer should send a value, not rely on the template default)
    off_supported = "off" in class_stats
    if off_supported:
        direct_class, direct_lever = "off", pick_lever("off", ("false", "absent", "low"))
        operating_point = op
        operating_lever = pick_lever(op, ("low", "medium", "true", "high", "absent")) if op else None
    else:                                    # always-on thinker: single arm at the cheapest bounded class
        cands = bounded or [cid for cid in class_stats if cid != "off"]
        direct_class = min(cands, key=lambda c: class_stats[c]["think_tokens_max"]) if cands else None
        if direct_class:
            all_same = len(class_stats) == 1          # every lever behaves identically (deepseek/deepcoder) → `false` is honest
            direct_lever = pick_lever(direct_class, ("false", "low") if all_same else ("low", "medium", "false", "true", "high", "absent"))
        else:
            direct_lever = None
        operating_point, operating_lever = None, None

    raw = {lv: {iid: {"tt": r.get("think_tokens_est", 0), "tc": len(r.get("thinking") or ""),
                      "ok": bool(r.get("correct")), "st": bool(r.get("starved")),
                      "done": r.get("done_reason"), "wall": r.get("wall"),
                      **({"lk": True} if r.get("leaked") else {})}
                for iid, r in res.items()} if res and not any(r.get("unsupported") for r in res.values())
           else {"unsupported": True}
           for lv, res in per_lever.items()}

    default_thinks = lever_class.get("absent") not in ("off", "unsupported", None)
    declared = meta.get("declared")
    return {
        "probed_at": meta.get("probed_at") or time.strftime("%Y-%m-%d"),
        "ollama": meta.get("ollama"), "digest": meta.get("digest"), "probe_sha": meta.get("probe_sha"),
        "derive_version": DERIVE_VERSION,
        "declared": declared,          # /api/show `thinking` at probe time (normalized); None = undeclared
        "levers_probed": sorted(per_lever, key=_lever_order),
        "leaked_levers": sorted((lv for c in class_stats.values() if c.get("leaked") for lv in c["levers"]),
                                key=_lever_order),
        "declared_mismatch": declared_mismatch(declared, lever_class, class_stats, default_thinks, off_supported,
                                               meta.get("ollama")),
        "off_supported": off_supported,
        "always_on": not off_supported,
        "default_thinks": default_thinks,
        "think_unbounded": bool(thinking) and not bounded,
        "classes": lever_class,
        "class_stats": class_stats,
        "direct_class": direct_class, "direct_lever": direct_lever,
        "operating_point": operating_point, "operating_lever": operating_lever,
        "raw": raw,      # compact per-lever measurements → `--rederive` recomputes everything above offline
    }


DECLARED_SINCE = (0, 34, 3)   # first Ollama whose /api/show carries `thinking`


def _ver(v) -> tuple:
    try:
        return tuple(int(x) for x in str(v).split("-")[0].split(".")[:3])
    except ValueError:
        return ()


def declared_mismatch(declared, lever_class, class_stats, default_thinks, off_supported, ollama=None) -> list[str]:
    """Every way the /api/show declaration disagrees with the measurement — informational; the measured
    profile is authoritative. [] = the declaration matches what was measured."""
    if declared is None:
        if _ver(ollama) and _ver(ollama) < DECLARED_SINCE:
            return []                  # the server could not declare anything — absence is not a finding
        return ["no think levels declared by /api/show"]
    vals, out = set(declared.get("values") or []), []
    off_levers = (class_stats.get("off") or {}).get("levers") or []
    if "false" in vals and not off_supported:
        out.append("declares `false`, but no lever turned thinking off")
    if "false" not in vals and off_supported:
        out.append(f"`false` not declared, yet {','.join(off_levers)} turn(s) thinking off")
    d = declared.get("default")
    if d is not None and (d != "false") != default_thinks:
        out.append(f"declared default `{d}`, but measured default {'thinks' if default_thinks else 'does not think'}")
    for cid, c in class_stats.items():
        if cid != "off" and not vals.intersection(c["levers"]) and set(c["levers"]) != {"absent"}:
            out.append(f"{cid} ({','.join(c['levers'])}) behaves distinctly but only via undeclared levers")
    for lv in sorted(vals, key=_lever_order):
        if lever_class.get(lv) == "unsupported":
            out.append(f"declares `{lv}`, but the server rejected it")
    return out


def rederive(prof: dict) -> dict | None:
    """Rebuild a profile from its stored `raw` (no model calls). None if the profile has no raw."""
    raw = prof.get("raw")
    if not raw:
        return None
    per_lever = {}
    for lv, res in raw.items():
        if res.get("unsupported"):
            per_lever[lv] = {"_": {"unsupported": True}}
            continue
        per_lever[lv] = {iid: {"think_tokens_est": r["tt"], "thinking": "x" * r["tc"], "correct": r["ok"],
                               "starved": r["st"], "done_reason": r["done"], "wall": r["wall"],
                               "leaked": r.get("lk", False)}
                         for iid, r in res.items()}
    meta = {k: prof.get(k) for k in ("probed_at", "ollama", "digest", "probe_sha", "declared")}
    return derive(None, per_lever, meta)


# ── registry I/O ──────────────────────────────────────────────────────────────

def probe_sha() -> str:
    return hashlib.sha256(PROBE_FILE.read_bytes()).hexdigest()[:12]


def ollama_meta():
    try:
        ver = requests.get(f"{ollama_host}/api/version", timeout=10).json().get("version")
    except Exception:
        ver = None
    digests = {}
    try:
        for m in requests.get(f"{ollama_host}/api/tags", timeout=10).json().get("models", []):
            digests[m.get("name") or m.get("model")] = (m.get("digest") or "")[:16]
    except Exception:
        pass
    return ver, digests


_NOT_CHECKED = object()


def is_stale(prof, digest, ver, psha, declared=_NOT_CHECKED) -> str | None:
    if not prof:
        return "no profile"
    if prof.get("derive_version") != DERIVE_VERSION and not prof.get("raw"):
        return "derivation changed (no raw to re-derive from)"
    if prof.get("probe_sha") != psha:
        return "probe set changed"
    if digest and prof.get("digest") != digest:
        return "weights changed"
    a, b = str(prof.get("ollama") or ""), str(ver or "")
    if a.split(".")[:2] != b.split(".")[:2]:
        return f"ollama {a}→{b}"
    # Checked AFTER the version gate: a server older than 0.34.3 declares nothing, so on a downgrade the
    # version reason is the true one. A profile predating the field reads as None = "undeclared".
    if declared is not _NOT_CHECKED and prof.get("declared") != declared:
        return "declared think levels changed"
    return None


def fmt_report(model, prof) -> str:
    L = [f"  {model}: off_supported={prof['off_supported']} default_thinks={prof['default_thinks']} "
         f"direct={prof['direct_class']}({prof['direct_lever']}) think={prof['operating_point']}({prof['operating_lever']})"
         + ("  ⚠ think UNBOUNDED" if prof["think_unbounded"] else "")]
    dec = prof.get("declared")
    L.append(f"    declared: " + (f"levels={','.join(dec.get('values') or [])} default={dec.get('default')}"
                                   if dec else "none"))
    for cid, c in prof["class_stats"].items():
        items = " ".join(f"{k}:{'✓' if v['correct'] else '✗'}" for k, v in c["items"].items())
        L.append(f"    {cid:7s} levers={','.join(c['levers']):28s} think_tok≤{c['think_tokens_max']:5d} "
                 f"score={c['score']}/{len(c['items'])} {'bounded' if not c['starved'] else 'STARVED'} "
                 f"wall_p50={c['wall_p50']}s  [{items}]" + ("  ⚠ leaked" if c.get("leaked") else ""))
    unsup = [lv for lv, cid in prof["classes"].items() if cid == "unsupported"]
    if unsup:
        L.append(f"    unsupported levers: {', '.join(unsup)}")
    if prof.get("leaked_levers"):
        L.append(f"    ⚠ LEAKED: thinking arrives inside content at {','.join(prof['leaked_levers'])} "
                 f"— never used as an arm while a clean lever exists; batteries would grade raw <think> text")
    for mm in prof.get("declared_mismatch") or []:
        L.append(f"    ≠ declared vs measured: {mm}")
    return "\n".join(L)


if __name__ == "__main__":
    dry = _flag("--dry-run")
    spec = json.loads(PROBE_FILE.read_text())
    items, options = spec["items"], dict(spec.get("options") or {})
    options.setdefault("num_predict", THINK_PROBE_NUM_PREDICT)
    registry = json.loads(REGISTRY.read_text())
    ver, digests = ollama_meta()
    psha = probe_sha()

    # Offline re-derivation: a derivation change (DERIVE_VERSION) or --rederive rebuilds every stored
    # profile from its `raw` block — no model calls, no re-probe (the think-spec's --rescore analogue).
    redone = []
    for m in registry:
        prof = m.get("think_profile")
        if prof and prof.get("raw") and (_flag("--rederive") or prof.get("derive_version") != DERIVE_VERSION):
            new = rederive(prof)
            if new:
                m["think_profile"] = new
                redone.append(m["name"])
    if redone:
        print(f"re-derived {len(redone)} profile(s) offline (derive v{DERIVE_VERSION}): {redone}")
        if not dry:
            REGISTRY.write_text(json.dumps(registry, indent=2) + "\n")
        for n in redone:
            print(fmt_report(n, next(m["think_profile"] for m in registry if m["name"] == n)))
    if _flag("--rederive"):
        sys.exit(0)

    thinking_models = [m for m in registry if "thinking" in (m.get("capabilities") or [])
                       and m.get("role") in ("worker", "router")]
    # Declared levels are read LIVE (not from models.json, which is only as fresh as the last registry
    # sync): they feed both the staleness check and the per-model lever set.
    _declared: dict = {}
    def declared_of(name):
        if name not in _declared:
            _declared[name] = fetch_declared(ollama_host, name)
        return _declared[name]
    def stale(m):
        return is_stale(m.get("think_profile"), digests.get(m["name"]), ver, psha, declared_of(m["name"]))
    if model_args:
        by = {m["name"]: m for m in registry}
        targets = []
        for n in model_args:
            entry = by.get(n)
            if entry is None:                       # not in models.json yet → ask Ollama directly
                try:
                    caps = requests.post(f"{ollama_host}/api/show", json={"model": n}, timeout=30).json().get("capabilities") or []
                except Exception:
                    caps = []
                entry = {"name": n, "capabilities": caps, "_unregistered": True}
                print(f"  ⚠ {n}: not in models.json — probing, but the profile can only be persisted after "
                      f"`python3 update_registry.py` adds it (report only this time)")
            if "thinking" not in (entry.get("capabilities") or []):
                print(f"  ↷ {n}: no `thinking` capability — nothing to probe (legacy think=False applies)")
                continue
            targets.append(entry)
        if _flag("--stale-only"):
            # the orchestrator's automatic probe phase: scope to the run's --models selection, but a fresh
            # profile is still a no-op — only an explicit `probe --models …` re-probes a fresh model
            fresh = [m for m in targets if not stale(m)]
            for m in fresh:
                print(f"  ↷ {m['name']}  profile fresh — skipped")
            targets = [m for m in targets if m not in fresh]
    elif _flag("--all"):
        targets = thinking_models
    elif _flag("--starved"):      # re-probe models whose profile has a class that starved (pre-escalation probes)
        targets = [m for m in thinking_models
                   if any(c.get("starved") for c in ((m.get("think_profile") or {}).get("class_stats") or {}).values())]
    else:
        targets = [m for m in thinking_models if stale(m)]
    no_cap = [m["name"] for m in registry if m.get("role") in ("worker", "router")
              and "thinking" not in (m.get("capabilities") or []) and m.get("think_profile")]

    print(f"think_probe — ollama={ver} probe_sha={psha} | {len(targets)} model(s) to probe"
          + (" [dry-run]" if dry else ""), flush=True)
    if not targets:
        print("  nothing to do (" + ("every selected model has a fresh profile or no thinking capability"
                                     if model_args else "every thinking-capable model has a fresh profile")
              + "; --all to re-probe)")
        sys.exit(0)

    profiles = {}
    for m in targets:
        name = m["name"]
        why = stale(m) or "requested"
        declared = declared_of(name)
        extra = [lv for lv in (declared or {}).get("values") or [] if lv not in THINK_LEVERS]
        print(f"\n▶ {name}  ({why})" + (f"  + declared-only level(s): {','.join(extra)}" if extra else ""), flush=True)
        per_lever = {}
        for lever in (*THINK_LEVERS, *extra):
            res = {}
            for it in items:
                try:
                    r = call(name, it, lever, options)
                except Exception as e:
                    r = {"unsupported": True, "error": str(e)[:160], "wall": 0}
                res[it["id"]] = r
                if r.get("unsupported"):
                    break
                if r.get("wall", 0) >= SLOW_ITEM_S:
                    print(f"    {lever:7s} · {it['id']} {r['wall']:.0f}s think_tok≈{r.get('think_tokens_est', 0)}", flush=True)
            per_lever[lever] = res
            if any(r.get("unsupported") for r in res.values()):
                print(f"    {lever:7s} unsupported ({list(res.values())[-1].get('error','')[:60]})", flush=True)
                continue
            # Escalate STARVED items: the 4096 probe cap sits below Qwen's (32k) and IBM's (8192) own
            # budgets, and reasoning length is long-tailed on MLX (qwen3.5:4b answered bat_ball at 1857
            # tokens one run and starved at 4096 the next). "Unbounded" must mean unbounded at 16384.
            for budget in PROBE_ESCALATION:
                starved = [it for it in items if res.get(it["id"], {}).get("starved")]
                if not starved:
                    break
                esc = {**options, "num_predict": budget, "num_ctx": max(int(options.get("num_ctx", 16384)), budget + 4096)}
                for it in starved:
                    try:
                        r = call(name, it, lever, esc)
                        r["budget"] = budget
                    except Exception as e:
                        r = {**res[it["id"]], "escalation_error": str(e)[:120]}
                    res[it["id"]] = r
                print(f"    {lever:7s} ↑ re-ran {len(starved)} starved item(s) at num_predict={budget}: "
                      + " ".join(f"{it['id']}:{'✓' if res[it['id']].get('correct') else '✗'}{'!' if res[it['id']].get('starved') else ''}" for it in starved), flush=True)
            marks = " ".join(f"{k}:{'✓' if r['correct'] else '✗'}{'!' if r['starved'] else ''}" for k, r in res.items())
            print(f"    {lever:7s} think_tok≤{max(r['think_tokens_est'] for r in res.values()):5d}  {marks}"
                  + ("  ⚠ leaked <think> into content" if any(r.get("leaked") for r in res.values()) else ""), flush=True)
        unload(name)
        prof = derive(name, per_lever, {"ollama": ver, "digest": digests.get(name), "probe_sha": psha,
                                        "declared": declared})
        if not prof["class_stats"]:
            print(f"  ✗ {name}: every lever was rejected or errored — no profile written", flush=True)
            continue
        profiles[name] = prof
        print(fmt_report(name, prof), flush=True)
        # Persist per model, not once at the end: a Stop mid-fleet used to discard every profile already
        # measured (2026-09-22: three aborted runs re-probed the same models from scratch each time).
        if not dry and not m.get("_unregistered"):
            m["think_profile"] = prof             # m IS the registry entry (targets are drawn from it)
            REGISTRY.write_text(json.dumps(registry, indent=2) + "\n")

    if dry:
        print("\n[dry-run] models.json not written")
        sys.exit(0)
    print(f"\n✓ wrote think_profile for {len(profiles)} model(s) → {REGISTRY}")
