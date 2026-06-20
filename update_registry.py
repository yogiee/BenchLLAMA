#!/usr/bin/env python3
"""
BenchLLAMA — Registry Updater

Queries the Ollama API for installed models and their capabilities,
then syncs models.json. Every entry now carries a `capabilities` array so the
capability-targeted batteries (vision, embedding) can select models directly.

Roles are pipeline LANES, not batteries:
  - worker / router → general chat models, graded by the standard suite + A–D
  - utility         → capability-routed specialists (vision, embedding, …); never
                      fed the chat suite. Which battery they run is decided by
                      their `capabilities` array (vision → Battery V, embedding →
                      Battery EMB), exactly as a worker's caps decide B/C/D.

Lanes (decided from capabilities + architecture):
  - completion → general chat models    → role=worker (gate may promote to router)
  - vision     → vision/OCR specialists → role=utility (run by Battery V — `vision` cap)
  - embedding  → embedding models        → role=utility (run by Battery EMB — `embedding` cap)
  - skipped    → pure image-gen / unknown → never added (nothing text-gradeable)

Sync rules:
  - New models are added with the default role for their lane.
  - Existing entries keep their role (respects manual edits + router promotions);
    disk_gb and capabilities are always refreshed from Ollama.
  - Models no longer installed are PRUNED by default. `/api/tags` (what `ollama list`
    reads) returns every installed model regardless of load state, so an absent entry
    has genuinely been `ollama rm`'d — not merely unloaded. Pass --keep-missing to
    retain absent entries instead (e.g. when syncing against a different/remote Ollama
    host that doesn't hold your full local set).

Note: a "specialist" lane model (e.g. glm-ocr) reports `completion` but is excluded
from the chat pipeline by architecture — it still belongs in the vision battery, so
it is admitted here rather than dropped.

Usage:
  python3 update_registry.py
  python3 update_registry.py --ollama http://host:11434
  python3 update_registry.py --keep-missing   # retain entries not installed locally
  python3 update_registry.py --dry-run
"""

import json
import sys
import requests
from pathlib import Path

REPO     = Path(__file__).parent
REGISTRY = REPO / "models.json"

# Architectures that report "completion" but are not general-purpose chat models.
# glmocr: OCR-specialist (glm-ocr); fails all chat benchmarks despite the flag,
# but IS a legitimate vision battery contender — admitted to the 'vision' lane.
EXCLUDED_ARCHITECTURES = {"glmocr"}

# Default role per lane. completion → worker (router promotion happens via the gate);
# every specialist lane → the single 'utility' role (capability picks its battery).
LANE_DEFAULT_ROLE = {"completion": "worker", "vision": "utility", "embedding": "utility"}


def _arg(name, default=None):
    if name in sys.argv:
        idx = sys.argv.index(name)
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--"):
            return sys.argv[idx + 1]
    return default

host         = _arg("--ollama", "http://localhost:11434")
dry_run      = "--dry-run" in sys.argv
keep_missing = "--keep-missing" in sys.argv


def classify(caps, arch):
    """Assign an installed model to a benchmark lane from its capabilities.

    Returns the lane name ('completion' | 'vision' | 'embedding') or None to skip.
    Order matters: a completion model on an excluded architecture is an OCR/vision
    specialist, not a chat model, so it lands in the vision lane.
    """
    caps = set(caps)
    if "completion" in caps and arch not in EXCLUDED_ARCHITECTURES:
        return "completion"
    if "completion" in caps and arch in EXCLUDED_ARCHITECTURES:
        return "vision"
    if "embedding" in caps:
        return "embedding"
    if "vision" in caps:
        return "vision"
    return None  # pure image-gen (flux/z-image) or unknown — nothing to grade


def fetch_installed(host):
    """Returns {name: {"disk_gb": float, "capabilities": set, "arch": str}}"""
    r = requests.get(f"{host}/api/tags", timeout=10)
    r.raise_for_status()
    result = {}
    for m in r.json().get("models", []):
        name    = m["name"]
        disk_gb = round(m.get("size", 0) / 1e9, 1)
        try:
            show     = requests.post(f"{host}/api/show", json={"name": name}, timeout=10)
            body     = show.json()
            caps     = set(body.get("capabilities", []))
            arch     = body.get("modelinfo", {}).get("general.architecture", "") or \
                       body.get("details", {}).get("family", "")
        except Exception:
            caps, arch = set(), ""
        result[name] = {"disk_gb": disk_gb, "capabilities": caps, "arch": arch}
    return result


def load_registry():
    if not REGISTRY.exists():
        return {}
    return {e["name"]: e for e in json.load(REGISTRY.open())}


if __name__ == "__main__":
    print(f"Querying {host} ...", flush=True)
    try:
        installed = fetch_installed(host)
    except Exception as e:
        sys.exit(f"Could not reach Ollama at {host}: {e}")

    # ── Classify every installed model into a lane ─────────────────────────────
    laned   = {}   # name -> (lane, info)
    skipped = {}   # name -> info  (pure image-gen / unknown)
    for name, info in installed.items():
        lane = classify(info["capabilities"], info["arch"])
        if lane is None:
            skipped[name] = info
        else:
            laned[name] = (lane, info)

    by_lane = {"completion": 0, "vision": 0, "embedding": 0}
    for _, (lane, _) in laned.items():
        by_lane[lane] += 1

    print(f"{len(installed)} models found  —  "
          f"{by_lane['completion']} completion, "
          f"{by_lane['vision']} vision, "
          f"{by_lane['embedding']} embedding, "
          f"{len(skipped)} skipped\n")

    if skipped:
        print("Skipped (no text/vision/embedding capability — image-gen or unknown):")
        for name, info in skipped.items():
            caps = ", ".join(sorted(info["capabilities"])) or "—"
            print(f"  {name:<32}  {info['disk_gb']:>5} GB  [{caps}]")
        print()

    existing = load_registry()
    proposed = []
    added, updated, unchanged, missing, pruned = [], [], [], [], []

    for name, (lane, info) in laned.items():
        caps_list = sorted(info["capabilities"])
        if name in existing:
            entry = dict(existing[name])
            changed = False
            if entry.get("disk_gb") != info["disk_gb"]:
                entry["disk_gb"] = info["disk_gb"]
                changed = True
            if entry.get("capabilities") != caps_list:
                entry["capabilities"] = caps_list
                changed = True
            # Role = lane assignment. Completion models keep their gated role
            # (preserves manual edits + router promotions). Specialists are always
            # 'utility' — migrate any deprecated role=vision/embedding here too.
            if lane == "completion":
                entry.setdefault("role", "worker")
                entry.setdefault("extended_roles", [])   # earned by Battery E (coder)
            elif entry.get("role") != "utility":
                entry["role"] = "utility"
                changed = True
            (updated if changed else unchanged).append(name)
        else:
            entry = {
                "name":         name,
                "disk_gb":      info["disk_gb"],
                "role":         LANE_DEFAULT_ROLE[lane],
                "capabilities": caps_list,
            }
            if lane == "completion":
                entry["extended_roles"] = []   # earned by Battery E (coder)
            added.append((name, lane))
        proposed.append(entry)

    for name, entry in existing.items():
        if name not in laned:
            if keep_missing:
                missing.append(name)
                proposed.append(entry)  # retained (--keep-missing)
            else:
                pruned.append(name)     # no longer installed → dropped

    # ── Print summary ──────────────────────────────────────────────────────────

    if added:
        print("New models added:")
        for name, lane in added:
            info = laned[name][1]
            caps = ", ".join(sorted(info["capabilities"]))
            role = LANE_DEFAULT_ROLE[lane]
            print(f"  + {name:<32}  {info['disk_gb']:>5} GB  role={role:<9} [{caps}]")
        print()

    if updated:
        print("Refreshed (disk size or capabilities):")
        for name in updated:
            print(f"  ~ {name}")
        print()

    if missing:
        print("In registry but not installed (kept — --keep-missing):")
        for name in missing:
            print(f"  ? {name}")
        print()

    if pruned:
        print("Pruned (no longer installed — removed from registry):")
        for name in pruned:
            print(f"  - {name}")
        print("  (pass --keep-missing to retain absent entries instead)")
        print()

    if not added and not updated and not missing and not pruned:
        print("Registry is already up to date.")

    # ── Write ──────────────────────────────────────────────────────────────────

    if dry_run:
        print("\n── dry run: proposed models.json ──────────────────────────")
        print(json.dumps(proposed, indent=2))
    else:
        REGISTRY.write_text(json.dumps(proposed, indent=2) + "\n")
        print(f"models.json written  ({len(proposed)} models)")
        new_completion = [n for n, lane in added if lane == "completion"]
        if new_completion:
            print(f"\n  {len(new_completion)} new completion model(s) added as role=worker.")
            print("  Run the standard suite — router promotion happens automatically if the gate passes.")
        new_specialist = [n for n, lane in added if lane in ("vision", "embedding")]
        if new_specialist:
            print(f"  {len(new_specialist)} new specialist(s) added — run './bench.sh vision' / './bench.sh embedding'.")
