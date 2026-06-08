#!/usr/bin/env python3
"""
BenchLLAMA — Registry Updater

Queries the Ollama API for installed models and their capabilities,
then syncs models.json:
  - New completion models are added with role=null (edit before running)
  - Existing entries keep their role; disk_gb is refreshed from Ollama
  - Models no longer installed are flagged but kept (in case they're temporarily unloaded)
  - Non-completion models (image, embedding) are listed but never added

Usage:
  python3 update_registry.py
  python3 update_registry.py --ollama http://host:11434
  python3 update_registry.py --dry-run
"""

import json
import sys
import requests
from pathlib import Path

REPO     = Path(__file__).parent
REGISTRY = REPO / "models.json"

def _arg(name, default=None):
    if name in sys.argv:
        idx = sys.argv.index(name)
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--"):
            return sys.argv[idx + 1]
    return default

host    = _arg("--ollama", "http://localhost:11434")
dry_run = "--dry-run" in sys.argv


def fetch_installed(host):
    """Returns {name: {"disk_gb": float, "capabilities": set}}"""
    r = requests.get(f"{host}/api/tags", timeout=10)
    r.raise_for_status()
    result = {}
    for m in r.json().get("models", []):
        name    = m["name"]
        disk_gb = round(m.get("size", 0) / 1e9, 1)
        try:
            show = requests.post(f"{host}/api/show", json={"name": name}, timeout=10)
            caps = set(show.json().get("capabilities", []))
        except Exception:
            caps = set()
        result[name] = {"disk_gb": disk_gb, "capabilities": caps}
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

    completion = {n: i for n, i in installed.items() if "completion" in i["capabilities"]}
    skipped    = {n: i for n, i in installed.items() if "completion" not in i["capabilities"]}

    print(f"{len(installed)} models found  —  "
          f"{len(completion)} completion, {len(skipped)} skipped\n")

    if skipped:
        print("Skipped (no completion capability):")
        for name, info in skipped.items():
            caps = ", ".join(sorted(info["capabilities"])) or "—"
            print(f"  {name:<32}  {info['disk_gb']:>5} GB  [{caps}]")
        print()

    existing = load_registry()
    proposed = []
    added, updated, unchanged, missing = [], [], [], []

    for name, info in completion.items():
        if name in existing:
            entry = dict(existing[name])
            if entry.get("disk_gb") != info["disk_gb"]:
                entry["disk_gb"] = info["disk_gb"]
                updated.append(name)
            else:
                unchanged.append(name)
        else:
            entry = {"name": name, "disk_gb": info["disk_gb"], "role": None}
            added.append(name)
        proposed.append(entry)

    for name in existing:
        if name not in completion:
            missing.append(name)
            proposed.append(existing[name])  # keep as-is

    # ── Print summary ─────────────────────────────────────────────────────────

    if added:
        print("New models added (role=null — set before running):")
        for name in added:
            info = completion[name]
            caps = ", ".join(sorted(info["capabilities"]))
            print(f"  + {name:<32}  {info['disk_gb']:>5} GB  [{caps}]")
        print()

    if updated:
        print("Disk size refreshed:")
        for name in updated:
            print(f"  ~ {name:<32}  → {completion[name]['disk_gb']} GB")
        print()

    if missing:
        print("In registry but not installed (kept):")
        for name in missing:
            print(f"  ? {name}")
        print()

    if not added and not updated and not missing:
        print("Registry is already up to date.")

    # ── Write ─────────────────────────────────────────────────────────────────

    if dry_run:
        print("\n── dry run: proposed models.json ──────────────────────────")
        print(json.dumps(proposed, indent=2))
    else:
        REGISTRY.write_text(json.dumps(proposed, indent=2) + "\n")
        print(f"models.json written  ({len(proposed)} models)")
        if added:
            print(f"\n  {len(added)} new model(s) have role=null.")
            print("  Edit models.json and set each to 'worker' or 'router' before running.")
