#!/usr/bin/env python3
"""
BenchLLAMA — Upgrade sts.json to the real STS-B dev slice (standard benchmark).

Downloads the sentence-transformers STS benchmark TSV and writes the `dev` split
(human-scored 0–5, normalised to 0–1) to sts.json, overwriting the curated seed.
This is the "standard slice" half of the hybrid dataset strategy.

  python3 suites/embedding/fetch.py            # default: up to 200 dev pairs
  python3 suites/embedding/fetch.py --n 500    # more pairs
  python3 suites/embedding/fetch.py --split test

Falls back with a clear message (and leaves the seed in place) if offline.
"""

import gzip
import io
import json
import sys
import requests
from pathlib import Path

HERE = Path(__file__).parent
URL  = "https://sbert.net/datasets/stsbenchmark.tsv.gz"


def _arg(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def main():
    n     = int(_arg("--n", "200"))
    split = _arg("--split", "dev")
    print(f"Downloading STS-B from {URL} ...", flush=True)
    try:
        raw = requests.get(URL, timeout=30)
        raw.raise_for_status()
        text = gzip.decompress(raw.content).decode("utf-8")
    except Exception as e:
        sys.exit(f"Fetch failed ({e}). Seed sts.json left untouched — run build_seed.py "
                 f"if you need to regenerate it.")

    # columns: split, genre, dataset, year, sid, score, sentence1, sentence2
    pairs = []
    for line in text.splitlines():
        cols = line.split("\t")
        if len(cols) < 8 or cols[0] != split:
            continue
        try:
            score = float(cols[5])
        except ValueError:
            continue  # skips the header row and any malformed line
        if not (0.0 <= score <= 5.0):
            continue
        pairs.append({"a": cols[6], "b": cols[7], "score": round(score / 5.0, 4)})
        if len(pairs) >= n:
            break

    if not pairs:
        sys.exit(f"No '{split}' pairs parsed — STS-B format may have changed. Seed left in place.")

    out = {"source": f"STS-B {split} (sentence-transformers, n={len(pairs)})", "pairs": pairs}
    (HERE / "sts.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(pairs)} STS-B {split} pairs → {HERE/'sts.json'}")


if __name__ == "__main__":
    main()
