# Image-Gen Output (Battery I)

Generated images from **Battery I** (`imagegen.py`) land here, organised as:

```
imagegen_images/<YYYY-MM-DD>/<model-slug>/<Pn>_<seed>.png
```

- **This folder is tracked** (via `.gitkeep`) so it exists on a fresh clone and the web dashboard's
  **Image Review** tab has a stable target to serve from (`/api/imagegen/img/…`).
- **The PNGs are gitignored** — generated artifacts, not source. Only this README + `.gitkeep` are committed.

The paired scores live in `results/imagegen_<date>.json` / `.md` (also gitignored); human review grades
land in `results/imagegen_<date>_human.json` (sidecar, written by the Image Review tab).

Reference-only — image models are characterised on performance + prompt-adherence, never ranked.
See `suites/imagegen/spec.md`.
