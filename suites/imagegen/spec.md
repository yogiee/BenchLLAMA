# Battery I — Image-Gen Characterization (`imagegen.py`) — SPEC

**Status:** designed 2026-06-29; **BUILT 2026-07-04** (`imagegen.py`). Contract re-probed live before
build (`/api/generate` → base64 PNG + `load_duration`/`total_duration`; `/api/ps` → `size_vram`; VLM
checklist parseable). Opt-in — `./bench.sh imagegen`, or `./bench.sh all --with-imagegen` to append it
to a full run; never in default `all`. Image models admitted to the registry via the new `image` lane.

## Purpose & scope
Characterise local **image-generation** models on **performance** + **prompt-adherence** — explicitly
**NOT quality** (image quality is subjective; even the best VLM misreads details a human catches, so we
never grade "is it good?"). This measures *how well the model captured the prompt's concrete details* and
*how fast/heavy it is*.

- **Reference only.** Results get their own table in `master.md` labelled **"reference, not ranked"** —
  image models never enter the primary ranking lanes.
- **Opt-in.** `./bench.sh imagegen` — a conscious choice. **NEVER in `all`.**
- **Infrequent.** Run when a new image-gen model lands; ~hourly-scale cost is acceptable (see Cost).

## Selection
Capability-routed by the **`image`** capability (which `update_registry.py` already *detects* but
lists-and-skips — `imagegen.py` is the one consumer that selects by it). Current models:
- `x/z-image-turbo:latest` — ZImagePipeline, 10.3B, FP8, 12.8 GB. *Claimed strength: photorealism.*
- `x/flux2-klein:9b` — Flux2KleinPipeline, 17.4B, FP4. *Claimed strength: design / illustration / text.*

## Generation
- **5 prompts × 3 images = 15 images/model.** Varied seeds, **seeds logged** (image gen is stochastic;
  averaging adherence over 3 smooths out a single unlucky/lucky seed — matters most for text + fine detail).
- **Each model's own default steps/resolution**, *reported alongside* the numbers. Reference table →
  realism beats forced parity; turbo *being* faster is the point of turbo, so we don't handicap it to
  flux's step count. Speed is therefore **not apples-to-apples by design** — report the spec, not a rank.
- **Invocation (probe-confirmed 2026-06-29):** `POST /api/generate {model, prompt, stream:false}` →
  returns `image` (base64), `load_duration`, `total_duration`. No `eval_count`/steps (no tokens).

## Metrics

### Perf (objective, straight from the API)
| Metric | Source |
|--------|--------|
| `load_s` | `load_duration` (cold load) |
| `s_per_image` | `total_duration`; `gen_s` = `total_duration − load_duration` |
| `vram_gb` / `ram_gb` | `/api/ps` `size_vram` / `size` |
Aggregate = median over the 15 images (per-model).

### Adherence (VLM-graded — `qwen2.5vl:3b`, the V-battery champion at 1.00)
- Per image, the VLM answers **yes / no / unsure** for each **✓-core** element of that prompt.
- `adherence%` = ✓-core captured / total ✓-core, averaged over the 3 images, then over prompts.
- **⚠-fine** elements are *also* graded but are **advisory only** → they go to the human-spot-check list,
  never into the confident %. (VLMs are unreliable on breath-mist, white-flecks, motion-blur, buckles, etc.)
- Grader is **blind to which model produced the image** (sees image + checklist only).

### Text — two distinct tests (the prompts gave us both)
- **Fidelity** (P3, P4): OCR the image (`minicpm-v4.6:1b` fast-OCR, cross-check `qwen2.5vl:3b`), fuzzy-match
  (ratio) to the **exact target string**. Measures *correctness*.
- **Rendering / legibility** (P5): is there **legible** text on the billboards in a **real script
  (Latin or CJK)** — readable, not gibberish? No target string. Measures *capability* (can it render text at all).

### Human-spot-check list
All ⚠-fine elements + every VLM "unsure" + any garbled-text case → surfaced in the report for the user's
eye. The VLM is a first-pass filter, never the verdict.

## Hypothesis (built into the prompt split — report confirms or refutes)
- P1, P2 are **text-free pure photoreal** → expect `z-image-turbo` to lead.
- P3, P4 are **design + text** → expect `flux2-klein` to lead.
- P5 is the **crossover** (photoreal + text rendering).

## The 5 prompts (final)

### P1 · Photoreal — Animal  *(text: none)*
> At golden hour, capture a close-up of a red fox in untouched snow, focusing on the sharp texture of its
> fur and visible breath mist. Emphasize the enigmatic, mysterious nature... subtle, serene elements like
> frosty air and gently falling snowflakes onto the fox's back. Use a 'fluffed' and 'ruffled' expression...
> Sigma 85mm Art lens, at f1.4, shallow depth of field while keeping the fox completely in focus.
- **✓ core:** red fox · untouched snow · golden-hour warm light · close-up framing · shallow DoF (bg soft / fox sharp)
- **⚠ fine:** visible breath mist · falling snowflakes · snow on its back · crisp individual-fur texture

### P2 · Photoreal — Human  *(text: none)*
> Candid, documentary-style portrait of an elderly fisherman at harbor dock, early morning. Weathered face,
> deep-set eyes, wrinkles; long unruly grey beard with white flecks; worn yellow raincoat with stains/tears.
> Still quiet morning; rough dock textures, worn ropes, mistiness. Soft muted golden light; rim-light from
> left-behind on face/hair edges. 50mm Nikon, f/2.4 — fisherman tack-sharp, harbor melts to bokeh but stays identifiable.
- **✓ core:** elderly man · weathered/wrinkled face · long unruly grey beard · worn yellow raincoat · harbor/dock setting · soft morning light · candid portrait · bokeh bg (harbor readable)
- **⚠ fine:** white flecks in beard · stains/tears on coat · rim-light from left-behind · worn ropes / rough dock texture · misty air

### P3 · Design — Poster + Text  *(text fidelity)*
> Minimalist jazz poster: gold saxophone silhouette on deep navy; clean sans-serif heading (Helvetica/Arial)
> proportional to the instrument. Sax: smooth curved lines, subtle shading, metal texture. Navy: slight
> gradient/texture for depth. Woman: confident/playful expression, curly hair, heels with sheen. Subtle
> lighting depth; faint grid texture; slight motion/energy around the woman.
- **✓ core:** poster/graphic style · gold saxophone silhouette · deep-navy background · sans-serif heading present · woman present · curly hair · heels
- **⚠ fine:** sax metal shading/curve detail · bg gradient/faint-grid · heading proportional to sax · figure · expression · motion-blur/energy around woman
- **Text target (OCR exact):** `BLUE NOTE, FRIDAY 9PM`

### P4 · Design — Hoarding Render  *(text fidelity)*
> Photorealistic highway billboard, daytime. **On the billboard ad:** background of cyan sky + park; woman in
> a deep-cut yellow summer dress (subtle fabric texture, jewelry/accessories), confident relaxed expression,
> direct gaze at camera, soft smile; soft-drink bottle clearly visible & appealing. **Real scene around the
> billboard:** cars on the highway emphasising motion/speed (blur/streaks). Subtle haze/mist in sky+park;
> highly detailed and realistic.
- **✓ core (on the ad):** photoreal billboard · daytime · cyan sky · park backdrop *on the ad* · woman in yellow dress · soft-drink bottle clearly visible · bold ad text present
- **✓ core (scene):** highway/road context · cars present
- **⚠ fine:** deep-cut dress + jewelry · direct-gaze/smiling expression · car motion-blur/speed streaks · atmospheric haze/mist · realistic billboard scale/perspective
- **Text target (OCR exact):** `FRESH COLA. FRESH SUMMER.`

### P5 · Neutral — Futuristic Metro  *(text legibility, no target)*
> Highly detailed near-futuristic cityscape at dusk. Kilometer-high skyscrapers (angular/curved/organic).
> Neon/holo billboards, vibrant colors/patterns, prominent — random brand text. People in revealing
> cyberpunk clothing (fabric/buckles/zippers, cyberpunk colors). Lighting effects; wet reflective pavement
> with rain/mist.
- **✓ core:** near-futuristic cityscape · dusk light · very tall skyscrapers · neon/holo billboards (vibrant, prominent) · people present · wet reflective pavement/reflections
- **⚠ fine:** angular/curved/organic building forms · cyberpunk revealing outfits (fabric/buckle/zipper detail) · rain/mist effects
- **Text test (legibility, not match):** is there **human-readable** text on the billboards in a real
  script (Latin or CJK) — not gibberish?

## Output
- `results/imagegen_<date>.json` + `.md` (gitignored, like other results). Optionally dual-write to the new
  SQLite store under battery key `image` (consistent with the storage migration; keeps history of re-runs).
- A **reference-only** table in `master.md`: model · params/quant · `load_s` · `s/image` (@reported spec) ·
  `vram` · `adherence%` · `text-fidelity` (P3/P4) · `text-legibility` (P5) · per-model notes + hypothesis verdict.

## Cost (from the 2026-06-29 probe: ~95 s/image incl. 7.7 s load on z-image-turbo; flux slower)
5 prompts × 3 images × 2 models = **30 images ≈ ~50 min** generation + VLM grading + OCR. Opt-in,
infrequent — acceptable per design.

## Build sequence
1. **`imagegen.py` perf-only** — trivial; just capture `load_duration`/`total_duration`/`/api/ps`. Could
   land standalone first (gives the load/VRAM/s-per-image reference table immediately).
2. **Adherence + text layer** — generation harness + VLM checklist grading (reuses `vision.py` VLM plumbing)
   + OCR text-match/legibility + the report.
3. **Wiring (lands with step 1 — NOT added yet; a dropdown option without `imagegen.py` is a dead button):**
   - `bench.sh` → `imagegen` command.
   - `orchestrator.py` `build_phases` → `if cmd == "imagegen": return [("Image Gen (Battery I)", _cmd(REPO/"imagegen.py", *x), "cap:image")]`,
     and add `COMMANDS`/`"image"` routing + a `BATTERY_LABELS`-style entry if needed.
   - `web/index.html` → suite dropdown `<option value="imagegen">Image Gen</option>`.
   - `update_registry.py` → admit `image`-cap models to a perf-only lane (currently lists-but-skips).
