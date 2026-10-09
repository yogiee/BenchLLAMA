# Spec input — embedding battery (EMB): scenarios and method

Written 2026-10-09 as input to the suite redesign. The scenarios below were chosen by the user.
The method is the HOW proposal. Nothing here is built yet: this note is folded into the new spec
first, then implemented.

The standing constraints are unchanged: **standard = NORMALIZED, aptitude = OPTIMAL**, and the
suite models a non-expert developer's real local-LLM workflow (Ollama, then OpenWebUI, then
OpenCode or Continue).

---

## 0. Neutrality rule: real-world results are learning points, not fixtures

BenchLLAMA stays consumer-neutral. A consumer's own evaluation (MemoryCentral's 2026-10-09 A/B
on its private corpus) can show us **where our instrument mispredicts reality**. We then fix the
instrument with **synthetic, neutral data** that reproduces the *shape* of the real case. We never
adopt a consumer's corpus, queries or relevance grades as a battery, not even as an optional
local-only tier.

What MemoryCentral's A/B taught us, and where each lesson lands:

| MemoryCentral finding (real notes, 679 memories) | Lands in |
|---|---|
| Our filler tail probe predicted v1 > v2 on buried facts. On dense real notes the reverse held (+0.035 MRR buried, +0.06–0.08 paraphrased detail). | §2 S2: facts among related facts |
| Chunk size moved retrieval as much as the model did (v2: 4000 → 1500 chars, buried MRR 0.483 → 0.600). A bigger window is not better retrieval. | §2 S1/S2 chunk axis; §2 S3 demoted to a fit check |
| RAM: a single snapshot is wrong in both directions. llama.cpp keeps its grown buffers (v1 2.37 GB settled); MLX frees per batch (v2 bf16 peak 2.76, settled 0.73; nvfp4 peak 3.16). | §3 RAM peak + settled |
| Prompt prefixes help EmbeddingGemma-2 and are mixed on v1. | §1 arms |
| Every embeddinggemma-2 tag is MLX, so it is Apple Silicon only. | §3 `platforms` |
| On n≈30 queries almost nothing is significant; paired bootstrap separated effect from noise. | §4 statistics |
| Probe-construction bug: a "buried" sentence sliced mid-text leaves a word-stump fragment. | §2 S2 construction rule |

Not taken: judged-coverage bias (our labels are complete by construction), corpus freezing (ours is
deterministic already), and the MiniLM/ONNX engine facts (the harness is Ollama-only).

## 1. Persona evidence → two arms

Verified 2026-10-09 against the OpenWebUI and Continue docs (Context7):

- **Continue deprecated codebase indexing (`@codebase`).** Its agents use file-search and grep tools
  now. OpenCode also doesn't embed. **Code search is therefore out of scope.**
- **OpenWebUI document RAG ("Knowledge", uploaded files) is the persona's embedding use.**
  - Defaults: `character` splitter, **1000 chars, overlap 100**, Markdown header splitting **on**, **top-K 3**.
  - Recommended starting config: `token` splitter (tiktoken), **2000 tokens, overlap 200**, top-K 15,
    or **5 for local models** with small contexts.
  - Its built-in default embedder is sentence-transformers MiniLM on CPU, not Ollama. The persona we
    measure is the one who points it at Ollama (the docs' own example is `nomic-embed-text`).
  - Prefixes are opt-in, via `RAG_EMBEDDING_QUERY_PREFIX` / `RAG_EMBEDDING_CONTENT_PREFIX`.
    By default none are sent.

This maps directly onto the two-mode principle:

| | **Standard arm (normalized = out of the box)** | **Aptitude arm (optimal = configured)** |
|---|---|---|
| Request | `/api/embed` with **no options** | `num_ctx` = `num_batch` = the enforced window (read from the runtime's own 413/400, as EMB v3 does) |
| Prefixes | none | the model's documented prefixes, if any (measured both ways; kept only if they help **that** model) |
| Chunking | OpenWebUI default: 1000 chars / 100, header-split | the model's best chunk size from the sweep (§2) |
| Top-K graded | 3 | 3 and 5 |

The difference between the arms is a user-facing number in its own right: **how much you gain by
configuring**.

⚠ Traps the standard arm must reproduce faithfully, not paper over: with no options, Ollama's
`num_batch` default (2048) caps encoder models. embeddinggemma-2 cuts any input over 8,192 tokens
down to 2,048 (measured 10-08). And OpenWebUI's *recommended* 2000-token chunk exceeds the
**trained** window of MiniLM (256) and granite:30m (512). S3 exists to catch exactly that.

## 2. Scenarios (user-selected 2026-10-09: all four)

All corpora are synthetic, seeded and deterministic, built offline like `suites/longctx/build.py`.

### S1 — Document Q&A ("will it find the right passage in my documents?")

- **Corpus:** manual- or handbook-style Markdown documents (headings, numbered settings, tables
  flattened to prose), from a few pages to about 40 pages each. The realistic hard case is
  **near-duplicate sections**: the same setting described for two product versions or two
  environments with different values. That is the document-world analog of Battery G v2's decoy
  relays.
- **Chunking:** a reimplementation of OpenWebUI's pipeline, with Markdown header split first, then
  a recursive character split at 1000/100 (standard). The recommended 2000-token/200 setting is
  graded too, because it's what the docs tell the persona to switch to.
- **Queries:** two kinds per answer section. A *lookup* query (close to the source wording) and a
  *paraphrase* query (no shared content words beyond the entity).
- **Grade:** **hit@3** (default top-K), hit@5 and MRR. A query passes if any retrieved chunk
  contains the whole answer sentence.

### S2 — Notes & memory search ("will it find the note I wrote about this?")

- **Corpus:** dense short notes (200–1500 chars), many per topic, so every fact sits **among related
  facts**, with same-format decoys. This replaces filler as the hard case. The filler tail probe
  stays only as a diagnostic.
- **Queries:** *buried verbatim* (the fact sits in the second half of a long note) and *paraphrased
  detail*.
- **Chunk axis:** 500 / 1500 / 4000 chars, with heading-aligned splits and 10% overlap. A note's score
  is its **best chunk** (max-collapse, which is how note search tools rank). The output is each model's
  curve plus its best chunk size, which feeds the aptitude arm.
- **Construction rule:** plant and slice **whole sentences only**, never a mid-text cut.

### S3 — Long-document fit check ("can it read a chunk this size whole?")

- **Reuses the EMB v3 machinery as is:** real-token buckets, an exact `cut` count per document from
  `prompt_eval_count`, and the window read from the runtime's own error.
- **Recast as pass/fail per chunk setting:** does the model read 1000 chars, 2000 tokens and 8k tokens
  whole, under each arm? The headline is a plain-language **"safe max chunk size"**.
- **Not a ranking key.** Today's `embedding_long` (mean tail MRR to 16k) rewards window length as such,
  and MemoryCentral showed that a bigger window is not better retrieval. This is the
  "rank by duty, not composite" lesson again.

### S4 — Short-text similarity ("find similar questions / duplicates")

- Keep the real STS-B slice (n=200, the only well-powered short task).
- **The rest is too small to rank on.** Evidence from the 2026-10-08 run: triplet n=16 (one item ≈ 6
  points), retrieval 12 queries over 33 docs, clustering 36 items. bf16 and nvfp4 of the *same* model
  (vector cosine 0.984 per MemoryCentral) differ by **0.11 clustering purity** (0.53 vs 0.64). That is
  noise, yet it moves the short composite. granite:30m's 0.84 "win" stands on 12 queries.
- Grow triplet and clustering to ≥200 items, or fold short retrieval into S1/S2 and drop it here.

## 3. Cross-cutting measurements (both arms)

- **RAM peak + settled**, measured under production-length input. Peak = continuous footprint sampling
  during the longest chunks. Settled = a reading about 3 s after the workload. Report both; never one
  snapshot (`bench_utils.measured_ram_gb` takes one today, which is the MLX trap).
  `quality_per_gb` currently divides by **disk** size; use settled RAM instead.
  ⚠ The same peak/settled issue applies to MLX chat models.
- **Indexing time in user terms:** "time to index 100 pages" at the arm's chunk setting, plus query
  latency p50. EMB's current `emb/s` is measured on one-line texts and says nothing about indexing
  real documents.
- **`platforms`** per model, derived from the runner/format in `/api/tags` (`mlx` → Apple Silicon only)
  and exported, so consumers never get an MLX pick off Apple Silicon.
- `dim` from the API response (the `ollama show` "embedding length" can be wrong: embeddinggemma-2
  shows 512 but returns 768).

## 4. Statistics

- **≥200 queries per scenario.** Paired-bootstrap CIs per query for every pairwise comparison that a
  ranking or a verdict relies on.
- Before trusting any ranking key, **check its tie distribution** (the recurring defect shape in
  CLAUDE.md). Quants of the same model are the built-in control: if they separate by more than the
  CI, the instrument is measuring noise.

## 5. Open — for the user (the WHAT)

1. **Ranking shape:** per-scenario lists (Document Q&A / Notes / Similarity), one combined list, or
   per-scenario lists plus a plain "best all-rounder" pick. S3 stays a fit flag either way.
2. **Which arm leads the headline:** out-of-the-box (what the persona will actually get) or configured
   (what's possible)?
3. **tiktoken as a dependency** for the exact 2000-token OpenWebUI setting, versus a measured
   chars-per-token approximation (about 4 for English prose under cl100k). This is a HOW detail, but
   it adds a dependency, so it's flagged.
4. Language scope: English-only (current) unless multilingual RAG is part of the persona.
