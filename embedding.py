#!/usr/bin/env python3
"""
BenchLLAMA — Battery EMB (Embedding)

Capability-routed: runs every model whose `capabilities` include `embedding`
(role=utility in models.json). Fully objective — no chat, no LLM judge.

Four tasks, scored with numpy only (no scipy/sklearn):
  sts        — graded sentence-pair similarity → Spearman(cosine, human score)
  triplet    — anchor/positive/negative        → % where sim(a,p) > sim(a,n)
  retrieval  — docs + queries w/ relevance       → recall@5, MRR, nDCG@10
  clustering — labelled sentences                → centroid purity

Plus operational metrics that decide "earns its keep":
  vector dim · embeddings/sec · disk MB · composite quality · quality-per-GB

Datasets live in suites/embedding/*.json (run build_seed.py once; fetch.py
upgrades sts.json to the real STS-B dev slice — the standard half of the hybrid).

Usage:
  python3 embedding.py                                   # all embedding models
  python3 embedding.py --models nomic-embed-text:latest  # specific
  python3 embedding.py --force                            # ignore 24h resume window
  python3 embedding.py --ollama http://host:11434
"""

import json
import sys
import time
import requests
import numpy as np
from pathlib import Path
from datetime import date
from bench_utils import latest_result

REPO        = Path(__file__).parent
RESULTS_DIR = REPO / "results"
DATA_DIR    = REPO / "suites" / "embedding"
STATUS_FILE = RESULTS_DIR / "status.json"
RESULTS_DIR.mkdir(exist_ok=True)

# ── CLI ─────────────────────────────────────────────────────────────────────────

def _flag(name): return name in sys.argv
def _arg(name, default=None):
    if name in sys.argv:
        idx = sys.argv.index(name)
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--"):
            return sys.argv[idx + 1]
    return default

force       = _flag("--force")
ollama_host = _arg("--ollama", "http://localhost:11434")
model_args  = []
if "--models" in sys.argv:
    idx = sys.argv.index("--models")
    model_args = [a for a in sys.argv[idx + 1:] if not a.startswith("--")]

TIMEOUT  = 120
BATCH    = 64
TOPK     = 5     # recall@K
NDCG_K   = 10

# ── Model selection (by capability) ──────────────────────────────────────────────

def load_models_by_cap(cap):
    path = REPO / "models.json"
    if not path.exists():
        sys.exit(f"models.json not found at {path} — run update_registry.py first")
    return [(m["name"], m.get("disk_gb", 0.0))
            for m in json.load(path.open())
            if cap in m.get("capabilities", [])]

# ── Datasets ─────────────────────────────────────────────────────────────────────

def _load(name):
    p = DATA_DIR / name
    if not p.exists():
        sys.exit(f"{p} missing — run: python3 suites/embedding/build_seed.py")
    return json.load(p.open())

# ── Ollama embed ─────────────────────────────────────────────────────────────────

def embed_texts(model, texts):
    """Return (np.ndarray [n, dim] float32, wall_seconds). Batched."""
    vecs, wall = [], 0.0
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        t0 = time.time()
        r = requests.post(f"{ollama_host}/api/embed",
                          json={"model": model, "input": chunk}, timeout=TIMEOUT)
        wall += time.time() - t0
        r.raise_for_status()
        emb = r.json().get("embeddings")
        if not emb:
            raise RuntimeError(f"{model}: empty embeddings response")
        vecs.extend(emb)
    arr = np.asarray(vecs, dtype=np.float32)
    return arr, wall

def _normalize(a):
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return a / n

def _cos_matrix(a, b):
    return _normalize(a) @ _normalize(b).T

# ── numpy stats (no scipy) ───────────────────────────────────────────────────────

def _rankdata(a):
    """Average ranks (scipy.stats.rankdata 'average' equivalent)."""
    a = np.asarray(a, float)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), dtype=int); inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    counts = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (counts[dense] + counts[dense - 1] + 1)

def _spearman(x, y):
    if len(x) < 2:
        return 0.0
    rx, ry = _rankdata(x), _rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])

def _kmeans(X, k, start, iters=50):
    """Deterministic spherical k-means (cosine) from one farthest-first seed.
    Returns (labels, objective) where objective = summed point→centroid cosine."""
    n = len(X)
    chosen = [start]
    dist = np.full(n, np.inf)
    for _ in range(1, k):
        dist = np.minimum(dist, 1.0 - X @ X[chosen[-1]])
        chosen.append(int(np.argmax(dist)))
    C = X[chosen].copy()
    labels = np.full(n, -1)
    for _ in range(iters):
        new = np.argmax(X @ C.T, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            pts = X[labels == j]
            if len(pts):
                c = pts.mean(axis=0)
                nrm = np.linalg.norm(c)
                C[j] = c / nrm if nrm else C[j]
    obj = float(sum((X[labels == j] @ C[j]).sum() for j in range(k) if (labels == j).any()))
    return labels, obj

def _kmeans_best(X, k, n_init=8):
    """Best-of-N k-means (MTEB-style n_init) with spread, deterministic seeds.
    Keeps the tightest clustering (highest objective) — robust to init luck."""
    n = len(X)
    starts = sorted({int(round(i * (n - 1) / max(1, n_init - 1))) for i in range(n_init)})
    best_lab, best_obj = None, -np.inf
    for s in starts:
        lab, obj = _kmeans(X, k, start=s)
        if obj > best_obj:
            best_obj, best_lab = obj, lab
    return best_lab

# ── Task scorers ─────────────────────────────────────────────────────────────────

def score_sts(model):
    data  = _load("sts.json")
    pairs = data["pairs"]
    a, _  = embed_texts(model, [p["a"] for p in pairs])
    b, w  = embed_texts(model, [p["b"] for p in pairs])
    cos   = np.sum(_normalize(a) * _normalize(b), axis=1)
    gold  = np.array([p["score"] for p in pairs], dtype=float)
    rho   = _spearman(cos, gold)
    return {"spearman": round(rho, 4), "n": len(pairs), "source": data.get("source", "?")}, w

def score_triplet(model):
    data = _load("triplet.json")
    tri  = data["triplets"]
    anc, _ = embed_texts(model, [t["anchor"]   for t in tri])
    pos, _ = embed_texts(model, [t["positive"] for t in tri])
    neg, w = embed_texts(model, [t["negative"] for t in tri])
    an, pn, nn = _normalize(anc), _normalize(pos), _normalize(neg)
    sap = np.sum(an * pn, axis=1)
    san = np.sum(an * nn, axis=1)
    correct = int(np.sum(sap > san))
    margin  = float(np.mean(sap - san))
    return {"accuracy": round(correct / len(tri), 4), "correct": correct,
            "n": len(tri), "mean_margin": round(margin, 4),
            "source": data.get("source", "?")}, w

def score_retrieval(model):
    data  = _load("retrieval.json")
    docs  = data["docs"]; queries = data["queries"]
    doc_ids = [d["id"] for d in docs]
    dmat, w1 = embed_texts(model, [d["text"] for d in docs])
    qmat, w2 = embed_texts(model, [q["text"] for q in queries])
    sims = _cos_matrix(qmat, dmat)   # [nq, ndoc]
    recalls, rrs, ndcgs = [], [], []
    for i, q in enumerate(queries):
        rel = set(q["relevant"])
        order = np.argsort(-sims[i])
        ranked = [doc_ids[j] for j in order]
        topk = ranked[:TOPK]
        recalls.append(len(rel & set(topk)) / len(rel))
        rr = 0.0
        for rank, did in enumerate(ranked, 1):
            if did in rel:
                rr = 1.0 / rank
                break
        rrs.append(rr)
        dcg = sum((1.0 if did in rel else 0.0) / np.log2(rank + 1)
                  for rank, did in enumerate(ranked[:NDCG_K], 1))
        idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(len(rel), NDCG_K) + 1))
        ndcgs.append(dcg / idcg if idcg else 0.0)
    return {f"recall@{TOPK}": round(float(np.mean(recalls)), 4),
            "mrr": round(float(np.mean(rrs)), 4),
            f"ndcg@{NDCG_K}": round(float(np.mean(ndcgs)), 4),
            "n_queries": len(queries), "n_docs": len(docs),
            "source": data.get("source", "?")}, (w1 + w2)

def score_clustering(model):
    data  = _load("clustering.json")
    items = data["items"]
    texts = [it["text"] for it in items]
    labs  = [it["label"] for it in items]
    mat, w = embed_texts(model, texts)
    mat = _normalize(mat)
    label_set = sorted(set(labs))
    k = len(label_set)

    # Unsupervised k-means purity (MTEB-style): cluster blind, then score each
    # discovered cluster by its majority true label. Harder than nearest-labelled-
    # centroid — fuzzy embeddings merge/split adjacent topics and purity drops.
    assign = _kmeans_best(mat, k)
    purity = 0
    for j in range(k):
        idx = [i for i in range(len(labs)) if assign[i] == j]
        if not idx:
            continue
        counts = {}
        for i in idx:
            counts[labs[i]] = counts.get(labs[i], 0) + 1
        purity += max(counts.values())
    purity /= len(labs)

    # intra vs inter cosine separation (secondary signal)
    full = mat @ mat.T
    intra, inter = [], []
    for i in range(len(items)):
        for j2 in range(i + 1, len(items)):
            (intra if labs[i] == labs[j2] else inter).append(full[i, j2])
    sep = float(np.mean(intra) - np.mean(inter)) if intra and inter else 0.0
    return {"purity": round(purity, 4), "separation": round(sep, 4),
            "n": len(items), "k": k, "metric": "kmeans_majority",
            "source": data.get("source", "?")}, w

# ── Per-model runner ─────────────────────────────────────────────────────────────

def _composite(tests):
    """Mean of the four sub-scores, each mapped to [0,1]. Spearman [-1,1]→[0,1]."""
    sts  = (tests["sts"]["spearman"] + 1) / 2
    trip = tests["triplet"]["accuracy"]
    ndcg = tests["retrieval"][f"ndcg@{NDCG_K}"]
    pur  = tests["clustering"]["purity"]
    return round(float(np.mean([sts, trip, ndcg, pur])), 4)

def run_model(model_name, disk_gb):
    print(f"\n{'='*60}\nMODEL: {model_name}  ({disk_gb} GB disk)  [embedding]\n{'='*60}", flush=True)
    result = {"model": model_name, "disk_gb": disk_gb, "tests": {}, "errors": []}
    total_texts = total_wall = 0
    dim = None
    try:
        for tid, fn in [("sts", score_sts), ("triplet", score_triplet),
                        ("retrieval", score_retrieval), ("clustering", score_clustering)]:
            print(f"  [{tid}]", end=" ", flush=True)
            res, wall = fn(model_name)
            result["tests"][tid] = res
            total_wall += wall
            print(json.dumps(res), flush=True)
        # one extra call to capture dim + a clean throughput sample
        probe, w = embed_texts(model_name, ["dimension probe"])
        dim = int(probe.shape[1])
        # rough throughput: total items embedded across tasks / total embed wall
        sts_n = result["tests"]["sts"]["n"]; tri_n = result["tests"]["triplet"]["n"]
        ret = result["tests"]["retrieval"]; clu_n = result["tests"]["clustering"]["n"]
        total_texts = sts_n * 2 + tri_n * 3 + ret["n_docs"] + ret["n_queries"] + clu_n + 1
        total_wall += w
    except Exception as e:
        print(f"\n  ✗ FAILED: {e}", flush=True)
        result["errors"].append(str(e))
        return result

    result["dim"]         = dim
    result["emb_per_sec"] = round(total_texts / total_wall, 1) if total_wall else None
    result["composite"]   = _composite(result["tests"])
    result["quality_per_gb"] = round(result["composite"] / disk_gb, 4) if disk_gb else None
    print(f"\n  dim={dim}  emb/s={result['emb_per_sec']}  "
          f"composite={result['composite']}  quality/GB={result['quality_per_gb']}", flush=True)
    return result

# ── Status + unload ──────────────────────────────────────────────────────────────

def _ws(model, phase):
    try:
        STATUS_FILE.write_text(json.dumps({"model": model, "phase": phase, "ts": time.time()}))
    except Exception:
        pass

def unload(model):
    try:
        requests.post(f"{ollama_host}/api/embed",
                      json={"model": model, "input": [""], "keep_alive": 0}, timeout=15)
    except Exception:
        pass

# ── Summary ──────────────────────────────────────────────────────────────────────

def write_summary(results, out_md):
    lines = [
        f"# Embedding Battery (EMB) — {out_md.stem}", "",
        "Objective retrieval/similarity benchmark. Datasets: STS-B slice (sts) + "
        "curated technical sets (triplet / retrieval / clustering).", "",
        "## Verdict — quality per GB", "",
        "| Model | Disk | Dim | emb/s | STS ρ | Triplet | Recall@5 | nDCG@10 | Purity | **Composite** | **Qual/GB** |",
        "|-------|-----:|----:|------:|------:|--------:|---------:|--------:|-------:|------:|------:|",
    ]
    for r in sorted(results, key=lambda x: x.get("composite", 0), reverse=True):
        if r.get("errors"):
            lines.append(f"| `{r['model']}` | {r['disk_gb']}GB | — | — | — | — | — | — | — | ERROR | — |")
            continue
        t = r["tests"]
        lines.append(
            f"| `{r['model']}` | {r['disk_gb']}GB | {r.get('dim','?')} | {r.get('emb_per_sec','?')} "
            f"| {t['sts']['spearman']} | {t['triplet']['accuracy']} | {t['retrieval']['recall@5']} "
            f"| {t['retrieval']['ndcg@10']} | {t['clustering']['purity']} "
            f"| **{r.get('composite','?')}** | **{r.get('quality_per_gb','?')}** |")
    lines += ["", "## Per-task detail", ""]
    for r in results:
        lines += [f"### `{r['model']}`", "", "```json", json.dumps(r["tests"], indent=2), "```", ""]
    out_md.write_text("\n".join(lines))
    print(f"MD → {out_md}", flush=True)

# ── Entrypoint ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TODAY    = date.today().isoformat()
    OUT_JSON = RESULTS_DIR / f"embedding_{TODAY}.json"
    OUT_MD   = RESULTS_DIR / f"embedding_{TODAY}.md"

    if model_args:
        reg = {m["name"]: m for m in json.load((REPO / "models.json").open())}
        MODELS = [(m, reg.get(m, {}).get("disk_gb", 0.0)) for m in model_args]
    else:
        MODELS = load_models_by_cap("embedding")

    if not MODELS:
        sys.exit("No embedding-capable models found. Run update_registry.py, "
                 "or pass --models <name>.")

    print(f"BenchLLAMA Battery EMB — {TODAY}", flush=True)
    print(f"ollama={ollama_host} | models: {[m[0] for m in MODELS]}", flush=True)
    print(f"Output: {OUT_JSON}", flush=True)

    # Resume SOURCE = today's file if present, else most recent embedding within
    # 24h (cross-day). Writes today's file, carrying prior results forward.
    all_results, completed = [], set()
    source = OUT_JSON if OUT_JSON.exists() else (None if force else latest_result(RESULTS_DIR, "embedding", False, 24))
    if source is not None and not force:
        try:
            loaded      = json.load(source.open())
            all_results = [r for r in loaded if not r.get("errors")]   # retry replaces, no dupes
            completed   = {r["model"] for r in all_results}
            if completed:
                via = "" if source == OUT_JSON else f" (carried from {source.name})"
                print(f"  Resuming — done{via}: {sorted(completed)}", flush=True)
        except Exception:
            all_results, completed = [], set()

    for model_name, disk_gb in MODELS:
        if model_name in completed:
            print(f"  ↷ {model_name} — already done, skipping", flush=True)
            continue
        _ws(model_name, "running")
        try:
            r = run_model(model_name, disk_gb)
        except Exception as e:
            print(f"  ✗ {model_name} FAILED: {e}", flush=True)
            r = {"model": model_name, "disk_gb": disk_gb, "errors": [str(e)], "tests": {}}
        all_results = [x for x in all_results if x["model"] != model_name] + [r]
        OUT_JSON.write_text(json.dumps(all_results, indent=2))
        unload(model_name)
        time.sleep(1)

    _ws("", "done")
    write_summary(all_results, OUT_MD)
    print(f"\n{'='*60}\nDONE\nJSON → {OUT_JSON}\nMD   → {OUT_MD}", flush=True)
