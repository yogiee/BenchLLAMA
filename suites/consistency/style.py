#!/usr/bin/env python3
"""
BenchLLAMA — Battery F deterministic toolkit (conversational consistency).

The objective detectors the F1–F5 scores ride on. No model in the loop; stdlib only.
Everything here is WITHIN-RUN relative or a high-precision lexical count — the design
that keeps F largely immune to cross-run rollout divergence (see the §0 build gate).

  • style_vector(text)   — IDENTITY style markers only (hedge / first-person / exclaim /
                           directness rates). Deliberately excludes content-driven features
                           (sentence length, vocabulary) so a model can shift *content*
                           register without being scored as inconsistent.
  • lexicon_hits         — high-precision phrase counter.
  • detect_stance        — extract the T1 committed choice (the load-bearing detector;
                           F2-flip and F4 both ride on it — gate validates its hit-rate).
  • token_overlap        — T6↔T2 callback fidelity (content-token recall).
"""

import re
import json
from pathlib import Path

HERE = Path(__file__).parent
_STOP = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with", "is",
    "it", "this", "that", "you", "your", "i", "my", "me", "we", "he", "she", "they",
    "be", "as", "at", "by", "if", "so", "do", "does", "did", "will", "would", "can",
    "could", "what", "which", "how", "why", "their", "its", "are", "was", "were", "not",
}


def load_script():
    return json.load((HERE / "script.json").open())


def load_lexicon():
    return json.load((HERE / "lexicon.json").open())


def _words(text):
    return re.findall(r"[a-z']+", (text or "").lower())


def _sentences(text):
    return [s for s in re.split(r"[.!?\n]+", text or "") if s.strip()]


def lexicon_hits(text, terms):
    """Count occurrences of each term (case-insensitive; word-boundary for single words,
    substring for multi-word phrases). Returns total hit count."""
    low = (text or "").lower()
    n = 0
    for t in terms:
        if " " in t or "'" in t:
            n += low.count(t)
        else:
            n += len(re.findall(r"\b" + re.escape(t) + r"\b", low))
    return n


def style_vector(text, lex):
    """Identity style markers as per-unit rates (NOT content-driven)."""
    w = _words(text)
    nw = max(1, len(w))
    sents = _sentences(text)
    ns = max(1, len(sents))
    hedge = lexicon_hits(text, lex["hedge_words"]) / nw
    first = lexicon_hits(text, lex["first_person_stance"]) / nw
    exclaim = (text or "").count("!") / ns
    # directness ≈ imperative-mood sentence starts (sentence begins with an imperative verb)
    imp = set(v for v in lex["imperative_verbs"] if " " not in v)
    direct = sum(1 for s in sents if (s.strip().split() or [""])[0].lower().strip(",.:;") in imp) / ns
    return {"hedge": hedge, "first_person": first, "exclaim": exclaim, "direct": direct}


def vector_drift(vec, baseline):
    """Mean per-dim absolute drift from baseline, normalized by (baseline + floor) so a
    small-baseline dim isn't infinitely sensitive. ~0 = identical, grows with drift."""
    keys = baseline.keys()
    return sum(abs(vec[k] - baseline[k]) / (baseline[k] + 0.10) for k in keys) / len(baseline)


def mean_vector(vecs):
    keys = vecs[0].keys()
    return {k: sum(v[k] for v in vecs) / len(vecs) for k in keys}


def detect_stance(text, options):
    """Return (key, label) of the option the text commits to, or (None, None).
    Most keyword hits wins; tie broken by earliest first occurrence. The load-bearing
    detector — if this misses, F2-flip and F4 are unreliable for the model."""
    low = (text or "").lower()
    best, best_hits, best_pos = None, 0, 1e9
    for opt in options:
        hits = sum(low.count(k) for k in opt["keywords"])
        pos = min((low.find(k) for k in opt["keywords"] if low.find(k) >= 0), default=1e9)
        if hits > best_hits or (hits == best_hits and hits > 0 and pos < best_pos):
            best, best_hits, best_pos = opt, hits, pos
    if best and best_hits > 0:
        return best["key"], best["label"]
    return None, None


def token_overlap(later, earlier):
    """Fraction of `earlier`'s content tokens that reappear in `later` (recall).
    Used for T6 (callback) referencing the T2 plan."""
    e = {w for w in _words(earlier) if len(w) > 3 and w not in _STOP}
    if not e:
        return 0.0
    l = {w for w in _words(later) if w not in _STOP}
    return len(e & l) / len(e)


def is_degenerate(text):
    """T3/T8 'didn't collapse' check: empty, ultra-short, or dominated by disclaimer tells."""
    w = _words(text)
    if len(w) < 8:
        return True
    return lexicon_hits(text, load_lexicon()["disclaimer_tells"]) >= 2 and len(w) < 30
