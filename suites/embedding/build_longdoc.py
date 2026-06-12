#!/usr/bin/env python3
"""
BenchLLAMA — Long-document retrieval corpus builder (length-stratified tail probe).

Writes retrieval_long.json — the dataset that gives Battery EMB a *context-window*
dimension the short retrieval.json (33 docs, ~60 chars each) never exercised.

Motivation: a 512-token model and an 8k-token model look identical on short docs
because no input ever approaches either limit. The 2026-06-13 MemoryCentral
embedding eval showed this directly — granite-embedding:30m "won" the battery,
then truncated 66% of a real 191-memory corpus (dropping 78k chars) because its
512-token window never engaged on the toy-length benchmark corpus.

This corpus fixes that. Docs are generated at five length buckets (nominal token
counts realised at ~4 chars/token):

    bucket(tok)   chars     who truncates here
    256           1024      nobody
    512           2048      granite (~1400-char cap)
    1024          4096      granite
    2048          8192      granite; embeddinggemma borderline (2048-tok window)
    4096         16384      granite, embeddinggemma; only 8k+-token models hold

Each doc carries TWO unique, DISTINCTIVE, OFF-TOPIC nuggets — a unique city
(head) and a unique codename (tail). Distinctive real-world entities (not opaque
alphanumeric codes) because embeddings collapse codes like "AH-1001"/"AH-1002"
into near-identical subwords and can't rank them. Off-topic from the doc's bulk
runbook/API prose so the bulk text can NOT answer the nugget query — that's what
makes truncation detectable: drop the tail sentence and its city/codename is the
only thing in the whole doc that matched the query, so the doc falls out of the
top-K.

  • HEAD nugget — a unique city, planted in the FIRST ~200 chars. Survives even
    aggressive truncation, so its query is a CONTROL: confirms the model can
    retrieve the doc at all, independent of window size.
  • TAIL nugget — a unique codename, planted as the LAST sentence. Only encoded if
    the model's window covers the whole doc. Its query is the TRUNCATION PROBE.

NOTE: this Ollama build silently truncates over-long input rather than returning
HTTP 500, so the effective window is measured BEHAVIOURALLY (the bucket at which
tail recall craters = max_clean_bucket_tok in embedding.py), not by probing for
errors.

Per bucket, per model:
    head_recall ≈ 1.0 always (if the model is competent)
    tail_recall craters once doc_len > the model's true window

The gap (head_recall − tail_recall) is the truncation signature; embedding.py's
window probe (max_input_chars) disambiguates "small window" from "weak model".

Deterministic, offline, reproducible — no network, no Math.random equivalent.

  python3 suites/embedding/build_longdoc.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent

# Nominal token buckets → char length at ~4 chars/token. The window probe in
# embedding.py reports each model's *true* char cap, so this approximation is
# transparent rather than load-bearing.
CHARS_PER_TOKEN = 4
BUCKETS_TOK = [256, 512, 1024, 2048, 4096]

DOCS_PER_BUCKET = 20        # within-bucket pool size → recall@1 baseline 0.05
TAIL_RESERVE_CHARS = 200    # space kept at the end for the tail nugget sentence

# System names (intro label only — not the queried signal). Indexed within-bucket.
NAMES = [
    "Halcyon", "Meridian", "Cobalt", "Tessera", "Praxis", "Ardent", "Vega", "Lyra",
    "Orion", "Draco", "Corvus", "Phoenix", "Strata", "Keystone", "Lattice", "Quorum",
    "Bastion", "Fulcrum", "Aurora", "Nimbus", "Tempest", "Solstice", "Zephyr", "Quasar",
    "Helix", "Cipher", "Vertex", "Polaris", "Cascade", "Onyx", "Apex", "Beacon",
    "Citadel", "Delta", "Echo", "Forge", "Glacier", "Horizon", "Ion", "Juno",
    "Kestrel", "Lumen", "Mosaic", "Nexus", "Oasis", "Pinnacle", "Quill", "Rampart",
    "Summit", "Talon",
]

# Five themes, one per bucket, each with six distinct fictional system names so a
# query resolves to exactly one doc. Filler is theme-flavoured generic prose,
# cycled deterministically to pad to the target length.
THEMES = [
    {
        "key": "runbook",
        "label": "operations runbook",
        "names": ["Halcyon", "Meridian", "Cobalt", "Tessera", "Praxis", "Ardent"],
        "filler": [
            "The service is deployed across three availability zones behind a regional load balancer.",
            "Health checks poll the readiness endpoint every fifteen seconds and drain on failure.",
            "Rollouts proceed canary-first, holding ten percent of traffic before a full promotion.",
            "Secrets are mounted from the vault at boot and never written to the container filesystem.",
            "Each pod requests two cores and four gigabytes, with autoscaling on sustained CPU.",
            "Structured logs ship to the aggregator and are retained for thirty days by default.",
            "Database migrations run in a separate job that gates the application rollout.",
            "Back-pressure is applied at the queue consumer when downstream latency rises.",
            "Alerts page the on-call engineer when error rate exceeds one percent for five minutes.",
            "Blue-green cutover keeps the previous version warm for an instant rollback.",
            "Connection pools are sized to the database's max-connections minus a safety margin.",
            "Feature flags gate risky paths and default to the safe branch when the flag service is down.",
        ],
    },
    {
        "key": "apiref",
        "label": "API reference",
        "names": ["Vega", "Lyra", "Orion", "Draco", "Corvus", "Phoenix"],
        "filler": [
            "All endpoints accept and return JSON encoded as UTF-8 over HTTPS only.",
            "Authentication uses a bearer token passed in the Authorization header on every request.",
            "Rate limits are enforced per API key and surfaced in the X-RateLimit response headers.",
            "Pagination is cursor-based; clients follow the next link until it is null.",
            "Timestamps are ISO-8601 in UTC and fields are returned in camelCase.",
            "A 429 response includes a Retry-After header indicating when to resume.",
            "Idempotency keys make retried POST requests safe against duplicate writes.",
            "Webhook payloads are signed with an HMAC the receiver must verify before trusting.",
            "Partial responses are supported through a fields query parameter for sparse fieldsets.",
            "Errors follow a problem-details envelope with a type, title, and detail member.",
            "Bulk operations cap at five hundred items per request to bound server work.",
            "Deprecated fields remain for two minor versions before removal under semver.",
        ],
    },
    {
        "key": "adr",
        "label": "architecture decision record",
        "names": ["Strata", "Keystone", "Lattice", "Quorum", "Bastion", "Fulcrum"],
        "filler": [
            "The team weighed a monolith against services and chose a modular monolith to start.",
            "Read traffic dominates writes by roughly forty to one in the observed workload.",
            "Strong consistency was preferred over availability for the billing path.",
            "An event log is the source of truth; projections are rebuilt from it on demand.",
            "The decision trades higher write latency for simpler reasoning about ordering.",
            "Caching sits in front of the read model with a short, explicit time-to-live.",
            "Schema changes are additive first, with destructive steps deferred to a later release.",
            "The chosen queue guarantees at-least-once delivery, so consumers are idempotent.",
            "Cross-service calls are budgeted against a strict end-to-end latency target.",
            "Observability was treated as a first-class requirement, not an afterthought.",
            "The alternative of a third-party platform was rejected on data-residency grounds.",
            "A reversal cost was estimated so the decision could be revisited without lock-in.",
        ],
    },
    {
        "key": "postmortem",
        "label": "incident postmortem",
        "names": ["Aurora", "Nimbus", "Tempest", "Solstice", "Zephyr", "Quasar"],
        "filler": [
            "The incident began when a deploy shipped a config that halved the connection pool.",
            "Latency climbed as requests queued waiting for a free database connection.",
            "The first alert fired four minutes after the regression reached production.",
            "On-call mitigated by rolling back, which restored the pool to its prior size.",
            "A contributing factor was a missing canary gate on configuration-only changes.",
            "Customer impact was elevated error rates on the checkout path for nineteen minutes.",
            "The runbook lacked a step for config rollback, slowing the response.",
            "Monitoring covered request latency but not pool saturation directly.",
            "The fix adds a saturation metric and a hard alert before exhaustion.",
            "A follow-up enforces canary analysis for config as well as code.",
            "No data was lost; the failure was availability rather than integrity.",
            "The timeline was reconstructed from structured logs and the deploy audit trail.",
        ],
    },
    {
        "key": "research",
        "label": "research note",
        "names": ["Helix", "Cipher", "Vertex", "Polaris", "Cascade", "Onyx"],
        "filler": [
            "The study compared dense retrieval against a sparse lexical baseline on a held-out set.",
            "Embeddings were normalised before cosine similarity to remove magnitude effects.",
            "Recall improved with larger context until the encoder's window saturated.",
            "Beyond the window, appended content was silently dropped and recall plateaued.",
            "Chunking long documents recovered most of the lost recall at a storage cost.",
            "The ablation isolated window size from model capacity by holding parameters fixed.",
            "Hard negatives sharing surface words were the main source of ranking errors.",
            "Quality-per-gigabyte favoured the smaller model until inputs exceeded its window.",
            "Throughput scaled roughly linearly with batch size up to the memory ceiling.",
            "The evaluation used self-contained synthetic facts to avoid leakage from pretraining.",
            "Results held across two random seeds, suggesting the effect was not init luck.",
            "The note recommends gating model choice on window fit before optimising for speed.",
        ],
    },
]


# Rich, distinctive, OFF-TOPIC nuggets. Each is built from THREE distinctive tokens
# (drawn from disjoint pools, indexed by doc) so the sentence carries ~3 unique
# content words — enough lexical signal to beat mean-pooling dilution in a 16k-char
# doc (verified: a 1-token nugget drowns; a 3-token rich sentence ranks #1 even at
# doc-doc cos 0.99). Head pools and tail pools are different categories so head and
# tail queries can't cross-match. None of these words appear in the theme filler.
PROFESSIONS = [
    "marine biologists", "glaciologists", "cartographers", "seismologists",
    "beekeepers", "lexicographers", "astronomers", "vintners", "falconers",
    "blacksmiths", "archivists", "botanists", "cryptographers", "ornithologists",
    "sommeliers", "geologists", "paleontologists", "herbalists", "locksmiths",
    "shipwrights", "calligraphers", "mycologists", "horologists", "weavers",
    "foragers", "surveyors", "luthiers", "apiarists", "hydrologists", "typographers",
    "stonemasons", "glassblowers", "cobblers", "tanners", "coopers", "fletchers",
    "wheelwrights", "thatchers", "chandlers", "gilders", "etchers", "bookbinders",
    "perfumers", "cartwrights", "brewers", "distillers", "dyers", "joiners",
    "saddlers", "milliners",
]
PHENOMENA = [
    "bioluminescent plankton", "glacial meltwater", "migratory storks",
    "volcanic basalt", "desert mirages", "tidal bores", "aurora curtains",
    "peat bogs", "salt flats", "coral spawning", "monsoon winds",
    "limestone caverns", "geyser fields", "mangrove roots", "fjord currents",
    "dune migration", "lichen growth", "thermal vents", "meteor showers",
    "river deltas", "kelp forests", "ash plumes", "cloud forests",
    "mineral springs", "frost heaves", "ocean gyres", "marsh reeds",
    "canyon strata", "tundra moss", "reef shoals",
    "sea ice", "dust storms", "pollen drifts", "magma chambers", "brine pools",
    "snow squalls", "sediment cores", "algal blooms", "cave pearls", "sun pillars",
    "rip currents", "hailstorms", "sinkholes", "glacier calving", "fog banks",
    "mud volcanoes", "salt marshes", "gypsum dunes", "iron springs", "granite tors",
]
PLACES = [
    "the Faroe Islands", "Patagonia", "the Atlas Mountains", "Hokkaido",
    "the Yucatan", "Tasmania", "Lapland", "the Azores", "the Gobi",
    "Madagascar", "the Hebrides", "Anatolia", "the Andes", "Borneo",
    "the Pyrenees", "Zanzibar", "Kamchatka", "Newfoundland", "the Serengeti",
    "the Dolomites", "Sardinia", "the Yukon", "Crete", "the Galapagos",
    "the Carpathians", "Sumatra", "the Falklands", "Greenland", "the Maldives",
    "the Kalahari",
    "the Outer Banks", "Svalbard", "the Cevennes", "Kyushu", "the Altai",
    "Corsica", "the Cyclades", "Tierra del Fuego", "the Karakoram", "Jutland",
    "the Apennines", "Lombok", "the Cairngorms", "Nova Scotia", "the Tatras",
    "Bali", "the Urals", "Flores", "the Balearics", "Honshu",
]
ERAS = [
    "Etruscan", "Byzantine", "Phoenician", "Minoan", "Babylonian", "Nabataean",
    "Sumerian", "Mughal", "Carolingian", "Ottoman", "Mayan", "Norse", "Aztec",
    "Khmer", "Assyrian", "Celtic", "Persian", "Numidian", "Thracian", "Edo-period",
    "Tang-dynasty", "Songhai", "Hittite", "Olmec", "Polynesian", "Gothic",
    "Mycenaean", "Zapotec", "Visigothic", "Achaemenid",
    "Roman", "Hellenistic", "Frankish", "Lydian", "Parthian", "Sassanid",
    "Umayyad", "Abbasid", "Chola", "Gupta", "Maurya", "Jomon", "Shang", "Zhou",
    "Inca", "Toltec", "Moche", "Nazca", "Wari", "Iberian",
]
ARTIFACTS = [
    "pottery shard", "bronze astrolabe", "woven tapestry", "stone tablet",
    "carved amulet", "silver chalice", "clay seal", "ivory comb", "glass bead",
    "iron key", "jade figurine", "leather scroll", "copper mirror", "marble bust",
    "lacquer box", "feather headdress", "bone flute", "gold pendant", "ceramic vase",
    "wooden mask", "mosaic tile", "engraved ring", "painted urn", "embroidered banner",
    "crystal lens", "terracotta lamp", "beaded necklace", "obelisk fragment",
    "enameled brooch", "sandstone relief",
    "bronze dagger", "silver brooch", "clay figurine", "stone seal", "ivory plaque",
    "gold torc", "glass vial", "iron buckle", "jade disc", "bone needle",
    "copper bowl", "marble frieze", "wooden idol", "ceramic jug", "painted shield",
    "woven sash", "carved lintel", "amber pendant", "slate tablet", "flint blade",
]
INSTITUTIONS = [
    "Bologna museum", "Uppsala archive", "Coimbra library", "Leiden observatory",
    "Ghent cathedral", "Kyoto monastery", "Toledo seminary", "Bruges guildhall",
    "Salamanca university", "Aarhus institute", "Pavia academy", "Trondheim college",
    "Utrecht conservatory", "Padua hall", "Krakow foundation", "Lund repository",
    "Heidelberg vault", "Bergen society", "Delft workshop", "Verona atelier",
    "Modena collection", "Siena chapterhouse", "Graz gallery", "Nantes archive",
    "Turku museum", "Cork institute", "Basel foundation", "Liege museum",
    "Maribor library", "Ravenna collection",
    "Lisbon archive", "Naples library", "Antwerp museum", "Geneva institute",
    "Vienna academy", "Prague repository", "Mainz vault", "Lyon college",
    "Seville cathedral", "Bremen society", "Tartu observatory", "Oviedo seminary",
    "Parma gallery", "Rouen guildhall", "Cadiz foundation", "Lecce atelier",
    "Trieste collection", "Ferrara hall", "Zadar chapterhouse", "Bath museum",
]


def _nuggets(i):
    """Return (head_sentence, head_query, tail_sentence, tail_query) for doc i."""
    prof, phen, place = PROFESSIONS[i], PHENOMENA[i], PLACES[i]
    era, art, inst    = ERAS[i], ARTIFACTS[i], INSTITUTIONS[i]
    head_s = (f"A historical note records that this system was first prototyped by "
              f"{prof} researching {phen} near {place}. ")
    head_q = f"Which document mentions {prof} researching {phen} near {place}?"
    tail_s = (f"A closing remark credits the original concept to a {era} {art} "
              f"preserved at the {inst}.")
    tail_q = f"Which document credits its concept to a {era} {art} at the {inst}?"
    return head_s, head_q, tail_s, tail_q


def _build_doc(theme, name, target_chars, head_s, tail_s):
    """Assemble one doc ~target_chars long. Rich head nugget (off-topic) in the
    first ~250 chars; rich tail nugget (off-topic) as the final sentence. The bulk
    is generic theme prose; only the head/tail nugget answers its query, so when a
    model's window truncates the doc the tail nugget is gone and its query can no
    longer find it (the truncation signal)."""
    intro = (f"System {name} — {theme['label']}. "
             f"This document describes the operating characteristics of {name}. ")
    parts = [intro, head_s]
    cur = len(intro) + len(head_s)
    budget = target_chars - TAIL_RESERVE_CHARS
    i, fill = 0, theme["filler"]
    while cur < budget:
        s = fill[i % len(fill)] + " "
        parts.append(s)
        cur += len(s)
        i += 1
    parts.append(tail_s)
    return "".join(parts)


def main():
    # v2: scoring is WITHIN-BUCKET (same-length distractors only) to neutralise
    # length bias, so nugget entities only need to be unique *within a bucket*, not
    # globally — indexed by within-bucket position j (0..DOCS_PER_BUCKET-1), reused
    # across buckets. Lets n rise to 20/bucket on the existing 50-entry pools.
    docs, queries = [], []
    gidx = 0
    for b_tok in BUCKETS_TOK:
        target_chars = b_tok * CHARS_PER_TOKEN
        theme = THEMES[BUCKETS_TOK.index(b_tok)]
        for j in range(DOCS_PER_BUCKET):
            name = NAMES[j]
            head_s, head_q, tail_s, tail_q = _nuggets(j)
            text = _build_doc(theme, name, target_chars, head_s, tail_s)
            did = f"L{gidx:02d}"
            docs.append({
                "id": did,
                "bucket_tok": b_tok,
                "target_chars": target_chars,
                "actual_chars": len(text),
                "theme": theme["key"],
                "name": name,
                "text": text,
            })
            # HEAD query — control: nugget at ~char 250, survives truncation
            queries.append({"id": f"{did}-head", "bucket_tok": b_tok, "zone": "head",
                            "text": head_q, "relevant": [did]})
            # TAIL query — truncation probe: nugget is the final sentence
            queries.append({"id": f"{did}-tail", "bucket_tok": b_tok, "zone": "tail",
                            "text": tail_q, "relevant": [did]})
            gidx += 1

    out = {
        "source": "seed (curated, length-stratified)",
        "chars_per_token": CHARS_PER_TOKEN,
        "buckets_tok": BUCKETS_TOK,
        "docs_per_bucket": DOCS_PER_BUCKET,
        "docs": docs,
        "queries": queries,
    }
    (HERE / "retrieval_long.json").write_text(json.dumps(out, indent=2))

    print(f"Wrote retrieval_long.json to {HERE}/")
    print(f"  {len(docs)} docs across {len(BUCKETS_TOK)} buckets "
          f"({DOCS_PER_BUCKET}/bucket), {len(queries)} queries "
          f"({len(queries)//2} head + {len(queries)//2} tail)")
    print("  bucket(tok)  target_chars  actual_char_range")
    for b in BUCKETS_TOK:
        sizes = [d["actual_chars"] for d in docs if d["bucket_tok"] == b]
        print(f"    {b:<11} {b*CHARS_PER_TOKEN:<13} {min(sizes)}–{max(sizes)}")


if __name__ == "__main__":
    main()
