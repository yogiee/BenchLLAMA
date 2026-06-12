#!/usr/bin/env python3
"""
BenchLLAMA — Embedding seed dataset builder.

Writes four curated JSON datasets used by embedding.py (Battery EMB):
  sts.json        — graded sentence-pair similarity   → Spearman correlation
  triplet.json    — anchor / positive / negative      → % correctly ordered
  retrieval.json  — docs + queries w/ relevance        → recall@k / MRR / nDCG
  clustering.json — labelled sentences                 → centroid purity

These are SEED sets so the battery runs offline out-of-the-box. For the standard
slice (the "hybrid" half), run fetch.py to overwrite sts.json with the real
STS-B dev split. Datasets are technical/RAG-flavoured to match real use
(OllamaMCP / LookingGlass document + code retrieval).

  python3 suites/embedding/build_seed.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent

# ── STS: graded similarity, score normalised to [0,1] ──────────────────────────
# 0.0 = unrelated, 1.0 = paraphrase. Seed pairs only; fetch.py upgrades to STS-B.
STS = [
    ("A man is playing a guitar.", "A person plays an acoustic guitar.", 0.92),
    ("The cat is sleeping on the couch.", "A cat naps on the sofa.", 0.90),
    ("How do I reset my password?", "What's the way to change my login credentials?", 0.80),
    ("The server returned a 500 error.", "The backend crashed with an internal error.", 0.78),
    ("Python is a popular programming language.", "Many developers write code in Python.", 0.74),
    ("The stock market fell sharply today.", "Equity prices dropped a lot this afternoon.", 0.82),
    ("She booked a flight to Tokyo.", "He cancelled his trip to Berlin.", 0.20),
    ("The recipe needs two cups of flour.", "Add 240 grams of plain flour to the bowl.", 0.66),
    ("Quantum entanglement links two particles.", "The dog chased the ball in the park.", 0.02),
    ("This function returns a list of users.", "The method outputs an array of user records.", 0.85),
    ("The weather is sunny and warm.", "It is a bright, hot day outside.", 0.88),
    ("Our revenue grew 12% last quarter.", "Sales increased by roughly an eighth in Q3.", 0.72),
    ("The database index speeds up queries.", "Indexing makes lookups run faster.", 0.84),
    ("He repaired the leaking faucet.", "The plumber fixed the dripping tap.", 0.80),
    ("Machine learning models need data.", "Training an ML model requires examples.", 0.78),
    ("The movie was boring and too long.", "I found the film dull and overlong.", 0.86),
    ("Paris is the capital of France.", "France's capital city is Paris.", 0.95),
    ("The API rate limit is 100 requests.", "You can make up to a hundred calls before throttling.", 0.74),
    ("A bird flew over the lake.", "An airplane landed at the airport.", 0.12),
    ("Encrypt the data before storing it.", "Save the records without any protection.", 0.15),
    ("The team shipped the feature on time.", "They delivered the release by the deadline.", 0.83),
    ("Coffee contains caffeine.", "Tea also has some caffeine in it.", 0.45),
    ("The car wouldn't start this morning.", "My vehicle failed to turn over today.", 0.85),
    ("Reduce memory usage by streaming.", "Stream the data to lower RAM consumption.", 0.86),
    ("The library closes at nine.", "The museum opens at ten.", 0.18),
]

# ── Triplet: sim(anchor, positive) should exceed sim(anchor, negative). ─────────
# Negatives are chosen to share surface words where possible (harder).
TRIPLET = [
    ("How do I cache API responses?",
     "Storing API results avoids repeated network calls.",
     "How do I cache flour for baking bread?"),
    ("The deployment failed during the build step.",
     "CI broke while compiling the project.",
     "The deployment of troops failed during the war."),
    ("Best way to sort a large list in Python",
     "Use sorted() or list.sort() for ordering sequences.",
     "Best way to sort laundry by colour"),
    ("My laptop battery drains too fast.",
     "The notebook's charge runs down quickly.",
     "The river drains into the sea too fast."),
    ("Explain how indexing improves database reads.",
     "An index lets the engine skip a full table scan.",
     "Explain how indexing improves a book's usability."),
    ("How to handle a 429 rate-limit response?",
     "Back off and retry when the API throttles you.",
     "How to handle a rude customer at the counter?"),
    ("What causes memory leaks in long-running services?",
     "Unreleased references keep heap memory from being freed.",
     "What causes water leaks in old plumbing?"),
    ("Convert a JSON string to an object.",
     "Parse the JSON text into a structured value.",
     "Convert dollars to euros for the trip."),
    ("The model overfits on the training set.",
     "It memorises training data and generalises poorly.",
     "The tailor overfits the suit to the mannequin."),
    ("Schedule a recurring task every night.",
     "Run the job automatically on a nightly cron.",
     "Schedule a dentist appointment for next night out."),
    ("Compress images without losing quality.",
     "Use lossless encoding to shrink files safely.",
     "Compress a spring without losing tension."),
    ("Authenticate users with a token.",
     "Verify identity using a bearer credential.",
     "Authenticate a painting as a genuine original."),
    ("Why is my SQL query so slow?",
     "The query lacks an index and scans every row.",
     "Why is my morning jog so slow lately?"),
    ("Roll back the migration after an error.",
     "Revert the schema change when it fails.",
     "Roll back the carpet after the party."),
    ("Stream large files to avoid loading them in memory.",
     "Process the file in chunks instead of all at once.",
     "Stream the concert live to fans at home."),
    ("Set up retries for flaky network calls.",
     "Automatically re-attempt requests that time out.",
     "Set up chairs for the flaky wedding guests."),
]

# ── Retrieval: each query has one or more relevant docs by id. ──────────────────
RETRIEVAL_DOCS = [
    ("d1",  "To reset your password, open Settings, choose Security, then 'Change password'."),
    ("d2",  "Our API enforces a rate limit of 100 requests per minute per API key."),
    ("d3",  "Postgres uses B-tree indexes by default to accelerate equality and range queries."),
    ("d4",  "Use exponential backoff with jitter when retrying failed HTTP requests."),
    ("d5",  "A memory leak occurs when allocated objects are never released back to the heap."),
    ("d6",  "JSON Web Tokens encode claims and are signed to verify their authenticity."),
    ("d7",  "Streaming reads a file in fixed-size chunks so it never loads fully into RAM."),
    ("d8",  "Database migrations should be reversible; always provide a down migration."),
    ("d9",  "Gzip compression reduces payload size for text responses over HTTP."),
    ("d10", "Connection pooling reuses open database connections to cut latency."),
    ("d11", "The bakery's sourdough needs a 24-hour cold proof in the refrigerator."),
    ("d12", "Mount Everest is the highest mountain above sea level on Earth."),
    ("d13", "Photosynthesis converts sunlight, water, and CO2 into glucose and oxygen."),
    ("d14", "The 2008 financial crisis was triggered by subprime mortgage defaults."),
    ("d15", "Regular stretching improves flexibility and reduces injury risk."),
    ("d16", "Use prepared statements to prevent SQL injection attacks."),
    ("d17", "Caching frequently-read values in Redis lowers load on the primary database."),
    ("d18", "A load balancer distributes incoming traffic across multiple servers."),
    ("d19", "TLS encrypts data in transit between the client and the server."),
    ("d20", "Pagination returns results in pages to avoid huge single responses."),
]
RETRIEVAL_QUERIES = [
    ("q1", "How do I change my account password?",            ["d1"]),
    ("q2", "What happens if I send too many API requests?",   ["d2"]),
    ("q3", "How can I make my SQL lookups faster?",            ["d3", "d17"]),
    ("q4", "What's the right way to retry a failed request?", ["d4"]),
    ("q5", "Why does my service keep using more and more memory?", ["d5"]),
    ("q6", "How are auth tokens verified?",                    ["d6", "d19"]),
    ("q7", "How do I process a file too big for memory?",      ["d7"]),
    ("q8", "How do I protect against SQL injection?",          ["d16"]),
    ("q9", "How do I reduce HTTP response size?",              ["d9"]),
    ("q10", "How do I spread traffic across servers?",         ["d18"]),
]

# ── Clustering: labelled sentences across well-separated topics. ────────────────
CLUSTERING = [
    # databases
    ("Add an index to speed up the WHERE clause.", "databases"),
    ("The query planner chose a sequential scan.", "databases"),
    ("Normalize the schema to avoid duplicate rows.", "databases"),
    ("A foreign key enforces referential integrity.", "databases"),
    ("Vacuum reclaims space from dead tuples.", "databases"),
    ("Joins combine rows from two related tables.", "databases"),
    # cooking
    ("Whisk the eggs before folding in the flour.", "cooking"),
    ("Let the dough rest for thirty minutes.", "cooking"),
    ("Sear the steak on high heat for two minutes.", "cooking"),
    ("Season the soup with salt and fresh thyme.", "cooking"),
    ("Preheat the oven to 200 degrees Celsius.", "cooking"),
    ("Caramelize the onions slowly over low heat.", "cooking"),
    # astronomy
    ("The telescope captured a distant spiral galaxy.", "astronomy"),
    ("A solar eclipse occurs when the Moon blocks the Sun.", "astronomy"),
    ("Jupiter is the largest planet in the solar system.", "astronomy"),
    ("Light from that star took millions of years to arrive.", "astronomy"),
    ("Black holes warp spacetime around them.", "astronomy"),
    ("The rover collected samples from the Martian surface.", "astronomy"),
    # finance
    ("The central bank raised interest rates again.", "finance"),
    ("Diversify the portfolio to spread risk.", "finance"),
    ("Quarterly earnings beat analyst expectations.", "finance"),
    ("Inflation eroded the currency's purchasing power.", "finance"),
    ("The bond yield rose after the announcement.", "finance"),
    ("Investors moved capital into safer assets.", "finance"),
    # fitness
    ("Do three sets of ten squats with good form.", "fitness"),
    ("Stretch your hamstrings after a long run.", "fitness"),
    ("Cardio raises your heart rate for endurance.", "fitness"),
    ("Rest days let muscles recover and grow.", "fitness"),
    ("Stay hydrated during a high-intensity workout.", "fitness"),
    ("Proper posture prevents back injuries when lifting.", "fitness"),
]


def main():
    (HERE / "sts.json").write_text(json.dumps(
        {"source": "seed (curated)",
         "pairs": [{"a": a, "b": b, "score": s} for a, b, s in STS]}, indent=2))
    (HERE / "triplet.json").write_text(json.dumps(
        {"source": "seed (curated)",
         "triplets": [{"anchor": a, "positive": p, "negative": n} for a, p, n in TRIPLET]}, indent=2))
    (HERE / "retrieval.json").write_text(json.dumps(
        {"source": "seed (curated)",
         "docs":    [{"id": i, "text": t} for i, t in RETRIEVAL_DOCS],
         "queries": [{"id": i, "text": t, "relevant": r} for i, t, r in RETRIEVAL_QUERIES]}, indent=2))
    (HERE / "clustering.json").write_text(json.dumps(
        {"source": "seed (curated)",
         "items": [{"text": t, "label": l} for t, l in CLUSTERING]}, indent=2))

    print(f"Wrote seed datasets to {HERE}/")
    print(f"  sts.json        {len(STS)} pairs")
    print(f"  triplet.json    {len(TRIPLET)} triplets")
    print(f"  retrieval.json  {len(RETRIEVAL_DOCS)} docs / {len(RETRIEVAL_QUERIES)} queries")
    print(f"  clustering.json {len(CLUSTERING)} sentences / {len(set(l for _, l in CLUSTERING))} clusters")
    print("\nFor the standard STS-B slice, run: python3 suites/embedding/fetch.py")


if __name__ == "__main__":
    main()
