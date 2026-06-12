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

# ── Retrieval: HARD. Hard-negative distractors (lexical overlap with the query
#    but wrong) + paraphrase-gap queries (share no keywords with the answer) so
#    recall/nDCG/MRR separate models instead of saturating at 1.0. ─────────────
RETRIEVAL_DOCS = [
    # password / account / auth — heavy lexical overlap between these
    ("d1",  "To reset your password, open Settings → Security and choose 'Change password'."),
    ("d2",  "Account lockout policy: five failed login attempts trigger a 30-minute lock."),
    ("d3",  "To lock your account and block all sign-ins, use Privacy → Freeze account."),
    ("d4",  "Delete your account permanently under Settings → Danger Zone."),
    ("d5",  "Passwords must be at least 12 characters and include one symbol."),
    ("d6",  "Two-factor authentication sends a one-time code to your phone at each login."),
    # database performance
    ("d7",  "Add a B-tree index so the query planner skips a full table scan."),
    ("d8",  "Cache hot query results in Redis to avoid repeated database hits."),
    ("d9",  "Upgrade to an NVMe SSD to cut database file read latency."),
    ("d10", "Run VACUUM nightly to reclaim disk space from dead tuples."),
    ("d11", "Connection pooling reuses open sockets to lower per-request overhead."),
    ("d12", "Shard the table across nodes to spread write load."),
    # HTTP / API
    ("d13", "Back off exponentially and retry when you receive HTTP 429 (rate limited)."),
    ("d14", "A 500 Internal Server Error means an unhandled exception on the server."),
    ("d15", "Use gzip to compress large JSON response bodies over HTTP."),
    ("d16", "Paginate results with limit and offset to avoid returning huge payloads."),
    ("d17", "Set a 30-second client timeout to abort stalled requests."),
    # memory
    ("d18", "A memory leak is allocated objects that are never released back to the heap."),
    ("d19", "Stream a file in fixed-size chunks so it never fully loads into RAM."),
    ("d20", "Increase the JVM heap with -Xmx to allow larger allocations."),
    ("d21", "Take a heap dump to find which objects dominate retained memory."),
    # security
    ("d22", "Use parameterized queries to prevent SQL injection."),
    ("d23", "Hash passwords with bcrypt before storing them."),
    ("d24", "Escape HTML output to prevent cross-site scripting (XSS)."),
    ("d25", "TLS encrypts data in transit between the client and the server."),
    ("d26", "Validate and sanitize all user input on the server side."),
    ("d27", "A load balancer distributes incoming requests across multiple backend servers."),
    # off-topic filler (grows the corpus, lowers accidental hits)
    ("d28", "Mount Everest is the highest mountain above sea level on Earth."),
    ("d29", "The bakery's sourdough needs a 24-hour cold proof in the refrigerator."),
    ("d30", "Photosynthesis converts sunlight, water, and CO2 into glucose and oxygen."),
    ("d31", "The 2008 financial crisis was triggered by subprime mortgage defaults."),
    ("d32", "Regular stretching improves flexibility and reduces injury risk."),
    ("d33", "A solar eclipse occurs when the Moon passes between the Earth and the Sun."),
]
RETRIEVAL_QUERIES = [
    # paraphrase gap to the answer; hard negatives (shown) share surface words
    ("q1",  "I'm locked out and can't sign in — how do I get back in?",     ["d1"]),       # vs d2/d3 (lock*)
    ("q2",  "My read query is slow — how do I speed it up?",                ["d7", "d8"]), # vs d9/d10/d11/d12
    ("q3",  "The API keeps saying I've sent too many requests.",            ["d13"]),      # vs d14/d17
    ("q4",  "RAM usage climbs forever until the service crashes.",          ["d18", "d21"]), # vs d19/d20
    ("q5",  "Stop attackers from injecting SQL through my form.",           ["d22"]),      # vs d26/d24
    ("q6",  "Block other people from signing into my account while I'm away.", ["d3"]),    # vs d1/d2/d4
    ("q7",  "Shrink the size of my JSON API responses.",                    ["d15"]),      # vs d16
    ("q8",  "What's the safe way to store user passwords?",                 ["d23"]),      # vs d5/d22
    ("q9",  "Protect my web app against XSS.",                              ["d24"]),      # vs d26/d22
    ("q10", "Process a CSV far bigger than my available memory.",           ["d19"]),      # vs d18/d20
    ("q11", "Spread incoming web traffic across several servers.",          ["d27"]),      # vs d12/d11
    ("q12", "Keep the database from running out of disk over time.",        ["d10"]),      # vs d9/d7
]

# ── Clustering: HARD. Six ADJACENT sub-topics in two families (programming
#    languages; physical sciences) that share vocabulary, so centroid purity
#    drops below 1.0 and separates models. ────────────────────────────────────
CLUSTERING = [
    # Family A — programming languages (share function / variable / callback / heap)
    ("Build the sequence with a single list comprehension.", "python"),
    ("A decorator wraps a function to extend its behaviour.", "python"),
    ("The GIL serialises thread execution in CPython.", "python"),
    ("Use a virtual environment to isolate package dependencies.", "python"),
    ("Generators yield items lazily to save memory.", "python"),
    ("Unpack the tuple into several variables at once.", "python"),
    ("Promises handle asynchronous work without nested callbacks.", "javascript"),
    ("Prefer const and let over var for block scoping.", "javascript"),
    ("The event loop drains the callback queue on each tick.", "javascript"),
    ("Destructure the object to pull out named fields.", "javascript"),
    ("Arrow functions capture the surrounding this binding.", "javascript"),
    ("Use async and await to flatten promise chains.", "javascript"),
    ("The borrow checker enforces ownership at compile time.", "rust"),
    ("Propagate errors with Result and the question-mark operator.", "rust"),
    ("Lifetimes describe how long a reference stays valid.", "rust"),
    ("Pattern-match an enum with the match expression.", "rust"),
    ("Box moves a value onto the heap.", "rust"),
    ("Traits define shared behaviour across types.", "rust"),
    # Family B — physical sciences (share reaction / energy / cell / atom)
    ("A covalent bond shares electrons between two atoms.", "chemistry"),
    ("The reaction is exothermic and releases heat.", "chemistry"),
    ("Acids donate protons when dissolved in water.", "chemistry"),
    ("Balancing the equation conserves every atom.", "chemistry"),
    ("A catalyst lowers the activation energy of a reaction.", "chemistry"),
    ("The periodic table groups elements by electron shells.", "chemistry"),
    ("Mitochondria generate the ATP that powers the cell.", "biology"),
    ("DNA stores the instructions for building proteins.", "biology"),
    ("Enzymes catalyse reactions inside living cells.", "biology"),
    ("Natural selection favours traits that aid survival.", "biology"),
    ("The cell membrane controls what enters and leaves.", "biology"),
    ("Ribosomes translate messenger RNA into proteins.", "biology"),
    ("Force equals mass times acceleration.", "physics"),
    ("Energy is conserved within a closed system.", "physics"),
    ("A photon carries a quantum of electromagnetic energy.", "physics"),
    ("Momentum is the product of mass and velocity.", "physics"),
    ("Gravity curves spacetime around massive bodies.", "physics"),
    ("Friction converts kinetic energy into heat.", "physics"),
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
