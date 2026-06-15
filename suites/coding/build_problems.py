#!/usr/bin/env python3
"""
BenchLLAMA — Battery E coding problem set builder.

Writes problems.json (consumed by aptitude.py Battery E). Python-only
discriminating core — the subset that actually *separates* local coders:

  E1 generate_basic        generation to an exact signature, edge-case checks
  E2 debug_fix             read + repair a subtle bug (off-by-one / input
                           mutation / bad base case)
  E5 test_writing          write a real test suite — MUTATION-graded (kill rate)
  E7 instruction_adherence honor hard constraints (line cap, no extra defs,
                           stdlib-only) AND stay correct

DRAFT for OllamaMCP review (the consumer validates Battery E against real
`local_code` usage before it locks — see feedback_consumer_validates_battery).
Difficulty is calibrated to what small local models realistically deliver, not
Leetcode puzzles.

Every problem carries a reference solution / reference tests used ONLY to
self-validate the set at build time — they are never shown to a model. Run:

  python3 suites/coding/build_problems.py            # build + validate + write
  python3 suites/coding/build_problems.py --check    # validate only, no write
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import harness as H  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "problems.json"


def fence(code, lang="python"):
    return f"```{lang}\n" + code + "\n```"


# ── E1 — generate_basic ────────────────────────────────────────────────────────

E1 = [
    {
        "id": "e1_rle",
        "category": "E1",
        "entry": "run_length_encode",
        "prompt": (
            "Write a Python function `run_length_encode(s)` that returns a list of "
            "(character, count) tuples for each run of consecutive equal characters "
            "in the string `s`.\n"
            "Example: 'aaabbc' -> [('a', 3), ('b', 2), ('c', 1)]. "
            "The empty string returns []."
        ),
        "gate": {"require_symbol": "run_length_encode"},
        "reference": (
            "def run_length_encode(s):\n"
            "    out = []\n"
            "    for ch in s:\n"
            "        if out and out[-1][0] == ch:\n"
            "            out[-1] = (ch, out[-1][1] + 1)\n"
            "        else:\n"
            "            out.append((ch, 1))\n"
            "    return out\n"
        ),
        "checks": [
            "assert run_length_encode('') == []",
            "assert run_length_encode('a') == [('a', 1)]",
            "assert run_length_encode('aaabbc') == [('a', 3), ('b', 2), ('c', 1)]",
            "assert run_length_encode('abc') == [('a', 1), ('b', 1), ('c', 1)]",
            "assert run_length_encode('aaaa') == [('a', 4)]",
            "assert run_length_encode('aabbaa') == [('a', 2), ('b', 2), ('a', 2)]",
        ],
    },
    {
        "id": "e1_most_frequent",
        "category": "E1",
        "entry": "most_frequent",
        "prompt": (
            "Write a Python function `most_frequent(xs)` returning the element that "
            "appears most often in the list `xs`. If several elements tie for the "
            "highest count, return the smallest of them."
        ),
        "gate": {"require_symbol": "most_frequent"},
        "reference": (
            "from collections import Counter\n"
            "def most_frequent(xs):\n"
            "    c = Counter(xs)\n"
            "    return min(c, key=lambda k: (-c[k], k))\n"
        ),
        "checks": [
            "assert most_frequent([1, 1, 2, 2, 3]) == 1",
            "assert most_frequent([4, 4, 4, 2, 2]) == 4",
            "assert most_frequent([7]) == 7",
            "assert most_frequent([5, 3, 5, 3, 1]) == 3",
            "assert most_frequent([-1, -1, -2]) == -1",
            "assert most_frequent([2, 2, 1, 1, 3, 3]) == 1",
        ],
    },
    {
        "id": "e1_balanced",
        "category": "E1",
        "entry": "is_balanced",
        "prompt": (
            "Write a Python function `is_balanced(s)` that returns True if every "
            "bracket in the string `s` is correctly matched and nested, considering "
            "the three bracket types (), [], and {}. Other characters are ignored. "
            "The empty string is balanced."
        ),
        "gate": {"require_symbol": "is_balanced"},
        "reference": (
            "def is_balanced(s):\n"
            "    pairs = {')': '(', ']': '[', '}': '{'}\n"
            "    stack = []\n"
            "    for ch in s:\n"
            "        if ch in '([{':\n"
            "            stack.append(ch)\n"
            "        elif ch in ')]}':\n"
            "            if not stack or stack.pop() != pairs[ch]:\n"
            "                return False\n"
            "    return not stack\n"
        ),
        "checks": [
            "assert is_balanced('') == True",
            "assert is_balanced('()') == True",
            "assert is_balanced('([{}])') == True",
            "assert is_balanced('(]') == False",
            "assert is_balanced('(()') == False",
            "assert is_balanced('a(b)c[d]') == True",
            "assert is_balanced(')(') == False",
        ],
    },
]


# ── E2 — debug_fix ─────────────────────────────────────────────────────────────

def _e2(id, entry, desc, buggy, reference, checks):
    return {
        "id": id, "category": "E2", "entry": entry,
        "prompt": (desc + "\n\n" + fence(buggy) +
                   "\n\nReturn the corrected function with the same name and signature."),
        "buggy": buggy,
        "gate": {"require_symbol": entry},
        "reference": reference,
        "checks": checks,
    }


E2 = [
    _e2(
        "e2_binary_search", "binary_search",
        "The function below should return the index of `target` in the sorted list "
        "`xs`, or -1 if it is absent. It has a bug — fix it.",
        "def binary_search(xs, target):\n"
        "    lo, hi = 0, len(xs) - 1\n"
        "    while lo < hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if xs[mid] == target:\n"
        "            return mid\n"
        "        elif xs[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1\n",
        "def binary_search(xs, target):\n"
        "    lo, hi = 0, len(xs) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if xs[mid] == target:\n"
        "            return mid\n"
        "        elif xs[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1\n",
        [
            "assert binary_search([1, 2, 3, 4, 5], 1) == 0",
            "assert binary_search([1, 2, 3, 4, 5], 5) == 4",
            "assert binary_search([1, 2, 3, 4, 5], 3) == 2",
            "assert binary_search([1, 2, 3, 4, 5], 6) == -1",
            "assert binary_search([], 1) == -1",
            "assert binary_search([7], 7) == 0",
        ],
    ),
    _e2(
        "e2_running_max", "running_max",
        "The function below should return a new list of the running maxima of `xs` "
        "(each element is the largest value seen so far). It must NOT modify the "
        "caller's input list. It has a bug — fix it.",
        "def running_max(xs):\n"
        "    for i in range(1, len(xs)):\n"
        "        xs[i] = max(xs[i], xs[i - 1])\n"
        "    return xs\n",
        "def running_max(xs):\n"
        "    out = list(xs)\n"
        "    for i in range(1, len(out)):\n"
        "        out[i] = max(out[i], out[i - 1])\n"
        "    return out\n",
        [
            "assert running_max([1, 3, 2, 5, 4]) == [1, 3, 3, 5, 5]",
            "assert running_max([]) == []",
            "assert running_max([5]) == [5]",
            "assert running_max([2, 2, 1]) == [2, 2, 2]",
            "src = [3, 1, 2]\nrunning_max(src)\nassert src == [3, 1, 2]",
        ],
    ),
    _e2(
        "e2_power_of_two", "is_power_of_two",
        "The function below should return True if the integer `n` is a positive power "
        "of two (1, 2, 4, 8, ...), else False. It mishandles some inputs — fix it.",
        "def is_power_of_two(n):\n"
        "    if n == 1:\n"
        "        return True\n"
        "    if n % 2 != 0:\n"
        "        return False\n"
        "    return is_power_of_two(n // 2)\n",
        "def is_power_of_two(n):\n"
        "    if n <= 0:\n"
        "        return False\n"
        "    if n == 1:\n"
        "        return True\n"
        "    if n % 2 != 0:\n"
        "        return False\n"
        "    return is_power_of_two(n // 2)\n",
        [
            "assert is_power_of_two(1) == True",
            "assert is_power_of_two(2) == True",
            "assert is_power_of_two(8) == True",
            "assert is_power_of_two(1024) == True",
            "assert is_power_of_two(6) == False",
            "assert is_power_of_two(0) == False",
            "assert is_power_of_two(-4) == False",
        ],
    ),
]


# ── E5 — test_writing (mutation-graded) ────────────────────────────────────────

def _e5(id, entry, desc, clean, mutants, reference_tests):
    return {
        "id": id, "category": "E5", "entry": entry,
        "prompt": (
            f"Write a thorough test suite for a Python function `{entry}` described "
            f"below. {desc}\n\n"
            "Write ONLY test functions named `test_*` that use plain `assert` "
            "statements (no pytest, no imports, no test framework). Assume "
            f"`{entry}` is already defined and in scope — do not define it yourself. "
            "Cover normal cases, edge cases, and boundaries."
        ),
        "clean_impl": clean,
        "mutants": mutants,
        "reference_tests": reference_tests,   # build-time validation only
    }


E5 = [
    _e5(
        "e5_clamp", "clamp",
        "`clamp(x, lo, hi)` returns `x` constrained to the inclusive range "
        "[lo, hi]: `lo` if x < lo, `hi` if x > hi, otherwise `x`.",
        "def clamp(x, lo, hi):\n"
        "    if x < lo:\n        return lo\n"
        "    if x > hi:\n        return hi\n"
        "    return x\n",
        [
            "def clamp(x, lo, hi):\n    if x < lo:\n        return lo\n    return x\n",      # ignores hi
            "def clamp(x, lo, hi):\n    return x\n",                                          # no clamping
            "def clamp(x, lo, hi):\n    if x > lo:\n        return lo\n    if x < hi:\n        return hi\n    return x\n",  # swapped logic
        ],
        [
            "def test_within():\n    assert clamp(5, 0, 10) == 5",
            "def test_below():\n    assert clamp(-3, 0, 10) == 0",
            "def test_above():\n    assert clamp(15, 0, 10) == 10",
            "def test_edges():\n    assert clamp(0, 0, 10) == 0 and clamp(10, 0, 10) == 10",
        ],
    ),
    _e5(
        "e5_count_vowels", "count_vowels",
        "`count_vowels(s)` returns the number of vowels (a, e, i, o, u — case-insensitive) "
        "in the string `s`. 'y' is NOT a vowel.",
        "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n",
        [
            "def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiou')\n",          # misses uppercase
            "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiouy')\n",  # counts y
            "def count_vowels(s):\n    return sum(1 for c in s.lower() if c not in 'aeiou')\n",  # consonants
        ],
        [
            "def test_basic():\n    assert count_vowels('hello') == 2",
            "def test_upper():\n    assert count_vowels('AEIOU') == 5",
            "def test_empty():\n    assert count_vowels('') == 0",
            "def test_none():\n    assert count_vowels('rhythm') == 0",
            "def test_mixed():\n    assert count_vowels('Apple') == 2",
        ],
    ),
    _e5(
        "e5_is_palindrome", "is_palindrome",
        "`is_palindrome(s)` returns True iff the string `s` reads identically forwards and "
        "backwards (exact, case-sensitive comparison). The empty string is a palindrome.",
        "def is_palindrome(s):\n    return s == s[::-1]\n",
        [
            "def is_palindrome(s):\n    return True\n",                       # always true
            "def is_palindrome(s):\n    return s != s[::-1]\n",               # negated
            "def is_palindrome(s):\n    return s.upper() == s[::-1]\n",       # breaks lowercase palindromes
        ],
        [
            "def test_pal():\n    assert is_palindrome('racecar') is True",
            "def test_not():\n    assert is_palindrome('abc') is False",
            "def test_empty():\n    assert is_palindrome('') is True",
            "def test_two():\n    assert is_palindrome('ab') is False",
            "def test_odd():\n    assert is_palindrome('aba') is True",
        ],
    ),
    _e5(
        "e5_fib", "fib",
        "`fib(n)` returns the n-th Fibonacci number, 0-indexed: fib(0)=0, fib(1)=1, "
        "fib(n)=fib(n-1)+fib(n-2).",
        "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
        [
            "def fib(n):\n    a, b = 1, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",   # wrong seed
            "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a\n    return a\n",       # wrong recurrence
            "def fib(n):\n    a, b = 0, 1\n    for _ in range(n - 1):\n        a, b = b, a + b\n    return a\n",  # off-by-one
        ],
        [
            "def test_zero():\n    assert fib(0) == 0",
            "def test_one():\n    assert fib(1) == 1",
            "def test_two():\n    assert fib(2) == 1",
            "def test_six():\n    assert fib(6) == 8",
            "def test_ten():\n    assert fib(10) == 55",
        ],
    ),
    _e5(
        "e5_merge_sorted", "merge_sorted",
        "`merge_sorted(a, b)` merges two already-sorted lists `a` and `b` into a "
        "single sorted list containing all elements (duplicates kept).",
        "def merge_sorted(a, b):\n"
        "    i = j = 0\n    out = []\n"
        "    while i < len(a) and j < len(b):\n"
        "        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n"
        "        else:\n            out.append(b[j]); j += 1\n"
        "    out.extend(a[i:]); out.extend(b[j:])\n"
        "    return out\n",
        [
            # drops the tail (no extend)
            "def merge_sorted(a, b):\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    return out\n",
            # reversed comparison
            "def merge_sorted(a, b):\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n        if a[i] >= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    out.extend(a[i:]); out.extend(b[j:])\n    return out\n",
            # off-by-one bound → IndexError on some inputs
            "def merge_sorted(a, b):\n    i = j = 0\n    out = []\n    while i <= len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    out.extend(a[i:]); out.extend(b[j:])\n    return out\n",
        ],
        [
            "def test_basic():\n    assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]",
            "def test_empty():\n    assert merge_sorted([], [1, 2]) == [1, 2] and merge_sorted([1, 2], []) == [1, 2]",
            "def test_dups():\n    assert merge_sorted([1, 1], [1, 2]) == [1, 1, 1, 2]",
            "def test_uneven():\n    assert merge_sorted([1, 2, 3, 9], [4]) == [1, 2, 3, 4, 9]",
        ],
    ),
]


# ── E7 — instruction_adherence (hard constraints + correctness) ────────────────

E7 = [
    {
        "id": "e7_dedupe",
        "category": "E7",
        "entry": "dedupe",
        "prompt": (
            "Implement `dedupe(xs)` that returns the elements of list `xs` with "
            "duplicates removed, preserving first-seen order.\n"
            "HARD CONSTRAINTS (these are graded):\n"
            "  • at most 5 lines of code\n"
            "  • no helper functions or extra top-level definitions\n"
            "  • standard library only\n"
            "Return only the function."
        ),
        "gate": {"require_symbol": "dedupe", "max_lines": 5, "forbid_extra_defs": True},
        "reference": "def dedupe(xs):\n    return list(dict.fromkeys(xs))\n",
        "checks": [
            "assert dedupe([1, 2, 1, 3, 2]) == [1, 2, 3]",
            "assert dedupe([]) == []",
            "assert dedupe([5, 5, 5]) == [5]",
            "assert dedupe(['a', 'b', 'a']) == ['a', 'b']",
            "assert dedupe([3, 1, 2]) == [3, 1, 2]",
        ],
    },
    {
        "id": "e7_fizzbuzz",
        "category": "E7",
        "entry": "fizzbuzz",
        "prompt": (
            "Implement `fizzbuzz(n)` returning a list of strings for the integers "
            "1..n: 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' "
            "for multiples of both, otherwise the number as a string.\n"
            "HARD CONSTRAINTS (these are graded):\n"
            "  • at most 8 lines of code\n"
            "  • no helper functions or extra top-level definitions\n"
            "Return only the function."
        ),
        "gate": {"require_symbol": "fizzbuzz", "max_lines": 8, "forbid_extra_defs": True},
        "reference": (
            "def fizzbuzz(n):\n"
            "    out = []\n"
            "    for i in range(1, n + 1):\n"
            "        s = ('Fizz' if i % 3 == 0 else '') + ('Buzz' if i % 5 == 0 else '')\n"
            "        out.append(s or str(i))\n"
            "    return out\n"
        ),
        "checks": [
            "assert fizzbuzz(1) == ['1']",
            "assert fizzbuzz(3) == ['1', '2', 'Fizz']",
            "assert fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']",
            "assert fizzbuzz(15)[-1] == 'FizzBuzz'",
            "assert fizzbuzz(0) == []",
            "assert len(fizzbuzz(15)) == 15",
        ],
    },
    {
        "id": "e7_chunk",
        "category": "E7",
        "entry": "chunk",
        "prompt": (
            "Implement `chunk(xs, n)` that splits list `xs` into consecutive "
            "sublists of length `n` (the final sublist may be shorter).\n"
            "HARD CONSTRAINTS (these are graded):\n"
            "  • at most 6 lines of code\n"
            "  • no helper functions or extra top-level definitions\n"
            "  • standard library only\n"
            "Return only the function."
        ),
        "gate": {"require_symbol": "chunk", "max_lines": 6, "forbid_extra_defs": True},
        "reference": "def chunk(xs, n):\n    return [xs[i:i + n] for i in range(0, len(xs), n)]\n",
        "checks": [
            "assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]",
            "assert chunk([], 3) == []",
            "assert chunk([1, 2, 3], 1) == [[1], [2], [3]]",
            "assert chunk([1, 2, 3, 4], 4) == [[1, 2, 3, 4]]",
            "assert chunk([1, 2, 3, 4, 5, 6], 3) == [[1, 2, 3], [4, 5, 6]]",
        ],
    },
]

# ── E3 — multi_language (JavaScript / SQL / PHP · generate + debug) ─────────────
# Valid target (generate) and broken input (debug) per language. JS uses an
# assert(eq(...)) harness; PHP uses _eq(...) boolean checks; SQL is graded by exact
# row-set match against a trusted setup. Python is already covered by E1/E2.

E3 = [
    # ---- JavaScript ----
    {
        "id": "e3_js_chunk", "category": "E3", "lang": "javascript", "subtype": "generate",
        "entry": "chunk_array",
        "prompt": (
            "Write a JavaScript function `chunk_array(arr, n)` that splits the array "
            "`arr` into consecutive subarrays each of length `n` (the last may be "
            "shorter). Example: chunk_array([1,2,3,4,5], 2) -> [[1,2],[3,4],[5]]. "
            "Return only the function."
        ),
        "gate": {"require_symbol": "chunk_array"},
        "reference": ("function chunk_array(arr, n){\n"
                      "  const out = [];\n"
                      "  for (let i = 0; i < arr.length; i += n) out.push(arr.slice(i, i + n));\n"
                      "  return out;\n}"),
        "checks": [
            "assert(eq(chunk_array([1,2,3,4,5],2),[[1,2],[3,4],[5]]))",
            "assert(eq(chunk_array([],3),[]))",
            "assert(eq(chunk_array([1,2,3],1),[[1],[2],[3]]))",
            "assert(eq(chunk_array([1,2,3,4],4),[[1,2,3,4]]))",
        ],
    },
    {
        "id": "e3_js_sum_even", "category": "E3", "lang": "javascript", "subtype": "debug",
        "entry": "sum_even",
        "prompt": ("The JavaScript function below should return the sum of the even "
                   "numbers in `arr`, but it returns the wrong total. Fix it."),
        "buggy": ("function sum_even(arr){\n"
                  "  let total = 0;\n"
                  "  for (let i = 0; i < arr.length; i++){\n"
                  "    if (arr[i] % 2 === 1) total += arr[i];\n"
                  "  }\n  return total;\n}"),
        "reference": ("function sum_even(arr){\n"
                      "  let total = 0;\n"
                      "  for (let i = 0; i < arr.length; i++){\n"
                      "    if (arr[i] % 2 === 0) total += arr[i];\n"
                      "  }\n  return total;\n}"),
        "gate": {"require_symbol": "sum_even"},
        "checks": [
            "assert(eq(sum_even([1,2,3,4]),6))",
            "assert(eq(sum_even([]),0))",
            "assert(eq(sum_even([2,4,6]),12))",
            "assert(eq(sum_even([1,3,5]),0))",
            "assert(eq(sum_even([-2,-1,2]),0))",
        ],
    },
    # ---- SQL ----
    {
        "id": "e3_sql_totals", "category": "E3", "lang": "sql", "subtype": "generate",
        "prompt": (
            "Given a table `sales(region TEXT, amount INTEGER)`, write a SQL query "
            "that returns each region with its total amount, ordered by total "
            "descending. Output columns: region, total."
        ),
        "setup": ("CREATE TABLE sales(region TEXT, amount INTEGER);"
                  "INSERT INTO sales VALUES ('west',100),('east',50),('west',30),('east',40);"),
        "reference": "SELECT region, SUM(amount) AS total FROM sales GROUP BY region ORDER BY total DESC",
        "expected": [["west", 130], ["east", 90]],
    },
    {
        "id": "e3_sql_filter", "category": "E3", "lang": "sql", "subtype": "debug",
        "prompt": (
            "The query below should return the names of employees in the 'eng' "
            "department who earn more than 100. It returns the wrong rows — fix it.\n\n"
            + fence("SELECT name FROM emp WHERE dept='eng' OR salary > 100", "sql")
            + "\n\nReturn the corrected query."
        ),
        "setup": ("CREATE TABLE emp(name TEXT, dept TEXT, salary INTEGER);"
                  "INSERT INTO emp VALUES ('Ann','eng',120),('Bob','eng',90),('Cal','sales',150);"),
        "buggy": "SELECT name FROM emp WHERE dept='eng' OR salary > 100",
        "reference": "SELECT name FROM emp WHERE dept='eng' AND salary > 100",
        "expected": [["Ann"]],
    },
    # ---- PHP ----
    {
        "id": "e3_php_word_count", "category": "E3", "lang": "php", "subtype": "generate",
        "entry": "word_count",
        "prompt": (
            "Write a PHP function `word_count($s)` that returns the number of "
            "whitespace-separated words in the string `$s`. An empty or "
            "whitespace-only string returns 0. Return only the function."
        ),
        "gate": {"require_symbol": "word_count"},
        "reference": ("function word_count($s){\n"
                      "  $t = trim($s);\n"
                      "  if ($t === '') return 0;\n"
                      "  return count(preg_split('/\\s+/', $t));\n}"),
        "checks": [
            "_eq(word_count('hello world'),2)",
            "_eq(word_count(''),0)",
            "_eq(word_count('   '),0)",
            "_eq(word_count('one two three'),3)",
            "_eq(word_count('  spaced  out  '),2)",
        ],
    },
    {
        "id": "e3_php_factorial", "category": "E3", "lang": "php", "subtype": "debug",
        "entry": "factorial",
        "prompt": ("The PHP function below should return the factorial of a "
                   "non-negative integer `$n` (with 0! = 1), but it is wrong. Fix it."),
        "buggy": ("function factorial($n){\n"
                  "  $r = 1;\n"
                  "  for ($i = 1; $i < $n; $i++){\n"
                  "    $r *= $i;\n"
                  "  }\n  return $r;\n}"),
        "reference": ("function factorial($n){\n"
                      "  $r = 1;\n"
                      "  for ($i = 1; $i <= $n; $i++){\n"
                      "    $r *= $i;\n"
                      "  }\n  return $r;\n}"),
        "gate": {"require_symbol": "factorial"},
        "checks": [
            "_eq(factorial(0),1)",
            "_eq(factorial(1),1)",
            "_eq(factorial(5),120)",
            "_eq(factorial(3),6)",
            "_eq(factorial(6),720)",
        ],
    },
]

# E2/E3 debug problems show the broken code in the prompt.
for _p in E3:
    if _p.get("subtype") == "debug" and "buggy" in _p and "```" not in _p["prompt"]:
        _lang = "javascript" if _p["lang"] == "javascript" else _p["lang"]
        _p["prompt"] = (_p["prompt"] + "\n\n" + fence(_p["buggy"], _lang) +
                        "\n\nReturn the corrected code with the same name and signature.")

# ── E9 — markup_quality (HTML / CSS) — objective rubric, not execution ──────────
# Generation of clean, valid, complete markup: validity gate (tidy / tinycss2) +
# required-selector coverage (no skipping) + cleanliness (node budget, forbidden
# junk, unused/missing CSS rules). A core local_code workload (templates/pages).

E9 = [
    {
        "id": "e9_html_contact", "category": "E9", "lang": "html", "kind": "html",
        "subtype": "generate",
        "prompt": (
            "Write a complete, valid HTML5 document for a contact page. It must include: "
            "a <header> with an <h1>; a <nav> with exactly 3 links; a <main> containing a "
            "<form> with a <label> and an email input (<input type=\"email\">) and a submit "
            "button (<button type=\"submit\">); and a <footer>. Keep it clean and semantic "
            "— no inline style attributes, no <script>, no tables for layout, no decorative "
            "filler. Output only the HTML."
        ),
        "require_selectors": [["header h1", 1], ["nav a", 3],
                              ["main form input[type=email]", 1],
                              ["main form label", ">=1"],
                              ["button[type=submit]", 1], ["footer", 1]],
        "max_nodes": 32,
        "forbid_selectors": ["script", "[style]", "center", "font", "marquee", "table"],
        "reference": (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head><meta charset=\"utf-8\">"
            "<title>Contact</title></head>\n<body>\n"
            "<header><h1>Contact Us</h1></header>\n"
            "<nav><a href=\"#home\">Home</a><a href=\"#about\">About</a>"
            "<a href=\"#contact\">Contact</a></nav>\n<main>\n<form>\n"
            "<label for=\"email\">Email</label>\n"
            "<input type=\"email\" id=\"email\" name=\"email\">\n"
            "<button type=\"submit\">Send</button>\n</form>\n</main>\n"
            "<footer><p>&copy; 2026</p></footer>\n</body>\n</html>\n"
        ),
    },
    {
        "id": "e9_html_cards", "category": "E9", "lang": "html", "kind": "html",
        "subtype": "generate",
        "prompt": (
            "Write a complete, valid HTML5 document showing a list of exactly 3 product "
            "cards. Use a <ul> where each <li> contains an <article class=\"card\"> with an "
            "<h2> (product name) and a <p> (price). No inline style attributes, no <script>, "
            "no tables. Output only the HTML."
        ),
        # Graduated: core structure is anchored on `.card` (any container element) so a
        # reasonable `li.card`/`div.card` choice still earns most of the coverage; the
        # exact semantic `article.card` is ONE additional point, not all-or-nothing.
        "require_selectors": [["ul li", 3], [".card", 3], [".card h2", 3],
                              [".card p", 3], ["article.card", 3]],
        "max_nodes": 40,
        "forbid_selectors": ["script", "[style]", "center", "font", "table"],
        "reference": (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head><meta charset=\"utf-8\">"
            "<title>Products</title></head>\n<body>\n<main>\n<ul>\n"
            "<li><article class=\"card\"><h2>Widget</h2><p>$9.99</p></article></li>\n"
            "<li><article class=\"card\"><h2>Gadget</h2><p>$19.99</p></article></li>\n"
            "<li><article class=\"card\"><h2>Gizmo</h2><p>$29.99</p></article></li>\n"
            "</ul>\n</main>\n</body>\n</html>\n"
        ),
    },
    {
        "id": "e9_css_components", "category": "E9", "lang": "css", "kind": "css",
        "subtype": "generate",
        "html_context": (
            "<div id=\"hero\"><h1 class=\"title\">Welcome</h1></div>"
            "<div class=\"card\"><p>Body</p><button class=\"btn\">Buy</button></div>"
        ),
        "css_required_selectors": [".card", ".btn", "#hero"],
        "prompt": (
            "Write a CSS stylesheet for the HTML below. Define rules for exactly these "
            "three selectors: `.card` (padding and a border), `.btn` (a background colour "
            "and padding), and `#hero` (a larger font-size). Do not add rules for anything "
            "else — no unused or empty rules. Output only the CSS.\n\n"
            + fence("<div id=\"hero\"><h1 class=\"title\">Welcome</h1></div>\n"
                    "<div class=\"card\"><p>Body</p><button class=\"btn\">Buy</button></div>", "html")
        ),
        "reference": ("#hero { font-size: 2rem; }\n"
                      ".card { padding: 16px; border: 1px solid #ccc; }\n"
                      ".btn { background: #0066cc; padding: 8px 12px; }\n"),
    },
]

PROBLEMS = E1 + E2 + E5 + E7 + E3 + E9


# ── self-validation against the harness ────────────────────────────────────────

def _spec(p):
    g = p.get("gate", {})
    return H.GateSpec(
        require_symbol=g.get("require_symbol"),
        allow=g.get("allow", ()),
        max_lines=g.get("max_lines"),
        forbid_extra_defs=g.get("forbid_extra_defs", False),
    )


def validate():
    errs = []
    for p in PROBLEMS:
        cat, pid = p["category"], p["id"]
        if cat in ("E1", "E2", "E7"):
            # reference must clear the gate and pass every hidden check
            g = H.gate(p["reference"], _spec(p))
            if not g.ok:
                errs.append(f"{pid}: reference fails gate ({g.reason} {g.details})")
                continue
            r = H.run_checks(p["reference"], p["checks"],
                             allowed_imports=_spec(p).allowed_imports, timeout=5)
            if r.error or r.passed != r.total:
                errs.append(f"{pid}: reference {r.passed}/{r.total} err={r.error} {r.detail}")
            # E2 sanity: the buggy original must FAIL at least one check (else no bug)
            if cat == "E2":
                rb = H.run_checks(p["buggy"], p["checks"],
                                  allowed_imports=_spec(p).allowed_imports, timeout=5)
                if rb.error is None and rb.passed == rb.total:
                    errs.append(f"{pid}: buggy original passes all checks — not a real bug")
        elif cat == "E3":
            lang = p["lang"]
            if not H.runtime_available(lang):
                print(f"  ⚠ {pid}: runtime for {lang} unavailable — skipping validation "
                      f"(battery will graceful-skip too)")
                continue
            if lang == "sql":
                g = H.gate_sql(p["reference"])
                if not g.ok:
                    errs.append(f"{pid}: reference fails gate_sql ({g.reason})")
                r = H.run_sql_query(p["setup"], p["reference"], p["expected"], timeout=5)
                if r.error or r.passed != 1:
                    errs.append(f"{pid}: reference sql {r.passed}/1 err={r.error} {r.detail}")
                if p.get("subtype") == "debug":
                    rb = H.run_sql_query(p["setup"], p["buggy"], p["expected"], timeout=5)
                    if rb.error is None and rb.passed == 1:
                        errs.append(f"{pid}: buggy query already returns expected — not a real bug")
            else:  # javascript / php
                gate_fn = H.gate_js if lang in ("javascript", "js") else H.gate_php
                run_fn = H.run_js_checks if lang in ("javascript", "js") else H.run_php_checks
                g = gate_fn(p["reference"], _spec(p))
                if not g.ok:
                    errs.append(f"{pid}: reference fails gate ({g.reason} {g.details})")
                    continue
                r = run_fn(p["reference"], p["checks"], timeout=8)
                if r.error or r.passed != r.total:
                    errs.append(f"{pid}: reference {r.passed}/{r.total} err={r.error} {r.detail}")
                if p.get("subtype") == "debug":
                    rb = run_fn(p["buggy"], p["checks"], timeout=8)
                    if rb.error is None and rb.passed == rb.total:
                        errs.append(f"{pid}: buggy original passes all checks — not a real bug")
        elif cat == "E9":
            lang = p["lang"]
            if not H.runtime_available(lang):
                print(f"  ⚠ {pid}: markup tooling for {lang} unavailable — skipping validation")
                continue
            ref_resp = fence(p["reference"], lang)
            out = H.grade_markup(ref_resp, p)
            if out.get("score", 0) is None or out.get("score", 0) < 0.95:
                errs.append(f"{pid}: reference markup scored {out.get('score')} "
                            f"(gate={out.get('gate')}, coverage={out.get('coverage')})")
        elif cat == "E5":
            # reference tests must pass the clean impl and KILL every mutant
            rc = H.run_test_functions(p["clean_impl"], "\n".join(p["reference_tests"]), timeout=5)
            if rc.error or rc.passed != rc.total or rc.total == 0:
                errs.append(f"{pid}: reference tests on clean = {rc.passed}/{rc.total} err={rc.error}")
            for i, m in enumerate(p["mutants"]):
                rm = H.run_test_functions(m, "\n".join(p["reference_tests"]), timeout=5)
                killed = rm.error is not None or rm.passed < rm.total
                if not killed:
                    errs.append(f"{pid}: mutant #{i} survives reference tests — not distinguishable")
    return errs


def main():
    check_only = "--check" in sys.argv
    by_cat = {}
    for p in PROBLEMS:
        by_cat.setdefault(p["category"], 0)
        by_cat[p["category"]] += 1
    print(f"Battery E problem set — {len(PROBLEMS)} problems: " +
          ", ".join(f"{k}={v}" for k, v in sorted(by_cat.items())))
    print("Validating against harness...")
    errs = validate()
    if errs:
        print(f"\n✗ {len(errs)} validation error(s):")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    print("✓ all references pass gates + checks; all E2 bugs real; all E5 mutants killable")
    if check_only:
        return
    OUT.write_text(json.dumps(PROBLEMS, indent=2))
    print(f"→ wrote {OUT}")


if __name__ == "__main__":
    main()
