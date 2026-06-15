#!/usr/bin/env python3
"""
BenchLLAMA — self-test for the coding grading harness.

Plain asserts, stdlib only (no pytest dep — the repo installs requests+textual).
Run:  python3 suites/coding/test_harness.py
Exits non-zero if any check fails. No model in the loop.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import harness as H  # noqa: E402

PASS, FAIL = 0, 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}  {extra}")


# ── extract_code ───────────────────────────────────────────────────────────────
print("\nextract_code")
check("python fence", H.extract_code("blah\n```python\nx = 1\n```\nend") == "x = 1")
check("untagged fence", H.extract_code("```\ny = 2\n```") == "y = 2")
check("no fence → whole text", H.extract_code("z = 3").strip() == "z = 3")
check("prefers lang tag",
      H.extract_code("```js\na\n```\n```python\nb\n```", lang="python") == "b")
check("empty → empty", H.extract_code("") == "")

# ── gate ───────────────────────────────────────────────────────────────────────
print("\ngate")
spec = H.GateSpec(require_symbol="f")
check("clean passes", H.gate("def f(x):\n    return x + 1", spec).ok)
check("syntax fails", H.gate("def f(:", spec).reason == "syntax")
check("missing symbol", H.gate("def g(x):\n    return x", spec).reason == "missing_symbol")
check("forbidden import os",
      H.gate("import os\ndef f(x):\n    return x", spec).reason == "forbidden_import")
check("forbidden from-import",
      H.gate("from subprocess import run\ndef f(x):\n    return x", spec).reason == "forbidden_import")
check("eval blocked",
      H.gate("def f(x):\n    return eval('x')", spec).reason == "forbidden_name")
check("__import__ blocked",
      H.gate("def f(x):\n    return __import__('os')", spec).reason == "forbidden_name")
check("dunder attr blocked",
      H.gate("def f(x):\n    return x.__class__.__bases__", spec).reason == "forbidden_attr")
check("allowed import ok", H.gate("import math\ndef f(x):\n    return math.sqrt(x)", spec).ok)
check("extra import via allow",
      H.gate("import numpy\ndef f(x):\n    return x", H.GateSpec(require_symbol="f", allow=["numpy"])).ok)

spec_strict = H.GateSpec(require_symbol="f", max_lines=2, forbid_extra_defs=True)
check("extra def blocked",
      H.gate("def helper():\n    return 1\ndef f():\n    return helper()", spec_strict).reason == "extra_defs")
check("line cap blocked",
      H.gate("def f():\n    a = 1\n    b = 2\n    return a + b", spec_strict).reason == "too_many_lines")
check("within cap ok", H.gate("def f():\n    return 1", spec_strict).ok)

# ── run_checks: correct vs buggy ──────────────────────────────────────────────
print("\nrun_checks")
good = "def add(a, b):\n    return a + b"
checks = ["assert add(2, 3) == 5", "assert add(-1, 1) == 0", "assert add(0, 0) == 0"]
r = H.run_checks(good, checks, timeout=5)
check("correct solution → 3/3", r.passed == 3 and r.total == 3, repr(r))

buggy = "def add(a, b):\n    return a - b"
r = H.run_checks(buggy, checks, timeout=5)
check("buggy solution → partial", r.passed < 3 and r.error is None, repr(r))

# regression: `from typing import ...` must not crash (guard blocked typing's
# internal `import sys`); whitelisted module + its transitive deps pre-loaded.
typed = "from typing import List, Tuple\ndef add(a: int, b: int) -> int:\n    return a + b"
r = H.run_checks(typed, checks, timeout=5)
check("typing import does NOT crash (regression)", r.passed == 3 and r.error is None, repr(r))

# regression: trailing demo code that raises must not sink the whole evaluation.
with_demo = good + "\nprint(add(1, 2))\nprint(add([], []))"   # 2nd demo would TypeError
r = H.run_checks(with_demo, checks, timeout=5)
check("trailing demo code stripped, not crashed (regression)", r.passed == 3 and r.error is None, repr(r))

# ── runtime import guard (defense-in-depth: bypasses the static gate) ──────────
print("\nruntime import guard")
sneaky = "import os\ndef leak():\n    return os.listdir('/')\nleak()"
r = H.run_checks(sneaky, ["assert True"], timeout=5)
check("runtime os import blocked", r.error is not None, repr(r))

dyn = "m = __import__('socket')\ndef f():\n    return m"
r = H.run_checks(dyn, ["assert True"], timeout=5)
check("runtime __import__ blocked", r.error is not None, repr(r))

# ── timeout (CPU spin must not hang the harness) ──────────────────────────────
print("\ntimeout")
spin = "def f():\n    pass\nwhile True:\n    pass"
import time as _t
t0 = _t.time()
r = H.run_checks(spin, ["assert True"], timeout=2)
elapsed = _t.time() - t0
check("infinite loop → error", r.error is not None, repr(r))
check("returns within ~timeout", elapsed < 8, f"elapsed={elapsed:.1f}s")

# ── run_test_functions: mutation grading ──────────────────────────────────────
print("\nrun_test_functions (mutation grading)")
clean = "def is_even(n):\n    return n % 2 == 0"
mutant = "def is_even(n):\n    return n % 2 == 1"   # inverted
model_tests = (
    "def test_even():\n    assert is_even(4)\n"
    "def test_odd():\n    assert not is_even(3)\n"
    "def test_zero():\n    assert is_even(0)\n"
)
r_clean = H.run_test_functions(clean, model_tests, timeout=5)
check("tests pass clean impl", r_clean.passed == 3 and r_clean.total == 3, repr(r_clean))
r_mut = H.run_test_functions(mutant, model_tests, timeout=5)
check("mutant is killed (≥1 test fails)", r_mut.passed < r_mut.total, repr(r_mut))

# regression: model redefines the function under test in its own test file — our
# supplied impl must still win (late binding), so mutation grading still works.
tests_with_def = "def is_even(n):\n    return n % 2 == 0\n" + model_tests
r_clean2 = H.run_test_functions(clean, tests_with_def, timeout=5)
check("redefine: tests still pass clean impl", r_clean2.passed == r_clean2.total and r_clean2.total >= 3, repr(r_clean2))
r_mut2 = H.run_test_functions(mutant, tests_with_def, timeout=5)
check("redefine: mutant STILL killed (impl wins, not model's def)", r_mut2.passed < r_mut2.total, repr(r_mut2))

# ── recover_parseable + per-test map (E5 rework) ──────────────────────────────
print("\nrecover_parseable + per-test")
check("clean code unchanged", H.recover_parseable("def f():\n    return 1") == "def f():\n    return 1")
truncated = "def test_a():\n    assert True\ndef test_b():\n    assert is_even(2)\ndef test_c(:"
rec = H.recover_parseable(truncated)
check("truncated trailing def recovered", "test_b" in rec and "test_c" not in rec, repr(rec))
# per-test map: each test_* reported individually
impl = "def is_even(n):\n    return n % 2 == 0"
mixed = ("def test_ok():\n    assert is_even(2)\n"
         "def test_bad():\n    assert is_even(3)\n"      # fails the clean impl (out-of-contract)
         "def test_zero():\n    assert is_even(0)\n")
r = H.run_test_functions(impl, mixed, timeout=5)
pt = r.detail.get("per_test", {})
check("per_test reports each test", pt.get("test_ok") is True and pt.get("test_bad") is False
      and pt.get("test_zero") is True, repr(pt))

# ── grade_generation: full path ───────────────────────────────────────────────
print("\ngrade_generation (end-to-end)")
resp = "Here is the function:\n```python\ndef square(n):\n    return n * n\n```"
out = H.grade_generation(resp, ["assert square(3) == 9", "assert square(0) == 0"],
                         H.GateSpec(require_symbol="square"), timeout=5)
check("full pass → score 1.0", out["score"] == 1.0, repr(out))

resp_bad = "```python\nimport os\ndef square(n):\n    return n * n\n```"
out = H.grade_generation(resp_bad, ["assert square(3) == 9"],
                         H.GateSpec(require_symbol="square"), timeout=5)
check("gate fail → score 0, no exec", out["score"] == 0.0 and out["exec"] is None, repr(out))

# ── SQL (stdlib sqlite3) ──────────────────────────────────────────────────────
print("\nSQL")
SETUP = ("CREATE TABLE emp(id INTEGER, name TEXT, dept TEXT, salary INTEGER);"
         "INSERT INTO emp VALUES (1,'Ann','eng',100),(2,'Bob','eng',120),(3,'Cal','sales',90);")
r = H.run_sql_query(SETUP, "SELECT name FROM emp WHERE dept='eng' ORDER BY salary DESC",
                    [["Bob"], ["Ann"]], timeout=5)
check("correct query matches rows", r.passed == 1 and r.error is None, repr(r))
r = H.run_sql_query(SETUP, "SELECT name FROM emp WHERE dept='eng'", [["Cal"]], timeout=5)
check("wrong result → 0", r.passed == 0 and r.error is None, repr(r))
check("gate blocks DROP", H.gate_sql("SELECT 1; DROP TABLE emp").reason == "forbidden_sql")
check("gate blocks ATTACH", H.gate_sql("ATTACH DATABASE 'x' AS y").reason == "forbidden_sql")
check("gate allows SELECT", H.gate_sql("SELECT name FROM emp ORDER BY id").ok)
g = H.grade_sql("```sql\nSELECT name FROM emp ORDER BY id\n```", SETUP,
                [["Ann"], ["Bob"], ["Cal"]], timeout=5)
check("grade_sql full path → 1.0", g["score"] == 1.0, repr(g))

# ── JavaScript (node + vm) ────────────────────────────────────────────────────
print(f"\nJavaScript (node {'present' if H._HAS_NODE else 'MISSING — skipped'})")
if H._HAS_NODE:
    js_good = "function reverseString(s){return s.split('').reverse().join('');}"
    r = H.run_js_checks(js_good, ["assert(eq(reverseString('abc'),'cba'))",
                                  "assert(eq(reverseString(''),''))"], timeout=5)
    check("correct JS → 2/2", r.passed == 2 and r.error is None, repr(r))
    js_bad = "function reverseString(s){return s;}"
    r = H.run_js_checks(js_bad, ["assert(eq(reverseString('abc'),'cba'))"], timeout=5)
    check("buggy JS → 0/1", r.passed == 0 and r.error is None, repr(r))
    check("gate blocks require", H.gate_js("const fs=require('fs')").reason == "forbidden_js")
    check("gate blocks process", H.gate_js("process.exit(1)").reason == "forbidden_js")
    spec_js = H.GateSpec(require_symbol="add")
    check("gate finds symbol", H.gate_js("function add(a,b){return a+b;}", spec_js).ok)
    check("gate missing symbol", H.gate_js("function sub(a,b){return a-b;}", spec_js).reason == "missing_symbol")
    # safety: even if a bad require slipped past the gate, vm context has no require
    r = H.run_js_checks("var x = (typeof require);", ["assert(eq(x,'undefined'))"], timeout=5)
    check("vm context has no require", r.passed == 1, repr(r))
    t0 = _t.time()
    r = H.run_js_checks("function f(){while(true){}}\nf();", ["assert(true)"], timeout=4)
    check("JS infinite loop → error", r.error is not None and _t.time() - t0 < 10, repr(r))

# ── PHP (php CLI) ─────────────────────────────────────────────────────────────
print(f"\nPHP (php {'present' if H._HAS_PHP else 'MISSING — skipped'})")
if H._HAS_PHP:
    php_good = "function slugify($s){return strtolower(str_replace(' ','-',trim($s)));}"
    r = H.run_php_checks(php_good, ["_eq(slugify(' Hello World '),'hello-world')",
                                    "_eq(slugify('A B'),'a-b')"], timeout=8)
    check("correct PHP → 2/2", r.passed == 2 and r.error is None, repr(r))
    php_bad = "function slugify($s){return $s;}"
    r = H.run_php_checks(php_bad, ["_eq(slugify('A B'),'a-b')"], timeout=8)
    check("buggy PHP → 0/1", r.passed == 0 and r.error is None, repr(r))
    check("gate blocks exec", H.gate_php("function f(){return exec('ls');}").reason == "forbidden_php")
    check("gate blocks file_get_contents",
          H.gate_php("function f(){return file_get_contents('/etc/passwd');}").reason == "forbidden_php")
    spec_php = H.GateSpec(require_symbol="add")
    check("gate finds symbol", H.gate_php("function add($a,$b){return $a+$b;}", spec_php).ok)
    # regression: forbidden tokens inside comments/strings must NOT trip the gate
    php_comment = ("function factorial($n){\n  // depending on requirements you might "
                   "throw\n  $r=1; for($i=1;$i<=$n;$i++){$r*=$i;} return $r;\n}")
    check("PHP comment 'requirements' does NOT block (regression)",
          H.gate_php(php_comment, H.GateSpec(require_symbol="factorial")).ok, repr(H.gate_php(php_comment)))
    check("PHP string with 'system' does NOT block",
          H.gate_php("function f(){ return 'use the system menu'; }").ok)
    g = H.grade_php("```php\n<?php\nfunction dbl($n){return $n*2;}\n```",
                    ["_eq(dbl(3),6)"], H.GateSpec(require_symbol="dbl"), timeout=8)
    check("grade_php full path → 1.0", g["score"] == 1.0, repr(g))

# ── HTML markup (tidy + bs4) ──────────────────────────────────────────────────
print(f"\nHTML markup (libs {'present' if H._markup_ready() else 'MISSING — skipped'}, "
      f"tidy {'present' if H._HAS_TIDY else 'absent'})")
if H._markup_ready():
    html_spec = {"kind": "html",
                 "require_selectors": [["header h1", 1], ["nav a", 3],
                                       ["main form input[type=email]", 1],
                                       ["button[type=submit]", 1]],
                 "max_nodes": 30, "forbid_selectors": ["script", "[style]", "center", "font"]}
    good_html = ("```html\n<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
                 "<title>Contact</title></head><body>"
                 "<header><h1>Contact</h1></header>"
                 "<nav><a href=\"#a\">A</a><a href=\"#b\">B</a><a href=\"#c\">C</a></nav>"
                 "<main><form><input type=\"email\" aria-label=\"email\">"
                 "<button type=\"submit\">Send</button></form></main>"
                 "</body></html>\n```")
    g = H.grade_markup(good_html, html_spec)
    check("good HTML → high score + full coverage", g["score"] > 0.9 and g["coverage"] == 1.0, repr(g))

    missing = good_html.replace('<a href="#c">C</a>', "")   # only 2 nav links
    g = H.grade_markup(missing, html_spec)
    check("missing element → coverage drops", g["coverage"] < 1.0, repr(g))

    junky = good_html.replace("</main>", "</main><script>alert(1)</script><center>x</center>")
    g = H.grade_markup(junky, html_spec)
    check("junk (script/center) penalized", g["score"] < H.grade_markup(good_html, html_spec)["score"], repr(g.get("junk")))

    if H._HAS_TIDY:
        # Tidy auto-recovers tag-soup as WARNINGS (it still renders); only severe
        # markup is an "error". So validity captures warnings as a soft signal.
        warn_html = good_html.replace('<input type="email" aria-label="email">',
                                      '<input type="email"><img src=x>')
        g = H.grade_markup(warn_html, html_spec)
        check("tidy warnings detected (img issues)",
              g.get("validity", {}).get("warnings", 0) >= 1, repr(g.get("validity")))

# ── CSS markup (tinycss2) ─────────────────────────────────────────────────────
print(f"\nCSS markup (tinycss2 {'present' if H._css_ready() else 'MISSING — skipped'})")
if H._css_ready():
    ctx = "<div class='card'><button class='btn'>Go</button><p id='lead'>hi</p></div>"
    css_spec = {"kind": "css", "html_context": ctx,
                "css_required_selectors": [".card", ".btn", "#lead"]}
    good_css = "```css\n.card{padding:8px}\n.btn{color:blue}\n#lead{font-size:14px}\n```"
    g = H.grade_markup(good_css, css_spec)
    check("good CSS → high score + full coverage", g["score"] > 0.9 and g["coverage"] == 1.0, repr(g))

    missing_css = "```css\n.card{padding:8px}\n.btn{color:blue}\n```"   # missing #lead
    g = H.grade_markup(missing_css, css_spec)
    check("CSS missing required selector → coverage drops", g["coverage"] < 1.0, repr(g))

    unused_css = good_css.replace("#lead{font-size:14px}", "#lead{font-size:14px}\n.ghost{color:red}\n.empty{}")
    g = H.grade_markup(unused_css, css_spec)
    check("CSS unused/empty rules penalized", g["css"]["unused"] >= 1 or g["css"]["empty"] >= 1, repr(g.get("css")))

    invalid_css = "```css\n.card{{{ color:: }\n```"
    g = H.grade_markup(invalid_css, css_spec)
    check("invalid CSS → validity gate fail (score 0)", g["score"] == 0.0, repr(g.get("gate")))

# ── summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*40}\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
