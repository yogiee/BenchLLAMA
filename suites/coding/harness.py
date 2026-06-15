#!/usr/bin/env python3
"""
BenchLLAMA — Coding grading harness (Battery E).

Three-stage, execution-graded pipeline shared by every coding test:

    1. EXTRACT   pull code from the first ``` fence (fall back to whole response)
    2. GATE      cheap structural checks, NO execution:
                   • parses (ast.parse)
                   • required symbol/signature present
                   • import WHITELIST (allow-list, fail-closed)
                   • no eval / exec / compile / __import__ / dunder escapes
                   • per-test constraints (line cap, "no extra defs")
                 → fail the gate ⇒ score 0, execution skipped
    3. EXECUTE   run in a sandboxed subprocess against hidden checks:
                   • fresh tmpdir cwd, scrubbed env, sys.executable
                   • RLIMIT_CPU + RLIMIT_AS + RLIMIT_FSIZE (best-effort on macOS)
                   • wall-clock timeout, whole process-group killed on expiry
                   • runtime __import__ guard re-enforces the whitelist
                 → score = fraction of hidden checks passing (pass@1)

SAFETY MODEL (single-user local dev tool): the static import whitelist is the
load-bearing boundary; it is re-enforced at runtime by an __import__ guard so a
dynamic `__import__("os")` is blocked even if the static gate is bypassed. The
RLIMITs are defense-in-depth (RLIMIT_AS is unreliable on darwin — never the sole
guard). Stronger isolation (container / sandbox-exec) is a future option, not v1.

This module has NO model in the loop and NO third-party deps — stdlib only, so it
is unit-testable on its own (see test_harness.py).
"""

import ast
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ── Runtime detection (multi-language tier; graceful skip when absent) ──────────
_NODE = shutil.which("node")
_PHP = shutil.which("php")
_TIDY = shutil.which("tidy")
_HAS_NODE = _NODE is not None
_HAS_PHP = _PHP is not None
_HAS_TIDY = _TIDY is not None
# SQL runs on Python's built-in sqlite3 module (always present) — no CLI dependency,
# and no exposure to the sqlite3 CLI's .shell/.system dot-commands.


def _markup_ready():
    """bs4 + soupsieve (DOM + selectors) — required for HTML grading."""
    try:
        import bs4  # noqa: F401
        return True
    except Exception:
        return False


def _css_ready():
    """tinycss2 — required for CSS grading."""
    try:
        import tinycss2  # noqa: F401
        return True
    except Exception:
        return False


def runtime_available(lang):
    """True if the executor/grader for `lang` can run on this host."""
    lang = (lang or "").lower()
    if lang in ("js", "javascript", "node"):
        return _HAS_NODE
    if lang == "php":
        return _HAS_PHP
    if lang in ("sql", "sqlite", "py", "python"):
        return True
    if lang in ("html", "html_css"):
        return _markup_ready()
    if lang == "css":
        return _css_ready()
    return False

# ── Import whitelist ───────────────────────────────────────────────────────────
# Allow-list of stdlib modules a pure-compute solution can legitimately need.
# Anything touching the OS, network, filesystem, or process table is excluded by
# omission (fail-closed). A specific problem may extend this via GateSpec.allow.
DEFAULT_ALLOWED_IMPORTS = frozenset({
    "math", "cmath", "collections", "itertools", "functools", "heapq", "bisect",
    "re", "string", "typing", "dataclasses", "enum", "fractions", "decimal",
    "statistics", "operator", "numbers", "array", "copy", "random", "json",
    "datetime", "textwrap", "unicodedata", "abc", "types",
})

# Names that are never acceptable in candidate code — code-exec / sandbox escapes.
FORBIDDEN_NAMES = frozenset({"eval", "exec", "compile", "__import__", "open", "input",
                             "globals", "locals", "vars", "memoryview", "breakpoint"})
# Attribute access that reaches into the runtime to escape the sandbox.
FORBIDDEN_ATTRS = frozenset({
    "__globals__", "__builtins__", "__subclasses__", "__bases__", "__mro__",
    "__class__", "__dict__", "__code__", "__closure__", "__getattribute__",
    "__reduce__", "__reduce_ex__",
})


# ── Stage 1: extract ───────────────────────────────────────────────────────────

def extract_code(text, lang=None):
    """Return code from the first ``` fence; fall back to the whole response.

    If `lang` is given and a fence tagged with that language exists, prefer it;
    otherwise take the first fence of any tag.
    """
    if not text:
        return ""
    fences = []  # (tag, body)
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        stripped = lines[i].lstrip()
        if stripped.startswith("```"):
            tag = stripped[3:].strip().lower()
            body = []
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("```"):
                body.append(lines[i])
                i += 1
            fences.append((tag, "\n".join(body)))
        i += 1
    if not fences:
        return text.strip()
    if lang:
        for tag, body in fences:
            if tag.startswith(lang.lower()):
                return body.strip()
    return fences[0][1].strip()


# ── Stage 2: gate ──────────────────────────────────────────────────────────────

class GateSpec:
    """Per-problem structural constraints checked before any execution."""

    def __init__(self, require_symbol=None, allow=(), max_lines=None,
                 forbid_extra_defs=False):
        self.require_symbol = require_symbol          # str function/var that must exist at module level
        self.allow = frozenset(allow)                 # extra modules on top of DEFAULT_ALLOWED_IMPORTS
        self.max_lines = max_lines                    # cap on non-blank, non-comment physical lines
        self.forbid_extra_defs = forbid_extra_defs    # only require_symbol may be a top-level def/class

    @property
    def allowed_imports(self):
        return DEFAULT_ALLOWED_IMPORTS | self.allow


class GateResult:
    def __init__(self, ok, reason="", details=None):
        self.ok = ok
        self.reason = reason
        self.details = details or {}

    def __repr__(self):
        return f"GateResult(ok={self.ok}, reason={self.reason!r})"


def _code_line_count(code):
    n = 0
    for ln in code.split("\n"):
        s = ln.strip()
        if s and not s.startswith("#"):
            n += 1
    return n


def gate(code, spec=None):
    """Run cheap structural checks. Returns GateResult(ok, reason, details)."""
    spec = spec or GateSpec()
    if not code or not code.strip():
        return GateResult(False, "empty")

    # parses?
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return GateResult(False, "syntax", {"error": str(e)})

    allowed = spec.allowed_imports
    top_defs = []      # module-level function/class names
    top_names = set()  # all module-level bound names (def/class/assign)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top_defs.append(node.name)
            top_names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    top_names.add(t.id)

    # walk the whole tree for imports + forbidden constructs
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in allowed:
                    return GateResult(False, "forbidden_import", {"module": alias.name})
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top not in allowed:
                return GateResult(False, "forbidden_import", {"module": node.module})
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                return GateResult(False, "forbidden_name", {"name": node.id})
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS:
                return GateResult(False, "forbidden_attr", {"attr": node.attr})

    # required symbol present?
    if spec.require_symbol and spec.require_symbol not in top_names:
        return GateResult(False, "missing_symbol", {"symbol": spec.require_symbol})

    # no extra defs?
    if spec.forbid_extra_defs:
        extras = [d for d in top_defs if d != spec.require_symbol]
        if extras:
            return GateResult(False, "extra_defs", {"defs": extras})

    # line cap?
    if spec.max_lines is not None:
        n = _code_line_count(code)
        if n > spec.max_lines:
            return GateResult(False, "too_many_lines", {"lines": n, "cap": spec.max_lines})

    return GateResult(True, "ok", {"lines": _code_line_count(code)})


# ── Stage 3: execute (sandboxed subprocess) ────────────────────────────────────

class ExecResult:
    def __init__(self, passed=0, total=0, error=None, detail=None):
        self.passed = passed
        self.total = total
        self.error = error                 # None on clean run; else "timeout"/"crash"/msg
        self.detail = detail or {}

    @property
    def score(self):
        return (self.passed / self.total) if self.total else 0.0

    def __repr__(self):
        return f"ExecResult(passed={self.passed}/{self.total}, error={self.error!r})"


# Static runner executed inside the sandbox. Reads _job.json, writes _out.json.
# Re-enforces the import whitelist at runtime via an __import__ guard.
_RUNNER_SRC = r'''
import json, builtins, sys

job = json.load(open("_job.json"))
_real_import = builtins.__import__

def _guard(name, *a, **k):
    if name.split(".")[0] not in set(job.get("allowed_imports", [])):
        raise ImportError("import %r blocked by sandbox" % name)
    return _real_import(name, *a, **k)

def emit(obj):
    json.dump(obj, open("_out.json", "w"))

try:
    mode = job["mode"]
    if mode == "checks":
        # Run the candidate solution, then each independent check against a
        # fresh copy of its namespace (checks cannot pollute one another).
        # Plain exec (not runpy) so the runner's own internals never route
        # through the __import__ guard, which is active ONLY for candidate code.
        # Pre-load whitelisted modules (and their transitive stdlib deps) BEFORE
        # arming the guard, so a candidate's `from typing import X` doesn't trip the
        # guard on typing's own internal `import sys`. After this, the guard only
        # blocks NEW top-level imports the candidate makes that aren't allowed.
        for _m in job.get("allowed_imports", []):
            try:
                _real_import(_m)
            except Exception:
                pass
        builtins.__import__ = _guard
        ns = {}
        exec(open("solution.py").read(), ns)
        passed, errors = 0, []
        checks = job["checks"]
        for i, chk in enumerate(checks):
            try:
                exec(chk, dict(ns))
                passed += 1
            except Exception as e:
                errors.append({"i": i, "err": repr(e)[:200]})
        emit({"passed": passed, "total": len(checks), "errors": errors})
    elif mode == "tests":
        # Mutation grading (E5): run the model's test_* functions against a
        # supplied implementation (clean reference or a planted mutant).
        # Exec the model's tests FIRST, then the supplied impl, so our impl wins
        # (late binding) even if the model redefined the function under test —
        # otherwise a model-supplied definition shadows the mutant and nothing
        # ever gets killed.
        # Pre-load whitelisted modules (and their transitive stdlib deps) BEFORE
        # arming the guard, so a candidate's `from typing import X` doesn't trip the
        # guard on typing's own internal `import sys`. After this, the guard only
        # blocks NEW top-level imports the candidate makes that aren't allowed.
        for _m in job.get("allowed_imports", []):
            try:
                _real_import(_m)
            except Exception:
                pass
        builtins.__import__ = _guard
        g = {}
        exec(open("test.py").read(), g)
        exec(open("impl.py").read(), g)
        fns = [(k, v) for k, v in list(g.items()) if k.startswith("test") and callable(v)]
        per = {}
        for name, fn in fns:
            try:
                fn()
                per[name] = True
            except Exception:
                per[name] = False
        emit({"per_test": per, "passed": sum(per.values()), "total": len(per)})
    elif mode == "sql":
        # SQL on the stdlib sqlite3 module (no CLI). The candidate is a query
        # STRING run by sqlite3 — not exec'd as code — so the import guard is not
        # needed here. :memory: db, extensions off, runaway-query abort.
        import sqlite3
        conn = sqlite3.connect(":memory:")
        try:
            conn.enable_load_extension(False)
        except Exception:
            pass
        steps = [0]
        def _abort():
            steps[0] += 1
            return 1 if steps[0] > 200000 else 0
        conn.set_progress_handler(_abort, 1000)
        try:
            conn.executescript(job["setup"])
            cur = conn.execute(job["query"])
            rows = [list(r) for r in cur.fetchall()]
            exp = [list(r) for r in job["expected"]]
            emit({"passed": 1 if rows == exp else 0, "total": 1, "rows": rows[:50]})
        except Exception as e:
            emit({"error": "sql_error", "msg": repr(e)[:200]})
    else:
        emit({"error": "bad_mode"})
except Exception as e:
    builtins.__import__ = _real_import
    emit({"error": "crash", "msg": repr(e)[:200]})
'''


def _preexec(cpu_s, mem_bytes, fsize_bytes):
    """Run in the child before exec: new session + resource caps (best-effort)."""
    try:
        os.setsid()
    except Exception:
        pass
    for res, val in (
        (resource.RLIMIT_CPU, (cpu_s, cpu_s)),
        (resource.RLIMIT_AS, (mem_bytes, mem_bytes)),
        (resource.RLIMIT_FSIZE, (fsize_bytes, fsize_bytes)),
    ):
        try:
            resource.setrlimit(res, val)
        except Exception:
            pass  # darwin rejects some of these — wall timeout + whitelist still hold


_MB = 1024 * 1024
# PATH covers Homebrew + Herd shims so `node`/`php` wrappers resolve their real binary.
_BASE_PATH = "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"


def _spawn(argv, cwd, *, timeout, mem_mb, capture=False, merge_stderr=False, extra_env=None):
    """Run argv in cwd under a scrubbed env + resource caps, in its own process
    group (killed wholesale on timeout). Returns {timed_out, stdout}."""
    env = {
        "PATH": _BASE_PATH, "HOME": cwd, "TMPDIR": cwd,
        "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8",
        "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1",
    }
    if extra_env:
        env.update(extra_env)
    cpu_s = max(1, int(timeout) + 1)
    proc = subprocess.Popen(
        argv, cwd=cwd, env=env,
        stdout=(subprocess.PIPE if capture else subprocess.DEVNULL),
        stderr=(subprocess.STDOUT if (capture and merge_stderr)
                else (subprocess.PIPE if capture else subprocess.DEVNULL)),
        preexec_fn=lambda: _preexec(cpu_s, mem_mb * _MB, 16 * _MB),
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except Exception:
            proc.kill()
        proc.communicate()
        return {"timed_out": True, "stdout": ""}
    return {"timed_out": False,
            "stdout": (out.decode("utf-8", "replace") if (capture and out) else "")}


def _run_filebased(files, argv, *, timeout, mem_mb, extra_env=None):
    """Write `files` into a fresh tmpdir, run argv (relative paths), parse _out.json."""
    tmp = tempfile.mkdtemp(prefix="benchllama_code_")
    try:
        for name, src in files.items():
            (Path(tmp) / name).write_text(src)
        res = _spawn(argv, tmp, timeout=timeout, mem_mb=mem_mb, extra_env=extra_env)
        if res["timed_out"]:
            return {"error": "timeout"}
        out = Path(tmp) / "_out.json"
        if not out.exists():
            return {"error": "no_output"}
        try:
            return json.loads(out.read_text())
        except Exception:
            return {"error": "bad_output"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run_job(files, job, *, timeout=10.0, mem_mb=1024):
    """Python/SQL runner: candidate files + job + static runner → parse _out.json."""
    payload = dict(files)
    payload["_job.json"] = json.dumps(job)
    payload["_runner.py"] = _RUNNER_SRC
    return _run_filebased(payload, [sys.executable, "-I", "-S", "_runner.py"],
                          timeout=timeout, mem_mb=mem_mb)


def _strip_module_demos(code):
    """Drop top-level bare expression statements — demo `print(...)` / example calls
    / module docstring that models append after the function. They execute during
    `exec(solution)` and can crash the whole evaluation (e.g. a demo call with an
    edge input that raises). Imports, defs, classes, assignments, and `if __name__`
    blocks are preserved, so real logic is untouched."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    tree.body = [n for n in tree.body if not isinstance(n, ast.Expr)]
    try:
        return ast.unparse(tree)
    except Exception:
        return code


def run_checks(solution_code, checks, *, allowed_imports=None, timeout=10.0, mem_mb=1024):
    """Execute `solution_code`, then each check snippet. pass@1 = passed/total.

    Used by E1 (generate), E2 (debug), E7 (constrained generation). Each check is
    a standalone snippet (typically `assert fn(...) == ...`) run against a fresh
    copy of the solution's module namespace.
    """
    job = {
        "mode": "checks",
        "checks": list(checks),
        "allowed_imports": sorted(allowed_imports or DEFAULT_ALLOWED_IMPORTS),
    }
    res = _run_job({"solution.py": _strip_module_demos(solution_code)}, job,
                   timeout=timeout, mem_mb=mem_mb)
    if "error" in res:
        return ExecResult(0, len(checks), error=res["error"], detail=res)
    return ExecResult(res["passed"], res["total"], detail={"errors": res.get("errors", [])})


def recover_parseable(code):
    """Largest parseable prefix of `code` — drops a truncated/garbage trailing tail
    (e.g. an incomplete `def` a model ran out of tokens on) so the rest still runs.
    Returns `code` unchanged if it already parses, "" if nothing parses."""
    try:
        ast.parse(code)
        return code
    except SyntaxError:
        pass
    lines = code.split("\n")
    for end in range(len(lines) - 1, 0, -1):
        prefix = "\n".join(lines[:end])
        try:
            ast.parse(prefix)
            return prefix
        except SyntaxError:
            continue
    return ""


def run_test_functions(impl_code, test_code, *, allowed_imports=None, timeout=10.0, mem_mb=1024):
    """Run the model's `test_*` functions against `impl_code`, INDIVIDUALLY. Returns
    ExecResult whose detail["per_test"] maps each test name → pass/fail bool, plus
    passed/total counts.

    Used by E5 mutation grading: run vs the clean impl to find which tests are valid
    (pass clean), then vs each mutant. Per-test results let the grader drop tests
    that fail the clean impl instead of zeroing the whole suite.
    """
    job = {
        "mode": "tests",
        "allowed_imports": sorted(allowed_imports or DEFAULT_ALLOWED_IMPORTS),
    }
    res = _run_job({"impl.py": impl_code, "test.py": test_code}, job,
                   timeout=timeout, mem_mb=mem_mb)
    if "error" in res:
        return ExecResult(0, 0, error=res["error"], detail=res)
    return ExecResult(res["passed"], res["total"], detail={"per_test": res.get("per_test", {})})


# ── Convenience: full extract → gate → execute for the common case ─────────────

def grade_generation(response, checks, spec=None, *, lang="python", timeout=10.0, mem_mb=1024):
    """End-to-end for E1/E2/E7: extract code, gate it, execute against checks.

    Returns dict: {code, gate: {...}, exec: {...}, score}. A gate failure short-
    circuits to score 0 with exec=None (no execution attempted).
    """
    code = extract_code(response, lang=lang)
    g = gate(code, spec)
    if not g.ok:
        return {"code": code, "gate": {"ok": False, "reason": g.reason, "details": g.details},
                "exec": None, "score": 0.0}
    allowed = spec.allowed_imports if spec else None
    ex = run_checks(code, checks, allowed_imports=allowed, timeout=timeout, mem_mb=mem_mb)
    return {
        "code": code,
        "gate": {"ok": True, "reason": "ok", "details": g.details},
        "exec": {"passed": ex.passed, "total": ex.total, "error": ex.error,
                 "errors": ex.detail.get("errors", [])},
        "score": round(ex.score, 4),
    }


# ══ Multi-language tier (E3) — JavaScript · SQL · PHP ════════════════════════════
# Same EXTRACT → GATE → EXECUTE contract per language. Each executor returns an
# ExecResult; each grader returns the same dict shape as grade_generation().
# Languages whose runtime is absent return error="runtime_missing" (graceful skip).

# Sanitize source before a forbidden-token scan: drop comments and blank string
# literals so tokens inside comments/strings (e.g. "requirements", a URL) don't
# trip the gate. Word-boundary / call-pattern regexes then avoid substring false
# positives (the bug where a `// ...requirements...` comment blocked valid PHP).
def _sanitize_for_scan(code, lang):
    lang = lang.lower()
    if lang in ("php", "js", "javascript"):
        code = re.sub(r"/\*.*?\*/", " ", code, flags=re.S)
        code = re.sub(r"//[^\n]*", " ", code)
        if lang == "php":
            code = re.sub(r"#[^\n]*", " ", code)
    elif lang in ("sql", "sqlite"):
        code = re.sub(r"/\*.*?\*/", " ", code, flags=re.S)
        code = re.sub(r"--[^\n]*", " ", code)
    code = re.sub(r'"(\\.|[^"\\])*"', '""', code)   # blank double-quoted strings
    code = re.sub(r"'(\\.|[^'\\])*'", "''", code)   # blank single-quoted strings
    return code


# ── SQL (Python stdlib sqlite3) ────────────────────────────────────────────────
# A "solution" is a single query run against a trusted setup; graded by exact
# row-set match. Writes / schema mutation / extension loading are gated out, and
# sqlite3.execute() already refuses multiple statements (kills `;`-injection).
_SQL_FORBIDDEN_RE = [
    r"\b(attach|detach|pragma|vacuum|load_extension)\b",
    r"\b(drop|alter|insert|update|delete|replace|create|truncate)\s",
]


def gate_sql(code, spec=None):
    if not code or not code.strip():
        return GateResult(False, "empty")
    scan = _sanitize_for_scan(code, "sql").lower()
    for pat in _SQL_FORBIDDEN_RE:
        m = re.search(pat, scan)
        if m:
            return GateResult(False, "forbidden_sql", {"match": m.group(0).strip()})
    return GateResult(True, "ok")


def run_sql_query(setup, query, expected, *, timeout=10.0):
    res = _run_job({}, {"mode": "sql", "setup": setup, "query": query, "expected": expected},
                   timeout=timeout)
    if "error" in res:
        return ExecResult(0, 1, error=res["error"], detail=res)
    return ExecResult(res["passed"], res["total"], detail={"rows": res.get("rows")})


def grade_sql(response, setup, expected, *, timeout=10.0):
    """E3 SQL: extract query → gate (no writes/escapes) → run vs expected rows."""
    code = extract_code(response, lang="sql")
    g = gate_sql(code)
    if not g.ok:
        return {"code": code, "lang": "sql",
                "gate": {"ok": False, "reason": g.reason, "details": g.details},
                "exec": None, "score": 0.0}
    ex = run_sql_query(setup, code, expected, timeout=timeout)
    return {"code": code, "lang": "sql", "gate": {"ok": True, "reason": "ok"},
            "exec": {"passed": ex.passed, "total": ex.total, "error": ex.error,
                     "rows": ex.detail.get("rows")},
            "score": round(ex.score, 4)}


# ── JavaScript (node + vm) ─────────────────────────────────────────────────────
# Candidate runs inside a fresh vm context: no require / process / fs / module are
# in scope (they are module-wrapper injections, absent from a bare context). The
# static gate blocks the known vm-escape tokens; node runs with code-generation
# from strings disabled and a small heap cap. vm is not a hard security boundary —
# the gate + scrubbed env + rlimits are the real containment.
_JS_FORBIDDEN_RE = [
    r"\brequire\s*\(", r"\bimport\b", r"\bprocess\b", r"\bchild_process\b",
    r"\bglobalThis\b", r"\bglobal\b", r"\beval\s*\(", r"\bFunction\s*\(",
    r"__proto__", r"\bconstructor\b", r"\bfetch\s*\(", r"\bXMLHttpRequest\b",
    r"\bWebAssembly\b", r"\bDeno\b", r"\bBun\b", r"\bmodule\.", r"\b__dirname\b",
]

_RUNNER_JS = r'''
const vm = require('vm');
const fs = require('fs');
const job = JSON.parse(fs.readFileSync('_job.json', 'utf8'));
const solution = fs.readFileSync('solution.js', 'utf8');
const prelude =
  "function assert(c,m){if(!c)throw new Error(m||'assert failed');}" +
  "function eq(a,b){return JSON.stringify(a)===JSON.stringify(b);}";
const sandbox = {};
vm.createContext(sandbox, { codeGeneration: { strings: false, wasm: false } });
let results;
try {
  vm.runInContext(prelude + "\n" + solution, sandbox, { timeout: 4000 });
  let passed = 0; const errors = [];
  job.checks.forEach((chk, i) => {
    try { vm.runInContext(chk, sandbox, { timeout: 2000 }); passed++; }
    catch (e) { errors.push({ i: i, err: String(e).slice(0, 200) }); }
  });
  results = { passed: passed, total: job.checks.length, errors: errors };
} catch (e) {
  results = { error: 'crash', msg: String(e).slice(0, 200) };
}
fs.writeFileSync('_out.json', JSON.stringify(results));
'''


def gate_js(code, spec=None):
    if not code or not code.strip():
        return GateResult(False, "empty")
    scan = _sanitize_for_scan(code, "javascript")
    for pat in _JS_FORBIDDEN_RE:
        m = re.search(pat, scan)
        if m:
            return GateResult(False, "forbidden_js", {"match": m.group(0)})
    if spec and spec.require_symbol:
        n = re.escape(spec.require_symbol)
        pat = r"(function\s+%s\b|(?:const|let|var)\s+%s\b|\b%s\s*=)" % (n, n, n)
        if not re.search(pat, code):
            return GateResult(False, "missing_symbol", {"symbol": spec.require_symbol})
    return GateResult(True, "ok")


def run_js_checks(solution_code, checks, *, timeout=10.0, mem_mb=512):
    if not _HAS_NODE:
        return ExecResult(0, len(checks), error="runtime_missing")
    files = {"solution.js": solution_code, "_runner.js": _RUNNER_JS,
             "_job.json": json.dumps({"checks": list(checks)})}
    argv = [_NODE, "--max-old-space-size=256",
            "--disallow-code-generation-from-strings", "_runner.js"]
    res = _run_filebased(files, argv, timeout=timeout, mem_mb=mem_mb)
    if "error" in res:
        return ExecResult(0, len(checks), error=res["error"], detail=res)
    return ExecResult(res["passed"], res["total"], detail={"errors": res.get("errors", [])})


def grade_js(response, checks, spec=None, *, timeout=10.0, mem_mb=512):
    code = extract_code(response, lang="javascript")
    g = gate_js(code, spec)
    if not g.ok:
        return {"code": code, "lang": "javascript",
                "gate": {"ok": False, "reason": g.reason, "details": g.details},
                "exec": None, "score": 0.0}
    ex = run_js_checks(code, checks, timeout=timeout, mem_mb=mem_mb)
    return {"code": code, "lang": "javascript", "gate": {"ok": True, "reason": "ok"},
            "exec": {"passed": ex.passed, "total": ex.total, "error": ex.error,
                     "errors": ex.detail.get("errors", [])},
            "score": round(ex.score, 4)}


# ── PHP (php CLI, hardened) ─────────────────────────────────────────────────────
# Candidate + check harness are concatenated into one script; results are echoed
# between sentinels (json_encode/echo need no file functions, so file/exec/network
# functions are disabled via -d disable_functions and confined by -d open_basedir).
# Match dangerous *calls* (name + `(`) and the include/require constructs — with
# word boundaries and comment/string stripping, so identifiers and prose can't
# false-trip (a `// ...requirements...` comment must NOT block valid code).
_PHP_FORBIDDEN_RE = [
    (r"\b(exec|shell_exec|system|passthru|popen|proc_open|eval|assert|"
     r"file_get_contents|file_put_contents|fopen|fwrite|fread|fgets|unlink|"
     r"fsockopen|putenv|getenv|scandir|glob|opendir|readdir|mkdir|rmdir|chmod|"
     r"symlink|phpinfo|ini_set|dl)\s*\("),
    r"\bcurl_\w+\s*\(",
    r"\bstream_socket_\w+\s*\(",
    r"\b(include|require)(_once)?\b",
    r"`",  # backtick = shell execution
]

_PHP_DISABLE = ("exec,shell_exec,system,passthru,popen,proc_open,fopen,fread,fwrite,"
                "file_get_contents,file_put_contents,unlink,curl_exec,curl_init,"
                "fsockopen,stream_socket_client,putenv,getenv,scandir,glob,opendir,"
                "readdir,mkdir,rmdir,chmod,symlink,link,dl,phpinfo,proc_open")


def gate_php(code, spec=None):
    if not code or not code.strip():
        return GateResult(False, "empty")
    scan = _sanitize_for_scan(code, "php")
    for pat in _PHP_FORBIDDEN_RE:
        m = re.search(pat, scan, re.I)
        if m:
            return GateResult(False, "forbidden_php", {"match": m.group(0)})
    if spec and spec.require_symbol:
        if not re.search(r"function\s+%s\b" % re.escape(spec.require_symbol), code):
            return GateResult(False, "missing_symbol", {"symbol": spec.require_symbol})
    return GateResult(True, "ok")


def run_php_checks(solution_code, checks, *, timeout=10.0, mem_mb=512):
    if not _HAS_PHP:
        return ExecResult(0, len(checks), error="runtime_missing")
    sol = re.sub(r"^\s*<\?php", "", solution_code).replace("?>", "")
    parts = [
        "<?php",
        "function _eq($a,$b){return json_encode($a)===json_encode($b);}",
        sol,
        "$__res=array('passed'=>0,'total'=>%d,'errors'=>array());" % len(checks),
    ]
    for i, chk in enumerate(checks):
        parts.append(
            "try{ if((%s)){$__res['passed']++;}else{$__res['errors'][]=array('i'=>%d);} }"
            "catch(\\Throwable $e){$__res['errors'][]=array('i'=>%d,'err'=>substr($e->getMessage(),0,160));}"
            % (chk, i, i))
    parts.append('echo "__OUT__".json_encode($__res)."__END__";')
    tmp = tempfile.mkdtemp(prefix="benchllama_code_")
    try:
        (Path(tmp) / "runner.php").write_text("\n".join(parts))
        argv = [_PHP, "-n",
                "-d", "disable_functions=" + _PHP_DISABLE,
                "-d", "open_basedir=" + tmp,
                "-d", "allow_url_fopen=0", "-d", "allow_url_include=0",
                "-d", "memory_limit=%dM" % mem_mb,
                "-d", "max_execution_time=%d" % int(timeout),
                "runner.php"]
        res = _spawn(argv, tmp, timeout=timeout, mem_mb=mem_mb, capture=True)
        if res["timed_out"]:
            return ExecResult(0, len(checks), error="timeout")
        m = re.search(r"__OUT__(.*)__END__", res["stdout"], re.S)
        if not m:
            return ExecResult(0, len(checks), error="no_output",
                              detail={"stdout": res["stdout"][:200]})
        try:
            data = json.loads(m.group(1))
        except Exception:
            return ExecResult(0, len(checks), error="bad_output")
        return ExecResult(data["passed"], data["total"], detail={"errors": data.get("errors", [])})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def grade_php(response, checks, spec=None, *, timeout=10.0, mem_mb=512):
    code = extract_code(response, lang="php")
    g = gate_php(code, spec)
    if not g.ok:
        return {"code": code, "lang": "php",
                "gate": {"ok": False, "reason": g.reason, "details": g.details},
                "exec": None, "score": 0.0}
    ex = run_php_checks(code, checks, timeout=timeout, mem_mb=mem_mb)
    return {"code": code, "lang": "php", "gate": {"ok": True, "reason": "ok"},
            "exec": {"passed": ex.passed, "total": ex.total, "error": ex.error,
                     "errors": ex.detail.get("errors", [])},
            "score": round(ex.score, 4)}


# ══ Markup tier (E9) — HTML / CSS, objective rubric (no execution) ═══════════════
# Generation correctness for markup is graded by a deterministic rubric over a
# parsed DOM / stylesheet — NOT by running it and NOT by an LLM judge:
#   • VALIDITY  — HTML Tidy reports 0 errors (won't-render gate); CSS parses clean
#   • COVERAGE  — required CSS selectors resolve in the DOM (catches *skipping*)
#   • CLEANLINESS — node budget, forbidden junk (script/inline-style/deprecated
#                   tags), and HTML↔CSS cross-check (unused or missing rules)
# score = validity_gate × (0.7·coverage + 0.3·(1 − junk_penalty))

_MARKUP_COVERAGE_W = 0.7
_MARKUP_CLEAN_W = 0.3
_TIDY_WARN_FACTOR = 0.02      # each Tidy warning nudges the junk penalty
_JUNK_HIT_WEIGHT = 0.34       # each forbidden-pattern hit


def _slice_html(text):
    """Strip prose around the document if the model didn't fence its output."""
    low = text.lower()
    for marker in ("<!doctype", "<html"):
        i = low.find(marker)
        if i != -1:
            return text[i:]
    return text


def _run_tidy(html):
    """Return {available, errors, warnings, raw} from HTML Tidy (validity gate)."""
    if not _HAS_TIDY:
        return {"available": False, "errors": 0, "warnings": 0}
    tmp = tempfile.mkdtemp(prefix="benchllama_code_")
    try:
        (Path(tmp) / "index.html").write_text(html)
        # No -q: it suppresses the "Tidy found N warnings and M errors!" summary.
        res = _spawn([_TIDY, "-errors", "index.html"], tmp,
                     timeout=10, mem_mb=256, capture=True, merge_stderr=True)
        text = res["stdout"] or ""
        m = re.search(r"Tidy found (\d+) warning[s]? and (\d+) error", text)
        if m:
            return {"available": True, "warnings": int(m.group(1)),
                    "errors": int(m.group(2)), "raw": text[:600]}
        return {"available": True,
                "errors": len(re.findall(r"- Error:", text)),
                "warnings": len(re.findall(r"- Warning:", text)), "raw": text[:600]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _css_analyze(css):
    """Parse CSS with tinycss2 → (selectors, n_rules, n_empty, n_errors)."""
    import tinycss2
    rules = tinycss2.parse_stylesheet(css, skip_comments=True, skip_whitespace=True)
    n_err = sum(1 for r in rules if r.type == "error")   # top-level (bad selectors / @-rules)
    selectors, n_empty, n_rules = [], 0, 0
    for r in rules:
        if r.type != "qualified-rule":
            continue
        n_rules += 1
        parsed = tinycss2.parse_declaration_list(
            r.content, skip_comments=True, skip_whitespace=True)
        n_err += sum(1 for d in parsed if d.type == "error")   # malformed declarations
        if not any(d.type == "declaration" for d in parsed):
            n_empty += 1
        sel = tinycss2.serialize(r.prelude).strip()
        selectors += [s.strip() for s in sel.split(",") if s.strip()]
    return selectors, n_rules, n_empty, n_err


def grade_markup(response, spec, *, timeout=10.0):
    """E9 — grade an HTML or CSS generation against an objective rubric.

    spec keys (by kind):
      html: require_selectors=[[sel, count|">=1"], ...], max_nodes=int,
            forbid_selectors=[sel, ...]
      css : html_context=str, css_required_selectors=[sel, ...]
    """
    kind = spec.get("kind", "html")
    if kind == "css":
        return _grade_css(response, spec)
    return _grade_html(response, spec)


def _grade_html(response, spec):
    if not _markup_ready():
        return {"lang": "html", "skipped": True, "reason": "markup_libs_missing", "score": None}
    from bs4 import BeautifulSoup
    code = extract_code(response, lang="html")
    if "```" not in response and "<" in code:
        code = _slice_html(code)

    tidy = _run_tidy(code)
    if tidy.get("available") and tidy["errors"] > 0:
        return {"code": code, "lang": "html",
                "gate": {"ok": False, "reason": "html_invalid",
                         "details": {"tidy_errors": tidy["errors"]}},
                "validity": tidy, "score": 0.0}

    soup = BeautifulSoup(code, "html5lib")
    req = spec.get("require_selectors", [])
    sat, sel_detail = 0, []
    for sel, want in req:
        n = len(soup.select(sel))
        ok = (n >= 1) if want in (">=1", None) else (n == want)
        sat += 1 if ok else 0
        sel_detail.append({"selector": sel, "want": want, "found": n, "ok": ok})
    coverage = sat / len(req) if req else 1.0

    junk_hits = [s for s in spec.get("forbid_selectors", []) if soup.select(s)]
    nodes = len(soup.find_all(True))
    budget = spec.get("max_nodes")
    over = (max(0, nodes - budget) / budget) if budget else 0.0
    warns = tidy.get("warnings", 0) if tidy.get("available") else 0
    junk_penalty = min(1.0, len(junk_hits) * _JUNK_HIT_WEIGHT + over + warns * _TIDY_WARN_FACTOR)

    score = _MARKUP_COVERAGE_W * coverage + _MARKUP_CLEAN_W * (1 - junk_penalty)
    return {
        "code": code, "lang": "html", "gate": {"ok": True, "reason": "ok"},
        "validity": tidy,
        "coverage": round(coverage, 4), "selectors": sel_detail,
        "junk": {"forbidden_hits": junk_hits, "nodes": nodes, "budget": budget,
                 "tidy_warnings": warns, "penalty": round(junk_penalty, 4)},
        "score": round(max(0.0, score), 4),
    }


def _grade_css(response, spec):
    if not _css_ready():
        return {"lang": "css", "skipped": True, "reason": "tinycss2_missing", "score": None}
    code = extract_code(response, lang="css")
    selectors, n_rules, n_empty, n_err = _css_analyze(code)
    if n_err > 0 or n_rules == 0:
        return {"code": code, "lang": "css",
                "gate": {"ok": False, "reason": "css_invalid",
                         "details": {"errors": n_err, "rules": n_rules}}, "score": 0.0}

    # coverage: each required selector must be defined by some rule
    req = spec.get("css_required_selectors", [])
    norm = {s.replace(" ", "") for s in selectors}
    sat = sum(1 for r in req if r.replace(" ", "") in norm)
    coverage = sat / len(req) if req else 1.0

    # cleanliness: empty rules + selectors that match nothing in the context HTML
    unused = 0
    ctx = spec.get("html_context")
    if ctx and _markup_ready():
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(ctx, "html5lib")
        for s in selectors:
            try:
                if not soup.select(s):
                    unused += 1
            except Exception:
                pass  # soupsieve can't parse exotic selector — don't penalize
    total = max(1, len(selectors))
    junk_penalty = min(1.0, n_empty / total + unused / total)

    score = _MARKUP_COVERAGE_W * coverage + _MARKUP_CLEAN_W * (1 - junk_penalty)
    return {
        "code": code, "lang": "css", "gate": {"ok": True, "reason": "ok"},
        "coverage": round(coverage, 4),
        "css": {"rules": n_rules, "empty": n_empty, "unused": unused,
                "penalty": round(junk_penalty, 4)},
        "score": round(max(0.0, score), 4),
    }
