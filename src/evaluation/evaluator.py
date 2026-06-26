import ast
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path


_MEMORY_LIMIT_BYTES = 1024 ** 3
_CPU_TIME_LIMIT_SEC = 30
_STACK_LIMIT_BYTES = 64 * 1024 * 1024

_LIMIT_BOOTSTRAP = (
    "import resource as _r; "
    f"_r.setrlimit(_r.RLIMIT_AS, ({_MEMORY_LIMIT_BYTES}, {_MEMORY_LIMIT_BYTES})); "
    f"_r.setrlimit(_r.RLIMIT_DATA, ({_MEMORY_LIMIT_BYTES}, {_MEMORY_LIMIT_BYTES})); "
    f"_r.setrlimit(_r.RLIMIT_CPU, ({_CPU_TIME_LIMIT_SEC}, {_CPU_TIME_LIMIT_SEC})); "
    f"_r.setrlimit(_r.RLIMIT_STACK, ({_STACK_LIMIT_BYTES}, {_STACK_LIMIT_BYTES}))\n"
)


_LEETCODE_PREAMBLE = '''\
from typing import *
from collections import *
import heapq, math, itertools, functools, bisect, string, re, random, operator, copy
from math import comb, perm, gcd, lcm, factorial, sqrt, isqrt, ceil, floor, inf, log, log2
from itertools import combinations, permutations, product, accumulate, chain, count, groupby, pairwise
from heapq import heappush, heappop, heapify, heappushpop, heapreplace, nlargest, nsmallest
from functools import reduce, lru_cache, cache, cmp_to_key
from bisect import bisect_left, bisect_right, insort, insort_left, insort_right


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        vals, cur = [], self
        while cur:
            vals.append(cur.val)
            cur = cur.next
        return str(vals)


def list_node(values):
    dummy = ListNode(0)
    cur = dummy
    for v in values:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next


def is_same_list(a, b):
    while a and b:
        if a.val != b.val:
            return False
        a, b = a.next, b.next
    return a is None and b is None


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        return f"TreeNode({self.val})"


def tree_node(values):
    if not values:
        return None
    root = TreeNode(values[0])
    queue = [root]
    i = 1
    while queue and i < len(values):
        node = queue.pop(0)
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root


def is_same_tree(p, q):
    if p is None and q is None:
        return True
    if p is None or q is None:
        return False
    return (
        p.val == q.val
        and is_same_tree(p.left, q.left)
        and is_same_tree(p.right, q.right)
    )


class Node:
    def __init__(self, val=0, neighbors=None, children=None,
                 next=None, left=None, right=None, random=None):
        self.val = val
        self.neighbors = neighbors or []
        self.children = children or []
        self.next = next
        self.left = left
        self.right = right
        self.random = random
'''


class _DiagAssertRewriter(ast.NodeTransformer):
    def __init__(self, candidate_name: str | None):
        self.cand = candidate_name

    def _find_input(self, expr):
        if not self.cand:
            return None
        for n in ast.walk(expr):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == self.cand
            ):
                return ", ".join(ast.unparse(a) for a in n.args)
        return None

    def visit_Assert(self, node):
        test = node.test
        src = ast.unparse(test)

        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
            out_expr = test.left
            exp_expr = test.comparators[0]
            check = "_out == _exp"
        elif isinstance(test, ast.Call) and len(test.args) >= 2:
            out_expr = test.args[0]
            exp_expr = test.args[1]
            helper = ast.unparse(test.func)
            check = f"{helper}(_out, _exp)"
        else:
            return ast.parse(
                f"if not ({src}):\n"
                f"    print('  Failed:   ' + {src!r})\n"
                f"    raise AssertionError({src!r})"
            ).body

        in_args = self._find_input(out_expr)
        in_line = (
            f"    print('  Input:    ' + {in_args!r})\n" if in_args is not None else ""
        )
        out_src = ast.unparse(out_expr)
        exp_src = ast.unparse(exp_expr)
        return ast.parse(
            f"_out = ({out_src})\n"
            f"_exp = ({exp_src})\n"
            f"if not ({check}):\n"
            f"{in_line}"
            f"    print(f'  Output:   {{_out!r}}')\n"
            f"    print(f'  Expected: {{_exp!r}}')\n"
            f"    raise AssertionError({src!r})"
        ).body


def extract_signature(test_str: str, entry_point: str) -> str:
    if not test_str or not entry_point:
        return ""
    try:
        tree = ast.parse(test_str)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "candidate"
        ):
            params = [kw.arg for kw in node.keywords if kw.arg]
            if params:
                return f"def {entry_point}({', '.join(params)}):"
    return f"def {entry_point}():"


def _prepare_assertion(assertion: str) -> tuple[str | None, str]:
    try:
        tree = ast.parse(assertion)
    except SyntaxError:
        return None, assertion

    entry_point = None
    param = None
    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "check"
            and node.value.args
            and isinstance(node.value.args[0], ast.Name)
        ):
            entry_point = node.value.args[0].id
        elif isinstance(node, ast.FunctionDef) and node.name == "check":
            if node.args.args:
                param = node.args.args[0].arg

    tree = _DiagAssertRewriter(param).visit(tree)
    ast.fix_missing_locations(tree)
    return entry_point or param, ast.unparse(tree)


def _alias_candidate(code: str, candidate: str | None) -> str:
    if not candidate:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    names = [n.name for n in funcs]
    if not names or candidate in names:
        return code
    target = candidate.replace("_", "").lower()
    matched = next((n for n in names if n.replace("_", "").lower() == target), None)
    if matched is None and len(names) == 1:
        matched = names[0]
    if matched is None and len(names) > 1:
        called_by_others = set()
        for fn in funcs:
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id != fn.name:
                    called_by_others.add(node.func.id)
        entries = [n for n in names if n not in called_by_others]
        matched = entries[-1] if entries else names[-1]
    if matched is None:
        return code
    return code + f"\n{candidate} = {matched}\n"


def execute_assertion(code: str, assertion: str, timeout: int = 10) -> tuple:
    candidate, rewritten = _prepare_assertion(assertion)
    code = _alias_candidate(code, candidate)
    full_code = _LIMIT_BOOTSTRAP + _LEETCODE_PREAMBLE + "\n" + code + "\n\n" + rewritten + "\n"
    workdir = tempfile.mkdtemp(prefix="exec_")
    script = os.path.join(workdir, "solution.py")
    with open(script, "w") as f:
        f.write(full_code)
    proc = subprocess.Popen(
        [sys.executable, script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        cwd=workdir,
        env={"PATH": "/usr/bin:/bin", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"},
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.communicate()
        return "timeout", "execution timed out"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if proc.returncode == 0:
        return "pass", ""
    combined = "\n".join(p for p in (out.rstrip(), err.rstrip()) if p)
    return "fail", combined


def _format_error(err: str) -> str:
    if not err:
        return ""
    lines = err.splitlines()
    diag = [l for l in lines if l.lstrip().startswith(("Input:", "Output:", "Expected:", "Failed:"))]
    if diag:
        return "\n".join(diag)
    name_match = re.search(r"NameError: name '(\w+)' is not defined", err)
    if name_match:
        return f"NameError: '{name_match.group(1)}' not defined"
    last = lines[-1] if lines else ""
    assert_line = next((l.strip() for l in lines if l.strip().startswith("assert ")), "")
    if assert_line and last:
        return f"{assert_line}  →  {last}"
    return last or err[:200]


def _classify(outcome: str, error: str) -> str:
    if outcome == "pass":
        return "pass"
    if outcome == "timeout":
        return "timeout"
    if "SyntaxError" in error or "IndentationError" in error:
        return "syntax_error"
    if "AssertionError" in error:
        return "wrong_answer"
    if "NameError" in error and "not defined" in error:
        return "missing_function"
    return "runtime_error"


def run_test_cases(code: str, test_cases: list) -> dict:
    passed = 0
    details = []
    errors = []
    categories = []
    for assertion in test_cases:
        outcome, error = execute_assertion(code, assertion)
        details.append(outcome)
        errors.append(error)
        categories.append(_classify(outcome, error))
        if outcome == "pass":
            passed += 1
    return {
        "passed": passed,
        "total": len(test_cases),
        "details": details,
        "errors": errors,
        "categories": categories,
    }


def failing_cases(result: dict, limit: int = 3) -> list:
    """Pull the first `limit` failing (category, error, assertion) triples from a
    run_test_cases result, de-duplicated by error message so the report stays short."""
    out = []
    seen = set()
    details = result.get("details", [])
    errors = result.get("errors", [])
    cats = result.get("categories", [])
    for i, outcome in enumerate(details):
        if outcome == "pass":
            continue
        err = (errors[i] if i < len(errors) else "") or "(no error message)"
        cat = cats[i] if i < len(cats) else "runtime_error"
        key = (cat, err[:160])
        if key in seen:
            continue
        seen.add(key)
        out.append((cat, err))
        if len(out) >= limit:
            break
    return out


def write_failure_report(path, title: str, entries: list) -> None:
    """Write a human-readable Markdown report of per-problem outcomes for debugging.

    entries: list of dicts, each with:
      header (str), solved (bool), note (str, optional), code (str, optional),
      fails (list[(category, error)], optional)
    Solved problems get a one-line ✓; failed problems show the generated code and
    the first few distinct failing test cases so the root cause is visible."""
    n_solved = sum(1 for e in entries if e.get("solved"))
    lines = [
        f"# {title}",
        "",
        f"Solved: **{n_solved}/{len(entries)}**.  Failed problems below show the generated "
        f"code and the first distinct failing assertions.",
        "",
    ]
    for e in entries:
        mark = "✓" if e.get("solved") else "✗"
        lines.append(f"## {mark} {e.get('header', '')}")
        if e.get("note"):
            lines.append(e["note"])
        if not e.get("solved"):
            code = (e.get("code") or "").strip()
            lines.append("")
            lines.append("```python")
            lines.append(code[:4000] if code else "# (no code was extracted from the model output)")
            lines.append("```")
            fails = e.get("fails", [])
            if fails:
                lines.append("")
                lines.append("Failing cases:")
                for cat, err in fails:
                    snippet = " ".join((err or "").split())[:300]
                    lines.append(f"- **{cat}** — {snippet}")
        lines.append("")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines))
    