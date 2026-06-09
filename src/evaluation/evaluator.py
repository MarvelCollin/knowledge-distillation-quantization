import ast
import os
import re
import resource
import subprocess
import tempfile


_MEMORY_LIMIT_BYTES = 1024 ** 3
_CPU_TIME_LIMIT_SEC = 30


def _apply_subprocess_limits() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (_MEMORY_LIMIT_BYTES, _MEMORY_LIMIT_BYTES))
    resource.setrlimit(resource.RLIMIT_DATA, (_MEMORY_LIMIT_BYTES, _MEMORY_LIMIT_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (_CPU_TIME_LIMIT_SEC, _CPU_TIME_LIMIT_SEC))
    resource.setrlimit(resource.RLIMIT_STACK, (64 * 1024 * 1024, 64 * 1024 * 1024))


_LEETCODE_PREAMBLE = '''\
from typing import *
from collections import *
import heapq, math, itertools, functools, bisect, string, re, random, operator, copy


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
    names = [
        n.name for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not names or candidate in names:
        return code
    target = candidate.replace("_", "").lower()
    matched = next((n for n in names if n.replace("_", "").lower() == target), None)
    if matched is None and len(names) == 1:
        matched = names[0]
    if matched is None:
        return code
    return code + f"\n{candidate} = {matched}\n"


def execute_assertion(code: str, assertion: str, timeout: int = 10) -> tuple:
    candidate, rewritten = _prepare_assertion(assertion)
    code = _alias_candidate(code, candidate)
    full_code = _LEETCODE_PREAMBLE + "\n" + code + "\n\n" + rewritten + "\n"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            tmp = f.name
        result = subprocess.run(
            ["python3", tmp],
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            preexec_fn=_apply_subprocess_limits,
        )
        if result.returncode == 0:
            return "pass", ""
        out = result.stdout.rstrip()
        err = result.stderr.rstrip()
        combined = "\n".join(p for p in (out, err) if p)
        return "fail", combined
    except subprocess.TimeoutExpired:
        return "timeout", "execution timed out"
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


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
    