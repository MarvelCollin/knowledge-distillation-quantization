import subprocess
import tempfile
import os

_PREAMBLE = (
    "from typing import *\n"
    "from collections import *\n"
    "import heapq\nimport math\nimport itertools\nimport functools\n\n"
)


def execute_assertion(code: str, assertion: str, timeout: int = 5) -> tuple:
    full_code = _PREAMBLE + code + "\n" + assertion
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
        )
        if result.returncode == 0:
            return "pass", ""
        return "fail", (result.stderr.strip() or result.stdout.strip())
    except subprocess.TimeoutExpired:
        return "timeout", "execution timed out"
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def run_test_cases(code: str, test_cases: list) -> dict:
    passed = 0
    details = []
    errors = []
    for assertion in test_cases:
        outcome, error = execute_assertion(code, assertion)
        details.append(outcome)
        errors.append(error)
        if outcome == "pass":
            passed += 1
    return {"passed": passed, "total": len(test_cases), "details": details, "errors": errors}
