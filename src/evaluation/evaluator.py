import subprocess


def execute_assertion(code: str, assertion: str, timeout: int = 5) -> tuple:
    full_code = code + "\n" + assertion
    try:
        result = subprocess.run(
            ["python3", "-c", full_code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return "pass", ""
        return "fail", (result.stderr.strip() or result.stdout.strip())
    except subprocess.TimeoutExpired:
        return "timeout", "execution timed out"


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
