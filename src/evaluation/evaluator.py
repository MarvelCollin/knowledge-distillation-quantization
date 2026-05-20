import subprocess


def execute_assertion(code: str, assertion: str, timeout: int = 5) -> str:
    full_code = code + "\n" + assertion
    try:
        result = subprocess.run(
            ["python3", "-c", full_code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return "pass" if result.returncode == 0 else "fail"
    except subprocess.TimeoutExpired:
        return "timeout"


def run_test_cases(code: str, test_cases: list) -> dict:
    passed = 0
    details = []
    for assertion in test_cases:
        outcome = execute_assertion(code, assertion)
        details.append(outcome)
        if outcome == "pass":
            passed += 1
    return {"passed": passed, "total": len(test_cases), "details": details}
