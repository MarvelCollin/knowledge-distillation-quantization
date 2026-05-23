import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import json
import re
import yaml
import torch
import argparse

from tqdm import tqdm

from src.data.dataset import create_datasets
from src.evaluation.evaluator import run_test_cases
from src.student.model import StudentModel


def extract_function(code: str) -> str:
    """Keep only the first function definition; strip \r\n and trailing test/print code."""
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    lines = code.split('\n')
    result = []
    for line in lines:
        if not result:
            result.append(line)
        elif not line or line[0] in (' ', '\t'):
            result.append(line)
        else:
            break
    while result and not result[-1].strip():
        result.pop()
    return '\n'.join(result)


_SYSTEM = (
    "You are a coding assistant. Write a standalone Python function that solves the problem. "
    "Output ONLY the function definition, nothing else. No class, no explanation, "
    "no markdown, no triple backticks. Start directly with 'def'."
)


def _extract_fn_name(test_cases: list) -> str | None:
    for tc in test_cases:
        for name in re.findall(r'\bcheck\((\w+)\)', tc):
            if name != 'candidate':
                return name
        m = re.search(r'\bassert\s+(\w+)\s*\(', tc)
        if m and m.group(1) != 'candidate':
            return m.group(1)
    return None


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def generate_solution(student: StudentModel, prompt: str, test_cases: list,
                      max_new_tokens: int, device: torch.device) -> str:
    expected = _extract_fn_name(test_cases)

    clean_prompt = prompt.rstrip()
    if clean_prompt.endswith("def"):
        clean_prompt = clean_prompt[:-3].rstrip()

    user_content = clean_prompt
    if expected:
        user_content = clean_prompt + f"\n\nName the function `{expected}`."

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_content},
    ]
    fmt = student.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    student.eval()
    with torch.no_grad():
        inputs = student.tokenizer(fmt, return_tensors="pt").to(device)
        out = student.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=student.tokenizer.eos_token_id,
        )

    gen = out[0][inputs["input_ids"].shape[1]:]
    code = student.tokenizer.decode(gen, skip_special_tokens=True).strip()

    fence = re.search(r'```(?:python)?\s*\n?(.*?)(?:```|$)', code, re.DOTALL | re.IGNORECASE)
    if fence:
        code = fence.group(1).strip()

    m = re.search(r'def \w', code)
    if m:
        code = code[m.start():]
    elif not code.startswith("def "):
        code = "def " + code.lstrip()

    code = extract_function(code)

    if expected:
        old = re.match(r'def (\w+)\(', code)
        if old and old.group(1) != expected:
            old_name = re.escape(old.group(1))
            code = re.sub(r'\b' + old_name + r'\b', expected, code)

    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show full generated code and error messages for each test")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student = StudentModel(
        model_name=args.checkpoint,
        max_length=config["student"]["max_length"],
    )
    student.to(device)

    _, val_dataset = create_datasets(config, student.tokenizer)

    total_passed = 0
    total_tests = 0
    problems_solved = 0
    all_results = []

    print(f"\nEvaluating {len(val_dataset)} problems...\n")
    print(f"{'─' * 60}")

    for i in tqdm(range(len(val_dataset)), desc="Evaluating", leave=False):
        prompt = val_dataset.get_prompt(i)
        test_cases = val_dataset.get_test_cases(i)

        generated_code = generate_solution(student, prompt, test_cases, config["evaluation"]["max_new_tokens"], device)
        result = run_test_cases(generated_code, test_cases)

        total_passed += result["passed"]
        total_tests += result["total"]
        if result["passed"] == result["total"] and result["total"] > 0:
            problems_solved += 1

        status_icon = "✓" if result["passed"] == result["total"] else "✗"
        detail_str = "  ".join(
            f"[{'✓' if d == 'pass' else '✗' if d == 'fail' else '⏱'}]" for d in result["details"]
        )
        code_preview = generated_code.replace('\n', ' | ')[:100]
        tqdm.write(
            f"  {status_icon} Problem {i:>2}  {result['passed']}/{result['total']} passed  {detail_str}\n"
            f"    CODE: {code_preview}"
        )

        if args.verbose:
            tqdm.write(f"\n  ── Problem {i} full output ──")
            tqdm.write(generated_code)
            for j, (assertion, outcome, error) in enumerate(
                zip(test_cases, result["details"], result["errors"])
            ):
                icon = "✓" if outcome == "pass" else "⏱" if outcome == "timeout" else "✗"
                tqdm.write(f"  [{icon}] test {j}: {assertion}")
                if error:
                    for err_line in error.splitlines():
                        tqdm.write(f"       {err_line}")
            tqdm.write("")

        all_results.append({
            "problem_idx": i,
            "passed": result["passed"],
            "total": result["total"],
            "details": result["details"],
            "errors": result["errors"],
            "generated_code": generated_code,
        })

    pass_rate = total_passed / max(total_tests, 1)
    problem_pass_rate = problems_solved / max(len(val_dataset), 1)

    print(f"{'─' * 60}")
    print(f"  Test cases passed : {total_passed}/{total_tests}  ({pass_rate:.1%})")
    print(f"  Problems solved   : {problems_solved}/{len(val_dataset)}  ({problem_pass_rate:.1%})")
    print(f"{'─' * 60}\n")

    eval_dir = Path(config["evaluation"]["output_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    results_path = eval_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "pass_rate": pass_rate,
            "problem_pass_rate": problem_pass_rate,
            "total_passed": total_passed,
            "total_tests": total_tests,
            "problems_solved": problems_solved,
            "num_problems": len(val_dataset),
            "per_problem": all_results,
        }, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
