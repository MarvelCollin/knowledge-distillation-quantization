import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import json
import yaml
import torch
import argparse

from tqdm import tqdm

from src.data.dataset import create_datasets
from src.evaluation.evaluator import run_test_cases
from src.student.model import StudentModel


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def generate_solution(student: StudentModel, prompt: str, max_new_tokens: int, device: torch.device) -> str:
    student.eval()
    with torch.no_grad():
        inputs = student.tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = student.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
        )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return student.tokenizer.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", required=True)
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

        generated_code = generate_solution(student, prompt, config["evaluation"]["max_new_tokens"], device)
        result = run_test_cases(generated_code, test_cases)

        total_passed += result["passed"]
        total_tests += result["total"]
        if result["passed"] == result["total"] and result["total"] > 0:
            problems_solved += 1

        status_icon = "✓" if result["passed"] == result["total"] else "✗"
        detail_str = "  ".join(
            f"[{'✓' if d == 'pass' else '✗' if d == 'fail' else '⏱'}]" for d in result["details"]
        )
        tqdm.write(
            f"  {status_icon} Problem {i:>2}  {result['passed']}/{result['total']} passed  {detail_str}"
        )

        all_results.append({
            "problem_idx": i,
            "passed": result["passed"],
            "total": result["total"],
            "details": result["details"],
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
