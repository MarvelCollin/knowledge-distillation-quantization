import ast
import json
import re
from pathlib import Path

from datasets import load_dataset

from src.evaluation.evaluator import extract_signature
from src.utils.reasoning import build_signature_user_content

PROMPT_TEMPLATE = (
    "Write a solution in Python to solve the following problem.\n"
    "Your answer must be a Python function only. Do not use any other language.\n\n"
    "Problem: {text}\n\n"
)

DIFFICULTY_ORDER = {"Easy": 0, "Medium": 1, "Hard": 2}


def build_user_content(problem: dict) -> str:
    base = PROMPT_TEMPLATE.format(text=problem["text"])
    return build_signature_user_content(base, problem.get("signature", ""))


def extract_test_cases(test_str: str, entry_point: str) -> list:
    if not test_str.strip() or not entry_point:
        return []
    tree = ast.parse(test_str)
    check_func = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "check"
    )
    other_top_level = [n for n in tree.body if n is not check_func]
    cases = []
    for stmt in check_func.body:
        single_check = ast.FunctionDef(
            name="check",
            args=check_func.args,
            body=[stmt],
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        module = ast.Module(body=other_top_level + [single_check], type_ignores=[])
        ast.fix_missing_locations(module)
        src = ast.unparse(module)
        cases.append(src + f"\ncheck({entry_point})")
    return cases


def _parse_item(item: dict) -> dict:
    entry_point = (item.get("entry_point") or "").strip()
    ep_clean = re.search(r'(\w+)$', entry_point)
    entry_point = ep_clean.group(1) if ep_clean else entry_point
    return {
        "text": (item.get("problem_description") or "").strip(),
        "code": (item.get("response") or "").strip(),
        "test_cases": extract_test_cases(item.get("test") or "", entry_point),
        "entry_point": entry_point,
        "difficulty": (item.get("difficulty") or "").strip(),
    }


def load_problems(config: dict) -> list:
    dataset_name = config["data"]["dataset_name"]
    max_samples = config["data"]["max_samples"]

    cache_file = Path("cache") / f"problems_{dataset_name.replace('/', '_')}_{max_samples}.json"
    src_files = [
        Path(__file__),
        Path(__file__).parents[1] / "evaluation" / "evaluator.py",
    ]
    src_mtime = max(f.stat().st_mtime for f in src_files)
    if cache_file.exists() and cache_file.stat().st_mtime >= src_mtime:
        problems = json.loads(cache_file.read_text())
        print(f"Loaded {len(problems)} problems from processed cache ({cache_file.name}).")
        return problems

    print(f"Loading {dataset_name}...")
    raw = load_dataset(dataset_name, split="train")
    raw = raw.select(range(min(max_samples, len(raw))))

    problems = []
    for item in raw:
        p = _parse_item(item)
        if not p["test_cases"] or not p["code"] or not p["text"]:
            continue
        p["signature"] = extract_signature(p["test_cases"][0], p["entry_point"])
        problems.append(p)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(problems))
    print(f"  Loaded {len(problems)} problems with test cases (processed cache written).")
    return problems


def load_test_problems(n: int, dataset_name: str, difficulty: str = "all") -> list:
    allowed = {d.strip().lower() for d in difficulty.replace("+", ",").split(",") if d.strip()}
    filter_on = bool(allowed) and "all" not in allowed
    diff_label = f"{'+'.join(sorted(allowed))} " if filter_on else ""
    print(f"Loading {n} {diff_label}problems from {dataset_name} test split (separate from training data)...")
    raw = load_dataset(dataset_name, split="test")
    problems = []
    for item in raw:
        p = _parse_item(item)
        if filter_on and p["difficulty"].lower() not in allowed:
            continue
        if not p["test_cases"] or not p["text"]:
            continue
        problems.append(p)
        if len(problems) >= n:
            break
    print(f"  Loaded {len(problems)} problems.")
    return problems
