import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from src.config import load_config
from src.data.problems import load_test_problems
from src.evaluation.evaluator import run_test_cases
from src.utils.reasoning import extract_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--num-problems", type=int, default=228)
    parser.add_argument("--out", default="outputs/eval/broken_tests.json")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    problems = load_test_problems(args.num_problems, config["data"]["dataset_name"])

    def score(item):
        idx, prob = item
        code = extract_code(prob["code"]) or prob["code"]
        r = run_test_cases(code, prob["test_cases"])
        causes = sorted({c for c in r.get("categories", []) if c and c != "pass"})
        return {
            "idx": idx,
            "passed": r.get("passed", 0),
            "total": r.get("total", 0),
            "causes": causes,
            "difficulty": prob.get("difficulty", ""),
        }

    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(score, enumerate(problems)))

    broken = [r for r in results if r["passed"] < r["total"] or r["total"] == 0]
    verified = len(results) - len(broken)
    cause_hist = Counter(c for r in broken for c in r["causes"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(broken, indent=1))

    print(f"Canonical-solution verification over {len(results)} problems:")
    print(f"  verified (canonical passes all tests): {verified}")
    print(f"  broken (canonical fails >=1 test)    : {len(broken)}")
    print(f"  broken causes: {dict(cause_hist.most_common())}")
    print(f"  broken list -> {out}")


if __name__ == "__main__":
    main()
