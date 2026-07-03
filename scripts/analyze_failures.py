#!/usr/bin/env python3
"""Re-score the cached FAILING teacher trajectories and bucket why they fail.

Categories aren't persisted in the cache, so this re-runs the test harness on each
non-passing cached trajectory and reports the dominant failure cause per trajectory
(syntax_error / wrong_answer / timeout / runtime_error / missing_function), plus a few
syntax_error examples if any exist.

Run inside Docker:
    sudo docker compose run --rm train python scripts/analyze_failures.py
"""
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from src.config import load_config
from src.data.dataset import load_problems
from src.evaluation.evaluator import run_test_cases
from src.utils.reasoning import extract_code

_CATS = ("syntax_error", "wrong_answer", "runtime_error", "missing_function", "timeout")


def _dominant(categories: list) -> str:
    non_pass = [c for c in categories if c and c != "pass"]
    if not non_pass:
        return "pass"
    return Counter(non_pass).most_common(1)[0][0]


def main() -> None:
    load_dotenv()
    config = load_config("config/config.yaml")
    cache_dir = Path(config["data"]["teacher_cache_dir"])
    problems = load_problems(config)

    failing = []
    for idx, prob in enumerate(problems):
        f = cache_dir / f"{idx}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        tp, tt = d.get("test_passed"), d.get("test_total")
        if not (tt is not None and tt > 0 and tp == tt):
            failing.append((idx, d.get("text", ""), prob["test_cases"]))

    print(f"Re-scoring {len(failing)} cached failing trajectories...")

    def _score(item):
        idx, text, tcs = item
        code = extract_code(text)
        r = run_test_cases(code, tcs)
        return idx, _dominant(r.get("categories", [])), r, code

    dominant_hist = Counter()
    any_hist = Counter()
    syntax_examples = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for idx, dom, r, code in ex.map(_score, failing):
            dominant_hist[dom] += 1
            for c in set(r.get("categories", [])):
                if c and c != "pass":
                    any_hist[c] += 1
            if dom == "syntax_error" and len(syntax_examples) < 3:
                syntax_examples.append((idx, code))

    print(f"\n{'=' * 60}")
    print(f"Failure cause breakdown  (dominant category per trajectory)")
    print(f"{'=' * 60}")
    n = max(len(failing), 1)
    for cat in _CATS:
        c = dominant_hist.get(cat, 0)
        print(f"  {cat:<16}: {c:>4}  ({c / n:5.1%})")
    other = sum(v for k, v in dominant_hist.items() if k not in _CATS and k != "pass")
    if other:
        print(f"  {'other':<16}: {other:>4}")
    reran_pass = dominant_hist.get("pass", 0)
    if reran_pass:
        print(f"\n  ⚠ {reran_pass} trajectories now PASS on re-score (flaky/timeout-bound tests) —")
        print(f"    create_datasets(rescore=True) recovers these as labels.")

    print(f"\n  (any-test-had-category, non-exclusive): "
          + ", ".join(f"{k}={v}" for k, v in any_hist.most_common()))

    if syntax_examples:
        print(f"\n  Syntax-error examples:")
        for idx, code in syntax_examples:
            snippet = (code or "").strip().splitlines()
            head = " ".join(snippet[:2])[:160] if snippet else "(no code extracted)"
            print(f"    prompt {idx}: {head}")


if __name__ == "__main__":
    main()
