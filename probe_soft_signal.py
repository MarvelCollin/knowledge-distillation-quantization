import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import argparse
import math
import statistics as st

import yaml
from dotenv import load_dotenv

from src.data.dataset import PROMPT_TEMPLATE, load_problems
from src.teacher.local_teacher import LocalTeacherModel


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def distribution_stats(logprob_rows: list) -> tuple:
    eff_k = []
    top1 = []
    for row in logprob_rows:
        if not row:
            continue
        probs = [math.exp(v) for v in row.values()]
        total = sum(probs)
        if total <= 0.0:
            continue
        probs = [p / total for p in probs]
        entropy = -sum(p * math.log(p) for p in probs if p > 0.0)
        eff_k.append(math.exp(entropy))
        top1.append(max(probs))
    return eff_k, top1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--problems", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)

    problems = load_problems(config)[: args.problems]
    prompts = [PROMPT_TEMPLATE.format(text=p["text"]) for p in problems]
    print(f"Probing teacher soft signal on {len(prompts)} problems at temperature={args.temperature}...")

    teacher = LocalTeacherModel(
        model_path=config["teacher"]["local_model_path"],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=config["teacher"]["top_p"],
        top_logprobs=config["teacher"]["top_logprobs"],
        student_tokenizer=None,
        gpu_memory_utilization=config["teacher"]["gpu_memory_utilization"],
        max_model_len=config["teacher"]["max_model_len"],
    )
    results = teacher.get_responses_batch(prompts)
    teacher.shutdown()

    rows = []
    for r in results:
        rows.extend(r["logprobs"])
    eff_k, top1 = distribution_stats(rows)

    eff_k.sort()
    top1.sort()
    n = len(eff_k)
    median = lambda a: a[n // 2]
    frac = lambda a, f: sum(1 for x in a if f(x)) / n

    print()
    print(f"  positions measured            : {n}")
    print(f"  effective-k  median / mean    : {median(eff_k):.2f} / {st.mean(eff_k):.2f}")
    print(f"  top-1 prob   median / mean    : {median(top1):.3f} / {st.mean(top1):.3f}")
    print(f"  near-deterministic (top1>0.9) : {frac(top1, lambda x: x > 0.9):.1%}")
    print(f"  genuinely soft     (top1<0.7) : {frac(top1, lambda x: x < 0.7):.1%}")
    print()
    print(f"  baseline at temperature=0.6 (current cache): effective-k median 1.00, 77% near-deterministic")
    print()
    if median(eff_k) >= 2.0:
        print("  VERDICT: soft signal recovered — rebuild the cache at this temperature to make KD effective.")
    else:
        print("  VERDICT: still near-deterministic — the KD ceiling is intrinsic; train as reasoning-trace SFT.")


if __name__ == "__main__":
    main()
