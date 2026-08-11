import json
import os
import random
from math import comb

random.seed(12345)
D = "outputs_evalplus_reasoning"

def load(dataset, name):
    path = os.path.join(D, dataset, f"{name}_eval_results.json")
    d = json.load(open(path))
    return {tid: samples[0]["plus_status"] == "pass" for tid, samples in d["eval"].items()}

def mcnemar_exact(A, B):
    keys = sorted(set(A) & set(B))
    b = sum(1 for k in keys if A[k] and not B[k])
    c = sum(1 for k in keys if B[k] and not A[k])
    n = b + c
    if n == 0:
        return b, c, 1.0
    m = min(b, c)
    tail = sum(comb(n, i) for i in range(m + 1)) * (0.5 ** n)
    return b, c, min(1.0, 2 * tail)

def boot_ci(A, B, iters=20000):
    keys = sorted(set(A) & set(B))
    diffs = [(1 if B[k] else 0) - (1 if A[k] else 0) for k in keys]
    n = len(diffs)
    sums = sorted(sum(diffs[random.randrange(n)] for _ in range(n)) for _ in range(iters))
    return sum(diffs), sums[int(0.025 * iters)], sums[int(0.975 * iters)]

NAMES = dict(
    o_bf="Qwen_Qwen2.5-1.5B",
    d_bf="outputs_general_v2_final_last",
    o_w4="outputs_fq_gptq4_original",
    d_w4="outputs_fq_gptq4_distilled",
)

TESTS = [
    ("HumanEval+ bf16 gap : orig vs distilled", "humaneval", "o_bf", "d_bf"),
    ("HumanEval+ W4 gap   : orig vs distilled", "humaneval", "o_w4", "d_w4"),
    ("MBPP+ bf16 gap      : orig vs distilled", "mbpp", "o_bf", "d_bf"),
    ("MBPP+ W4 gap        : orig vs distilled", "mbpp", "o_w4", "d_w4"),
]

def main():
    print("=" * 92)
    for title, dataset, a_key, b_key in TESTS:
        A = load(dataset, NAMES[a_key])
        B = load(dataset, NAMES[b_key])
        n_pass_a, n_pass_b = sum(A.values()), sum(B.values())
        b, c, p = mcnemar_exact(A, B)
        obs, lo, hi = boot_ci(A, B)
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(title)
        print(f"    pass {n_pass_a}/{len(A)} -> {n_pass_b}/{len(B)} | discordant b={b:>2} c={c:>2} "
              f"| McNemar p={p:.4g} [{star}] | dpass={obs:+d} 95%CI[{lo:+d},{hi:+d}]")
        print()

if __name__ == "__main__":
    main()
