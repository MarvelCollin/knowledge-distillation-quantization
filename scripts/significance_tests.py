import json, os, random
from math import comb

random.seed(12345)
D = os.environ.get("EVAL_DIR", "outputs/eval/intermediate")


def load(f):
    d = json.load(open(os.path.join(D, f)))
    return {p["idx"]: bool(p["solved"]) for p in d["per_problem"]}


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


F = dict(
    oI="Student_original_Qwen2.5-Coder-1.5B-Instruct.json",
    dI="Student_distilled_Qwen2.5-Coder-1.5B-Instruct.json",
    dI_i8="Student_distilled_Qwen2.5-Coder-1.5B-Instruct_int8.json",
    dI_i8b="Student_distilled_Qwen2.5-Coder-1.5B-Instruct_int8_s42.json",
    dI_i4="Student_distilled_Qwen2.5-Coder-1.5B-Instruct_int4.json",
    dI_i4b="Student_distilled_Qwen2.5-Coder-1.5B-Instruct_int4_s42.json",
    oI_i4="Student_original_Qwen2.5-Coder-1.5B-Instruct_int4.json",
    qoff_bf="Student_distilled_Qwen2.5-Coder-1.5B-Instruct_qead_off_bf16.json",
    qoff_bfb="Student_distilled_Qwen2.5-Coder-1.5B-Instruct_qead_off_bf16_s42.json",
    qoff_i4="Student_distilled_Qwen2.5-Coder-1.5B-Instruct_qead_off_int4.json",
    qoff_i4b="Student_distilled_Qwen2.5-Coder-1.5B-Instruct_qead_off_int4_s42.json",
    oB="Student_original_Qwen2.5-1.5B.json",
    dB="Student_distilled_Qwen2.5-1.5B.json",
    dB_i4="Student_distilled_Qwen2.5-1.5B_int4.json",
    dB_i4b="Student_distilled_Qwen2.5-1.5B_base_distilled_int4_s42.json",
    dB_i8="Student_distilled_Qwen2.5-1.5B_int8.json",
    oB_i4="Student_original_Qwen2.5-1.5B_int4.json",
    dB_g4="Student_distilled_Qwen2.5-1.5B_gptq4.json",
    oB_g4="Student_original_Qwen2.5-1.5B_gptq4.json",
    dB_rtn4fq="Student_distilled_Qwen2.5-1.5B_rtn4fq.json",
    qat_bf="Student_distilled_Qwen2.5-1.5B_qatbf16.json",
    qat_bfb="Student_distilled_Qwen2.5-1.5B_qatbf16_s42.json",
    qat_rtn4="Student_distilled_Qwen2.5-1.5B_qatrtn4.json",
    qat_rtn4b="Student_distilled_Qwen2.5-1.5B_qatrtn4_s42.json",
)

TESTS = [
    ("1  Instruct KD @bf16      : orig(22) vs distilled(36)", "oI", ["dI"]),
    ("2  Base KD @bf16          : orig(12) vs distilled(28)", "oB", ["dB"]),
    ("3  Instruct INT4 erasure  : distilled bf16(36) vs INT4(17/23)", "dI", ["dI_i4", "dI_i4b"]),
    ("4  Base INT4 erasure      : distilled bf16(28) vs INT4(7/5)", "dB", ["dB_i4", "dB_i4b"]),
    ("5  Instruct gap @INT4     : orig INT4(21) vs distilled INT4(17/23)", "oI_i4", ["dI_i4", "dI_i4b"]),
    ("6  Base gap @INT4         : orig INT4(6) vs distilled INT4(7/5)", "oB_i4", ["dB_i4", "dB_i4b"]),
    ("7  Instruct INT8 preserve : distilled bf16(36) vs INT8(30/35)", "dI", ["dI_i8", "dI_i8b"]),
    ("8  QEAD bf16 (instruct)   : on(36) vs off(30/35)", "dI", ["qoff_bf", "qoff_bfb"]),
    ("9  QEAD INT4 (instruct)   : on(17/23) vs off(16/20)", "dI_i4", ["qoff_i4", "qoff_i4b"]),
    ("B1 Real GPTQ4 base        : distilled bf16(28) vs GPTQ4(1)", "dB", ["dB_g4"]),
    ("B2 GPTQ4 gap (base)       : orig GPTQ4(1) vs distilled GPTQ4(1)", "oB_g4", ["dB_g4"]),
    ("Q1 QAT bf16 vs standard KD bf16  : dB(28) vs QAT-KD bf16(10/11)", "dB", ["qat_bf", "qat_bfb"]),
    ("Q2 QAT RTN4 vs standard KD RTN4  : rtn4fq(2) vs QAT-KD RTN4(3/1)", "dB_rtn4fq", ["qat_rtn4", "qat_rtn4b"]),
]

def main():
    L = {k: load(v) for k, v in F.items()}
    nsolv = lambda k: sum(L[k].values())
    print("=" * 92)
    for title, a, blist in TESTS:
        print(title)
        for bk in blist:
            b, c, p = mcnemar_exact(L[a], L[bk])
            obs, lo, hi = boot_ci(L[a], L[bk])
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"    vs {bk:<8} solved {nsolv(a):>2}->{nsolv(bk):<2} | discordant b={b:>2} c={c:>2} "
                  f"| McNemar p={p:.4g} [{star}] | dsolved={obs:+d} 95%CI[{lo:+d},{hi:+d}]")
        print()

if __name__ == "__main__":
    main()
