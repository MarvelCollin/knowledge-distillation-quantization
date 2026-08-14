import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import significance_tests as st

random.seed(12345)

NULL_TESTS = [
    ("5  Instruct KD gap @INT4      : orig INT4 vs distilled INT4", "oI_i4", ["dI_i4", "dI_i4b"]),
    ("6  Base KD gap @INT4          : orig INT4 vs distilled INT4", "oB_i4", ["dB_i4", "dB_i4b"]),
    ("B2 Base KD gap @GPTQ-W4 (real): orig vs distilled", "oB_g4", ["dB_g4"]),
    ("B3 Base KD gap @RTN-W4 (real) : orig vs distilled", "oB_rtn4fq", ["dB_rtn4fq"]),
    ("7  Instruct INT8 preservation : distilled bf16 vs INT8", "dI", ["dI_i8", "dI_i8b"]),
    ("8  QEAD @bf16 (instruct)      : on vs off", "dI", ["qoff_bf", "qoff_bfb"]),
    ("9  QEAD @INT4 (instruct)      : on vs off", "dI_i4", ["qoff_i4", "qoff_i4b"]),
    ("Q2 QAT-KD @RTN-W4 vs standard : standard vs QAT", "dB_rtn4fq", ["qat_rtn4", "qat_rtn4b"]),
]

def paired_diffs(A, B):
    keys = sorted(set(A) & set(B))
    return [(1 if B[k] else 0) - (1 if A[k] else 0) for k in keys]

def boot_ci(diffs, conf, iters=20000):
    n = len(diffs)
    if n == 0:
        return 0, 0, 0
    sums = sorted(sum(diffs[random.randrange(n)] for _ in range(n)) for _ in range(iters))
    lo_i = int(((1.0 - conf) / 2.0) * iters)
    hi_i = int((1.0 - (1.0 - conf) / 2.0) * iters) - 1
    return sum(diffs), sums[lo_i], sums[hi_i]

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--margin", type=float, default=None, help="Equivalence margin in problems. If given, report a TOST verdict against it as well as the data-driven delta*.")
    ap.add_argument("--iters", type=int, default=20000, help="Bootstrap resamples.")
    ap.add_argument("--n-problems", type=int, default=228,
                    help="Denominator for expressing delta* as a percentage.")
    args = ap.parse_args()

    L = {k: st.load(v) for k, v in st.F.items()}
    missing = [k for k, v in L.items() if v is None]
    if missing:
        print("skipping (file not found): %s\n" % ", ".join(sorted(missing)))

    print("=" * 96)
    print("EQUIVALENCE BOUNDS ON THE NULL RESULTS")
    print("90%% bootstrap CI = the TOST interval at alpha=0.05; delta* = max(|lo|,|hi|)")
    print("Read delta* as: a residual difference larger than this is excluded at 95%% confidence.")
    print("=" * 96)
    print()

    for title, a, blist in NULL_TESTS:
        print(title)
        if L.get(a) is None:
            print("    skipped (missing %s)\n" % a)
            continue
        for bk in blist:
            if L.get(bk) is None:
                print("    vs %-10s skipped (missing file)" % bk)
                continue
            diffs = paired_diffs(L[a], L[bk])
            obs, lo90, hi90 = boot_ci(diffs, 0.90, args.iters)
            _, lo95, hi95 = boot_ci(diffs, 0.95, args.iters)
            delta = max(abs(lo90), abs(hi90))
            pct = 100.0 * delta / float(args.n_problems)
            line = ("    vs %-10s dsolved=%+d | 90%%CI[%+d,%+d] | 95%%CI[%+d,%+d] "
                    "| delta*=%.0f problems (%.1f%% of %d)"
                    % (bk, obs, lo90, hi90, lo95, hi95, delta, pct, args.n_problems))
            if args.margin is not None:
                ok = (lo90 > -args.margin) and (hi90 < args.margin)
                line += " | TOST@%.3g: %s" % (args.margin, "EQUIVALENT" if ok else "not equivalent")
            print(line)
        print()

    print("-" * 96)
    print("Note: delta* is bounded below by the bootstrap's granularity (whole problems).")
    print("A null with delta* of 5 on 228 problems excludes any effect larger than ~2.2%.")

if __name__ == "__main__":
    main()
