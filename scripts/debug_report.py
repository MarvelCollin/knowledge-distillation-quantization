import sys
import json
import glob
import argparse
from pathlib import Path


def _idx(path):
    stem = Path(path).stem
    return int(stem) if stem.isdigit() else 1 << 30


def teacher_cache(cache_dir, cap):
    files = sorted(glob.glob(f"{cache_dir}/*.json"), key=_idx)
    total = passed = trunc = empty = 0
    fails = []
    toks = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        total += 1
        tp, tt = d.get("test_passed"), d.get("test_total")
        ok = bool(tt) and tp == tt
        if ok:
            passed += 1
        n = len(d.get("tokens") or [])
        toks.append(n)
        truncated = n >= cap - 50
        if truncated:
            trunc += 1
        if not (d.get("text") or "").strip():
            empty += 1
        if not ok:
            fails.append((Path(f).stem, tp, tt, n, truncated))
    toks.sort()
    print(f"== TEACHER CACHE: {cache_dir} ==")
    print(f"   files={total}  pass={passed} ({100 * passed / max(total, 1):.0f}%)  "
          f"truncated={trunc}  empty={empty}")
    if toks:
        print(f"   token len  p50={toks[len(toks) // 2]}  "
              f"p90={toks[int(len(toks) * 0.9)]}  max={toks[-1]}  (cap {cap})")
    if fails:
        print(f"   failing/truncated problems ({len(fails)}):")
        for stem, tp, tt, n, tr in fails[:40]:
            print(f"     idx {stem}: {tp}/{tt} tests  tokens={n}{'  TRUNCATED' if tr else ''}")
    print()


def eval_model(path):
    d = json.load(open(path))
    print(f"== {d.get('name', '?')}  ({Path(path).name}) ==")
    print(f"   pass@1={d.get('pass_at_1', 0):.1%}  pass@{d.get('k', '?')}={d.get('pass_at_k', 0):.1%}  "
          f"test={d.get('test_pass_rate', 0):.1%}  "
          f"solved={d.get('problems_solved')}/{d.get('num_problems')}  "
          f"truncated={d.get('truncated')}")
    print(f"   failure_counts (test-case level): {d.get('failure_counts')}")
    fails = [r for r in d.get("per_problem", []) if not r.get("solved")]
    if fails:
        print(f"   failing problems ({len(fails)}):")
        for r in fails[:50]:
            cats = r.get("categories") or []
            errs = r.get("errors") or []
            cat = next((c for c in cats if c and c != "pass"), "?")
            err = next((e for e in errs if e), "")
            err = " ".join(err.split())[:140]
            tr = "  TRUNCATED" if r.get("truncated") else ""
            print(f"     #{r.get('idx')} [{r.get('difficulty', '?')}] "
                  f"{r.get('num_passing', 0)}/{r.get('num_samples', 1)} pass  {cat}: {err}{tr}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Root-cause failure breakdown for teacher cache + eval results.")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--eval-dir", default="outputs/eval/intermediate")
    ap.add_argument("--cap", type=int, default=9600)
    args = ap.parse_args()

    if args.cache_dir:
        teacher_cache(args.cache_dir, args.cap)
    for p in sorted(glob.glob(f"{args.eval_dir}/*.json")):
        eval_model(p)


if __name__ == "__main__":
    main()
