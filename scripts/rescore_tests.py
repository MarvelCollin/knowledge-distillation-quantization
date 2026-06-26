import sys
import os
import re
import json
import glob
import random
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.dataset import load_problems
from src.utils.reasoning import extract_code
from src.evaluation.evaluator import run_test_cases


def main():
    ap = argparse.ArgumentParser(
        description="Re-run unit tests on FAILED teacher cache entries with the current harness; "
                    "recover any that now pass (e.g. after a harness/aliasing fix)."
    )
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--sample", type=int, default=0,
                    help="If >0, measure recovery on a random sample without writing.")
    ap.add_argument("--apply", action="store_true",
                    help="Re-test ALL failed entries and UPDATE the cache for recoveries.")
    args = ap.parse_args()

    config = load_config(args.config)
    cache_dir = config["data"]["teacher_cache_dir"]
    problems = load_problems(config)

    pp = re.compile(rb'"test_passed":\s*(\d+)')
    tt = re.compile(rb'"test_total":\s*(\d+)')
    failed = []
    for f in glob.glob(cache_dir + "/*.json"):
        idx = int(Path(f).stem)
        with open(f, "rb") as fh:
            sz = os.fstat(fh.fileno()).st_size
            fh.seek(max(0, sz - 400))
            tail = fh.read()
        a, b = pp.search(tail), tt.search(tail)
        ok = a and b and int(b.group(1)) > 0 and int(a.group(1)) == int(b.group(1))
        if not ok and idx < len(problems):
            failed.append(idx)

    total_failed = len(failed)
    if args.sample > 0:
        random.seed(0)
        failed = random.sample(failed, min(args.sample, len(failed)))

    def recheck(idx):
        d = json.load(open(f"{cache_dir}/{idx}.json"))
        code = extract_code(d.get("text", ""))
        r = run_test_cases(code, problems[idx]["test_cases"])
        ok = r["total"] > 0 and r["passed"] == r["total"]
        return idx, ok, r["total"], d

    recovered = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for idx, ok, total, d in ex.map(recheck, failed):
            if ok:
                recovered += 1
                if args.apply:
                    d["test_passed"] = total
                    d["test_total"] = total
                    with open(f"{cache_dir}/{idx}.json", "w") as fh:
                        json.dump(d, fh)

    mode = "APPLIED — cache updated" if args.apply else f"DRY-RUN sample (no write)"
    print(f"\n[{mode}]  re-tested {len(failed)} of {total_failed} failed  ->  "
          f"RECOVERED {recovered} now passing ({100 * recovered / max(len(failed), 1):.0f}%)")


if __name__ == "__main__":
    main()
