import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.problems import load_problems
from src.data.teacher_cache import rescore_failed_cache


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
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="Per-test-case timeout in seconds (default 30 — generous, so "
                         "slow-but-correct solutions mislabeled as timeout at 5s recover).")
    args = ap.parse_args()

    config = load_config(args.config)
    cache_dir = config["data"]["teacher_cache_dir"]
    problems = load_problems(config)

    recovered, tested, total_failed = rescore_failed_cache(
        cache_dir, problems, apply=args.apply, sample=args.sample, timeout=args.timeout
    )

    mode = "APPLIED — cache updated" if args.apply else "DRY-RUN (no write)"
    print(f"\n[{mode}]  re-tested {tested} of {total_failed} failed  ->  "
          f"RECOVERED {recovered} now passing ({100 * recovered / max(tested, 1):.0f}%)")


if __name__ == "__main__":
    main()
