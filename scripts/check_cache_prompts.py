#!/usr/bin/env python3
"""Report how many cached teacher trajectories still match the current build_user_content.

A mismatch means the prompt text changed since the trajectory was cached (e.g. a code
edit to build_user_content / PROMPT_TEMPLATE). _is_valid then treats the entry as invalid
and re-generates it — which, for a previously-passing entry, risks a stochastic rejection
retry failing and (before the monotonic-save fix) overwriting good data.

Run inside Docker:
    sudo docker compose run --rm train python scripts/check_cache_prompts.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from src.config import load_config
from src.data.dataset import build_user_content, load_problems


def main() -> None:
    load_dotenv()
    config = load_config("config/config.yaml")
    cache_dir = Path(config["data"]["teacher_cache_dir"])
    problems = load_problems(config)

    total = len(problems)
    cached = matched = mismatched = 0
    mism_passing = mism_failing = 0
    match_passing = 0
    missing = 0

    for idx, prob in enumerate(problems):
        f = cache_dir / f"{idx}.json"
        if not f.exists():
            missing += 1
            continue
        cached += 1
        d = json.loads(f.read_text())
        tp, tt = d.get("test_passed"), d.get("test_total")
        passing = tt is not None and tt > 0 and tp == tt
        if d.get("prompt") == build_user_content(prob):
            matched += 1
            match_passing += int(passing)
        else:
            mismatched += 1
            mism_passing += int(passing)
            mism_failing += int(not passing)

    print(f"\n{'=' * 60}")
    print(f"Teacher cache prompt-match report  ({cache_dir})")
    print(f"{'=' * 60}")
    print(f"  problems (config max_samples) : {total}")
    print(f"  cached files                  : {cached}")
    print(f"  missing (never generated)     : {missing}")
    print(f"  prompt MATCHES current builder: {matched}   (passing: {match_passing})")
    print(f"  prompt MISMATCH (stale)       : {mismatched}")
    print(f"      of which passing (at risk of pointless re-gen): {mism_passing}")
    print(f"      of which failing                             : {mism_failing}")
    if mism_passing:
        print(f"\n  ⚠ {mism_passing} passing trajectories are cached under a STALE prompt. They")
        print(f"    will be re-generated every run. The monotonic-save fix protects them from")
        print(f"    being downgraded, but they still cost time and their logprobs are for the")
        print(f"    old prompt text. If the change was intentional, consider regenerating; if")
        print(f"    cosmetic, revert build_user_content to reuse the cache as-is.")
    else:
        print(f"\n  ✓ No passing trajectory is stranded under a stale prompt.")


if __name__ == "__main__":
    main()
