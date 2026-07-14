#!/usr/bin/env python3
"""Report how many cached teacher prompts match the current builder; saves to outputs/eval/diagnostics/cache_prompts.txt."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from src.config import load_config
from src.data.problems import build_user_content, load_problems
from src.data.teacher_cache import fully_passed

REPORT_PATH = Path("outputs/eval/diagnostics/cache_prompts.txt")


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
        passing = fully_passed(d.get("test_passed"), d.get("test_total"))
        if d.get("prompt") == build_user_content(prob):
            matched += 1
            match_passing += int(passing)
        else:
            mismatched += 1
            mism_passing += int(passing)
            mism_failing += int(not passing)

    lines = [
        "=" * 60,
        f"Teacher cache prompt-match report  ({cache_dir})",
        f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        f"  problems (config max_samples) : {total}",
        f"  cached files                  : {cached}",
        f"  missing (never generated)     : {missing}",
        f"  prompt MATCHES current builder: {matched}   (passing: {match_passing})",
        f"  prompt MISMATCH (stale)       : {mismatched}",
        f"      of which passing (at risk of pointless re-gen): {mism_passing}",
        f"      of which failing                             : {mism_failing}",
    ]
    if mism_passing:
        lines += [
            "",
            f"  ⚠ {mism_passing} passing trajectories are cached under a STALE prompt. They",
            "    will be re-generated every run. The monotonic-save fix protects them from",
            "    being downgraded, but they still cost time and their logprobs are for the",
            "    old prompt text. If the change was intentional, consider regenerating; if",
            "    cosmetic, revert build_user_content to reuse the cache as-is.",
        ]
    else:
        lines += ["", "  ✓ No passing trajectory is stranded under a stale prompt."]

    report = "\n".join(lines)
    print("\n" + report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report + "\n")
    print(f"\n  Report saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
