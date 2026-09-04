#!/usr/bin/env bash
# pipefail matters: without it a failing python is masked by tee succeeding,
# and the run would report success while producing an empty result file.
set -euo pipefail

# Code Ocean runs this with the working directory at /code. Relative paths keep
# it working when the capsule is downloaded and run locally.
DATA=../data

# significance_tests.py skips records it cannot open, so a missing or unmounted
# data directory would otherwise produce an empty run that still exits zero.
count=$(find "$DATA" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
if [ "$count" -lt 80 ]; then
    echo "error: expected at least 80 evaluation records in $DATA, found $count" >&2
    exit 1
fi

EVAL_DIR="$DATA" python3 -u significance_tests.py | tee ../results/significance_tests.txt
