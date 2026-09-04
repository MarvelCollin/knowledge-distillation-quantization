# Significance tests for "Efficient Reasoning for Competitive Programming"

Reproduces every paired significance test reported in the paper from the per-problem
execution records of each evaluated model.

## What this capsule does

`significance_tests.py` reads one JSON file per evaluated configuration. Each file lists, for
all 228 held-out LeetCode problems, whether that configuration solved the problem. The script
recomputes:

- Exact McNemar tests for every paired comparison in the paper
- Paired bootstrap 95% confidence intervals on the difference in problems solved
- The same tests restricted to the 116-problem verified subset

## Data

`data/` holds 83 evaluation records, one per configuration (student track x precision x seed).
Each entry is `{"idx": <problem id>, "solved": <bool>}`. Only the fields the tests consume are
included; the full generations they were derived from are large and are kept in the project
repository instead.

## Reproducing

```bash
EVAL_DIR=../data python code/significance_tests.py
```

`significance_tests.py` uses only the Python standard library. Runtime is a few seconds.

Results are deterministic: the bootstrap is seeded with `random.seed(12345)`.

## Related artifacts

Full source code, training pipeline, and weight-space measurement scripts:
https://github.com/MarvelCollin/knowledge-distillation-quantization at tag `v1.0-paper`

## License

Apache License 2.0.
