# Project History and Results

Efficient Reasoning for Competitive Programming via Knowledge Distillation.

Student: Qwen2.5-Coder-1.5B. Teacher: R1-Distill-Qwen-7B. Eval: LeetCode 228 problems, 5 samples, temperature 0.6, plain prompt (same protocol for every model).

## Final result

| Model | Solved | pass@1 | pass@5 | test-case | truncated |
|---|---|---|---|---|---|
| Original 1.5B (untrained) | 22/228 | 4.1% | 9.6% | 15.3% | 5 |
| Distilled (two teacher, final) | 36/228 | 8.2% | 15.8% | 24.3% | 3 |
| Teacher R1-7B (ceiling) | 126/228 | 37.3% | 55.3% | 63.2% | 0 |

Distillation roughly doubles the student (pass@1 x2.0, pass@5 x1.6, +14 problems solved) and the student retains 28.6% of a teacher 4.7x its size.

The final model is the two teacher mix trained with seed 7. Weights at `outputs/final_last_seed7`.

## Files

- [methods.md](methods.md) how every method compared
- [findings.md](findings.md) why the number stopped where it did, and what would move it
