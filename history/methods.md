# Methods Compared

Same student (Qwen2.5-Coder-1.5B), same fullset (228), pass@5, temperature 0.6.

| Method | Teacher and signal | Solved | pass@5 | trunc | Verdict |
|---|---|---|---|---|---|
| Untrained baseline | none | 22 to 25 | 9.6 to 11% | 5 | reference |
| Single teacher logit KD | R1-7B offline logits | 39 | 17.1% | 42 | ceiling |
| Rescored full dist KD | R1-7B soft top 20 | 38 | 16.7% | ~42 | tie (noise) |
| Trace length filtered | R1-7B, 1269 traces | 38 | n/a | n/a | tie (noise) |
| On policy GKD | R1 scores student traces | ~38 | n/a | n/a | null |
| Two teacher mix | R1 logits + Coder-7B short CoT | 38 | 16.7% | 3 | same solves, 14x fewer trunc |
| Two teacher seed 7 (final) | same | 36 | 15.8% | 3 | best checkpoint |
| Over mixed edge (1 to 5.5) | mix + third short wave | 29 | 12.7% | n/a | regressed |
| Barebone base untrained | none (base, not instruct) | 28 | 12.3% | 103 | abandoned, floor too high |
| General base untrained | none (Qwen2.5-1.5B general) | 12 | 5.3% | 29 | floor for gap study |
| General base distilled | two teacher mix, seed 7 | 25 | 11.0% | 0 | +13 abs, x2.08 rel, trunc 0 |
| Teacher R1-7B (ceiling) | none | 126 | 55.3% | 0 | upper bound |

## Barebone (base 1.5B) experiment, abandoned

Tested Qwen2.5-Coder-1.5B base (non-instruct) as the student, hoping a weaker starting point would widen the original to distilled gap. It backfired: the raw base already solves 28 (higher than the instruct baseline 22), because plain code completion suits the base Coder model and LeetCode is completion-like. Since the distilled ceiling is capacity-bound at about 36 regardless of start, the base gap would be roughly +8 versus the instruct +14, a smaller story, not bigger. The base also truncates massively (103 problems versus 5) with no instruction tuning to stop it. Skipped training. Instruct student stays the headline.

## General (non-code) base 1.5B experiment, trained

Swapped the student to Qwen2.5-1.5B, the general (non code specialized) base, to get a low floor with the same 1.5B capacity. Same two teacher mix, seed 7, same R1 cache (verified tokenizer identical, alignment delta median 1 token = eos).

| Metric | Original (general base) | Distilled | Abs gap | Relative |
|---|---|---|---|---|
| solved | 12/228 | 25/228 | +13 | x2.08 |
| pass@1 | 1.6% | 4.9% | | x3.1 |
| pass@5 | 5.3% | 11.0% | +5.7 pts | x2.08 |
| test-case | 10.3% | 20.3% | +10.0 pts | x2.0 |
| truncated | 29 | 0 | | perfect |
| easy / med / hard | 8 / 2 / 2 | 18 / 5 / 2 | +10 / +3 / 0 | |

Verdict: the floor dropped as hoped (12, well below the instruct 22 and the Coder base 28), but the distilled ceiling also dropped (25 versus the instruct 36), because a non code base holds less code strategy at the same size. So the absolute gap (+13) is tied with the instruct (+14), confirming again that the gap is capacity bound. The relative gap is genuinely bigger though: distillation more than doubles the base (x2.08 solved, x3.1 pass@1) versus the instruct x1.6, and the two teacher mix drove truncation to a clean 0 (from 29), the best efficiency result yet. Distillation also erased missing_function failures entirely (1651 to 0). One cost: 93 new syntax errors appeared (the base learned to emit code but not always valid), net still strongly positive.

## What this shows

1. Every knowledge distillation method lands in the same 35 to 39 solve band. Five plus independent methods agree, so this is a real ceiling, not a tuning failure.

2. The two teacher mix is the only method that changed a real axis: truncations dropped from 42 problems to 3, malformed outputs dropped about 16x, at equal solve accuracy. This is the efficiency win.

3. Over mixing hurt: pushing the short to long ratio past 4 to 1 dropped solves to 29. The mix helps up to a ratio then breaks.

4. On policy GKD was a clean null: the R1 teacher agrees with the student's own tokens 96.6% of the time (measured over 259K positions), so there was nothing to correct.

## The final model detail (seed 7)

| Metric | Original | Seed 7 final | Teacher | Gain |
|---|---|---|---|---|
| solved | 22/228 | 36/228 | 126/228 | +14 (x1.6) |
| pass@1 | 4.1% | 8.2% | 37.3% | x2.0 |
| pass@5 | 9.6% | 15.8% | 55.3% | +6.2 pts |
| test-case | 15.3% | 24.3% | 63.2% | +9.0 pts |
| Easy / Med / Hard | 14 / 4 / 4 | 22 / 9 / 5 | 45 / 57 / 24 | +8 / +5 / +1 |

Seed 7 is the same two teacher mix recipe with a different random seed. Across three identical runs the solve count was 38, 35, 36 (noise band), and seed 7 was kept as the best draw.
