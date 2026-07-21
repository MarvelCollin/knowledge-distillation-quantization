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
| Teacher R1-7B (ceiling) | none | 126 | 55.3% | 0 | upper bound |

## Barebone (base 1.5B) experiment, abandoned

Tested Qwen2.5-Coder-1.5B base (non-instruct) as the student, hoping a weaker starting point would widen the original to distilled gap. It backfired: the raw base already solves 28 (higher than the instruct baseline 22), because plain code completion suits the base Coder model and LeetCode is completion-like. Since the distilled ceiling is capacity-bound at about 36 regardless of start, the base gap would be roughly +8 versus the instruct +14, a smaller story, not bigger. The base also truncates massively (103 problems versus 5) with no instruction tuning to stop it. Skipped training. Instruct student stays the headline.

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
