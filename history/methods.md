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
| General base v2 (best-val ep2) | mix, lr 1e-5, 3ep, hard x2 | 22 | 9.6% | 6 | wrong ckpt picked by val loss |
| General base v2 (final_last ep3) | same run, last epoch ckpt | 28 | 12.3% | 1 | base best: easy 19, hard 5 |
| General base two stage | short CoT SFT then mix, init-from | 26 | 11.4% | 1 | ordering no better than mixing |
| General base v2 seed 42 | same v2 recipe, seed 42, final_last | 28 | 12.3% | 2 | 28 replicated, not a lucky draw |
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

## General base v2 (higher lr, longer schedule, hard oversample), regressed

Attempt to push the general base past 25: lr 5e-6 to 1e-5, epochs 2 to 3, warmup 100, plus hard R1 traces duplicated 2x (+256 samples). The best-val checkpoint (epoch 2, val 0.9375) scored only 22 (easy 14, trunc back to 6, degeneration loops returned), which first looked like a regression. But evaluating the last-epoch checkpoint (`outputs_general_v2/final_last`) told the real story: 28 solved, the base track best. Easy 19 (above v1's 18), hard 5 (from 2, now equal to the instruct distilled hard count), truncation 1, pass@5 12.3%.

Two lessons. First, val loss (pure R1 traces) measures R1 imitation, not solving; the two are decoupled. Best-val checkpoint selection picked an inferior mid-run snapshot, and epoch 2 to 3 went from 22 to 28 solves while val loss got worse. For the base track, always evaluate `final_last`, and treat val loss as a monitoring signal only. Second, the v2 recipe (higher lr, 3 epochs, hard oversample) does work for the base student once the right checkpoint is read: +3 total and +3 hard over v1. Base best is now `outputs_general_v2/final_last` at 28 (12 to 28, x2.33 relative).

## General base two stage curriculum, null

Tested whether ordering beats mixing for a raw base: stage 1 pure SFT on the 5385 Coder short CoT samples only (2 epochs, lr 1e-5, no R1 signal), then stage 2 full mix (2 epochs, lr 5e-6) initialised from the stage 1 last checkpoint. Result 26 solved (easy 18, medium 4, hard 4, trunc 1), between v1 (25) and v2 final_last (28), inside the noise band. Ordering is not better than mixing. Stage 1 did make the model structurally tidier (missing_function test failures 421 to 58, the lowest ever) but tidiness does not convert to solves. Side observation: after stage 1 the val loss on R1 traces was 1.2275, worse than the raw base 1.1402, then stage 2 recovered it to 1.0425, more evidence that R1 imitation and capability move independently.

Failure autopsy of v2 final_last vs two stage: both models fail identically. About 73% of failed samples are wrong_answer (clean, complete, running code with wrong logic), syntax errors are zero in both, and only about 12 of 1140 samples are near misses passing 90%+ of tests. The base recipes changed how the model learns, not what kind of model comes out. The only measurable difference is consistency: v2 final_last lands 71 fully correct samples vs 56 for two stage, same knowledge, more reliable execution. Three independent recipes (v1 mix 25, v2 28, two stage 26) converge to a 25 to 28 band, the base track mirror of the instruct 35 to 39 ceiling. Capacity bound, recipe exhausted; remaining item is a seed rerun of the v2 recipe to confirm 28 is not a lucky draw.

Seed rerun result (seed 42, identical v2 recipe, final_last): 28 solved again, pass@5 12.3% identical, val loss 0.9372 vs 0.9375, near identical training dynamics. 28 is the real level, not a draw. Caveat inside the total: the difficulty composition swings across seeds (seed 7: easy 19 / med 4 / hard 5; seed 42: easy 17 / med 9 / hard 2), so per-difficulty counts on the base track are noise at the plus or minus 3 to 5 level and should not be reported as findings, only the total. The hard=5 in seed 7 was partly a lucky composition. Base track final: 12 to 28 (x2.33), replicated across two seeds, band 25 to 28 across recipes. Track closed.

## PTQ INT8 (Phase 2): simulated quantization, first grid

Implementation note: real compressed INT8 checkpoints (llm-compressor W8A16 pack-quantized and W8A8 int-quantized) load in vLLM 0.6.6 but generate deterministic garbage on both the Marlin WNA16 and CUTLASS W8A8 kernel paths; the written checkpoints were verified byte-level correct (true int8 weights, sane per-channel scales), so the fault is vLLM 0.6.x's reader. Pivoted to simulated PTQ: per-channel symmetric int8 round-trip on every Linear weight (lm_head excluded), saved as plain bf16 (`scripts/quantize_int8.py`). Numerically equivalent to W8A16 inference for accuracy; the real INT8 footprint (2.12 GB vs 2.91 GB bf16) is cited from the compressed checkpoint. Verified: 86% of weights changed, mean |delta| 4e-4, embeddings untouched, coherent generation.

Results (228 problems, pass@5, one sampling draw per cell):

| Model | bf16 solved | INT8 solved | delta | test rate bf16 to INT8 |
|---|---|---|---|---|
| Instruct original | 22 | 24 | +2 | 15.3% to 19.7% |
| Instruct distilled (seed 7 two teacher) | 36 | 30 | -6 | 24.3% to 24.4% |
| General base distilled (v2 final_last) | 28 | 31 | +3 | 22.6% to 21.9% |

Reading: only the instruct distilled delta (-6) is at the edge of the noise band; original and base rows show no measurable degradation. The interesting detail is that the instruct distilled test case rate is unchanged (24.3 to 24.4 over ~19k test executions) while solves dropped: INT8 did not remove per-test competence, it removed consistency, knocking borderline problems off the all-tests-green edge.

Second sampling draw on the instruct distilled INT8 row (seed 42): 35 solved, pass@5 15.4%, test rate 25.8%. The INT8 draws {30, 35} overlap the bf16 seed band (35 to 38), so the -6 was mostly sampling noise. Phase 2 verdict: at 1.5B, per-channel W8 weight quantization causes no measurable degradation for any model (original, distilled, or base) on this eval. Clean deployment result, but it removes the contrast Phase 3 needs: if nothing degrades at INT8, QEAD-on vs QEAD-off cannot differ on solves. Decision: tighten to INT4 (group-128 symmetric, `--bits 4` in quantize_int8.py, the standard W4 granularity) to induce measurable degradation, then run the 2x2 there. Paper framing: robust at INT8, differentiated at INT4.

## PTQ INT4: degradation appears, and it is asymmetric

INT4 grid (group-128 symmetric W4, 228 problems, pass@5, one draw per cell):

| Model | bf16 | INT8 | INT4 | INT4 delta vs bf16 | trunc |
|---|---|---|---|---|---|
| Instruct original | 22 | 24 | 21 | -1 | 2 |
| Instruct distilled | 36 | 30 / 35 | 17 | -19 | 22 |
| Base distilled | 28 | 31 | 7 | -21 | 158 |

Headline finding: the original instruct model is essentially INT4-robust (-1, within noise) while both distilled models collapse (roughly halved or worse). Distillation dramatically increases quantization fragility at 4-bit. Failure mode is visible in truncation counts: base distilled hits 158/1140 truncated samples with degenerate output (including bracket spam deep enough to overflow CPython's parser stack, fixed in reasoning.py by treating MemoryError/RecursionError as unparseable). The distilled weight distributions evidently occupy sharper/more outlier-heavy configurations that group-128 W4 cannot represent; the pretrained instruct weights are flatter and survive.

This sets up Phase 3 cleanly: the current distilled checkpoints are QEAD-on, so the 2x2 question is whether QEAD-off distillation degrades even worse at INT4. If yes, QEAD partially mitigates a real fragility that distillation itself introduces (an honest and novel framing: distillation creates INT4 fragility, QEAD recovers part of it). If QEAD-off is the same, QEAD does not help and the paper reports the fragility finding itself, with INT8 robustness as the deployment recommendation.

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
