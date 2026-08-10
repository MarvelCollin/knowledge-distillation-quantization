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
| Instruct QEAD-off | mix, seed 7, uniform token weights | 32.5 (30, 35) | 15.4% | 0-1 | null vs QEAD-on 36 |
| General base QEAD-off | v2 recipe, seed 7, uniform weights | 26 | 11.4% | 0 | null vs QEAD-on 28 |
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

Headline finding: the original instruct model is essentially INT4-robust (-1, within noise) while both distilled models collapse (roughly halved or worse). Distillation dramatically increases quantization fragility at 4-bit. Failure mode is visible in truncation counts: base distilled hits 158/1140 truncated samples with degenerate output (including bracket spam deep enough to overflow CPython's parser stack, fixed in reasoning.py by treating MemoryError/RecursionError as unparseable). The mechanism was assumed at the time to be sharper / more outlier-heavy distilled weight distributions that group-128 W4 cannot represent, with flatter pretrained weights surviving. **That assumption was later measured directly and falsified** — see "Weight-level mechanism" below. Note also that the backbone-survival half of the claim holds only on the instruct track (22 to 21); the base original halved (12 to 6), so it did not survive either.

This sets up Phase 3 cleanly: the current distilled checkpoints are QEAD-on, so the 2x2 question is whether QEAD-off distillation degrades even worse at INT4. If yes, QEAD partially mitigates a real fragility that distillation itself introduces (an honest and novel framing: distillation creates INT4 fragility, QEAD recovers part of it). If QEAD-off is the same, QEAD does not help and the paper reports the fragility finding itself, with INT8 robustness as the deployment recommendation.

## INT4 grid completed: quantization erases the distillation gain

Strengthening runs: second INT4 draw on instruct distilled (seed 42) scored 23 (first draw 17, bf16 band 35 to 38), confirming the collapse is real, roughly 36 to 20. Base original rows: INT8 11 (bf16 12, free like everywhere else), INT4 6 (bf16 12, halved, truncation 29 to 70).

Full 2x3 grid (solved / 228):

| Model | bf16 | INT8 | INT4 |
|---|---|---|---|
| Instruct original | 22 | 24 | 21 |
| Instruct distilled | 36 | 30 / 35 | 17 / 23 |
| Base original | 12 | 11 | 6 |
| Base distilled | 28 | 31 | 7 |

Unified finding: at bf16 the distilled models lead the originals by +14 (instruct) and +16 (base). At INT4 the gap vanishes on both tracks: instruct 20 vs 21, base 7 vs 6, statistically identical. W4 quantization erases the distillation gain completely, while the pretrained backbone capability survives (instruct fully at -1, base partially at -6). INT8 preserves the gain fully on all four models. The knowledge added by KD is stored in weight structure that 4-bit rounding destroys; this is a sharper claim than "distilled models degrade more" and is the paper's headline pending the Phase 3 QEAD-off ablation (does error-aware weighting change where the knowledge is stored, or is the fragility inherent to KD).

## Failure mixture across formats (base model, 116,170 test executions per format)

Per test execution category share:

| Category | bf16 original | bf16 distilled | INT8 distilled | INT4 distilled |
|---|---|---|---|---|
| pass | 10.9% | 20.7% | 20.4% | 5.2% |
| wrong_answer | 60.7% | 62.7% | 62.7% | 38.2% |
| runtime_error | 12.7% | 12.7% | 13.8% | 30.3% |
| missing_function | 14.0% | 0.9% | 0.6% | 21.4% |
| syntax_error | 0.1% | 0.0% | 0.1% | 1.7% |
| timeout | 1.6% | 3.0% | 2.4% | 3.2% |

Per sample (dominant failure of each of the 1140 generations): fully passed 18 / 59 / 62 / 15; empty extracted code 118 / 0 / 0 / 155.

Two conclusions. First, the INT8 column is indistinguishable from bf16 distilled in every category (wrong_answer 62.7 vs 62.7, fully passed 59 vs 62): the +3 solve delta has no mechanism behind it, confirming it is sampling noise, not improvement. Second, INT4 changes the kind of failure, not just the amount: wrong_answer (the competent failure mode, clean running code with wrong logic) drops to 38.2% while structural failures explode (missing_function 0.9 to 21.4%, runtime errors 12.7 to 30.3%, syntax errors 39 to 1986 executions, empty extractions 0 to 155). Distillation had specifically eliminated missing_function and empty output; INT4 resurrects both above the untrained original's level. bf16/INT8 distilled fails like a programmer with wrong ideas; INT4 distilled fails like a broken text generator. This is the quantitative core of the fragility claim (fig5).

## Phase 3: the QEAD ablation is a null on both tracks

The 2x2 (QEAD-on vs QEAD-off) x (bf16 vs INT4) was run on both student tracks. QEAD-off means uniform per-token weights over the response instead of quantization-error weights; the KLD path is byte-identical between modes (unit tested), same teacher cache, same seed 7, same recipe per track. INT8 was dropped from the matrix because Phase 2 showed nothing degrades there, so there is no contrast to measure.

| Track | Variant | bf16 | INT4 | Absolute drop | Retention |
|---|---|---|---|---|---|
| Instruct | QEAD-on | 36 | 20 (draws 17, 23) | -16 | 55.6% |
| Instruct | QEAD-off | 32.5 (draws 30, 35) | 18 (draws 16, 20) | -14.5 | 55.4% |
| Base | QEAD-on | 28 | 6 (draws 7, 5) | -22 | 21.4% |
| Base | QEAD-off | 26 | 8 | -18 | 30.8% |

Two separate claims fail here, and both fail cleanly.

First, QEAD does not improve distillation quality. The instruct QEAD-off first draw was 30 against QEAD-on's 36, a +6 that sat at the edge of the noise band and looked like it might be real. The second draw (seed 42) came back 35, so the draws are {30, 35} against 36: indistinguishable, and overlapping the same band the QEAD-on training seeds themselves occupied (38, 35, 36). The base track had already said the same thing more quietly at 28 vs 26. Two tracks, two nulls.

Second, QEAD does not confer quantization robustness, which was the original point of the method. On the instruct track, where all four cells are now double-drawn, retention is 55.6 percent for QEAD-on and 55.4 percent for QEAD-off. The two students keep the same fraction of their bf16 accuracy to within two tenths of a point.

The base track's apparent 9.4 point gap in the opposite direction (21.4 vs 30.8) is a single-draw artifact. It rests on one QEAD-off INT4 measurement of 8 against a two-draw mean of 6, and the instruct cells now show directly that single INT4 draws scatter by 4 to 6 solves ({17, 23} and {16, 20}). A lone 8 against a mean of 6 is inside that scatter. Where the sampling is adequate the two variants are identical; where it is thin the difference points the wrong way for the hypothesis anyway.

The cleanest way to see it is that the instruct INT4 gap (20 vs 16) is just the bf16 gap carried forward, not widened. If error-aware weighting were protecting the weights from 4-bit rounding, quantization would separate the two students further apart than they already are. It does not.

Verdict: QEAD is a complete null, and the paper reports it as the negative control that rules out the obvious mitigation. This does not weaken the headline finding, it sharpens it. The INT4 erasure is not an artifact of one training recipe, and it is not fixable by reweighting which tokens the distillation loss attends to. The knowledge that quantization destroys is not concentrated in the tokens QEAD upweights.

## Weight-level mechanism: the KD update is smaller than one 4-bit quantization step

Measured with `scripts/analyze_weight_quant.py` across all 196 Linear projection weights per model (lm_head and embeddings excluded, matching `quantize_int8.py`, whose quantizer is imported rather than reimplemented so the error measured is exactly the error the eval grid experienced). Four checkpoint pairs, CPU only, zero eval runs.

### The sharpness hypothesis is false

| Track | Statistic | Original | Distilled |
|---|---|---|---|
| Base | excess kurtosis | 2.50648 | 2.50679 |
| Base | outlier ratio (>4 sigma) | 0.00143645 | 0.00143645 |
| Base | W4 error (nmse) | 0.0161585 | 0.0161443 |
| Instruct | excess kurtosis | 3.50113 | 3.50118 |
| Instruct | outlier ratio (>4 sigma) | 0.00149576 | 0.00149582 |
| Instruct | W4 error (nmse) | 0.0164836 | 0.0164809 |

The distributions are identical to five significant figures on both tracks, and the distilled models quantize marginally *better* than their originals (base: distilled quantizes worse in 0 of 196 layers). Distillation does not sharpen the weights, does not add outliers, and does not make the model harder to quantize.

### What actually happens

The distillation update is tiny compared to the quantization perturbation, and far below the 4-bit grid resolution.

| Pair | KD update | W4 perturbation | W8 perturbation | cos W4 | cos W8 |
|---|---|---|---|---|---|
| Base (original to distilled) | 0.892% | 12.67% (14.2x) | 0.983% (1.1x) | 0.194 | 0.718 |
| Instruct (original to distilled) | 0.459% | 12.80% (27.9x) | 1.008% (2.2x) | 0.110 | 0.589 |
| Base QEAD-on to QEAD-off | 0.323% | (39.2x) | (3.0x) | 0.144 | 0.464 |
| Instruct QEAD-on to QEAD-off | 0.141% | (90.5x) | (7.1x) | 0.084 | 0.347 |

All magnitudes are relative Frobenius norm, averaged over layers. `cos` is the cosine between the pre-quantization update (W_distilled - W_original) and the same difference after both models are quantized — how much of the update's direction survives rounding.

Expressed in quantization-step units (step inferred as rms error times sqrt(12)), the picture is unambiguous:

| Pair | KD update as fraction of W8 step | of W4 step |
|---|---|---|
| Base | 26% | 2.0% |
| Instruct | 13% | 1.0% |
| Base QEAD on/off | 9.4% | 0.74% |
| Instruct QEAD on/off | 4.0% | 0.32% |

**Distillation moves the weights by roughly 2% of one 4-bit quantization step, but 26% of an 8-bit step.** At W8 the update is a meaningful fraction of a step, crosses bin boundaries often, and is transmitted (cos 0.59 to 0.72). At W4 it is far below the grid resolution, almost never changes which bin a weight lands in, and is not transmitted (cos 0.08 to 0.19, where 0 is an unrelated direction).

Per-weight fate at W4 on the base track: 45.8% of weights have their KD change erased outright, and only 1.35% are amplified beyond 2x. But that 1.35% is enough to make the difference between the two *quantized* models 4.14% of weight norm — 4.6x larger than the true 0.892% update it replaced. So W4 does not shrink the distillation update, it substitutes a larger and differently-directed perturbation built from incidental bin flips. This explains a detail the earlier framing could not: the INT4 distilled model does not behave like the untrained original, it behaves worse in kind (base distilled INT4 truncation 158 vs base original INT4 70, missing_function 21.4% vs the original's 14.0%). It is the original plus a large structureless perturbation, not the original restored.

At W8 the amplified fraction is higher (7.3%) precisely because the step is small enough for the KD nudge to routinely cross a boundary — which is why the update survives there.

### Why QEAD could not have worked

The weight difference between the QEAD-on and QEAD-off checkpoints is 39x (base) to 90x (instruct) smaller than the W4 perturbation, and under 1% of a single quantization step. Whatever error-aware token weighting did to the weights sits roughly two orders of magnitude below the noise floor it was meant to protect against.

The Phase 3 null is therefore not "the mitigation was tried and did not help" but "the mitigation was incapable of mattering at this precision, and here is the measurement." Reported that way it is a considerably stronger negative result.

### Internal consistency and scope

The base KD update (0.892%) is roughly twice the instruct update (0.459%), matching that the base student had more to learn (12 to 28, x2.33) than the instruct student (22 to 36, x1.64).

Scope limit: this is a weight-space measurement. That a destroyed weight-space direction causes the observed behavioural collapse is an inference, not a measurement. The functional half needs an activation-side probe (output divergence against the bf16 reference on a fixed prompt set), which is not yet run.

Open caveat: the instruct pair used `outputs/final_last`, since `outputs/final_last_seed7` named in this repo no longer exists on disk. If that directory was overwritten by a later run, the instruct row describes a checkpoint whose eval numbers are not the recorded 36. The base track is unaffected.

## Calibrated PTQ does not rescue the distilled model (GPTQ vs RTN, base track)

The recorded W4 grid used naive round-to-nearest, so the erasure could have been an artifact of the weakest available quantizer rather than a property of 4-bit quantization. GPTQ (Hessian-weighted error compensation, 256 calibration sequences drawn from the train-split R1 traces) was run against RTN in the same toolchain (`llm-compressor`, `scripts/quantize_real.py --save-mode fake`), both dequantized back to bf16 so they run on the same inference path that produced every other number here.

| Precision | Original | Distilled | KD gap (solved) | Original test rate | Distilled test rate | KD gap (test rate) |
|---|---|---|---|---|---|---|
| bf16 | 12 | 28 | +16 | 10.3% | 22.0% | +11.7 pts |
| RTN-W4 | 2 | 2 | 0 | 1.1% | 1.3% | +0.2 pts |
| GPTQ-W4 | 3 | 2 | -1 | 1.4% | 1.4% | 0.0 pts |

Calibration buys essentially nothing at this scale: RTN and GPTQ land within a solve of each other on both rows, and the distilled model sits at or below its own untrained baseline under both. The KD gap is erased on solve count and on test-case rate alike, and the test-case rates agree to the first decimal, which removes the resolution objection that applies to near-floor solve counts.

This is the strongest form of the headline. The erasure is not an artifact of naive rounding; it survives an error-compensating, calibration-based quantizer.

Methodological caveats, all of which belong in the paper:

- `llm-compressor`'s `W4A16` is not numerically identical to `int_roundtrip_` despite both being nominally group-128 symmetric. Base distilled scores 2 (test 1.3%) under the former and 7 (test 4.8%) under the latter. They agree qualitatively — the gain is destroyed either way — but not quantitatively, so this table is read within-toolchain and the recorded simulated grid stands separately.
- Real compressed-tensors checkpoints could not be used. vLLM 0.6.6 loads them but its Marlin path corrupts generation with systematically duplicated closing delimiters followed by repetition loops, on both RTN and GPTQ, while the same weights saved as dequantized bf16 generate cleanly. That is a runtime defect, not a result, and it is why deployment memory and throughput measurements remain unavailable.
- Calibration data is drawn from the LeetCode train split via the passing R1 traces; the 228 eval problems come from the test split. Disjoint by construction.

**Significance (added 2026-08-10, `scripts/significance_tests.py`):** the KD gap at both real quantizers is now a confirmed null, not an eyeballed one. RTN-W4 gap (orig 2 vs distilled 2): McNemar p=1, Δsolved 0, 95% CI [−4,+4]. GPTQ-W4 gap (orig 3 vs distilled 2): McNemar p=1, Δsolved −1, 95% CI [−4,+2]. Both bracket zero comfortably, consistent with the bf16 gap's CI ([+6,+26]) being nowhere near either.

## Stage 8 pre-registration: quantization-aware distillation (written 2026-08-04, before any run)

Recorded before running so the success criterion cannot be chosen after seeing the numbers. The QEAD ablation cost weeks partly because a first draw of 30 looked like a real +6 until the second draw returned 35; the criterion below is fixed in advance and both cells are double-drawn from the start.

**Intervention.** Fake-quantize the student's weights in the forward pass (group-128 symmetric W4, straight-through estimator on the backward), so the optimiser searches over weights that survive rounding. Implemented in `src/distillation/qat.py`, gated by `training.qat` / `--qat`, with the off path verified byte-identical (`scripts/smoke_qat.py`).

**Why this and not weight-distribution regularization.** The mechanism analysis found no weight statistic that separates distilled from original models — kurtosis and outlier ratio are identical to five significant figures — so there is nothing to regularize toward. What it did find is that the update sits at ~2% of a W4 step. The intervention must therefore act on where the weights land relative to the grid, which is what training through the quantizer does.

**The 2x2** (base track, `config_general.yaml` v2 recipe, seed 7, `final_last`):

| Variant | bf16 | W4 | Retention |
|---|---|---|---|
| Standard KD (recorded) | 28 | 2 (llm-compressor RTN) / 7 (simulated RTN) | 7% / 25% |
| Quantization-aware KD | ? | ? | ? |

**Success criterion, pre-declared.** QAT-KD succeeds if its W4 cell beats standard KD's W4 cell by more than the observed draw scatter, on **test-case pass rate** rather than solve count — near-floor solve counts (2 of 228) have no resolution, and the base-track cells sit there. Standard KD at W4 scores 1.3% test rate against 22.0% at bf16. A W4 test rate above roughly 8% would be a real recovery; anything under about 3% is noise.

**Pre-declared honest failure case.** QAT may cost bf16 accuracy while improving retention. Deciding now: a result of bf16 24 / W4 18 **counts as success**, because the deployable artifact is the W4 model and the paper's claim is about what survives quantization, not about the bf16 ceiling. A result that improves retention only by lowering the bf16 score without raising the W4 score is **not** success — that is the trap the QEAD ablation fell into, where a bf16 gap carried forward unchanged was mistaken for protection.

**Both cells double-drawn** (seeds 1234 and 42) before any conclusion is recorded.

## Stage 8 result: quantization-aware distillation fails (run 2026-08-10)

Trained per the pre-registration above (`--qat --offline`, base track, seed 7, 3 epochs, `outputs_qat_base/final_last`, 9h05m wall-clock, best_val_loss 0.9939). Quantized with the same RTN-W4 fake-quant path as every other real-PTQ cell in this project. Both bf16 and W4 double-drawn (seeds 1234, 42) before conclusions, as declared.

| Variant | bf16 (solved / test rate) | W4, RTN real (solved / test rate) | Retention (test rate) |
|---|---|---|---|
| Standard KD (recorded) | 28 / 22.0% | 2 / 1.3% | 5.9% |
| Quantization-aware KD | 10, 11 / 13.5%, 14.0% | 3, 1 / 3.2%, 3.3% | ~23.6% |

Paired significance (`scripts/significance_tests.py`, McNemar + bootstrap CI over the 228 problems):

- **The bf16 cost is real and large.** QAT-KD vs standard KD at bf16: both draws significant (p=0.0014, p=0.0033), Δsolved −18 / −17, 95% CI [−29,−8] / [−28,−7] — excludes 0 comfortably.
- **The W4 gain is not distinguishable from standard KD's W4 cell.** Both draws p=1 (ns), Δsolved +1 / −1, 95% CI [−3,+5] / [−5,+2] — includes 0. Caveat: only 4-5 discordant problems, so power is weak; the two W4 test-rate draws (3.2%, 3.3%) agree closely with each other, which is weak evidence the rate itself moved even though solve-count can't confirm it.

**Verdict against the pre-registration: fail.** The declared success bar was W4 test rate above ~8%; QAT-KD reaches 3.2-3.3%, short of it and only marginally past the ~3% noise floor also declared in advance. The retention percentage looks better than standard KD's (23.6% vs 5.9%), but that is arithmetic, not evidence: the denominator (bf16) collapsed by a large, confirmed margin, while the numerator's (W4) apparent rise is not confirmed. This is the shape of result the pre-registration's honest-failure clause was written to catch, even though it does not match the clause's literal wording (W4 did move, nominally).

Training through a straight-through W4 fake-quantizer for 3 epochs, at this model scale and data budget, damages general capability more than it protects distilled knowledge from rounding. No fix is proposed; per `plan-phase4.md` §3, this project reports the finding rather than chasing a working mitigation. This closes the question outcome A left open: neither calibrated PTQ (Stage 4) nor quantization-aware training (Stage 8) rescues the distilled model at W4.

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
