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

**Scope correction (2026-08-23): this section is base-track only, and it does not generalise.** Running the same calibrated GPTQ on the instruct track gives a gap of +6 and +12 across two draws, the second significant, with neither draw able to exclude the bf16 gap of +14. The claim that calibration does not rescue distilled knowledge is therefore demonstrated here and contradicted there. See "Calibrated PTQ on the instruct track" below before quoting the sentence above about this being the strongest form of the headline; on the evidence now available that sentence holds for the general-base student and not as a general statement.

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

## Stage 7 result: generality on EvalPlus — the erasure is task-complexity-dependent, not absolute (run 2026-08-11)

Evaluated the same four base-track checkpoints (original vs distilled x bf16 vs the calibrated GPTQ-W4 checkpoints from Stage 4) on HumanEval+ (164 problems) and MBPP+ (378 problems), pass@1 greedy. Generation used this project's own budget-forced think/code protocol (`scripts/evalplus_reasoning_codegen.py`, reusing `src/evaluation/generation.py`), not EvalPlus's own generic completion instruction. An earlier attempt using `evalplus.codegen` directly found a KD gap of essentially zero at both precisions, which turned out to be a measurement artifact: reading `evalplus/provider/vllm.py` and `utility.py` showed its default prompt is a generic "write a self-contained function" chat instruction that never triggers the model's trained `<think>...</think>` reasoning phase. Every other number in this project, including the whole LeetCode grid, comes from a protocol that always invokes that phase; that first measurement is discarded in favor of the reasoning-protocol run below.

| Benchmark | Precision | Original | Distilled | KD gap (pass@1) |
|---|---|---|---|---|
| HumanEval+ | bf16 | 33.5% | 37.2% | +3.7 pts |
| HumanEval+ | GPTQ-W4 | 7.3% | 9.1% | +1.8 pts |
| MBPP+ | bf16 | 47.1% | 55.8% | +8.7 pts |
| MBPP+ | GPTQ-W4 | 12.7% | 17.7% | +5.0 pts |

Paired significance (`scripts/evalplus_significance.py`, same McNemar + bootstrap approach as Stage 5, over the "+" pass@1 flags, base + extra tests):

| Comparison | McNemar p | Delta-pass [95% CI] |
|---|---|---|
| HumanEval+ bf16 gap | 0.39 (ns) | +6 [-5,+17] |
| HumanEval+ W4 gap | 0.58 (ns) | +3 [-4,+10] |
| MBPP+ bf16 gap | **0.0004** (\*\*\*) | +33 [+15,+51] |
| MBPP+ W4 gap | **0.013** (\*) | +19 [+5,+33] |

MBPP+ (378 problems, more statistical power) confirms both gaps: distillation lifts pass@1 significantly at bf16, and the advantage survives, only partially attenuated, at W4 -- the CI excludes 0 at both precisions. HumanEval+ (164 problems) shows the same direction but doesn't reach significance at this sample size; treat it as consistent with, not independently confirming, the MBPP+ result.

**This reframes the generality claim.** The LeetCode headline is complete erasure: the KD gap goes from +16 solves at bf16 to a confirmed null under both real quantizers (B2/B3 in `notes/significance_tests.md`). On MBPP+ the gap only shrinks -- retaining roughly half its magnitude by point estimate (49% on HumanEval+, 57% on MBPP+) rather than collapsing, and that retained gap is itself statistically significant. **The erasure is strongest on LeetCode-style competitive programming and partial on shorter, simpler function-completion tasks.** That is a more precise and more defensible claim than "the erasure generalizes" -- it says something about *what kind* of quantization fragility this is, plausibly tied to task complexity or generation length rather than being a blanket property of distilled-then-quantized weights.

## Activation-side probe: the erasure is confirmed behaviorally, not just inferred (run 2026-08-11)

Closes the scope limit stated in "Weight-level mechanism" above: that a destroyed weight-space direction causes the observed behavioural collapse was an inference from weight statistics, not a measurement. This probe measures it directly.

**Design.** Teacher-force each of {original, distilled} x {bf16, GPTQ-W4} over the same 100 fixed, already-verified-passing R1 traces (train split, `cache/teacher_logprobs_r1_full/`), one forward pass per sequence via vLLM's `prompt_logprobs` (no sampling, no code execution -- `scripts/activation_probe.py`). Records the model's own top-20 next-token distribution at every completion-token position, teacher-forced over the real reference continuation.

**The first pairing (each model vs its own quantized self) was uninformative by construction.** bf16-vs-W4 self-agreement was nearly identical for both tracks (original 81.4%, distilled 81.0%) -- expected, since a model's own bf16-to-W4 divergence is dominated by rounding noise on the *shared* bulk weight distribution (identical between original and distilled to five significant figures), not by the tiny KD-specific delta. This pairing cannot detect whether the KD signal specifically survives.

**The pairing that matters: does the original-distilled gap collapse under quantization?** If W4 erases distillation-specific behavior, the two quantized models should converge toward each other more than the two bf16 models did.

| Comparison | Top-1 agreement | dist top-1 missing from other's top-20 |
|---|---|---|
| dist_bf16 vs orig_bf16 | 80.6% | 0.4% |
| dist_w4 vs orig_w4 | 81.0% | 0.2% |

Both metrics move the same direction: agreement rises, and the rate at which the distilled model's preferred token falls out of the original's top-20 entirely is halved. Significance is bootstrapped over the 100 independent sequences, not the 316,104 raw positions (which are correlated within a sequence and would overstate the sample size) -- same discipline as the rest of this project's paired tests.

**Mean agreement-rate shift (W4 minus bf16): +0.48 pts, 95% CI [+0.31, +0.66] -- excludes 0.** 73 of 100 sequences show increased dist-orig agreement under W4.

**This is the activation-level confirmation of erasure.** The effect is modest per-token (well under one percentage point) but statistically unambiguous at n=100 independent sequences, and modest per-token shifts are exactly what should compound, over a full multi-hundred-token generation, into the large sequence-level accuracy collapse already measured (28 solved to 2, both real quantizers). The weight-space finding (KD update ~2% of a W4 step, cosine 0.19) and the eval-level finding (KD gap 12/28 to 2/2) are now connected by a direct behavioural measurement, not just an inference bridging them.

## Efficiency table: teacher vs distilled student (measured 2026-08-11)

Both models profiled identically (`scripts/efficiency_table.py`): 24 fixed prompts, temperature
0.6, top_p 0.95, max 512 new tokens, vLLM 0.6.6, bf16, single RTX 3090.

| Model | On-disk (bf16) | Weights VRAM | Throughput (24-prompt batch) |
|---|---|---|---|
| Teacher (R1-Distill-Qwen-7B) | 14.19 GB | 14.22 GiB | 994 tok/s |
| Student, distilled (1.5B) | 2.88 GB | 2.91 GiB | 2417 tok/s |

The distilled student is ~4.9x smaller and ~2.4x faster than the teacher at bf16, for a fraction of
the teacher's reasoning capacity (28/228 vs the teacher's 126/228 ceiling) -- expected at this size
gap, and the point of the deployment argument, not a counterexample to it.

**Theoretical INT8/W4 size** (bit-width only; real compressed-checkpoint throughput and VRAM cannot
be measured on this stack -- the vLLM 0.6.6 Marlin runtime defect corrupts generation from real
compressed-tensors checkpoints, see "Calibrated PTQ does not rescue the distilled model" above):

| Model | bf16 | INT8 (theoretical) | W4 (theoretical) |
|---|---|---|---|
| Teacher | 14.19 GB | 7.09 GB | 3.55 GB |
| Student, distilled | 2.88 GB | 1.44 GB | 0.72 GB |

At INT8 -- the precision this project's own grid shows is free (no measurable degradation on any
model tested, the finding this table exists to support) -- the distilled student's theoretical
footprint is ~9.9x smaller than the teacher's bf16 footprint. This is the practical form of the
paper's deployment recommendation: distill to 1.5B, serve at INT8, skip W4.

## Stage 10 pre-registration: precision dose-response (written 2026-08-15, before any run)

Recorded before running. Every result so far is consistent with the step-size account but none of it
*tests* the account, because the grid only has its two endpoints. This stage turns the mechanism into
a prediction: if erasure is caused by the KD update being small relative to the quantization step,
then sweeping the step must move the gain back, at a location the existing measurements already fix
in advance.

**Amendment, same day, still before any eval run.** The first draft of this entry assumed the step halves
per added bit and predicted 2/4/8/16/32%. That is wrong: symmetric quantization uses qmax = 2^(bits-1) - 1,
so the step scales as 1/qmax and the W4-to-W8 ratio is 127/7 = 18.1x, not 16x. `scripts/smoke_quantgrid.py`
measured the quantizer directly and the per-bit ratios match the qmax ratios exactly, so the corrected
figures below are 2.0/4.3/8.9/18.0/36.3%. The declared crossover of W6-W7 is unchanged -- the bracket moves
by well under one bit. Recorded as an amendment rather than a silent rewrite; zero eval cells existed at
the time of the correction, and the git history shows both versions.

**A confound this stage also fixes.** The recorded grid pairs W8 at per-output-channel granularity
with W4 at group-128 (`quantize_int8.py`, original `int_roundtrip_`), so the headline INT8-preserves /
INT4-erases contrast varies bit-width and granularity together, and the 26% / 2.0% step fractions
inherit that. Evidence it matters: 26/2.0 = 13.0x, where four bits at fixed granularity gives 18.1x
(measured, `smoke_quantgrid.py`). Granularity therefore accounts for 18.1/13.0 = 1.40x on its own --
per-channel scales off the row max, group-128 off the group max, so the per-channel step is the larger
of the two. Stage 10 sweeps at **fixed group-128 throughout**, so the
new ladder is single-variable and the old contrast can be restated cleanly.

**Intervention.** No training. `scripts/quantize_int8.py` with bit-width and granularity as
independent axes (`--bits 3-8`, `--group-size`, defaults reproducing the old pairing). Base track
first, both checkpoints, same eval harness as every other cell.

**Prediction.** At fixed granularity the step scales as 1/qmax with qmax = 2^(bits-1) - 1, not as a clean
halving per bit -- the -1 matters at low bit-width. Verified against the quantizer itself: measured
per-bit ratios 2.14 / 2.07 / 2.03 / 2.02 match 15/7, 31/15, 63/31, 127/63 exactly. Anchoring on the
measured base-track W4-g128 figure of 2.0%:

| Bits (g128) | qmax | KD update as % of step | Predicted gain | cos (transmission) |
|---|---|---|---|---|
| W4 | 7 | 2.0% (measured) | absent (measured: 6 vs 6, test rate 1.3% vs 1.1%) | 0.194 (measured) |
| W5 | 15 | 4.3% | absent or trace | |
| W6 | 31 | 8.9% | partial | |
| W7 | 63 | 18.0% | mostly present | |
| W8 | 127 | 36.3% | present | 0.718 at per-channel (measured) |

The W8-g128 cell predicts 36.3% against the 26% measured at per-channel, in the right direction:
group-128 gives the smaller step, so the same update is a larger fraction of it. That 36.3/26 = 1.40x
is the same granularity factor measured independently above, which is the internal consistency check.

**Pre-declared crossover: W6 or W7.** Defined as the lowest bit-width at which the paired bootstrap 95%
CI on the distilled-minus-original difference excludes zero, same paired bootstrap and same 228 problems
as every other significance row in this project (`scripts/significance_tests.py`).

**Primary metric is test-case pass rate, not solve count** — same reasoning as Stage 8, since the base
track's low cells sit near the floor where solve counts have no resolution. Solve count secondary.
Declared noise floors, from this project's own scatter: ~3% test rate, ±4 to 6 solves single-draw.

**Second, independent prediction.** The behavioural crossover should coincide with cosine transmission
crossing roughly 0.4 to 0.5, interpolating monotonically between the measured 0.194 (W4) and 0.718 (W8).
Run `analyze_weight_quant.py` at every bit-width so weight-space and eval land on one axis. This is a
separate falsifiable claim from the solve-count one and should be judged separately.

**Third prediction, the group-size axis -- a negative control.** Measured (`smoke_quantgrid.py`): at fixed
W4, granularity 128 -> 64 -> 32 shrinks the step by only 1.21x in total, against 2.14x for a single added
bit. So W4-g32 puts the KD update at 2.42% of a step, against 2.0% at g128 -- still deep in the erasure
regime. **Predict no recovery anywhere on this axis**, while the bit-width axis recovers over the same
runs. This is the sharper form of the claim: three cells nominally at "4-bit" that all stay erased, next
to a ladder at fixed granularity that does not, isolates step size from bit-width as the operative
variable. Recovery at g32 would falsify the account as stated.

**Pre-declared falsification.** The step-size account is wrong, and must be rewritten rather than
softened, if any of the following holds:

- The curve is non-monotone in bit-width beyond the declared noise floor — e.g. a gain at W5 or W6
  larger than at W8.
- **W8 at g128 fails to preserve the gain.** The recorded W8 preservation was measured at per-channel;
  if it does not reproduce at group-128 then granularity, not bit-width, was doing the work, and the
  paper's central contrast is misattributed. This is the outcome that would cost the most, which is
  exactly why it is written down first.
- The gain is fully present already at W5 (4% of a step). Directionally consistent, but it would put the
  threshold far below what the 2%-vs-26% bracket implies, and the quantitative form of the account would
  not survive.

**Draw discipline, declared in advance.** The ladder runs single-draw to locate the crossover; the two
cells bracketing it are then double-drawn (seeds 1234 and 42) before any conclusion is recorded. Fixing
this now so draws cannot be added after seeing which cell is inconvenient — the failure mode the QEAD
ablation demonstrated, where a first draw of 30 read as a real +6 until the second returned 35.

**Back-compatibility check, to run before anything else.** Re-quantize one existing base checkpoint at
`--bits 4 --group-size 128` and confirm byte-identity against the W4 checkpoint already on the server.
If that fails, no Stage 10 number is comparable to any recorded number and the generalized quantizer is
wrong.

## Calibrated PTQ on the instruct track: the erasure does NOT replicate (run 2026-08-22 to 2026-08-23)

The calibrated-PTQ section above was measured on the base track only. Repeating it on the instruct track was intended as a routine two-track confirmation. It is not one, and the honest reading is that the real-quantizer erasure claim holds on the base track and fails on the instruct track.

| Track | bf16 gap | Real GPTQ-W4 gap [95% CI] | McNemar p | CI excludes own bf16 gap? |
|---|---|---|---|---|
| Base | +16 | -1 [-5, +2] | 1.0 | **yes**, erasure demonstrated |
| Instruct, draw 1 | +14 | +6 [-2, +14] | 0.24 | no |
| Instruct, draw 2 | +14 | **+12 [+4, +21]** | **0.012** | no |

On the instruct track the distilled model holds 29 and 31 solved against 36 at bf16, while its original stays flat (22 at bf16, 23 and 19 under GPTQ). The gap attenuates from +14 to roughly +9, about 64 percent retained, and the second draw is significantly positive. Neither instruct draw can exclude the full bf16 gap, so nothing here supports erasure; the first draw is simply underpowered (18 discordant problems, CI spanning -2 to +14, which contains both zero and the whole bf16 effect).

This is consistent in direction with the EvalPlus result, where the erasure was also partial rather than complete, and it should be stated the same way: **complete erasure is demonstrated on the general-base track under both real quantizers, and is not demonstrated on the instruct track under calibrated GPTQ.**

A second discrepancy sits underneath it. On the instruct track the simulated quantizer puts the distilled model at 17 and 23, while real GPTQ puts it at 29 and 31. The simulated quantizer is markedly more destructive here, which is the opposite direction from the base track, where real GPTQ (2) was more destructive than simulated (7). Since the headline grid of `quantize_int8.py` is simulated throughout, this is a live methodological caveat and not a footnote: the two toolchains do not agree on how much the instruct distilled model loses at 4 bits.

### Weight-space follow-up: the mechanism is scoped to weight-error-minimising quantizers

Two measurements were run to test whether the Stage 10 step-size account explains the instruct GPTQ behaviour. It does not, and the reason is instructive.

First, the matched instruct cosine ladder (`scripts/cos_ladder.py`, 196 Linear weights, g128), against the base ladder already recorded:

| Bits | Instruct kd_cos | Base kd_cos | Instruct noise_rel | Base noise_rel |
|---|---|---|---|---|
| W4 | 0.110 | 0.194 | 0.1280 | 0.1267 |
| W5 | 0.226 | 0.307 | 0.0598 | 0.0592 |
| W6 | 0.351 | 0.438 | 0.0290 | 0.0287 |
| W7 | 0.501 | 0.617 | 0.0142 | 0.0141 |
| W8 | 0.687 | 0.807 | 0.0071 | 0.0070 |

`noise_rel` agrees between tracks to three decimals at every bit-width, as it must: same quantizer, same grid, so the perturbation is a property of the precision and not of the checkpoint. Transmission is uniformly lower on the instruct track for the only remaining reason, that its KD update is about half the size (0.459 percent of weight norm against 0.892 percent). The independent GPTQ measurement returns `kd_rel_before` = 0.0046, confirming the instruct update size from a different code path.

Second, transmission through the real GPTQ checkpoints (`scripts/gptq_transmission.py`, which reads four checkpoints off disk and quantizes nothing, so the number is whatever the toolchain produced). Compared against the matched simulated cell above, same checkpoint pair:

| Quantizer | kd_cos | noise_rel | kd_rel_after | frac_amplified | Behavioural gap |
|---|---|---|---|---|---|
| Simulated W4 g128 | 0.110 | 0.128 | n/a | n/a | erased (-4, +2) |
| Real GPTQ W4 | 0.022 | 0.157 | 0.167 | 0.62 | retained (+6, +12) |

**GPTQ is five times worse on transmission and 23 percent worse on weight perturbation, and preserves the distillation gain where round-to-nearest destroys it.** It also replaces a 0.46 percent update with a 16.7 percent difference pointing elsewhere, amplifying 62 percent of per-weight changes beyond 2x.

This is coherent once the objective is taken seriously. GPTQ does not minimise weight error; it minimises layer output error under the Hessian of each layer's inputs, and deliberately accepts larger weight displacement along directions the activations do not excite. Large weight-space perturbation with small functional perturbation is the method working as designed.

The consequence is a scope limit and it should be stated as one: **the step-size and cosine-transmission account explains weight-error-minimising quantization, which covers the entire simulated grid, the W4-to-W8 ladder, the group-size negative control and the Stage 8 QAT failure, and it does not transfer to activation-aware quantizers.** Everything the ladder established stands; GPTQ marks the boundary rather than contradicting the interior.

One interpretive caution belongs with the number. `kd_cos` = 0.022 should not be read as "GPTQ destroys the KD direction". For two models quantized independently by a calibration-dependent method, the weight-space difference is dominated by each model's own Hessian-driven decisions rather than by the update separating them. Round-to-nearest per-group scales partially cancel between similar checkpoints; GPTQ's sequential compensation does not. The honest statement is that `kd_cos` loses meaning for calibration-dependent quantizers, not that it measured destruction.

### A silent failure that nearly entered the record

The matching real-RTN cell for the instruct track was run and produced original 24 and distilled 35, a gap of +11 with p=0.035, apparently supporting retention. **Those numbers are invalid and are discarded.** `scripts/probe_quant_grid.py` counts distinct values inside one quantization group, where a group-128 W4 checkpoint admits at most 16; the RTN checkpoints returned 119 to 125, byte-identical to the unquantized bf16 reference. The weights were never rounded, so the cell measured a bf16 model, which is why the distilled score of 35 sat one solve below bf16's 36.

Cause: `build_recipe` maps `--method rtn` to `QuantizationModifier`, which only attaches scales and zero points and defers the rounding to `save_compressed=True`. Under `--save-mode fake` the rounding therefore never happens, and `strip_quant_params` then removes the attached parameters, leaving a plain copy of the input. `GPTQModifier` is unaffected because it rewrites weights in place during calibration, which the grid probe confirms (13 to 15 distinct values per group). The recorded base-track RTN cell from 2026-08-04 is sound: it scored 2 against 28 at bf16, which an unquantized checkpoint cannot do, so the behaviour changed with the llm-compressor version pulled in when `evalplus` was added on 2026-08-10. The same upgrade caused the `attention_mask` KeyError fixed in `quantize_real.py` on 2026-08-23.

Two mitigations are in place. `quantize_real.py` now runs the grid probe automatically after any `--save-mode fake` build and exits non-zero rather than emitting a checkpoint that is not on the expected grid. Real-toolchain RTN is not available on the instruct track under the current library, and is not pursued: `quantize_int8.py --bits 4 --group-size 128` already is group-128 RTN, and the real-RTN cell existed only to match GPTQ's toolchain.

The general lesson is worth keeping. This failure produced no error, a correctly sized checkpoint, a loadable model, and a plausible eval number in the direction the run was testing. Nothing except a direct check of the weight values would have caught it, and every fake-quant checkpoint in this project is stored as bf16, so dtype and config inspection prove nothing.

## Stage 10 result: the precision dose-response, all three predictions hold (run 2026-08-20 to 2026-08-22)

Run as pre-registered above: base track, both checkpoints, bit-width swept at **fixed group-128** so the ladder is single-variable, same eval harness and 228 problems as every other cell, primary metric test-case pass rate with solve count secondary.

**Back-compatibility gate, run before anything else as declared.** `scripts/smoke_quantgrid.py` reports legacy equivalence 8/8 exact, and the measured per-bit step ratios (2.14 / 2.07 / 2.03 / 2.02) match the qmax ratios 15/7, 31/15, 63/31, 127/63 exactly. The behavioural check agrees: W4-g128 scores 6 (original) and 7 (distilled), reproducing the recorded simulated W4 grid cell for cell. Every Stage 10 number is therefore comparable to the recorded grid.

### The ladder (single draw, seed 1234)

| Bits | Original | Distilled | Gap solved [95% CI] | McNemar p | Gap test rate |
|---|---|---|---|---|---|
| W4 | 6 / 7.5% | 7 / 4.8% | +1 [-3, +5] | 1.0 | -2.7 pts |
| W5 | 12 / 9.5% | 22 / 15.2% | +10 [+3, +17] | 0.013 | +5.7 pts |
| W6 | 8 / 12.7% | 22 / 19.2% | +14 [+6, +22] | 0.0013 | +6.5 pts |
| W7 | 14 / 13.4% | 29 / 23.0% | +15 [+6, +24] | 0.0015 | +9.6 pts |
| W8 | 15 / 10.0% | 25 / 23.1% | +10 [+1, +19] | 0.052 | +13.1 pts |

The distilled model's own recovery is a clean monotone saturation toward its bf16 value (test rate 4.8, 15.2, 19.2, 23.0, 23.1 percent against 22.0 at bf16), and the gap on the primary metric is monotone across the whole ladder.

### Double draw on the bracketing cells, and why the crossover is W6 rather than W5

The single-draw ladder put the lowest CI-excludes-zero cell at W5, which would have placed the crossover one bit below the pre-registered W6-to-W7 bracket. The declared draw discipline settled it.

| Cell | Draw | Original | Distilled | Gap solved [95% CI] | McNemar p |
|---|---|---|---|---|---|
| W4 | seed 1234 | 6 / 7.5% | 7 / 4.8% | +1 [-3, +5] | 1.0 |
| W4 | seed 42 | 8 / 8.1% | 5 / 4.5% | -3 [-8, +2] | 0.45 |
| W5 | seed 1234 | 12 / 9.5% | 22 / 15.2% | +10 [+3, +17] | 0.013 |
| W5 | seed 42 | 14 / 6.9% | 20 / 16.2% | +6 [-3, +15] | 0.29 |

W4 is unambiguous: both draws non-significant, mean gap -1, erasure confirmed. W5 does not replicate: the first draw excludes zero and the second does not, mean gap +8 solved. This is the same shape as the QEAD ablation, where a first draw of 30 read as a real +6 until the second returned 35, and it is exactly what the draw discipline was declared in advance to catch. **The lowest bit-width whose single draw excludes zero and is not contradicted by a second draw is W6, inside the pre-registered W6-to-W7 bracket.**

**Correction (2026-08-23).** An earlier version of the sentence above said the gap "replicates" at W6. That is wrong and is retracted: W6, W7 and W8 are all single-draw on this track, because the declared discipline sent the double-draws to the two cells bracketing the apparent crossover, which were W4 and W5. W6 has therefore never been replicated, and the base crossover rests on one draw. This matters for the cross-track comparison with Stage 11 below, where the instruct cells were double-drawn and did not replicate: the base result looks cleaner in part because it was tested less. Stated honestly, the base transition is bracketed at W5 to W6 rather than located at W6.

An internal inconsistency in the pre-registration surfaces here and is recorded rather than resolved after the fact. Test-case pass rate was declared the primary metric, but the crossover was defined through the paired bootstrap CI, which `significance_tests.py` computes on solve flags. The two disagree at W5: on solve count W5 is unconfirmed, while on test rate both draws show a solid positive gap (+5.7 and +9.3 points, against a declared noise floor near 3 percent). The defensible statement is therefore bracketed rather than pointwise: **erased through W4, in transition at W5 to W6, fully recovered by W7 to W8.**

No third draw was added at W5. The pre-registration forbids adding draws after seeing which cell is inconvenient, and a tiebreak run chosen because the first two disagreed is precisely that. The ambiguity is reported as the result.

### Falsification checks, all three declared in advance

- **W8 at g128 fails to preserve the gain.** Not triggered, and this was the outcome written down first because it would have cost the most. W8-g128 preserves (+10 solved, CI [+1, +19], test rate +13.1 points), so the headline INT8-preserves and INT4-erases contrast is not an artifact of the recorded grid pairing per-channel W8 with group-128 W4. The granularity confound is closed.
- **Non-monotone in bit-width beyond the noise floor.** Not triggered on the primary metric: the test-rate gap climbs -2.7, +5.7, +6.5, +9.6, +13.1. On solve count W7 (+15) exceeds W8 (+10) by 5, which sits inside the declared single-draw floor of 4 to 6 solves, and the choice of test rate as primary is what keeps this clean.
- **Gain fully present already at W5.** Not triggered. W5 recovers roughly half the bf16 gap (+8 solved of +16, +7.5 points of +11.7 on test rate), which is the "partial" regime, not full presence.

### The second prediction: cosine transmission (`scripts/cos_ladder.py`, 196 Linear weights, no GPU)

| Bits | kd_cos | noise_rel | frac_erased |
|---|---|---|---|
| W4 | 0.194 | 0.127 | 0.458 |
| W5 | 0.307 | 0.059 | 0.443 |
| W6 | 0.438 | 0.029 | 0.414 |
| W7 | 0.617 | 0.014 | 0.360 |
| W8 | 0.807 | 0.007 | 0.276 |

The prediction was that the behavioural crossover should coincide with cosine transmission crossing roughly 0.4 to 0.5. It crosses that band at W6 (0.438). **Two independent pre-registered predictions, one behavioural and one in weight space, both land on W6.** Weight-space anchors reproduce the earlier measurement exactly (`noise_rel` 0.127 at W4 against the recorded 12.67 percent W4 perturbation; `kd_cos` 0.194 at W4 against the recorded 0.194), and W8 at g128 gives 0.807 against 0.718 at per-channel, higher at the finer granularity as the independently measured 1.40x step-size factor requires.

### The third prediction: the group-size negative control (run 2026-08-22)

The sharpest form of the claim. Granularity is a second way to shrink the quantization step, but a far weaker one: measured on the quantizer itself, g128 to g32 shrinks the step by only 1.21x in total, against 2.14x for a single added bit. The prediction was therefore no recovery anywhere on the granularity axis, while the bit-width axis recovers over the same runs, and recovery at g32 would falsify the account as stated.

| Cell | Original | Distilled | Gap solved [95% CI] | McNemar p | Gap test rate | Trunc |
|---|---|---|---|---|---|---|
| W4 g128 | 6 / 7.5% | 7 / 4.8% | +1 [-3, +5] | 1.0 | -2.7 pts | 157 |
| W4 g64 | 8 / 9.5% | 8 / 7.5% | 0 [-5, +5] | 1.0 | -2.1 pts | 121 |
| W4 g32 | 9 / 10.5% | 6 / 11.6% | -3 [-8, +2] | 0.45 | +1.1 pts | 30 |

Every interval on the granularity axis contains zero, against W5 at +10 [+3, +17] and W6 at +14 [+6, +22] on the bit axis. **Prediction confirmed: three cells all nominally 4-bit stay erased while the bit ladder recovers, which isolates step size rather than the nominal bit-width label as the operative variable.**

### The quantitative form of the account

Combining both axes gives the threshold the two-endpoint bracket could not. Expressing every configuration as the KD update's size relative to one quantization step:

| Configuration | KD update as % of step | KD gap | Distilled trunc |
|---|---|---|---|
| W4 g128 | 2.0% | absent | 157 |
| W4 g64 | 2.2% | absent | 121 |
| W4 g32 | 2.4% | absent | 30 |
| W5 g128 | 4.3% | unconfirmed (draws disagree) | 1 |
| W6 g128 | 8.9% | present | 0 |
| W7 g128 | 18.0% | present | 0 |
| W8 g128 | 36.3% | present | 2 |

The gap recovers somewhere between roughly 4 and 9 percent of a step. The granularity axis tops out at 2.4 percent, well short of that, which is why no amount of group tightening recovers the gain at 4 bits. This supersedes the original 2-percent-versus-26-percent bracket with a bounded threshold, and it retroactively explains the Stage 8 QAT-KD failure: that run trained through a group-128 W4 fake-quantizer, a step fraction of 2.0 percent, squarely inside the erasure regime, so the optimiser was searching against a grid too coarse for the update to survive.

### Coherence and gap recovery are the same variable at different thresholds

An earlier draft of this entry claimed generation coherence was a cliff confined to W4 that vanished in a single bit, on the basis of the ladder truncation counts alone (157, 1, 0, 0, 2 for W4 through W8). The group-size control corrects that. Truncation also falls sharply *within* W4 as granularity tightens, 157 to 121 to 30, so coherence is not bound to bit-width either: it responds to step size like the gap does, but recovers at a materially lower threshold, largely repaired by about 2.4 percent of a step where the gap needs roughly 9 percent. Both effects therefore reduce to the single step-size variable with two different thresholds, which is a simpler account than two unrelated phenomena. Note the dissociation this produces at W4 g32: coherent output (truncation 30, test rate 11.6 percent, the highest of any 4-bit cell) with the distillation gain still entirely absent. A model can be repaired enough to stop degenerating while remaining unable to express what distillation taught it.

### Deviations and infrastructure notes

W5 could not be evaluated under the stock engine configuration. vLLM 0.6.6 asserts in `scheduler.py::_schedule_running` (`assert len(self._async_stopped) == 0`) when its async output processor races chunked prefill during preemption, and the enormous phase-2 prompts produced by W5's degenerate think traces trigger that path reliably. A `--sync-output` flag was added to `compare_eval.py` (disables async output processing, default off, no other cell affected) and W5 was run with it. The flag changes when output tokens are processed rather than what is sampled, and the load-bearing quantity here, the distilled-minus-original gap, is measured within a single invocation under identical settings, so the within-rung comparison is unaffected. Recorded as a deviation because the cross-rung curve carries it. The crash is itself corroborating evidence that the W4-to-W5 degeneration regime is severe enough to destabilise the serving stack.

Scope note on the negative control: the granularity sweep runs entirely inside the simulated quantizer of `quantize_int8.py`, which is the same quantizer that produced the main INT8/INT4 grid and the whole W4 to W8 ladder, so it is the correct control for those numbers. It says nothing independently about how `llm-compressor`'s GPTQ and RTN behave under changed granularity; `quantize_real.py` accepts a `--group-size` argument but warns that the recipe does not apply it, so the real-toolchain cells of Section "Calibrated PTQ" were all run at that toolchain's own default and granularity was never swept there.

## Stage 11 pre-registration: the instruct crossover sits one bit higher (written 2026-08-23, before any run)

Recorded before running. Stage 10 located the base-track crossover at W6 and fitted a threshold on that track alone. The instruct weight-space measurements above fix a prediction for a second student in advance, derived quantitatively rather than chosen after seeing an eval.

**Two independent routes, both from already-measured quantities, both giving the same answer.**

Cosine route: the 0.4-to-0.5 transmission band that coincided with the base behavioural crossover at W6 (0.438) is not reached on the instruct track until W7 (0.501); instruct W6 is only 0.351.

Step-fraction route: the instruct KD update is 0.459 percent of weight norm against the base 0.892 percent, a factor of 0.515, so it sits at about 1.03 percent of a W4 step where the base sits at 2.0 percent. Applying the step ratios measured in `smoke_quantgrid.py` (W4 to W6 divides the step by 4.43, W4 to W7 by 9.00), the instruct update reaches 4.6 percent of a step at W6 and 9.3 percent at W7. Stage 10's recovery band is 4 to 9 percent, so instruct enters it at W6 and clears it at W7.

**Prediction: the instruct behavioural crossover is W7, marginally W6.** Recovery below W6 or absence at W8 both falsify the account's transfer across students.

**Intervention.** No training. `quantize_int8.py --bits {6,7} --group-size 128` on the instruct original and the instruct distilled checkpoint, evaluated with the same harness, 228 problems, `--num-samples 5`, as every other cell. Only the decisive pair is run: Stage 10 already establishes that W4 is erased and W8 preserved, and the full ladder is not needed to locate a crossover that is bracketed in advance.

**Crossover definition and metrics, unchanged from Stage 10.** Lowest bit-width whose paired bootstrap 95 percent CI on distilled-minus-original excludes zero. Primary metric test-case pass rate, solve count secondary. Declared noise floors: about 3 percent test rate, plus or minus 4 to 6 solves single-draw.

**Draw discipline, declared in advance.** Single draw at each of W6 and W7 to locate; whichever cell the crossover lands on is then double-drawn (seeds 1234 and 42) together with the cell below it, before any conclusion is recorded. No tiebreak draw is added after seeing a disagreement, per the Stage 10 W5 precedent.

**Pre-declared falsification.** The cross-track transfer fails, and must be rewritten rather than softened, if the gap is already present at W5 or below, if it is still absent at W8, or if the curve is non-monotone in bit-width beyond the declared noise floor.

**What a confirmation would and would not show.** It would show that a threshold fitted on one student predicts, quantitatively and one bit out, where a second student with a differently sized update recovers. It would not extend the account to activation-aware quantizers, which the GPTQ measurement above places outside its scope regardless of this outcome.

## Stage 11 result: the instruct crossover is not located, and the transition is bracketed (run 2026-08-23)

Run exactly as pre-registered: instruct original and instruct distilled at W6 and W7, group-128, same harness and 228 problems, single draw to locate and then both cells double-drawn at seeds 1234 and 42 before recording.

| Cell | Draw | Original | Distilled | Gap solved [95% CI] | McNemar p |
|---|---|---|---|---|---|
| W6 | 1234 | 20 / 19.3% | 28 / 23.5% | +8 [-2, +17] | 0.15 |
| W6 | 42 | 24 / 19.2% | 30 / 23.6% | +6 [-2, +14] | 0.24 |
| W7 | 1234 | 22 / 18.8% | 36 / 23.4% | **+14 [+4, +24]** | **0.013** |
| W7 | 42 | 22 / 18.9% | 30 / 26.7% | +8 [+0, +16] | 0.096 |

**The prediction is not confirmed.** On the single draws the result matched the pre-registration exactly, W6 non-significant and W7 significant, which is a crossover at W7 and the predicted one-bit offset from the base track. The second draw at W7 returned +8 with a CI whose lower bound falls on zero, so W7 does not replicate. Under the declared criterion, applied with the declared draw discipline, neither W6 nor W7 certifies and the instruct crossover is not located.

What the data does support is a real and graded transition. Every one of the four cells is positive, and the means rise monotonically toward the bf16 gap of +14: W6 gives +7 (draws +8, +6) and W7 gives +11 (draws +14, +8). On test-case rate the same ordering holds, W6 at +4.2 and +4.4 points and W7 at +4.6 and +7.9 against a bf16 gap of +9.0. The recovery is not in doubt; its location within one bit is.

**The honest cross-track statement, with both tracks treated equally:** the base transition is bracketed at W5 to W6 and the instruct transition at W6 to W7. The direction of the offset is consistent with the prediction derived from update size, and the magnitude is consistent with one bit, but neither track has a crossover certified under replication, so the offset is suggestive rather than established.

Two things follow that are worth stating plainly rather than discovering later.

First, the asymmetry noted in the correction above. Base W6 was never double-drawn while instruct W6 and W7 were, so the base crossover survived only because it faced less scrutiny. Any comparison between the tracks should either double-draw base W6 and W7 to equalise treatment, or report both as brackets. It should not present a single-draw base crossover next to a double-drawn instruct non-replication as though the two were measured the same way.

Second, a power limit that applies to the whole ladder and is now visible twice. At 228 problems with gaps of roughly +6 to +14 solves and 18 to 28 discordant problems per comparison, the paired bootstrap CI is about plus or minus 9 solves wide. That is wider than the difference between adjacent bit-widths in the transition region, so this design can establish that recovery happens across a two-bit window but cannot resolve which single bit it happens at. Locating a crossover to one bit would need either more problems or more samples per problem, not more bit-widths. The pre-registration's crossover definition was therefore sharper than the measurement could deliver, which is a design lesson rather than a result.

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
