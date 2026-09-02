# Pre-registration: external replication of the precision threshold

Registered 2026-09-02, before any evaluation cell of this experiment was run.
Governs the four cells described below. Follows the discipline of the four
registrations already reported in the paper.

## Why this experiment

Every result in the paper is measured on students we distilled ourselves, from one
teacher and one corpus. The weight-space account claims more than that: it claims the
survival of a distillation increment under rounding is decided by how large that
increment is relative to the quantization step, whoever produced it.

`logs/weightstats_r1d15b.json` (generated 2026-08-14) already measures the pair
`Qwen/Qwen2.5-Math-1.5B` -> `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, a checkpoint
distilled by another group, from another teacher, on another corpus:

| quantity | our general-base track | R1-Distill-1.5B |
|---|---|---|
| update, fraction of weight norm | 0.0089 | 0.1915 |
| `cos_4` | 0.194 | 0.716 |
| `cos_8` | 0.718 | 0.996 |
| `noise_rel_4` | 0.1267 | 0.1267 |

The perturbation is identical because it is a property of the grid. The transmission
differs because the update is about 21 times larger.

## Prediction

The account predicts the **opposite outcome** to the paper's headline result on this
checkpoint. Because `cos_4 = 0.716` sits far above the 0.4-0.5 band at which recovery
occurs on both of our tracks, the distillation gain of R1-Distill-1.5B over its own
backbone **should survive 4-bit quantization**.

This is an out-of-sample prediction. It is not a restatement of the erasure result and
cannot be confirmed by it.

## Intervention

No training. Four evaluation cells, all under the harness, seed, sample budget and
problem set used everywhere else in the paper:

| cell | model | precision |
|---|---|---|
| 1 | `Qwen/Qwen2.5-Math-1.5B` | bf16 |
| 2 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | bf16 |
| 3 | `Qwen/Qwen2.5-Math-1.5B` | W4, group-128, simulated |
| 4 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | W4, group-128, simulated |

The simulated quantizer of `scripts/quantize_int8.py --bits 4 --group-size 128` is used,
the same one that produced the main grid and the whole precision ladder, so the result is
expressed in the same units as Table `tab:threshold`. Calibration-based quantizers are
excluded by Section `sec:scope`, which places them outside this account.

## Metrics and criterion, unchanged from the earlier registrations

Test-case pass rate primary, solved count secondary. Paired bootstrap over problems,
20,000 resamples, on the distilled-minus-original difference. Declared noise floors of
approximately 3 percentage points of test rate and +/- 4 to 6 solved problems on a single
draw.

**The gain is declared present at W4** if the paired bootstrap 95% interval on the
distilled-minus-original difference at W4 excludes zero, and **absent** if it contains
zero while the bf16 interval excludes it.

## Pre-declared falsification

If the gain is **absent** at W4 on this checkpoint, the transmission account is wrong or
materially incomplete, and must be rewritten rather than qualified. A measured `cos_4` of
0.716 with no surviving gain would mean directional transmission does not predict survival
across checkpoints, which is the account's central claim.

## What confirmation would and would not establish

It would establish that the threshold is a property of the interaction between an update
and a grid rather than of our distillation recipe, and it would make the paper's mechanism
predictive rather than explanatory.

It would not establish anything about activation-aware quantizers, which Section
`sec:scope` places outside the account regardless of this outcome, nor about model scales
other than 1.5B.

## Draw discipline

Single draw per cell to locate. If the bf16 gap and the W4 gap fall on opposite sides of
the criterion, both W4 cells are re-run at a second seed before anything is recorded. No
third draw is added after a disagreement is seen.

## Recording

Whatever the outcome, it is reported as recorded. Two of the four registrations already in
the paper were not confirmed and are reported as such.
