# On-Policy GKD Experiment

Date: 2026-07-16
Branch: `onpolicy-gkd`

## Goal
Test whether on-policy Generalized Knowledge Distillation (GKD) — training the student on its
own generated solutions corrected by the R1 teacher's token distribution — improves over offline
trace distillation. Motivation: offline KD plateaued at 38–39/228; on-policy is the literature's
answer to train/inference exposure bias.

## Pipeline
1. `scripts/onpolicy_generate.py` — distilled student (`outputs/final`) generates one solution per
   train problem to `cache/onpolicy_r1_gen`.
2. `scripts/rescore_cache.py --src cache/onpolicy_r1_gen --dst cache/onpolicy_r1` — R1 teacher
   scores the student's tokens (`prompt_logprobs`, top-20), no test labels (pure KD).
3. `scripts/train.py --offline --onpolicy` — pure-KD fine-tune (`alpha=1.0`, no CE) from
   `outputs/final` on the student's own trajectories, saved to `outputs/final_last`.

## Config
- onpolicy: temperature 0.7, top_p 0.95, max_tokens 16384
- training: alpha 1.0 (pure distill), init from `outputs/final`, 1628 traces, 2 epochs
- teacher: R1-Distill-Qwen-7B

## Result

### Baseline (offline distilled)
| role | solved | pass@1 | pass@5 | testcase | trunc p/s |
|---|---|---|---|---|---|
| original | 25/228 | 4.5% | 11.0% | 19.9% | 0/0 |
| distilled | 38/228 | 7.3% | 16.7% | 23.1% | 42/54 |

Dominant failure: `wrong_answer=175`.

### On-policy training signal
- n_train 1465, 2 epochs, baseline_val 0.0407
- distill loss: 0.024 -> 0.022 (near zero)
- teacher_frac: 1.0 (teacher signal present on 100% of batches)

### On-policy eval
- approx 38/228 — UNCHANGED from offline baseline. On-policy gave no gain.

## Why it did not help (deepcheck over 259,367 token positions of `cache/onpolicy_r1`)
| measurement | value |
|---|---|
| teacher argmax == student's actual token | 96.6% |
| teacher mean prob on student's actual token | 0.948 |
| student token NOT in teacher top-k | 0.0% |

The R1 teacher endorses the R1-distilled student's own tokens ~96.6% of the time, so there is
almost nothing to correct. This is why the distill loss was ~0.02, the weights barely changed,
and the eval was unchanged.

## Conclusion
1. On-policy GKD gives no gain for a same-lineage teacher/student (R1 teacher, R1-distilled
   student). On-policy requires the teacher to disagree with the student on the student's own
   outputs; here they agree.
2. The student's failures are strategic (wrong overall approach), not local (wrong token). Each
   token in a wrong solution is locally plausible, so the teacher agrees token-by-token even on
   wrong solutions. Token-level KD (offline or on-policy) cannot fix strategic errors.
3. The offline distilled model (38–39/228) is the pure-KD ceiling from this teacher.
4. Exceeding it needs a sequence-level signal (test-based rejection), which is out of the pure-KD
   scope, or a stronger/different teacher.

## Checkpoints and artifacts
- `outputs/final` — offline distilled model (38/228), the KD result. KEEP.
- `outputs/final_last` — on-policy model (~38/228, no gain). KEEP as ablation artifact.
- `outputs/final_offline_bak` — backup of offline model (byte-identical to `outputs/final`).
- `cache/onpolicy_r1`, `cache/onpolicy_r1_gen` — on-policy caches, removed to reclaim ~24 GB;
  regenerable from the code on branch `onpolicy-gkd`.
