# Research Progress

Title: "Efficient Reasoning for Competitive Programming via Knowledge Distillation and Post-Training Quantization of Large Language Models"

## Project Scope

Distill a large reasoning teacher (DeepSeek-R1-Distill-Qwen-7B, bf16, long-CoT) plus a short-CoT
helper teacher (Qwen2.5-Coder-7B-Instruct) into a small student (Qwen2.5-1.5B general base as the
headline track; Qwen2.5-Coder-1.5B-Instruct as the ceiling-evidence track) on LeetCode coding
problems, then measure how Post-Training Quantization interacts with distilled knowledge.

**Scope revised 2026-07-29.** The original goal was to show that the QEAD-distilled student degrades
less after quantization than a naively distilled one. That hypothesis was tested on both tracks and
**falsified** (Phase 3 below). The paper's contribution is now the quantization-interaction finding
itself, with the QEAD ablation reported as the negative control that rules out the obvious fix.

### Findings claimed as contributions
- **INT4 erases the distillation gain** — on two independent student tracks, W4 group-128
  quantization removes the entire KD improvement while leaving the pretrained backbone intact.
  INT8 preserves it fully. This is the headline.
- **The erasure is a failure-*mode* change, not a magnitude change** — quantified via failure-mixture
  analysis: INT4 shifts failures from wrong_answer (competent, running code) to structural
  (missing_function, empty output, degenerate loops).
- **Error-aware token weighting does not prevent it** — QEAD, the natural mitigation, shows no
  measurable retention benefit on either track (negative result, Phase 3).

### Method components (implemented, no longer claimed as the contribution)
- **QEAD token weighting** — simulate INT8 quantization error on student logits per position,
  weight the distillation loss so quantization-sensitive tokens get more attention.
- **Teacher-confidence weighting** — entropy-based: multiply QEAD weights by
  (1 − normalized_entropy(teacher)), so tokens where the teacher is uncertain get lower weight.
- **AST signature-hint extraction** — parse test cases to extract function signatures, inject into
  eval prompt to prevent name mismatches.

### Borrowed / adapted techniques
| Technique | Source | Our adaptation |
|---|---|---|
| Skew-KL loss | DistiLLM (arXiv 2402.03898, ICML 2024) | We compute KL(mixture ‖ teacher) where mixture = λ·student + (1−λ)·teacher. DistiLLM defines SKL as KL(teacher ‖ mixture) — arguments swapped. Our variant is still stable (mixture prevents zero-denominator) but is technically a different objective. Frame as "inspired by" in writeup. |
| Adaptive skew λ | DistiLLM-2 (arXiv 2503.07067) | Per-sample λ = tanh(KL_gap / 4), same as paper. |
| Rejection filtering | DeepSeek-R1 (arXiv 2501.12948) | Only keep teacher traces that pass all unit tests. |
| Curriculum ordering | Self-Paced KD (arXiv 2408.03680) | ENABLED, length-based (`curriculum: length`). |
| Thinking-budget forcing | s1 (arXiv 2502.04267) | 75% think / 25% code budget split. |
| Top-k logit caching | Sparse Logit Sampling (arXiv 2503.16870) | Top-20 token-id + logprob cached per position, remapped to student vocab. |

### Key architectural difference from referenced KD papers

Both MiniLLM (arXiv 2306.08543) and DistiLLM are **on-policy** methods: the student trains on its
own generated samples. Our pipeline is **fully offline** — the teacher generates once, caches to
disk, and the student trains on teacher reference sequences as a fixed dataset. This is a deliberate
choice: a fixed, reproducible teacher signal is needed to isolate the quantization-error effect that
QEAD targets. On-policy sampling would make that signal noisy.

| Dimension | MiniLLM | DistiLLM | Ours |
|---|---|---|---|
| Student trains on | Own samples | Own samples (buffered) | Teacher's cached traces |
| Policy | On-policy (RL/PG) | Adaptive off-policy | Fully offline |
| Loss | Reverse KL | Skew-KL or Skew-Reverse-KL | Skew-mixture KL (custom direction) |
| Teacher access | Full logits, live | Full logits, live | Top-20 cached logits (sparse) |
| Regularizer | L_PT on pretrain corpus | — | Task CE on reference solution |
| Teacher needed at train time | Yes | Yes | No (cache only) |

---

## Current State (as of 2026-07-29)

### Teacher caches (all offline — R1 teacher weights DELETED, never regenerate non-offline)
- `cache/teacher_logprobs_r1_full/` — 2600 files, **1809 pass all unit tests** (69%): Easy 89%,
  Medium 70%, Hard 47% (283/596 passing hard traces)
- `cache/short_cot_coder7b/` — Coder-7B short-CoT helper traces, ~5760 usable train samples
- Two-teacher mix: 1628 R1 (logit KD) + 5385 short-CoT (CE) train, val stays 181 pure-R1

### Evaluation protocol
- Full LeetCode test split: 228 problems (48 easy / 101 medium / 79 hard), pass@5, temp 0.6
- 112/228 problems have broken references → verified-116 subset is the paper headline metric
- Teacher ceiling: R1-7B = 126/228 (55.3% pass@5)

### Headline results (228 problems, pass@5)

| Model | Solved | Verified-116 | Notes |
|---|---|---|---|
| General base original (Qwen2.5-1.5B) | 12 | 11 (9.5%) | **headline track** floor |
| General base distilled (v2 final_last) | 28 | 27 (23.3%) | ×2.33 relative; beats instruct *original* |
| Instruct original (Coder-1.5B-Instruct) | 22 | — | ceiling-evidence track |
| Instruct distilled, best (R1-only KD) | 39 | 31.9% | |
| Instruct distilled, two-teacher (seed 7) | 36 | — | truncation 42 → 3 |
| Teacher R1-7B | 126 | — | upper bound |

**Framing decision (2026-07-29):** lead with the base track — a general non-code 1.5B model reaches
28/228 after distillation, surpassing the code-specialized instruct model's native 22/228. The
instruct track stays as the ceiling evidence (5 converging methods, on-policy probe), since that
evidence exists only there.

### Full quantization grid (solved / 228, pass@5)

| Model | bf16 | INT8 | INT4 |
|---|---|---|---|
| Instruct original | 22 | 24 | 21 |
| Instruct distilled | 36 | 32.5 *(30, 35)* | 20 *(17, 23)* |
| Base original | 12 | 11 | 6 |
| Base distilled | 28 | 31 | 6 *(7, 5)* |

KD gap by precision: bf16 +14 / +16 → INT8 ~+8.5 / +20 → **INT4 −1 / +1**.

### QEAD ablation 2×2, both tracks (Phase 3, complete)

| Track | Variant | bf16 | INT4 | Drop | Retention |
|---|---|---|---|---|---|
| Instruct | QEAD-on | 36 | 20 *(17, 23)* | −16 | **55.6%** |
| Instruct | QEAD-off | 32.5 *(30, 35)* | 18 *(16, 20)* | −14.5 | **55.4%** |
| Base | QEAD-on | 28 | 6 *(7, 5)* | −22 | 21.4% |
| Base | QEAD-off | 26 | 8 | −18 | 30.8% |

**Both QEAD claims fail.** (a) No distillation-quality benefit: instruct 36 vs 32.5, base 28 vs 26 —
both inside noise. The instruct first draw of 30 looked like a real +6 until the seed-42 draw returned
35. (b) No quantization robustness: on instruct, where all four cells are double-drawn, retention is
55.6% vs 55.4% — identical to two tenths of a point. Base's opposite-signed 9.4-pt gap is a
single-draw artifact: it rests on one QEAD-off INT4 measurement of 8 against a two-draw mean of 6,
and the instruct cells show single INT4 draws scatter by 4-6 solves ({17,23} and {16,20}).

### Key findings (see `history/findings.md`)
- Pure-KD ceiling on the 1.5B instruct student is proven exhausted (~38-39): on-policy probe showed
  96.6% teacher-student token agreement; 4 configs converge to the same score. Bigger student (3B)
  is the real lever.
- Teacher swap to OCR-Nemotron-7B regressed and was reverted — R1 cache is canonical.
- Truncation was degeneration loops, not token budget; the short-CoT mix fixed it (42 → 3 → 0).
- Val loss (pure-R1 val set) measures R1 imitation, not solving — decoupled from solve count.
  Best-val checkpoint selection picks inferior snapshots on the base track; always eval `final_last`.
- Base track converged to its own ceiling band (25-28 across three recipes: v1 mix 25, v2 28,
  two-stage curriculum 26), mirroring the instruct 35-39 band. ~73% of failures are wrong_answer
  with zero syntax errors — capacity-bound, not recipe-bound.
- **INT8 is free at 1.5B** for every model tested; the apparent ±3-6 solve swings have no mechanism
  behind them (failure mixture is identical to bf16 in every category).
- **INT4 erases the KD gain on both tracks** while the pretrained backbone survives. KD knowledge
  lives in weight structure that 4-bit rounding destroys.
- **QEAD is a complete null** (negative result, both tracks). It improves neither distillation
  quality (bf16 36 vs 32.5 instruct, 28 vs 26 base) nor quantization retention (55.6% vs 55.4% on
  instruct, all cells double-drawn). The knowledge INT4 destroys is not concentrated in the tokens
  QEAD upweights.

### Active run
None. Phase 3 fully closed 2026-07-30 — both instruct second draws landed (bf16 35, INT4 20) and
confirmed the null at 55.6% vs 55.4% retention.
Next GPU work: 3B student (Phase 4) if the paper needs a scaling result.

---

## Roadmap Status

### Phase 1: Clean end-to-end run — DONE
- [x] Teacher cache built (2600 problems, 1809 passing)
- [x] Full 3-way 228-problem compare with teacher included
- [x] Two-teacher mix pipeline (R1 long-CoT logit KD + Coder-7B short-CoT CE)
- [x] Verified-116 eval subset for paper-grade numbers

### Phase 1.5: Base-model gap study — wrapping up
- [x] General base (Qwen2.5-1.5B) baseline 12/228 and distilled 25/228
- [x] v2 recipe (higher LR, 3 epochs, hard oversample) — best: 28/228 from `final_last`
- [x] Two-stage curriculum (short-CoT SFT → mix) — 26/228, null vs mixing
- [x] Checkpoint-selection lesson: eval `final_last`, val loss decoupled from solves
- [x] Seed rerun of v2 recipe — 28/228 replicated (seed 42), band confirmed; track closed
- [ ] Difficulty-awareness conditioning (postponed by design)

### Phase 2: PTQ INT8/INT4 evaluation — DONE (2026-07-27)
- [x] INT8 via `scripts/quantize_int8.py` — simulated per-channel symmetric W8 fake-quant
      (real compressed INT8 checkpoints are unreadable by vLLM 0.6.x; verified writer-side correct)
- [x] `--original` / `--tag` / `--skip-distilled` flags in compare_eval.py
- [x] Instruct original bf16 22 → INT8 24 (+2, no measurable degradation)
- [x] Instruct distilled bf16 36 → INT8 30 (−6, edge of noise; test-rate unchanged → consistency loss)
- [x] Base distilled bf16 28 → INT8 31 (+3, no measurable degradation)
- [x] Second sampling seed on instruct-distilled INT8: 35 → the −6 was noise; INT8 draws {30, 35}
      overlap the bf16 band. **Verdict: no measurable INT8 degradation for any model at 1.5B.**
- [x] INT4 grid: instruct original 22→21 (**robust**), instruct distilled 36→17, base distilled
      28→7 (trunc 158, degenerate loops). **Distillation dramatically increases INT4 fragility.**
- [x] Second INT4 draw instruct distilled: 23 (draws {17, 23} vs bf16 35-38 — collapse confirmed)
- [x] Second INT4 draw base distilled: 5 (draws {7, 5}, trunc 150)
- [x] Base original INT8 11 / INT4 6 (bf16 12) — grid complete
- [x] Failure-mixture analysis across 4 formats (~116k test executions each) — INT8 ≡ bf16 in every
      category; INT4 shifts wrong_answer 62.7→38.2% and missing_function 0.9→21.4%
- [x] Base verified-116 numbers: original 11 (9.5%), distilled 27 (23.3%)
- [x] Parser stack-overflow fix in `src/utils/reasoning.py` (INT4 bracket spam overflowed `ast.parse`)
- **Headline: INT4 erases the distillation gain on both tracks (instr 20 vs 21, base 6 vs 6);
  INT8 preserves it fully everywhere**

### Phase 3: 2×2 QEAD ablation — DONE (2026-07-29), result is NEGATIVE
- [x] `qead: false` config flag / `--no-qead` CLI flag (uniform weights over response tokens;
      KLD path unchanged, verified identical between modes)
- [x] Train QEAD-off instruct student (`outputs_qead_off`, seed 7, 2ep) — bf16 30, INT4 16
- [x] Train QEAD-off base student (`outputs_general_qead_off`, seed 7, 3ep, val 0.9402) — bf16 26, INT4 8
- [x] Full 2×2 × 2 tracks at {bf16, INT4} (INT8 dropped — no degradation there to differentiate)
- [x] Second draw instruct QEAD-off bf16 (seed 42): 35 → draws {30, 35} vs QEAD-on 36. The +6 was a
      low draw; **no distillation-quality benefit either.**
- [x] Second draw instruct QEAD-off INT4 (seed 42): 20 → draws {16, 20}, mean 18. All four instruct
      cells now double-drawn.
- [x] **Both original claims FALSIFIED.** No bf16 quality gain (36 vs 32.5 instruct, 28 vs 26 base);
      no INT4 retention gain (55.6% vs 55.4% on the fully double-drawn instruct track). QEAD is a
      complete null. Base's opposite-signed gap is a single-draw artifact — see the table above.
- [ ] *Optional:* second INT4 draw on base QEAD-off (n=1 at 8) — the only remaining single-drawn
      cell in the ablation. Would tighten the base row but cannot change the verdict.

### Paper status
- §1-§5 (intro, method, KD ceiling, gap study, quantization grid + failure mixture) — fully
  data-backed, ready to draft now.
- §6 (QEAD ablation) — data complete, write as an honest negative control.
- **Title needs revision.** The current title promises QEAD as the contribution; the evidence does
  not support that. Something closer to *"Quantization Erases Knowledge Distillation Gains in Small
  Code Models"* fits what was actually shown.

### Phase 4: Scaling + supporting experiments — not started
- [ ] 3B student (`config_3b.yaml`, `outputs_3b/` empty) — the proven lever for hard problems;
      measure untrained 3B baseline first. Now also tests whether the INT4 erasure is scale-dependent.
- [ ] Efficiency table: size / VRAM / tokens-per-sec for teacher vs student bf16 vs student INT8
      (INT8 is the deployment recommendation, so this table supports the practical claim)
- [ ] Statistical rigor: pass@5 with multiple seeds on the verified-116 subset
- ~~Ablate confidence weighting / adaptive λ~~ — deprioritized; QEAD is no longer the contribution
- ~~Correlate per-token QEAD weights with INT8 prediction flips~~ — moot given the Phase 3 null
