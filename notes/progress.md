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
- **Neither calibrated PTQ nor quantization-aware training rescues it** — GPTQ (Stage 4) and
  QAT-KD (Stage 8) both fail to recover the W4 gap; QAT-KD additionally costs a large, statistically
  confirmed bf16 accuracy drop (negative result, Phase 4).
- **The erasure is LeetCode-specific in its completeness, not a blanket property of W4** — on
  HumanEval+/MBPP+ (Stage 7), the KD gap only partially attenuates under GPTQ-W4 rather than going
  to zero, significant on MBPP+ at both precisions (378 problems).
- **The erasure is confirmed at the activation level, not just inferred from weights** —
  teacher-forced over 100 fixed reference sequences, the distilled and original models'
  next-token distributions measurably converge under W4 (agreement-rate shift +0.48 pts, 95% CI
  [+0.31, +0.66], bootstrapped over the 100 independent sequences). Small per-token effect,
  statistically unambiguous, consistent with compounding into the large sequence-level collapse.

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
- **INT4 erases the KD gain on both tracks — under the *simulated* quantizer.** Note the
  backbone-survival half holds only on the instruct track (22 → 21); the base original halved
  (12 → 6). State it as an instruct result. **Qualified 2026-08-23:** under *real* calibrated
  GPTQ the erasure replicates on the base track but not on the instruct track (gap +6 / +12,
  neither draw excluding the bf16 +14). The two quantizers disagree on the instruct track, so
  "both tracks" is accurate only for the simulated grid and must be said that way.
- **Mechanism, measured (2026-08-02):** the distillation update is ~0.9% (base) / ~0.5% (instruct)
  of weight norm — **2% of one W4 quantization step, but 26% of a W8 step**. At W8 the update
  crosses bin boundaries often and survives (direction cosine 0.72 / 0.59); at W4 it is below the
  grid resolution and does not (0.19 / 0.11). W4 also *substitutes* a larger perturbation than it
  erases (4.14% vs the true 0.89%), built from ~1.35% incidental bin flips — which is why INT4
  distilled models fail worse than the untrained original rather than reverting to it.
- **The sharpness hypothesis is falsified.** Distilled and original weights are identical in
  kurtosis and outlier ratio to five significant figures, and distilled models quantize marginally
  *better*. Distillation does not make weights harder to quantize; its update is simply too small
  to survive a coarse grid.
- **QEAD is a complete null** (negative result, both tracks). It improves neither distillation
  quality (bf16 36 vs 32.5 instruct, 28 vs 26 base) nor quantization retention (55.6% vs 55.4% on
  instruct, all cells double-drawn). The weight-level analysis explains why: the QEAD-on vs
  QEAD-off weight difference is 39-90× smaller than the W4 perturbation, under 1% of one
  quantization step — two orders of magnitude below the noise floor it was meant to counter.
- **Quantization-aware distillation (Stage 8) also fails, and worse than QEAD.** Training through a
  straight-through W4 fake-quantizer (base track, seed 7) misses its own pre-registered success bar
  (W4 test rate 3.2-3.3% vs the declared ~8% threshold) and significantly damages bf16 accuracy
  (28 → 10/11 solved, p<0.005, 95% CI excludes 0) while the W4 change itself is not distinguishable
  from standard KD's W4 cell (p=1, CI includes 0). See `history/methods.md` → "Stage 8 result:
  quantization-aware distillation fails".

### Active run
None. **Stage 10 (precision dose-response) closed 2026-08-22 with all three pre-registered
predictions holding.** Prior closures: Phase 3 2026-07-30, mechanism analysis 2026-08-02,
calibrated PTQ 2026-08-04, Stage 8 QAT-KD (failed) 2026-08-10, EvalPlus + activation probe +
efficiency table 2026-08-11.

**All planned experiments are now complete.** Remaining work is writing, not measurement:
Stage 10 is not yet in the paper (no dose-response content exists in the Overleaf draft), and
three optional gap-fillers remain (instruct-track GPTQ, verified-116 significance, the
`final_last_seed7` checkpoint caveat). None of the three gate submission.

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
  data-backed, ready to draft now. **Remove the sharpness claim wherever it appears** — falsified
  2026-08-02.
- New mechanism section — data complete (weight-level analysis), and it is the paper's answer to
  "why", which the draft previously could not give.
- §6 (QEAD ablation) — data complete, write as an honest negative control, now with the
  quantitative reason it was never able to work.
- **Title needs revision.** The current title promises QEAD as the contribution; the evidence does
  not support that. Something closer to *"Quantization Erases Knowledge Distillation Gains in Small
  Code Models"* fits what was actually shown.

### Phase 4: Mechanism, real PTQ, rigor — COMPLETE 2026-08-22 (plan: `notes/plan-phase4.md`)
- [x] **Stage 10 precision dose-response — DONE 2026-08-22, ALL THREE PREDICTIONS HOLD.**
      Bit ladder W4-W8 at fixed g128, double-draw on the bracketing cells, cosine ladder, and the
      group-size negative control. (1) Behavioural crossover replicates at **W6**, inside the
      pre-registered W6-W7 bracket — W5 was significant on one draw and not the other, so it is
      reported as unconfirmed and no tiebreak draw was added. (2) Cosine transmission crosses the
      predicted 0.4-0.5 band at **W6** (0.438), so behaviour and weight geometry agree on the same
      bit-width from independent measurements. (3) Negative control: W4 at g128/g64/g32 all stay
      erased (every CI contains 0) while the bit axis recovers — step size, not the nominal "4-bit"
      label, is the operative variable. **Yields a bounded threshold: the gap recovers between ~4%
      and ~9% of a quantization step**, superseding the old 2%-vs-26% bracket, and retroactively
      explains the Stage 8 QAT failure (it trained at a 2.0% step fraction, inside the erasure
      regime). Also corrects the earlier "coherence is a W4 cliff" claim: truncation falls 157→121→30
      *within* W4 as granularity tightens, so coherence and gap recovery are the same variable at
      different thresholds. See `history/methods.md` → "Stage 10 result".
- [x] **Instruct-track calibrated PTQ — DONE 2026-08-23, the erasure does NOT replicate.**
      Real GPTQ-W4 on the instruct track gives gap **+6 [−2,+14] ns** and **+12 [+4,+21] p=0.012**
      across two draws; distilled holds 29/31 solved against 36 at bf16 while the original stays
      flat. **Neither draw can exclude the bf16 gap of +14**, so erasure is unsupported here,
      whereas the base track's CI [−5,+2] does exclude its +16. Revised claim: **complete erasure
      is demonstrated on the general-base track under both real quantizers and NOT on the instruct
      track under calibrated GPTQ.** Also exposes a live methodological caveat: on instruct the
      simulated quantizer puts distilled at 17/23 while real GPTQ puts it at 29/31, the opposite
      direction from the base track's discrepancy. The headline grid is simulated throughout.
      The matching real-RTN cell was **discarded as invalid** (checkpoint was never quantized;
      `--method rtn` + `--save-mode fake` is a silent no-op on current llm-compressor). See
      `history/methods.md` → "Calibrated PTQ on the instruct track".
- [x] **Quantization grid gate** (`scripts/probe_quant_grid.py`, 2026-08-23). Counts distinct values
      per quantization group directly from the safetensors header; group-128 W4 admits at most 16,
      unquantized bf16 shows ~120. Now runs automatically inside `quantize_real.py` after any
      `--save-mode fake` build and exits non-zero on failure. Added because the RTN no-op above
      produced no error, a correct-size loadable checkpoint, and a plausible eval number.
- [x] **Weight-level mechanism analysis** (`scripts/analyze_weight_quant.py`, 2026-08-02, 0 eval
      runs). Four pairs × 196 Linear weights. Falsified the sharpness hypothesis; established the
      step-size mechanism and the quantitative explanation of the QEAD null. See
      `history/methods.md` → "Weight-level mechanism". Outputs in `logs/weightstats_*.json`.
- [x] **Calibrated PTQ (GPTQ vs RTN) — DONE 2026-08-04, headline SURVIVES.** Base track, same
      toolchain, fake-quant onto the trusted bf16 path. KD gap: bf16 +16 → RTN-W4 0 → **GPTQ-W4 −1**.
      Test-case rate gap +11.7 pts → 0.0 pts. Calibration does not rescue distilled knowledge; the
      distilled model sits at or below its own untrained baseline under both quantizers. The erasure
      is not an artifact of naive rounding. See `history/methods.md` → "Calibrated PTQ does not
      rescue the distilled model".
- [x] **Quantization-aware distillation (Stage 8) — DONE 2026-08-10, FAILS its pre-registration.**
      Trained through a straight-through W4 fake-quantizer, base track, double-drawn. Misses the
      declared success bar (W4 test rate 3.2-3.3% vs ~8% threshold) and significantly costs bf16
      accuracy (28 → 10/11 solved, p<0.005) while the W4 change is not distinguishable from standard
      KD's W4 cell (p=1). See `history/methods.md` → "Stage 8 result: quantization-aware
      distillation fails". No fix survives; the finding-paper framing from `plan-phase4.md` §3
      is confirmed as the right call.
- [x] **Activation-side probe — DONE 2026-08-11, erasure confirmed behaviorally.** Teacher-forced
      original/distilled × bf16/GPTQ-W4 over 100 fixed passing R1 traces (`scripts/activation_probe.py`,
      vLLM `prompt_logprobs`, single forward pass per sequence). The original-distilled output-agreement
      rate rises under W4 (+0.48 pts, 95% CI [+0.31,+0.66], bootstrapped over the 100 sequences) —
      the two models' behavior measurably converges, the activation-level signature of erasure.
      See `history/methods.md` → "Activation-side probe: the erasure is confirmed behaviorally".
- [ ] Real compressed checkpoints + simulated-vs-real agreement check (validates the fake-quant
      methodology every recorded number depends on)

### Supporting experiments — remaining and cut
- [x] **Efficiency table — DONE 2026-08-11.** Teacher vs distilled student, identical profiling
      conditions (`scripts/efficiency_table.py`). Student is ~4.9x smaller, ~2.4x faster than the
      teacher at bf16 (14.19GB/994 tok/s vs 2.88GB/2417 tok/s). Theoretical INT8 footprint (bit-width
      only — real compressed-checkpoint throughput is blocked by the Marlin defect, same limitation
      as Stage 6) is ~9.9x smaller than the teacher's bf16 size. See `history/methods.md` →
      "Efficiency table: teacher vs distilled student".
- [x] **Statistical rigor — DONE 2026-08-10, extended to verified-116 on 2026-08-22.** Paired exact
      McNemar + bootstrap CI (`scripts/significance_tests.py`) covers every
      bf16/INT4/INT8/RTN4/GPTQ4/QAT comparison in the grid. **Re-run on the verified-116 subset
      (the paper's headline metric) and every conclusion replicates**: significant stays significant,
      null stays null, point estimates move by at most 2 solves. Load-bearing contrast on
      verified-116: base KD gap **+18 [+10,+26] at INT8** vs **+2 / 0 (ns) at INT4**. Side-by-side
      table in `notes/significance_tests.md`.
- [x] **Second benchmark (EvalPlus) — DONE 2026-08-11, generality claim reframed, not simply
      confirmed.** HumanEval+/MBPP+, pass@1 greedy, using this project's own reasoning protocol
      (an EvalPlus-native run first showed a ≈0 gap at both precisions — a measurement artifact,
      since EvalPlus's default prompt never invokes the model's trained think phase). Under the
      matched protocol: KD gap is positive at bf16 on both benchmarks, and — unlike LeetCode's
      complete erasure — only partially attenuates under GPTQ-W4 (retains ~49-57% by point
      estimate). MBPP+'s gap is significant at both precisions (McNemar p=0.0004 bf16, p=0.013 W4,
      378 problems); HumanEval+ trends the same direction but doesn't reach significance at 164
      problems. See `history/methods.md` → "Stage 7 result: generality on EvalPlus".
- ~~3B student~~ — cut per `notes/plan-phase4.md`. A bf16 full fine-tune of 3B does not fit the
  single 24 GB card alongside 32k-token traces, and any workaround (LoRA, shortened context,
  offload) makes the run protocol-mismatched to every 1.5B number in the paper. Declared as a
  limitation with the VRAM reason stated.
- ~~Second model family~~ — infeasible, not merely unaffordable: the R1 cache stores top-20 token
  ids remapped to the Qwen2.5 vocabulary and the teacher weights are deleted, so a non-Qwen student
  cannot consume it and re-caching is impossible. Declared as a limitation.
- ~~Ablate confidence weighting / adaptive λ~~ — deprioritized; QEAD is no longer the contribution
- ~~Correlate per-token QEAD weights with INT8 prediction flips~~ — moot given the Phase 3 null
