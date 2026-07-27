# Research Progress

Title: "Efficient Reasoning for Competitive Programming via Knowledge Distillation and Post-Training Quantization of Large Language Models"

## Project Scope

Distill a large reasoning teacher (DeepSeek-R1-Distill-Qwen-7B, bf16, long-CoT) plus a short-CoT
helper teacher (Qwen2.5-Coder-7B-Instruct) into a small student (Qwen2.5-Coder-1.5B-Instruct;
gap-study track: Qwen2.5-1.5B general base) on LeetCode coding problems, then apply Post-Training
Quantization (PTQ INT8 via bitsandbytes) and show that the QEAD-distilled student degrades less
after quantization than a naively distilled one.

### Novel contributions (ours)
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

## Current State (as of 2026-07-23)

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

| Model | Solved | Notes |
|---|---|---|
| Instruct original (Coder-1.5B-Instruct) | 22 | |
| Instruct distilled, best (R1-only KD) | 39 | 31.9% on verified-116 |
| Instruct distilled, two-teacher (seed 7) | 36 | truncation 42 → 3 |
| General base original (Qwen2.5-1.5B) | 12 | low-floor gap study |
| General base distilled, best (v2 final_last) | 28 | ×2.33 relative, hard 2→5, trunc 1 |
| Teacher R1-7B | 126 | upper bound |

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

### Active run
None. Base track closed 2026-07-25 (28/228 replicated across seeds 7 and 42).
Next GPU work: PTQ INT8 evaluation (Phase 2), then 3B student (Phase 4).

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

### Phase 2: PTQ INT8 evaluation — mostly done (2026-07-25)
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
- [ ] Phase 3: train QEAD-off distilled, eval bf16 + INT4 — does QEAD mitigate the fragility?
- [x] Second INT4 draw instruct distilled: 23 (draws {17, 23} vs bf16 35-38 — collapse confirmed)
- [x] Base original INT8 11 / INT4 6 (bf16 12) — grid complete
- **Headline: INT4 erases the distillation gain on both tracks (instr 20 vs 21, base 7 vs 6);
  INT8 preserves it fully everywhere**

### Phase 3: 2×2 QEAD headline experiment — not started
- [x] `qead: false` config flag / `--no-qead` CLI flag (uniform weights over response tokens;
      KLD path unchanged, verified identical between modes)
- [ ] Train student with `qead: off` (uniform weights) — same teacher cache, reusable
- [ ] Run the 2×2 matrix: {QEAD on, QEAD off} × {bf16, INT8}
- [ ] **Key claim: Δpass-rate(bf16→INT8) is smaller for QEAD student**

### Phase 4: Scaling + supporting experiments — not started
- [ ] 3B student (`config_3b.yaml`, `outputs_3b/` empty) — the proven lever for hard problems;
      measure untrained 3B baseline first
- [ ] Ablate confidence weighting (on/off) and adaptive λ (on/off)
- [ ] Correlate per-token QEAD weights with actual INT8 prediction flips (preempts reviewer attack)
- [ ] Efficiency table: size / VRAM / tokens-per-sec for teacher vs student bf16 vs student INT8
- [ ] Statistical rigor: pass@5 with multiple seeds on the verified-116 subset
