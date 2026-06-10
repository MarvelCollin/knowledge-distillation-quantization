# Research Progress

Title: "Efficient Reasoning for Competitive Programming via Knowledge Distillation and Post-Training Quantization of Large Language Models"

## Project Scope

Distill a large reasoning teacher (DeepSeek-R1-Distill-Qwen-7B, bf16) into a small student
(DeepSeek-R1-Distill-Qwen-1.5B) on LeetCode coding problems, then apply Post-Training Quantization
(PTQ INT8 via bitsandbytes) and show that the QEAD-distilled student degrades less after quantization
than a naively distilled one.

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
| Curriculum ordering | Self-Paced KD (arXiv 2408.03680) | Currently DISABLED (`curriculum: none`). |
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

## Current State (as of 2026-06-10)

### Teacher cache (`cache/teacher_logprobs_r1_7b/`)
- 1000 files cached, **481 pass all unit tests** (48.1%)
- After rejection filtering: ~433 train / ~48 val samples
- ⚠️ 48% pass rate is low — half the dataset is discarded. Consider: increase `max_tokens` (currently
  8192), retry failed problems, or investigate why they fail.

### Trained checkpoints
| Checkpoint | Date | Notes |
|---|---|---|
| `outputs/final` | Jun 9 | Latest training run (R1-7B teacher, α=0.5, T=2.0) |
| `outputs/final_T2_alpha05` | Jun 4 | Earlier run, same hyperparams based on name |

### Evaluation results

**Completed 3-way comparison (Jun 5) — 30 problems, pass@5, teacher SKIPPED:**

| Model | Solved | pass@5 | test_pass_rate |
|---|---|---|---|
| Student (original) | 27/30 (90%) | 90.0% | 70.0% |
| Student (distilled) | 28/30 (93.3%) | 93.3% | 53.3% |

Mixed signal: distilled solves +1 problem but lower raw test pass rate. n=30 is too small.

**In-progress comparison (Jun 10) — 100 problems, pass@5, teacher SKIPPED:**

Only Student (original) completed so far:

| | pass@1 | pass@5 | solved | truncated |
|---|---|---|---|---|
| Student (original) | 35.4% | 60% | 39/100 | 35 |

Notably lower than the 30-problem run → earlier run likely got an easy problem sample.
35/100 outputs truncated (hit token budget) — worth monitoring.

**Older evaluate.py run (May 29):** 6/20 solved (30%) — earlier checkpoint or config.

### Known issues to fix
- `compare_eval.py` comparison runs all skip the teacher (`--skip-teacher`). Need a full 3-way run.
- `notes.md` contains the Self-Instruct/Alpaca/WizardLM notes from a synthetic-data exploration
  session — not directly used in the pipeline but useful if dataset augmentation is needed later.

---

## Roadmap Status

### Phase 1: Clean end-to-end run — ~70%
- [x] Teacher cache built (1000 problems, 481 passing)
- [x] Training completed (2 checkpoint runs)
- [ ] compare_eval with 100+ problems (in progress, student-original done)
- [ ] **Include teacher in comparison** (all runs so far skip it)
- [ ] Verify distilled checkpoint is from the latest training run (Jun 9 `outputs/final`)

### Phase 2: PTQ INT8 evaluation — not started
- [ ] Add `--load-in-8bit` flag to `evaluate.py` / `compare_eval.py` using `BitsAndBytesConfig`
- [ ] Run student (original) bf16 vs INT8 as degradation baseline
- [ ] Run student (distilled) bf16 vs INT8 — this is the headline number
- [ ] (Optional) Also evaluate GPTQ/AWQ 4-bit for stronger compression story

### Phase 3: 2×2 QEAD headline experiment — not started
- [ ] Add `qead_enabled: true/false` config flag (~5 lines in `train.py`)
- [ ] Train student with `qead: off` (uniform weights) — same teacher cache, reusable
- [ ] Run the 2×2 matrix: {QEAD on, QEAD off} × {bf16, INT8}
- [ ] **Key claim: Δpass-rate(bf16→INT8) is smaller for QEAD student**

### Phase 4: Supporting experiments — not started
- [ ] Ablate confidence weighting (on/off) and adaptive λ (on/off)
- [ ] Correlate per-token QEAD weights with actual INT8 prediction flips (preempts reviewer attack)
- [ ] Efficiency table: size / VRAM / tokens-per-sec for teacher vs student bf16 vs student INT8
- [ ] Statistical rigor: pass@5 with multiple seeds, ≥100 test problems
- [ ] (Deferred) Synthetic data augmentation via Evol-Instruct — only if dataset size is the bottleneck
