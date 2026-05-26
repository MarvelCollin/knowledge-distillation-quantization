# Methodology: Knowledge Distillation for Code Generation

This document surveys recent (2023-2026) knowledge distillation (KD) research for code-generation LLMs and maps each technique to our concrete setup: a **Qwen2.5-Coder-3B-Instruct** student trained from a **local Qwen 14B** teacher on the [`newfacade/LeetCodeDataset`](https://huggingface.co/datasets/newfacade/LeetCodeDataset). Both teacher and student share the **same Qwen tokenizer/vocabulary**, which unlocks several techniques that string-level top-k caching alone cannot.

---

## 1. Landscape of KD Methods for Code LLMs

### 1.1 Divergence-based logit distillation

| Method | Loss | Sampling | Key idea |
|---|---|---|---|
| Vanilla KD (Hinton) | Forward KL `FKL(T‖S)` | Teacher / fixed | Mass-covering, encourages student to cover all teacher modes — produces blurry, hallucinated code. |
| MiniLLM (arXiv 2306.08543) | Reverse KL `RKL(S‖T)` | On-policy student | Mode-seeking, student focuses on high-probability teacher regions; uses REINFORCE with teacher logprob as reward. |
| GKD (DeepMind, 2023) | FKL / RKL / JSD | Mixture `π = λ·p_θ + (1-λ)·p_data` | DAgger-style: interpolate teacher-forced and student-generated tokens. |
| DistiLLM (ICML 2024, arXiv 2402.03898) | **Skew KL** `D_SKL = KL(p‖λp + (1-λ)q)` | Adaptive off-policy buffer | Numerically stable when student probs are near 0; theoretically bounded gradient. |
| DistiLLM-2 (ICML 2025 Oral, arXiv 2503.07067) | Asymmetric: `(1-β)·SKL(teacher-gen) + β·SRKL(student-gen)` | Both teacher and student outputs | "Pull-up" student on teacher outputs, "push-down" on its own bad outputs. Curriculum schedule on α and β. |
| AKL / ToDi / EOPD (2025) | Head/tail partitioned KL, entropy-gated blend | On-policy | Adaptively choose FKL vs RKL per token based on entropy. |

### 1.2 Code-specific extensions

- **Execution-feedback distillation** (GenX, SWE-RM): use unit-test pass/fail as a binary or scalar reward to filter or re-weight teacher samples.
- **Rejection fine-tuning (RFT)**: keep only teacher trajectories that pass the test cases — a degenerate but strong baseline.
- **Self-Paced KD for Code (arXiv 2408.03680)**: rank samples by difficulty (teacher loss / student-teacher gap) and curriculum from easy → hard.
- **OSS-Instruct / Evol-Instruct (Magicoder, WizardCoder)**: synthesize new (problem, solution) pairs from open-source seeds, then distill on the augmented mix.

### 1.3 Beyond logits

- **Feature / hidden-state distillation**: minimize `‖h_S^l − W·h_T^l‖²` between aligned layers. Practical when teacher and student share architecture family (our case: Qwen2 / Qwen2.5).
- **Attention transfer**: match attention map summaries `Σ|a_T|²` between layers.
- **Sparse Logit Sampling (arXiv 2503.16870)**: cache only top-k teacher tokens but with importance-weighted normalization — what we already do for `k=5`, easy to push to `k=50–100`.

---

## 2. What Our Project Already Does

Code references:
- [src/distillation/loss.py:30](src/distillation/loss.py#L30) — `skew_kld_loss` implements DistiLLM-style Skew KLD with `skew_lambda=0.1`.
- [src/distillation/qead.py:14](src/distillation/qead.py#L14) — `compute_qead_weights` re-weights tokens by simulated INT8 quantization error (novel).
- [src/distillation/loss.py:61](src/distillation/loss.py#L61) — convex mix `α·L_distill + (1-α)·L_task_CE`, with `α=0.7`.
- [train.py:211](train.py#L211) — `filter_failed_teacher` discards teacher samples failing dataset test cases (basic rejection sampling).
- [src/teacher/local_teacher.py:11](src/teacher/local_teacher.py#L11) — local Qwen 14B teacher with top-5 logprob extraction, cached to JSON.
- [src/data/dataset.py:38](src/data/dataset.py#L38) — prompt-masked SFT labels (`-100` on prompt tokens).

**Stack summary:** off-policy distillation, fixed top-5 teacher logprobs cached once, Skew-FKL + CE + QEAD weighting, execution-filter on teacher samples.

---

## 3. Recommended Upgrades — Ranked by Impact / Cost

The teacher and student share the **same tokenizer** (Qwen family). This is the single biggest under-exploited fact in the current pipeline: we are paying the cost of string-level top-k caching when we could be using direct token-id logits.

### 3.1 [HIGH] Replace string-based top-k with token-id top-k or full vocab slice

**Problem.** [src/distillation/loss.py:18](src/distillation/loss.py#L18) does `tokenizer.encode(token_str, ...)` per (token, top-k) entry — slow, lossy on multi-token strings (silently dropped via `len(ids) == 1`), and capped at `top_logprobs=5` which leaves ~99.99% of vocab mass uncovered.

**Fix.** In [src/teacher/local_teacher.py:64](src/teacher/local_teacher.py#L64) we already have raw `scores`. Cache `top_logprobs.indices` and `top_logprobs.values` as int/float arrays directly (e.g. top-50 or top-100), keyed by token id. At training time, scatter into a `[seq, vocab]` tensor with no tokenizer round-trip. Eliminates the silent drop in [src/distillation/loss.py:21](src/distillation/loss.py#L21) and removes the dominant CPU cost of `build_teacher_distribution`.

**Expected gain.** Sharper teacher distribution, no token-mismatch loss, ~10× faster batch prep. Single biggest quality + speed win.

### 3.2 [HIGH] Add on-policy / student-generated rollouts (DistiLLM-2 style)

**Problem.** We are 100% off-policy: teacher logprobs come from a one-shot cache built before training starts, so the student never gets corrective signal on tokens *it would actually produce* (classic exposure bias — exactly the issue MiniLLM was built to solve).

**Fix.** Every N steps, sample a short continuation `y_s` from the student under the same prompt, then either:
- (a) run a single teacher forward over `(prompt, y_s)` to obtain teacher logits over student tokens — possible because we keep the teacher loaded only at cache-build time today; reload it periodically, or
- (b) cheaper: keep just teacher logprobs along `y_s` cached after a periodic teacher pass.

Combine into the DistiLLM-2 contrastive objective:

```
L = (1-β)·SKL(p_T, q_S; y_t)   +   β·SRKL(p_T, q_S; y_s)
     ^^^^^^^^^^^^^^^^^^^^^^^         ^^^^^^^^^^^^^^^^^^^^
     "pull up" on teacher data       "push down" on bad student data
```

With β increasing on a schedule `β = clip(epoch/E + step/T, β_0, 1)` (DistiLLM-2 §3).

**Expected gain.** Largest quality improvement reported in the literature for small code LLMs; directly attacks the exposure-bias problem [notes/recap.md](notes/recap.md) already identified.

### 3.3 [HIGH] Execution-aware token / sample weighting

**Problem.** [train.py:211-214](train.py#L211-L214) only does *binary* filtering on test-case pass. A teacher solution that passes 9/10 tests is still useful — currently discarded.

**Fix.** Replace the boolean filter with a continuous sample weight `w_i = pass_rate_i` (or `w_i = exp(γ·pass_rate_i)`), multiplied into `qead_weights` before normalization in [train.py:232](train.py#L232). Also weight tokens *before* the first failing test trace higher than tokens after.

**Expected gain.** Recovers ~40-60% of currently discarded samples, weighted by reliability. Cheap.

### 3.4 [MED] Self-paced / curriculum sample ordering

**Problem.** Random shuffle on `train_loader` — small student sees hard problems first, often diverging on long chain-of-thought completions early.

**Fix.** Score each cached sample by either (a) teacher response length, or (b) student CE on the reference solution after a brief warmup. Sort and present easy → hard. The "Adaptive CoT Distillation Based on LLM Performance" paper ([notes/recap.md](notes/recap.md)) supports the same intuition for code.

**Expected gain.** Faster convergence, lower variance; no extra compute.

### 3.5 [MED] Hidden-state (feature) distillation

**Problem.** We only match output distributions. Qwen14B and Qwen2.5-Coder-3B share the architecture family, so intermediate representations are alignable with a small linear projection.

**Fix.** Add `L_hidden = ‖W·h_S^{l_S} − h_T^{l_T}‖²` for a chosen pair of layers (e.g., student layer 18 ↔ teacher layer 30). Requires teacher to be present during the student forward pass — fine to enable only for short fine-tuning bursts (e.g. last epoch), or precompute teacher hidden states once for the cached prompt + reference solution and store on disk.

**Expected gain.** Reported 1-3% HumanEval gain in attention/feature transfer studies; moderate disk cost.

### 3.6 [MED] Adaptive `α` per sample (DistiLLM-2 curriculum)

**Problem.** Fixed `alpha = 0.7` and `skew_lambda = 0.1` for all samples ([config/config.yaml:23](config/config.yaml#L23)).

**Fix.** Make skew `α_t` larger for samples where teacher and student disagree strongly (hard), smaller where they already agree (easy):

```
α_t  ←  1 − (1 − α_0) · m / (log p_T(y) − log q_S(y))
```

Implement in [src/distillation/loss.py:30](src/distillation/loss.py#L30) as a per-sample tensor instead of a scalar.

### 3.7 [LOW] Replay buffer of past student rollouts (DistiLLM §4 / "Adaptive Off-Policy")

When (3.2) is in place, store recent student generations in a bounded FIFO. Re-sample with probability proportional to `exp(−‖θ_now − θ_at_capture‖)` (or just an age decay) so stale rollouts get dropped. Improves sample efficiency 2-3× per teacher pass.

### 3.8 [LOW] OSS-Instruct / Evol-Instruct data augmentation

Once core distillation is solid, expand the 200-sample LeetCode set by having the **teacher** generate new (problem, test cases, solution) tuples seeded from open-source code snippets (Magicoder recipe). Filter with the same execution check. Most useful if val pass-rate plateaus due to data scarcity rather than model capacity.

---

## 4. Suggested Implementation Order

1. **3.1** (token-id top-k) — pure refactor, no new hyperparameters, immediate quality + speed lift.
2. **3.3** (continuous pass-rate weighting) — one-line change in the train loop.
3. **3.4** (curriculum ordering) — one-time sort.
4. **3.2** (on-policy DistiLLM-2 contrastive) — biggest research lift; needs periodic teacher reload and a β schedule.
5. **3.6** (per-sample α).
6. **3.5** (feature distillation) — only if (1)-(4) saturate.
7. **3.7, 3.8** — scale-out work.

---

## 5. Concrete Config Deltas (proposed)

```yaml
teacher:
  top_logprobs: 50           # was 5 — caches token-id logits, see §3.1
  cache_format: token_ids    # new — bypass tokenizer re-encode
training:
  alpha_init: 0.3            # DistiLLM-2 α_0
  beta_init: 0.0             # DistiLLM-2 SRKL weight, climbs to 1
  on_policy_every: 50        # steps between student-rollout passes
  curriculum: pass_rate_then_length
  sample_weight: pass_rate   # was: binary filter
  hidden_distill_layers: [18, 30]   # student/teacher pair; null disables
```

---

## 6. References

- [MiniLLM: On-Policy Distillation of LLMs (Gu et al., 2023)](https://arxiv.org/abs/2306.08543)
- [DistiLLM: Towards Streamlined Distillation for LLMs (Ko et al., ICML 2024)](https://arxiv.org/abs/2402.03898)
- [DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs (Ko et al., ICML 2025 Oral)](https://arxiv.org/abs/2503.07067)
- [A Survey of On-Policy Distillation for LLMs (2025)](https://arxiv.org/html/2604.00626v3)
- [Sparse Logit Sampling: Accelerating KD in LLMs (2025)](https://arxiv.org/pdf/2503.16870)
- [Self-Paced KD for Lightweight Code LLMs (2024)](https://arxiv.org/pdf/2408.03680)
- [Magicoder: OSS-Instruct (Wei et al., ICML 2024)](https://arxiv.org/abs/2312.02120)
- [Attention and Feature Transfer KD (Nature Sci. Reports, 2023)](https://www.nature.com/articles/s41598-023-43986-y)
- [DistillKit (Arcee AI) — open-source KD toolkit](https://github.com/arcee-ai/DistillKit)

---

## 7. Optimization Log — Problems Encountered & How We Solved Them

This section is a project changelog. Each row is a real problem we hit while building the pipeline, the research paper that informed the fix, and where the fix lives in code.

### 7.1 Problem: top-k string caching was lossy and slow

**Symptom.** Cache stored `{token_str: logprob}` dicts. At train time we re-encoded each `token_str` through the student tokenizer. ~34 % of top-k strings spanned multiple student tokens and were silently dropped (`len(ids) == 1` filter). With `top_logprobs=5`, real coverage was sometimes 3-4 tokens per position.

**Solution.** Cache `top_k_ids` and `top_k_vals` arrays directly in the student-vocab space at cache-build time. Bump `top_logprobs` 5 → 20. Loader has a fast `index_add_` path when these are present, falls back to string path for the old 180 caches (zero migration).
**Inspired by.** [Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs (arXiv 2503.16870, 2025)](https://arxiv.org/pdf/2503.16870) — same insight: caching teacher logits in token-id form gives <10 % overhead vs CE.
**Implemented in.** [src/teacher/local_teacher.py:78-122](src/teacher/local_teacher.py#L78-L122), [src/distillation/loss.py:8-53](src/distillation/loss.py#L8-L53).

### 7.2 Problem: a single scalar `skew_lambda` was wrong for every sample

**Symptom.** Hard samples (large teacher-student gap) need a more conservative mixture for gradient stability; easy samples want a sharper teacher signal. A constant `λ=0.1` over-corrects on easy and under-corrects on hard.

**Solution.** Per-sample `λ_i = base + (max − base) · tanh(KLD_i / 4)`, computed in `torch.no_grad()` so it acts as a constant multiplier per sample.
**Inspired by.** [DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs (Ko et al., ICML 2025 Oral, arXiv 2503.07067)](https://arxiv.org/abs/2503.07067) §3 — curriculum-based α update `α_t ← 1 − (1−α₀)·m/(log p_T − log q_S)`. We use a smoother tanh-clamped variant suited to off-policy training.
**Implemented in.** [src/distillation/loss.py:82-104](src/distillation/loss.py#L82-L104).

### 7.3 Problem: random shuffle made the small student diverge on hard samples first

**Symptom.** With `shuffle=True`, the first batches could include 300-token reasoning traces while the student was still cold. Loss spiked, gradient norm clipped frequently.

**Solution.** Sort samples by teacher response length, present easy → hard via a `_FixedOrderSampler`. Falls back to problem-text length when a cache file is missing.
**Inspired by.** [Self-Paced KD for Lightweight Code LLMs (arXiv 2408.03680, 2024)](https://arxiv.org/pdf/2408.03680) + the "Adaptive CoT Distillation Based on LLM Performance" paper from [notes/recap.md](notes/recap.md) — small students benefit from difficulty-graded curriculum.
**Implemented in.** [src/data/dataset.py:40-58](src/data/dataset.py#L40-L58), [train.py:160-184](train.py#L160-L184).

### 7.4 Problem: every teacher token weighted equally, even where teacher itself was uncertain

**Symptom.** QEAD already weights by student-side quantization error, but ignored the teacher-side signal quality. A position where the teacher distribution is near-uniform (top-5 all ≈0.2) is uninformative; distilling against it just pulls the student toward noise.

**Solution.** Multiply QEAD weights by `(1 − H_norm(teacher))`, then re-normalize per row. Confident teacher positions get more loss weight; uncertain ones get less.
**Inspired by.** Token Importance Filtering / TIP discussed in the [On-Policy Distillation Survey (arXiv 2604.00626, 2025)](https://arxiv.org/html/2604.00626v3) §4. The survey lists entropy-based filtering as a standard preprocessing step for dense logit KD.
**Implemented in.** [src/distillation/qead.py:23-32](src/distillation/qead.py#L23-L32), [train.py:268-273](train.py#L268-L273).

### 7.5 Problem: no reasoning support; prompts actively suppressed CoT

**Symptom.** Teacher (Qwen2.5-Coder-14B) and student (Qwen2.5-Coder-3B) were both instruct models. System prompts said *"Output ONLY a raw Python function definition. No explanation."* — perfect for code-only, fatal if we want reasoning distillation.

**Solution.** Switch to DeepSeek-R1-Distill-Qwen-7B teacher + DeepSeek-R1-Distill-Qwen-1.5B student (same Qwen2.5 tokenizer family — no other plumbing changes). New `reasoning:` config section toggles a CoT-allowing system prompt and a shared `strip_thinking()` extractor that splits on the first `</think>`. Old non-reasoning pipeline still works when `reasoning.enabled: false`. Separate `cache/teacher_logprobs_reasoning/` dir keeps the old code-only cache for A/B comparison.
**Inspired by.** [DeepSeek-R1 paper (2025)](https://arxiv.org/abs/2501.12948) — the exact recipe used to distill R1's reasoning into smaller Qwen2.5 students via pure SFT on teacher-generated reasoning traces.
**Implemented in.** [src/utils/reasoning.py](src/utils/reasoning.py), [src/teacher/local_teacher.py](src/teacher/local_teacher.py), [src/teacher/teacher_api.py](src/teacher/teacher_api.py), [train.py](train.py), [evaluate.py](evaluate.py), [compare_eval.py](compare_eval.py).

### 7.6 Problem: 20 GB VRAM ceiling once reasoning sequences arrived

**Symptom.** `teacher_dist` is `[B, T, V]` fp32 = `B × T × 152064 × 4` bytes. With `B=2, T=4096` that's 5 GB before counting model/grads/optimizer/loss intermediates. Easily OOM on 20 GB.

**Solution.** `batch_size: 1`, `gradient_accumulation_steps: 4` (same effective batch), `max_length: 2048` (covers most reasoning traces; longer ones drop via `filter_failed_teacher`), `adaptive_skew: false` (saves the second `[B,T,V]` fp32 allocation inside the no-grad block). Net peak ≈ 15.7 GB.
**Inspired by.** Standard practice — no specific paper. Empirical memory accounting.
**Implemented in.** [config/config.yaml:25-42](config/config.yaml#L25-L42).

### 7.7 Problem: an unsafe vocab-equality shortcut

**Symptom.** During audit, found a `_same_vocab` shortcut that compared `tokenizer.vocab_size` and trusted teacher token IDs as student IDs. Correct for Qwen2 ↔ Qwen2.5 (both 152064), but would silently corrupt IDs for any cross-family pair with a coincidental vocab size match.

**Solution.** Dropped the shortcut. Always decode → re-encode through the student tokenizer. Cost is microseconds per top-k entry, run once at cache build.
**Inspired by.** N/A — pure correctness audit, not paper-derived.
**Implemented in.** [src/teacher/local_teacher.py:95-110](src/teacher/local_teacher.py#L95-L110).

---

## 8. Further Optimizations — When 2.6 k + Reasoning Becomes the Bottleneck

The full LeetCode set is ~2.6 k samples. With a 7 B reasoning teacher emitting 1.5-3 k tokens per response, naïve HF generation gives roughly:

```
2,600 samples × 2,000 tokens / 30 tok/s ≈ 48 hours just for cache build
```

Training adds more on top. Below are six research-backed ways to cut this, ranked by impact.

### 8.1 [HUGE] Less-Is-More: subsample to 500-1000 high-quality traces

For reasoning distillation, *quality* dominates *quantity*. Multiple 2025 papers demonstrate that 500-1000 carefully chosen samples ≥ 100 k random samples:
- **LIMO (COLM 2025)** — 817 samples → 95.6 % on MATH500, 63.3 % on AIME24, using **1 %** of prior training data ([arXiv 2502.03387](https://arxiv.org/abs/2502.03387)).
- **s1: Simple Test-Time Scaling** — 1 000 samples beat models trained on 800 k.
- **LIMR (Less Is More for RL)** — 1 k samples with RL gave >100 % AIME improvement over LIMO/s1 baselines ([arXiv 2502.11886](https://arxiv.org/pdf/2502.11886)).

**Action for our project.** Build cache only on the top 500-1000 LeetCode problems ranked by (i) difficulty=Medium (the sweet spot, per LIMO §4), (ii) problem-statement length, (iii) test-case count. Reduces cache time **5×** and gives equal-or-better final pass-rate.

### 8.2 [HUGE] vLLM for teacher cache build

`transformers.generate()` is sequential per request. vLLM's PagedAttention + continuous batching gives **up to 24× throughput** on the same hardware. vLLM's `SamplingParams(logprobs=20)` returns exactly the top-k we cache today — drop-in replacement for the HF generate call in [src/teacher/local_teacher.py:62](src/teacher/local_teacher.py#L62).

**Action for our project.** Add a `--engine vllm` switch to local_teacher's precompute path. Estimated cache build: 48 h → **~5 h** on a single 20 GB GPU.
**Inspired by.** [Sparse Logit Sampling (arXiv 2503.16870, 2025)](https://arxiv.org/pdf/2503.16870) §5 explicitly recommends vLLM + cached logits for KD pipelines; [vLLM project docs](https://docs.vllm.ai/).

### 8.3 [BIG] LoRA / QLoRA student training (Tina recipe)

Full fine-tuning of a 1.5 B student needs ~9 GB in weights + grads + optimizer state. LoRA freezes the base, trains only low-rank adapters → roughly **10× memory drop**, **3-5× faster** training step.

For reasoning specifically, the **Tina paper (2025)** distilled DeepSeek-R1 reasoning into smaller models with **LoRA r=32-64 + GRPO** on just 7 k examples for **<\$25 total compute**.

**Action for our project.** Wrap the student in `peft.get_peft_model(student.model, LoraConfig(r=32, target_modules=["q_proj","k_proj","v_proj","o_proj"]))`. Keeps the existing distillation loss unchanged.

### 8.4 [BIG] Sequence packing with FlashAttention-2

Our current loader pads every sample to `max_length=2048`. Reasoning traces vary from 200 to 2000 tokens — average ~50 % padding waste. The HuggingFace `DataCollatorWithFlattening` packs multiple short samples into one flat sequence, using `position_ids` and a block-diagonal attention mask so they don't cross-contaminate. With FA2 this is essentially free.

**Action for our project.** Switch from `padding="max_length"` in [src/data/dataset.py:69](src/data/dataset.py#L69) to dynamic padding + `DataCollatorWithFlattening`. Typical throughput uplift on variable-length data: **1.5-2×**.
**Inspired by.** [Enhancing Training Efficiency Using Packing with Flash Attention (arXiv 2407.09105, 2024)](https://arxiv.org/abs/2407.09105); [HuggingFace blog](https://huggingface.co/blog/packing-with-FA2).

### 8.5 [MED] CoT compression — shorter teacher traces

Reasoning models often emit redundant `"wait, let me reconsider"` loops. Compressing the teacher's CoT before caching shrinks every downstream cost (cache size, training seq length, memory).

- **DRP (Distilled Reasoning Pruning, arXiv 2505.13975, 2025)** — teacher-side step pruning. Cut GSM8K reasoning from 917 → 328 tokens *while improving* accuracy 91.7 → 94.1 %.
- **TokenSkip** — semantic importance scoring drops unimportant reasoning tokens.
- **AutoL2S** — auto long/short reasoning, **−71 %** length with minimal accuracy loss.

**Action for our project.** Post-process cached `text` field with a regex that collapses self-reflection loops, OR a one-shot prompt to the teacher *"summarize your reasoning in ≤ 500 tokens"*, then re-cache logprobs over the shortened trace. Halves cache size and training seq length.

### 8.6 [MED] Two-stage teacher: cheap pre-filter → expensive re-generate

Use a small (1.5 B) teacher to *attempt* every sample, run test cases, keep only the ones that pass. Then have the big (7 B) teacher re-generate cached logprobs **only on the survivors**. This is essentially **rejection fine-tuning** applied at cache-build time.

**Action for our project.** Run [src/teacher/local_teacher.py](src/teacher/local_teacher.py) twice: once with `DeepSeek-R1-Distill-Qwen-1.5B` (fast), once with `7B` on samples whose 1.5B attempt passed. Saves ~40-60 % of expensive teacher generations.

### 8.7 Fast-path config preset for the 2.6 k full set

When 8.1 + 8.2 + 8.3 are in place, the realistic config is roughly:

```yaml
data:
  max_samples: 1000              # LIMO sweet spot (was 200)
  curated_subset: medium_only    # difficulty filter
teacher:
  engine: vllm                   # batched generation
  top_logprobs: 20
training:
  use_lora: true
  lora_r: 32
  lora_alpha: 64
  lora_targets: [q_proj, k_proj, v_proj, o_proj]
  batch_size: 4                  # LoRA lets us go back up
  gradient_accumulation_steps: 2
  packing: flatten_fa2           # DataCollatorWithFlattening
```

Estimated end-to-end wall-clock on 20 GB:

| Stage | Today (sequential HF, 200 samples) | After 8.1+8.2+8.3 (1000 LIMO samples) |
|---|---|---|
| Teacher cache build | ~3 h | ~1 h (vLLM, even with 5× more samples) |
| Student training (1 epoch) | ~1 h | ~30 min (LoRA + packing) |
| **Total** | **~4 h** | **~1.5 h** |

Same order of magnitude for the full 2.6 k set: **~3-4 h** total wall-clock, vs ~50+ h naïvely.

### Additional references for §8

- [LIMO: Less Is More for Reasoning (COLM 2025, arXiv 2502.03387)](https://arxiv.org/abs/2502.03387)
- [LIMR: Less Is More for RL Scaling (arXiv 2502.11886, 2025)](https://arxiv.org/pdf/2502.11886)
- [DRP: Distilled Reasoning Pruning (arXiv 2505.13975, 2025)](https://arxiv.org/html/2505.13975)
- [AutoL2S: Auto Long-Short Reasoning (arXiv 2505.22662, 2025)](https://arxiv.org/html/2505.22662)
- [CODI: CoT Compression via Self-Distillation (arXiv 2502.21074, 2025)](https://arxiv.org/html/2502.21074)
- [Packing with Flash Attention 2 (arXiv 2407.09105, 2024)](https://arxiv.org/abs/2407.09105)
- [vLLM: Efficient LLM Inference with PagedAttention](https://docs.vllm.ai/)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL (arXiv 2501.12948, 2025)](https://arxiv.org/abs/2501.12948)

---

## 9. End-to-End Flow — The Big Picture

Four nested loops: **project → cache build → training step → evaluation**. Each arrow is one concrete thing the code does.

### 9.1 Project-level pipeline (runs once)

```
LeetCode dataset (HF: newfacade/LeetCodeDataset)
  → filter samples with valid test cases + code + problem text       [src/data/dataset.py:102]
  → train / val split (90 / 10)                                      [src/data/dataset.py:119]
  → build teacher logprob cache  (one teacher pass over all train)   [§9.2]
  → sort train indices by teacher response length  (curriculum)      [src/data/dataset.py:40]
  → student SFT loop  (epoch × steps)                                [§9.3]
  → save final checkpoint                                            [train.py:312]
  → evaluate on val split                                            [§9.4]
  → compare student-original vs teacher vs student-distilled         [compare_eval.py]
```

### 9.2 Teacher cache build (runs once per training sample)

```
problem.text
  → PROMPT_TEMPLATE.format(...)                                       [src/data/dataset.py:9]
  → messages = [system=reasoning_prompt, user=prompt]                [src/utils/reasoning.py]
  → tokenizer.apply_chat_template(...)                                [local_teacher.py:50]
  → model.generate(do_sample=False, max_new_tokens=8192,
                   output_scores=True, return_dict_in_generate=True)  [local_teacher.py:59]
  → per generated token:
        decode token_id → token_str
        log_softmax(scores) → topk(20) indices + values
        decode each top-k id → top-k strings
        re-encode each top-k string with student tokenizer
            → keep if single token + in range
            → store (student_id, logprob)
  → JSON: {prompt, text, tokens, logprobs[str→lp], top_k_ids, top_k_vals}
  → write cache/teacher_logprobs_reasoning/{idx}.json                 [local_teacher.py:159]
```

### 9.3 Training step (per batch — runs N × epochs times)

**Front half — fetch & forward:**
```
sampler yields idx (curriculum order)
  → dataset.__getitem__(idx)
  → tokenize(prompt + reference_solution, max_length=2048, pad)
  → labels = input_ids, mask prompt with -100, mask pad with -100     [src/data/dataset.py:60]
  → batch{input_ids, attention_mask, labels, prompt_length, idx}
  → student(input_ids, attention_mask)
  → student_logits  shape [B=1, T=2048, V=152064]
```

**Middle — build distillation signal:**
```
labels != -100  →  response_mask  [B, T]
  → compute_qead_weights(student_logits, response_mask)               [src/distillation/qead.py:14]
        student_logits → simulated INT8 quantize → ||orig − quant||₂
        → per-token weight (mask × error), row-normalized
  → qead_weights  [B, T]

for each sample i in batch:
    load cache file {idx}.json
      → if file missing OR prompt mismatch OR empty logprobs:  skip
      → if filter_failed_teacher:
            strip_thinking(text, "</think>")
            → extract code  → run_test_cases
            → if not all pass:  skip
      → align: dist_start = prompt_length - 1
      → if cache has top_k_ids / top_k_vals (new format):
            FAST PATH: index_add_(exp(vals / T)) → row-normalize       [src/distillation/loss.py:19]
      → else (old 180 caches):
            STRING PATH: re-encode each top-k string via student tokenizer
                         exp(logprob / T) → row-normalize              [src/distillation/loss.py:41]
      → write into teacher_dist[i, dist_start : dist_start+aligned]

teacher_dist  shape [B, T, V=152064]
```

**Back half — weight, mix, loss, step:**
```
valid_teacher_mask = teacher_dist.sum(-1) > 1e-8
  → qead_weights *= valid_teacher_mask                                [train.py:269]
  → qead_weights *= teacher_confidence_weights(teacher_dist)          [src/distillation/qead.py:23]
       (1 − normalized entropy → 1 for peaked teacher, 0 for uniform)
  → row-normalize qead_weights to sum=1 per sample

if adaptive_skew:
    adaptive_skew_lambda(student, teacher, qead_weights, base, max)   [src/distillation/loss.py:82]
      → per-sample KL(teacher ‖ student) → tanh-scaled λ ∈ [base, max]
else:
    λ = skew_lambda (scalar 0.1)

skew_kld_loss(student_logits, teacher_dist, qead_weights, λ, T=2.0)  [src/distillation/loss.py:56]
  → student_probs = softmax(logits / T)
  → mixture = λ·student_probs + (1−λ)·teacher_dist        (DistiLLM)
  → KL(mixture ‖ teacher_dist) per token
  → weighted sum / weight sum  →  L_distill

task_ce_loss(student_logits, labels)                                  [src/distillation/loss.py:107]
  → standard shifted cross-entropy, ignores -100  →  L_task

L_total = α · L_distill + (1 − α) · L_task                            [src/distillation/loss.py:117]

(L_total / grad_accum).backward()
  → every grad_accum steps:
        clip_grad_norm_(max_grad_norm)
        optimizer.step()  (Adafactor)
        scheduler.step()  (linear warmup)
        optimizer.zero_grad()
  → every save_steps:    student.save(checkpoint-N)                   [train.py:304]
  → every eval_steps:    run_validation(val_loader)                   [train.py:307]
```

### 9.4 Evaluation (per val problem)

```
problem.text
  → PROMPT_TEMPLATE.format(...)
  → infer expected function name from test cases                      [evaluate.py:43]
  → messages = [system=reasoning_prompt, user=prompt + "Name fn `X`"]
  → chat template → student.generate(max_new_tokens=2048, greedy)
  → decode generated tokens
  → if reasoning_enabled:  strip_thinking(code, "</think>")           [src/utils/reasoning.py]
  → strip ```python fence
  → keep from first `def \w`
  → keep only the contiguous first function body                      [evaluate.py:19]
  → rename function if name ≠ expected
  → run_test_cases(code, test_cases)                                  [src/evaluation/evaluator.py:35]
  → record pass / fail
aggregate:
  → test-case pass rate, problem solve rate
  → save outputs/eval/results.json
```

### 9.5 What flows through the whole system

A single training token's journey:

```
LeetCode problem text
  → teacher generates reasoning + code  (cache build, once)
  → teacher's top-20 logprobs at that token position  (cached as token_ids)
  → loaded into teacher_dist[B, T, V]  at training time
  → mixed with student's own softmax via Skew KL  (DistiLLM)
  → weighted by:  QEAD (student quant-error) × confidence (1 − teacher entropy) × valid mask
  → contributes to L_distill
  → combined with L_task (CE vs the reference solution)
  → backpropagated through student model
  → gradient accumulates over 4 micro-batches
  → Adafactor updates student weights
  → after many steps:  student emits similar reasoning + correct code at eval time
```

That's the whole picture.

---

## 10. FAQ — Fine-tune teacher first, or distill directly?

**Question:** Should we fine-tune the teacher (`DeepSeek-R1-Distill-Qwen-7B`) on LeetCode first, *then* distill to the student? Or distill directly from the off-the-shelf teacher?

**Answer:** **Direct distillation.** This is what the current pipeline does and what every recent reasoning-distillation paper does.

### The two options compared

```
Option A:  LeetCode → fine-tune 7B teacher → cache → distill to 1.5B student
Option B:  LeetCode → off-the-shelf 7B teacher → cache → distill to 1.5B student   ← what we have
```

| Factor | A: fine-tune then distill | B: direct distill (current) |
|---|---|---|
| Extra compute | +10-20 h fine-tuning the 7B | 0 |
| Dataset size required | ≥ 10 k samples to safely fine-tune 7B | works at any size (LIMO: 817 samples) |
| Overfitting risk | **HIGH** — 7B on 200-2 600 samples → catastrophic forgetting; teacher loses general reasoning prior, student inherits the over-fit signal | LOW — teacher stays general; only the student adapts |
| Paper standard | Older domain-adaptation work (2020-2022, weak teachers) | [DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948), [LIMO](https://arxiv.org/abs/2502.03387), [s1](https://arxiv.org/abs/2502.04267), Sky-T1, OpenR1 |

### Why direct distillation wins here

The teacher was already trained on **800 k** DeepSeek-R1 reasoning traces. Adding 2 600 LeetCode samples with bf16 full fine-tuning would shift the teacher's distribution toward LeetCode-style answers and lose the broader reasoning prior — and then the student would inherit that damaged signal. **The point of distillation is to transfer existing teacher knowledge into a smaller model, not to improve the teacher.**

### How we get teacher-specialization benefits *without* fine-tuning

Two mechanisms already in the pipeline give the upside of a domain-specialized teacher with none of the cost or risk:

1. **`filter_failed_teacher`** ([train.py:241](train.py#L241)) — drops teacher traces that fail the unit tests. Same rejection-sampling recipe DeepSeek used to filter 800 k → ~600 k R1 traces before training their distilled models.
2. **`teacher_confidence_weight`** ([src/distillation/qead.py:23](src/distillation/qead.py#L23)) — down-weights positions where the teacher itself was uncertain.

Together: keep only the teacher's *good* traces, and within each kept trace, weight tokens by how confident the teacher was. This is "soft teacher specialization" at zero extra compute.

### When Option A would actually be worth reconsidering

All three would need to be true:
- Dataset ≥ 10 k samples (we have 200-2 600)
- Target domain is narrow and **off-distribution** for the teacher (e.g. proprietary DSL R1 has never seen)
- Compute budget allows RL on the teacher (per the [Tina paper](https://arxiv.org/pdf/2502.11886) — fine-tune via GRPO, then distill)

None apply to our LeetCode + R1-Distill setup.

### Decision rule

Run [compare_eval.py](compare_eval.py) first. If the off-the-shelf teacher scores **< 50 %** on the LeetCode test split, revisit Option A. A properly-functioning R1-distilled 7 B should land **70 %+** without any further training.
