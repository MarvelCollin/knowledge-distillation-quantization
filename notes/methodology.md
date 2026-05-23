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
