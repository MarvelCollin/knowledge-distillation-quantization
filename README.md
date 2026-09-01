# Efficient Reasoning for Competitive Programming

Code and measurement artifacts for the paper *"Efficient Reasoning for Competitive Programming:
Precision Limits of Knowledge Distillation and Post-Training Quantization of Large Language
Models."*

The study distills a DeepSeek-R1-Distill-Qwen-7B teacher into a 1.5B student on unit-test-verified
LeetCode traces, then asks whether the distillation gain survives post-training quantization.
It does at INT8 and does not at 4 bits, and the weight-space measurements here explain why.

## What this repository contains

| Path | Contents |
|---|---|
| `src/distillation/` | Sparse top-k Skew-KLD loss, QEAD token weighting, quantization-aware distillation |
| `src/teacher/` | Teacher serving and top-k logit cache builder |
| `src/student/`, `src/data/` | Student model wrapper, dataset, batching, cache indexing |
| `src/evaluation/` | Generation and sandboxed unit-test execution harness |
| `scripts/` | Training, evaluation, quantization, and every measurement script behind the paper |
| `config/` | Run configurations (base track, general track, 3B, barebone) |
| `logs/` | Weight-space statistics, cosine transmission ladders, GPTQ transmission measurements |
| `notes/`, `history/`, `experiments/` | Pre-registrations, findings log, and method notes |

### Scripts behind specific results

| Script | Produces |
|---|---|
| `scripts/train.py` | Distillation training |
| `scripts/evaluate.py`, `scripts/compare_eval.py` | Held-out solve rates on the 228-problem split |
| `scripts/analyze_weight_quant.py` | Weight-space statistics (kurtosis, tail mass, NMSE, step fraction) |
| `scripts/cos_ladder.py` | Directional transmission of the update at each bit width |
| `scripts/probe_quant_grid.py` | The precision grid and the W4-to-W8 ladder |
| `scripts/quantize_real.py`, `scripts/quantize_int8.py`, `scripts/quantize_delta.py` | Real-toolchain quantization and split-precision reconstruction |
| `scripts/gptq_transmission.py`, `scripts/compare_quantizers.py` | Calibration-based quantizer comparison |
| `scripts/activation_probe.py` | Teacher-forced activation-space agreement |
| `scripts/significance_tests.py`, `scripts/tost_equivalence.py` | Paired significance and equivalence tests |
| `scripts/evalplus_reasoning_codegen.py`, `scripts/evalplus_significance.py` | HumanEval+ and MBPP+ generality check |
| `scripts/analyze_failures.py`, `scripts/make_figures.py` | Failure taxonomy and paper figures |

## Requirements

A CUDA GPU, Docker with the NVIDIA container runtime, and a Hugging Face token for the base
model downloads. Python dependencies are pinned in `requirements.txt` (PyTorch, Transformers,
vLLM, EvalPlus).

## Running

```bash
cp .env.example .env          # add your HF_TOKEN
docker compose build

docker compose run --rm train                                  # distillation
docker compose run --rm evaluate                               # held-out evaluation
docker compose run --rm compare_eval                           # original vs distilled
```

`./menu.sh` wraps the same pipeline in an interactive menu covering cache building, training,
evaluation, the quantization grid, and the weight-space measurements.

Problem statements are not redistributed here. They are pulled from the
[LeetCodeDataset](https://arxiv.org/abs/2504.14655) release at runtime.

## What is not in this repository

These artifacts are too large for Git and are **available from the corresponding author on
request**:

- **The four student checkpoints** (instruction-tuned and general-base, each distilled and
  undistilled), roughly 3 GB each
- **The sparse top-k teacher logit cache**
- **Per-problem execution records** for each evaluation run

The aggregate measurements those runs produced — weight-space statistics, transmission ladders,
and quantizer comparisons — are included under `logs/`, and every script needed to regenerate
the rest is in `scripts/`.

## License

Code is released under the Apache License 2.0. Teacher traces derived from the LeetCodeDataset
release follow the terms of that release.

## Citation

```bibtex
@article{collin2026efficient,
  title   = {Efficient Reasoning for Competitive Programming: Precision Limits of Knowledge
             Distillation and Post-Training Quantization of Large Language Models},
  author  = {Collin, Marvel and Tjahyadi, Bertrand Geraldo and Sanjaya, Helena Aurelia
             and Suhartono, Derwin},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Under review}
}
```
