# Removed artifacts log

Storage cleanup on 2026-08-02 (disk was 99% full, 2.5G free). Items below are from
the knowledge-distillation project, were unused for >1 month (mtime before 2026-07-02),
and are all re-downloadable from HuggingFace. Removed to reclaim space; re-pull if needed.

## 2026-08-02

Removed from `/root/.cache/huggingface/hub` (root-run HF cache; active copies of the
student/base models still live in `/home/hendrik-n/.cache/huggingface`, downloaded 2026-07):

| Model | Size | Last modified | Note |
|---|---|---|---|
| Qwen2.5-Coder-3B-Instruct | 5.8G | 2026-05-19 | 3B-student roadmap model; re-pull when 3B work starts |
| DeepSeek-R1-Distill-Qwen-1.5B | 3.4G | 2026-05-25 | R1 1.5B distill (not the protected 7B teacher cache) |
| Qwen2.5-Coder-1.5B-Instruct | 2.9G | 2026-06-20 | student model; fresh copy remains in /home cache |

Reclaimed: ~12G. All re-downloadable via `huggingface-cli download <repo>`.

Note: R1 teacher artifacts in `/home/kolin/.../cache/` (teacher_logprobs_r1_full 23G,
teacher-model-r1-7b 15G) are PROTECTED and were NOT touched.

### Docker images (rule: >1 month old, >1GB)

Removed unused knowledge-distillation Docker images (0 containers running):

| Image | Size | Age |
|---|---|---|
| knowledge-distillation-quantization-compare_eval:latest | 10.9G | ~7 weeks |
| knowledge-distillation-quantization-evaluate:latest | 10.9G | ~7 weeks |

Rebuildable from the repo Dockerfiles. The train image (26.5G, 8 days old) was KEPT.

