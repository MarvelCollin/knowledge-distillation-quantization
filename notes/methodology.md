
Dataset split
LeetCode train split → 90% train / 10% val

Prompt format (DeepSeek-R1, arXiv 2501.12948)
→ system prompt allow <think>...</think>, user prompt minta function Python only

Teacher caching pakai top-k token-id logits (Sparse Logit Sampling, arXiv 2503.16870 — preprint)
→ simpan top-20 token-id + logprob teacher di setiap posisi, bukan string, biar 10× lebih cepat saat training

Rejection filtering (DeepSeek-R1, arXiv 2501.12948)
→ buang teacher response yang gagal unit test, sisakan yang lulus saja

Curriculum ordering (Self-Paced KD for Lightweight Code LLMs, arXiv 2408.03680)
→ sort sample dari response pendek ke panjang, student lihat yang mudah dulu sebelum yang susah

Student model: DeepSeek-R1-Distill-Qwen-1.5B (DeepSeek-R1, arXiv 2501.12948)
→ small reasoning model bf16, satu vocab family sama teacher

QEAD token weighting (custom, no paper)
→ simulate INT8 quantization error per posisi → kasih bobot tinggi di token yang sensitive ke quantization

Teacher confidence weighting (custom, entropy-based)
→ kalikan QEAD weights dengan (1 − normalized_entropy(teacher)) → token yang teacher-nya bingung dapat bobot rendah

Skew-KL loss (DistiLLM, arXiv 2402.03898 — ICML 2024)
→ KL antara λ·student + (1−λ)·teacher vs teacher — lebih stabil dari forward-KL biasa

Adaptive skew lambda (DistiLLM-2, arXiv 2503.07067 — preprint)
→ per-sample λ = tanh(KL/4) — sample yang gap student-teacher besar dapat λ lebih konservatif

Task cross-entropy loss (standard SFT)
→ standard CE student logits vs reference solution text

Convex mix (Hinton et al., NeurIPS 2015)
→ L_total = α · L_distill + (1 − α) · L_task dengan α = 0.3

Adafactor optimizer (Shazeer & Stern, ICML 2018, arXiv 1804.04235)
→ memory-efficient optimizer (gak simpan momentum penuh kayak AdamW)

Linear warmup + linear decay (Devlin et al., BERT, NAACL 2019)
→ LR ramp up 5 step, lalu decay linear sampai akhir training

Gradient checkpointing (Chen et al., arXiv 1604.06174)
→ trade compute untuk save VRAM ~30%, recompute activations saat backward

Gradient clipping (Pascanu et al., ICML 2013)
→ clip grad norm di 1.0 biar gak explode

Gradient accumulation (standard)
→ akumulasi 4 micro-batch sebelum optimizer.step() — effective batch size 4

Inference-time budget forcing (s1: Simple Test-Time Scaling, arXiv 2502.04267 — preprint)
→ split thinking budget 75% / code budget 25%, force-inject  ```python kalau model masih thinking pas budget habis

Signature hint extraction (custom, no paper)
→ parse test cases dengan AST, extract param names, masukin ke eval prompt biar model gak salah signature