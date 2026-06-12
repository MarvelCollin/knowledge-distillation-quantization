
Dataset split
LeetCode train split → 90% train / 10% val

Prompt format (DeepSeek-R1, arXiv 2501.12948)
→ system prompt allow <think>...</think>, user prompt minta function Python only

Teacher caching pakai top-k token-id logits (Sparse Logit Sampling, arXiv 2503.16870 — preprint)
→ simpan top-20 token-id + logprob teacher di setiap posisi, bukan string, biar 10× lebih cepat saat training

Rejection filtering (DeepSeek-R1, arXiv 2501.12948)
→ buang teacher response yang gagal unit test, sisakan yang lulus saja

Curriculum ordering (Self-Paced KD for Lightweight Code LLMs, arXiv 2408.03680) — ENABLED
→ sort sample dari response pendek ke panjang (easy first). Aktif di config (curriculum: length); fixed-order sampler di train.py, sort by teacher response token_count.

Teacher model: DeepSeek-R1-Distill-Qwen-7B bf16 (DeepSeek-R1, arXiv 2501.12948)
→ reasoning teacher (handle both reasoning chain + code generation), 7B params (vs 1.5B student → 4.6× scale). Pure bf16 no quantization (~15GB VRAM, fit 20GB constraint). Same Qwen2.5 family dengan student → tokenizer compatible untuk top-k logit transfer. Trade-off vs 14B INT8: smaller model tapi pure precision, hindari distillation noise dari quantization teacher.

Student model: DeepSeek-R1-Distill-Qwen-1.5B (DeepSeek-R1, arXiv 2501.12948)
→ small reasoning model bf16, satu vocab family (Qwen2.5) sama teacher. Sama-sama output `<think>...</think>\n\`\`\`python ... \`\`\`` — clean_teacher_cache strip after-code text dan keep think+code structure.

QEAD token weighting (custom, no paper)
→ simulate INT8 quantization error per posisi → kasih bobot tinggi di token yang sensitive ke quantization

Teacher confidence weighting (custom, entropy-based)
→ kalikan QEAD weights dengan (1 − normalized_entropy(teacher)) → token yang teacher-nya bingung dapat bobot rendah

Skew-KL loss (DistiLLM, arXiv 2402.03898 — ICML 2024)
→ KL antara λ·student + (1−λ)·teacher vs teacher — lebih stabil dari forward-KL biasa

Adaptive skew lambda (DistiLLM-2, arXiv 2503.07067 — preprint)
→ per-sample λ = 0.2 + 0.3·tanh(KL/4), interpolasi skew_lambda (0.2) → skew_lambda_max (0.5) — sample yang gap student-teacher besar dapat λ lebih konservatif

Task cross-entropy loss (standard SFT)
→ standard CE student logits vs reference solution text

Convex mix (Hinton et al., NeurIPS 2015)
→ L_total = α · L_distill + (1 − α) · L_task dengan α = 0.7 (naikkan distill weight setelah teacher swap ke R1-Distill-Qwen-7B)

Adafactor optimizer (Shazeer & Stern, ICML 2018, arXiv 1804.04235)
→ memory-efficient optimizer (gak simpan momentum penuh kayak AdamW)

Linear warmup + linear decay (Devlin et al., BERT, NAACL 2019)
→ LR ramp up 10 step, lalu decay linear sampai akhir training

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

Best-checkpoint selection by validation loss (standard ML practice)
→ save student HANYA saat val_loss improve. Sebelumnya save tiap N step + end-of-train overwrite, bisa load checkpoint lebih buruk dari best-seen. Final checkpoint = best val_loss across training.


────────────────────────────────────────────────────────────
Applied speed optimizations
────────────────────────────────────────────────────────────

SDPA attention with FlashAttention-2 backend (Dao, arXiv 2307.08691 — 2023; PyTorch SDPA dispatch)
→ `attn_implementation="sdpa"` di student + teacher + eval. PyTorch SDPA otomatis dispatch ke FlashAttention-2 kernel di GPU SM 8.0+ (Ampere/Hopper). ~1.5-1.8× attention speedup tanpa install flash-attn package — aman buat runtime CUDA image tanpa nvcc.

Async data loading (PyTorch DataLoader standard practice)
→ train_loader + val_loader pakai `num_workers=2, persistent_workers=True, pin_memory=True`. Tokenization + collation overlap dengan GPU compute, hilangkan sequential I/O bottleneck. Estimasi ~10% iter-time speedup.

Mid-train test_eval disabled (custom)
→ `test_eval_steps=0`. Sebelumnya 10 (subprocess code-exec tiap 10 optimizer step → ~5 menit per cycle, ~30-40% wall-time overhead). Final test eval lewat compare_eval.py setelah training selesai.

Total expected: ~2.4× wall-clock training (3:32 → ~1:30).

────────────────────────────────────────────────────────────
Considered, deferred
────────────────────────────────────────────────────────────

flash-attn explicit package (Dao, arXiv 2307.08691)
→ marginal ~10% gain di atas SDPA, tapi butuh nvcc → switch Dockerfile ke `cuda:12.1-devel` (+3GB image). Skip karena SDPA sudah cukup.

Liger Kernel (Hsu et al., arXiv 2410.10989 — NeurIPS 2024 ENLSP)
→ Triton-fused RMSNorm/RoPE/SwiGLU/FusedLinearCrossEntropy. ~20% speed + ~60% less memory di Qwen-class. Perlu monkey-patch spesifik Qwen2ForCausalLM; risiko silent no-op kalau model class beda. Defer sampai pipeline stabil.

torch.compile / TorchDynamo (Ansel et al., ASPLOS 2024)
→ 1.3-1.5× speedup tapi first-compile 30-60s overhead + kadang konflik dengan HF generate dan gradient checkpointing. Defer.

Selective activation recomputation (Korthikanti et al., NVIDIA Megatron-LM, arXiv 2205.05198 — MLSys 2023)
→ alternatif full gradient_checkpointing. Setelah teacher swap response ~200 token, VRAM probably cukup tanpa checkpointing → estimasi 1.3-1.5× faster di backward. Belum diuji, defer.

Sparse top-k KL loss (GKD, Agarwal et al., arXiv 2306.13649 — ICLR 2024; exploit sparsity dari arXiv 2503.16870)
→ teacher distribution non-zero cuma di ≤20 indices per posisi, tapi current `skew_kld_loss` compute full-vocab softmax + KL (152064 ops × seq × batch). Alternatif: KL di union top-k indices teacher + sample student top-k → ~50 indices, 3000× lebih sedikit ops. Custom refactor besar, perlu test compatibility dengan QEAD weights. Defer.
    
QLoRA / LoRA fine-tuning (Hu et al., arXiv 2106.09685 — ICLR 2022; Dettmers et al., arXiv 2305.14314 — NeurIPS 2023)
→ train rank-r adapter (r=32-64), freeze base → ~5× speedup. Tapi changes experiment dari full FT ke PEFT. Defer sampai full-FT baseline jelas.

vLLM teacher cache build (Kwon et al., arXiv 2309.06180 — SOSP 2023)
→ 5-10× cache build speedup (HF generate ~10-25 tok/s vs vLLM ~200-500 tok/s). Tapi ekstraksi top_k logprobs perlu hook custom di vLLM. Cache build cuma 1× setup cost, defer.

8-bit AdamW (Dettmers et al., arXiv 2110.02861 — ICLR 2022)
→ bnb.optim.AdamW8bit, optimizer state quantized 4×. Adafactor saat ini sudah memory-efficient, marginal gain. Defer.

Pre-tokenized dataset cache (standard practice)
→ `__getitem__` re-tokenize per epoch. Pre-tokenize sekali simpan tensor → ~5-10% speedup. Marginal, defer.

Effective batch size scaling (Goyal et al., arXiv 1706.02677 — Facebook AI)
→ saat ini batch=1 × grad_accum=4 = effective 4. Setelah teacher swap response pendek, VRAM ada margin untuk naikkan ke batch=4 × grad_accum=2 = effective 8 → konvergensi lebih cepat dalam total step. Pertimbangkan setelah pipeline stabil.
────────────────────────────────────────────────────────────