import math

import torch
import torch.nn.functional as F


def build_teacher_topk(
    top_k_ids: list,
    top_k_vals: list,
    vocab_size: int,
    k: int,
    device: torch.device,
    temperature: float,
) -> tuple:
    inv_t = 1.0 / max(temperature, 1e-6)
    seq_len = len(top_k_ids)
    ids_rows = [[0] * k for _ in range(seq_len)]
    probs_rows = [[0.0] * k for _ in range(seq_len)]
    for t, (ids, vals) in enumerate(zip(top_k_ids, top_k_vals)):
        acc = {}
        for tok_id, val in zip(ids, vals):
            if 0 <= tok_id < vocab_size:
                p = math.exp(val * inv_t)
                if p > 0.0:
                    acc[tok_id] = acc.get(tok_id, 0.0) + p
        if not acc:
            continue
        items = list(acc.items())[:k]
        total = sum(p for _, p in items)
        id_row = ids_rows[t]
        prob_row = probs_rows[t]
        for j, (tok_id, p) in enumerate(items):
            id_row[j] = tok_id
            prob_row[j] = p / total
    ids_out = torch.tensor(ids_rows, dtype=torch.long, device=device)
    probs_out = torch.tensor(probs_rows, dtype=torch.float, device=device)
    return ids_out, probs_out


def _lambda_view(skew_lambda):
    if torch.is_tensor(skew_lambda) and skew_lambda.dim() == 1:
        return skew_lambda.view(-1, 1, 1)
    return skew_lambda


def skew_kld_per_token(
    student_logits: torch.Tensor,
    teacher_ids: torch.Tensor,
    teacher_probs: torch.Tensor,
    lam,
    temperature: float,
) -> torch.Tensor:
    scaled = student_logits / temperature
    log_z = torch.logsumexp(scaled, dim=-1, keepdim=True).float()
    student_at = (torch.gather(scaled, -1, teacher_ids).float() - log_z).exp()
    mixture = (lam * student_at + (1 - lam) * teacher_probs).clamp(min=1e-10)
    safe_teacher = teacher_probs.clamp(min=1e-10)
    return (teacher_probs * (safe_teacher.log() - mixture.log())).sum(dim=-1)


def skew_kld_loss(
    student_logits: torch.Tensor,
    teacher_ids: torch.Tensor,
    teacher_probs: torch.Tensor,
    qead_weights: torch.Tensor,
    skew_lambda,
    temperature: float,
) -> torch.Tensor:
    lam = _lambda_view(skew_lambda)
    kld_per_token = skew_kld_per_token(student_logits, teacher_ids, teacher_probs, lam, temperature)
    return (qead_weights * kld_per_token).sum() / qead_weights.sum().clamp(min=1e-8)


def compute_loss_chunked_backward(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    teacher_ids: torch.Tensor,
    teacher_probs: torch.Tensor,
    qead_weights: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    skew_lambda,
    temperature: float,
    loss_scale: float = 1.0,
    chunk_size: int = 1024,
) -> tuple:
    hidden_d = hidden.detach().requires_grad_(True)
    batch, seq_len, _ = hidden_d.shape
    vocab = lm_head_weight.size(0)

    w_sum = qead_weights.sum().clamp(min=1e-8)
    shift_labels = torch.full_like(labels, -100)
    shift_labels[..., :-1] = labels[..., 1:]
    per_sample_valid = (shift_labels != -100).sum(dim=-1).clamp(min=1)

    lam = _lambda_view(skew_lambda)

    distill_val = 0.0
    task_val = 0.0
    for s in range(0, seq_len, chunk_size):
        e = min(s + chunk_size, seq_len)
        lc = F.linear(hidden_d[:, s:e], lm_head_weight)

        kld_per_token = skew_kld_per_token(lc, teacher_ids[:, s:e], teacher_probs[:, s:e], lam, temperature)
        distill_chunk = (qead_weights[:, s:e] * kld_per_token).sum() / w_sum

        task_flat = F.cross_entropy(
            lc.reshape(-1, vocab),
            shift_labels[:, s:e].reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(batch, -1)
        task_chunk = (task_flat.sum(dim=-1) / per_sample_valid).mean()

        chunk_loss = alpha * distill_chunk + (1 - alpha) * task_chunk
        (chunk_loss * loss_scale).backward()

        distill_val += distill_chunk.item()
        task_val += task_chunk.item()

    if hidden_d.grad is not None:
        hidden.backward(gradient=hidden_d.grad)

    total_val = alpha * distill_val + (1 - alpha) * task_val
    return total_val, distill_val, task_val
