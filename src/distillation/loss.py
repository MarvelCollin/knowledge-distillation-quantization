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


def skew_kld_loss(
    student_logits: torch.Tensor,
    teacher_ids: torch.Tensor,
    teacher_probs: torch.Tensor,
    qead_weights: torch.Tensor,
    skew_lambda,
    temperature: float,
) -> torch.Tensor:
    scaled = student_logits / temperature
    log_z = torch.logsumexp(scaled, dim=-1, keepdim=True).float()
    student_at = (torch.gather(scaled, -1, teacher_ids).float() - log_z).exp()

    if torch.is_tensor(skew_lambda) and skew_lambda.dim() == 1:
        lam = skew_lambda.view(-1, 1, 1)
    else:
        lam = skew_lambda

    mixture = (lam * student_at + (1 - lam) * teacher_probs).clamp(min=1e-10)
    safe_teacher = teacher_probs.clamp(min=1e-10)

    kld_per_token = (teacher_probs * (safe_teacher.log() - mixture.log())).sum(dim=-1)
    return (qead_weights * kld_per_token).sum() / qead_weights.sum().clamp(min=1e-8)


def adaptive_skew_lambda(
    student_logits: torch.Tensor,
    teacher_ids: torch.Tensor,
    teacher_probs: torch.Tensor,
    qead_weights: torch.Tensor,
    base_lambda: float,
    max_lambda: float,
    temperature: float,
) -> torch.Tensor:
    with torch.no_grad():
        scaled = student_logits / temperature
        log_z = torch.logsumexp(scaled, dim=-1, keepdim=True).float()
        student_at = torch.gather(scaled, -1, teacher_ids).float() - log_z
        safe_teacher = teacher_probs.clamp(min=1e-10)
        per_token_kld = (teacher_probs * (safe_teacher.log() - student_at)).sum(dim=-1)
        w_sum = qead_weights.sum(dim=-1).clamp(min=1e-8)
        per_sample_kld = (qead_weights * per_token_kld).sum(dim=-1) / w_sum
        gate = torch.tanh(per_sample_kld / 4.0)
        lam = base_lambda + (max_lambda - base_lambda) * gate
        return lam.clamp(min=base_lambda, max=max_lambda)


def task_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_labels = torch.full_like(labels, -100)
    shift_labels[..., :-1] = labels[..., 1:]
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


def compute_total_loss(
    student_logits: torch.Tensor,
    teacher_ids: torch.Tensor,
    teacher_probs: torch.Tensor,
    qead_weights: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    skew_lambda,
    temperature: float,
) -> tuple:
    l_distill = skew_kld_loss(student_logits, teacher_ids, teacher_probs, qead_weights, skew_lambda, temperature)
    l_task = task_ce_loss(student_logits, labels)
    l_total = alpha * l_distill + (1 - alpha) * l_task
    return l_total, l_distill, l_task
