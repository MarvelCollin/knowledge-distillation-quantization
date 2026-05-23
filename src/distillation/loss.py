import math

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer


def build_teacher_distribution(
    logprobs_per_token: list,
    tokenizer: PreTrainedTokenizer,
    vocab_size: int,
    device: torch.device,
    temperature: float,
    top_k_ids: list = None,
    top_k_vals: list = None,
) -> torch.Tensor:
    inv_t = 1.0 / max(temperature, 1e-6)

    if top_k_ids is not None and top_k_vals is not None and len(top_k_ids) > 0:
        seq_len = len(top_k_ids)
        dist = torch.zeros(seq_len, vocab_size, device=device)
        for t in range(seq_len):
            ids = top_k_ids[t]
            vals = top_k_vals[t]
            if not ids:
                continue
            idx_t = torch.as_tensor(ids, dtype=torch.long, device=device)
            val_t = torch.as_tensor(vals, dtype=torch.float, device=device)
            mask = (idx_t >= 0) & (idx_t < vocab_size)
            if not mask.any():
                continue
            idx_t = idx_t[mask]
            val_t = val_t[mask]
            probs = torch.exp(val_t * inv_t)
            dist[t].index_add_(0, idx_t, probs)
            row_sum = dist[t].sum()
            if row_sum > 1e-8:
                dist[t] /= row_sum
        return dist

    seq_len = len(logprobs_per_token)
    dist = torch.zeros(seq_len, vocab_size, device=device)
    for t, top_k in enumerate(logprobs_per_token):
        for token_str, logprob in top_k.items():
            if not token_str:
                continue
            ids = tokenizer.encode(token_str, add_special_tokens=False)
            if len(ids) == 1 and 0 <= ids[0] < vocab_size:
                dist[t, ids[0]] += math.exp(logprob * inv_t)
        row_sum = dist[t].sum()
        if row_sum > 1e-8:
            dist[t] /= row_sum
    return dist


def skew_kld_loss(
    student_logits: torch.Tensor,
    teacher_dist: torch.Tensor,
    qead_weights: torch.Tensor,
    skew_lambda,
    temperature: float,
) -> torch.Tensor:
    student_probs = F.softmax(student_logits / temperature, dim=-1)

    is_empty = teacher_dist.sum(dim=-1, keepdim=True) < 1e-8
    uniform = torch.ones_like(teacher_dist) / teacher_dist.size(-1)
    safe_teacher = torch.where(is_empty.expand_as(teacher_dist), uniform, teacher_dist)
    safe_teacher = safe_teacher.clamp(min=1e-10)

    if torch.is_tensor(skew_lambda) and skew_lambda.dim() == 1:
        lam = skew_lambda.view(-1, 1, 1)
    else:
        lam = skew_lambda

    mixture = lam * student_probs + (1 - lam) * safe_teacher
    mixture = mixture.clamp(min=1e-10)

    kld_per_token = (mixture * (mixture.log() - safe_teacher.log())).sum(dim=-1)
    return (qead_weights * kld_per_token).sum() / qead_weights.sum().clamp(min=1e-8)


def adaptive_skew_lambda(
    student_logits: torch.Tensor,
    teacher_dist: torch.Tensor,
    qead_weights: torch.Tensor,
    base_lambda: float,
    max_lambda: float,
    temperature: float,
) -> torch.Tensor:
    with torch.no_grad():
        student_probs = F.softmax(student_logits.float() / temperature, dim=-1)
        is_empty = teacher_dist.sum(dim=-1, keepdim=True) < 1e-8
        safe_teacher = torch.where(
            is_empty.expand_as(teacher_dist),
            torch.full_like(teacher_dist, 1.0 / teacher_dist.size(-1)),
            teacher_dist,
        ).clamp(min=1e-10)
        student_safe = student_probs.clamp(min=1e-10)
        per_token_kld = (safe_teacher * (safe_teacher.log() - student_safe.log())).sum(dim=-1)
        w_sum = qead_weights.sum(dim=-1).clamp(min=1e-8)
        per_sample_kld = (qead_weights * per_token_kld).sum(dim=-1) / w_sum
        scaled = torch.tanh(per_sample_kld / 4.0)
        lam = base_lambda + (max_lambda - base_lambda) * scaled
        return lam.clamp(min=base_lambda, max=max_lambda)


def task_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def compute_total_loss(
    student_logits: torch.Tensor,
    teacher_dist: torch.Tensor,
    qead_weights: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    skew_lambda,
    temperature: float,
) -> tuple:
    l_distill = skew_kld_loss(student_logits, teacher_dist, qead_weights, skew_lambda, temperature)
    l_task = task_ce_loss(student_logits, labels)
    l_total = alpha * l_distill + (1 - alpha) * l_task
    return l_total, l_distill, l_task
