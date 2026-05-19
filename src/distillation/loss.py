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
) -> torch.Tensor:
    seq_len = len(logprobs_per_token)
    dist = torch.zeros(seq_len, vocab_size, device=device)

    for t, top_k in enumerate(logprobs_per_token):
        for token_str, logprob in top_k.items():
            ids = tokenizer.encode(token_str, add_special_tokens=False)
            if len(ids) == 1 and 0 <= ids[0] < vocab_size:
                dist[t, ids[0]] += math.exp(logprob / temperature)
        row_sum = dist[t].sum()
        if row_sum > 1e-8:
            dist[t] /= row_sum

    return dist


def skew_kld_loss(
    student_logits: torch.Tensor,
    teacher_dist: torch.Tensor,
    qead_weights: torch.Tensor,
    skew_lambda: float,
    temperature: float,
) -> torch.Tensor:
    student_probs = F.softmax(student_logits / temperature, dim=-1)

    is_empty = teacher_dist.sum(dim=-1, keepdim=True) < 1e-8
    uniform = torch.ones_like(teacher_dist) / teacher_dist.size(-1)
    safe_teacher = torch.where(is_empty.expand_as(teacher_dist), uniform, teacher_dist)
    safe_teacher = safe_teacher.clamp(min=1e-10)

    mixture = skew_lambda * student_probs + (1 - skew_lambda) * safe_teacher
    mixture = mixture.clamp(min=1e-10)

    kld_per_token = (mixture * (mixture.log() - safe_teacher.log())).sum(dim=-1)
    return (qead_weights * kld_per_token).sum() / qead_weights.sum().clamp(min=1e-8)


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
    skew_lambda: float,
    temperature: float,
) -> tuple:
    l_distill = skew_kld_loss(student_logits, teacher_dist, qead_weights, skew_lambda, temperature)
    l_task = task_ce_loss(student_logits, labels)
    l_total = alpha * l_distill + (1 - alpha) * l_task
    return l_total, l_distill, l_task
