import torch
import torch.nn.functional as F


def simulate_int8_quantization(tensor: torch.Tensor) -> torch.Tensor:
    min_val = tensor.amin(dim=-1, keepdim=True)
    max_val = tensor.amax(dim=-1, keepdim=True)
    scale = ((max_val - min_val) / 255.0).clamp(min=1e-8)
    q = tensor.sub(min_val)
    q.div_(scale).round_().clamp_(0, 255).mul_(scale).add_(min_val)
    return q


def compute_qead_weights(logits: torch.Tensor, response_mask: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
    batch, seq_len, vocab = logits.shape
    flat_logits = logits.detach().reshape(-1, vocab)
    idx = response_mask.reshape(-1).nonzero(as_tuple=True)[0]
    error = torch.zeros(batch * seq_len, device=logits.device)
    # Process in chunks to avoid a single (seq_len, vocab) float32 allocation (~4-5 GB at
    # max sequence length), which was causing CUDA OOM near the end of each epoch.
    for start in range(0, len(idx), chunk_size):
        chunk_idx = idx[start : start + chunk_size]
        rows = flat_logits.index_select(0, chunk_idx).float()
        rows_q = simulate_int8_quantization(rows)
        error[chunk_idx] = torch.norm(rows.sub_(rows_q), p=2, dim=-1)
        del rows, rows_q
    error = error.reshape(batch, seq_len)
    row_sums = error.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return error / row_sums


def teacher_confidence_weights(teacher_probs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        nonempty = teacher_probs.sum(dim=-1) > 1e-8
        safe = teacher_probs.clamp(min=1e-10)
        entropy = -(teacher_probs * safe.log()).sum(dim=-1)
        k_eff = (teacher_probs > 1e-10).float().sum(dim=-1).clamp(min=2.0)
        max_entropy = torch.log(k_eff)
        norm_entropy = (entropy / max_entropy).clamp(0.0, 1.0)
        confidence = 1.0 - norm_entropy
        return torch.where(nonempty, confidence, torch.zeros_like(confidence))
