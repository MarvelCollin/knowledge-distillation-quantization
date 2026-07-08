import torch
import torch.nn.functional as F


def simulate_int8_quantization(tensor: torch.Tensor) -> torch.Tensor:
    min_val = tensor.amin(dim=-1, keepdim=True)
    max_val = tensor.amax(dim=-1, keepdim=True)
    scale = ((max_val - min_val) / 255.0).clamp(min=1e-8)
    q = tensor.sub(min_val)
    q.div_(scale).round_().clamp_(0, 255).mul_(scale).add_(min_val)
    return q


def fused_qead_kld_pass(hidden: torch.Tensor, lm_head_weight: torch.Tensor,
                        response_mask: torch.Tensor, teacher_ids: torch.Tensor,
                        teacher_probs: torch.Tensor, temperature: float,
                        compute_kld: bool = True, chunk_size: int = 1024,
                        int8_rows: int = 512) -> tuple:
    with torch.no_grad():
        batch, seq_len, dim = hidden.shape
        device = hidden.device
        error = torch.zeros(batch, seq_len, device=device)
        per_token_kld = torch.zeros(batch, seq_len, device=device) if compute_kld else None

        for s in range(0, seq_len, chunk_size):
            e = min(s + chunk_size, seq_len)
            logits = F.linear(hidden[:, s:e].detach(), lm_head_weight)

            if compute_kld:
                scaled = logits / temperature
                log_z = torch.logsumexp(scaled, dim=-1, keepdim=True).float()
                student_at = torch.gather(scaled, -1, teacher_ids[:, s:e]).float() - log_z
                safe_teacher = teacher_probs[:, s:e].clamp(min=1e-10)
                per_token_kld[:, s:e] = (teacher_probs[:, s:e] * (safe_teacher.log() - student_at)).sum(dim=-1)

            flat = logits.reshape(-1, logits.size(-1))
            idx = response_mask[:, s:e].reshape(-1).nonzero(as_tuple=True)[0]
            err_flat = torch.zeros(flat.size(0), device=device)
            for r in range(0, len(idx), int8_rows):
                rows_idx = idx[r : r + int8_rows]
                rows = flat.index_select(0, rows_idx).float()
                rows_q = simulate_int8_quantization(rows)
                err_flat[rows_idx] = torch.norm(rows.sub_(rows_q), p=2, dim=-1)
                del rows, rows_q
            error[:, s:e] = err_flat.reshape(batch, e - s)
            del logits, flat, err_flat

        row_sums = error.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return error / row_sums, per_token_kld


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
