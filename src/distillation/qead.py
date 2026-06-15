import torch
import torch.nn.functional as F


def simulate_int8_quantization(tensor: torch.Tensor) -> torch.Tensor:
    min_val = tensor.min(dim=-1, keepdim=True).values
    max_val = tensor.max(dim=-1, keepdim=True).values
    scale = (max_val - min_val) / 255.0
    scale = scale.clamp(min=1e-8)
    quantized = torch.round((tensor - min_val) / scale).clamp(0, 255)
    return quantized * scale + min_val


def compute_qead_weights(logits: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    logits_fp = logits.float().detach()
    logits_q = simulate_int8_quantization(logits_fp)
    error = torch.norm(logits_fp - logits_q, p=2, dim=-1)
    error = error * response_mask.float()
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
