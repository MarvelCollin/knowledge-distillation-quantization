import torch
import torch.nn as nn
from torch.nn.utils import parametrize

DEFAULT_BITS = 4
DEFAULT_GROUP = 128
SKIP = ("lm_head",)

class FakeQuantWeight(nn.Module):
    def __init__(self, bits: int = DEFAULT_BITS, group_size: int = DEFAULT_GROUP):
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.qmax = 2 ** (bits - 1) - 1

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if w.ndim != 2 or w.shape[1] % self.group_size != 0:
            return w
        out_f, in_f = w.shape
        wf = w.float().reshape(out_f, in_f // self.group_size, self.group_size)
        scale = wf.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / self.qmax
        q = wf.div(scale).round().clamp(-self.qmax - 1, self.qmax).mul(scale)
        q = q.reshape(out_f, in_f).to(w.dtype)
        return w + (q - w).detach()

def _targets(model: nn.Module) -> list:
    return [m for name, m in model.named_modules()
            if isinstance(m, nn.Linear) and not any(s in name for s in SKIP)]

def apply_qat(model: nn.Module, bits: int = DEFAULT_BITS,
              group_size: int = DEFAULT_GROUP) -> int:
    n = 0
    for mod in _targets(model):
        if not parametrize.is_parametrized(mod, "weight"):
            parametrize.register_parametrization(mod, "weight", FakeQuantWeight(bits, group_size))
            n += 1
    return n

def qat_spec(model: nn.Module) -> dict | None:
    for mod in model.modules():
        if parametrize.is_parametrized(mod, "weight"):
            p = mod.parametrizations.weight[0]
            if isinstance(p, FakeQuantWeight):
                return {"bits": p.bits, "group_size": p.group_size}
    return None

def remove_qat(model: nn.Module) -> int:
    n = 0
    for mod in list(model.modules()):
        if parametrize.is_parametrized(mod, "weight"):
            parametrize.remove_parametrizations(mod, "weight", leave_parametrized=False)
            n += 1
    return n
