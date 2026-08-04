import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from quantize_int8 import int_roundtrip_
from src.distillation.qat import apply_qat, qat_spec, remove_qat

FAILURES = []

def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)

def build() -> nn.Module:
    torch.manual_seed(0)
    m = nn.Sequential()
    m.add_module("q_proj", nn.Linear(1536, 1536, bias=False))
    m.add_module("down_proj", nn.Linear(8960, 1536, bias=False))
    m.add_module("lm_head", nn.Linear(1536, 512, bias=False))
    return m

def main() -> None:
    print("=== QAT smoke ===")
    m = build()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    n = apply_qat(m)
    remove_qat(m)
    after = m.state_dict()
    same_keys = set(before) == set(after)
    same_vals = same_keys and all(torch.equal(before[k], after[k]) for k in before)
    check("apply+remove restores weights bit-exactly", same_vals)
    check("no parametrization keys leak into state_dict",
          not any("parametrizations" in k for k in after))
    check("lm_head excluded, both others wrapped", n == 2, f"wrapped {n}")

    m = build()
    linears = {name: mod for name, mod in m.named_modules() if isinstance(mod, nn.Linear)}
    orig = {name: mod.weight.data.float().clone() for name, mod in linears.items()}
    ref = {name: int_roundtrip_(w.clone(), 4) for name, w in orig.items()}
    apply_qat(m)
    for name, mod in linears.items():
        got = mod.weight.detach().float()
        if name == "lm_head":
            check("lm_head left unquantized", torch.equal(got, orig[name]))
        else:
            check(f"{name}: matches quantize_int8.int_roundtrip_ exactly",
                  torch.equal(got, ref[name]),
                  f"max|diff|={float((got - ref[name]).abs().max()):.3e}")

    m = build()
    apply_qat(m)
    x = torch.randn(4, 1536)
    loss = m.q_proj(x).square().mean()
    loss.backward()
    g = m.q_proj.parametrizations.weight.original.grad
    check("gradient reaches the underlying weight (STE)",
          g is not None and torch.isfinite(g).all() and g.abs().sum() > 0)

    w = torch.randn(8, 128, requires_grad=True)
    from src.distillation.qat import FakeQuantWeight
    FakeQuantWeight(4, 128)(w).sum().backward()
    check("STE gradient is identity, not zero",
          w.grad is not None and torch.allclose(w.grad, torch.ones_like(w.grad)))

    m = build()
    apply_qat(m, bits=4, group_size=128)
    spec = qat_spec(m)
    check("qat_spec reports the active settings", spec == {"bits": 4, "group_size": 128}, str(spec))
    remove_qat(m)
    check("qat_spec is None once removed", qat_spec(m) is None)

    print(f"\n{'ALL PASS' if not FAILURES else 'FAILED: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)

if __name__ == "__main__":
    main()