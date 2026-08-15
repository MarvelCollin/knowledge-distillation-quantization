import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from quantize_int8 import int_roundtrip_

def legacy_roundtrip_(weight: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    if bits == 4:
        out_features, in_features = weight.shape
        w = weight.reshape(out_features, in_features // 128, 128)
        scale = w.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / qmax
        return w.div(scale).round_().clamp_(-qmax - 1, qmax).mul_(scale).reshape(out_features, in_features)
    scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return weight.div(scale).round_().clamp_(-qmax - 1, qmax).mul_(scale)

def resolve_group_size(bits: int, group_size=None) -> int:
    return group_size if group_size is not None else (128 if bits == 4 else 0)

def step(w: torch.Tensor, bits: int, group_size: int) -> float:
    err = (int_roundtrip_(w.clone(), bits, group_size) - w)
    return err.pow(2).mean().sqrt().item() * (12 ** 0.5)

def main() -> int:
    torch.manual_seed(0)
    shapes = [(1536, 1536), (1536, 8960), (8960, 1536), (256, 4096)]
    failures = []

    for shape in shapes:
        w = torch.randn(*shape, dtype=torch.float32) * 0.02

        for bits in (8, 4):
            got = int_roundtrip_(w.clone(), bits, resolve_group_size(bits))
            want = legacy_roundtrip_(w.clone(), bits)
            if not torch.equal(got, want):
                failures.append(f"{shape} bits={bits}: default path diverges from legacy")

    print(f"legacy equivalence: {len(shapes) * 2 - len(failures)}/{len(shapes) * 2} exact")

    w = torch.randn(1536, 8960, dtype=torch.float32) * 0.02

    steps = {b: step(w, b, 128) for b in (4, 5, 6, 7, 8)}
    print("\nstep size at fixed group-128 (rms error * sqrt(12)):")
    prev = None
    for b in (4, 5, 6, 7, 8):
        ratio = f"{prev / steps[b]:.2f}x smaller than W{b - 1}" if prev else ""
        print(f"  W{b}: {steps[b]:.3e}  {ratio}")
        if prev is not None and steps[b] >= prev:
            failures.append(f"W{b} step is not smaller than W{b - 1}")
        prev = steps[b]

    gsteps = {g: step(w, 4, g) for g in (128, 64, 32)}
    print("\nstep size at fixed W4, varying granularity:")
    for g in (128, 64, 32):
        print(f"  g{g}: {gsteps[g]:.3e}")
    if not (gsteps[32] < gsteps[64] < gsteps[128]):
        failures.append("finer group size did not shrink the step")

    pc8 = step(w, 8, 0)
    print(f"\nW4-g128 step / W8-per-channel step: {steps[4] / pc8:.1f}x")
    print(f"W4-g128 step / W8-g128 step:        {steps[4] / steps[8]:.1f}x (pure bit-width would be 16x)")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nOK: defaults reproduce the legacy grid; both sweep axes move the step monotonically.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
