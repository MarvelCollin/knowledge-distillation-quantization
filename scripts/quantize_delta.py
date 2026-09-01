import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def int_roundtrip_(weight: torch.Tensor, bits: int, group_size: int = 0) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    if group_size:
        out_features, in_features = weight.shape
        if in_features % group_size != 0:
            raise ValueError(
                f"in_features {in_features} is not divisible by group size {group_size}")
        w = weight.reshape(out_features, in_features // group_size, group_size)
        scale = w.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / qmax
        return w.div(scale).round_().clamp_(-qmax - 1, qmax).mul_(scale).reshape(out_features, in_features)
    scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return weight.div(scale).round_().clamp_(-qmax - 1, qmax).mul_(scale)


def step_size(weight: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    if group_size:
        out_features, in_features = weight.shape
        w = weight.reshape(out_features, in_features // group_size, group_size)
        return w.abs().amax(dim=2).clamp(min=1e-8) / qmax
    return weight.abs().amax(dim=1).clamp(min=1e-8) / qmax


def main():
    parser = argparse.ArgumentParser(
        description="Split-precision PTQ: quantize the pretrained backbone on a coarse grid "
                    "while carrying the distillation update on its own, much narrower scale. "
                    "Reconstructs W = Q_base(W_orig) + Q_delta(W_dist - W_orig) for every Linear "
                    "and writes a plain bf16 checkpoint that vLLM loads as a normal model. "
                    "This is the weight-side intervention that the step-size account predicts "
                    "should retain the distillation gain at 4-bit backbone precision.")
    parser.add_argument("--original", required=True,
                        help="Undistilled backbone checkpoint (the model distillation started from).")
    parser.add_argument("--distilled", required=True,
                        help="Distilled checkpoint. Non-Linear parameters are taken from here.")
    parser.add_argument("--out", required=True, help="Output dir for the reconstructed checkpoint.")
    parser.add_argument("--base-bits", type=int, choices=[3, 4, 5, 6, 7, 8], default=4,
                        help="Bit-width for the backbone grid (default: 4).")
    parser.add_argument("--base-group", type=int, default=128,
                        help="Backbone granularity along in_features; 0 = per-output-channel (default: 128).")
    parser.add_argument("--delta-bits", type=int, choices=[2, 3, 4, 5, 6, 7, 8, 16], default=16,
                        help="Bit-width for the distillation update on its own scale; "
                             "16 keeps it in bf16 (default: 16).")
    parser.add_argument("--delta-group", type=int, default=128,
                        help="Update granularity along in_features; 0 = per-output-channel (default: 128).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the step-size diagnostics and exit without writing a checkpoint.")
    parser.add_argument("--joint", action="store_true",
                        help="Control arm: quantize the distilled weights directly, W = Q_base(W_dist), "
                             "ignoring --delta-bits. Reproduces the standard PTQ grid for a matched "
                             "comparison built by this same script.")
    args = parser.parse_args()

    original = AutoModelForCausalLM.from_pretrained(
        args.original, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.distilled, torch_dtype=torch.bfloat16, trust_remote_code=True)

    orig_linear = {name: mod for name, mod in original.named_modules()
                   if isinstance(mod, torch.nn.Linear) and name != "lm_head"}

    n = 0
    joint_frac_sum = 0.0
    own_frac_sum = 0.0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if not (isinstance(mod, torch.nn.Linear) and name != "lm_head"):
                continue
            w_orig = orig_linear[name].weight.data.float()
            w_dist = mod.weight.data.float()
            delta = w_dist - w_orig

            joint_step = step_size(w_dist, args.base_bits, args.base_group).mean().item()
            own_step = step_size(delta, args.delta_bits if args.delta_bits < 16 else 8,
                                 args.delta_group).mean().item()
            delta_rms = delta.pow(2).mean().sqrt().item()
            joint_frac_sum += delta_rms / joint_step
            own_frac_sum += delta_rms / own_step

            if args.joint:
                new_w = int_roundtrip_(w_dist, args.base_bits, args.base_group)
            else:
                base_q = int_roundtrip_(w_orig, args.base_bits, args.base_group)
                delta_q = delta if args.delta_bits == 16 else int_roundtrip_(
                    delta, args.delta_bits, args.delta_group)
                new_w = base_q + delta_q

            mod.weight.data = new_w.to(torch.bfloat16)
            n += 1

    base_gran = f"group-{args.base_group}" if args.base_group else "per-output-channel"
    if args.joint:
        print(f"[joint control] W = Q{args.base_bits}({base_gran}) applied to distilled weights, "
              f"{n} Linear layers.")
    else:
        delta_gran = f"group-{args.delta_group}" if args.delta_group else "per-output-channel"
        delta_desc = "bf16 (unquantized)" if args.delta_bits == 16 else f"INT{args.delta_bits} {delta_gran}"
        print(f"[split precision] W = Q{args.base_bits}({base_gran}) on backbone "
              f"+ {delta_desc} on the distillation update, {n} Linear layers.")
    print(f"  update size as a fraction of one backbone step : {joint_frac_sum / n:6.2%}")
    print(f"  update size as a fraction of one update step   : {own_frac_sum / n:6.2%}")
    print("  (the second number is what the split grid gives the update; the first is what the "
          "joint grid gives it)")

    if args.dry_run:
        print("dry run: no checkpoint written")
        return

    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.distilled, trust_remote_code=True).save_pretrained(args.out)
    print(f"checkpoint written to {args.out}")


if __name__ == "__main__":
    main()
