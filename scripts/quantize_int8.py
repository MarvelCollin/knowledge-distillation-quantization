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


def main():
    parser = argparse.ArgumentParser(
        description="Simulated PTQ: intN round-trip on every Linear weight (lm_head excluded), "
                    "saved as a plain bf16 checkpoint that vLLM loads as a normal model. "
                    "Bit-width (--bits) and granularity (--group-size) are independent axes. "
                    "The defaults reproduce the original grid: INT8 per-channel symmetric "
                    "(≡ W8A16 accuracy) and INT4 group-128 symmetric (≡ W4A16 accuracy). "
                    "(Real compressed INT8 checkpoints are unreadable by vLLM 0.6.x — both "
                    "Marlin WNA16 and CUTLASS W8A8 paths dequantize wrongly.)")
    parser.add_argument("--model", required=True,
                        help="Source model: HF hub id or local checkpoint dir.")
    parser.add_argument("--out", required=True,
                        help="Output dir for the quantization-simulated checkpoint.")
    parser.add_argument("--bits", type=int, choices=[3, 4, 5, 6, 7, 8], default=8,
                        help="Weight quantization bit-width (default: 8).")
    parser.add_argument("--group-size", type=int, default=None,
                        help="Quantization granularity along in_features: 0 = per-output-channel, "
                             "128 = standard W4 grouping. Defaults preserve the original grid "
                             "(0 for --bits 8, 128 for --bits 4); set explicitly to sweep "
                             "granularity independently of bit-width.")
    args = parser.parse_args()

    group_size = args.group_size if args.group_size is not None else (128 if args.bits == 4 else 0)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True)

    n = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and name != "lm_head":
                mod.weight.data = int_roundtrip_(
                    mod.weight.data.float(), args.bits, group_size).to(torch.bfloat16)
                n += 1
    gran = f"group-{group_size}" if group_size else "per-output-channel"
    print(f"Applied INT{args.bits} {gran} round-trip to {n} Linear layers (lm_head excluded).")

    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.model, trust_remote_code=True).save_pretrained(args.out)
    print(f"INT{args.bits} ({gran}) simulated checkpoint written to {args.out}")

if __name__ == "__main__":
    main()
