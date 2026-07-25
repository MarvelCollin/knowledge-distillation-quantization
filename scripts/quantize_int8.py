import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def int_roundtrip_(weight: torch.Tensor, bits: int) -> torch.Tensor:
    # Quantize to intN and dequantize back, so the stored bf16 weights carry exact
    # quantization error. INT8: per-output-channel symmetric (standard W8 PTQ).
    # INT4: group-128 symmetric (standard W4 granularity, as in GPTQ/AWQ).
    qmax = 2 ** (bits - 1) - 1
    if bits == 4:
        out_features, in_features = weight.shape
        w = weight.reshape(out_features, in_features // 128, 128)
        scale = w.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / qmax
        return w.div(scale).round_().clamp_(-qmax - 1, qmax).mul_(scale).reshape(out_features, in_features)
    scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    return weight.div(scale).round_().clamp_(-qmax - 1, qmax).mul_(scale)


def main():
    parser = argparse.ArgumentParser(
        description="Simulated PTQ: intN round-trip on every Linear weight (lm_head excluded), "
                    "saved as a plain bf16 checkpoint that vLLM loads as a normal model. "
                    "INT8 = per-channel symmetric (≡ W8A16 accuracy); INT4 = group-128 symmetric "
                    "(≡ W4A16 accuracy). (Real compressed INT8 checkpoints are unreadable by "
                    "vLLM 0.6.x — both Marlin WNA16 and CUTLASS W8A8 paths dequantize wrongly.)")
    parser.add_argument("--model", required=True,
                        help="Source model: HF hub id or local checkpoint dir.")
    parser.add_argument("--out", required=True,
                        help="Output dir for the quantization-simulated checkpoint.")
    parser.add_argument("--bits", type=int, choices=[8, 4], default=8,
                        help="Weight quantization bit-width (default: 8).")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True)

    n = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and name != "lm_head":
                mod.weight.data = int_roundtrip_(mod.weight.data.float(), args.bits).to(torch.bfloat16)
                n += 1
    print(f"Applied INT{args.bits} round-trip to {n} Linear layers (lm_head excluded).")

    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.model, trust_remote_code=True).save_pretrained(args.out)
    print(f"INT{args.bits}-simulated checkpoint written to {args.out}")


if __name__ == "__main__":
    main()
