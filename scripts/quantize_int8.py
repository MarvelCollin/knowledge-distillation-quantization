import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def int8_roundtrip_(weight: torch.Tensor) -> torch.Tensor:
    # Per-output-channel symmetric INT8 (the standard W8 PTQ scheme): quantize to
    # int8 and dequantize back, so the stored bf16 weights carry exact INT8 error.
    scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    return weight.div(scale).round_().clamp_(-128, 127).mul_(scale)


def main():
    parser = argparse.ArgumentParser(
        description="Simulated PTQ INT8: per-channel symmetric int8 round-trip on every Linear "
                    "weight (lm_head excluded), saved as a plain bf16 checkpoint. Numerically "
                    "equivalent to W8A16 inference; loads in vLLM as a normal model. "
                    "(Real compressed INT8 checkpoints are unreadable by vLLM 0.6.x — "
                    "both Marlin WNA16 and CUTLASS W8A8 paths dequantize wrongly.)")
    parser.add_argument("--model", required=True,
                        help="Source model: HF hub id or local checkpoint dir.")
    parser.add_argument("--out", required=True,
                        help="Output dir for the INT8-simulated checkpoint.")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True)

    n = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and name != "lm_head":
                mod.weight.data = int8_roundtrip_(mod.weight.data.float()).to(torch.bfloat16)
                n += 1
    print(f"Applied INT8 round-trip to {n} Linear layers (lm_head excluded).")

    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.model, trust_remote_code=True).save_pretrained(args.out)
    print(f"INT8-simulated checkpoint written to {args.out}")


if __name__ == "__main__":
    main()
