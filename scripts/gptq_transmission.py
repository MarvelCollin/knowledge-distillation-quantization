"""Cosine transmission of the KD update through a REAL quantizer.

`cos_ladder.py` answers this for the simulated round-trip by quantizing the bf16
checkpoints itself. This script answers it for a quantizer whose output already
exists on disk (GPTQ, AWQ, real RTN): it reads the four checkpoints directly and
never quantizes anything, so the number is whatever the real toolchain produced.

Metrics match `analyze_weight_quant.quant_survival` exactly, so kd_cos here is
directly comparable to the simulated ladder (base track, g128: 0.194 at W4,
0.438 at W6, 0.807 at W8).

Usage:
  python3 scripts/gptq_transmission.py \
    --original-bf16 Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --distilled-bf16 outputs/final_last \
    --original-quant outputs_gptq/instruct_orig \
    --distilled-quant outputs_gptq/instruct_dist \
    --out logs/gptq_transmission_instruct.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from analyze_weight_quant import linear_weights, load_tensor, resolve, tensor_index


def mean(v: list) -> float:
    v = [x for x in v if x == x]
    return sum(v) / len(v) if v else float("nan")


def survival(wo, wd, qo, qd) -> dict:
    """Same definitions as analyze_weight_quant.quant_survival, but with the
    quantized tensors supplied rather than computed."""
    kd, kdq = wd - wo, qd - qo
    kd_n, kdq_n = kd.pow(2).sum().sqrt(), kdq.pow(2).sum().sqrt()
    denom = (kd_n * kdq_n).item()
    base = wd.pow(2).sum().sqrt()
    d, dq = kd.abs(), kdq.abs()
    return {
        "noise_rel": ((qd - wd).pow(2).sum().sqrt() / base).item() if base > 0 else float("nan"),
        "kd_cos": (kd * kdq).sum().item() / denom if denom > 0 else float("nan"),
        "kd_rel_after": (kdq_n / base).item() if base > 0 else float("nan"),
        "kd_rel_before": (kd_n / base).item() if base > 0 else float("nan"),
        "frac_erased": (dq < 0.1 * d).double().mean().item(),
        "frac_amplified": (dq > 2.0 * d).double().mean().item(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--original-bf16", required=True)
    ap.add_argument("--distilled-bf16", required=True)
    ap.add_argument("--original-quant", required=True)
    ap.add_argument("--distilled-quant", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-layers", type=int, default=0)
    args = ap.parse_args()

    idx = {k: tensor_index(resolve(v)) for k, v in (
        ("ob", args.original_bf16), ("db", args.distilled_bf16),
        ("oq", args.original_quant), ("dq", args.distilled_quant))}

    names = linear_weights(idx["ob"])
    if args.max_layers:
        names = names[: args.max_layers]

    acc, skipped = {}, 0
    for i, name in enumerate(names):
        if not all(name in idx[k] for k in idx):
            skipped += 1
            continue
        t = {k: load_tensor(idx[k][name], name).float() for k in idx}
        s = survival(t["ob"], t["db"], t["oq"], t["dq"])
        for k, v in s.items():
            acc.setdefault(k, []).append(v)
        del t
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(names)}", flush=True)

    rows = {k: mean(v) for k, v in acc.items()}
    n = len(acc.get("kd_cos", []))

    print(f"\nreal-quantizer transmission over {n} Linear weights"
          f"{f' ({skipped} skipped, absent from one checkpoint)' if skipped else ''}")
    for k in ("kd_cos", "noise_rel", "kd_rel_before", "kd_rel_after",
              "frac_erased", "frac_amplified"):
        if k in rows:
            print(f"  {k:<16} {rows[k]:.4f}")
    print("\ncompare simulated g128 ladder (base track): "
          "kd_cos 0.194 (W4), 0.307 (W5), 0.438 (W6), 0.617 (W7), 0.807 (W8)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"original_bf16": args.original_bf16, "distilled_bf16": args.distilled_bf16,
                   "original_quant": args.original_quant, "distilled_quant": args.distilled_quant,
                   "n_layers": n, "skipped": skipped, "means": rows}, f, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
