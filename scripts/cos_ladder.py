import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from analyze_weight_quant import (linear_weights, load_tensor, quant_survival, resolve, tensor_index)

def mean(v: list) -> float:
    v = [x for x in v if x == x]
    return sum(v) / len(v) if v else float("nan")

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cosine transmission of the KD update across a bit-width / group-size "
                    "ladder. Stage 10 second prediction: the behavioural crossover should "
                    "coincide with kd_cos crossing roughly 0.4-0.5.")
    ap.add_argument("--original", required=True)
    ap.add_argument("--distilled", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--max-layers", type=int, default=0)
    args = ap.parse_args()

    orig_idx = tensor_index(resolve(args.original))
    dist_idx = tensor_index(resolve(args.distilled))
    names = linear_weights(orig_idx)
    if args.max_layers:
        names = names[: args.max_layers]

    acc = {b: {"kd_cos": [], "noise_rel": [], "frac_erased": []} for b in args.bits}

    for i, name in enumerate(names):
        wo = load_tensor(orig_idx[name], name).float()
        wd = load_tensor(dist_idx[name], name).float()
        for b in args.bits:
            s = quant_survival(wo, wd, b, args.group_size)
            for k in acc[b]:
                acc[b][k].append(s.get(k, float("nan")))
        del wo, wd
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(names)}", flush=True)

    rows = {str(b): {k: mean(v) for k, v in acc[b].items()} for b in args.bits}

    print(f"\ngroup-{args.group_size}, {len(names)} Linear weights")
    print(f"{'bits':>5} {'kd_cos':>9} {'noise_rel':>11} {'frac_erased':>13}")
    for b in args.bits:
        r = rows[str(b)]
        print(f"{b:>5} {r['kd_cos']:>9.3f} {r['noise_rel']:>11.4f} {r['frac_erased']:>13.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"original": args.original, "distilled": args.distilled,
                   "group_size": args.group_size, "n_layers": len(names),
                   "means": rows}, f, indent=2)
    print(f"\nwritten to {args.out}")

if __name__ == "__main__":
    main()
