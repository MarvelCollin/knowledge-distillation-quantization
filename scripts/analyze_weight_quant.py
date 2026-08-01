#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from quantize_int8 import int_roundtrip_

PROJ_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

def resolve(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(path, allow_patterns=["*.safetensors*", "*.json"]))

def tensor_index(root: Path) -> dict:
    from safetensors import safe_open

    index = root / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        return {name: root / shard for name, shard in weight_map.items()}

    single = root / "model.safetensors"
    if not single.exists():
        raise FileNotFoundError(
            f"No safetensors checkpoint under {root}. This script does not read .bin files; "
            "re-save the checkpoint with save_pretrained(safe_serialization=True)."
        )
    with safe_open(single, framework="pt") as f:
        return {name: single for name in f.keys()}

def load_tensor(shard: Path, name: str) -> torch.Tensor:
    from safetensors import safe_open
    with safe_open(shard, framework="pt") as f:
        return f.get_tensor(name)

def linear_weights(idx: dict) -> list:
    names = [n for n in idx if n.startswith("model.layers.") and n.endswith(".weight")]
    return sorted(names, key=lambda n: (int(n.split(".")[2]), n.split(".")[-2]))

def shape_stats(w: torch.Tensor, sigma_k: float) -> dict:
    x = w.reshape(-1).double()
    mu = x.mean()
    centered = x - mu
    var = centered.pow(2).mean()
    sigma = var.sqrt()
    kurt = (centered.pow(4).mean() / var.pow(2) - 3.0) if var > 0 else torch.tensor(0.0)
    outlier = (centered.abs() > sigma_k * sigma).double().mean() if sigma > 0 else torch.tensor(0.0)
    return {
        "std": sigma.item(),
        "max_abs": x.abs().max().item(),
        "excess_kurtosis": kurt.item(),
        "outlier_ratio": outlier.item(),
    }

def quant_error(w: torch.Tensor, bits: int) -> float:
    w32 = w.float()
    if bits == 4 and w32.shape[1] % 128 != 0:
        return float("nan")
    q = int_roundtrip_(w32, bits)
    denom = w32.pow(2).sum()
    return (w32 - q).pow(2).sum().item() / denom.item() if denom > 0 else float("nan")

def pearson(xs: list, ys: list) -> float:
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    d = (x.pow(2).sum().sqrt() * y.pow(2).sum().sqrt()).item()
    return (x * y).sum().item() / d if d > 0 else float("nan")

def rank(v: list) -> list:
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out

def correlate(xs: list, ys: list) -> dict:
    pairs = [(a, b) for a, b in zip(xs, ys) if a == a and b == b]
    if len(pairs) < 3:
        return {"n": len(pairs), "pearson": float("nan"), "spearman": float("nan"), "p_value": None}
    xs2, ys2 = [p[0] for p in pairs], [p[1] for p in pairs]
    r = pearson(xs2, ys2)
    rho = pearson(rank(xs2), rank(ys2))
    p = None
    try:
        from scipy import stats
        p = float(stats.pearsonr(xs2, ys2)[1])
    except Exception:
        pass
    return {"n": len(pairs), "pearson": r, "spearman": rho, "p_value": p}

def mean(v: list) -> float:
    v = [x for x in v if x == x]
    return sum(v) / len(v) if v else float("nan")

def median(v: list) -> float:
    v = sorted(x for x in v if x == x)
    if not v:
        return float("nan")
    m = len(v) // 2
    return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", required=True, help="Reference checkpoint (HF id or local dir).")
    ap.add_argument("--distilled", required=True, help="Comparison checkpoint (HF id or local dir).")
    ap.add_argument("--label", default="pair", help="Name for this pair, used in output.")
    ap.add_argument("--out", required=True, help="Path for the full per-layer JSON.")
    ap.add_argument("--sigma-k", type=float, default=4.0, help="Outlier threshold in sigmas.")
    ap.add_argument("--max-layers", type=int, default=0, help="Smoke test: stop after N tensors.")
    args = ap.parse_args()

    orig_root, dist_root = resolve(args.original), resolve(args.distilled)
    orig_idx, dist_idx = tensor_index(orig_root), tensor_index(dist_root)

    names = linear_weights(orig_idx)
    missing = [n for n in names if n not in dist_idx]
    if missing:
        raise SystemExit(f"{len(missing)} tensors missing from --distilled, e.g. {missing[:3]}")
    if args.max_layers:
        names = names[: args.max_layers]

    print(f"pair={args.label}  original={args.original}  distilled={args.distilled}")
    print(f"{len(names)} Linear weights (lm_head + embeddings excluded), sigma_k={args.sigma_k}")
    print("streaming layer by layer, this takes a few minutes on CPU...", flush=True)

    layers = []
    for i, name in enumerate(names):
        wo = load_tensor(orig_idx[name], name)
        wd = load_tensor(dist_idx[name], name)
        if wo.shape != wd.shape:
            raise SystemExit(f"shape mismatch on {name}: {tuple(wo.shape)} vs {tuple(wd.shape)}")

        wo32, wd32 = wo.float(), wd.float()
        delta = (wd32 - wo32).pow(2).sum().sqrt().item()
        base_norm = wo32.pow(2).sum().sqrt().item()

        rec = {
            "name": name,
            "layer": int(name.split(".")[2]),
            "proj": name.split(".")[-2],
            "shape": list(wo.shape),
            "original": {**shape_stats(wo32, args.sigma_k),
                         "w4_nmse": quant_error(wo32, 4), "w8_nmse": quant_error(wo32, 8)},
            "distilled": {**shape_stats(wd32, args.sigma_k),
                          "w4_nmse": quant_error(wd32, 4), "w8_nmse": quant_error(wd32, 8)},
            "kd_delta_fro": delta,
            "kd_delta_rel": delta / base_norm if base_norm > 0 else float("nan"),
        }
        rec["w4_nmse_diff"] = rec["distilled"]["w4_nmse"] - rec["original"]["w4_nmse"]
        rec["kurtosis_diff"] = rec["distilled"]["excess_kurtosis"] - rec["original"]["excess_kurtosis"]
        layers.append(rec)

        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(names)}", flush=True)
        del wo, wd, wo32, wd32

    def col(path, src=None):
        return [(r[src][path] if src else r[path]) for r in layers]

    n_sharper = sum(1 for r in layers if r["w4_nmse_diff"] > 0)
    summary = {
        "pair": args.label,
        "original": args.original,
        "distilled": args.distilled,
        "n_layers": len(layers),
        "sigma_k": args.sigma_k,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "means": {
            "w4_nmse_original": mean(col("w4_nmse", "original")),
            "w4_nmse_distilled": mean(col("w4_nmse", "distilled")),
            "w8_nmse_original": mean(col("w8_nmse", "original")),
            "w8_nmse_distilled": mean(col("w8_nmse", "distilled")),
            "kurtosis_original": mean(col("excess_kurtosis", "original")),
            "kurtosis_distilled": mean(col("excess_kurtosis", "distilled")),
            "outlier_original": mean(col("outlier_ratio", "original")),
            "outlier_distilled": mean(col("outlier_ratio", "distilled")),
            "kd_delta_rel": mean(col("kd_delta_rel")),
        },
        "medians": {
            "w4_nmse_original": median(col("w4_nmse", "original")),
            "w4_nmse_distilled": median(col("w4_nmse", "distilled")),
            "kurtosis_original": median(col("excess_kurtosis", "original")),
            "kurtosis_distilled": median(col("excess_kurtosis", "distilled")),
        },
        "layers_where_distilled_quantizes_worse": f"{n_sharper}/{len(layers)}",
        "correlations": {
            "kd_delta_rel__vs__w4_nmse_distilled":
                correlate(col("kd_delta_rel"), col("w4_nmse", "distilled")),
            "kd_delta_rel__vs__w4_nmse_diff":
                correlate(col("kd_delta_rel"), col("w4_nmse_diff")),
            "kurtosis_diff__vs__w4_nmse_diff":
                correlate(col("kurtosis_diff"), col("w4_nmse_diff")),
            "kd_delta_rel__vs__kurtosis_diff":
                correlate(col("kd_delta_rel"), col("kurtosis_diff")),
        },
        "by_proj": {
            p: {
                "n": len([r for r in layers if r["proj"] == p]),
                "kd_delta_rel": mean([r["kd_delta_rel"] for r in layers if r["proj"] == p]),
                "w4_nmse_orig": mean([r["original"]["w4_nmse"] for r in layers if r["proj"] == p]),
                "w4_nmse_dist": mean([r["distilled"]["w4_nmse"] for r in layers if r["proj"] == p]),
            }
            for p in PROJ_TYPES if any(r["proj"] == p for r in layers)
        },
    }

    max_layer = max(r["layer"] for r in layers)
    bounds = [(0, max_layer // 3), (max_layer // 3 + 1, 2 * max_layer // 3), (2 * max_layer // 3 + 1, max_layer)]
    summary["by_depth"] = {
        f"{lo}-{hi}": {
            "kd_delta_rel": mean([r["kd_delta_rel"] for r in layers if lo <= r["layer"] <= hi]),
            "w4_nmse_orig": mean([r["original"]["w4_nmse"] for r in layers if lo <= r["layer"] <= hi]),
            "w4_nmse_dist": mean([r["distilled"]["w4_nmse"] for r in layers if lo <= r["layer"] <= hi]),
        }
        for lo, hi in bounds
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "layers": layers}, indent=2))

    m, md = summary["means"], summary["medians"]
    print(f"\n=== {args.label}: {len(layers)} layers ===")
    print(f"{'metric':<28}{'original':>14}{'distilled':>14}")
    for lbl, a, b in [
        ("W4 nmse (mean)", m["w4_nmse_original"], m["w4_nmse_distilled"]),
        ("W4 nmse (median)", md["w4_nmse_original"], md["w4_nmse_distilled"]),
        ("W8 nmse (mean)", m["w8_nmse_original"], m["w8_nmse_distilled"]),
        ("excess kurtosis (mean)", m["kurtosis_original"], m["kurtosis_distilled"]),
        ("outlier ratio (mean)", m["outlier_original"], m["outlier_distilled"]),
    ]:
        print(f"{lbl:<28}{a:>14.6g}{b:>14.6g}")
    print(f"{'mean rel KD delta':<28}{'':>14}{m['kd_delta_rel']:>14.6g}")
    print(f"distilled quantizes worse in {summary['layers_where_distilled_quantizes_worse']} layers")

    print("\n--- correlations across layers ---")
    for k, c in summary["correlations"].items():
        p = "n/a" if c["p_value"] is None else f"{c['p_value']:.3g}"
        print(f"{k:<42} r={c['pearson']:+.3f}  rho={c['spearman']:+.3f}  n={c['n']}  p={p}")

    print("\n--- by projection (mean) ---")
    print(f"{'proj':<12}{'kd_delta_rel':>14}{'w4_orig':>12}{'w4_dist':>12}")
    for p, v in summary["by_proj"].items():
        print(f"{p:<12}{v['kd_delta_rel']:>14.5g}{v['w4_nmse_orig']:>12.5g}{v['w4_nmse_dist']:>12.5g}")

    print("\n--- by depth third (mean) ---")
    print(f"{'layers':<12}{'kd_delta_rel':>14}{'w4_orig':>12}{'w4_dist':>12}")
    for d, v in summary["by_depth"].items():
        print(f"{d:<12}{v['kd_delta_rel']:>14.5g}{v['w4_nmse_orig']:>12.5g}{v['w4_nmse_dist']:>12.5g}")

    top = sorted(layers, key=lambda r: -r["kd_delta_rel"])[:5]
    print("\n--- 5 layers KD changed most ---")
    for r in top:
        print(f"{r['name']:<44} rel_delta={r['kd_delta_rel']:.4g}  w4_diff={r['w4_nmse_diff']:+.3g}")

    print(f"\nfull per-layer JSON -> {out}")

if __name__ == "__main__":
    main()
