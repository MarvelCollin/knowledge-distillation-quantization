import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_weight_quant import resolve, tensor_index, load_tensor, linear_weights

def levels(w, max_elems=200000):
    flat = w.flatten()
    if flat.numel() > max_elems:
        flat = flat[:max_elems]
    return torch.unique(flat)

def level_report(w):
    lv = levels(w)
    n = int(lv.numel())
    out = {"distinct_levels_sampled": n}
    if n == 0:
        return out
    lo, hi = float(lv.min()), float(lv.max())
    out["level_min"] = lo
    out["level_max"] = hi
    span = max(abs(lo), abs(hi))
    out["symmetry_offset"] = (abs(lo + hi) / span) if span > 0 else 0.0
    out["looks_symmetric"] = bool(out["symmetry_offset"] < 0.05)
    return out

def inferred_group_size(w_orig, w_q, max_row_len=4096):
    if w_orig.dim() != 2:
        return None
    row_o = w_orig[0][:max_row_len].to(torch.float32)
    row_q = w_q[0][:max_row_len].to(torch.float32)
    err = (row_q - row_o).abs()
    if err.numel() < 4:
        return None
    boundaries = []
    window = 8
    prev = None
    for i in range(0, err.numel() - window, window):
        m = float(err[i:i + window].max())
        if prev is not None and prev > 0 and abs(m - prev) / prev > 0.25:
            boundaries.append(i)
        prev = m
    if len(boundaries) < 2:
        return None
    gaps = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
    gaps.sort()
    return int(gaps[len(gaps) // 2])

def nmse(a, b):
    d = (a.to(torch.float32) - b.to(torch.float32))
    den = float((b.to(torch.float32) ** 2).sum())
    return float((d ** 2).sum()) / den if den > 0 else 0.0

def relfro(a, b):
    d = (a.to(torch.float32) - b.to(torch.float32))
    den = float(torch.linalg.norm(b.to(torch.float32)))
    return float(torch.linalg.norm(d)) / den if den > 0 else 0.0

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", required=True, help="Unquantized bf16 checkpoint.")
    ap.add_argument("--sim", required=True, help="Simulated round-trip W4 checkpoint (quantize_int8.py --bits 4).")
    ap.add_argument("--real", required=True, help="Real-toolchain W4 checkpoint saved fake-quant (quantize_real.py).")
    ap.add_argument("--out", required=True, help="Path for the per-layer JSON.")
    ap.add_argument("--max-layers", type=int, default=0, help="Smoke test: stop after N tensors.")
    args = ap.parse_args()

    po, ps, pr = resolve(args.original), resolve(args.sim), resolve(args.real)
    io, isim, ireal = tensor_index(po), tensor_index(ps), tensor_index(pr)
    names = linear_weights(io)
    if args.max_layers:
        names = names[:args.max_layers]

    rows = []
    for i, name in enumerate(names):
        if name not in isim or name not in ireal:
            continue
        wo = load_tensor(po / io[name], name)
        ws = load_tensor(ps / isim[name], name)
        wr = load_tensor(pr / ireal[name], name)
        rows.append({
            "layer": name,
            "nmse_sim_vs_orig": nmse(ws, wo),
            "nmse_real_vs_orig": nmse(wr, wo),
            "relfro_sim_vs_real": relfro(ws, wr),
            "sim_levels": level_report(ws),
            "real_levels": level_report(wr),
            "sim_group_size": inferred_group_size(wo, ws),
            "real_group_size": inferred_group_size(wo, wr),
        })
        if (i + 1) % 25 == 0:
            print("  ...%d/%d" % (i + 1, len(names)))

    if not rows:
        print("No overlapping linear weights found -- check the three paths.")
        return

    def avg(k):
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        return sum(vals) / len(vals) if vals else float("nan")

    print()
    print("=" * 78)
    print("SIMULATED vs REAL W4 -- %d linear layers" % len(rows))
    print("=" * 78)
    print("mean NMSE, simulated vs original : %.6g" % avg("nmse_sim_vs_orig"))
    print("mean NMSE, real vs original      : %.6g" % avg("nmse_real_vs_orig"))
    print("mean relative Frobenius, sim<->real: %.6g" % avg("relfro_sim_vs_real"))
    print()
    s0, r0 = rows[0]["sim_levels"], rows[0]["real_levels"]
    print("first layer (%s):" % rows[0]["layer"])
    print("  simulated : %s distinct levels, symmetric=%s (offset %.4f)"
          % (s0.get("distinct_levels_sampled"), s0.get("looks_symmetric"),
             s0.get("symmetry_offset", float("nan"))))
    print("  real      : %s distinct levels, symmetric=%s (offset %.4f)"
          % (r0.get("distinct_levels_sampled"), r0.get("looks_symmetric"),
             r0.get("symmetry_offset", float("nan"))))
    gs_sim = [r["sim_group_size"] for r in rows if r["sim_group_size"]]
    gs_real = [r["real_group_size"] for r in rows if r["real_group_size"]]
    if gs_sim:
        gs_sim.sort()
        print("  inferred group size, simulated : %d (median over %d layers)"
              % (gs_sim[len(gs_sim) // 2], len(gs_sim)))
    if gs_real:
        gs_real.sort()
        print("  inferred group size, real      : %d (median over %d layers)"
              % (gs_real[len(gs_real) // 2], len(gs_real)))

    print()
    print("Interpretation: if the two NMSEs differ materially, one quantizer rounds")
    print("harder than the other and the eval gap is a rounding-strength difference.")
    print("If they match but relfro_sim_vs_real is large, the grids are offset from")
    print("each other (symmetric vs asymmetric, or misaligned groups) rather than coarser.")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"layers": rows}, fh, indent=2)
    print("\nfull per-layer JSON -> %s" % out)

if __name__ == "__main__":
    main()
