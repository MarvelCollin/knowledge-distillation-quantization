"""Check that a saved checkpoint really sits on a low-bit grid.

Fake-quant checkpoints are stored as bf16, so the only way to tell a quantized
checkpoint from an unquantized one is to count distinct values inside a single
quantization group: group-128 W4 admits at most 16, while unquantized weights
show close to 128. Reads the safetensors header directly, so it needs no torch.

Usage: python3 scripts/probe_quant_grid.py <ckpt-dir> [<ckpt-dir> ...]
"""
import json
import os
import struct
import sys

ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4}


def probe(path: str, rows: int = 6, group: int = 128) -> list:
    idx = os.path.join(path, "model.safetensors.index.json")
    name = None
    if os.path.exists(idx):
        weight_map = json.load(open(idx))["weight_map"]
        name = next(k for k in weight_map if k.endswith("self_attn.q_proj.weight"))
        f = os.path.join(path, weight_map[name])
    else:
        f = os.path.join(path, "model.safetensors")
    with open(f, "rb") as fh:
        hdr_len = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hdr_len))
        base = 8 + hdr_len
        if name is None:
            name = next(k for k in hdr if k.endswith("self_attn.q_proj.weight"))
        meta = hdr[name]
        size = ITEMSIZE[meta["dtype"]]
        in_f = meta["shape"][1]
        start = meta["data_offsets"][0]
        counts = []
        for r in range(rows):
            fh.seek(base + start + r * in_f * size)
            raw = fh.read(group * size)
            counts.append(len({raw[i:i + size] for i in range(0, len(raw), size)}))
    return meta["dtype"], meta["shape"], counts


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    print(f"group-128 W4 => at most 16 distinct values per group; unquantized => near 128\n")
    for path in sys.argv[1:]:
        try:
            dtype, shape, counts = probe(path)
            verdict = "QUANTIZED" if max(counts) <= 16 else "NOT quantized"
            print(f"{path:<40} {dtype} {shape}  uniq/group={counts}  -> {verdict}")
        except Exception as exc:
            print(f"{path:<40} ERROR {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
