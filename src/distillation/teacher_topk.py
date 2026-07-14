from pathlib import Path

import torch

from src.data.teacher_cache import load_cached
from src.distillation.loss import build_teacher_topk


def preload_teacher_topk(cache_dir: str, original_indices: list,
                         vocab_size: int, topk: int, distill_temp: float) -> dict:
    newest_json = max((f.stat().st_mtime for f in Path(cache_dir).glob("*.json")), default=0.0)
    snap_path = Path(cache_dir) / f"topk_snapshot_v{vocab_size}_k{topk}_t{distill_temp}.pt"
    teacher_topk = None
    if snap_path.exists() and snap_path.stat().st_mtime >= newest_json:
        snap = torch.load(snap_path)
        if snap.get("indices") == list(original_indices):
            teacher_topk = snap["topk"]
            print(f"Loaded teacher top-k snapshot ({snap_path.name}).")
    if teacher_topk is None:
        teacher_topk = {}
        for cache_idx in original_indices:
            cached = load_cached(cache_dir, cache_idx)
            if not cached or not cached.get("top_k_ids") or not cached.get("top_k_vals"):
                continue
            teacher_topk[cache_idx] = build_teacher_topk(
                cached["top_k_ids"], cached["top_k_vals"],
                vocab_size, topk, torch.device("cpu"), distill_temp,
            )
        torch.save({"indices": list(original_indices), "topk": teacher_topk}, snap_path)
    return teacher_topk


def scatter_teacher_topk(teacher_topk: dict, sample_idxs, prompt_lengths,
                         original_indices: list, seq_len: int, topk: int, device) -> tuple:
    teacher_ids = torch.zeros(len(sample_idxs), seq_len, topk, dtype=torch.long, device=device)
    teacher_probs = torch.zeros(len(sample_idxs), seq_len, topk, device=device)

    for i in range(len(sample_idxs)):
        local_idx = sample_idxs[i].item()
        cache_idx = original_indices[local_idx]
        entry = teacher_topk.get(cache_idx)
        if entry is None:
            continue
        ids_full, probs_full = entry
        pl = prompt_lengths[i].item()
        dist_start = pl - 1
        aligned_len = min(ids_full.size(0), seq_len - dist_start)
        if aligned_len <= 0:
            continue
        teacher_ids[i, dist_start:dist_start + aligned_len] = ids_full[:aligned_len].to(device)
        teacher_probs[i, dist_start:dist_start + aligned_len] = probs_full[:aligned_len].to(device)

    return teacher_ids, teacher_probs
