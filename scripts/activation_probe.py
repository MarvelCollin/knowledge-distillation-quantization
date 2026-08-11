import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json

from src.data.teacher_cache import fully_passed, tail_counts

OUT_DIR = Path("outputs_activation_probe")
CACHE_DIR = Path("cache/teacher_logprobs_r1_full")
TOP_K = 20
MAX_LEN = 4096

def _load_fixed_sequences(n: int):
    files = sorted(CACHE_DIR.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 1 << 30)
    seqs = []
    for f in files:
        tp, tt = tail_counts(f, 400)
        if not fully_passed(tp, tt):
            continue
        d = json.loads(f.read_text())
        if d.get("prompt") and d.get("text"):
            seqs.append((int(f.stem), d["prompt"], d["text"]))
        if len(seqs) >= n:
            break
    return seqs

def score(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    seqs = _load_fixed_sequences(args.n)
    print(f"loaded {len(seqs)} fixed passing R1 sequences from {CACHE_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, dtype=args.dtype, max_model_len=MAX_LEN,
              gpu_memory_utilization=0.70, trust_remote_code=True,
              enforce_eager=True, enable_chunked_prefill=True,
              max_num_batched_tokens=1024, enable_prefix_caching=False)

    params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1, prompt_logprobs=TOP_K)
    cap = MAX_LEN - 1

    token_prompts = []
    metas = []
    for idx, prompt, text in seqs:
        p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        r_ids = tokenizer(text, add_special_tokens=False).input_ids
        full_ids = (p_ids + r_ids)[:cap]
        if len(full_ids) <= len(p_ids):
            continue
        token_prompts.append({"prompt_token_ids": full_ids})
        metas.append((idx, len(p_ids), len(full_ids)))

    outputs = llm.generate(token_prompts, params, use_tqdm=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for (idx, prompt_len, total_len), out in zip(metas, outputs):
        positions = []
        for pos in range(prompt_len, total_len):
            row = out.prompt_logprobs[pos]
            if not row:
                continue
            topk = {tid: lp.logprob for tid, lp in row.items()}
            positions.append({"token_id": out.prompt_token_ids[pos], "topk": topk})
        saved.append({"idx": idx, "positions": positions})

    out_path = OUT_DIR / f"{args.tag}.json"
    out_path.write_text(json.dumps(saved))
    n_pos = sum(len(s["positions"]) for s in saved)
    print(f"scored {len(saved)} sequences, {n_pos} completion-token positions -> {out_path}")

def compare(tag_a, tag_b):
    a = json.loads((OUT_DIR / f"{tag_a}.json").read_text())
    b = json.loads((OUT_DIR / f"{tag_b}.json").read_text())
    b_by_idx = {s["idx"]: s["positions"] for s in b}

    agree = 0
    fell_out = 0
    n = 0
    for sa in a:
        pb = b_by_idx.get(sa["idx"])
        if pb is None or len(pb) != len(sa["positions"]):
            continue
        for pa_pos, pb_pos in zip(sa["positions"], pb):
            top1_a = max(pa_pos["topk"], key=pa_pos["topk"].get)
            top1_b = max(pb_pos["topk"], key=pb_pos["topk"].get)
            n += 1
            if top1_a == top1_b:
                agree += 1
            if top1_a not in pb_pos["topk"]:
                fell_out += 1

    print(f"{tag_a} vs {tag_b}: {n} matched positions")
    print(f"  top-1 agreement                                : {agree}/{n} ({agree / max(n,1):.1%})")
    print(f"  {tag_a} top-1 token missing from {tag_b} top-{TOP_K}   : {fell_out}/{n} ({fell_out / max(n,1):.1%})")

def _seq_agreement(pos_a, pos_b):
    n = len(pos_a)
    if n == 0:
        return 0.0
    agree = sum(
        1 for x, y in zip(pos_a, pos_b)
        if max(x["topk"], key=x["topk"].get) == max(y["topk"], key=y["topk"].get)
    )
    return agree / n

def compare_gap(dist_bf, orig_bf, dist_w4, orig_w4):
    import random
    random.seed(12345)

    def load(tag):
        return {s["idx"]: s["positions"] for s in json.loads((OUT_DIR / f"{tag}.json").read_text())}

    A_bf, B_bf, A_w4, B_w4 = load(dist_bf), load(orig_bf), load(dist_w4), load(orig_w4)
    idxs = sorted(set(A_bf) & set(B_bf) & set(A_w4) & set(B_w4))

    deltas = []
    for idx in idxs:
        pa_bf, pb_bf, pa_w4, pb_w4 = A_bf[idx], B_bf[idx], A_w4[idx], B_w4[idx]
        if len(pa_bf) != len(pb_bf) or len(pa_w4) != len(pb_w4):
            continue
        deltas.append(_seq_agreement(pa_w4, pb_w4) - _seq_agreement(pa_bf, pb_bf))

    n = len(deltas)
    mean_delta = sum(deltas) / n
    iters = 20000
    boots = sorted(sum(deltas[random.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    lo, hi = boots[int(0.025 * iters)], boots[int(0.975 * iters)]
    n_pos = sum(1 for d in deltas if d > 0)

    print(f"Gap-convergence test over {n} independent sequences "
          f"({dist_bf}/{orig_bf} agreement-rate -> {dist_w4}/{orig_w4} agreement-rate delta):")
    print(f"  mean delta (w4_agree - bf16_agree): {mean_delta:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  sequences where dist-orig agreement increased under W4: {n_pos}/{n} ({n_pos / n:.1%})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--tag")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "auto"])
    ap.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"))
    ap.add_argument("--compare-gap", nargs=4,
                     metavar=("DIST_BF16", "ORIG_BF16", "DIST_W4", "ORIG_W4"))
    args = ap.parse_args()

    if args.compare_gap:
        compare_gap(*args.compare_gap)
        return
    if args.compare:
        compare(*args.compare)
        return
    if not args.model or not args.tag:
        ap.error("--model and --tag are required unless --compare/--compare-gap is given")
    score(args)

if __name__ == "__main__":
    main()
