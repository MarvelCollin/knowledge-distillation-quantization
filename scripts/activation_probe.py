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
              gpu_memory_utilization=0.9, trust_remote_code=True)

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--tag")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "auto"])
    ap.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"))
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return
    if not args.model or not args.tag:
        ap.error("--model and --tag are required unless --compare is given")
    score(args)

if __name__ == "__main__":
    main()
