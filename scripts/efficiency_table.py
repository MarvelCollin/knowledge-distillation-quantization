import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time

PROMPTS = [
    "Write a Python function that returns the nth Fibonacci number using dynamic programming.",
    "Write a Python function that checks whether a given string is a valid palindrome, "
    "ignoring non-alphanumeric characters.",
    "Write a Python function that finds the longest common subsequence of two strings.",
    "Write a Python function that merges two sorted linked lists into one sorted linked list.",
] * 6

def dir_size_gb(path):
    p = Path(path)
    if p.is_file():
        return p.stat().st_size / 1024 ** 3
    total = sum(f.stat().st_size for f in p.rglob("*.safetensors"))
    return total / 1024 ** 3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    disk_gb = dir_size_gb(args.model)

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.85, trust_remote_code=True)

    params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=args.max_new_tokens)
    t0 = time.time()
    outputs = llm.generate(PROMPTS, params, use_tqdm=False)
    elapsed = time.time() - t0

    n_out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    toks_per_sec = n_out_tokens / elapsed

    print()
    print(f"=== efficiency: {args.tag} ({args.model}) ===")
    print(f"  on-disk safetensors size (bf16) : {disk_gb:.2f} GB")
    print(f"  weights VRAM                    : see 'model weights take X GiB' in the vLLM log above")
    print(f"  batch: {len(PROMPTS)} prompts, {n_out_tokens} output tokens in {elapsed:.1f}s")
    print(f"  aggregate throughput             : {toks_per_sec:.0f} tok/s")
    print()
    print(f"  theoretical INT8 size (bit-width only) : {disk_gb / 2:.2f} GB")
    print(f"  theoretical W4 size (bit-width only)   : {disk_gb / 4:.2f} GB")
    print("  (real compressed-checkpoint throughput/VRAM: blocked by the vLLM 0.6.6 Marlin")
    print("   runtime defect on this stack -- declared limitation, not measured here)")

if __name__ == "__main__":
    main()
