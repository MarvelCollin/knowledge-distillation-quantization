import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.reasoning import SYSTEM_PROMPT

PROMPTS = [
    "Write a Python function two_sum(nums, target) that returns the indices of the two "
    "numbers adding up to target.",
    "Write a Python function is_palindrome(s) that returns True if the string reads the same "
    "forwards and backwards, ignoring case and non-alphanumeric characters.",
    "Write a Python function merge_intervals(intervals) that merges all overlapping intervals "
    "and returns the merged list sorted by start.",
]

SHINGLE = 24

def distinct_ratio(text: str) -> float:
    if len(text) < SHINGLE * 2:
        return 1.0
    shingles = [text[i:i + SHINGLE] for i in range(len(text) - SHINGLE)]
    return len(set(shingles)) / len(shingles)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16"],
                    help="auto lets vLLM pick from the checkpoint config. compare_eval.py "
                         "hardcodes bfloat16 -- test auto and float16 on quantized checkpoints.")
    ap.add_argument("--quantization", default=None,
                    help="Force a vLLM quantization backend (e.g. gptq_marlin, awq_marlin, "
                         "compressed-tensors). Omit to let vLLM infer from the config.")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--chars", type=int, default=300, help="Characters printed per sample.")
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="Distinct-shingle ratio below which a sample is called degenerate.")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in PROMPTS
    ]

    kwargs = dict(model=args.model, dtype=args.dtype, max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_mem, trust_remote_code=True)
    if args.quantization:
        kwargs["quantization"] = args.quantization

    print(f"model={args.model}  dtype={args.dtype}  quantization={args.quantization or 'inferred'}")
    llm = LLM(**kwargs)

    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
                        use_tqdm=False)

    bad = 0
    for i, out in enumerate(outs):
        text = out.outputs[0].text
        ratio = distinct_ratio(text)
        verdict = "OK" if ratio >= args.threshold else "DEGENERATE"
        bad += verdict == "DEGENERATE"
        print(f"\n--- sample {i + 1}  distinct={ratio:.3f}  {verdict} ---")
        print(text[:args.chars].replace("\n", "\\n"))

    print(f"\n=== {len(outs) - bad}/{len(outs)} coherent ===")
    if bad:
        print("FAIL: this checkpoint is not usable. That is a tooling bug, not a result.")
        print("Try --dtype float16, or --quantization <backend>, before blaming the checkpoint.")
        sys.exit(1)
    print("PASS: safe to evaluate.")

if __name__ == "__main__":
    main()
