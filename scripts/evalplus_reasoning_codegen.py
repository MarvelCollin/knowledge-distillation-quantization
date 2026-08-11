import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer

from src.evaluation.generation import build_eval_prompts, budget_forced_generate
from src.utils.reasoning import extract_code

def _signature_line(prompt: str, entry_point: str) -> str:
    try:
        tree = ast.parse(prompt)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            node.body = [ast.Pass()]
            return ast.unparse(node).split("\n")[0]
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=["humaneval", "mbpp"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--think-ratio", type=float, default=0.5)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "auto"])
    ap.add_argument("--limit", type=int, default=None,
                     help="Only run the first N problems (smoke testing).")
    args = ap.parse_args()

    if args.dataset == "humaneval":
        from evalplus.data import get_human_eval_plus
        tasks = get_human_eval_plus()
    else:
        from evalplus.data import get_mbpp_plus
        tasks = get_mbpp_plus()

    task_ids = list(tasks)
    if args.limit:
        task_ids = task_ids[:args.limit]
    problems = [
        {
            "text": tasks[tid]["prompt"],
            "entry_point": tasks[tid]["entry_point"],
            "signature": _signature_line(tasks[tid]["prompt"], tasks[tid]["entry_point"]),
        }
        for tid in task_ids
    ]
    n_no_sig = sum(1 for p in problems if not p["signature"])

    print(f"loaded {len(task_ids)} {args.dataset} problems ({n_no_sig} without a parsed signature)")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts, signatures = build_eval_prompts(problems, tokenizer)

    from vllm import LLM
    llm = LLM(model=args.model, dtype=args.dtype, max_model_len=4096,
              gpu_memory_utilization=0.9, trust_remote_code=True)

    grid = budget_forced_generate(
        llm, prompts, signatures, num_samples=1,
        temperature=0.0, top_p=1.0, max_new_tokens=args.max_new_tokens,
        think_ratio=args.think_ratio, seed=1234,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_empty = 0
    n_truncated = 0
    with out.open("w") as f:
        for tid, (text, truncated) in zip(task_ids, (g[0] for g in grid)):
            code = extract_code(text)
            if not code.strip():
                n_empty += 1
            if truncated:
                n_truncated += 1
            f.write(json.dumps({"task_id": tid, "solution": code}) + "\n")

    print(f"wrote {len(task_ids)} samples to {out}")
    print(f"empty extractions: {n_empty}/{len(task_ids)}  truncated: {n_truncated}/{len(task_ids)}")

if __name__ == "__main__":
    main()
