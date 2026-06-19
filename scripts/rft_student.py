import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import shutil
import time

from dotenv import load_dotenv
from transformers import AutoTokenizer

from src.config import load_config
from src.data.dataset import PROMPT_TEMPLATE, load_problems
from src.distillation.rft import is_passing, persist, read_entry, sft_payload, solves
from src.evaluation.generation import build_eval_prompts, budget_forced_generate
from src.utils.gpu import gpu_total_gb, gpu_used_gb
from src.utils.reasoning import extract_code


def main():
    parser = argparse.ArgumentParser(
        description="Student-first RFT: sample the student on the train problems with the "
                    "eval-time budget-forced generation, keep the first solution that passes "
                    "all unit tests as an SFT target. Falls back to the teacher's cached "
                    "passing solution only where the student cannot solve the problem."
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--teacher-cache", default=None)
    parser.add_argument("--out-cache", default="cache/rft_student_r1")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=6144)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-teacher-fallback", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    if args.max_samples is not None:
        config["data"]["max_samples"] = args.max_samples

    teacher_cache = args.teacher_cache or config["data"]["teacher_cache_dir"]
    if Path(args.out_cache).resolve() == Path(teacher_cache).resolve():
        raise SystemExit("  ERROR: --out-cache must differ from the teacher cache.")
    out = Path(args.out_cache)
    out.mkdir(parents=True, exist_ok=True)

    problems = load_problems(config)
    if args.limit is not None:
        problems = problems[:args.limit]
    n = len(problems)
    max_length = config["student"]["max_length"]

    print("═" * 62)
    print("  Student-first RFT (rejection-sampling fine-tuning)")
    print("═" * 62)
    print(f"  student      : {args.checkpoint}")
    print(f"  problems     : {n}")
    print(f"  samples/prob : {args.num_samples}  temp={args.temperature}  top_p={args.top_p}")
    print(f"  teacher cache: {teacher_cache}  (fallback={'off' if args.no_teacher_fallback else 'on'})")
    print(f"  out cache    : {args.out_cache}")
    print("═" * 62)

    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_gb = gpu_total_gb()
    gpu_util = max(0.30, min(0.85, (total_gb - gpu_used_gb()) / total_gb - 0.05))
    max_model_len = args.max_new_tokens + 2048
    print(f"  Loading student into vLLM (gpu_util={gpu_util:.2f}, max_model_len={max_model_len})...")
    llm = LLM(
        model=args.checkpoint,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_util,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )

    prompts, signatures = build_eval_prompts(problems, tokenizer)
    print(f"\n  Sampling {n} problems × {args.num_samples} candidates via vLLM...")
    t0 = time.time()
    grid = budget_forced_generate(
        llm, prompts, signatures, num_samples=args.num_samples,
        temperature=args.temperature, top_p=args.top_p,
        max_new_tokens=args.max_new_tokens, seed=args.seed,
    )
    print(f"  Generation done in {(time.time() - t0) / 60:.1f} min. Scoring + writing...")

    raw_prompts = [PROMPT_TEMPLATE.format(text=p["text"]) for p in problems]
    student_solved = 0
    for i, prob in enumerate(problems):
        tests = prob["test_cases"]
        for final_text, _ in grid[i]:
            if solves(extract_code(final_text), tests):
                persist(out, i, sft_payload(tokenizer, raw_prompts[i], final_text, len(tests), max_length))
                student_solved += 1
                break

    print(f"\n  Student solved {student_solved}/{n} ({student_solved / max(n, 1):.1%}).")

    teacher_filled = 0
    if not args.no_teacher_fallback:
        tc = Path(teacher_cache)
        for i in range(n):
            if (out / f"{i}.json").exists():
                continue
            src = read_entry(tc / f"{i}.json")
            if src is not None and is_passing(src):
                shutil.copyfile(tc / f"{i}.json", out / f"{i}.json")
                teacher_filled += 1
        print(f"  Teacher fallback filled {teacher_filled} more (problems the student failed).")

    total = len(list(out.glob("*.json")))
    print()
    print("═" * 62)
    print("  RFT summary")
    print("═" * 62)
    print(f"  Student-own solutions : {student_solved}  (pure SFT, no drift)")
    print(f"  Teacher fallback      : {teacher_filled}  (KD where student failed)")
    print(f"  Combined cache size   : {total}/{n}  → {args.out_cache}")
    print("═" * 62)
    print(f"\n  Train on it:")
    print(f"    docker compose run --rm train python scripts/train.py --offline "
          f"--cache-dir {args.out_cache} --max-samples {n}")


if __name__ == "__main__":
    main()
