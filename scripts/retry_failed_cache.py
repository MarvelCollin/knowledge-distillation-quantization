"""Rejection-sample ONLY the failed teacher cache entries.

Loads the teacher model, finds all failed indices, regenerates solutions
with rejection sampling, tests them, and overwrites the cache file if a
passing solution is found.  Existing passing entries are never touched.

Usage:
    python scripts/retry_failed_cache.py                 # default: 3 attempts per fail
    python scripts/retry_failed_cache.py --attempts 5    # more attempts
"""
import gc
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer

from src.config import load_config
from src.data.dataset import build_user_content, load_problems, _failed_cache_indices
from src.evaluation.evaluator import run_test_cases
from src.teacher.local_teacher import LocalTeacherModel
from src.utils.reasoning import extract_code
from src.utils.runlog import RunLog, eta_str, show_progress_bars

RETRY_SYSTEM_PROMPT = (
    "You are a reasoning coding assistant. First think step by step inside <think>...</think>. "
    "After </think>, output a single standalone Python function inside ```python ... ```. "
    "Do NOT wrap it in a class. Do NOT use class Solution. Output only the function."
)

RETRY_PROMPT_TEMPLATE = (
    "Write a solution in Python to solve the following problem.\n"
    "Your answer must be a standalone Python function (not a class method).\n"
    "Consider edge cases: empty inputs, single elements, negative values, large inputs.\n"
    "Read the constraints carefully.\n\n"
    "Problem: {text}\n\n"
)


def _build_retry_prompt(problem: dict, tokenizer) -> str:
    from src.utils.reasoning import build_signature_user_content
    base = RETRY_PROMPT_TEMPLATE.format(text=problem["text"])
    content = build_signature_user_content(base, problem.get("signature", ""))
    messages = [
        {"role": "system", "content": RETRY_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--attempts", type=int, default=3,
                        help="Max rejection samples per failed problem.")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    cache_dir = config["data"]["teacher_cache_dir"]
    problems = load_problems(config)

    failed = _failed_cache_indices(cache_dir, len(problems))
    if not failed:
        print("No failed cache entries. Nothing to retry.")
        return

    print(f"Found {len(failed)} failed cache entries out of {len(problems)} total.")

    student_tokenizer = AutoTokenizer.from_pretrained(
        config["student"]["model_name"], trust_remote_code=True
    )
    teacher = LocalTeacherModel(
        model_path=config["teacher"]["local_model_path"],
        max_tokens=config["teacher"]["max_tokens"],
        temperature=args.temperature,
        top_p=args.top_p,
        top_logprobs=config["teacher"]["top_logprobs"],
        student_tokenizer=student_tokenizer,
        gpu_memory_utilization=config["teacher"].get("gpu_memory_utilization", 0.85),
        max_model_len=config["teacher"].get("max_model_len", 10240),
        kv_cache_dtype=config["teacher"].get("kv_cache_dtype", "auto"),
        enable_chunked_prefill=config["teacher"].get("enable_chunked_prefill", False),
        max_num_batched_tokens=config["teacher"].get("max_num_batched_tokens", None),
        max_num_seqs=config["teacher"].get("max_num_seqs", None),
    )

    cache_path = Path(cache_dir)
    formatted_prompts = [_build_retry_prompt(problems[i], teacher.tokenizer) for i in failed]
    original_prompts = [build_user_content(problems[i]) for i in failed]
    tests = [problems[i]["test_cases"] for i in failed]
    diffs = [problems[i].get("difficulty", "?") for i in failed]

    recovered = 0
    still_failed = 0
    diff_recovered = Counter()
    diff_still = Counter()
    total_start = time.time()

    def _generate_batch(formatted):
        outputs = teacher.llm.generate(
            formatted, teacher.sampling_params, use_tqdm=show_progress_bars())
        return [teacher._extract_result(o) for o in outputs]

    for chunk_start in range(0, len(failed), args.chunk_size):
        chunk_end = min(chunk_start + args.chunk_size, len(failed))
        chunk_indices = failed[chunk_start:chunk_end]
        chunk_formatted = formatted_prompts[chunk_start:chunk_end]
        chunk_orig = original_prompts[chunk_start:chunk_end]
        chunk_tests = tests[chunk_start:chunk_end]
        chunk_diffs = diffs[chunk_start:chunk_end]

        chunk_idx = chunk_start // args.chunk_size + 1
        total_chunks = (len(failed) + args.chunk_size - 1) // args.chunk_size
        chunk_t0 = time.time()
        print(f"\n[chunk {chunk_idx}/{total_chunks}] retrying {len(chunk_indices)} failed problems "
              f"(up to {args.attempts} attempts each)...")

        chunk_recovered = 0
        still_orig_pos_prev = list(range(len(chunk_indices)))

        for attempt in range(args.attempts):
            to_try = still_orig_pos_prev
            if not to_try:
                break

            results = _generate_batch([chunk_formatted[i] for i in to_try])

            def _score(pair):
                result, tcs = pair
                code = extract_code(result.get("text", ""))
                return run_test_cases(code, tcs, timeout=30.0)

            test_pairs = [(results[j], chunk_tests[to_try[j]]) for j in range(len(to_try))]
            with ThreadPoolExecutor(max_workers=min(16, len(test_pairs))) as ex:
                scores = list(ex.map(_score, test_pairs))

            next_round = []
            for j, (orig_pos, score) in enumerate(zip(to_try, scores)):
                idx = chunk_indices[orig_pos]
                if score["total"] and score["total"] > 0 and score["passed"] == score["total"]:
                    payload = {
                        "prompt": chunk_orig[orig_pos],
                        "max_tokens": teacher.max_tokens,
                        **results[j],
                        "test_passed": score["total"],
                        "test_total": score["total"],
                    }
                    with open(cache_path / f"{idx}.json", "w") as f:
                        json.dump(payload, f)
                    chunk_recovered += 1
                else:
                    next_round.append(orig_pos)

            print(f"    attempt {attempt+1}/{args.attempts}: "
                  f"tried {len(to_try)}, recovered {len(to_try)-len(next_round)}, "
                  f"still failing {len(next_round)}")
            still_orig_pos_prev = next_round

        import re as _re
        for orig_pos in range(len(chunk_indices)):
            diff = chunk_diffs[orig_pos]
            f = cache_path / f"{chunk_indices[orig_pos]}.json"
            try:
                with open(f, "rb") as fh:
                    sz = fh.seek(0, 2)
                    fh.seek(max(0, sz - 400))
                    tail = fh.read()
                mp = _re.search(rb'"test_passed":\s*(\d+)', tail)
                mt = _re.search(rb'"test_total":\s*(\d+)', tail)
                if mp and mt and int(mt.group(1)) > 0 and int(mp.group(1)) == int(mt.group(1)):
                    diff_recovered[diff] += 1
                else:
                    diff_still[diff] += 1
            except Exception:
                diff_still[diff] += 1

        recovered += chunk_recovered
        chunk_still = len(chunk_indices) - chunk_recovered
        still_failed += chunk_still

        elapsed = time.time() - chunk_t0
        total_elapsed = time.time() - total_start
        done = chunk_end
        eta_sec = total_elapsed / done * (len(failed) - done) if done > 0 else 0
        print(
            f"  chunk {chunk_idx}/{total_chunks} done in {elapsed:.0f}s  "
            f"|  recovered {chunk_recovered}/{len(chunk_indices)}  "
            f"|  total recovered {recovered}/{done} so far  "
            f"|  ETA {eta_str(eta_sec)}"
        )

    print(f"\nRetry complete: recovered {recovered}/{len(failed)} previously-failed entries.")
    print(f"Still failed: {still_failed}")
    print()
    order = {"Easy": 0, "Medium": 1, "Hard": 2}
    all_diffs = sorted(set(list(diff_recovered) + list(diff_still)), key=lambda d: order.get(d, 3))
    print(f"  {'Difficulty':<10} {'Recovered':>10} {'Still fail':>10}")
    print(f"  {'─'*34}")
    for d in all_diffs:
        r = diff_recovered.get(d, 0)
        s = diff_still.get(d, 0)
        print(f"  {d:<10} {r:>10} {s:>10}")

    teacher.shutdown()
    del teacher
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
