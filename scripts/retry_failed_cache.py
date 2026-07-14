import gc
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Thread

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer

from src.config import load_config
from src.data.problems import DIFFICULTY_ORDER, build_user_content, load_problems
from src.data.teacher_cache import failed_cache_indices, rescore_failed_cache
from src.evaluation.evaluator import run_test_cases
from src.teacher.local_teacher import LocalTeacherModel
from src.utils.reasoning import build_retry_prompt, extract_code
from src.utils.runlog import eta_str, show_progress_bars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--attempts", type=int, default=3,
                        help="Candidates per failed problem (generated in one vLLM call via n=attempts).")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--no-rescore", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    cache_dir = config["data"]["teacher_cache_dir"]
    problems = load_problems(config)

    if not args.no_rescore:
        print("Step 1: Rescore failed entries (CPU, no GPU needed)...")
        t0 = time.time()
        recovered, tested, total_failed = rescore_failed_cache(
            cache_dir, problems, apply=True, timeout=30.0)
        print(f"  Rescore done in {time.time() - t0:.0f}s: "
              f"recovered {recovered}/{total_failed} entries.\n")

    failed = failed_cache_indices(cache_dir, len(problems))
    if not failed:
        print("No failed cache entries. Nothing to retry.")
        return

    retry_model_len = min(
        config["teacher"].get("max_model_len", 10240),
        args.max_tokens + 4096,
    )

    print(f"Step 2: Retry {len(failed)} failed entries "
          f"({args.attempts} candidates each, chunk_size={args.chunk_size}, "
          f"max_tokens={args.max_tokens}, max_model_len={retry_model_len})")

    student_tokenizer = AutoTokenizer.from_pretrained(
        config["student"]["model_name"], trust_remote_code=True
    )
    teacher = LocalTeacherModel.from_config(
        config, student_tokenizer,
        temperature=args.temperature,
        top_p=args.top_p,
        max_model_len=retry_model_len,
    )

    from vllm import SamplingParams
    retry_params = SamplingParams(
        n=args.attempts,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        logprobs=config["teacher"]["top_logprobs"],
        repetition_penalty=1.05,
    )

    cache_path = Path(cache_dir)
    formatted_prompts = [build_retry_prompt(problems[i], teacher.tokenizer) for i in failed]
    original_prompts = [build_user_content(problems[i]) for i in failed]
    tests = [problems[i]["test_cases"] for i in failed]
    diffs = [problems[i].get("difficulty", "?") for i in failed]

    recovered = 0
    still_failed = 0
    diff_recovered = Counter()
    diff_still = Counter()
    total_start = time.time()
    test_workers = min(os.cpu_count() or 16, 64)

    bg_thread = None
    bg_error = [None]

    def _test_and_save(chunk_indices, chunk_diffs, chunk_orig, chunk_tests,
                       outputs, chunk_idx, total_chunks, chunk_t0):
        nonlocal recovered, still_failed
        try:
            all_tasks = []
            for j, vllm_out in enumerate(outputs):
                for k, gen in enumerate(vllm_out.outputs):
                    result = teacher._extract_one(gen)
                    all_tasks.append((j, k, result, chunk_tests[j]))

            def _score(task):
                _, _, result, tcs = task
                code = extract_code(result.get("text", ""))
                return run_test_cases(code, tcs, timeout=30.0)

            with ThreadPoolExecutor(max_workers=test_workers) as ex:
                all_scores = list(ex.map(_score, all_tasks))

            by_problem = {}
            for task, score in zip(all_tasks, all_scores):
                j, k, result, _ = task
                by_problem.setdefault(j, []).append((k, result, score))

            chunk_recovered = 0
            chunk_diff_r = Counter()
            chunk_diff_s = Counter()
            for j in range(len(chunk_indices)):
                idx = chunk_indices[j]
                diff = chunk_diffs[j]
                candidates = sorted(by_problem.get(j, []), key=lambda x: x[0])

                found = False
                for _, result, score in candidates:
                    if score["total"] and score["total"] > 0 and score["passed"] == score["total"]:
                        payload = {
                            "prompt": chunk_orig[j],
                            "max_tokens": teacher.max_tokens,
                            **result,
                            "test_passed": score["total"],
                            "test_total": score["total"],
                        }
                        with open(cache_path / f"{idx}.json", "w") as f:
                            json.dump(payload, f)
                        chunk_recovered += 1
                        diff_recovered[diff] += 1
                        chunk_diff_r[diff] += 1
                        found = True
                        break

                if not found:
                    diff_still[diff] += 1
                    chunk_diff_s[diff] += 1

            recovered += chunk_recovered
            still_failed += len(chunk_indices) - chunk_recovered

            elapsed = time.time() - chunk_t0
            total_elapsed = time.time() - total_start
            done = min(chunk_idx * args.chunk_size, len(failed))
            eta_sec = total_elapsed / max(done, 1) * (len(failed) - done)

            print(
                f"  chunk {chunk_idx}/{total_chunks} done in {elapsed:.0f}s  "
                f"|  recovered {chunk_recovered}/{len(chunk_indices)} "
                f"({args.attempts} candidates each)  "
                f"|  total {recovered}/{recovered + still_failed}  "
                f"|  ETA {eta_str(eta_sec)}"
            )
            for d in sorted(set(list(chunk_diff_r) + list(chunk_diff_s)),
                           key=lambda d: DIFFICULTY_ORDER.get(d, 3)):
                cr = chunk_diff_r.get(d, 0)
                cs = chunk_diff_s.get(d, 0)
                tr = diff_recovered.get(d, 0)
                ts = diff_still.get(d, 0)
                print(f"    {d:<8} {cr}/{cr+cs} recovered  "
                      f"(total {tr}/{tr+ts}  {tr/max(tr+ts,1):.0%})")

        except Exception as e:
            bg_error[0] = e

    for chunk_start in range(0, len(failed), args.chunk_size):
        chunk_end = min(chunk_start + args.chunk_size, len(failed))
        ci = failed[chunk_start:chunk_end]
        cf = formatted_prompts[chunk_start:chunk_end]
        co = original_prompts[chunk_start:chunk_end]
        ct = tests[chunk_start:chunk_end]
        cd = diffs[chunk_start:chunk_end]

        chunk_idx = chunk_start // args.chunk_size + 1
        total_chunks = (len(failed) + args.chunk_size - 1) // args.chunk_size
        chunk_t0 = time.time()
        print(f"\n[chunk {chunk_idx}/{total_chunks}] generating {len(ci)} x {args.attempts} "
              f"candidates via vLLM (shared prefix KV cache)...")

        outputs = teacher.llm.generate(cf, retry_params, use_tqdm=show_progress_bars())

        if bg_thread is not None:
            bg_thread.join()
            if bg_error[0] is not None:
                raise bg_error[0]

        bg_thread = Thread(
            target=_test_and_save,
            args=(ci, cd, co, ct, outputs, chunk_idx, total_chunks, chunk_t0),
        )
        bg_thread.start()

    if bg_thread is not None:
        bg_thread.join()
        if bg_error[0] is not None:
            raise bg_error[0]

    total_min = (time.time() - total_start) / 60
    print(f"\nRetry complete in {total_min:.1f} min: "
          f"recovered {recovered}/{len(failed)} previously-failed entries.")
    print(f"Still failed: {still_failed}")
    print()
    all_diffs = sorted(set(list(diff_recovered) + list(diff_still)), key=lambda d: DIFFICULTY_ORDER.get(d, 3))
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
