import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer
from vllm import SamplingParams

from src.config import load_config
from src.data.problems import build_user_content, load_problems
from src.data.teacher_cache import failed_cache_indices
from src.evaluation.evaluator import run_test_cases
from src.teacher.local_teacher import LocalTeacherModel
from src.utils.reasoning import build_retry_prompt, extract_code


def main():
    load_dotenv()
    config = load_config("config/config.yaml")
    cache_dir = config["data"]["teacher_cache_dir"]
    problems = load_problems(config)

    failed = failed_cache_indices(cache_dir, len(problems))
    if not failed:
        print("SKIP: no failed cache entries to test with")
        return

    smoke_n = min(4, len(failed))
    smoke_indices = failed[:smoke_n]
    attempts = 3
    max_tokens = 8192
    retry_model_len = min(
        config["teacher"].get("max_model_len", 10240),
        max_tokens + 4096,
    )

    print(f"=== Smoke test: retry {smoke_n} problems, {attempts} candidates each ===")
    print(f"  max_tokens     = {max_tokens}")
    print(f"  max_model_len  = {retry_model_len}")
    print(f"  failed total   = {len(failed)}")
    print()

    student_tokenizer = AutoTokenizer.from_pretrained(
        config["student"]["model_name"], trust_remote_code=True
    )
    t_load = time.time()
    teacher = LocalTeacherModel.from_config(
        config, student_tokenizer,
        temperature=1.0,
        top_p=0.95,
        max_model_len=retry_model_len,
    )
    load_secs = time.time() - t_load
    print(f"  model loaded in {load_secs:.1f}s")

    retry_params = SamplingParams(
        n=attempts,
        temperature=1.0,
        top_p=0.95,
        max_tokens=max_tokens,
        logprobs=config["teacher"]["top_logprobs"],
        repetition_penalty=1.05,
    )

    prompts = [build_retry_prompt(problems[i], teacher.tokenizer) for i in smoke_indices]
    prompt_lens = [len(teacher.tokenizer.encode(p)) for p in prompts]
    print(f"  prompt token lengths: {prompt_lens}")

    t_gen = time.time()
    outputs = teacher.llm.generate(prompts, retry_params, use_tqdm=True)
    gen_secs = time.time() - t_gen

    print(f"\n  generation done in {gen_secs:.1f}s  ({gen_secs / smoke_n:.1f}s/problem)")

    recovered = 0
    for j, idx in enumerate(smoke_indices):
        diff = problems[idx].get("difficulty", "?")
        tcs = problems[idx]["test_cases"]
        orig_prompt = build_user_content(problems[idx])
        best_score = None
        best_result = None

        for gen in outputs[j].outputs:
            result = teacher._extract_one(gen)
            out_tokens = len(gen.token_ids)
            code = extract_code(result.get("text", ""))
            score = run_test_cases(code, tcs, timeout=30.0)
            passed = score["passed"] == score["total"] and score["total"] > 0
            status = "PASS" if passed else "FAIL"
            print(f"  [{idx}] {diff:<8} candidate: {out_tokens:>5} tokens  {status}  "
                  f"({score['passed']}/{score['total']} tests)")
            if passed and best_result is None:
                best_result = result
                best_score = score

        if best_result is not None:
            recovered += 1
            print(f"  [{idx}] -> RECOVERED")
        else:
            print(f"  [{idx}] -> still failed")

    print(f"\n=== Results ===")
    print(f"  recovered    : {recovered}/{smoke_n}")
    print(f"  gen time     : {gen_secs:.1f}s total, {gen_secs / smoke_n:.1f}s/problem")
    print(f"  model load   : {load_secs:.1f}s")

    full_eta_min = gen_secs / smoke_n * len(failed) / 60
    print(f"  full run ETA : {full_eta_min:.0f} min for {len(failed)} problems")

    old_chunk_secs = 3828
    old_per_problem = old_chunk_secs / 64
    new_per_problem = gen_secs / smoke_n
    if new_per_problem > 0:
        speedup = old_per_problem / new_per_problem
        print(f"  speedup      : {speedup:.1f}x vs old run ({old_per_problem:.1f}s -> {new_per_problem:.1f}s per problem)")

    teacher.shutdown()
    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
