import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import gc
import json
import shutil
import time

import torch
import yaml
from dotenv import load_dotenv

from src.data.dataset import PROMPT_TEMPLATE, load_problems
from src.evaluation.evaluator import extract_signature, run_test_cases
from src.student.model import StudentModel
from src.teacher.local_teacher import LocalTeacherModel
from src.utils.reasoning import (
    SYSTEM_PROMPT,
    build_signature_user_content,
    extract_code,
    extract_fn_name,
    generate_with_thinking_cap,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def is_passing(entry: dict) -> bool:
    tp, tt = entry.get("test_passed"), entry.get("test_total")
    return bool(tt) and tp == tt


def tokenize_pieces(tokenizer, text: str) -> list:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [tokenizer.decode([i]) for i in ids]


def teacher_payload(prompt: str, res: dict, r: dict, max_tokens: int) -> dict:
    return {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "source": "rft_teacher",
        "text": res["text"],
        "tokens": res["tokens"],
        "logprobs": res.get("logprobs", []),
        "top_k_ids": res.get("top_k_ids"),
        "top_k_vals": res.get("top_k_vals"),
        "test_passed": r["passed"],
        "test_total": r["total"],
    }


def student_payload(tokenizer, prompt: str, code: str, r: dict, max_tokens: int) -> dict:
    text = f"```python\n{code}\n```"
    return {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "source": "rft_student",
        "text": text,
        "tokens": tokenize_pieces(tokenizer, text),
        "logprobs": [],
        "top_k_ids": None,
        "top_k_vals": None,
        "test_passed": r["passed"],
        "test_total": r["total"],
    }


def sample_student_code(student, prompt: str, test_cases: list,
                        max_new_tokens: int, temperature: float, top_p: float) -> str:
    expected = extract_fn_name(test_cases)
    signature = extract_signature(test_cases[0], expected) if test_cases else ""
    user_content = build_signature_user_content(prompt, signature)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    fmt = student.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    raw, _ = generate_with_thinking_cap(
        student.model, student.tokenizer, fmt, max_new_tokens,
        code_primer_signature=signature,
        do_sample=True, temperature=temperature, top_p=top_p,
    )
    return extract_code(raw)


def persist(out_cache: str, idx: int, payload: dict) -> None:
    with open(Path(out_cache) / f"{idx}.json", "w") as f:
        json.dump(payload, f)


def first_passing(candidates: list, test_cases: list):
    for res in candidates:
        code = extract_code(res["text"])
        r = run_test_cases(code, test_cases)
        if r["total"] > 0 and r["passed"] == r["total"]:
            return res, r
    return None


def seed_combined_cache(problems: list, src_cache: str, out_cache: str) -> list:
    src = Path(src_cache)
    out = Path(out_cache)
    out.mkdir(parents=True, exist_ok=True)

    copied = 0
    gaps = []
    for i in range(len(problems)):
        sf = src / f"{i}.json"
        if sf.exists() and is_passing(json.loads(sf.read_text())):
            df = out / f"{i}.json"
            if not df.exists():
                shutil.copyfile(sf, df)
                copied += 1
            continue
        df = out / f"{i}.json"
        if df.exists() and is_passing(json.loads(df.read_text())):
            continue
        gaps.append(i)

    print(f"  Seeded combined cache: {copied} passing teacher entries copied.")
    print(f"  Gap problems to fill : {len(gaps)} (teacher failed/missing).")
    return gaps


def run_teacher_phase(config, gaps, prompts, tests, args, out_cache) -> dict:
    from transformers import AutoTokenizer

    solved = {}
    student_tokenizer = AutoTokenizer.from_pretrained(
        config["student"]["model_name"], trust_remote_code=True
    )
    teacher = LocalTeacherModel(
        model_path=config["teacher"]["local_model_path"],
        max_tokens=config["teacher"]["max_tokens"],
        temperature=config["teacher"]["temperature"],
        top_p=config["teacher"].get("top_p", 0.95),
        top_logprobs=config["teacher"]["top_logprobs"],
        student_tokenizer=student_tokenizer,
        gpu_memory_utilization=config["teacher"].get("gpu_memory_utilization", 0.85),
        max_model_len=config["teacher"].get("max_model_len", 10240),
    )

    chunk = args.chunk_size
    t0 = time.time()
    for start in range(0, len(gaps), chunk):
        idxs = gaps[start:start + chunk]
        batch_prompts = [prompts[i] for i in idxs]
        print(f"\n[teacher {start + len(idxs)}/{len(gaps)}] sampling "
              f"{args.teacher_samples} candidates x {len(idxs)} prompts...")
        results = teacher.sample_candidates_batch(
            batch_prompts, n=args.teacher_samples,
            temperature=args.temperature, top_p=args.top_p,
        )
        for i, cands in zip(idxs, results):
            hit = first_passing(cands, tests[i])
            if hit is not None:
                res, r = hit
                solved[i] = teacher_payload(prompts[i], res, r, config["teacher"]["max_tokens"])
                persist(out_cache, i, solved[i])
        done = start + len(idxs)
        print(f"  teacher solved {len(solved)}/{done} gaps so far "
              f"({time.time() - t0:.0f}s elapsed).")

    teacher.shutdown()
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    return solved


def run_student_phase(config, remaining, prompts, tests, args, device, out_cache) -> dict:
    solved = {}
    student = StudentModel(
        model_name=args.checkpoint,
        max_length=config["student"]["max_length"],
    )
    student.to(device)
    student.eval()
    tok = student.tokenizer

    t0 = time.time()
    for n, i in enumerate(remaining, 1):
        for _ in range(args.student_samples):
            code = sample_student_code(
                student, prompts[i], tests[i],
                args.max_new_tokens, args.temperature, args.top_p,
            )
            r = run_test_cases(code, tests[i])
            if r["total"] > 0 and r["passed"] == r["total"]:
                solved[i] = student_payload(tok, prompts[i], code, r,
                                            config["student"]["max_length"])
                persist(out_cache, i, solved[i])
                break
        if n % 10 == 0 or n == len(remaining):
            print(f"  [student {n}/{len(remaining)}] solved {len(solved)} "
                  f"({time.time() - t0:.0f}s elapsed).")

    del student
    gc.collect()
    torch.cuda.empty_cache()
    return solved


def main():
    parser = argparse.ArgumentParser(description="RFT gap-filling (Tahap B)")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", default="outputs/final",
                        help="Student checkpoint to sample candidates from.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Problems to consider (defaults to config data.max_samples).")
    parser.add_argument("--source-cache", default=None,
                        help="Teacher cache to read passing entries / gaps from "
                             "(defaults to config data.teacher_cache_dir).")
    parser.add_argument("--out-cache", default="cache/combined_r1_7b",
                        help="Combined drop-in cache to write (point training at this).")
    parser.add_argument("--teacher-samples", type=int, default=4)
    parser.add_argument("--student-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Student generation budget per candidate.")
    parser.add_argument("--chunk-size", type=int, default=16,
                        help="Teacher prompts per vLLM batch.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of gap problems (smoke test).")
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--skip-student", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    if args.max_samples is not None:
        config["data"]["max_samples"] = args.max_samples

    src_cache = args.source_cache or config["data"]["teacher_cache_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("═" * 60)
    print("  RFT gap-filling (Tahap B)")
    print("═" * 60)
    print(f"  source cache : {src_cache}")
    print(f"  out cache    : {args.out_cache}")
    print(f"  checkpoint   : {args.checkpoint}")
    print(f"  samples      : teacher={args.teacher_samples}  student={args.student_samples}"
          f"  temp={args.temperature}  top_p={args.top_p}")
    print("═" * 60)

    problems = load_problems(config)
    gaps = seed_combined_cache(problems, src_cache, args.out_cache)
    if args.limit is not None:
        gaps = gaps[:args.limit]
        print(f"  --limit: processing first {len(gaps)} gaps only.")
    if not gaps:
        print("  No gaps to fill — combined cache is complete.")
        return

    prompts = {i: PROMPT_TEMPLATE.format(text=problems[i]["text"]) for i in gaps}
    tests = {i: problems[i]["test_cases"] for i in gaps}

    by_teacher = {}
    if not args.skip_teacher:
        by_teacher = run_teacher_phase(config, gaps, prompts, tests, args, args.out_cache)
        print(f"\n  Teacher phase: filled {len(by_teacher)}/{len(gaps)} gaps. "
              f"Freed vLLM, loading student...\n")

    remaining = [i for i in gaps if i not in by_teacher]
    by_student = {}
    if not args.skip_student and remaining:
        by_student = run_student_phase(config, remaining, prompts, tests, args, device, args.out_cache)

    solved = {**by_teacher, **by_student}
    total_combined = len(list(Path(args.out_cache).glob("*.json")))
    print()
    print("═" * 60)
    print("  RFT summary")
    print("═" * 60)
    print(f"  Gaps processed        : {len(gaps)}")
    print(f"  Filled by teacher     : {len(by_teacher)}")
    print(f"  Filled by student     : {len(by_student)}")
    print(f"  Still unsolved        : {len(gaps) - len(solved)}")
    print(f"  Combined cache size   : {total_combined} problems "
          f"({args.out_cache})")
    print("═" * 60)
    print(f"\n  Next: train on the combined cache by setting")
    print(f"        data.teacher_cache_dir: {args.out_cache}   in config/config.yaml")
    print(f"        (or keep both dirs for an SFT vs KD vs KD+RFT ablation).")


if __name__ == "__main__":
    main()
