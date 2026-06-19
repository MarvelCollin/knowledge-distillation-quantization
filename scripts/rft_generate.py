import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import gc
import shutil
import time

import torch
from dotenv import load_dotenv

from src.config import load_config
from src.data.dataset import PROMPT_TEMPLATE, load_problems
from src.distillation.rft import (
    assemble_solution_text,
    first_passing,
    is_passing,
    kd_payload,
    persist,
    read_entry,
    sft_payload,
    solves,
)
from src.evaluation.generation import generate_student_solution
from src.student.model import StudentModel
from src.teacher.local_teacher import LocalTeacherModel


def seed_combined_cache(problems, src_cache, out_cache):
    src = Path(src_cache)
    out = Path(out_cache)
    out.mkdir(parents=True, exist_ok=True)

    copied = 0
    gaps = []
    for i in range(len(problems)):
        sf = src / f"{i}.json"
        se = read_entry(sf) if sf.exists() else None
        df = out / f"{i}.json"
        if se is not None and is_passing(se):
            if not df.exists():
                shutil.copyfile(sf, df)
                copied += 1
            continue
        de = read_entry(df) if df.exists() else None
        if de is not None and is_passing(de):
            continue
        gaps.append(i)

    print(f"  Seeded combined cache: {copied} passing teacher entries copied.")
    print(f"  Gap problems to fill : {len(gaps)} (teacher failed/missing).")
    return gaps


def run_teacher_phase(config, gaps, prompts, tests, args, out_cache):
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
            res = first_passing(cands, tests[i])
            if res is not None:
                solved[i] = kd_payload(prompts[i], res, len(tests[i]), config["teacher"]["max_tokens"])
                persist(out_cache, i, solved[i])
        done = start + len(idxs)
        print(f"  teacher solved {len(solved)}/{done} gaps so far "
              f"({time.time() - t0:.0f}s elapsed).")

    teacher.shutdown()
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    return solved


def run_student_phase(config, remaining, prompts, tests, args, device, out_cache):
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
            raw, code = generate_student_solution(
                student, prompts[i], tests[i], args.max_new_tokens,
                do_sample=True, temperature=args.temperature, top_p=args.top_p,
            )
            if solves(code, tests[i]):
                text = assemble_solution_text(raw, code)
                solved[i] = sft_payload(tok, prompts[i], text, len(tests[i]),
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
    parser.add_argument("--checkpoint", default="outputs/final")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--source-cache", default=None)
    parser.add_argument("--out-cache", default="cache/combined_r1_7b")
    parser.add_argument("--teacher-samples", type=int, default=4)
    parser.add_argument("--student-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--skip-student", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    if args.max_samples is not None:
        config["data"]["max_samples"] = args.max_samples

    src_cache = args.source_cache or config["data"]["teacher_cache_dir"]
    if Path(args.out_cache).resolve() == Path(src_cache).resolve():
        raise SystemExit("  ERROR: --out-cache must differ from the source teacher cache.")
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
    print(f"  Combined cache size   : {total_combined} problems ({args.out_cache})")
    print("═" * 60)
    print(f"\n  Train on it:")
    print(f"    docker compose run --rm train python scripts/train.py --offline "
          f"--cache-dir {args.out_cache} --max-samples {len(problems)}")


if __name__ == "__main__":
    main()
