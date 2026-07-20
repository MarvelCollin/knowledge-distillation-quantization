import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from transformers import AutoTokenizer

from src.config import load_config
from src.data.problems import build_user_content, load_problems
from src.data.short_cot import EDGE_COT_SYSTEM_PROMPT, SHORT_COT_SYSTEM_PROMPT
from src.evaluation.evaluator import run_test_cases
from src.utils.reasoning import extract_code
from src.utils.runlog import RunLog


def passing_texts(problem: dict, candidates: list, max_workers: int) -> list:
    def _passes(text: str) -> bool:
        code = extract_code(text)
        if not code:
            return False
        r = run_test_cases(code, problem["test_cases"], timeout=5.0)
        return r["total"] > 0 and r["passed"] == r["total"]

    unique = list(dict.fromkeys(t for t in candidates if t.strip()))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        keep = list(ex.map(_passes, unique))
    return [t for t, ok in zip(unique, keep) if ok]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--edge", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    sc = config["short_cot"]
    system_prompt = EDGE_COT_SYSTEM_PROMPT if args.edge else SHORT_COT_SYSTEM_PROMPT
    seed_base = sc["seed"] + (1000 if args.edge else 0)
    out_dir = Path(sc["edge_cache_dir"] if args.edge else sc["cache_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(sc["model_name"], trust_remote_code=True)
    problems = load_problems(config)
    indices = list(range(len(problems)))
    if args.limit:
        indices = indices[:args.limit]
    pending = [i for i in indices if not (out_dir / f"{i}.json").exists()]
    print(f"Short-CoT generation ({sc['model_name']}): {len(pending)} pending / "
          f"{len(indices)} problems -> {out_dir}")
    if not pending:
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=sc["model_name"], trust_remote_code=True, dtype="bfloat16",
              max_model_len=sc["max_model_len"],
              gpu_memory_utilization=sc["gpu_memory_utilization"])
    n_samples = sc["samples"]
    sample_params = [
        SamplingParams(temperature=sc["temperature"], top_p=sc["top_p"],
                       max_tokens=sc["max_tokens"], n=1, seed=seed_base + j)
        for j in range(n_samples)
    ]

    runlog = RunLog("logs/short_cot.jsonl")
    runlog.event("gen_start", pending=len(pending), total=len(indices),
                 samples=sc["samples"], model=sc["model_name"])

    test_workers = min(os.cpu_count() or 16, 48)
    chunk = sc["gen_chunk_size"]
    written_texts = 0
    solved_problems = 0
    for start in range(0, len(pending), chunk):
        batch = pending[start:start + chunk]
        prompts = []
        params = []
        for i in batch:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_content(problems[i])},
            ]
            chat = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            prompts.extend([chat] * n_samples)
            params.extend(sample_params)
        outputs = llm.generate(prompts, params)
        for pos, i in enumerate(batch):
            group = outputs[pos * n_samples:(pos + 1) * n_samples]
            texts = passing_texts(
                problems[i], [g.outputs[0].text for g in group], test_workers)
            payload = {
                "prompt": build_user_content(problems[i]),
                "attempts": n_samples,
                "texts": texts,
            }
            (out_dir / f"{i}.json").write_text(json.dumps(payload))
            written_texts += len(texts)
            solved_problems += 1 if texts else 0
        done = min(start + chunk, len(pending))
        print(f"  generated {done}/{len(pending)} pending "
              f"({written_texts} passing texts, {solved_problems} problems with >=1 pass)")
        runlog.event("gen_chunk", done=done, pending=len(pending),
                     passing_texts=written_texts, solved_problems=solved_problems)

    runlog.event("gen_done", passing_texts=written_texts, solved_problems=solved_problems)
    print(f"Short-CoT generation complete: {written_texts} passing texts across "
          f"{solved_problems} problems -> {out_dir}")


if __name__ == "__main__":
    main()
