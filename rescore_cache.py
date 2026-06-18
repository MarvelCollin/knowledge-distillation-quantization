import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import argparse

import yaml
from dotenv import load_dotenv
from transformers import AutoTokenizer

from src.teacher.local_teacher import LocalTeacherModel


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-extract the true top-k teacher distribution from existing cached "
                    "trajectories via a single forward pass (vLLM prompt_logprobs), without "
                    "re-generating. Fixes caches whose logprobs were truncated by top_p/temperature."
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--src", default="cache/teacher_logprobs_r1_7b",
                        help="Source cache dir with existing (passing) trajectories to reuse.")
    parser.add_argument("--dst", default=None,
                        help="Output cache dir (default: data.teacher_cache_dir from config).")
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--gpu-mem", type=float, default=0.70)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--prefill-chunk", type=int, default=1024,
                        help="Chunked-prefill batch size; bounds the prompt_logprobs "
                             "compute memory regardless of sequence length.")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    dst = args.dst or config["data"]["teacher_cache_dir"]

    student_tokenizer = AutoTokenizer.from_pretrained(
        config["student"]["model_name"], trust_remote_code=True
    )

    teacher = LocalTeacherModel(
        model_path=config["teacher"]["local_model_path"],
        max_tokens=config["teacher"]["max_tokens"],
        temperature=1.0,
        top_p=1.0,
        top_logprobs=config["teacher"]["top_logprobs"],
        student_tokenizer=student_tokenizer,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.prefill_chunk,
    )
    teacher.rescore_and_cache(args.src, dst, chunk_size=args.chunk_size)
    teacher.shutdown()


if __name__ == "__main__":
    main()
