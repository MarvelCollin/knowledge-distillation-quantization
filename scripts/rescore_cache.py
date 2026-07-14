import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from dotenv import load_dotenv
from transformers import AutoTokenizer

from src.config import load_config
from src.teacher.local_teacher import LocalTeacherModel
from src.teacher.rescorer import rescore_and_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-extract the true top-k teacher distribution from existing cached "
                    "trajectories via a single forward pass (vLLM prompt_logprobs), without "
                    "re-generating. Fixes caches whose logprobs were truncated by top_p/temperature."
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--src", default="cache/teacher_logprobs_coder15b",
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

    teacher = LocalTeacherModel.from_config(
        config, student_tokenizer,
        temperature=1.0,
        top_p=1.0,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.prefill_chunk,
        max_num_seqs=None,
        enable_prefix_caching=False,
    )
    rescore_and_cache(teacher, args.src, dst, chunk_size=args.chunk_size)
    teacher.shutdown()


if __name__ == "__main__":
    main()
