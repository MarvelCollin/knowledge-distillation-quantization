import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse

from dotenv import load_dotenv
from transformers import AutoTokenizer

from src.config import load_config
from src.data.dataset import train_split_indices
from src.data.problems import build_user_content
from src.utils.reasoning import SYSTEM_PROMPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Generate for only the first N train problems (smoke test).")
    args = parser.parse_args()

    load_dotenv()
    config = load_config("config/config.yaml")
    op = config["onpolicy"]
    student_path = str(Path(config["training"]["output_dir"]) / "final")
    src_dir = Path(config["data"]["onpolicy_cache_dir"] + "_gen")
    src_dir.mkdir(parents=True, exist_ok=True)

    student_tokenizer = AutoTokenizer.from_pretrained(
        config["student"]["model_name"], trust_remote_code=True)
    problems, train_indices = train_split_indices(
        config, student_tokenizer, config["data"]["teacher_cache_dir"])
    if args.limit:
        train_indices = train_indices[:args.limit]

    prompts = []
    for i in train_indices:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_content(problems[i])},
        ]
        prompts.append(student_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True))

    from vllm import LLM, SamplingParams
    llm = LLM(model=student_path, trust_remote_code=True, dtype="bfloat16",
              max_model_len=config["student"]["max_length"], gpu_memory_utilization=0.85)
    params = SamplingParams(
        temperature=op["temperature"], top_p=op["top_p"], max_tokens=op["max_tokens"], n=1)
    outputs = llm.generate(prompts, params)

    written = 0
    for i, out in zip(train_indices, outputs):
        text = out.outputs[0].text
        if not text.strip():
            continue
        ids = student_tokenizer(text, add_special_tokens=False).input_ids
        payload = {
            "prompt": build_user_content(problems[i]),
            "max_tokens": op["max_tokens"],
            "text": text,
            "tokens": student_tokenizer.convert_ids_to_tokens(ids),
        }
        (src_dir / f"{i}.json").write_text(json.dumps(payload))
        written += 1

    print(f"Generated {written} on-policy student sequences -> {src_dir}")
    print(f"Next: python scripts/rescore_cache.py --src {src_dir} "
          f"--dst {config['data']['onpolicy_cache_dir']} --max-model-len 32768 --gpu-mem 0.80 --chunk-size 1")


if __name__ == "__main__":
    main()
