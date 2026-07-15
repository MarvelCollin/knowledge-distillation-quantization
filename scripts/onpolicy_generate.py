import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer

from src.config import load_config
from src.data.dataset import train_split_indices
from src.data.problems import build_user_content
from src.teacher.local_teacher import LocalTeacherModel
from src.teacher.rescorer import rescore_and_cache
from src.utils.gpu import cleanup_vllm, wait_for_gpu_freed
from src.utils.reasoning import SYSTEM_PROMPT


def main():
    load_dotenv()
    config = load_config("config/config.yaml")
    op = config["onpolicy"]
    student_path = str(Path(config["training"]["output_dir"]) / "final")
    onpolicy_dir = config["data"]["onpolicy_cache_dir"]
    src_dir = Path(onpolicy_dir + "_gen")
    src_dir.mkdir(parents=True, exist_ok=True)

    student_tokenizer = AutoTokenizer.from_pretrained(
        config["student"]["model_name"], trust_remote_code=True)
    problems, train_indices = train_split_indices(
        config, student_tokenizer, config["data"]["teacher_cache_dir"])

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

    cleanup_vllm(llm)
    gc.collect()
    torch.cuda.empty_cache()
    wait_for_gpu_freed(20.0)

    teacher = LocalTeacherModel.from_config(config, student_tokenizer)
    rescore_and_cache(teacher, str(src_dir), onpolicy_dir, chunk_size=op["rescore_chunk_size"])
    teacher.shutdown()


if __name__ == "__main__":
    main()
