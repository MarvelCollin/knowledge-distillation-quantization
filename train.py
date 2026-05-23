import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import os
import re
import yaml
import torch
import argparse
import torch.nn.functional as F

from dotenv import load_dotenv
import gc
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from transformers.optimization import Adafactor

from src.data.dataset import create_datasets
from src.distillation.loss import (
    adaptive_skew_lambda,
    build_teacher_distribution,
    compute_total_loss,
)
from src.distillation.qead import compute_qead_weights, teacher_confidence_weights
from src.evaluation.evaluator import run_test_cases
from src.student.model import StudentModel
from src.teacher.teacher_api import TeacherModel


def _extract_teacher_code(text: str) -> str:
    text = text.strip()
    fence = re.search(r'```(?:python)?\s*\n?(.*?)(?:```|$)', text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    m = re.search(r'def \w', text)
    if m:
        text = text[m.start():]
    elif not text.startswith("def "):
        text = "def " + text.lstrip()
    lines = text.split('\n')
    result = []
    for line in lines:
        if not result:
            result.append(line)
        elif not line or line[0] in (' ', '\t'):
            result.append(line)
        else:
            break
    while result and not result[-1].strip():
        result.pop()
    return '\n'.join(result)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_validation(student: StudentModel, val_loader: DataLoader, device: torch.device) -> float:
    student.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = student(input_ids, attention_mask)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += (shift_labels != -100).sum().item()

    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--offline", action="store_true",
                        help="Skip teacher API calls; use only existing cached responses.")
    parser.add_argument("--local", action="store_true",
                        help="Use local teacher model instead of API.")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = config["data"]["teacher_cache_dir"]

    if args.local:
        from transformers import AutoTokenizer
        from src.teacher.local_teacher import LocalTeacherModel
        _tokenizer = AutoTokenizer.from_pretrained(
            config["student"]["model_name"], trust_remote_code=True
        )
        train_dataset, val_dataset = create_datasets(config, _tokenizer)
        train_prompts = [train_dataset.get_prompt(i) for i in range(len(train_dataset))]

        local_path = config["teacher"].get("local_model_path", "cache/teacher-model")
        local_teacher = LocalTeacherModel(
            model_path=f"/workspace/{local_path}",
            max_tokens=config["teacher"]["max_tokens"],
            temperature=config["teacher"]["temperature"],
            top_logprobs=config["teacher"]["top_logprobs"],
            student_tokenizer=_tokenizer,
        )
        local_teacher.precompute_and_cache(train_prompts, cache_dir)
        del local_teacher
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Local teacher freed. GPU now: {torch.cuda.memory_allocated()/1024**3:.1f} GB. Loading student model...")

        student = StudentModel(
            model_name=config["student"]["model_name"],
            max_length=config["student"]["max_length"],
        )
        student.to(device)
    else:
        student = StudentModel(
            model_name=config["student"]["model_name"],
            max_length=config["student"]["max_length"],
        )
        student.to(device)

        train_dataset, val_dataset = create_datasets(config, student.tokenizer)
        train_prompts = [train_dataset.get_prompt(i) for i in range(len(train_dataset))]

        if args.offline:
            print("Offline mode: skipping teacher API. Using existing cache only.")
        else:
            api_key_env = config["teacher"].get("api_key_env", "TEACHER_API_KEY")
            teacher = TeacherModel(
                api_key=os.environ[api_key_env],
                model=config["teacher"]["model"],
                api_base=config["teacher"]["api_base"],
                max_tokens=config["teacher"]["max_tokens"],
                temperature=config["teacher"]["temperature"],
                top_logprobs=config["teacher"]["top_logprobs"],
                headers=config["teacher"].get("headers") or {},
                provider=config["teacher"].get("provider", "openrouter"),
            )
            teacher.precompute_and_cache(train_prompts, cache_dir)

    curriculum_mode = config["training"].get("curriculum", "none")
    if curriculum_mode == "length":
        order = train_dataset.curriculum_order(cache_dir)
        print(f"Curriculum: ordering {len(order)} samples by teacher response length (easy first).")

        class _FixedOrderSampler(Sampler):
            def __init__(self, indices):
                self.indices = list(indices)

            def __iter__(self):
                return iter(self.indices)

            def __len__(self):
                return len(self.indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            sampler=_FixedOrderSampler(order),
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, drop_last=True
        )
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"])

    optimizer = Adafactor(
        student.parameters(),
        lr=float(config["training"]["learning_rate"]),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )
    total_steps = len(train_loader) * config["training"]["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=total_steps,
    )

    vocab_size = student.model.config.vocab_size
    alpha = config["training"]["alpha"]
    skew_lambda = config["training"]["skew_lambda"]
    skew_lambda_max = config["training"].get("skew_lambda_max", skew_lambda)
    use_adaptive_skew = config["training"].get("adaptive_skew", False)
    use_confidence = config["training"].get("teacher_confidence_weight", False)
    distill_temp = config["training"]["distill_temperature"]
    grad_accum = config["training"]["gradient_accumulation_steps"]
    max_grad_norm = config["training"]["max_grad_norm"]
    max_length = config["student"]["max_length"]
    filter_failed_teacher = config["training"].get("filter_failed_teacher", False)

    global_step = 0
    student.model.gradient_checkpointing_enable()

    for epoch in range(config["training"]["num_epochs"]):
        student.train()
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            prompt_lengths = batch["prompt_length"]
            sample_idxs = batch["idx"]

            student_logits = student(input_ids, attention_mask)

            response_mask = labels != -100
            qead_weights = compute_qead_weights(student_logits, response_mask)

            teacher_dist = torch.zeros(len(sample_idxs), max_length, vocab_size, device=device)

            for i in range(len(sample_idxs)):
                idx = sample_idxs[i].item()
                cached = TeacherModel.load_cached(cache_dir, idx)
                if not cached or not cached.get("logprobs"):
                    continue
                if cached.get("prompt") != train_dataset.get_prompt(idx):
                    continue
                if filter_failed_teacher:
                    teacher_code = _extract_teacher_code(cached.get("text", ""))
                    test_cases = train_dataset.get_test_cases(idx)
                    if run_test_cases(teacher_code, test_cases)["passed"] < len(test_cases):
                        continue
                logprobs = cached["logprobs"]
                top_k_ids = cached.get("top_k_ids")
                top_k_vals = cached.get("top_k_vals")
                pl = prompt_lengths[i].item()
                dist_start = pl - 1
                aligned_len = min(len(logprobs), max_length - dist_start)
                if aligned_len <= 0:
                    continue
                kwargs = {}
                if top_k_ids is not None and top_k_vals is not None:
                    kwargs["top_k_ids"] = top_k_ids[:aligned_len]
                    kwargs["top_k_vals"] = top_k_vals[:aligned_len]
                partial = build_teacher_distribution(
                    logprobs[:aligned_len],
                    student.tokenizer,
                    vocab_size,
                    device,
                    distill_temp,
                    **kwargs,
                )
                teacher_dist[i, dist_start : dist_start + aligned_len] = partial

            valid_teacher = teacher_dist.sum(dim=-1) > 1e-8
            qead_weights = qead_weights * valid_teacher.float()
            if use_confidence:
                qead_weights = qead_weights * teacher_confidence_weights(teacher_dist)
            weight_sums = qead_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            qead_weights = qead_weights / weight_sums

            if use_adaptive_skew:
                effective_lambda = adaptive_skew_lambda(
                    student_logits, teacher_dist, qead_weights,
                    skew_lambda, skew_lambda_max, distill_temp,
                )
            else:
                effective_lambda = skew_lambda

            total, l_distill, l_task = compute_total_loss(
                student_logits, teacher_dist, qead_weights, labels, alpha, effective_lambda, distill_temp
            )

            (total / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    print(
                        f"step={global_step} "
                        f"total={total.item():.4f} "
                        f"distill={l_distill.item():.4f} "
                        f"task={l_task.item():.4f}"
                    )

                if global_step % config["training"]["save_steps"] == 0:
                    student.save(str(output_dir / f"checkpoint-{global_step}"))

                if global_step % config["training"]["eval_steps"] == 0:
                    val_loss = run_validation(student, val_loader, device)
                    print(f"val_loss={val_loss:.4f} at step {global_step}")
                    student.train()

    student.save(str(output_dir / "final"))


if __name__ == "__main__":
    main()
