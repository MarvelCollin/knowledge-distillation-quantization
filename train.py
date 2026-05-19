import os
import yaml
import torch
import argparse
import torch.nn.functional as F
from pathlib import Path

from dotenv import load_dotenv
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.data.dataset import create_datasets
from src.distillation.loss import build_teacher_distribution, compute_total_loss
from src.distillation.qead import compute_qead_weights
from src.student.model import StudentModel
from src.teacher.deepseek_api import DeepSeekTeacher


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
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    student = StudentModel(
        model_name=config["student"]["model_name"],
        max_length=config["student"]["max_length"],
    )
    student.to(device)

    train_dataset, val_dataset = create_datasets(config, student.tokenizer)

    teacher = DeepSeekTeacher(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        model=config["teacher"]["model"],
        api_base=config["teacher"]["api_base"],
        max_tokens=config["teacher"]["max_tokens"],
        temperature=config["teacher"]["temperature"],
        top_logprobs=config["teacher"]["top_logprobs"],
    )

    cache_dir = config["data"]["teacher_cache_dir"]
    train_prompts = [train_dataset.get_prompt(i) for i in range(len(train_dataset))]
    teacher.precompute_and_cache(train_prompts, cache_dir)

    train_loader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"])

    optimizer = AdamW(student.parameters(), lr=config["training"]["learning_rate"])
    total_steps = len(train_loader) * config["training"]["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=total_steps,
    )

    vocab_size = student.model.config.vocab_size
    alpha = config["training"]["alpha"]
    skew_lambda = config["training"]["skew_lambda"]
    distill_temp = config["training"]["distill_temperature"]
    grad_accum = config["training"]["gradient_accumulation_steps"]
    max_grad_norm = config["training"]["max_grad_norm"]
    max_length = config["student"]["max_length"]

    global_step = 0

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
                cached = DeepSeekTeacher.load_cached(cache_dir, sample_idxs[i].item())
                if not cached or not cached["logprobs"]:
                    continue
                logprobs = cached["logprobs"]
                pl = prompt_lengths[i].item()
                aligned_len = min(len(logprobs), max_length - pl)
                if aligned_len <= 0:
                    continue
                partial = build_teacher_distribution(
                    logprobs[:aligned_len],
                    student.tokenizer,
                    vocab_size,
                    device,
                    distill_temp,
                )
                teacher_dist[i, pl : pl + aligned_len] = partial

            valid_teacher = teacher_dist.sum(dim=-1) > 1e-8
            qead_weights = qead_weights * valid_teacher.float()
            weight_sums = qead_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            qead_weights = qead_weights / weight_sums

            total, l_distill, l_task = compute_total_loss(
                student_logits, teacher_dist, qead_weights, labels, alpha, skew_lambda, distill_temp
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
