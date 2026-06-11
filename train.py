import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml
import torch
import argparse
import random
import time
import numpy as np
import torch.nn.functional as F

from dotenv import load_dotenv
import gc
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor

from src.data.dataset import PROMPT_TEMPLATE, create_datasets, load_problems
from src.distillation.loss import (
    adaptive_skew_lambda,
    build_teacher_distribution,
    compute_total_loss,
)
from src.distillation.qead import compute_qead_weights, teacher_confidence_weights
from src.evaluation.evaluator import run_test_cases
from src.student.model import StudentModel
from src.teacher.local_teacher import LocalTeacherModel
from evaluate import generate_solution


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


def run_test_validation(student: StudentModel, val_dataset, device: torch.device,
                        num_problems: int, max_new_tokens: int) -> dict:
    student.eval()
    n = min(num_problems, len(val_dataset))
    passed_total = 0
    test_total = 0
    solved_count = 0

    for i in range(n):
        prompt = val_dataset.get_prompt(i)
        test_cases = val_dataset.get_test_cases(i)
        try:
            code = generate_solution(student, prompt, test_cases, max_new_tokens, device)
            r = run_test_cases(code, test_cases)
        except Exception as exc:
            print(f"  [test-val {i}] generation/exec error: {type(exc).__name__}: {exc}")
            continue
        passed_total += r["passed"]
        test_total += r["total"]
        if r["total"] > 0 and r["passed"] == r["total"]:
            solved_count += 1

    return {
        "test_pass_rate": passed_total / max(test_total, 1),
        "problems_solved": solved_count,
        "problems_total": n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--offline", action="store_true",
                        help="Skip teacher cache build; use only existing cached responses.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Override data.max_samples from config (total problems loaded before train/val split).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training.num_epochs from config.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override training.seed for multi-seed runs.")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)

    if args.max_samples is not None:
        config["data"]["max_samples"] = args.max_samples
    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
    if args.seed is not None:
        config["training"]["seed"] = args.seed

    seed = config["training"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"  Seed locked: {seed}")

    n_total = config["data"]["max_samples"]
    n_train = int(n_total * config["data"]["train_ratio"])
    print(f"  Run config: max_samples={n_total}  (~{n_train} train / {n_total - n_train} val)"
          f"  epochs={config['training']['num_epochs']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = config["data"]["teacher_cache_dir"]

    student_tokenizer = AutoTokenizer.from_pretrained(
        config["student"]["model_name"], trust_remote_code=True
    )

    if not args.offline:
        raw_problems = load_problems(config)
        raw_prompts = [PROMPT_TEMPLATE.format(text=p["text"]) for p in raw_problems]
        raw_tests = [p["test_cases"] for p in raw_problems]
        local_path = config["teacher"]["local_model_path"]
        local_teacher = LocalTeacherModel(
            model_path=f"/workspace/{local_path}",
            max_tokens=config["teacher"]["max_tokens"],
            temperature=config["teacher"]["temperature"],
            top_p=config["teacher"].get("top_p", 0.95),
            top_logprobs=config["teacher"]["top_logprobs"],
            student_tokenizer=student_tokenizer,
            gpu_memory_utilization=config["teacher"].get("gpu_memory_utilization", 0.85),
            max_model_len=config["teacher"].get("max_model_len", 10240),
        )
        chunk_size = config["teacher"].get("cache_chunk_size", 64)
        local_teacher.precompute_and_cache(raw_prompts, cache_dir,
                                           test_cases_per_prompt=raw_tests,
                                           chunk_size=chunk_size)
        local_teacher.shutdown()
        del local_teacher
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Local teacher freed. Loading student model...")
    else:
        print("Offline mode: skipping teacher cache build. Using existing cache only.")

    train_dataset, val_dataset = create_datasets(config, student_tokenizer, cache_dir)

    student = StudentModel(
        model_name=config["student"]["model_name"],
        max_length=config["student"]["max_length"],
    )
    student.to(device)

    curriculum_mode = config["training"].get("curriculum", "none")
    if curriculum_mode == "length":
        order = train_dataset.curriculum_order()
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
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )

    optimizer = Adafactor(
        student.parameters(),
        lr=float(config["training"]["learning_rate"]),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )
    optimizer_steps_per_epoch = len(train_loader) // config["training"]["gradient_accumulation_steps"]
    total_steps = optimizer_steps_per_epoch * config["training"]["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=total_steps,
    )

    import os
    n_cache_files = len(list(Path(cache_dir).glob("*.json"))) if os.path.isdir(cache_dir) else 0
    n_train_samples = len(train_dataset)
    n_val_samples = len(val_dataset)
    effective_batch = config["training"]["batch_size"] * config["training"]["gradient_accumulation_steps"]
    print()
    print("═" * 60)
    print("  Training data breakdown")
    print("═" * 60)
    print(f"  Teacher cache files       : {n_cache_files}")
    print(f"  Passing tests (usable)    : {n_train_samples + n_val_samples}  ({(n_train_samples + n_val_samples) / max(n_cache_files, 1):.1%} of cache)")
    print(f"  Train split (90%)         : {n_train_samples}")
    print(f"  Val split (10%)           : {n_val_samples}")
    print(f"  batch_size × grad_accum   : {config['training']['batch_size']} × {config['training']['gradient_accumulation_steps']} = {effective_batch} effective")
    print(f"  Optimizer steps / epoch   : {optimizer_steps_per_epoch}")
    print(f"  Total epochs              : {config['training']['num_epochs']}")
    print(f"  Total optimizer steps     : {total_steps}")
    print("═" * 60)
    print()

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
    test_eval_steps = config["training"].get("test_eval_steps", 0)
    test_eval_problems = config["training"].get("test_eval_problems", 5)
    test_eval_max_new_tokens = config["training"].get("test_eval_max_new_tokens", 1024)

    global_step = 0
    samples_seen = 0
    best_val_loss = float("inf")
    train_start = time.time()
    student.model.gradient_checkpointing_enable()

    for epoch in range(config["training"]["num_epochs"]):
        student.train()
        optimizer.zero_grad()
        epoch_start = time.time()
        epoch_samples_seen = 0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['training']['num_epochs']}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            prompt_lengths = batch["prompt_length"]
            sample_idxs = batch["idx"]

            student_logits = student(input_ids, attention_mask)

            response_mask = labels != -100
            predict_mask = torch.zeros_like(response_mask)
            predict_mask[:, :-1] = response_mask[:, 1:]
            qead_weights = compute_qead_weights(student_logits, predict_mask)

            teacher_dist = torch.zeros(len(sample_idxs), max_length, vocab_size, device=device)

            for i in range(len(sample_idxs)):
                local_idx = sample_idxs[i].item()
                cache_idx = train_dataset.original_indices[local_idx]
                cached = LocalTeacherModel.load_cached(cache_dir, cache_idx)
                if not cached or not cached.get("logprobs"):
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

            samples_seen += len(sample_idxs)
            epoch_samples_seen += len(sample_idxs)

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    epoch_pct = epoch_samples_seen / max(n_train_samples, 1) * 100
                    total_pct = global_step / max(total_steps, 1) * 100
                    print(
                        f"[ep {epoch + 1}/{config['training']['num_epochs']}] "
                        f"step={global_step}/{total_steps} ({total_pct:.0f}%) "
                        f"samples={epoch_samples_seen}/{n_train_samples} ({epoch_pct:.0f}% of epoch) "
                        f"total={total.item():.4f} "
                        f"distill={l_distill.item():.4f} "
                        f"task={l_task.item():.4f}"
                    )

                if global_step % config["training"]["eval_steps"] == 0:
                    val_loss = run_validation(student, val_loader, device)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        student.save(str(output_dir / "final"))
                        print(f"val_loss={val_loss:.4f} at step {global_step}  ✓ saved best")
                    else:
                        print(f"val_loss={val_loss:.4f} at step {global_step}  (best={best_val_loss:.4f})")
                    student.train()

                if test_eval_steps > 0 and global_step % test_eval_steps == 0:
                    stats = run_test_validation(
                        student, val_dataset, device,
                        num_problems=test_eval_problems,
                        max_new_tokens=test_eval_max_new_tokens,
                    )
                    print(
                        f"[TEST-VAL] step={global_step} "
                        f"solved={stats['problems_solved']}/{stats['problems_total']} "
                        f"test_pass_rate={stats['test_pass_rate']:.1%}"
                    )
                    student.train()

        epoch_elapsed = time.time() - epoch_start
        epoch_m, epoch_s = divmod(int(epoch_elapsed), 60)
        print()
        print(f"[epoch {epoch + 1}/{config['training']['num_epochs']} done] "
              f"samples seen: {epoch_samples_seen}/{n_train_samples}  "
              f"cumulative: {samples_seen}  "
              f"time: {epoch_m}m {epoch_s:02d}s  "
              f"best_val: {best_val_loss:.4f}")
        print()

    if best_val_loss == float("inf"):
        student.save(str(output_dir / "final"))
        print("No validation eval ran during training — saved final-state checkpoint.")
    else:
        print(f"Training complete. Best val_loss={best_val_loss:.4f} saved at outputs/final.")

    elapsed = time.time() - train_start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    print(f"  Wall-clock        : {hours}h {minutes:02d}m {seconds:02d}s")
    print(f"  Optimizer steps   : {global_step}")
    print(f"  Samples processed : {samples_seen} (across {config['training']['num_epochs']} epochs)")
    print(f"  Cache files used  : {n_train_samples + n_val_samples} of {n_cache_files} total")
    print(f"  Seed              : {seed}")


if __name__ == "__main__":
    main()
