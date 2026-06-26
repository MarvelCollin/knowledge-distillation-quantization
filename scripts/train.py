import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from transformers.optimization import Adafactor

from src.data.dataset import PROMPT_TEMPLATE, build_user_content, create_datasets, load_problems
from src.distillation.loss import (
    adaptive_skew_lambda,
    build_teacher_topk,
    compute_loss_chunked_backward,
    compute_total_loss,
)
from src.distillation.qead import compute_qead_weights, teacher_confidence_weights
from src.evaluation.evaluator import run_test_cases, failing_cases, write_failure_report
from src.student.model import StudentModel
from src.teacher.local_teacher import LocalTeacherModel
from src.config import load_config
from src.evaluation.generation import generate_student_solution


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
                        num_problems: int, max_new_tokens: int,
                        detail_path=None) -> dict:
    student.eval()
    n = min(num_problems, len(val_dataset))
    passed_total = 0
    test_total = 0
    solved_count = 0
    entries = []

    for i in range(n):
        prompt = val_dataset.get_prompt(i)
        test_cases = val_dataset.get_test_cases(i)
        code = ""
        try:
            _, code = generate_student_solution(student, prompt, test_cases, max_new_tokens)
            r = run_test_cases(code, test_cases)
        except Exception as exc:
            print(f"  [test-val {i}] generation/exec error: {type(exc).__name__}: {exc}")
            entries.append({
                "header": f"Problem {i}  —  generation/exec error",
                "solved": False, "code": code,
                "fails": [("runtime_error", f"{type(exc).__name__}: {exc}")],
            })
            continue
        passed_total += r["passed"]
        test_total += r["total"]
        solved = r["total"] > 0 and r["passed"] == r["total"]
        if solved:
            solved_count += 1
        entries.append({
            "header": f"Problem {i}  —  {r['passed']}/{r['total']} test cases",
            "solved": solved, "code": code,
            "fails": failing_cases(r, limit=3),
        })

    if detail_path is not None:
        try:
            write_failure_report(detail_path, f"Train selection eval — {Path(detail_path).stem}", entries)
        except Exception as exc:
            print(f"  [test-val] could not write detail report: {type(exc).__name__}: {exc}")

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
    parser.add_argument("--lr", type=float, default=None,
                        help="Override training.learning_rate from config.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override training.seed for multi-seed runs.")
    parser.add_argument("--cache-dir", default=None,
                        help="Override data.teacher_cache_dir.")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)

    if args.max_samples is not None:
        config["data"]["max_samples"] = args.max_samples
    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.cache_dir is not None:
        config["data"]["teacher_cache_dir"] = args.cache_dir

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
    if device.type == "cuda":
        _dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        _total_mem = torch.cuda.get_device_properties(_dev_idx).total_memory
        _cap_frac = min(22.0 * 1024 ** 3 / _total_mem, 1.0)
        torch.cuda.set_per_process_memory_fraction(_cap_frac, _dev_idx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"  CUDA memory cap  : 22 GB / {_total_mem / 1024**3:.1f} GB (fraction={_cap_frac:.3f})")
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = config["data"]["teacher_cache_dir"]

    student_tokenizer = AutoTokenizer.from_pretrained(
        config["student"]["model_name"], trust_remote_code=True
    )

    if not args.offline:
        raw_problems = load_problems(config)
        raw_prompts = [build_user_content(p) for p in raw_problems]
        raw_tests = [p["test_cases"] for p in raw_problems]
        local_teacher = LocalTeacherModel(
            model_path=config["teacher"]["local_model_path"],
            max_tokens=config["teacher"]["max_tokens"],
            temperature=config["teacher"]["temperature"],
            top_p=config["teacher"].get("top_p", 0.95),
            top_logprobs=config["teacher"]["top_logprobs"],
            student_tokenizer=student_tokenizer,
            gpu_memory_utilization=config["teacher"].get("gpu_memory_utilization", 0.85),
            max_model_len=config["teacher"].get("max_model_len", 10240),
        )
        chunk_size = config["teacher"].get("cache_chunk_size", 64)
        local_teacher.precompute_and_cache(
            raw_prompts, cache_dir,
            test_cases_per_prompt=raw_tests,
            chunk_size=chunk_size,
            rejection_samples=config["teacher"].get("rejection_samples", 0),
            rejection_temperature=config["teacher"].get("rejection_temperature", 1.0),
            rejection_top_p=config["teacher"].get("rejection_top_p", 0.95),
        )
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

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

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
            worker_init_fn=seed_worker,
            generator=loader_generator,
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
            worker_init_fn=seed_worker,
            generator=loader_generator,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    optimizer = Adafactor(
        student.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )
    optimizer_steps_per_epoch = len(train_loader) // config["training"]["gradient_accumulation_steps"]
    total_steps = optimizer_steps_per_epoch * config["training"]["num_epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=total_steps,
    )

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
    topk = config["teacher"]["top_logprobs"] + 1
    alpha = config["training"]["alpha"]
    skew_lambda = config["training"]["skew_lambda"]
    skew_lambda_max = config["training"].get("skew_lambda_max", skew_lambda)
    use_adaptive_skew = config["training"].get("adaptive_skew", False)
    use_confidence = config["training"].get("teacher_confidence_weight", False)
    distill_temp = config["training"]["distill_temperature"]
    grad_accum = config["training"]["gradient_accumulation_steps"]
    max_grad_norm = config["training"]["max_grad_norm"]
    eval_steps = config["training"]["eval_steps"]
    select_problems = config["training"]["select_problems"]
    select_max_new_tokens = config["training"]["select_max_new_tokens"]

    teacher_topk = {}
    for cache_idx in train_dataset.original_indices:
        cached = LocalTeacherModel.load_cached(cache_dir, cache_idx)
        if not cached or not cached.get("top_k_ids") or not cached.get("top_k_vals"):
            continue
        teacher_topk[cache_idx] = build_teacher_topk(
            cached["top_k_ids"], cached["top_k_vals"],
            vocab_size, topk, torch.device("cpu"), distill_temp,
        )
    topk_mb = sum(i.nbytes + p.nbytes for i, p in teacher_topk.values()) / 1024**2
    print(f"Preloaded teacher top-k for {len(teacher_topk)}/{n_train_samples} train samples ({topk_mb:.0f} MB CPU).")

    global_step = 0
    samples_seen = 0
    train_start = time.time()
    if config["training"].get("gradient_checkpointing", True):
        student.model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: ON (saves memory, ~30% slower)")
    else:
        print("  Gradient checkpointing: OFF (faster, higher memory)")

    neftune_alpha = float(config["training"].get("neftune_alpha", 0.0))
    if neftune_alpha > 0:
        def _neftune_hook(module, args, output):
            if module.training:
                dims = output.size(-2) * output.size(-1)
                mag = neftune_alpha / (dims ** 0.5)
                output = output + torch.empty_like(output).uniform_(-mag, mag)
            return output
        student.model.get_input_embeddings().register_forward_hook(_neftune_hook)
        print(f"  NEFTune: ON (alpha={neftune_alpha}) -- noisy embeddings during training only")

    base_stats = run_test_validation(
        student, val_dataset, device,
        num_problems=select_problems, max_new_tokens=select_max_new_tokens,
        detail_path=output_dir / "train_details" / "step_00000_base.md",
    )
    best_solve_rate = base_stats["test_pass_rate"]
    best_val_loss = run_validation(student, val_loader, device)
    student.save(str(output_dir / "final"))
    print(f"Baseline (untrained) selection: solve={best_solve_rate:.1%} "
          f"({base_stats['problems_solved']}/{base_stats['problems_total']})  "
          f"val_loss={best_val_loss:.4f}  → saved as initial best (floor).")

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
            seq_len = student_logits.size(1)

            response_mask = labels != -100
            predict_mask = torch.zeros_like(response_mask)
            predict_mask[:, :-1] = response_mask[:, 1:]
            qead_weights = compute_qead_weights(student_logits, predict_mask)

            teacher_ids = torch.zeros(len(sample_idxs), seq_len, topk, dtype=torch.long, device=device)
            teacher_probs = torch.zeros(len(sample_idxs), seq_len, topk, device=device)

            for i in range(len(sample_idxs)):
                local_idx = sample_idxs[i].item()
                cache_idx = train_dataset.original_indices[local_idx]
                entry = teacher_topk.get(cache_idx)
                if entry is None:
                    continue
                ids_full, probs_full = entry
                pl = prompt_lengths[i].item()
                dist_start = pl - 1
                aligned_len = min(ids_full.size(0), seq_len - dist_start)
                if aligned_len <= 0:
                    continue
                teacher_ids[i, dist_start : dist_start + aligned_len] = ids_full[:aligned_len].to(device)
                teacher_probs[i, dist_start : dist_start + aligned_len] = probs_full[:aligned_len].to(device)

            valid_teacher = teacher_probs.sum(dim=-1) > 1e-8
            qead_weights = qead_weights * valid_teacher.float()
            if use_confidence:
                qead_weights = qead_weights * teacher_confidence_weights(teacher_probs)
            weight_sums = qead_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            qead_weights = qead_weights / weight_sums

            if use_adaptive_skew:
                effective_lambda = adaptive_skew_lambda(
                    student_logits, teacher_ids, teacher_probs, qead_weights,
                    skew_lambda, skew_lambda_max, distill_temp,
                )
            else:
                effective_lambda = skew_lambda

            sample_alpha = alpha if bool(valid_teacher.any()) else 0.0

            # Chunked-over-sequence loss + backward: keeps the (chunk, vocab) softmax
            # working set small so peak VRAM stays under the 22 GB cap even on the
            # longest (max_length) sequences at the end of the curriculum.
            total_val, distill_val, task_val = compute_loss_chunked_backward(
                student_logits, teacher_ids, teacher_probs, qead_weights, labels,
                sample_alpha, effective_lambda, distill_temp, loss_scale=1.0 / grad_accum,
            )

            # Free large GPU tensors immediately to keep VRAM below 22 GB cap.
            del student_logits, teacher_ids, teacher_probs, qead_weights, valid_teacher
            del effective_lambda, weight_sums
            del response_mask, predict_mask

            samples_seen += len(sample_idxs)
            epoch_samples_seen += len(sample_idxs)

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    torch.cuda.empty_cache()

                if global_step % 10 == 0:
                    epoch_pct = epoch_samples_seen / max(n_train_samples, 1) * 100
                    total_pct = global_step / max(total_steps, 1) * 100
                    print(
                        f"[ep {epoch + 1}/{config['training']['num_epochs']}] "
                        f"step={global_step}/{total_steps} ({total_pct:.0f}%) "
                        f"samples={epoch_samples_seen}/{n_train_samples} ({epoch_pct:.0f}% of epoch) "
                        f"total={total_val:.4f} "
                        f"distill={distill_val:.4f} "
                        f"task={task_val:.4f}"
                    )

                if global_step % eval_steps == 0:
                    val_loss = run_validation(student, val_loader, device)
                    stats = run_test_validation(
                        student, val_dataset, device,
                        num_problems=select_problems,
                        max_new_tokens=select_max_new_tokens,
                        detail_path=output_dir / "train_details" / f"step_{global_step:05d}.md",
                    )
                    solve_rate = stats["test_pass_rate"]
                    line = (
                        f"step {global_step}/{total_steps}  "
                        f"solve={solve_rate:.1%} ({stats['problems_solved']}/{stats['problems_total']})  "
                        f"val_loss={val_loss:.4f}"
                    )
                    if (solve_rate, -val_loss) > (best_solve_rate, -best_val_loss):
                        best_solve_rate = solve_rate
                        best_val_loss = val_loss
                        student.save(str(output_dir / "final"))
                        print(line + "  ✓ saved best")
                    else:
                        print(line + f"  (best solve={best_solve_rate:.1%})")
                    student.train()

        epoch_elapsed = time.time() - epoch_start
        epoch_m, epoch_s = divmod(int(epoch_elapsed), 60)
        print()
        print(f"[epoch {epoch + 1}/{config['training']['num_epochs']} done] "
              f"samples seen: {epoch_samples_seen}/{n_train_samples}  "
              f"cumulative: {samples_seen}  "
              f"time: {epoch_m}m {epoch_s:02d}s  "
              f"best_solve: {best_solve_rate:.1%}")
        print()

    student.save(str(output_dir / "final_last"))
    print(f"Last-epoch (fully-trained) checkpoint saved at outputs/final_last for evaluation.")

    print(f"Training complete. Best solve_rate={best_solve_rate:.1%} (val_loss={best_val_loss:.4f}) "
          f"saved at outputs/final — guaranteed >= base ({base_stats['test_pass_rate']:.1%}).")

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
