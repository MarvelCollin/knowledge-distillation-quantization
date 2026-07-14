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
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from transformers.optimization import Adafactor

from src.data.batching import FixedBatchSampler, make_pad_collate, pack_by_budget, seed_worker
from src.data.dataset import create_datasets
from src.data.problems import build_user_content, load_problems
from src.distillation.loss import compute_loss_chunked_backward
from src.distillation.teacher_topk import preload_teacher_topk, scatter_teacher_topk
from src.distillation.qead import fused_qead_kld_pass, teacher_confidence_weights, entropy_focus_weights
from src.student.model import StudentModel
from src.teacher.cache_builder import precompute_and_cache
from src.teacher.local_teacher import LocalTeacherModel
from src.utils.runlog import RunLog, eta_str, show_progress_bars
from src.config import load_config


def run_validation(student: StudentModel, val_loader: DataLoader, device: torch.device) -> float:
    student.eval()
    total_loss = 0.0
    total_tokens = 0

    chunk_size = 1024
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            hidden = student.forward_hidden(input_ids, attention_mask)
            weight = student.lm_head_weight
            seq_len = hidden.size(1)
            shift_labels = torch.full_like(labels, -100)
            shift_labels[..., :-1] = labels[..., 1:]

            for s in range(0, seq_len, chunk_size):
                e = min(s + chunk_size, seq_len)
                logits_chunk = F.linear(hidden[:, s:e], weight)
                loss = F.cross_entropy(
                    logits_chunk.reshape(-1, logits_chunk.size(-1)),
                    shift_labels[:, s:e].reshape(-1),
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
        raw_difficulties = [p.get("difficulty", "") for p in raw_problems]
        local_teacher = LocalTeacherModel.from_config(config, student_tokenizer)
        chunk_size = config["teacher"].get("cache_chunk_size", 64)
        precompute_and_cache(
            local_teacher, raw_prompts, cache_dir,
            test_cases_per_prompt=raw_tests,
            difficulty_per_prompt=raw_difficulties,
            chunk_size=chunk_size,
            rejection_samples=config["teacher"].get("rejection_samples", 0),
            rejection_wave=config["teacher"].get("rejection_wave", 2),
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
    try:
        student.model = torch.compile(student.model, mode="reduce-overhead")
        print("  torch.compile: ON (reduce-overhead)")
    except Exception as e:
        print(f"  torch.compile: SKIP ({e})")


    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    pad_id = student_tokenizer.pad_token_id
    if pad_id is None:
        pad_id = student_tokenizer.eos_token_id
    pad_collate = make_pad_collate(pad_id)

    token_budget = int(config["training"].get("batch_token_budget", 16384))
    max_batch = int(config["training"].get("max_batch_size", 4))

    curriculum_mode = config["training"].get("curriculum", "none")
    if curriculum_mode == "length":
        order = train_dataset.curriculum_order()
        batches = pack_by_budget(order, train_dataset.lengths, token_budget, max_batch)
        print(f"Curriculum: {len(order)} samples by length (easy first), packed into "
              f"{len(batches)} batches (token_budget={token_budget}, max_batch={max_batch}).")

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=FixedBatchSampler(batches),
            collate_fn=pad_collate,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            drop_last=True,
            collate_fn=pad_collate,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=loader_generator,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=FixedBatchSampler(
            pack_by_budget(val_dataset.curriculum_order(), val_dataset.lengths, token_budget, max_batch)),
        collate_fn=pad_collate,
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
    effective_batch = config["training"]["batch_size"] * config["training"]["gradient_accumulation_steps"]
    optimizer_steps_per_epoch = max(1, len(train_dataset) // effective_batch)
    total_steps = optimizer_steps_per_epoch * config["training"]["num_epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=total_steps,
    )

    n_cache_files = len(list(Path(cache_dir).glob("*.json"))) if os.path.isdir(cache_dir) else 0
    n_train_samples = len(train_dataset)
    n_val_samples = len(val_dataset)
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
    use_entropy_focus = config["training"].get("entropy_focus", False)
    distill_temp = config["training"]["distill_temperature"]
    max_grad_norm = config["training"]["max_grad_norm"]
    eval_steps = config["training"]["eval_steps"]

    teacher_topk = preload_teacher_topk(
        cache_dir, train_dataset.original_indices, vocab_size, topk, distill_temp)
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

    best_val_loss = run_validation(student, val_loader, device)
    student.save(str(output_dir / "final"))
    torch.cuda.empty_cache()
    print(f"Baseline (untrained): val_loss={best_val_loss:.4f}  → saved as initial best (floor).")

    runlog = RunLog("logs/train_metrics.jsonl")
    runlog.event("train_start", total_steps=total_steps, n_train=n_train_samples,
                 epochs=config["training"]["num_epochs"], baseline_val_loss=round(best_val_loss, 4))
    loop_t0 = time.time()

    for epoch in range(config["training"]["num_epochs"]):
        student.train()
        optimizer.zero_grad()
        epoch_start = time.time()
        epoch_samples_seen = 0
        pending_samples = 0
        t_data = t_compute = t_val = 0.0
        t_mark = time.time()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['training']['num_epochs']}",
                                          disable=not show_progress_bars())):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            prompt_lengths = batch["prompt_length"]
            sample_idxs = batch["idx"]

            t_data += time.time() - t_mark
            t_mark = time.time()

            seq_len = input_ids.size(1)
            teacher_ids, teacher_probs = scatter_teacher_topk(
                teacher_topk, sample_idxs, prompt_lengths,
                train_dataset.original_indices, seq_len, topk, device)

            hidden = student.forward_hidden(input_ids, attention_mask)
            lm_head_weight = student.lm_head_weight

            response_mask = labels != -100
            predict_mask = torch.zeros_like(response_mask)
            predict_mask[:, :-1] = response_mask[:, 1:]
            qead_weights, per_token_kld = fused_qead_kld_pass(
                hidden, lm_head_weight, predict_mask, teacher_ids, teacher_probs,
                distill_temp, compute_kld=use_adaptive_skew,
            )

            valid_teacher = teacher_probs.sum(dim=-1) > 1e-8
            qead_weights = qead_weights * valid_teacher.float()
            if use_entropy_focus:
                qead_weights = qead_weights * entropy_focus_weights(teacher_probs)
            elif use_confidence:
                qead_weights = qead_weights * teacher_confidence_weights(teacher_probs)
            weight_sums = qead_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            qead_weights = qead_weights / weight_sums

            if use_adaptive_skew:
                per_sample_kld = (qead_weights * per_token_kld).sum(dim=-1) / qead_weights.sum(dim=-1).clamp(min=1e-8)
                gate = torch.tanh(per_sample_kld / 4.0)
                effective_lambda = (skew_lambda + (skew_lambda_max - skew_lambda) * gate).clamp(
                    min=skew_lambda, max=skew_lambda_max)
            else:
                effective_lambda = skew_lambda

            sample_alpha = alpha if bool(valid_teacher.any()) else 0.0

            loss_chunk = 512 if seq_len > 8192 else 1024
            total_val, distill_val, task_val = compute_loss_chunked_backward(
                hidden, lm_head_weight, teacher_ids, teacher_probs, qead_weights, labels,
                sample_alpha, effective_lambda, distill_temp,
                loss_scale=len(sample_idxs) / effective_batch,
                chunk_size=loss_chunk,
            )

            del hidden, teacher_ids, teacher_probs, qead_weights, valid_teacher
            del effective_lambda, weight_sums, per_token_kld
            del response_mask, predict_mask
            if seq_len > 8192:
                torch.cuda.empty_cache()

            samples_seen += len(sample_idxs)
            epoch_samples_seen += len(sample_idxs)
            pending_samples += len(sample_idxs)

            t_compute += time.time() - t_mark
            t_mark = time.time()

            if pending_samples >= effective_batch:
                pending_samples = 0
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    epoch_pct = epoch_samples_seen / max(n_train_samples, 1) * 100
                    total_pct = global_step / max(total_steps, 1) * 100
                    lr_now = scheduler.get_last_lr()[0]
                    eta = (time.time() - loop_t0) / global_step * (total_steps - global_step)
                    gpu_gb = torch.cuda.memory_reserved() / 1024 ** 3
                    print(
                        f"[ep {epoch + 1}/{config['training']['num_epochs']}] "
                        f"step={global_step}/{total_steps} ({total_pct:.0f}%) "
                        f"samples={epoch_samples_seen}/{n_train_samples} ({epoch_pct:.0f}% of epoch) "
                        f"total={total_val:.4f} "
                        f"distill={distill_val:.4f} "
                        f"task={task_val:.4f} "
                        f"lr={lr_now:.2e} "
                        f"gpu={gpu_gb:.1f}GB "
                        f"ETA {eta_str(eta)}"
                    )
                    runlog.event(
                        "step", step=global_step, epoch=epoch + 1,
                        total=round(total_val, 4), distill=round(distill_val, 4),
                        task=round(task_val, 4), lr=lr_now,
                        samples=samples_seen, gpu_gb=round(gpu_gb, 1),
                    )

                if global_step % eval_steps == 0:
                    v0 = time.time()
                    val_loss = run_validation(student, val_loader, device)
                    delta = val_loss - best_val_loss
                    improved = val_loss < best_val_loss
                    line = (f"step {global_step}/{total_steps}  val_loss={val_loss:.4f}  "
                            f"Δbest={delta:+.4f}")
                    if improved:
                        best_val_loss = val_loss
                        student.save(str(output_dir / "final"))
                        print(line + "  ✓ saved best")
                    else:
                        print(line + f"  (best={best_val_loss:.4f})")
                    runlog.event("val", step=global_step, val_loss=round(val_loss, 4),
                                 best=round(best_val_loss, 4), improved=improved)
                    student.train()
                    torch.cuda.empty_cache()
                    t_val += time.time() - v0

                t_mark = time.time()

        epoch_elapsed = time.time() - epoch_start
        epoch_m, epoch_s = divmod(int(epoch_elapsed), 60)
        print()
        print(f"[epoch {epoch + 1}/{config['training']['num_epochs']} done] "
              f"samples seen: {epoch_samples_seen}/{n_train_samples}  "
              f"cumulative: {samples_seen}  "
              f"time: {epoch_m}m {epoch_s:02d}s  "
              f"(compute {t_compute / 60:.1f}m | val+select {t_val / 60:.1f}m | data wait {t_data / 60:.1f}m)  "
              f"best_val_loss: {best_val_loss:.4f}")
        print()
        runlog.event("epoch_done", epoch=epoch + 1, secs=int(epoch_elapsed),
                     compute_min=round(t_compute / 60, 1), val_min=round(t_val / 60, 1),
                     data_wait_min=round(t_data / 60, 1), best_val_loss=round(best_val_loss, 4))

    student.save(str(output_dir / "final_last"))
    print(f"Last-epoch (fully-trained) checkpoint saved at outputs/final_last for evaluation.")

    print(f"Training complete. Best val_loss={best_val_loss:.4f} saved at outputs/final.")
    runlog.event("train_done", best_val_loss=round(best_val_loss, 4), steps=global_step)

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
