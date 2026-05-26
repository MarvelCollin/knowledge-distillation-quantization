import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml
import torch
import argparse
import torch.nn.functional as F

from dotenv import load_dotenv
import gc
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
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
from src.teacher.local_teacher import LocalTeacherModel
from src.utils.reasoning import extract_code
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
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)

    if args.max_samples is not None:
        config["data"]["max_samples"] = args.max_samples
    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs

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
    train_dataset, val_dataset = create_datasets(config, student_tokenizer)

    if not args.offline:
        train_prompts = [train_dataset.get_prompt(i) for i in range(len(train_dataset))]
        train_tests = [train_dataset.get_test_cases(i) for i in range(len(train_dataset))]
        local_path = config["teacher"]["local_model_path"]
        local_teacher = LocalTeacherModel(
            model_path=f"/workspace/{local_path}",
            max_tokens=config["teacher"]["max_tokens"],
            temperature=config["teacher"]["temperature"],
            top_p=config["teacher"].get("top_p", 0.95),
            top_logprobs=config["teacher"]["top_logprobs"],
            student_tokenizer=student_tokenizer,
        )
        local_teacher.precompute_and_cache(train_prompts, cache_dir,
                                           test_cases_per_prompt=train_tests)
        del local_teacher
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Local teacher freed. GPU now: {torch.cuda.memory_allocated()/1024**3:.1f} GB. Loading student model...")
    else:
        print("Offline mode: skipping teacher cache build. Using existing cache only.")

    student = StudentModel(
        model_name=config["student"]["model_name"],
        max_length=config["student"]["max_length"],
    )
    student.to(device)

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
    test_eval_steps = config["training"].get("test_eval_steps", 0)
    test_eval_problems = config["training"].get("test_eval_problems", 5)
    test_eval_max_new_tokens = config["training"].get("test_eval_max_new_tokens", 1024)

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
                cached = LocalTeacherModel.load_cached(cache_dir, idx)
                if not cached or not cached.get("logprobs"):
                    continue
                if cached.get("prompt") != train_dataset.get_prompt(idx):
                    continue
                if filter_failed_teacher:
                    cached_passed = cached.get("test_passed")
                    cached_total = cached.get("test_total")
                    if cached_passed is not None and cached_total is not None:
                        if cached_total == 0 or cached_passed < cached_total:
                            continue
                    else:
                        teacher_code = extract_code(cached.get("text", ""))
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
                    student.save(str(output_dir / "final"))

                if global_step % config["training"]["eval_steps"] == 0:
                    val_loss = run_validation(student, val_loader, device)
                    print(f"val_loss={val_loss:.4f} at step {global_step}")
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

    student.save(str(output_dir / "final"))


if __name__ == "__main__":
    main()
