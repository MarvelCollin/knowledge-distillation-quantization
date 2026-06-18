import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import argparse

import torch
import yaml
from dotenv import load_dotenv

from src.data.dataset import create_datasets
from src.distillation.loss import build_teacher_topk
from src.student.model import StudentModel
from src.teacher.local_teacher import LocalTeacherModel


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def diagnose(student, dataset, cache_dir, vocab, topk, distill_temp, device):
    disagree_n = disagree_tot = 0
    ent_agree_sum = ent_agree_n = 0.0
    kl_sum = kl_n = 0.0
    for i in range(len(dataset)):
        batch = dataset[i]
        input_ids = batch["input_ids"].unsqueeze(0).to(device)
        attn = batch["attention_mask"].unsqueeze(0).to(device)
        pl = batch["prompt_length"].item()
        orig_idx = dataset.original_indices[i]

        cached = LocalTeacherModel.load_cached(cache_dir, orig_idx)
        ids_full, probs_full = build_teacher_topk(
            cached["top_k_ids"], cached["top_k_vals"], vocab, topk, torch.device("cpu"), distill_temp
        )

        logits = student(input_ids, attn)[0].float()
        seq_len = logits.size(0)
        dist_start = pl - 1
        aligned = min(ids_full.size(0), seq_len - dist_start)
        if aligned <= 0:
            continue

        s = logits[dist_start:dist_start + aligned]
        tids = ids_full[:aligned].to(device)
        tprobs = probs_full[:aligned].to(device)
        valid = tprobs.sum(-1) > 1e-6
        if valid.sum() == 0:
            continue

        s_logprob = torch.log_softmax(s, dim=-1)
        student_top = s.argmax(-1)
        teacher_top = tids.gather(1, tprobs.argmax(-1, keepdim=True)).squeeze(1)
        agree = (student_top == teacher_top) & valid

        safe_t = tprobs.clamp(min=1e-10)
        tent = -(safe_t.log() * tprobs).sum(-1)
        s_at = torch.gather(s_logprob, 1, tids)
        kl = (tprobs * (safe_t.log() - s_at)).sum(-1)

        v = valid
        disagree_n += int((~agree & v).sum())
        disagree_tot += int(v.sum())
        ag = agree
        ent_agree_sum += float(tent[ag].sum())
        ent_agree_n += int(ag.sum())
        kl_sum += float(kl[v].sum())
        kl_n += int(v.sum())

    return {
        "disagree_rate": disagree_n / max(disagree_tot, 1),
        "mean_entropy_at_agreement": ent_agree_sum / max(ent_agree_n, 1),
        "mean_kl_teacher_student": kl_sum / max(kl_n, 1),
        "positions": disagree_tot,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test: does the teacher actually have something to teach THIS student? "
                    "Loads the real student and measures token-level divergence from the teacher."
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--new", required=True, help="Rescored (soft) cache dir.")
    parser.add_argument("--old", default=None, help="Original (one-hot) cache dir for contrast.")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distill_temp = config["training"]["distill_temperature"]
    topk = config["teacher"]["top_logprobs"] + 1

    student = StudentModel(
        model_name=config["student"]["model_name"],
        max_length=config["student"]["max_length"],
    )
    student.to(device)
    student.eval()
    vocab = student.model.config.vocab_size

    train_ds, val_ds = create_datasets(config, student.tokenizer, args.new)
    print(f"\nDiagnosing {len(train_ds)} train problems (distill_temperature={distill_temp})...\n")

    new_stats = diagnose(student, train_ds, args.new, vocab, topk, distill_temp, device)
    print("=== RESCORED (soft) teacher vs current student ===")
    print(f"  positions scored                 : {new_stats['positions']}")
    print(f"  student top-1 DISAGREES w/ teacher: {new_stats['disagree_rate']:.1%}   (trajectory teaching signal)")
    print(f"  teacher entropy at AGREEMENT pos  : {new_stats['mean_entropy_at_agreement']:.3f} nats   (soft bonus SFT cannot give)")
    print(f"  mean KL(teacher || student)       : {new_stats['mean_kl_teacher_student']:.3f}   (total per-token learning signal)")

    if args.old:
        old_train, _ = create_datasets(config, student.tokenizer, args.old)
        old_stats = diagnose(student, old_train, args.old, vocab, topk, distill_temp, device)
        print("\n=== OLD (one-hot) teacher vs current student ===")
        print(f"  student top-1 DISAGREES w/ teacher: {old_stats['disagree_rate']:.1%}")
        print(f"  teacher entropy at AGREEMENT pos  : {old_stats['mean_entropy_at_agreement']:.3f} nats   (~0 = no soft signal = pure SFT)")
        print(f"  mean KL(teacher || student)       : {old_stats['mean_kl_teacher_student']:.3f}")


if __name__ == "__main__":
    main()
