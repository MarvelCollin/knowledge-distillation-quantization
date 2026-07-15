import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoConfig, AutoTokenizer

from src.config import load_config
from src.data.dataset import create_datasets
from src.distillation.teacher_topk import preload_teacher_topk


def main():
    config = load_config("config/config.yaml")
    onpolicy_dir = config["data"]["onpolicy_cache_dir"]
    model_name = config["student"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    vocab_size = AutoConfig.from_pretrained(model_name, trust_remote_code=True).vocab_size

    train_ds, val_ds = create_datasets(config, tokenizer, onpolicy_dir, require_passing=False)
    print(f"on-policy datasets: train={len(train_ds)} val={len(val_ds)}")
    print(f"train indices: {list(train_ds.original_indices)}")

    topk = config["teacher"]["top_logprobs"] + 1
    teacher_topk = preload_teacher_topk(
        onpolicy_dir, train_ds.original_indices, vocab_size, topk,
        config["training"]["distill_temperature"])
    print(f"teacher top-k loaded for {len(teacher_topk)}/{len(train_ds)} train samples")

    ids, probs = next(iter(teacher_topk.values()))
    nonzero = int((probs.sum(dim=-1) > 0).sum().item())
    print(f"teacher_probs shape={tuple(probs.shape)} nonzero_positions={nonzero}")

    item = train_ds[0]
    resp = int((item["labels"] != -100).sum().item())
    print(f"dataset item input={tuple(item['input_ids'].shape)} response_tokens={resp}")
    print("on-policy data path OK")


if __name__ == "__main__":
    main()
