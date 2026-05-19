import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class CodeDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def get_prompt(self, idx: int) -> str:
        sample = self.samples[idx]
        return f"### Instruction:\n{sample['instruction']}\n\n### Response:\n"

    def get_reference(self, idx: int) -> str:
        return self.samples[idx]["output"]

    def __getitem__(self, idx: int) -> dict:
        prompt = self.get_prompt(idx)
        response = self.get_reference(idx)
        full_text = prompt + response

        full_enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        prompt_ids = self.tokenizer.encode(response, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=True)
        prompt_length = max(0, min(len(full_ids) - len(prompt_ids), self.max_length - 1))

        input_ids = full_enc.input_ids.squeeze(0)
        attention_mask = full_enc.attention_mask.squeeze(0)

        labels = input_ids.clone()
        labels[:prompt_length] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_length": torch.tensor(prompt_length, dtype=torch.long),
            "idx": torch.tensor(idx, dtype=torch.long),
        }


def create_datasets(config: dict, tokenizer) -> tuple:
    raw = load_dataset(config["data"]["dataset_name"], split="train")
    max_samples = config["data"]["max_samples"]
    raw = raw.select(range(min(max_samples, len(raw))))

    train_size = int(len(raw) * config["data"]["train_ratio"])
    train_raw = raw.select(range(train_size))
    val_raw = raw.select(range(train_size, len(raw)))

    max_length = config["student"]["max_length"]
    return CodeDataset(train_raw, tokenizer, max_length), CodeDataset(val_raw, tokenizer, max_length)
