import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

PROMPT_TEMPLATE = (
    "Write a solution in Python to solve the following problem.\n"
    "Your answer must be a Python function only. Do not use any other language.\n\n"
    "Problem: {text}\n\n"
)


def _extract_test_cases(test_str: str, entry_point: str) -> list:
    if not test_str.strip() or not entry_point:
        return []
    return [test_str + f"\ncheck({entry_point})"]


class CodingDataset(Dataset):
    def __init__(self, problems: list, tokenizer, max_length: int):
        self.problems = problems
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.problems)

    def get_prompt(self, idx: int) -> str:
        return PROMPT_TEMPLATE.format(text=self.problems[idx]["text"])

    def get_reference(self, idx: int) -> str:
        return self.problems[idx]["code"]

    def get_test_cases(self, idx: int) -> list:
        return self.problems[idx]["test_cases"]

    def curriculum_order(self, cache_dir: str) -> list:
        cache_path = Path(cache_dir)
        scored = []
        for i in range(len(self.problems)):
            difficulty = None
            f = cache_path / f"{i}.json"
            if f.exists():
                try:
                    d = json.loads(f.read_text())
                    toks = d.get("tokens") or []
                    if toks:
                        difficulty = len(toks)
                except Exception:
                    difficulty = None
            if difficulty is None:
                difficulty = len(self.problems[i]["text"])
            scored.append((difficulty, i))
        scored.sort(key=lambda x: x[0])
        return [i for _, i in scored]

    def __getitem__(self, idx: int) -> dict:
        prompt = self.get_prompt(idx)
        solution = self.get_reference(idx)
        full_text = prompt + solution

        full_enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        prompt_length = min(len(prompt_ids), self.max_length - 1)

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
    dataset_name = config["data"]["dataset_name"]
    max_samples = config["data"]["max_samples"]
    train_ratio = config["data"]["train_ratio"]
    max_length = config["student"]["max_length"]

    print(f"Loading {dataset_name}...")
    raw = load_dataset(dataset_name, split="train")
    raw = raw.select(range(min(max_samples, len(raw))))

    problems = []
    for item in raw:
        entry_point = (item.get("entry_point") or "").strip()
        ep_clean = re.search(r'(\w+)$', entry_point)
        entry_point = ep_clean.group(1) if ep_clean else entry_point
        test_cases = _extract_test_cases(item.get("test") or "", entry_point)
        code = (item.get("response") or "").strip()
        text = (item.get("problem_description") or "").strip()
        if not test_cases or not code or not text:
            continue
        problems.append({
            "text": text,
            "code": code,
            "test_cases": test_cases,
        })

    print(f"  Loaded {len(problems)} problems with test cases.")
    split = int(len(problems) * train_ratio)
    return (
        CodingDataset(problems[:split], tokenizer, max_length),
        CodingDataset(problems[split:], tokenizer, max_length),
    )
