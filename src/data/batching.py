import random

import numpy as np
import torch
from torch.utils.data import Sampler


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_pad_collate(pad_id: int):
    def pad_collate(items):
        max_len = max(it["input_ids"].size(0) for it in items)
        input_ids = torch.full((len(items), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(items), max_len), dtype=torch.long)
        labels = torch.full((len(items), max_len), -100, dtype=torch.long)
        for r, it in enumerate(items):
            n = it["input_ids"].size(0)
            input_ids[r, :n] = it["input_ids"]
            attention_mask[r, :n] = it["attention_mask"]
            labels[r, :n] = it["labels"]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_length": torch.stack([it["prompt_length"] for it in items]),
            "idx": torch.stack([it["idx"] for it in items]),
        }
    return pad_collate


def pack_by_budget(order: list, lengths: list, token_budget: int, max_batch: int) -> list:
    batches = []
    cur, cur_max = [], 0
    for i in order:
        length = lengths[i]
        new_max = max(cur_max, length)
        if cur and (len(cur) + 1 > max_batch or new_max * (len(cur) + 1) > token_budget):
            batches.append(cur)
            cur, cur_max = [i], length
        else:
            cur.append(i)
            cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


class FixedBatchSampler(Sampler):
    def __init__(self, batches: list):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)
