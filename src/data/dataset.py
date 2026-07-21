import torch
from torch.utils.data import Dataset

from src.data.problems import build_user_content, load_problems
from src.data.short_cot import is_short_key, load_short_cot_responses, short_problem_idx
from src.data.teacher_cache import load_passing_responses, rescore_failed_cache
from src.utils.reasoning import SYSTEM_PROMPT


class CodingDataset(Dataset):
    def __init__(self, problems: list, tokenizer, max_length: int,
                 teacher_responses: dict, original_indices: list,
                 lengths: list = None):
        self.problems = problems
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.teacher_responses = teacher_responses
        self.original_indices = original_indices
        self.lengths = lengths or []
        self._items = self._pretokenize()

    def _pretokenize(self) -> list:
        eos_id = self.tokenizer.eos_token_id
        items = []
        for idx in range(len(self.problems)):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_content(self.problems[idx])},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            response = self.teacher_responses[self.original_indices[idx]]["text"]

            prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
            response_ids = self.tokenizer(response, add_special_tokens=False).input_ids
            if eos_id is not None:
                response_ids = response_ids + [eos_id]

            prompt_length = min(len(prompt_ids), self.max_length - 1)
            prompt_ids = prompt_ids[:prompt_length]
            response_ids = response_ids[:self.max_length - len(prompt_ids)]

            combined = prompt_ids + response_ids
            input_ids = torch.tensor(combined, dtype=torch.long)
            attention_mask = torch.ones(len(combined), dtype=torch.long)
            labels = input_ids.clone()
            labels[:prompt_length] = -100

            items.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "prompt_length": torch.tensor(prompt_length, dtype=torch.long),
                "idx": torch.tensor(idx, dtype=torch.long),
            })
        return items

    def __len__(self) -> int:
        return len(self.problems)

    def get_prompt(self, idx: int) -> str:
        return build_user_content(self.problems[idx])

    def get_chat_prompt(self, idx: int) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.get_prompt(idx)},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def get_reference(self, idx: int) -> str:
        return self.teacher_responses[self.original_indices[idx]]["text"]

    def get_test_cases(self, idx: int) -> list:
        return self.problems[idx]["test_cases"]

    def curriculum_order(self) -> list:
        if self.lengths:
            scored = [(length, i) for i, length in enumerate(self.lengths)]
        else:
            scored = [(self.teacher_responses[orig_idx]["token_count"], i)
                      for i, orig_idx in enumerate(self.original_indices)]
        scored.sort(key=lambda x: x[0])
        return [i for _, i in scored]

    def __getitem__(self, idx: int) -> dict:
        return self._items[idx]


def create_datasets(config: dict, tokenizer, cache_dir: str, rescore: bool = False,
                    require_passing: bool = True, include_short_cot: bool = False) -> tuple:
    max_length = config["student"]["max_length"]
    train_ratio = config["data"]["train_ratio"]

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    problems = load_problems(config)
    if rescore:
        recovered, tested, total_failed = rescore_failed_cache(cache_dir, problems)
        print(f"  Rescored teacher cache (pure KD — re-tests labels only, trajectories untouched): "
              f"recovered {recovered}/{total_failed} previously-failed responses now passing.")
    responses = load_passing_responses(cache_dir, len(problems), require_passing)
    n_teacher = len(responses)
    if include_short_cot:
        responses.update(load_short_cot_responses(config["short_cot"], tokenizer))

    def problem_of(key: int) -> dict:
        return problems[short_problem_idx(key)] if is_short_key(key) else problems[key]

    kept_indices = []
    lengths_map = {}
    teacher_deltas = []
    prompt_len_cache = {}
    for i in sorted(responses.keys()):
        problem_idx = short_problem_idx(i) if is_short_key(i) else i
        prompt_len = prompt_len_cache.get(problem_idx)
        if prompt_len is None:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_content(problems[problem_idx])},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_len = min(len(tokenizer(prompt, add_special_tokens=False).input_ids), max_length - 1)
            prompt_len_cache[problem_idx] = prompt_len
        student_resp_len = len(tokenizer(responses[i]["text"], add_special_tokens=False).input_ids)
        if not is_short_key(i):
            teacher_deltas.append(student_resp_len - responses[i]["token_count"])
        if prompt_len + student_resp_len + 1 <= max_length:
            kept_indices.append(i)
            lengths_map[i] = prompt_len + student_resp_len + 1

    dropped = len(responses) - len(kept_indices)
    teacher_kept = [i for i in kept_indices if not is_short_key(i)]
    short_kept = [i for i in kept_indices if is_short_key(i)]
    print(f"  Kept {len(kept_indices)} responses ({len(teacher_kept)} teacher, {len(short_kept)} "
          f"short-CoT) from cache ({dropped} dropped: prompt+response exceeds max_length={max_length}).")
    if teacher_deltas:
        absd = sorted(abs(x) for x in teacher_deltas)
        n = len(absd)
        median = absd[n // 2]
        drift = sum(1 for x in absd if x > 2)
        benign = n - drift
        print(f"  KD alignment: teacher/student length delta median |Δ|={median} tok, max={absd[-1]} "
              f"({benign}/{n} benign |Δ|≤2 e.g. eos, {drift} drift |Δ|>2).")
        if drift and drift / n > 0.05:
            print(f"  ⚠ {drift} teacher samples ({100 * drift / n:.1f}%) drift >2 tokens — KD may be "
                  f"positionally misaligned there; verify the student tokenizer matches the cache build.")

    split = int(len(teacher_kept) * train_ratio)
    val_keys = teacher_kept[split:]
    val_problem_idxs = set(val_keys)
    train_keys = teacher_kept[:split] + [
        k for k in short_kept if short_problem_idx(k) not in val_problem_idxs]
    if short_kept:
        n_short_train = len(train_keys) - split
        print(f"  Mix: {split} teacher + {n_short_train} short-CoT train samples "
              f"(ratio 1:{n_short_train / max(split, 1):.1f}), val stays {len(val_keys)} teacher-only.")
    return (
        CodingDataset([problem_of(i) for i in train_keys], tokenizer, max_length,
                      responses, train_keys,
                      [lengths_map[i] for i in train_keys]),
        CodingDataset([problem_of(i) for i in val_keys], tokenizer, max_length,
                      responses, val_keys,
                      [lengths_map[i] for i in val_keys]),
    )


def train_split_indices(config: dict, tokenizer, cache_dir: str) -> tuple:
    train_dataset, _ = create_datasets(config, tokenizer, cache_dir)
    return load_problems(config), list(train_dataset.original_indices)
