import ast
import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from src.utils.reasoning import SYSTEM_PROMPT, THINK_END_TAG, build_signature_user_content, extract_code
from src.evaluation.evaluator import extract_signature, run_test_cases

PROMPT_TEMPLATE = (
    "Write a solution in Python to solve the following problem.\n"
    "Your answer must be a Python function only. Do not use any other language.\n\n"
    "Problem: {text}\n\n"
)

_CODE_FENCE_OPEN = "```python"
_CODE_FENCE_CLOSE = "```"
_BRIDGE_TOKEN = "\n"


def build_user_content(problem: dict) -> str:
    base = PROMPT_TEMPLATE.format(text=problem["text"])
    return build_signature_user_content(base, problem.get("signature", ""))


def _extract_test_cases(test_str: str, entry_point: str) -> list:
    if not test_str.strip() or not entry_point:
        return []
    tree = ast.parse(test_str)
    check_func = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "check"
    )
    other_top_level = [n for n in tree.body if n is not check_func]
    cases = []
    cases = []
    for stmt in check_func.body:
        single_check = ast.FunctionDef(
            name="check",
            args=check_func.args,
            body=[stmt],
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        module = ast.Module(body=other_top_level + [single_check], type_ignores=[])
        ast.fix_missing_locations(module)
        src = ast.unparse(module)
        cases.append(src + f"\ncheck({entry_point})")
    return cases


def clean_teacher_cache(cached: dict) -> dict:
    tokens = cached.get("tokens") or []
    if not tokens:
        return cached
    text = "".join(tokens)

    fence_open = text.find(_CODE_FENCE_OPEN)
    if fence_open < 0:
        return cached
    fence_close = text.find(_CODE_FENCE_CLOSE, fence_open + len(_CODE_FENCE_OPEN))
    if fence_close < 0:
        return cached
    fence_close_end = fence_close + len(_CODE_FENCE_CLOSE)

    close_pos = text.find(THINK_END_TAG)
    use_think_path = 0 <= close_pos < fence_open
    close_end = close_pos + len(THINK_END_TAG) if use_think_path else None

    keep_end = code_start = code_end = None
    char_pos = 0
    for i, tok in enumerate(tokens):
        if use_think_path and keep_end is None and char_pos >= close_end:
            keep_end = i
        if code_start is None and char_pos >= fence_open:
            code_start = i
        if code_end is None and char_pos >= fence_close_end:
            code_end = i
            break
        char_pos += len(tok)
    if code_end is None:
        code_end = len(tokens)
    if code_start is None:
        return cached

    logprobs = cached.get("logprobs") or []
    top_ids = cached.get("top_k_ids")
    top_vals = cached.get("top_k_vals")

    if use_think_path and keep_end is not None and keep_end < code_start:
        new_tokens = tokens[:keep_end] + [_BRIDGE_TOKEN] + tokens[code_start:code_end]
        new_logprobs = logprobs[:keep_end] + [{}] + logprobs[code_start:code_end]
        new_top_ids = (
            list(top_ids[:keep_end]) + [[]] + list(top_ids[code_start:code_end])
            if top_ids is not None else None
        )
        new_top_vals = (
            list(top_vals[:keep_end]) + [[]] + list(top_vals[code_start:code_end])
            if top_vals is not None else None
        )
    else:
        new_tokens = tokens[code_start:code_end]
        new_logprobs = logprobs[code_start:code_end]
        new_top_ids = list(top_ids[code_start:code_end]) if top_ids is not None else None
        new_top_vals = list(top_vals[code_start:code_end]) if top_vals is not None else None

    cleaned = dict(cached)
    cleaned["tokens"] = new_tokens
    cleaned["logprobs"] = new_logprobs
    cleaned["text"] = "".join(new_tokens)
    if new_top_ids is not None:
        cleaned["top_k_ids"] = new_top_ids
    if new_top_vals is not None:
        cleaned["top_k_vals"] = new_top_vals
    return cleaned


def _load_passing_indices(cache_dir: str, num_problems: int) -> dict:
    cache_path = Path(cache_dir)
    passing = {}
    for i in range(num_problems):
        f = cache_path / f"{i}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        tp = d.get("test_passed")
        tt = d.get("test_total")
        if tt is None or tp is None or tt == 0 or tp < tt:
            continue
        d = clean_teacher_cache(d)
        tokens = d.get("tokens") or []
        text = d.get("text") or ""
        if not tokens or not text:
            continue
        passing[i] = {"text": text, "token_count": len(tokens)}
    return passing


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


def load_problems(config: dict) -> list:
    dataset_name = config["data"]["dataset_name"]
    max_samples = config["data"]["max_samples"]

    cache_file = Path("cache") / f"problems_{dataset_name.replace('/', '_')}_{max_samples}.json"
    src_files = [
        Path(__file__),
        Path(__file__).parents[1] / "utils" / "reasoning.py",
        Path(__file__).parents[1] / "evaluation" / "evaluator.py",
    ]
    src_mtime = max(f.stat().st_mtime for f in src_files)
    if cache_file.exists() and cache_file.stat().st_mtime >= src_mtime:
        problems = json.loads(cache_file.read_text())
        print(f"Loaded {len(problems)} problems from processed cache ({cache_file.name}).")
        return problems

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
        signature = extract_signature(test_cases[0], entry_point) if test_cases else ""
        problems.append({
            "text": text,
            "code": code,
            "test_cases": test_cases,
            "entry_point": entry_point,
            "signature": signature,
            "difficulty": (item.get("difficulty") or "").strip(),
        })

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(problems))
    print(f"  Loaded {len(problems)} problems with test cases (processed cache written).")
    return problems


def _failed_cache_indices(cache_dir: str, num_problems: int) -> list:
    pp = re.compile(rb'"test_passed":\s*(\d+)')
    tt = re.compile(rb'"test_total":\s*(\d+)')
    cache_path = Path(cache_dir)
    failed = []
    for i in range(num_problems):
        f = cache_path / f"{i}.json"
        if not f.exists():
            continue
        with open(f, "rb") as fh:
            fh.seek(max(0, f.stat().st_size - 400))
            tail = fh.read()
        a, b = pp.search(tail), tt.search(tail)
        if not (a and b and int(b.group(1)) > 0 and int(a.group(1)) == int(b.group(1))):
            failed.append(i)
    return failed


def rescore_failed_cache(cache_dir: str, problems: list, apply: bool = True,
                         sample: int = 0, seed: int = 0, max_workers: int = 0,
                         timeout: float = 5.0) -> tuple:
    import os
    from concurrent.futures import ThreadPoolExecutor

    if max_workers <= 0:
        max_workers = min(os.cpu_count() or 16, 48)
    cache_path = Path(cache_dir)
    failed = _failed_cache_indices(cache_dir, len(problems))
    total_failed = len(failed)
    if sample > 0:
        import random
        random.seed(seed)
        failed = random.sample(failed, min(sample, len(failed)))

    def _recheck(i: int) -> bool:
        f = cache_path / f"{i}.json"
        d = json.loads(f.read_text())
        code = extract_code(d.get("text", ""))
        r = run_test_cases(code, problems[i]["test_cases"], timeout=timeout)
        if r["total"] > 0 and r["passed"] == r["total"]:
            if apply:
                d["test_passed"] = r["total"]
                d["test_total"] = r["total"]
                f.write_text(json.dumps(d))
            return True
        return False

    recovered = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for ok in ex.map(_recheck, failed):
            recovered += int(ok)
    return recovered, len(failed), total_failed


def create_datasets(config: dict, tokenizer, cache_dir: str, rescore: bool = False) -> tuple:
    max_length = config["student"]["max_length"]
    train_ratio = config["data"]["train_ratio"]

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    problems = load_problems(config)
    if rescore:
        recovered, tested, total_failed = rescore_failed_cache(cache_dir, problems)
        print(f"  Rescored teacher cache (pure KD — re-tests labels only, trajectories untouched): "
              f"recovered {recovered}/{total_failed} previously-failed responses now passing.")
    teacher_responses = _load_passing_indices(cache_dir, len(problems))

    kept_indices = []
    lengths_map = {}
    misaligned = 0
    for i in sorted(teacher_responses.keys()):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_content(problems[i])},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_len = min(len(tokenizer(prompt, add_special_tokens=False).input_ids), max_length - 1)
        student_resp_len = len(tokenizer(teacher_responses[i]["text"], add_special_tokens=False).input_ids)
        if student_resp_len != teacher_responses[i]["token_count"]:
            misaligned += 1
        if prompt_len + student_resp_len + 1 <= max_length:
            kept_indices.append(i)
            lengths_map[i] = prompt_len + student_resp_len + 1

    dropped = len(teacher_responses) - len(kept_indices)
    kept_problems = [problems[i] for i in kept_indices]
    print(f"  Kept {len(kept_problems)} problems with passing teacher responses from cache "
          f"({dropped} dropped: prompt+response exceeds max_length={max_length}).")
    if misaligned:
        pct = misaligned / max(len(teacher_responses), 1) * 100
        print(f"  ⚠ Teacher/student token-count mismatch on {misaligned}/{len(teacher_responses)} "
              f"({pct:.1f}%) responses — teacher logprobs may be positionally misaligned with the "
              f"student tokens on those samples.")

    split = int(len(kept_problems) * train_ratio)
    return (
        CodingDataset(kept_problems[:split], tokenizer, max_length,
                      teacher_responses, kept_indices[:split],
                      [lengths_map[i] for i in kept_indices[:split]]),
        CodingDataset(kept_problems[split:], tokenizer, max_length,
                      teacher_responses, kept_indices[split:],
                      [lengths_map[i] for i in kept_indices[split:]]),
    )
