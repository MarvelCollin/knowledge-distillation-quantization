import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.evaluation.evaluator import run_test_cases
from src.utils.reasoning import THINK_END_TAG, extract_code

PASSED_PATTERN = re.compile(rb'"test_passed":\s*(\d+)')
TOTAL_PATTERN = re.compile(rb'"test_total":\s*(\d+)')

_CODE_FENCE_OPEN = "```python"
_CODE_FENCE_CLOSE = "```"
_BRIDGE_TOKEN = "\n"


def fully_passed(passed, total) -> bool:
    return total is not None and total > 0 and passed == total


def tail_counts(path, size: int = 400) -> tuple:
    p = Path(path)
    if not p.exists():
        return None, None
    try:
        with open(p, "rb") as fh:
            end = fh.seek(0, 2)
            fh.seek(max(0, end - size))
            tail = fh.read()
    except Exception:
        return None, None
    mp = PASSED_PATTERN.search(tail)
    mt = TOTAL_PATTERN.search(tail)
    return (int(mp.group(1)) if mp else None, int(mt.group(1)) if mt else None)


def tail_fully_passed(path, size: int = 400) -> bool:
    return fully_passed(*tail_counts(path, size))


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


def load_cached(cache_dir: str, idx: int) -> dict | None:
    file_path = Path(cache_dir) / f"{idx}.json"
    if not file_path.exists():
        return None
    with open(file_path) as f:
        data = json.load(f)
    return clean_teacher_cache(data)


def load_passing_responses(cache_dir: str, num_problems: int) -> dict:
    cache_path = Path(cache_dir)
    passing = {}
    for i in range(num_problems):
        f = cache_path / f"{i}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        if not fully_passed(d.get("test_passed"), d.get("test_total")):
            continue
        d = clean_teacher_cache(d)
        tokens = d.get("tokens") or []
        text = d.get("text") or ""
        if not tokens or not text:
            continue
        passing[i] = {"text": text, "token_count": len(tokens)}
    return passing


def failed_cache_indices(cache_dir: str, num_problems: int) -> list:
    cache_path = Path(cache_dir)
    failed = []
    for i in range(num_problems):
        f = cache_path / f"{i}.json"
        if not f.exists():
            continue
        if not tail_fully_passed(f):
            failed.append(i)
    return failed


def rescore_failed_cache(cache_dir: str, problems: list, apply: bool = True,
                         sample: int = 0, seed: int = 0, max_workers: int = 0,
                         timeout: float = 5.0) -> tuple:
    if max_workers <= 0:
        max_workers = min(os.cpu_count() or 16, 48)
    cache_path = Path(cache_dir)
    failed = failed_cache_indices(cache_dir, len(problems))
    total_failed = len(failed)
    if sample > 0:
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
