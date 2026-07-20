import json
from itertools import zip_longest
from pathlib import Path

SHORT_KEY_BASE = 1_000_000
SHORT_KEY_STRIDE = 100

SHORT_COT_SYSTEM_PROMPT = (
    "You are a coding assistant. Think briefly inside <think>...</think> using at most a few "
    "short sentences. After </think>, output the final Python function inside "
    "```python ... ``` and nothing else."
)

EDGE_COT_SYSTEM_PROMPT = (
    "You are a coding assistant. Think briefly inside <think>...</think> about the edge cases: "
    "empty inputs, single elements, duplicates, extreme values, and inputs at the constraint "
    "limits. After </think>, output the final Python function that handles all of them inside "
    "```python ... ``` and nothing else."
)


def short_key(problem_idx: int, sample_idx: int) -> int:
    return SHORT_KEY_BASE + problem_idx * SHORT_KEY_STRIDE + sample_idx


def is_short_key(key: int) -> bool:
    return key >= SHORT_KEY_BASE


def short_problem_idx(key: int) -> int:
    return (key - SHORT_KEY_BASE) // SHORT_KEY_STRIDE


def _texts_by_problem(cache_dir: str) -> dict:
    texts = {}
    cache_path = Path(cache_dir)
    if not cache_path.is_dir():
        return texts
    for f in cache_path.glob("[0-9]*.json"):
        data = json.loads(f.read_text())
        if data["texts"]:
            texts[int(f.stem)] = data["texts"]
    return texts


def _interleave(waves: list) -> list:
    merged = []
    for row in zip_longest(*waves):
        merged.extend(t for t in row if t is not None)
    return merged


def load_short_cot_responses(sc_config: dict, tokenizer) -> dict:
    waves = [_texts_by_problem(sc_config["cache_dir"]),
             _texts_by_problem(sc_config["edge_cache_dir"])]
    cap = sc_config["max_per_problem"]
    responses = {}
    for problem_idx in sorted(set().union(*waves)):
        merged = _interleave([w.get(problem_idx, []) for w in waves])[:cap]
        for j, text in enumerate(merged):
            ids = tokenizer(text, add_special_tokens=False).input_ids
            responses[short_key(problem_idx, j)] = {"text": text, "token_count": len(ids)}
    return responses
