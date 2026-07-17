import json
from pathlib import Path

SHORT_KEY_BASE = 1_000_000
SHORT_KEY_STRIDE = 100

SHORT_COT_SYSTEM_PROMPT = (
    "You are a coding assistant. Think briefly inside <think>...</think> using at most a few "
    "short sentences. After </think>, output the final Python function inside "
    "```python ... ``` and nothing else."
)


def short_key(problem_idx: int, sample_idx: int) -> int:
    return SHORT_KEY_BASE + problem_idx * SHORT_KEY_STRIDE + sample_idx


def is_short_key(key: int) -> bool:
    return key >= SHORT_KEY_BASE


def short_problem_idx(key: int) -> int:
    return (key - SHORT_KEY_BASE) // SHORT_KEY_STRIDE


def load_short_cot_responses(cache_dir: str, tokenizer) -> dict:
    responses = {}
    cache_path = Path(cache_dir)
    if not cache_path.is_dir():
        return responses
    for f in cache_path.glob("[0-9]*.json"):
        problem_idx = int(f.stem)
        data = json.loads(f.read_text())
        for j, text in enumerate(data["texts"]):
            ids = tokenizer(text, add_special_tokens=False).input_ids
            responses[short_key(problem_idx, j)] = {"text": text, "token_count": len(ids)}
    return responses
