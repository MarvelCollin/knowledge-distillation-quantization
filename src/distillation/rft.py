import json
from pathlib import Path

from src.evaluation.evaluator import run_test_cases
from src.utils.reasoning import THINK_END_TAG, extract_code


def read_entry(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def is_passing(entry):
    tp, tt = entry.get("test_passed"), entry.get("test_total")
    return bool(tt) and tp == tt


def solves(code, test_cases):
    if not code or not test_cases:
        return False
    r = run_test_cases(code, test_cases)
    return r["total"] > 0 and r["passed"] == r["total"]


def first_passing(candidates, test_cases):
    for res in candidates:
        if solves(extract_code(res["text"]), test_cases):
            return res
    return None


def tokenize_pieces(tokenizer, text):
    return [tokenizer.decode([i]) for i in tokenizer.encode(text, add_special_tokens=False)]


def assemble_solution_text(raw, code):
    if THINK_END_TAG in raw:
        reasoning = raw.split(THINK_END_TAG)[0].rstrip()
        return f"{reasoning}\n{THINK_END_TAG}\n```python\n{code}\n```"
    return f"```python\n{code}\n```"


def sft_payload(tokenizer, prompt, text, n_tests, max_tokens, source="rft_student"):
    return {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "source": source,
        "text": text,
        "tokens": tokenize_pieces(tokenizer, text),
        "logprobs": [],
        "top_k_ids": None,
        "top_k_vals": None,
        "test_passed": n_tests,
        "test_total": n_tests,
    }


def kd_payload(prompt, res, n_tests, max_tokens, source="rft_teacher"):
    return {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "source": source,
        "text": res["text"],
        "tokens": res["tokens"],
        "logprobs": res.get("logprobs", []),
        "top_k_ids": res.get("top_k_ids"),
        "top_k_vals": res.get("top_k_vals"),
        "test_passed": n_tests,
        "test_total": n_tests,
    }


def persist(out_cache, idx, payload):
    Path(out_cache, f"{idx}.json").write_text(json.dumps(payload))
