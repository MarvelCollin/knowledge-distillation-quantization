import ast
import re

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


SYSTEM_PROMPT = (
    "You are a reasoning coding assistant. First think step by step inside <think>...</think>. "
    "After </think>, output the final Python function inside ```python ... ``` and nothing else."
)

THINK_END_TAG = "</think>"
THINK_BUDGET_RATIO = 0.75


def _build_code_primer(has_think_end: bool, signature: str) -> str:
    head = "\n```python\n" if has_think_end else "\n</think>\n```python\n"
    if signature:
        return f"{head}{signature}\n    "
    return head


class _StopOnSequence(StoppingCriteria):
    def __init__(self, sequence_ids: list, prompt_len: int):
        self.sequence_ids = sequence_ids
        self.prompt_len = prompt_len
        self.seq_len = len(sequence_ids)

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if input_ids.shape[1] - self.prompt_len < self.seq_len:
            return False
        tail = input_ids[0, -self.seq_len:].tolist()
        return tail == self.sequence_ids


def generate_with_thinking_cap(model, tokenizer, prompt_text: str,
                                max_new_tokens: int,
                                code_primer_signature: str = "",
                                **gen_kwargs) -> tuple[str, bool]:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    eos_id = tokenizer.eos_token_id

    think_budget = max(1, int(max_new_tokens * THINK_BUDGET_RATIO))
    code_budget = max(1, max_new_tokens - think_budget)

    think_end_ids = tokenizer.encode(THINK_END_TAG, add_special_tokens=False)
    stop = StoppingCriteriaList([_StopOnSequence(think_end_ids, prompt_len)])

    with torch.no_grad():
        phase1 = model.generate(
            **inputs,
            max_new_tokens=think_budget,
            pad_token_id=eos_id,
            stopping_criteria=stop,
            **gen_kwargs,
        )

    phase1_gen = phase1[0][prompt_len:]
    if phase1_gen.numel() > 0 and phase1_gen[-1].item() == eos_id:
        return tokenizer.decode(phase1_gen, skip_special_tokens=True), False

    phase1_text = tokenizer.decode(phase1_gen, skip_special_tokens=False)
    primer = _build_code_primer(THINK_END_TAG in phase1_text, code_primer_signature)
    suffix_ids = tokenizer.encode(primer, add_special_tokens=False, return_tensors="pt").to(model.device)
    primed = torch.cat([phase1, suffix_ids], dim=1)
    primed_len = primed.shape[1]
    attn = torch.ones_like(primed)

    with torch.no_grad():
        phase2 = model.generate(
            primed,
            attention_mask=attn,
            max_new_tokens=code_budget,
            pad_token_id=eos_id,
            **gen_kwargs,
        )

    gen_tokens = phase2[0][prompt_len:]
    truncated = (phase2.shape[1] - primed_len) >= code_budget
    raw = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return raw, truncated


def extract_fn_name(test_cases: list) -> str | None:
    for tc in test_cases:
        for name in re.findall(r'\bcheck\((\w+)\)', tc):
            if name != 'candidate':
                return name
        m = re.search(r'\bassert\s+(\w+)\s*\(', tc)
        if m and m.group(1) != 'candidate':
            return m.group(1)
    return None


def build_signature_user_content(base_prompt: str, signature: str) -> str:
    content = base_prompt.rstrip()
    if signature:
        content += f"\n\nImplement this exact signature:\n```python\n{signature}\n    ...\n```"
    return content


def strip_thinking(text: str) -> str:
    if not text:
        return ""
    idx = text.find(THINK_END_TAG)
    if idx < 0:
        return text
    return text[idx + len(THINK_END_TAG):].lstrip()


class _SelfStripper(ast.NodeTransformer):
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        return node


def _largest_parseable_prefix(code: str) -> str:
    lines = code.split('\n')
    end = len(lines)
    while end > 0:
        candidate = '\n'.join(lines[:end])
        try:
            ast.parse(candidate)
        except SyntaxError as exc:
            nxt = (exc.lineno - 1) if exc.lineno else end - 1
            end = nxt if 0 < nxt < end else end - 1
            continue
        return candidate.rstrip()
    return ""


def _unwrap_solution(code: str) -> str:
    if not code.strip():
        return code
    code = _largest_parseable_prefix(code)
    if not code:
        return ""
    tree = ast.parse(code)

    if not any(isinstance(n, ast.ClassDef) and n.name == "Solution" for n in tree.body):
        return code

    stripper = _SelfStripper()
    new_body = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == "__init__":
                        continue
                    if item.args.args and item.args.args[0].arg == "self":
                        item.args.args = item.args.args[1:]
                    new_body.append(stripper.visit(item))
                elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                    new_body.append(item)
        else:
            new_body.append(node)

    tree.body = new_body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def extract_code(text: str) -> str:
    if not text:
        return ""
    text = strip_thinking(text).strip()

    fence = re.search(r'```(?:python)?\s*\n?(.*?)(?:```|$)', text, re.DOTALL | re.IGNORECASE)
    if fence:
        code = fence.group(1).strip()
        m = re.search(r'^(?:class|def) \w', code, re.MULTILINE)
        if m:
            return _unwrap_solution(code[m.start():])
        return _unwrap_solution(code)

    m = re.search(r'^(?:def|class) \w', text, re.MULTILINE)
    if not m:
        return ""
    text = text[m.start():]

    lines = text.split('\n')
    result = []
    for line in lines:
        if not result:
            result.append(line)
        elif not line or line[0] in (' ', '\t'):
            result.append(line)
        else:
            break
    while result and not result[-1].strip():
        result.pop()
    return _unwrap_solution('\n'.join(result))
