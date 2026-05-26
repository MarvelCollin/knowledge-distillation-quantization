import re

SYSTEM_PROMPT = (
    "You are a reasoning coding assistant. First think step by step inside <think>...</think>. "
    "After </think>, output the final Python function inside ```python ... ``` and nothing else."
)

THINK_END_TAG = "</think>"


def strip_thinking(text: str) -> str:
    if not text:
        return ""
    idx = text.find(THINK_END_TAG)
    if idx < 0:
        return text
    return text[idx + len(THINK_END_TAG):].lstrip()


def extract_code(text: str) -> str:
    if not text:
        return ""
    text = strip_thinking(text).strip()

    fence = re.search(r'```(?:python)?\s*\n?(.*?)(?:```|$)', text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    m = re.search(r'def \w', text)
    if m:
        text = text[m.start():]
    elif not text.startswith("def "):
        text = "def " + text.lstrip()

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
    return '\n'.join(result)
