CODE_ONLY_SYSTEM_PROMPT = (
    "You are a coding assistant. Output ONLY a raw Python function definition. "
    "No explanation, no markdown, no triple backticks. Start directly with def."
)

REASONING_SYSTEM_PROMPT = (
    "You are a reasoning coding assistant. First think step by step inside <think>...</think>. "
    "After </think>, output the final Python function inside ```python ... ``` and nothing else."
)


def strip_thinking(text: str, end_tag: str = "</think>") -> str:
    if not text or not end_tag:
        return text or ""
    idx = text.find(end_tag)
    if idx < 0:
        return text
    return text[idx + len(end_tag):].lstrip()


def system_prompt(reasoning_enabled: bool) -> str:
    return REASONING_SYSTEM_PROMPT if reasoning_enabled else CODE_ONLY_SYSTEM_PROMPT
