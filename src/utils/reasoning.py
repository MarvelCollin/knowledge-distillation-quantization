import ast
import re


SYSTEM_PROMPT = (
    "You are a reasoning coding assistant. First think step by step inside <think>...</think>. "
    "After </think>, output the final Python function inside ```python ... ``` and nothing else."
)

THINK_END_TAG = "</think>"

RETRY_SYSTEM_PROMPT = (
    "You are a reasoning coding assistant. First think step by step inside <think>...</think>. "
    "After </think>, output a single standalone Python function inside ```python ... ```. "
    "Do NOT wrap it in a class. Do NOT use class Solution. Output only the function."
)

RETRY_PROMPT_TEMPLATE = (
    "Write a solution in Python to solve the following problem.\n"
    "Your answer must be a standalone Python function (not a class method).\n"
    "Consider edge cases: empty inputs, single elements, negative values, large inputs.\n"
    "Read the constraints carefully.\n\n"
    "Problem: {text}\n\n"
)


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
        content += f"\n\nImplement this exact signature:\n```python\n{signature}\n    pass\n```"
    return content


def build_retry_prompt(problem: dict, tokenizer) -> str:
    base = RETRY_PROMPT_TEMPLATE.format(text=problem["text"])
    content = build_signature_user_content(base, problem.get("signature", ""))
    messages = [
        {"role": "system", "content": RETRY_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


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


def _normalize_indent(code: str) -> str:
    return code.replace("\t", "    ")


def extract_code(text: str) -> str:
    if not text:
        return ""
    text = _normalize_indent(strip_thinking(text)).strip()

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
        elif re.match(r'^(?:def |class |@)', line):
            result.append(line)
        else:
            break
    while result and not result[-1].strip():
        result.pop()
    return _unwrap_solution('\n'.join(result))
