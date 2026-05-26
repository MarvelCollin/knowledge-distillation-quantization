import ast
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


class _SelfStripper(ast.NodeTransformer):
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        return node


def _unwrap_solution(code: str) -> str:
    if not code.strip():
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

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
    return _unwrap_solution('\n'.join(result))
