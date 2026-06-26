import ast, json, glob
from pathlib import Path
from src.config import load_config
from src.data.dataset import load_problems
from src.evaluation.evaluator import _alias_candidate, _prepare_assertion
from src.utils.reasoning import extract_code

config = load_config("config/config.yaml")
cache_dir = config["data"]["teacher_cache_dir"]
problems = load_problems(config)
files = sorted(glob.glob(f"{cache_dir}/*.json"), key=lambda p: int(Path(p).stem))

def defines(code, name):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return True
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return True
    return False

examples = []
n_missing = 0
for f in files:
    idx = int(Path(f).stem)
    d = json.loads(open(f).read())
    tp, tt = d.get("test_passed"), d.get("test_total")
    if not tt or tp == tt or idx >= len(problems):
        continue
    cap = d.get("max_tokens") or 0
    n = len(d.get("tokens") or [])
    if cap and n >= cap - 50:
        continue
    tcs = problems[idx]["test_cases"]
    cand, _ = _prepare_assertion(tcs[0]) if tcs else (None, "")
    if not cand:
        continue
    raw = d.get("text") or ""
    code = extract_code(raw)
    aliased = _alias_candidate(code, cand)
    if defines(aliased, cand):
        continue
    n_missing += 1
    if len(examples) < 10:
        try:
            tree = ast.parse(code) if code else None
            top = [type(x).__name__ + (":" + getattr(x, "name", "") if hasattr(x, "name") else "") for x in (tree.body if tree else [])]
        except SyntaxError:
            top = ["<SYNTAX ERROR>"]
        examples.append((idx, cand, problems[idx].get("entry_point"), bool(code), top, code[:600]))

print(f"static missing-candidate count (non-truncated failed): {n_missing}")
for idx, cand, ep, has_code, top, snippet in examples:
    print("\n" + "=" * 70)
    print(f"idx={idx} candidate={cand!r} entry_point={ep!r} extract_nonempty={has_code}")
    print(f"top-level nodes: {top}")
    print("--- code[:600] ---")
    print(snippet)
