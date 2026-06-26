import json, glob
from pathlib import Path
from src.config import load_config
from src.data.dataset import load_problems
from src.evaluation.evaluator import execute_assertion, _classify, _alias_candidate, _prepare_assertion
from src.utils.reasoning import extract_code

config = load_config("config/config.yaml")
cache_dir = config["data"]["teacher_cache_dir"]
problems = load_problems(config)

files = sorted(glob.glob(f"{cache_dir}/*.json"), key=lambda p: int(Path(p).stem))
toks = []
cached_caps = set()
missing_examples = []

for f in files:
    idx = int(Path(f).stem)
    d = json.loads(open(f).read())
    n = len(d.get("tokens") or [])
    toks.append(n)
    cached_caps.add(d.get("max_tokens"))
    tp, tt = d.get("test_passed"), d.get("test_total")
    if not tt or tp == tt:
        continue
    cap = d.get("max_tokens") or 0
    if cap and n >= cap - 50:
        continue
    if idx >= len(problems):
        continue
    code = extract_code(d.get("text") or "")
    tcs = problems[idx]["test_cases"]
    for assertion in tcs:
        outcome, error = execute_assertion(code, assertion)
        if outcome != "pass":
            cat = _classify(outcome, error)
            if cat == "missing_function" and len(missing_examples) < 8:
                cand, _ = _prepare_assertion(assertion)
                aliased = _alias_candidate(code, cand)
                missing_examples.append((idx, problems[idx]["entry_point"], cand, code, aliased, error))
            break

toks.sort()
print("=== cached max_tokens values present:", cached_caps)
print(f"=== token length: p50={toks[len(toks)//2]} p90={toks[int(len(toks)*0.9)]} "
      f"p95={toks[int(len(toks)*0.95)]} p99={toks[int(len(toks)*0.99)]} max={toks[-1]}")

for idx, ep, cand, code, aliased, error in missing_examples:
    print("\n" + "=" * 70)
    print(f"idx={idx}  entry_point={ep!r}  candidate(from test)={cand!r}")
    print(f"alias changed code? {'YES' if aliased != code else 'NO'}")
    print("--- extracted code (first 25 lines) ---")
    print("\n".join(code.splitlines()[:25]))
    print("--- error ---")
    print(" ".join((error or "").split())[:200])
