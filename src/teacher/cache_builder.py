import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Thread

from src.data.problems import DIFFICULTY_ORDER
from src.data.teacher_cache import PASSED_PATTERN, TOTAL_PATTERN, fully_passed, tail_fully_passed
from src.evaluation.evaluator import run_test_cases
from src.utils.gpu import gpu_used_gb
from src.utils.reasoning import extract_code
from src.utils.runlog import RunLog, eta_str


def _score_result(result: dict, test_cases: list) -> dict:
    try:
        return run_test_cases(extract_code(result.get("text", "")), test_cases)
    except Exception as exc:
        print(f"  test exec error: {type(exc).__name__}: {exc}")
        return {"passed": None, "total": None, "details": [], "errors": [], "categories": []}


def _dominant_cause(score: dict) -> str:
    non_pass = [c for c in score.get("categories", []) if c != "pass"]
    return Counter(non_pass).most_common(1)[0][0] if non_pass else "unknown"


def _is_valid_cache(cache_path: Path, idx: int, prompt: str,
                    max_tokens: int, rejection_samples: int) -> bool:
    f = cache_path / f"{idx}.json"
    if not f.exists():
        return False
    try:
        prefix = ('{"prompt": ' + json.dumps(prompt) + ', "max_tokens": ').encode()
        with open(f, "rb") as fh:
            head = fh.read(len(prefix) + 64)
            size = fh.seek(0, 2)
            fh.seek(max(0, size - 400))
            tail = fh.read()
        if not head.startswith(prefix):
            return False
        if b'"tokens": []' in head:
            return False
        m = re.match(rb"\d+", head[len(prefix):])
        if m is None:
            return False
        cached_max_tokens = int(m.group(0))
        mp = PASSED_PATTERN.search(tail)
        mt = TOTAL_PATTERN.search(tail)
        passed_all = (
            mp is not None and mt is not None
            and int(mt.group(1)) > 0
            and int(mp.group(1)) == int(mt.group(1))
        )
        if rejection_samples > 0 and not passed_all:
            return False
        if cached_max_tokens < max_tokens and not passed_all:
            return False
        return True
    except Exception:
        return False


def _rejection_recover(teacher, scored, test_cases_per_prompt, rejection_samples,
                       rejection_wave, rejection_temperature, rejection_top_p):
    remaining = [s for s in scored
                 if s[4] is not None and not fully_passed(s[3], s[4])]
    failed_count = len(remaining)
    if not remaining:
        return
    wave = max(1, rejection_wave)
    print(f"  rejection sampling {failed_count} failed prompts "
          f"(n={rejection_samples}, wave={wave}, temp={rejection_temperature})...")
    recovered = 0
    generated = 0
    while remaining and generated < rejection_samples:
        n_this = min(wave, rejection_samples - generated)
        candidate_lists = teacher.sample_candidates_batch(
            [s[1] for s in remaining], n_this,
            rejection_temperature, rejection_top_p)
        generated += n_this
        tasks = []
        for ri, (s, candidates) in enumerate(zip(remaining, candidate_lists)):
            tcs = test_cases_per_prompt[s[0]]
            for ci, candidate in enumerate(candidates):
                tasks.append((ri, ci, candidate, tcs))
        with ThreadPoolExecutor(max_workers=min(16, max(1, len(tasks)))) as ex:
            task_scores = list(ex.map(lambda t: _score_result(t[2], t[3]), tasks))
        by_prompt = {}
        for (ri, ci, candidate, _), score in zip(tasks, task_scores):
            by_prompt.setdefault(ri, []).append((ci, candidate, score))
        still = []
        for ri, s in enumerate(remaining):
            hit = False
            for ci, candidate, score in sorted(by_prompt.get(ri, []), key=lambda x: x[0]):
                if fully_passed(score["passed"], score["total"]):
                    s[2], s[3], s[4], s[5] = candidate, score["passed"], score["total"], score
                    recovered += 1
                    hit = True
                    break
            if not hit:
                best = max(by_prompt.get(ri, []),
                           key=lambda x: (x[2].get("passed") or 0), default=None)
                if best is not None and (best[2].get("passed") or 0) > (s[5].get("passed") or 0):
                    s[5] = best[2]
                still.append(s)
        remaining = still
    print(f"  rejection recovered {recovered}/{failed_count} prompts "
          f"(generated up to {generated}/{rejection_samples} per prompt).")


def precompute_and_cache(teacher, prompts: list, cache_dir: str,
                         test_cases_per_prompt: list = None,
                         difficulty_per_prompt: list = None,
                         chunk_size: int = 64,
                         rejection_samples: int = 0,
                         rejection_wave: int = 2,
                         rejection_temperature: float = 1.0,
                         rejection_top_p: float = 0.95) -> None:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    pending = [(idx, p) for idx, p in enumerate(prompts)
               if not _is_valid_cache(cache_path, idx, p, teacher.max_tokens, rejection_samples)]

    if not pending:
        print(f"All {len(prompts)} teacher responses already cached.")
        return

    print(f"Caching {len(pending)}/{len(prompts)} teacher responses via vLLM continuous batching...")
    print(f"  chunk_size={chunk_size}  (save progress every chunk)")
    runlog = RunLog("logs/cache_build.jsonl")
    runlog.event("build_start", pending=len(pending), total=len(prompts), chunk_size=chunk_size)
    total_start = time.time()
    pass_count = 0
    fail_count = 0

    bg_thread = None
    bg_error = [None]

    diff_pass = Counter()
    diff_fail = Counter()
    diff_causes = {}

    def _get_diff(idx):
        if difficulty_per_prompt and idx < len(difficulty_per_prompt):
            return difficulty_per_prompt[idx] or "?"
        return "?"

    def _test_and_save(chunk, results, chunk_idx, total_chunks, chunk_t0):
        nonlocal pass_count, fail_count
        try:
            scored = []
            if test_cases_per_prompt is not None:
                with ThreadPoolExecutor(max_workers=min(16, len(chunk))) as ex:
                    score_out = list(ex.map(
                        lambda ir: _score_result(ir[1], test_cases_per_prompt[ir[0]]),
                        [(idx, result) for (idx, _), result in zip(chunk, results)]))
                for (idx, prompt), result, score in zip(chunk, results, score_out):
                    scored.append([idx, prompt, result, score["passed"], score["total"], score])
            else:
                for (idx, prompt), result in zip(chunk, results):
                    scored.append([idx, prompt, result, None, None, None])

            if rejection_samples > 0 and test_cases_per_prompt is not None:
                _rejection_recover(teacher, scored, test_cases_per_prompt, rejection_samples,
                                   rejection_wave, rejection_temperature, rejection_top_p)

            chunk_pass = chunk_fail = 0
            chunk_diff_pass = Counter()
            chunk_diff_fail = Counter()
            chunk_fail_causes = {}
            for idx, prompt, result, test_passed, test_total, score in scored:
                diff = _get_diff(idx)
                new_pass = fully_passed(test_passed, test_total)
                if test_total is not None and not new_pass and tail_fully_passed(cache_path / f"{idx}.json"):
                    pass_count += 1
                    chunk_pass += 1
                    diff_pass[diff] += 1
                    chunk_diff_pass[diff] += 1
                    continue

                if test_total is not None:
                    if new_pass:
                        pass_count += 1
                        chunk_pass += 1
                        diff_pass[diff] += 1
                        chunk_diff_pass[diff] += 1
                    else:
                        fail_count += 1
                        chunk_fail += 1
                        diff_fail[diff] += 1
                        chunk_diff_fail[diff] += 1
                        if score is not None:
                            cause = _dominant_cause(score)
                            chunk_fail_causes.setdefault(diff, Counter())[cause] += 1
                            diff_causes.setdefault(diff, Counter())[cause] += 1

                payload = {"prompt": prompt, "max_tokens": teacher.max_tokens, **result}
                if test_total is not None:
                    payload["test_passed"] = test_passed
                    payload["test_total"] = test_total
                    if score is not None and not new_pass:
                        payload["fail_cause"] = _dominant_cause(score)
                with open(cache_path / f"{idx}.json", "w") as f:
                    json.dump(payload, f)

            chunk_elapsed = time.time() - chunk_t0
            done_count = min(chunk_idx * chunk_size, len(pending))
            total_elapsed = time.time() - total_start
            avg_per_prompt = total_elapsed / max(done_count, 1)
            eta_sec = avg_per_prompt * (len(pending) - done_count)

            tested = pass_count + fail_count
            rate = pass_count / max(tested, 1)

            print(
                f"  chunk {chunk_idx}/{total_chunks} done in {chunk_elapsed:.1f}s  "
                f"({chunk_elapsed / len(chunk):.1f}s/prompt)  "
                f"|  pass {chunk_pass}/{len(chunk)}  "
                f"|  total {pass_count}/{tested} ({rate:.1%})  "
                f"|  ETA {eta_str(eta_sec)}  "
                f"|  GPU {gpu_used_gb():.1f}GB"
            )
            all_diffs = sorted(set(list(chunk_diff_pass) + list(chunk_diff_fail)),
                               key=lambda d: DIFFICULTY_ORDER.get(d, 3))
            for d in all_diffs:
                cp = chunk_diff_pass.get(d, 0)
                cf = chunk_diff_fail.get(d, 0)
                tp = diff_pass.get(d, 0)
                tf = diff_fail.get(d, 0)
                causes_str = ""
                if d in chunk_fail_causes:
                    causes_str = "  " + ", ".join(
                        f"{c}={n}" for c, n in chunk_fail_causes[d].most_common())
                print(
                    f"    {d:<8} {cp}/{cp+cf} solved  "
                    f"(total {tp}/{tp+tf}  {tp/max(tp+tf,1):.0%})"
                    f"{causes_str}"
                )

            runlog.event(
                "chunk", idx=chunk_idx, of=total_chunks, secs=round(chunk_elapsed, 1),
                s_per_prompt=round(chunk_elapsed / len(chunk), 1),
                chunk_pass=chunk_pass, chunk_fail=chunk_fail,
                total_pass=pass_count, tested=tested, pass_rate=round(rate, 4),
                by_difficulty={
                    d: {"pass": diff_pass.get(d, 0), "fail": diff_fail.get(d, 0),
                        "causes": dict(diff_causes.get(d, {}))}
                    for d in set(list(diff_pass) + list(diff_fail))
                },
                gpu_gb=round(gpu_used_gb(), 1), eta_s=int(eta_sec),
            )
        except Exception as e:
            bg_error[0] = e

    for chunk_start in range(0, len(pending), chunk_size):
        chunk = pending[chunk_start:chunk_start + chunk_size]
        chunk_prompts = [p for _, p in chunk]

        chunk_idx = chunk_start // chunk_size + 1
        total_chunks = (len(pending) + chunk_size - 1) // chunk_size
        chunk_t0 = time.time()
        print(f"\n[chunk {chunk_idx}/{total_chunks}] generating {len(chunk)} prompts via vLLM...")

        results = teacher.get_responses_batch(chunk_prompts)

        if bg_thread is not None:
            bg_thread.join()
            if bg_error[0] is not None:
                raise bg_error[0]

        bg_thread = Thread(
            target=_test_and_save,
            args=(chunk, results, chunk_idx, total_chunks, chunk_t0),
        )
        bg_thread.start()

    if bg_thread is not None:
        bg_thread.join()
        if bg_error[0] is not None:
            raise bg_error[0]

    if test_cases_per_prompt is not None:
        tested = pass_count + fail_count
        rate = pass_count / max(tested, 1)
        total_min = (time.time() - total_start) / 60
        print(f"\n  Cache build complete in {total_min:.1f} min: "
              f"{pass_count}/{tested} pass all tests ({rate:.1%}).")
        runlog.event("build_done", minutes=round(total_min, 1),
                     total_pass=pass_count, tested=tested, pass_rate=round(rate, 4))
