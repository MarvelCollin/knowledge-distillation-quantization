#!/usr/bin/env python3
"""
Three-way evaluation: original student | teacher | distilled student.
Uses the LeetCodeDataset *test* split — never seen during training.

Usage (inside Docker):
    python scripts/compare_eval.py
    python scripts/compare_eval.py --distilled outputs/final
    python scripts/compare_eval.py --num-problems 30 --skip-teacher
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gc
import json
import math
import re
import time
import torch
import argparse
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from datasets import load_dataset
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from src.config import load_config
from src.data.dataset import _extract_test_cases
from src.evaluation.evaluator import (
    run_test_cases,
    failing_cases,
    write_failure_report,
)
from src.evaluation.generation import build_eval_prompts, budget_forced_generate
from src.utils.gpu import cleanup_vllm, gpu_total_gb, gpu_used_gb, wait_for_gpu_freed
from src.utils.reasoning import extract_code


_INTERMEDIATE = Path("outputs/eval/intermediate")

COLORS = {
    "Student (original)":            "#4e79a7",
    "Teacher (R1-Distill-Qwen-7B)":  "#f28e2b",
    "Student (distilled)":           "#59a14f",
}
FALLBACK_COLOR = "#888888"


def load_test_problems(n: int, dataset_name: str, difficulty: str = "all") -> list:
    # Accept one or several levels: "easy", "easy,medium", "easy+medium", "all".
    allowed = {d.strip().lower() for d in difficulty.replace("+", ",").split(",") if d.strip()}
    filter_on = bool(allowed) and "all" not in allowed
    diff_label = f"{'+'.join(sorted(allowed))} " if filter_on else ""
    print(f"Loading {n} {diff_label}problems from {dataset_name} test split (separate from training data)...")
    raw = load_dataset(dataset_name, split="test")
    problems = []
    for item in raw:
        if filter_on and (item.get("difficulty") or "").lower() not in allowed:
            continue
        entry_point = (item.get("entry_point") or "").strip()
        # Strip class prefix like "Solution().method" → just the method name
        ep_clean = re.search(r'(\w+)$', entry_point)
        entry_point = ep_clean.group(1) if ep_clean else entry_point
        test_str = item.get("test") or ""
        cases = _extract_test_cases(test_str, entry_point) if test_str.strip() and entry_point else []
        code = (item.get("response") or "").strip()
        text = (item.get("problem_description") or "").strip()
        if not cases or not code or not text:
            continue
        problems.append({
            "text": text,
            "code": code,
            "test_cases": cases,
            "difficulty": (item.get("difficulty") or ""),
            "entry_point": entry_point,
        })
        if len(problems) >= n:
            break
    print(f"  Loaded {len(problems)} problems.")
    return problems


def _cache_path(label: str) -> Path:
    safe = label.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    return _INTERMEDIATE / f"{safe}.json"


def _save_intermediate(label: str, data: dict) -> None:
    _INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    _cache_path(label).write_text(json.dumps(data, indent=2))


def _load_intermediate(label: str, dataset_name: str) -> dict | None:
    p = _cache_path(label)
    if p.exists():
        d = json.loads(p.read_text())
        if d.get("dataset") == dataset_name:
            return d
    return None


def _model_fingerprint(model_path: str) -> str:
    weights = Path(model_path) / "model.safetensors"
    if weights.exists():
        st = weights.stat()
        return f"{model_path}:{int(st.st_mtime)}:{st.st_size}"
    return model_path


_FAILURE_CATEGORIES = ("syntax_error", "wrong_answer", "runtime_error", "missing_function", "timeout")


def _pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((1.0 - k / i) for i in range(n - c + 1, n + 1))


def _difficulty_breakdown(results: list, k: int) -> list:
    """Per-difficulty pass@1 / pass@k / solved, so a single run shows where each
    model floors (e.g. Hard) and justifies any difficulty filtering in the report."""
    buckets = {}
    for r in results:
        d = (r.get("difficulty") or "unknown").strip().lower() or "unknown"
        buckets.setdefault(d, []).append(r)
    order = ["easy", "medium", "hard", "unknown"]
    rows = []
    for d in sorted(buckets, key=lambda x: order.index(x) if x in order else len(order)):
        rs = buckets[d]
        m = len(rs)
        p1 = sum(_pass_at_k(x.get("num_samples", 1), x.get("num_passing", int(x["solved"])), 1) for x in rs) / m
        pk = sum(_pass_at_k(x.get("num_samples", 1), x.get("num_passing", int(x["solved"])), k) for x in rs) / m
        rows.append({
            "difficulty": d,
            "num_problems": m,
            "pass_at_1": p1,
            "pass_at_k": pk,
            "solved": sum(1 for x in rs if x["solved"]),
        })
    return rows


def _dump_failure_report(label: str, results: list) -> None:
    """Write a readable Markdown breakdown of every problem (code + failing cases for
    the misses) to outputs/eval/details/<label>.md for root-cause debugging."""
    entries = []
    for r in results:
        diff = (r.get("difficulty") or "?")
        np_, ns = r.get("num_passing", int(r["solved"])), r.get("num_samples", 1)
        header = f"Problem {r['idx']}  [{diff}]  —  {np_}/{ns} samples passed"
        # pick a representative failing sample (first unsolved), else first sample
        samples = r.get("samples") or [r]
        rep = next((s for s in samples if not s.get("solved")), samples[0])
        note = None
        if r["solved"]:
            note = f"_first sample: {r['passed']}/{r['total']} test cases_"
        if any(s.get("truncated") for s in samples):
            note = (note or "") + "  ⚠ some samples truncated"
        entries.append({
            "header": header,
            "solved": bool(r["solved"]),
            "note": note,
            "code": rep.get("code", ""),
            "fails": failing_cases(rep, limit=3),
        })
    safe = label.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    out_path = _INTERMEDIATE.parent / "details" / f"{safe}.md"
    write_failure_report(out_path, f"Failure report — {label}", entries)
    print(f"  Detail report     : {out_path}")


def _finalise(label: str, results: list, dataset_name: str, k: int, meta: dict | None = None) -> dict:
    tp = sum(r["passed"] for r in results)
    tt = sum(r["total"] for r in results)
    solved = sum(1 for r in results if r["solved"])
    truncated = sum(1 for r in results if r.get("truncated"))
    n = len(results)
    failure_counts = {cat: 0 for cat in _FAILURE_CATEGORIES}
    for r in results:
        for cat in r.get("categories", []):
            if cat in failure_counts:
                failure_counts[cat] += 1

    n_samples = max((r.get("num_samples", 1) for r in results), default=1)
    pass1 = sum(_pass_at_k(r.get("num_samples", 1), r.get("num_passing", int(r["solved"])), 1) for r in results) / max(n, 1)
    passk = sum(_pass_at_k(r.get("num_samples", 1), r.get("num_passing", int(r["solved"])), k) for r in results) / max(n, 1)

    by_difficulty = _difficulty_breakdown(results, k)

    summary = {
        "name": label,
        "dataset": dataset_name,
        "k": k,
        "num_samples": n_samples,
        "pass_at_1": pass1,
        "pass_at_k": passk,
        "test_pass_rate": tp / max(tt, 1),
        "problem_solve_rate": solved / max(n, 1),
        "total_passed": tp,
        "total_tests": tt,
        "problems_solved": solved,
        "num_problems": n,
        "truncated": truncated,
        "failure_counts": failure_counts,
        "by_difficulty": by_difficulty,
        "per_problem": results,
    }
    if meta:
        summary.update(meta)
    print(f"\n  pass@1            : {pass1:.1%}")
    if k > 1:
        print(f"  pass@{k:<2}            : {passk:.1%}")
    print(f"  Problems solved   : {solved}/{n}  (any of {n_samples} samples)")
    print(f"  Test cases passed : {tp}/{tt}  ({summary['test_pass_rate']:.1%})  [first sample]")
    print(f"  Failure modes     : "
          + ", ".join(f"{c}={failure_counts[c]}" for c in _FAILURE_CATEGORIES if failure_counts[c]))
    if len(by_difficulty) > 1:
        print(f"  By difficulty     :")
        for d in by_difficulty:
            print(f"      {d['difficulty']:<8} pass@1={d['pass_at_1']:.1%}  "
                  f"pass@{k}={d['pass_at_k']:.1%}  solved={d['solved']}/{d['num_problems']}")
    if truncated:
        print(f"  ⚠ Truncated       : {truncated}/{n} hit the token cap — raise evaluation.max_new_tokens")
    _save_intermediate(label, summary)
    _dump_failure_report(label, results)
    return summary


def evaluate_model(label: str, model_path: str, problems: list,
                   max_new_tokens: int, device, is_teacher: bool = False,
                   dataset_name: str = "", num_samples: int = 1,
                   temperature: float = 0.7, top_p: float = 0.95,
                   k: int = 1, difficulty: str = "all", seed: int = 1234) -> dict:
    cache_key = {
        "num_problems": len(problems),
        "num_samples": num_samples,
        "k": k,
        "difficulty": difficulty,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "model_fingerprint": _model_fingerprint(model_path),
    }
    cached = _load_intermediate(label, dataset_name)
    if cached and all(cached.get(key) == val for key, val in cache_key.items()):
        print(f"\n  ✓ Reusing cached results for: {label}  (settings unchanged — not re-running)")
        return cached

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"\n{'═' * 62}")
    print(f"  Evaluating: {label}")
    print(f"  Model path: {model_path}")
    print(f"{'═' * 62}")

    total_gb = gpu_total_gb()
    pre_used = gpu_used_gb()
    print(f"  GPU before load: {pre_used:.1f}GB used / {total_gb:.1f}GB total")

    required_gb = 20.0
    free_gb = total_gb - pre_used
    if free_gb < required_gb:
        print(f"  WARNING: only {free_gb:.1f}GB free, need ≥{required_gb:.1f}GB. Forcing cleanup...")
        free_gb = wait_for_gpu_freed(target_free_gb=required_gb, max_retries=5, retry_sleep=3.0)
        if free_gb < required_gb:
            print(f"  ERROR: GPU still has only {free_gb:.1f}GB free after retries. Memory leak suspected.")
            raise RuntimeError(f"Insufficient GPU memory: need {required_gb:.1f}GB, have {free_gb:.1f}GB")

    available_for_vllm = free_gb / total_gb
    safety_margin = 0.05
    base_util = 0.82
    gpu_mem_util = min(base_util, available_for_vllm - safety_margin)
    gpu_mem_util = max(gpu_mem_util, 0.30)
    max_model_len = max_new_tokens + 2048

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading vLLM (gpu_memory_utilization={gpu_mem_util:.2f}, max_model_len={max_model_len})...")
    llm = None
    try:
        llm = LLM(
            model=model_path,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=False,
            enable_prefix_caching=True,
        )

        formatted_prompts, signatures = build_eval_prompts(problems, tokenizer)

        post_load = gpu_used_gb()
        print(f"  GPU after load: {post_load:.1f}GB used  (delta +{post_load - pre_used:.1f}GB)")

        eval_chunk_size = 3 if is_teacher else 10
        total_chunks = (len(formatted_prompts) + eval_chunk_size - 1) // eval_chunk_size
        print(f"  Generating {len(problems)} prompts × {num_samples} samples via vLLM "
              f"(budget-forced think/code, chunk_size={eval_chunk_size}, {total_chunks} chunks)...")
        gen_grid = [None] * len(formatted_prompts)
        for chunk_start in range(0, len(formatted_prompts), eval_chunk_size):
            chunk = formatted_prompts[chunk_start:chunk_start + eval_chunk_size]
            chunk_sigs = signatures[chunk_start:chunk_start + eval_chunk_size]
            chunk_idx = chunk_start // eval_chunk_size + 1
            chunk_t0 = time.time()
            print(f"  [chunk {chunk_idx}/{total_chunks}] generating {len(chunk)} prompts × {num_samples} samples...")
            chunk_grid = budget_forced_generate(
                llm, chunk, chunk_sigs, num_samples, temperature, top_p, max_new_tokens,
                seed=seed,
            )
            for off, row in enumerate(chunk_grid):
                gen_grid[chunk_start + off] = row
            chunk_elapsed = time.time() - chunk_t0
            done = chunk_start + len(chunk)
            avg = chunk_elapsed / len(chunk)
            eta_remaining = avg * (len(formatted_prompts) - done) * 1.05
            eta_m, eta_s = divmod(int(eta_remaining), 60)
            eta_h, eta_m = divmod(eta_m, 60)
            print(f"    chunk done in {chunk_elapsed:.1f}s ({avg:.1f}s/prompt avg)  |  ETA {eta_h}h {eta_m:02d}m {eta_s:02d}s  |  GPU {gpu_used_gb():.1f}GB")

        print(f"  Running {len(problems) * num_samples} test executions in parallel (max 16 workers)...")
        eval_tasks = []
        for i, prob in enumerate(problems):
            for j in range(num_samples):
                final_text, truncated = gen_grid[i][j]
                code = extract_code(final_text)
                eval_tasks.append((i, j, code, truncated, prob["test_cases"]))

        def _run_one(task):
            i, j, code, truncated, tcs = task
            r = run_test_cases(code, tcs)
            return i, j, code, truncated, r

        sample_records_by_problem = {i: [None] * num_samples for i in range(len(problems))}
        with ThreadPoolExecutor(max_workers=16) as ex:
            for fut in tqdm(ex.map(_run_one, eval_tasks), total=len(eval_tasks), desc="tests"):
                i, j, code, truncated, r = fut
                sample_solved = r["passed"] == r["total"] and r["total"] > 0
                sample_records_by_problem[i][j] = {
                    **r, "solved": sample_solved, "code": code, "truncated": truncated,
                }

        results = []
        for i, prob in enumerate(problems):
            sample_records = sample_records_by_problem[i]
            num_passing = sum(1 for s in sample_records if s["solved"])
            any_solved = num_passing > 0
            any_truncated = any(s["truncated"] for s in sample_records)
            first = sample_records[0]

            results.append({
                "idx": i,
                "difficulty": prob.get("difficulty", ""),
                "passed": first["passed"], "total": first["total"],
                "details": first["details"], "errors": first["errors"], "categories": first["categories"],
                "solved": any_solved, "code": first["code"], "truncated": any_truncated,
                "samples": sample_records, "num_samples": num_samples, "num_passing": num_passing,
            })

            diff_tag = f"[{prob.get('difficulty', '')}] " if prob.get("difficulty") else ""
            trunc_tag = " ⚠ TRUNCATED" if any_truncated else ""
            if num_samples > 1:
                print(
                    f"  {'✓' if any_solved else '✗'} [{i+1:>3}/{len(problems)}] {diff_tag}"
                    f"  {num_passing}/{num_samples} samples solved{trunc_tag}"
                )
            else:
                print(
                    f"  {'✓' if any_solved else '✗'} [{i+1:>3}/{len(problems)}] {diff_tag}"
                    f"  {first['passed']}/{first['total']} test cases{trunc_tag}"
                )

        summary = _finalise(label, results, dataset_name, k, meta=cache_key)

    finally:
        print(f"  Cleaning up vLLM...")
        cleanup_vllm(llm)
        llm = None
        del tokenizer
        for _ in range(3):
            gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)
        post_cleanup = gpu_used_gb()
        print(f"  GPU after cleanup: {post_cleanup:.1f}GB used  (started at {pre_used:.1f}GB)")
        residual = post_cleanup - pre_used
        if residual > 1.0:
            print(f"  ⚠ Residual leak detected: +{residual:.1f}GB above baseline. Forcing extra cleanup...")
            wait_for_gpu_freed(target_free_gb=total_gb - pre_used - 1.0, max_retries=3, retry_sleep=2.0)

    return summary


def _style_ax(ax) -> None:
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444466")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color("white")
    ax.yaxis.label.set_color("#aaaaaa")
    ax.xaxis.label.set_color("#aaaaaa")
    ax.grid(axis="y", color="#444466", linestyle="--", linewidth=0.5, zorder=0)


def plot_comparison(summaries: list, out_path: Path) -> None:
    names      = [s["name"] for s in summaries]
    test_rates = [s["test_pass_rate"] for s in summaries]
    k_value    = summaries[0].get("k", 1)
    solve_rates= [s.get("pass_at_k", s["problem_solve_rate"]) for s in summaries]
    solve_title = f"pass@{k_value}  (unbiased estimator)" if k_value > 1 else "Problem Solve Rate  (all tests pass)"
    n_problems = summaries[0]["num_problems"]
    matrix     = np.array([[r["solved"] for r in s["per_problem"]] for s in summaries],
                           dtype=float)
    colors     = [COLORS.get(n, FALLBACK_COLOR) for n in names]

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("#0f0f1a")
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[1, 1, 0.9],
        hspace=0.55, wspace=0.35,
        left=0.06, right=0.97, top=0.88, bottom=0.07,
    )

    x = np.arange(len(names))
    bar_w = 0.45

    def _bar(ax, rates, title, ylabel):
        _style_ax(ax)
        bars = ax.bar(x, [r * 100 for r in rates], width=bar_w, color=colors,
                      edgecolor="#222244", linewidth=0.8, zorder=3)
        ax.set_title(title, color="white", fontsize=12, pad=8)
        ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=10)
        ax.set_ylim(0, 115)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9, rotation=10)
        for bar, v in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2.5,
                    f"{v:.1%}", ha="center", va="bottom",
                    color="white", fontsize=12, fontweight="bold")

    _bar(fig.add_subplot(gs[0, 0]), test_rates,
         "Test-Case Pass Rate", "Pass rate (%)")
    _bar(fig.add_subplot(gs[0, 1]), solve_rates,
         solve_title, "Solved (%)")

    ax_delta = fig.add_subplot(gs[1, :])
    _style_ax(ax_delta)
    if len(summaries) >= 2:
        baseline = np.array([r["solved"] for r in summaries[0]["per_problem"]], dtype=float)
        for j, s in enumerate(summaries[1:], start=1):
            delta = np.array([r["solved"] for r in s["per_problem"]], dtype=float) - baseline
            col = COLORS.get(s["name"], FALLBACK_COLOR)
            ax_delta.bar(
                np.arange(n_problems) + (j - 1) * 0.3, delta,
                width=0.28, color=col, alpha=0.85, label=s["name"],
                edgecolor="#222244", linewidth=0.5, zorder=3,
            )
        ax_delta.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
        ax_delta.set_xticks(np.arange(n_problems))
        ax_delta.set_xticklabels(np.arange(n_problems), fontsize=7, color="#aaaaaa")
        ax_delta.set_xlabel("Problem index (LeetCode test split)", color="#aaaaaa", fontsize=10)
        ax_delta.set_title(
            f"Δ vs Student (original)  — +1 = newly solved, -1 = regression",
            color="white", fontsize=12, pad=8,
        )
        ax_delta.set_ylim(-1.4, 1.4)
        ax_delta.set_yticks([-1, 0, 1])
        ax_delta.set_yticklabels(["-1\n(regression)", "0\n(same)", "+1\n(improved)"],
                                 color="white", fontsize=8)
        leg = ax_delta.legend(loc="upper right", framealpha=0.2,
                              labelcolor="white", fontsize=9)
        leg.get_frame().set_edgecolor("#555577")

    ax_heat = fig.add_subplot(gs[2, :])
    _style_ax(ax_heat)
    im = ax_heat.imshow(matrix, aspect="auto", cmap="RdYlGn",
                        vmin=0, vmax=1, interpolation="nearest")
    ax_heat.set_yticks(range(len(names)))
    ax_heat.set_yticklabels(names, color="white", fontsize=10)
    ax_heat.set_xticks(range(n_problems))
    ax_heat.set_xticklabels(range(n_problems), fontsize=7, color="#aaaaaa")
    ax_heat.set_xlabel("Problem index (LeetCode test split)", color="#aaaaaa", fontsize=10)
    ax_heat.set_title(
        "Per-Problem Heatmap  (green = solved ✓, red = failed ✗)",
        color="white", fontsize=12, pad=8,
    )
    for m in range(len(names)):
        for p in range(n_problems):
            ax_heat.text(p, m, "✓" if matrix[m, p] else "✗",
                         ha="center", va="center", color="white", fontsize=8)
    cbar = fig.colorbar(im, ax=ax_heat, orientation="vertical",
                        fraction=0.012, pad=0.01)
    cbar.ax.yaxis.set_tick_params(color="white")
    for lbl in cbar.ax.yaxis.get_ticklabels():
        lbl.set_color("white")

    passk_header = f"pass@{k_value}" if k_value > 1 else "pass@1"
    summary_lines = [
        f"{'Model':<30}  {'pass@1':>8}  {passk_header:>8}  {'test-case':>10}  {'solved':>10}",
        "─" * 76,
    ]
    for s in summaries:
        summary_lines.append(
            f"{s['name']:<30}  {s['pass_at_1']:>7.1%}"
            f"  {s.get('pass_at_k', s['pass_at_1']):>7.1%}"
            f"  {s['test_pass_rate']:>9.1%}"
            f"  {s['problems_solved']:>3}/{s['num_problems']}"
        )

    fig.suptitle(
        "QEAD Knowledge Distillation — Evaluation on LeetCode Test Split\n"
        + "\n".join(summary_lines),
        color="white", fontsize=10, fontweight="bold",
        y=0.975, va="top", family="monospace",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nGraph saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-way model comparison on LeetCode test split")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--distilled", default=None,
                        help="Path to distilled student checkpoint (auto-detected if omitted)")
    parser.add_argument("--num-problems", type=int, default=30,
                        help="Number of LeetCode test problems to evaluate (default: 30)")
    parser.add_argument("--difficulty", default="all",
                        help="Difficulties to include, comma-separated: easy, medium, hard, "
                             "or all (default: all). e.g. --difficulty easy,medium")
    parser.add_argument("--skip-teacher", action="store_true",
                        help="Skip teacher evaluation to save time (~1 min/problem)")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Samples per problem for pass@k (default: 1 = greedy)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--k", type=int, default=None,
                        help="k for pass@k (defaults to --num-samples)")
    parser.add_argument("--out", default="outputs/eval/comparison.png")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Sampling seed for reproducible pass@k (static models give identical results each run)")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore and clear cached per-model results, re-evaluating every model from scratch.")
    args = parser.parse_args()
    k = args.k if args.k is not None else args.num_samples
    if k > args.num_samples:
        raise SystemExit(f"--k ({k}) cannot exceed --num-samples ({args.num_samples})")

    load_dotenv()

    if args.fresh:
        for p in _INTERMEDIATE.glob("*.json"):
            p.unlink()
        print("  --fresh: cleared cached intermediate results — every model will be re-evaluated.")

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_new_tokens = args.max_new_tokens or config["evaluation"]["max_new_tokens"]
    student_model  = config["student"]["model_name"]
    teacher_path   = config["teacher"]["local_model_path"]

    dataset_name = config["data"]["dataset_name"]
    problems = load_test_problems(args.num_problems, dataset_name, args.difficulty)

    summaries = []

    summaries.append(evaluate_model(
        "Student (original)", student_model,
        problems, max_new_tokens, device, is_teacher=False,
        dataset_name=dataset_name,
        num_samples=args.num_samples, temperature=args.temperature, top_p=args.top_p, k=k,
        difficulty=args.difficulty, seed=args.seed,
    ))

    if not args.skip_teacher:
        summaries.append(evaluate_model(
            "Teacher (R1-Distill-Qwen-7B)", teacher_path,
            problems, max_new_tokens, device, is_teacher=True,
            dataset_name=dataset_name,
            num_samples=args.num_samples, temperature=args.temperature, top_p=args.top_p, k=k,
            difficulty=args.difficulty, seed=args.seed,
        ))

    distilled_path = args.distilled
    if distilled_path is None:
        final_dir = Path(config["training"]["output_dir"]) / "final"
        if (final_dir / "model.safetensors").exists():
            distilled_path = str(final_dir)
            print(f"\nAuto-detected distilled checkpoint: {distilled_path}")
    if distilled_path:
        summaries.append(evaluate_model(
            "Student (distilled)", distilled_path,
            problems, max_new_tokens, device, is_teacher=False,
            dataset_name=dataset_name,
            num_samples=args.num_samples, temperature=args.temperature, top_p=args.top_p, k=k,
            difficulty=args.difficulty, seed=args.seed,
        ))
    else:
        print("\nNo distilled checkpoint found — run train.py first to generate one.")

    out_dir = Path(config["evaluation"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(summaries, indent=2))
    print(f"Full results → {out_dir / 'comparison.json'}")

    if len(summaries) >= 2:
        plot_comparison(summaries, Path(args.out))

    passk_header = f"pass@{k}" if k > 1 else "pass@1"
    print(f"\n{'═' * 72}")
    print(f"  {'Model':<30}  {'pass@1':>8}  {passk_header:>8}  {'test-case':>10}")
    print(f"{'─' * 72}")
    for s in summaries:
        print(
            f"  {s['name']:<30}  {s['pass_at_1']:>7.1%}"
            f"  {s.get('pass_at_k', s['pass_at_1']):>7.1%}"
            f"  {s['test_pass_rate']:>9.1%}"
        )
    print(f"{'═' * 72}")


if __name__ == "__main__":
    main()
