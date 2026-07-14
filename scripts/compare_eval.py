#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gc
import json
import math
import time
import torch
import argparse
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from src.config import load_config
from src.data.problems import load_test_problems
from src.evaluation.evaluator import (
    run_test_cases,
    failing_cases,
    write_failure_report,
)
from src.evaluation.generation import build_eval_prompts, budget_forced_generate, free_generate
from src.utils.gpu import cleanup_vllm, gpu_total_gb, gpu_used_gb, wait_for_gpu_freed
from src.utils.reasoning import extract_code


_INTERMEDIATE = Path("outputs/eval/intermediate")

MEM_CEILING_GB = 22.0

COLORS = {
    "Student original":  "#4e79a7",
    "Teacher":           "#f28e2b",
    "Student distilled": "#59a14f",
}
FALLBACK_COLOR = "#888888"


def _color(name: str) -> str:
    return next((c for k, c in COLORS.items() if name.startswith(k)), FALLBACK_COLOR)


def _safe_label(label: str) -> str:
    return label.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")


def _cache_path(label: str) -> Path:
    return _INTERMEDIATE / f"{_safe_label(label)}.json"


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


def _gen_cache_path(label: str) -> Path:
    return _INTERMEDIATE / f"{_safe_label(label)}_gen.json"


def _save_gen_grid(label: str, gen_grid: list, cache_key: dict) -> None:
    _INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    payload = {"cache_key": cache_key, "gen_grid": gen_grid}
    _gen_cache_path(label).write_text(json.dumps(payload))
    print(f"  Saved generation cache → {_gen_cache_path(label)}")


def _load_gen_grid(label: str, cache_key: dict) -> list | None:
    p = _gen_cache_path(label)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if d.get("cache_key") != cache_key:
        return None
    print(f"  ✓ Reusing cached generations for: {label}")
    return d["gen_grid"]


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


def _difficulty_breakdown(results: list, k: int = 5) -> list:
    buckets = {}
    for r in results:
        d = (r.get("difficulty") or "unknown").strip().lower() or "unknown"
        buckets.setdefault(d, []).append(r)
    order = ["easy", "medium", "hard", "unknown"]
    rows = []
    for d in sorted(buckets, key=lambda x: order.index(x) if x in order else len(order)):
        rs = buckets[d]
        tp = sum(x["passed"] for x in rs)
        tt = sum(x["total"] for x in rs)
        n = len(rs)
        p1 = sum(_pass_at_k(r.get("num_samples", 1), r.get("num_passing", int(r["solved"])), 1) for r in rs) / max(n, 1)
        pk = sum(_pass_at_k(r.get("num_samples", 1), r.get("num_passing", int(r["solved"])), min(k, r.get("num_samples", 1))) for r in rs) / max(n, 1)
        rows.append({
            "difficulty": d,
            "num_problems": n,
            "test_pass_rate": tp / max(tt, 1),
            "solved": sum(1 for x in rs if x["solved"]),
            "pass_at_1": p1,
            "pass_at_k": pk,
        })
    return rows


def _dump_failure_report(label: str, results: list) -> None:
    entries = []
    for r in results:
        diff = (r.get("difficulty") or "?")
        np_, ns = r.get("num_passing", int(r["solved"])), r.get("num_samples", 1)
        header = f"Problem {r['idx']}  [{diff}]  —  {np_}/{ns} samples passed"
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
    out_path = _INTERMEDIATE.parent / "details" / f"{_safe_label(label)}.md"
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
    print(f"\n{'─' * 70}")
    print(f"  {label}")
    print(f"{'─' * 70}")
    print(f"  test_pass_rate : {summary['test_pass_rate']:.1%}  ({tp}/{tt})")
    print(f"  pass@1         : {pass1:.1%}")
    if n_samples > 1:
        print(f"  pass@{k}         : {passk:.1%}  ({solved}/{n} solved)")
    print(f"  failure modes  : "
          + ", ".join(f"{c}={failure_counts[c]}" for c in _FAILURE_CATEGORIES if failure_counts[c]))
    if truncated:
        print(f"  truncated      : {truncated}/{n}")
    if len(by_difficulty) > 1:
        print()
        header = f"  {'difficulty':<10} {'problems':>8} {'test_pass':>10} {'pass@1':>8}"
        if n_samples > 1:
            header += f" {'pass@' + str(k):>8} {'solved':>8}"
        print(header)
        print(f"  {'─' * (len(header) - 2)}")
        for d in by_difficulty:
            row = f"  {d['difficulty']:<10} {d['num_problems']:>8} {d['test_pass_rate']:>9.1%} {d['pass_at_1']:>8.1%}"
            if n_samples > 1:
                row += f" {d['pass_at_k']:>8.1%} {d['solved']:>5}/{d['num_problems']}"
            print(row)
    print(f"{'─' * 70}")
    _save_intermediate(label, summary)
    _dump_failure_report(label, results)
    return summary


def evaluate_model(label: str, model_path: str, problems: list,
                   max_new_tokens: int, device, is_teacher: bool = False,
                   dataset_name: str = "", num_samples: int = 1,
                   temperature: float = 0.7, top_p: float = 0.95,
                   k: int = 1, difficulty: str = "all", seed: int = 1234,
                   think_ratio: float = 0.75) -> dict:
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

    gen_grid = _load_gen_grid(label, cache_key)

    if gen_grid is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        llm = None
    else:
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
        base_util = 0.90
        ceiling_util = MEM_CEILING_GB / total_gb
        gpu_mem_util = min(base_util, available_for_vllm - safety_margin, ceiling_util)
        gpu_mem_util = max(gpu_mem_util, 0.30)
        max_model_len = max_new_tokens + 2048

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"  Loading vLLM (gpu_memory_utilization={gpu_mem_util:.2f} "
              f"≈{gpu_mem_util * total_gb:.1f}GB, ceiling={MEM_CEILING_GB:.0f}GB, max_model_len={max_model_len})...")
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

            eval_chunk_size = 8 if is_teacher else 24
            gen_mode = "free" if is_teacher else "budget-forced"
            total_chunks = (len(formatted_prompts) + eval_chunk_size - 1) // eval_chunk_size
            print(f"  Generating {len(problems)} prompts × {num_samples} samples via vLLM "
                  f"({gen_mode} think/code, chunk_size={eval_chunk_size}, {total_chunks} chunks)...")
            gen_grid = [None] * len(formatted_prompts)
            for chunk_start in range(0, len(formatted_prompts), eval_chunk_size):
                chunk = formatted_prompts[chunk_start:chunk_start + eval_chunk_size]
                chunk_sigs = signatures[chunk_start:chunk_start + eval_chunk_size]
                chunk_idx = chunk_start // eval_chunk_size + 1
                chunk_t0 = time.time()
                print(f"  [chunk {chunk_idx}/{total_chunks}] generating {len(chunk)} prompts × {num_samples} samples...")
                if is_teacher:
                    chunk_grid = free_generate(
                        llm, chunk, num_samples, temperature, top_p, max_new_tokens,
                        seed=seed,
                    )
                else:
                    chunk_grid = budget_forced_generate(
                        llm, chunk, chunk_sigs, num_samples, temperature, top_p, max_new_tokens,
                        think_ratio=think_ratio, seed=seed,
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

            _save_gen_grid(label, gen_grid, cache_key)

        finally:
            print(f"  Cleaning up vLLM...")
            cleanup_vllm(llm)
            llm = None
            del tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            for _ in range(3):
                gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)

    if gen_grid is None or any(g is None for g in gen_grid):
        raise RuntimeError(f"Generation incomplete for {label} — missing entries in gen_grid")

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

    gen_cache = _gen_cache_path(label)
    if gen_cache.exists():
        gen_cache.unlink()

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
    solve_rates= [s["problem_solve_rate"] for s in summaries]
    solve_title = "Problem Solve Rate  (all tests pass)"
    n_problems = summaries[0]["num_problems"]
    matrix     = np.array([[r["solved"] for r in s["per_problem"]] for s in summaries],
                           dtype=float)
    colors     = [_color(n) for n in names]

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
         "Overall Test-Case Pass Rate", "Pass rate (%)")
    _bar(fig.add_subplot(gs[0, 1]), solve_rates,
         solve_title, "Solved (%)")

    ax_delta = fig.add_subplot(gs[1, :])
    _style_ax(ax_delta)
    if len(summaries) >= 2:
        baseline = np.array([r["solved"] for r in summaries[0]["per_problem"]], dtype=float)
        for j, s in enumerate(summaries[1:], start=1):
            delta = np.array([r["solved"] for r in s["per_problem"]], dtype=float) - baseline
            col = _color(s["name"])
            ax_delta.bar(
                np.arange(n_problems) + (j - 1) * 0.3, delta,
                width=0.28, color=col, alpha=0.85, label=s["name"],
                edgecolor="#222244", linewidth=0.5, zorder=3,
            )
        ax_delta.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
        ax_delta.set_xticks(np.arange(n_problems))
        ax_delta.set_xticklabels(np.arange(n_problems), fontsize=7, color="#aaaaaa")
        ax_delta.set_xlabel("Problem index (test split)", color="#aaaaaa", fontsize=10)
        ax_delta.set_title(
            f"Δ vs {summaries[0]['name']}  — +1 = newly solved, -1 = regression",
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
    ax_heat.set_xlabel("Problem index (test split)", color="#aaaaaa", fontsize=10)
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

    summary_lines = [
        f"{'Model':<48}  {'test-case':>10}  {'solved':>10}",
        "─" * 74,
    ]
    for s in summaries:
        summary_lines.append(
            f"{s['name']:<48}"
            f"  {s['test_pass_rate']:>9.1%}"
            f"  {s['problems_solved']:>3}/{s['num_problems']}"
        )

    fig.suptitle(
        f"QEAD Knowledge Distillation — Evaluation on LeetCode Test Split\n"
        + "\n".join(summary_lines),
        color="white", fontsize=10, fontweight="bold",
        y=0.975, va="top", family="monospace",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nGraph saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-way model comparison on the LeetCode test split")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--distilled", default=None,
                        help="Path to distilled student checkpoint (auto-detected if omitted)")
    parser.add_argument("--num-problems", type=int, default=228,
                        help="Number of LeetCode test problems to evaluate (default: 228 = full test split)")
    parser.add_argument("--difficulty", default="all",
                        help="Difficulties to include, comma-separated: easy, medium, hard, "
                             "or all (default: all). e.g. --difficulty easy,medium")
    parser.add_argument("--skip-teacher", action="store_true",
                        help="Skip teacher evaluation to save time (~1 min/problem)")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--think-ratio", type=float, default=0.75,
                        help="Fraction of the token budget for the think phase of budget-forced student decoding.")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Samples per problem for pass@k (default: 1 = greedy)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--k", type=int, default=None,
                        help="k for pass@k (defaults to --num-samples)")
    parser.add_argument("--out", default=None,
                        help="Graph output path (default: outputs/eval/comparison_<dataset>.png)")
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

    student_base = student_model.split("/")[-1]

    summaries.append(evaluate_model(
        f"Student original ({student_base})", student_model,
        problems, max_new_tokens, device, is_teacher=False,
        dataset_name=dataset_name,
        num_samples=args.num_samples, temperature=args.temperature, top_p=args.top_p, k=k,
        difficulty=args.difficulty, seed=args.seed, think_ratio=args.think_ratio,
    ))

    distilled_path = args.distilled
    if distilled_path is None:
        out_dir = Path(config["training"]["output_dir"])
        for name in ("final", "final_last"):
            cand = out_dir / name
            if (cand / "model.safetensors").exists():
                distilled_path = str(cand)
                print(f"\nAuto-detected distilled checkpoint: {distilled_path}")
                break
    if distilled_path:
        summaries.append(evaluate_model(
            f"Student distilled ({student_base})", distilled_path,
            problems, max_new_tokens, device, is_teacher=False,
            dataset_name=dataset_name,
            num_samples=args.num_samples, temperature=args.temperature, top_p=args.top_p, k=k,
            difficulty=args.difficulty, seed=args.seed, think_ratio=args.think_ratio,
        ))
    else:
        print("\nNo distilled checkpoint found — run train.py first to generate one.")

    if not args.skip_teacher:
        summaries.append(evaluate_model(
            "Teacher (R1-Distill-Qwen-7B)", teacher_path,
            problems, max_new_tokens, device, is_teacher=True,
            dataset_name=dataset_name,
            num_samples=args.num_samples, temperature=args.temperature, top_p=args.top_p, k=k,
            difficulty=args.difficulty, seed=args.seed,
        ))

    out_dir = Path(config["evaluation"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results_json = out_dir / "comparison.json"
    results_json.write_text(json.dumps(summaries, indent=2))
    print(f"Full results → {results_json}")

    if len(summaries) >= 2:
        out_png = Path(args.out) if args.out else out_dir / "comparison.png"
        plot_comparison(summaries, out_png)

    print(f"\n{'═' * 78}")
    print(f"  {'Model':<48}  {'test-case':>10}  {'solved':>10}")
    print(f"{'─' * 78}")
    for s in summaries:
        print(
            f"  {s['name']:<48}"
            f"  {s['test_pass_rate']:>9.1%}"
            f"  {s['problems_solved']:>3}/{s['num_problems']}"
        )
    print(f"{'═' * 78}")

    diffs = [d["difficulty"] for d in summaries[0]["by_difficulty"]]
    if len(diffs) > 1:
        print(f"\n  Solved by difficulty:")
        print(f"  {'Model':<48}" + "".join(f"{d:>12}" for d in diffs))
        for s in summaries:
            cells = "".join(f"{b['solved']}/{b['num_problems']:}".rjust(12)
                            for b in s["by_difficulty"])
            print(f"  {s['name']:<48}{cells}")

    short = {"syntax_error": "syntax", "wrong_answer": "wrong_answer",
             "runtime_error": "runtime", "missing_function": "missing_fn",
             "timeout": "timeout"}
    print(f"\n  Failure causes (failing test cases, first sample; truncated = problems hitting the token cap):")
    print(f"  {'Model':<48}" + "".join(f"{short[c]:>14}" for c in _FAILURE_CATEGORIES) + f"{'truncated':>14}")
    for s in summaries:
        fc = s["failure_counts"]
        row = "".join(f"{fc.get(c, 0):>14}" for c in _FAILURE_CATEGORIES)
        print(f"  {s['name']:<48}{row}{s['truncated']:>14}")


if __name__ == "__main__":
    main()
