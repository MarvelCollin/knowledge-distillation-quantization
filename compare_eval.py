#!/usr/bin/env python3
"""
Three-way evaluation: original student | teacher | distilled student.
Uses the MBPP *test* split — never seen during training.

Usage (inside Docker):
    python compare_eval.py
    python compare_eval.py --distilled outputs/checkpoint-20
    python compare_eval.py --num-problems 30 --skip-teacher
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gc
import json
import re
import yaml
import torch
import argparse

from datasets import load_dataset
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from src.data.dataset import PROMPT_TEMPLATE
from src.evaluation.evaluator import run_test_cases


def _extract_function(code: str) -> str:
    """Keep only the first function definition; strip \\r\\n and trailing test/print code."""
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    lines = code.split('\n')
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


def _extract_fn_name(test_cases: list) -> str | None:
    for tc in test_cases:
        m = re.search(r'\bassert\s+(\w+)\s*\(', tc)
        if m:
            return m.group(1)
    return None


_SYSTEM = (
    "You are a coding assistant. Output ONLY a raw Python function definition. "
    "No explanation, no markdown, no triple backticks. Start directly with def."
)

_INTERMEDIATE = Path("outputs/eval/intermediate")

COLORS = {
    "Student (original)":  "#4e79a7",
    "Teacher (14B)":       "#f28e2b",
    "Student (distilled)": "#59a14f",
}
FALLBACK_COLOR = "#888888"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_test_problems(n: int) -> list:
    print(f"Loading {n} problems from MBPP test split (separate from training data)...")
    raw = load_dataset("google-research-datasets/mbpp", split="test")
    problems = []
    for item in raw:
        cases = [t for t in item.get("test_list", []) if isinstance(t, str) and t.strip()]
        if not cases or not item.get("code", "").strip():
            continue
        problems.append({
            "text": item["text"].strip(),
            "code": item["code"].strip(),
            "test_cases": cases,
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


def _load_intermediate(label: str) -> dict | None:
    p = _cache_path(label)
    if p.exists():
        return json.loads(p.read_text())
    return None


def _finalise(label: str, results: list) -> dict:
    tp = sum(r["passed"] for r in results)
    tt = sum(r["total"] for r in results)
    solved = sum(1 for r in results if r["solved"])
    n = len(results)
    summary = {
        "name": label,
        "test_pass_rate": tp / max(tt, 1),
        "problem_solve_rate": solved / max(n, 1),
        "total_passed": tp,
        "total_tests": tt,
        "problems_solved": solved,
        "num_problems": n,
        "per_problem": results,
    }
    print(f"\n  Test cases passed : {tp}/{tt}  ({summary['test_pass_rate']:.1%})")
    print(f"  Problems solved   : {solved}/{n}  ({summary['problem_solve_rate']:.1%})")
    _save_intermediate(label, summary)
    return summary


def _run_problem(model, tokenizer, prompt: str, test_cases: list, max_new_tokens: int,
                 is_teacher: bool, device) -> str:
    expected = _extract_fn_name(test_cases)

    if is_teacher:
        user_content = prompt
        if expected:
            user_content = prompt + f"\n\nName the function `{expected}`."
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_content},
        ]
        fmt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(fmt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        code = tokenizer.decode(gen, skip_special_tokens=True).strip()
        if not code.startswith("def "):
            code = "def " + code.lstrip()
        code = _extract_function(code)
        if expected:
            old = re.match(r'def (\w+)\(', code)
            if old and old.group(1) != expected:
                old_name = re.escape(old.group(1))
                code = re.sub(r'\b' + old_name + r'\b', expected, code)
        return code
    else:
        if expected and prompt.endswith("def "):
            primed = prompt[:-4] + f"def {expected}("
            code_prefix = f"def {expected}("
        else:
            primed = prompt
            code_prefix = "def "
        inputs = tokenizer(primed, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(gen, skip_special_tokens=True)
        if code_prefix == "def ":
            code = "def " + decoded.lstrip()
        else:
            code = code_prefix + decoded
        return _extract_function(code)


def evaluate_model(label: str, model_path: str, problems: list,
                   max_new_tokens: int, device, is_teacher: bool = False) -> dict:
    cached = _load_intermediate(label)
    if cached and cached.get("num_problems") == len(problems):
        print(f"\n  Loaded cached results for: {label}")
        return cached

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'═' * 62}")
    print(f"  Evaluating: {label}")
    print(f"  Model path: {model_path}")
    print(f"{'═' * 62}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
    if is_teacher:
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    if not is_teacher:
        model = model.to(device)
    model.eval()

    if torch.cuda.is_available():
        print(f"  GPU allocated: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    results = []
    for i, prob in enumerate(tqdm(problems, desc=label)):
        prompt = PROMPT_TEMPLATE.format(text=prob["text"])
        try:
            code = _run_problem(model, tokenizer, prompt, prob["test_cases"], max_new_tokens, is_teacher, device)
        except Exception as exc:
            tqdm.write(f"  ✗ [{i+1:>2}/{len(problems)}] generation error: {exc}")
            results.append({"idx": i, "passed": 0, "total": len(prob["test_cases"]),
                            "solved": False, "details": [], "code": ""})
            continue

        r = run_test_cases(code, prob["test_cases"])
        solved = r["passed"] == r["total"] and r["total"] > 0
        results.append({"idx": i, **r, "solved": solved, "code": code})
        tqdm.write(
            f"  {'✓' if solved else '✗'} [{i+1:>2}/{len(problems)}]"
            f"  {r['passed']}/{r['total']} test cases"
        )

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return _finalise(label, results)


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
         "Problem Solve Rate  (all tests pass)", "Solved (%)")

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
        ax_delta.set_xlabel("Problem index (MBPP test split)", color="#aaaaaa", fontsize=10)
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
    ax_heat.set_xlabel("Problem index (MBPP test split)", color="#aaaaaa", fontsize=10)
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
        f"{'Model':<30}  {'pass@1':>8}  {'test-case':>10}  {'solved':>10}",
        "─" * 64,
    ]
    for s in summaries:
        summary_lines.append(
            f"{s['name']:<30}  {s['problem_solve_rate']:>7.1%}"
            f"  {s['test_pass_rate']:>9.1%}"
            f"  {s['problems_solved']:>3}/{s['num_problems']}"
        )

    fig.suptitle(
        "QEAD Knowledge Distillation — Evaluation on MBPP Test Split\n"
        + "\n".join(summary_lines),
        color="white", fontsize=10, fontweight="bold",
        y=0.975, va="top", family="monospace",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nGraph saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-way model comparison on MBPP test split")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--distilled", default=None,
                        help="Path to distilled student checkpoint (auto-detected if omitted)")
    parser.add_argument("--num-problems", type=int, default=30,
                        help="Number of MBPP test problems to evaluate (default: 30)")
    parser.add_argument("--skip-teacher", action="store_true",
                        help="Skip teacher evaluation to save time (~1 min/problem)")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--out", default="outputs/eval/comparison.png")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_new_tokens = args.max_new_tokens or config["evaluation"]["max_new_tokens"]
    student_model  = config["student"]["model_name"]
    teacher_path   = f"/workspace/{config['teacher']['local_model_path']}"

    problems = load_test_problems(args.num_problems)

    summaries = []

    summaries.append(evaluate_model(
        "Student (original)", student_model,
        problems, max_new_tokens, device, is_teacher=False,
    ))

    if not args.skip_teacher:
        summaries.append(evaluate_model(
            "Teacher (14B)", teacher_path,
            problems, max_new_tokens, device, is_teacher=True,
        ))

    distilled_path = args.distilled
    if distilled_path is None:
        ckpt_dir = Path(config["training"]["output_dir"])
        checkpoints = sorted(
            ckpt_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]),
        )
        if checkpoints:
            distilled_path = str(checkpoints[-1])
            print(f"\nAuto-detected distilled checkpoint: {distilled_path}")
    if distilled_path:
        summaries.append(evaluate_model(
            "Student (distilled)", distilled_path,
            problems, max_new_tokens, device, is_teacher=False,
        ))
    else:
        print("\nNo distilled checkpoint found — run train.py first to generate one.")

    out_dir = Path(config["evaluation"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(summaries, indent=2))
    print(f"Full results → {out_dir / 'comparison.json'}")

    if len(summaries) >= 2:
        plot_comparison(summaries, Path(args.out))

    print(f"\n{'═' * 62}")
    print(f"  {'Model':<30}  {'pass@1':>8}  {'test-case':>10}")
    print(f"{'─' * 62}")
    for s in summaries:
        print(
            f"  {s['name']:<30}  {s['problem_solve_rate']:>7.1%}"
            f"  {s['test_pass_rate']:>9.1%}"
        )
    print(f"{'═' * 62}")


if __name__ == "__main__":
    main()
