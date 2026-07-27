import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures"

C_ORIG = "#9e9e9e"
C_DIST = "#2c7fb8"
C_TEACH = "#31a354"
C_BAD = "#d95f02"

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def fig1_methods_ceiling():
    # All KD variants on the instruct student, fullset 228, pass@5 (solve counts)
    rows = [
        ("Untrained\nbaseline", 23.5, C_ORIG),          # 22-25 band, plot midpoint
        ("Single-teacher\nlogit KD", 39, C_DIST),
        ("Rescored\nfull-dist KD", 38, C_DIST),
        ("Trace-length\nfiltered", 38, C_DIST),
        ("On-policy\nGKD", 38, C_DIST),
        ("Two-teacher\nmix", 38, C_DIST),
        ("Two-teacher\nseed 7 (final)", 36, C_DIST),
        ("Over-mixed\n(ratio > 4:1)", 29, C_BAD),
        ("Teacher\nR1-7B", 126, C_TEACH),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    xs = range(len(rows))
    ax.bar(xs, [r[1] for r in rows], color=[r[2] for r in rows], width=0.65)
    ax.axhspan(35, 39, color=C_DIST, alpha=0.12, zorder=0)
    ax.text(len(rows) - 0.4, 44, "KD ceiling band 35-39", color=C_DIST, va="center",
            ha="right", fontsize=9)
    ax.axhline(23.5, color=C_ORIG, ls="--", lw=1)
    for x, (_, v, _) in zip(xs, rows):
        ax.text(x, v + 1.5, f"{v:g}", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_ylabel("Problems solved / 228 (pass@5)")
    ax.set_title("Five independent KD methods converge to the same ceiling (Qwen2.5-Coder-1.5B-Instruct student)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_methods_ceiling.png", dpi=200)


def fig2_gap_study():
    # Distillation gap across student floors: general base vs instruct
    students = ["Qwen2.5-1.5B\n(general base)", "Qwen2.5-Coder-1.5B\n-Instruct"]
    orig = [12, 22]
    dist = [28, 36]
    teach = 126
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    xs = [0, 1]
    w = 0.32
    ax.bar([x - w / 2 for x in xs], orig, w, color=C_ORIG, label="Original")
    ax.bar([x + w / 2 for x in xs], dist, w, color=C_DIST, label="Distilled (two-teacher mix)")
    ax.axhline(teach, color=C_TEACH, ls="--", lw=1.2)
    ax.text(1.28, teach - 6, "Teacher R1-7B: 126", color=C_TEACH, fontsize=9, ha="right")
    for x, o, d in zip(xs, orig, dist):
        ax.text(x - w / 2, o + 2, str(o), ha="center", fontsize=10)
        ax.text(x + w / 2, d + 2, str(d), ha="center", fontsize=10)
        ax.annotate("", xy=(x + w / 2, d - 1), xytext=(x - w / 2, o + 8),
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
        ax.text(x, max(o, d) + 9, f"+{d - o} (x{d / o:.2f})", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(students)
    ax.set_ylim(0, 140)
    ax.set_ylabel("Problems solved / 228 (pass@5)")
    ax.set_title("Absolute distillation gain is capacity-bound (+14 to +16)\nregardless of student floor")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_gap_study.png", dpi=200)


def fig3_quantization():
    # Full 2x3 grid; multi-draw cells plotted as the mean with an error bar:
    # instruct-distilled INT8 {30, 35}, INT4 {17, 23}
    models = ["Instruct\noriginal", "Instruct\ndistilled", "Base\noriginal", "Base\ndistilled"]
    bf16 = [22, 36, 12, 28]
    int8 = [24, 32.5, 11, 31]
    int4 = [21, 20, 6, 7]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    xs = [0, 1, 2, 3]
    w = 0.26
    ax.bar([x - w for x in xs], bf16, w, color="#444444", label="bf16")
    ax.bar(xs, int8, w, color=C_DIST, label="INT8 (per-channel W8)")
    ax.bar([x + w for x in xs], int4, w, color=C_BAD, label="INT4 (group-128 W4)")
    ax.errorbar([1], [32.5], yerr=2.5, fmt="none", ecolor="black", capsize=3, lw=1)
    ax.errorbar([1 + w], [20], yerr=3, fmt="none", ecolor="black", capsize=3, lw=1)
    for x, b, i8, i4 in zip(xs, bf16, int8, int4):
        ax.text(x - w, b + 0.8, f"{b:g}", ha="center", fontsize=9)
        ax.text(x, i8 + (3.5 if x == 1 else 0.8), f"{i8:g}", ha="center", fontsize=9)
        ax.text(x + w, i4 + (3.8 if x == 1 else 0.8), f"{i4:g}", ha="center", fontsize=9)
    # distilled-vs-original gap annotations per track
    for track, (xo, xd) in {"instruct": (0, 1), "base": (2, 3)}.items():
        gap_bf = bf16[xd] - bf16[xo]
        gap_i4 = int4[xd] - int4[xo]
        ax.text((xo + xd) / 2, max(bf16[xd], int8[xd]) + 4.5,
                f"KD gap bf16: +{gap_bf}\nKD gap INT4: {gap_i4:+d}",
                ha="center", fontsize=8.5, style="italic")
    ax.set_xticks(xs)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 46)
    ax.set_ylabel("Problems solved / 228 (pass@5)")
    ax.set_title("INT8 preserves the distillation gain; INT4 erases it on both tracks")
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_quantization.png", dpi=200)


def fig4_int4_failure_mode():
    # Truncation counts expose the INT4 failure mode (degenerate loops)
    models = ["Instruct\noriginal", "Instruct\ndistilled", "Base\ndistilled"]
    bf16 = [5, 3, 2]
    int4 = [2, 22, 158]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    xs = [0, 1, 2]
    w = 0.32
    ax.bar([x - w / 2 for x in xs], bf16, w, color="#444444", label="bf16")
    ax.bar([x + w / 2 for x in xs], int4, w, color=C_BAD, label="INT4")
    for x, b, i4 in zip(xs, bf16, int4):
        ax.text(x - w / 2, b + 2, str(b), ha="center", fontsize=9)
        ax.text(x + w / 2, i4 + 2, str(i4), ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(models)
    ax.set_ylabel("Truncated samples / 1140")
    ax.set_title("INT4 failure mode: degenerate generation loops,\nnot wrong answers (truncation counts)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_int4_failure_mode.png", dpi=200)


def fig5_failure_mixture():
    # Per-test-execution category shares, base model, 116,170 executions per format
    formats = ["bf16\noriginal", "bf16\ndistilled", "INT8\ndistilled", "INT4\ndistilled"]
    cats = ["pass", "wrong_answer", "runtime_error", "missing_function", "syntax_error", "timeout"]
    shares = {
        "pass":             [10.9, 20.7, 20.4, 5.2],
        "wrong_answer":     [60.7, 62.7, 62.7, 38.2],
        "runtime_error":    [12.7, 12.7, 13.8, 30.3],
        "missing_function": [14.0, 0.9, 0.6, 21.4],
        "syntax_error":     [0.1, 0.0, 0.1, 1.7],
        "timeout":          [1.6, 3.0, 2.4, 3.2],
    }
    colors = {
        "pass": C_TEACH, "wrong_answer": "#fdae6b", "runtime_error": C_BAD,
        "missing_function": "#756bb1", "syntax_error": "#e7298a", "timeout": "#666666",
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = range(len(formats))
    bottom = [0.0] * len(formats)
    for c in cats:
        ax.bar(xs, shares[c], 0.6, bottom=bottom, color=colors[c], label=c)
        bottom = [b + v for b, v in zip(bottom, shares[c])]
    for x, f in zip(xs, formats):
        ax.text(x, shares["pass"][x] / 2, f"{shares['pass'][x]:g}%", ha="center",
                fontsize=8.5, color="white", fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(formats)
    ax.set_ylabel("Share of test executions (%)")
    ax.set_ylim(0, 100)
    ax.set_title("INT4 changes the kind of failure, not just the amount\n(base model, 116k test executions per format)")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_failure_mixture.png", dpi=200)


if __name__ == "__main__":
    import os

    os.makedirs(OUT, exist_ok=True)
    fig1_methods_ceiling()
    fig2_gap_study()
    fig3_quantization()
    fig4_int4_failure_mode()
    fig5_failure_mixture()
    print(f"wrote 5 figures to {OUT}/")
