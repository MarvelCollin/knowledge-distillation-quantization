# Significance tests (paired, per-problem)

Computed 2026-08-03 from the saved per-problem eval records (`outputs/eval/intermediate/*.json`,
restored from `gdrive:kd-backup`). No GPU — pure post-processing of the `per_problem[].solved`
flags. All cells are pass@5 on the same 228 LeetCode problems, so every comparison is paired.

- **Test:** two-sided exact McNemar (binomial on discordant pairs b, c).
- **CI:** 20,000-sample paired bootstrap on Δsolved (B − A) over the 228 problems.
- b = A solves & B doesn't; c = B solves & A doesn't.
- Reproduce: `gdrive-pull outputs/eval/intermediate` then `EVAL_DIR=outputs/eval/intermediate python scripts/significance_tests.py`.

| # | Comparison (A → B) | solved | b | c | McNemar p | Δsolved [95% CI] | verdict |
|---|---|---|---|---|---|---|---|
| 1 | Instruct KD @bf16: orig → distilled | 22→36 | 3 | 17 | **0.0026** | +14 [+6, +23] | KD helps (sig.) |
| 2 | Base KD @bf16: orig → distilled | 12→28 | 6 | 22 | **0.0037** | +16 [+6, +26] | KD helps (sig.) |
| 3 | Instruct: distilled bf16 → INT4 (draw1/2) | 36→17/23 | 20/18 | 1/5 | **2e-5 / 0.011** | −19 [−28,−11] / −13 [−23,−4] | INT4 drop (sig.) |
| 4 | Base: distilled bf16 → INT4 (draw1/2) | 28→7/5 | 22/23 | 1/0 | **6e-6 / 2e-7** | −21 [−30,−12] / −23 [−32,−15] | INT4 drop (sig.) |
| 5 | Instruct gap @INT4: orig → distilled (d1/2) | 21→17/23 | 9/8 | 5/10 | 0.42 / 0.81 | −4 [−11,+3] / +2 [−6,+10] | **gap gone (ns)** |
| 6 | Base gap @INT4: orig → distilled (d1/2) | 6→7/5 | 2/2 | 3/1 | 1.0 / 1.0 | +1 [−3,+5] / −1 [−5,+2] | **gap gone (ns)** |
| 7 | Instruct INT8 preserve: bf16 → INT8 (d1/2) | 36→30/35 | 8/7 | 2/6 | 0.11 / 1.0 | −6 [−12,0] / −1 [−8,+6] | preserved (ns) |
| 8 | QEAD bf16 (instruct): on → off (d1/2) | 36→30/35 | 9/8 | 3/7 | 0.15 / 1.0 | −6 [−13,0] / −1 [−9,+7] | **null (ns)** |
| 9 | QEAD INT4 (instruct): on → off (d1/2) | 17→16/20 | 3/3 | 2/6 | 1.0 / 0.51 | −1 [−5,+3] / +3 [−3,+9] | **null (ns)** |
| B1 | Base distilled bf16 → **real GPTQ-W4** | 28→1 | 27 | 0 | **1.5e-8** | −27 [−37,−18] | GPTQ collapses too |
| B2 | Base gap @GPTQ-W4: orig → distilled | 1→1 | 1 | 1 | 1.0 | 0 [−3,+3] | **gap gone (ns)** |

## How to state it in the paper

The erasure claim is now rigorous via a **contrast of CIs**, not eyeballing:
- At **bf16** the KD advantage is significant and its CI excludes 0 (instruct +14 [+6,+23]; base +16 [+6,+26]).
- At **INT4** the same advantage's CI includes 0 (rows 5–6: instruct +2/−4, base ±1) — no detectable gain.
- The within-model **bf16→INT4 drop is itself significant** (rows 3–4, p ≤ 0.01).

Honest caveats:
- Rows 5–9 are *non-significant*, i.e. failure to reject, not proof of equality — discordant counts
  are small (n = b+c is 3–18), so power is limited. State as "no detectable difference," backed by
  the CIs including 0, not as "identical."
- Bonferroni at 0.05/11 ≈ 0.0045: rows 1, 2, 3-draw1, 4 (both), B1 survive; row 3-draw2 (0.011) does
  not, but its companion draw does. All nulls stay null. The story is unchanged.

## Bonus finding (item #3, real PTQ)

Real **GPTQ-W4** cells already exist (`*_gptq4`). GPTQ collapses the base distilled model *harder*
than naive RTN (28→**1** vs RTN 28→7; p=1.5e-8), and the original↔distilled gap is gone at GPTQ too
(B2, p=1.0). So on the base track the INT4 erasure is **not an RTN artifact** — a calibration-based
W4 method reproduces it. (Instruct-track GPTQ not yet run.)
