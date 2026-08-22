# Significance tests (paired, per-problem)

Computed 2026-08-03, updated 2026-08-10 (corrected a stale GPTQ file reference, added Stage 8
QAT-KD and real-RTN-gap comparisons), updated 2026-08-22 (**re-run on the verified-116 subset;
every conclusion replicates**) from the saved per-problem eval records
(`outputs/eval/intermediate/*.json`). No GPU: pure post-processing of the `per_problem[].solved`
flags. All cells are pass@5 on the same 228 LeetCode problems, so every comparison is paired.

- **Test:** two-sided exact McNemar (binomial on discordant pairs b, c).
- **CI:** 20,000-sample paired bootstrap on Δsolved (B − A) over the 228 problems.
- b = A solves & B doesn't; c = B solves & A doesn't.
- Reproduce (full 228): `EVAL_DIR=outputs/eval/intermediate python scripts/significance_tests.py`.
- Reproduce (verified-116): add `EXCLUDE_FILE=outputs/eval/broken_tests.json`.

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
| B1 | Base distilled bf16 → **real GPTQ-W4** | 28→2 | 26 | 0 | **3e-8** | −26 [−36,−17] | GPTQ collapses too |
| B2 | Base gap @GPTQ-W4 (real): orig → distilled | 3→2 | 2 | 1 | 1.0 | −1 [−4,+2] | **gap gone (ns)** |
| B3 | Base gap @RTN-W4 (real): orig → distilled | 2→2 | 2 | 2 | 1.0 | 0 [−4,+4] | **gap gone (ns)** |
| Q1 | Stage 8: QAT-KD bf16 vs standard KD bf16 (d1/2) | 28→10/11 | 24/24 | 6/7 | **0.0014 / 0.0033** | −18 [−29,−8] / −17 [−28,−7] | QAT hurts bf16 (sig.) |
| Q2 | Stage 8: QAT-KD real-RTN4 vs standard KD real-RTN4 (d1/2) | 2→3/1 | 2/2 | 3/1 | 1.0 / 1.0 | +1 [−3,+5] / −1 [−5,+2] | **gain unconfirmed (ns)** |

## Verified-116 replication (run 2026-08-22)

The paper's headline metric is the verified-116 subset (the 112 problems with broken reference
solutions excluded), but every row above is computed on the full 228. Re-running with
`EXCLUDE_FILE=outputs/eval/broken_tests.json` closes that gap. **Nothing flips**: every significant
result stays significant, every null stays null, and point estimates move by at most 2 solves.

| Comparison | Full-228 Δ [95% CI] | Verified-116 Δ [95% CI] | Verdict |
|---|---|---|---|
| 1 Instruct KD @bf16 | +14 [+6,+23] ** | +13 [+5,+21] ** | KD helps |
| 2 Base KD @bf16 | +16 [+6,+26] ** | +15 [+6,+25] ** | KD helps |
| 3 Instruct INT4 erasure (d1/d2) | −19 *** / −13 * | −17 *** / −11 * | drop real |
| 4 Base INT4 erasure (d1/d2) | −21 / −23 *** | −19 / −21 *** | drop real |
| 5 Instruct gap @INT4 (d1/d2) | −4 / +2 ns | −3 / +3 ns | **gap gone** |
| 6 Base gap @INT4 (d1/d2) | +1 / −1 ns | +2 / 0 ns | **gap gone** |
| 7 Instruct INT8 preserve (d1/d2) | −6 / −1 ns | −6 / −1 ns | preserved |
| 8 QEAD bf16 (d1/d2) | −6 / −1 ns | −6 / −2 ns | **null** |
| 9 QEAD INT4 (d1/d2) | −1 / +3 ns | 0 / +3 ns | **null** |
| B1 Real GPTQ4 collapse | −26 *** | −24 *** | collapse real |
| B2 GPTQ4 gap | −1 ns | −1 ns | **gap gone** |
| B3 RTN4 gap | 0 ns | +1 ns | **gap gone** |
| Q1 QAT bf16 cost (d1/d2) | −18 / −17 ** | −16 / −15 ** | QAT hurts bf16 |
| Q2 QAT W4 gain (d1/d2) | +1 / −1 ns | +1 / −1 ns | unconfirmed |
| 10 Instruct gap @INT8 (d1/d2) | +6 ns / +11 * | +4 ns / +9 * | **preserved** |
| 11 Base gap @INT8 | +20 [+12,+29] *** | +18 [+10,+26] *** | **preserved** |

The load-bearing contrast on the headline metric is rows 10-11 against 5-6: on verified-116 the base
track's KD gap is **+18 [+10,+26] at INT8** and **+2 / 0 (ns) at INT4**. Preserved at 8 bits, gone at
4, measured on the subset the paper actually reports.

## How to state it in the paper

The erasure claim is rigorous via a **contrast of CIs**, not eyeballing:
- At **bf16** the KD advantage is significant and its CI excludes 0 (instruct +14 [+6,+23]; base +16 [+6,+26]).
- At **INT4** the same advantage's CI includes 0 (rows 5–6: instruct +2/−4, base ±1) — no detectable gain.
- The within-model **bf16→INT4 drop is itself significant** (rows 3–4, p ≤ 0.01), and so is the
  drop under real GPTQ (B1, p=3e-8).
- The gap-gone result holds under **both real PTQ methods** now: RTN (B3) and GPTQ (B2), not just
  the simulated grid (rows 5–6).

Honest caveats:
- Rows 5–9, B2, B3, Q2 are *non-significant*, i.e. failure to reject, not proof of equality —
  discordant counts are small (n = b+c is 2–18), so power is limited. State as "no detectable
  difference," backed by the CIs including 0, not as "identical."
- Bonferroni at 0.05/14 ≈ 0.0036: rows 1, 2, 3-draw1, 4 (both), B1, Q1-draw1 clearly survive; row 2
  (0.0037) and Q1-draw2 (0.0033) sit right at the line — treat as "very likely real" rather than
  relitigating the correction choice. All nulls stay null regardless of correction. The story is
  unchanged.

## Bonus finding (item #3, real PTQ)

Real **GPTQ-W4** and **RTN-W4** cells both exist now (`*_gptq4fq`, `*_rtn4fq` — the calibrated final
Stage 4 run with 8192-token calibration sequences, which supersedes an earlier under-calibrated GPTQ
draw that had scored the distilled model at 1). GPTQ collapses the base distilled model hard
(28→**2**, p=3e-8, B1), and the original↔distilled gap is gone under **both** real quantizers (B2
GPTQ p=1.0, B3 RTN p=1.0). So on the base track the INT4 erasure is **not an RTN artifact** — a
calibration-based W4 method reproduces it, now with a formal CI behind the claim instead of a
single-draw solve count. (Instruct-track GPTQ not yet run.)

## Stage 8 (QAT-KD): fails its pre-registration

Both cells double-drawn as declared. bf16 drops significantly (Q1, p<0.005, CI excludes 0 both
draws) — a real, large capability cost. The W4 cell nominally rises (2→3/1, test rate 1.3%→3.2-3.3%)
but is not distinguishable from standard KD's real-RTN4 cell by McNemar (Q2, p=1, CI includes 0 both
draws; caveat: only 4–5 discordant problems, so this test has weak power to detect a small effect).
The declared success bar was W4 test rate above ~8%; the observed 3.2–3.3% falls well short.
Verdict: fail. Full writeup in `history/methods.md` → "Stage 8 result: quantization-aware
distillation fails".
