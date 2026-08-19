# Progress Report — August 2026

## One-line summary

We tried four ways to save the distilled knowledge from 4-bit quantization, all failed, and we proved *why* it fails: the knowledge is a nudge to the weights too small to survive 4-bit rounding. INT8 keeps it fully — that is the deployment recommendation.

---

## The finding, in plain terms

You distill knowledge into a small model and it goes from 12 → 28 solved. Then you shrink the model to 4-bit to make it small and fast, and the 28 falls back to ~6. **4-bit quantization erases what distillation taught it.**

August was spent doing two things: (1) trying to rescue the 4-bit knowledge from every angle, and (2) proving the mechanism behind the collapse.

- **Three rescues, all failed.** QEAD (train-side token weighting), GPTQ (smart shrinking), and QAT (train through the quantizer). None recovered the 4-bit knowledge. The collapse cannot be fixed from the training side *or* the shrinking side.
- **Why it fails — the step-size mechanism.** The distilled knowledge is a *tiny nudge* to the weights: only ~2% of a single 4-bit rounding step, but ~26% of an 8-bit step. So 4-bit rounding steamrolls the nudge; 8-bit keeps it.
- **Double-proven.** Seen in the weights (the tiny nudge) *and* confirmed in behavior (under 4-bit, the distilled and original models start predicting the same tokens again — they converge).
- **Not absolute — it depends on the task.** The erasure is complete on hard competitive-programming problems (LeetCode) but only partial on shorter, simpler function-completion tasks (MBPP+ keeps ~half the gain).
- **The escape hatch: use INT8.** 8-bit preserves the knowledge fully on every model tested. The recommendation is: distill to 1.5B, serve at INT8, skip 4-bit.

---

## What was done, in order

| Date | Work | Result |
|---|---|---|
| Aug 2 | Weight-level mechanism analysis | Disproved "distilled weights are spiky"; found the step-size mechanism (the *why*) |
| Aug 3 | Paired significance tests | Every result now has p-values + confidence intervals, not eyeballing |
| Aug 4 | Real GPTQ vs RTN quantization | Smart shrinking (GPTQ) erases the knowledge too — not an artifact of dumb rounding |
| Aug 10 | Quantization-aware distillation (QAT) | Failed its pre-registration — hurt normal accuracy without saving 4-bit |
| Aug 11 | Second/third benchmarks (HumanEval+, MBPP+) | Reframed the claim: erasure is task-complexity-dependent, not absolute |
| Aug 11 | Activation-side probe | Behavioral confirmation of erasure (models converge under 4-bit) |
| Aug 11 | Efficiency table | The payoff: student ~5× smaller, ~2.4× faster than the teacher |
| Aug 14–15 | Precision dose-response setup | Generalized quantizer (any bit-width, any group-size) + pre-registration written; **not run yet** |

---

## Key numbers

### The erasure survives real, smart quantization (base track)

| Precision | Original | Distilled | KD gap | Significance |
|---|---|---|---|---|
| bf16 | 12 | 28 | **+16** | p=0.004, CI [+6,+26] |
| RTN-W4 (dumb) | 2 | 2 | 0 | p=1, CI [−4,+4] |
| GPTQ-W4 (smart) | 3 | 2 | −1 | p=1, CI [−4,+2] |

The bf16 gain is real and significant; at 4-bit it is gone under both quantizers. Calibration buys nothing.

### The mechanism (why)

- Distilled weights are **not** sharper or more outlier-heavy than originals — identical to five significant figures.
- KD update size: **~2% of a 4-bit step, ~26% of an 8-bit step.**
- Direction survival (cosine): **0.19 at W4** (lost) vs **0.72 at W8** (kept).
- QEAD-on vs QEAD-off weight difference is 39–90× *smaller* than the 4-bit perturbation — it was mathematically incapable of protecting anything.

### The task-dependence (generality)

| Benchmark | Task type | bf16 gap | W4 gap | Verdict |
|---|---|---|---|---|
| LeetCode | Hard competitive programming | +16 | ~0 | Fully erased |
| MBPP+ | Simple function completion | +8.7 pts | +5.0 pts | Retains ~57% (significant) |
| HumanEval+ | Simple function completion | +3.7 pts | +1.8 pts | Same trend, not significant (small N) |

### The payoff (why it matters)

| Model | Size (bf16) | Speed | vs teacher |
|---|---|---|---|
| Teacher (R1-7B) | 14.19 GB | 994 tok/s | — |
| Student (1.5B) | 2.88 GB | 2417 tok/s | ~5× smaller, ~2.4× faster |
| Student at INT8 (theoretical) | 1.44 GB | — | ~10× smaller than teacher bf16 |

---

## Where the paper stands

The four risks flagged for IEEE Access earlier are now substantively addressed:

1. **"You only tested dumb quantization (RTN)"** → GPTQ run, erasure survives.
2. **"Only one benchmark"** → HumanEval+ and MBPP+ added (and they reshaped the claim for the better).
3. **"The mechanism is asserted, not measured"** → measured in weights *and* confirmed in behavior.
4. **"The efficiency framing is unbacked"** → efficiency table added.

The paper moved from "at the bar on findings, not on framing" to "genuinely close." What remains is finishing one experiment, one reframe, and the writing pass.

---

## Next steps

**1. Run the precision dose-response sweep (Stage 10) — highest value.**
The tools are built and the prediction is pre-registered (knowledge should return around 6–7 bit). Running it turns the step-size story from "an explanation that fits our two data points" into "a theory that predicted exactly where the knowledge comes back." This is the single strongest thing left to do, and it is ready to launch. Fill in the 5, 6, 7-bit cells on the base track, double-draw the two cells around the crossover.

**2. Run GPTQ on the instruct track — cheap gap-filler.**
Real GPTQ has only been run on the base track. The instruct track still has simulated-only numbers. One quantize + eval closes the asymmetry and lets the significance table cover both tracks under real PTQ.

**3. Reconcile the "erasure" → "task-complexity-dependent erasure" reframe across all docs and the title.**
The July notes and the working title still say "erasure," full stop. The MBPP+ result means the honest claim is *"strongest on complex tasks, partial on simple ones."* Every doc, the abstract, and the title need to say the same thing before drafting. (No GPU — writing only.)

**4. Verify the instruct checkpoint caveat — data integrity.**
The weight-mechanism analysis used `outputs/final_last` because `final_last_seed7` no longer exists on disk. Confirm the instruct mechanism row describes the checkpoint that actually scored 36, or re-run it on the correct one. Base track is unaffected.

**5. Re-run significance on the verified-116 subset.**
The paired tests currently cover the full 228. The paper's headline metric is verified-116, so the significance table should be regenerated on that subset too. Cheap post-processing, no GPU.

**6. Presentation pass for IEEE Access.**
Assemble the notes into IEEE format: defined notation, every claim tied to a numbered table/figure, a reproducibility statement (you are well-positioned — fixed caches, seeds, released configs). This venue desk-rejects on presentation, so budget real time here.

**Order:** start #1 (it needs GPU time and is the biggest scientific win), do #3 and #5 in parallel while it runs (writing only), then #2 and #4, then #6 once the numbers are frozen.
