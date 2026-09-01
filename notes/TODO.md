# TODO: path to submission (IEEE Access)

Last updated 2026-08-27.

**All required experiments are done, and all five writing items are done.**
What remains is one optional experiment, one in-flight experiment, and two
administrative items that need a human.

---

## Writing (5) — COMPLETE

- [x] **1. Add Stage 10 to the paper.** Done as Section "The Precision Dose-Response":
      ladder table, recovery-curve figure, double-draw table, three falsification checks,
      cosine transmission ladder, group-size negative control, bounded 4-9% threshold table,
      coherence/capability dissociation, deviations note.

- [x] **2. Add Stage 11 and the GPTQ scope limit.** Done as "Cross-Track Transfer of the
      Threshold" (honest non-replication, both tracks reported as brackets) and "Scope of the
      Weight-Space Account" (GPTQ preserves the gain on instruct; mechanism bounded to
      weight-error-minimising quantizers).

- [x] **3. Revise the erasure claim throughout.** Done in abstract, introduction contributions,
      results and conclusion. Both axes now stated everywhere: task complexity and quantizer family.

- [x] **4. Change the title.** Now "Efficient Reasoning for Competitive Programming: Precision
      Limits of Knowledge Distillation and Post-Training Quantization of Large Language Models".
      Only the connector changed; the original wording is otherwise intact.

- [x] **5. Polish.** Reproducibility statement added with a pre-registration record table.
      **The verified-116 sub-item was NOT done, and was reversed on evidence — see below.**

---

## Correction to this file's own earlier instruction

The previous version said to "swap the significance tables to verified-116, which is the
paper's headline metric". **That instruction was wrong and has been reversed.**

Checking `outputs/eval/broken_tests.json` against 98 eval records shows the 112 excluded
problems are not unsolvable. 110 of the 112 fail only because the shipped reference solution
does not define the function the tests call, which is a packaging defect. The tests themselves
are intact, and models pass them: **the teacher solves 33 of the 112**, and distilled students
solve 1 to 5 each.

Excluding them therefore discards real signal. The paper now uses the **full 228 split as the
primary scoring set**, with verified-116 reported as a robustness check and as the reference
whenever an absolute rate is compared against externally published numbers. Every conclusion is
identical on both denominators.

---

## Optional experiment (1)

- [ ] **Double-draw base-track W6 and W7** (~1 day GPU)
      Unchanged in substance. The paper now reports both transitions as brackets
      (base W5-W6, instruct W6-W7) and calls the one-bit offset suggestive rather than
      established, so skipping this costs only the sharper claim.

## In flight (1)

- [~] **Split-precision reconstruction.** Quantize the backbone at W4 while carrying the
      distillation update on its own scale: `scripts/quantize_delta.py`. Pre-registration is
      already written into the paper. Diagnostics measured: the update is 2.13% of a backbone
      step (matching the 2.0% recorded from an independent code path) against 2876% of its own
      step. Eval running; results section not yet written, so `sec:splitresult` is a dangling
      reference until it lands.

---

## Known limitations, all now declared in the paper

- Crossover cannot be located to one bit (±9 solve CI) — Sections "Cross-Track Transfer" and Limitations.
- Weight-space mechanism is RTN-scoped — Section "Scope of the Weight-Space Account".
- `kd_cos` is meaningless for calibration-dependent quantizers — same section.
- Simulated vs real toolchains disagree in opposite directions per track — Limitations, item 2.
- Instruct weight-geometry checkpoint provenance — now stated in the weight-space section.
- Real-toolchain RTN unavailable on the instruct track — Reproducibility Statement.
- 3B student and second model family — Limitations and Future Work, listed as foreclosed
  because the teacher weights were not retained and the cache is Qwen-vocabulary bound.
