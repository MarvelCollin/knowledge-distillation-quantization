# TODO: path to submission (IEEE Access)

Last updated 2026-08-23.

**All required experiments are done.** Nothing below blocks a submission except the writing.
One optional experiment remains, and it buys rigor rather than a new finding.

---

## Required: none (measurement is complete)

Every claim in the paper is backed by data already on disk. The full evidence base lives in
`outputs/eval/intermediate/` (83 files, kept local as of 2026-08-23), with statistics in
`notes/significance_tests.md` and narrative in `history/methods.md`.

---

## Optional experiment (1)

- [ ] **Double-draw base-track W6 and W7** (~1 day GPU)
      Base W6 is the cell recorded as the base crossover, and it was never double-drawn; the
      instruct cells were, and did not replicate. Publishing a cross-track offset where one side
      had half the scrutiny is a reviewer target. Running this either puts both tracks on equal
      footing, or confirms that both should be reported as brackets.
      **If you skip it:** report both transitions as brackets (base W5-W6, instruct W6-W7) and
      call the one-bit offset suggestive rather than established. This is already how
      `history/methods.md` states it, so skipping costs nothing but the sharper claim.

---

## Writing (5) — this is the real remaining work

- [ ] **1. Add Stage 10 to the paper.** A results subsection plus the recovery-curve figure.
      Currently absent entirely, and it is the strongest result in the project: a pre-registered
      dose-response with a bounded 4-9% step-fraction threshold and a group-size negative control.

- [ ] **2. Add Stage 11 and the GPTQ scope limit.** Stage 11 as the honest non-replication plus
      the bracketed cross-track statement; GPTQ as the boundary of the weight-space mechanism
      (it perturbs weights *more* than RTN yet preserves the gain, because it minimises output
      error, not weight error).

- [ ] **3. Revise the erasure claim throughout.** The biggest item. The draft argues erasure
      flatly, but it is now conditional on two axes:
      - **Task complexity** — MBPP+ retains ~57% of the gap at W4 (significant), LeetCode erases it.
      - **Quantizer family** — real GPTQ retains it on the instruct track; simulated RTN erases it.
      Touches the abstract, introduction, results and conclusion. Not a find-and-replace.

- [ ] **4. Change the title.** It still promises QEAD, which the project's own ablation nulled.

- [ ] **5. Polish.** 167 overfull-hbox LaTeX warnings (content running past the column edge),
      a reproducibility statement (strong here: fixed offline caches, declared seeds, released
      configs, and pre-registrations timestamped in git history), and swap the significance
      tables to verified-116, which is the paper's headline metric.

---

## Known limitations to declare in the paper

Not tasks. Write them down rather than leaving them for a reviewer to find.

- **Crossover cannot be located to one bit.** At 228 problems with gaps of +6 to +14 solves, the
  paired bootstrap CI is about ±9 solves, wider than the difference between adjacent bit-widths
  in the transition region. The design shows recovery happens over a two-bit window; resolving a
  single bit would need more problems or more samples per problem.
- **Weight-space mechanism is RTN-scoped.** It covers the simulated grid, the ladder, the
  group-size control and the QAT failure. It does not transfer to activation-aware quantizers.
- **`kd_cos` is meaningless for calibration-dependent quantizers.** For two independently GPTQ'd
  models the weight delta is dominated by each model's own Hessian decisions.
- **Simulated vs real toolchains disagree, in opposite directions per track.** Instruct: simulated
  17/23 vs real GPTQ 29/31. Base: simulated 7/5 vs real 2. The headline grid is simulated throughout.
- **No `final_last_seed7`.** The instruct weight-geometry row used `outputs/final_last`; the
  checkpoint named in earlier notes is absent from disk and from the Drive backup, so its eval
  number could not be re-verified. Base track unaffected.
- **Real-toolchain RTN unavailable on the instruct track.** `--method rtn` with `--save-mode fake`
  is a silent no-op on current llm-compressor. Gated now by `scripts/probe_quant_grid.py`.
- **3B student and second model family** — cut for VRAM and for the Qwen-vocabulary teacher cache
  respectively; see `notes/plan-phase4.md`.
