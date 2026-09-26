# Ku-Mellin-PATP — Reproducibility Repository

Verification and reproducibility code for the paper:

> **Closed-form moment propagation through fractional-power response surfaces
> via the Mellin transform**
> Serhii Zabolotnii, 2026.

**Repository:** <https://github.com/SZabolotnii/Ku-Mellin-PATP-code-supplement>
(`git clone https://github.com/SZabolotnii/Ku-Mellin-PATP-code-supplement.git`)

**One-line summary.** Classical moment-based uncertainty evaluation (MUET) queries
the Mellin transform `M_X(s) = E[X^{s-1}]` of the input density on the *integer
lattice* `s = k+1`. The signed-parity fractional-power **PATP** basis queries the
*same* `M_X` at *real* arguments `s = p_i(α)·k + 1` — an off-lattice continuation —
yielding a closed-form moment-propagation formula for fractional-power response
surfaces that polynomial MUET cannot reach. This repository reproduces every
numerical claim in the paper and carries the Lean 4 proofs of the supporting lemmas.

**Versions.** Version 2.0.0 corresponds to the **revised** manuscript. Version 1.2.0
(last commit `dd4bee1`) corresponds to the submitted version; its scripts are kept
unchanged (see [below](#scripts-of-the-submitted-version-kept)) as the record of that
version.

---

## What this repository reproduces

Map of the revised manuscript (its Table `tab:repro`). Table, figure and section numbers
follow the revised draft; the LaTeX labels in brackets are the stable reference. Output
paths are relative to the repository root.

| Manuscript artifact (revised) | Script | Output |
|---|---|---|
| **Table 1** (`tab:validity`) and the mixed-κ counterexample, **§5** Region of validity | `verification/cas/mixed_kappa_validity.py` | `verification/cas/mixed_kappa_validity.txt` |
| **Table 2** (`tab:mellin-distros`), **Appendix A** (`app:tn`, truncated normal) | `verification/cas/truncnormal_mellin.py` | `verification/cas/truncnormal_mellin.txt` |
| Pareto example, **§5.2** (`sec:heavytail`) | `verification/cas/heavy_tail_admissibility.py` | console |
| **Corollary 10** (`cor:removable`), **Figure 2** (`fig:cond-alpha`), **§5.3** (`sec:confluent`) | `experiments/p1_confluent_limit.py` | `experiments/results/p1_confluent.txt`, `experiments/results/p1_cond_vs_alpha.csv`, `outputs/fig_cond_alpha.pdf` |
| **Tables 3, 4, 6** (`tab:rq3`, `tab:rq3-baselines`, `tab:twod`), **§6.1–6.2**, **§6.4** (`sec:precision`), **§6.5** | `experiments/p3_reverify.py` | `experiments/results/p3_reverify.txt` |
| **Table 5** (`tab:balanced`), **§6.3** | `experiments/p3_balanced_benchmark.py` | `experiments/results/p3_balanced.txt` |
| **Table 7** (`tab:heavytail-bench`), **§6.6** | `experiments/p4_heavy_tail.py` | `experiments/results/p4_heavy_tail.txt` |
| **Tables 8, 9** (`tab:eng-q1`, `tab:eng-sweep`), **Figure 3** (`fig:engineering`), **§7** | `experiments/p2_engineering_cb.py` | `experiments/results/p2_engineering.txt`, `outputs/fig_engineering.pdf` |
| **§8** (`sec:realdata`): **Table 10** (`tab:realdata-decision`), **Figure 4** (`fig:realdata`); **Appendix E** (`app:realdata`): **Tables 12, 13** (`tab:realdata`, `tab:realdata-secondary`) | `experiments/realdata_split.py` (needs the CWRU data, see [`data/README.md`](data/README.md)) | `experiments/results/realdata_split.txt`, `outputs/fig_realdata.pdf` |
| Lean lemmas, **Appendix C** (`app:lean`) | `lake build PATP`; `./check_no_sorry.sh` | build log |
| Background to §5.3 and §6 (revision step 0: capacity-matched re-run of the benchmark and the α ≈ 1/2 mechanism check); not a table of the paper | `experiments/step0_capacity_matched.py`, `experiments/step0_mechanism.py` | `experiments/results/step0.txt`, `experiments/results/step0_mechanism.txt` |

Figure 1 (`fig:workflow`) is a schematic drawn in TikZ and has no script. Sections 1–4
carry background and proofs with no computed results; the numbers quoted in the
Limitations (§9) and the Conclusion (§10) are those of the rows above.
`step0_capacity_matched.py` is also a module: `p1_confluent_limit.py`,
`p2_engineering_cb.py` and `p3_balanced_benchmark.py` import its grid, exponent map and
α-optimiser.

### Output convention

- Scripts run from their own directory (`experiments/` or `verification/cas/`), as each
  header says; `run_all.py` does this.
- `experiments/results/<name>.txt` is the script's standard output, captured by
  `run_all.py` (the file is replaced only when the step exits 0). The two CAS scripts
  write `verification/cas/<name>.txt` themselves.
- These text outputs are **committed** — they are the files in which every number quoted
  in the paper can be found — so after a run `git diff` shows whether anything moved.
  Only wall-clock lines (`elapsed … s`, `wall time … s`, `… ms (min of 5)`) are expected
  to change.
- Figures are written to `outputs/` (git-ignored). The paper's copies are `fig_*.pdf`
  with the same names.

### Scripts of the submitted version (kept)

None of the scripts below is listed in the revised manuscript's `tab:repro`. They are
kept unchanged as the record of the submitted version, and because
`p3_reverify.py` imports `rq3_demo`, `twod_demo`, `alpha_sensitivity` and
`heavy_tail_admissibility` as modules: it fits each surrogate exactly as the submitted
scripts did and re-evaluates the moments in 50-digit arithmetic.

| Script | Artifact of the **submitted** version | Status in the revision |
|---|---|---|
| `experiments/rq3_demo.py` | Table 3 — manufactured-solution benchmark (float64 closed form) | superseded by `p3_reverify.py` (50-digit evaluation; the published M4 improvement at `j = 4`, −16 %, is +70.7 % in 50 digits) |
| `experiments/make_figure.py` | Figure 1 — relative error vs moment order (`outputs/fig_rq3.pdf`) | not in the revision |
| `experiments/twod_demo.py` | Table 4 — separable 2-D propagation | superseded by `p3_reverify.py` |
| `experiments/jorder_stress.py` | §7 — higher moment orders (`j <= 8`), conditioning | superseded by `p3_reverify.py` |
| `experiments/alpha_sensitivity.py` | §7 — stability of `α*` across optimiser settings | superseded by `p3_reverify.py` |
| `experiments/realdata_case.py` | Table 5 / §8 — in-sample real-data study (105 and 100, iid bootstrap `B=300`) | superseded by `realdata_split.py` (temporal split, records 105 and 106) |
| `experiments/mellin_error.py` | §8 — empirical-Mellin estimator error on the CWRU sample | not in the revision |
| `verification/cas/region_of_validity.py` | Table 1 — quadratic structure of `p_i(α)` | superseded by `mixed_kappa_validity.py`, which re-derives the same table in exact arithmetic |
| `verification/cas/multinomial_validity.py` | §5 — 765-pair grid scan of the multinomial validity minimum | superseded by `mixed_kappa_validity.py`, which finds 11 mixed tuples (e.g. `κ₁₀ = 1, κ₉ = 2`) where every single-index row passes but the mixed Mellin argument leaves the strip |
| `verification/cas/mellin_table.py` | Table 2 — closed-form Mellin transforms (uniform, Beta, triangular, truncated normal) | still valid for the closed-form rows; the truncated-normal rows and Appendix A are now `truncnormal_mellin.py` |
| `verification/cas/check_patp.py` | CAS cross-check of T1 (corner values), T2 (positivity), T3 (Mellin power rule) and the closed form vs Monte-Carlo | still valid; supporting check, not listed in `tab:repro` |

The companion `experiments/Results-*.md` files document the submitted version. The
float64 values of these scripts that sit on ill-conditioned sums change with the
NumPy/LAPACK build. For most cells only the last digits move (for example
`jorder_stress.py` at `j >= 5`), but the M4 row of `rq3_demo.py` at `j = 4` flips the sign
of its printed improvement: −15.6 % in the submitted run, +45.5 % with the pinned
versions of `requirements.txt`. That sensitivity is why the revision re-evaluates every
such number in 50-digit arithmetic; `p3_reverify.py` prints the published value, today's
float64 value and the 50-digit value (+70.7 %) side by side.

## Formal verification (Lean 4 / Mathlib v4.26.0)

The four supporting lemmas are formalized in Lean 4 and are **included here** under
`Lean/`:

| File | Lemma |
|---|---|
| `Lean/PATP/Param.lean` | **T1** — corner values `p_i(0)=1/i`, `p_i(1/2)=1`, `p_i(1)=i` |
| `Lean/PATP/Positivity.lean` | **T2** — `p_i(α) > 0` on `[0,1]` for `i ≤ 5`; witness `p_6(3/20) = -1/48 < 0` |
| `Lean/PATP/MellinPower.lean` | **T3** — Mellin power-substitution rule for `Y = X^c` |
| `Lean/PATP/AlphaOneReduction.lean` | **T4** — reduction of the PATP basis at `α = 1` |
| `Lean/PATP/Basis.lean` | the signed-parity family `φ_i(ξ;α)` (odd symmetry, used by T4) |

```bash
lake build PATP        # needs elan/lake; the Mathlib pin is in lake-manifest.json
./check_no_sorry.sh    # asserts no sorry/axiom/admit/native_decide/#exit in Lean/
```

`MELLIN_AUDIT.md` records which Mathlib lemmas the Mellin layer stands on
(`mellin`, `MellinConvergent`, `mellin_comp_rpow`, …).

`Param.lean` and `Basis.lean` are **vendored copies**; their canonical source is the
author's `Ku_PATP` project, which holds the Lean layer of the PATP foundation paper
([arXiv:2605.14610](https://arxiv.org/abs/2605.14610)). They are duplicated here so
this repository closes its own import graph and builds standalone.

The **main theorem** (closed-form propagation) is proven in the paper, not in Lean;
the bilateral Mellin pair and the integrability conditions are likewise proved in the
text only. The Lean layer covers the four supporting lemmas — the paper says so
(Appendix C), and so does this README.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python run_all.py                  # revised manuscript, then the submitted-version scripts
python run_all.py --revision-only  # only the scripts behind the revised manuscript
```

`run_all.py` runs every step, prints which table/figure each produces, and ends with a
per-step summary and timing. On the reference machine the revision steps take about
4 minutes, most of it the real-data bootstrap.

### Real-data step

The CWRU bearing data is **not bundled** (size + source terms). To reproduce §8 and
Appendix E, download records `105.mat` and `106.mat` into `data/` — see
[`data/README.md`](data/README.md), which also lists their SHA-256 hashes — then:

```bash
cd experiments && python realdata_split.py        # about 3 minutes
# or keep the files elsewhere:
CWRU_DATA_DIR=/path/to/cwru python realdata_split.py
```

The script checks both files against `data/MANIFEST.sha256` and prints `MATCH` or
`MISMATCH`. If the files are absent it prints a pointer and exits with code 2; no other
step depends on them.

## No leakage / fair comparison

- Every surrogate is fitted by ordinary least squares on the response surface alone
  (for the manufactured benchmarks, a 401-point grid on `[1e-6, 1]`); shape parameters
  (`α`, the log-power centre `c`) are optimised on the fit residual only. No optimiser
  sees a reference moment.
- The baselines include capacity-matched ones: poly3 (4 coefficients), poly4 (5, the
  parameter count of PATP-opt), fixed-`α` PATP, the generic fractional basis
  `{1, x^½, x, x^{3/2}}` and the confluent log-power basis; in the engineering example,
  Gauss quadrature rules compared at equal per-climate and at equal total numbers of
  response evaluations. The number of distinct Mellin evaluations each route consumes is
  reported next to its error.
- In the real-data study the Mellin transform is estimated on a training segment and
  the moments of a held-out test segment are predicted; normalisation constants come
  from the training segment only. The decision rule, the metrics and the bootstrap were
  fixed in the script's docstring before the first run, and the SHA-256 of that block is
  printed in the output.

## What is *not* claimed here

These are the paper's own limits (§9), with the file that shows each one.

- **Measured input.** The pre-registered out-of-sample comparison is at **parity** for
  every response: the PATP surrogate is closer to the true response than the
  polynomials, but that advantage is far below the sampling error of predicting a
  held-out segment (`experiments/results/realdata_split.txt`). The responses there are
  illustrative transformations, not calibrated physical models.
- **Quadrature.** In the engineering example, where the response can be evaluated, the
  Gauss rules in `V` and in `√V` beat PATP-opt at every `n` in {8, 12, 16, 24}
  (`experiments/results/p2_engineering.txt`, qualifier Q-a).
- **Negative results stay in.** `exp(-x)` (M7) is a genuine negative result: polynomial
  MUET wins on every moment there (Table 3). On the analytic responses of the balanced
  benchmark (class A) poly4 beats PATP-opt on 4 of 5 (`experiments/results/p3_balanced.txt`).
- **The degenerate point `α = 1/2`.** The pre-registered convergence criterion Q4 of the
  confluent limit failed, so the printed decision is the fallback — an excluded
  neighbourhood and extended precision — and the log-power family is reported as a
  comparison, not adopted as the default (`experiments/results/p1_confluent.txt`).
- **Heavy tails.** The enlarged admissible region gives accurate means and standard
  deviations only with `α` restricted below `1/2`; higher moments near the admissibility
  edge are not accurate (`experiments/results/p4_heavy_tail.txt`).
- **Competitors not run.** Polynomial chaos, higher-order unscented transforms and
  fractional-moment maximum entropy are positioned in the paper but not run.
- **Determinism.** The manufactured, heavy-tail and engineering benchmarks are
  deterministic (the Monte-Carlo cross-check of the 2-D example uses a fixed seed) and
  reproduce exactly. Sampling variability enters only in the real-data study, through a
  moving-block bootstrap (`B = 1000`, fixed seeds printed by the script).

## Environment

See [`SESSION_INFO.md`](SESSION_INFO.md) for tested versions. `requirements.txt` pins the
versions that produced the revised manuscript's numbers.

## License & citation

MIT (see [`LICENSE`](LICENSE)). If you use this code, please cite the paper and the
software — see [`CITATION.cff`](CITATION.cff).
