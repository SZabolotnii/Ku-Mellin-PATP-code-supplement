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

---

## What this repository reproduces

| Script | Paper artifact |
|---|---|
| `verification/cas/check_patp.py` | CAS cross-check of T1 (corner values), T2 (positivity), T3 (Mellin power rule), and the closed-form multinomial formula vs Monte-Carlo |
| `verification/cas/region_of_validity.py` | **Table 1** — quadratic structure of `p_i(α)` (discriminant, vertex, minimum) and per-`k` admissibility |
| `verification/cas/mellin_table.py` | **Table 2** — closed-form Mellin transforms of the metrology distributions (uniform, beta, triangular, truncated normal), validated numerically |
| `verification/cas/multinomial_validity.py` | **§5** — 765-pair scan establishing that the multinomial validity minimum reduces to the single-index minimum |
| `verification/cas/heavy_tail_admissibility.py` | **§5.1, Proposition 2** — admissibility on *unbounded* support: bounds on `P_S(α) = max_i p_i(α)`, plus the Pareto check where polynomial MUET needs `E[X^6] = ∞` while the closed form returns the moment |
| `experiments/rq3_demo.py` | **Table 3** — manufactured-solution benchmark (PATP-MUET vs polynomial MUET on 7 response surfaces, `X ~ U[0,1]`) |
| `experiments/make_figure.py` | **Figure 1** — relative error vs moment order (`outputs/fig_rq3.pdf`) |
| `experiments/twod_demo.py` | **Table 4** — separable 2-D propagation (independent inputs, per-axis factorization `E[(g1 g2)^j] = E[g1^j] E[g2^j]`) |
| `experiments/jorder_stress.py` | **§7** — higher moment orders (`j<=8`): exact closed form vs double-precision conditioning (`cond(A)`, 50-digit cross-check) |
| `experiments/alpha_sensitivity.py` | **§7** — stability of `α*` across optimiser tolerance, grid density, and two independent optimisers |
| `experiments/realdata_case.py` | **Table 5 / §8** — real-data case study: propagation through a measured, non-Gaussian vibration-power input (CWRU), with bootstrap CIs |
| `experiments/mellin_error.py` | **§8** — empirical-Mellin estimator error: unbiasedness, `1/N` variance rate, and strip-edge (`s -> 1/2+`) blow-up on the CWRU sample |

Companion `Results-*.md` files in `experiments/` carry the full per-moment tables
and honest-limits discussion (`Results-RQ3`, `Results-2D`, `Results-JOrder`,
`Results-AlphaSensitivity`, `Results-RealData`, `Results-MellinError`).

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

The **main theorem** (closed-form propagation, Theorem 2 in the paper) is proven
inline in the paper, not in Lean; its formalization is listed as future work. The
Lean layer covers the supporting lemmas only — the paper says so, and so does this
README.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

`run_all.py` runs every step and prints which table/figure each produces. Outputs
land in `outputs/` (git-ignored).

### Real-data step

The CWRU bearing data is **not bundled** (size + source terms). To reproduce
Table 5 / §8, download `105.mat` (and `100.mat`) into `data/` — see
[`data/README.md`](data/README.md) — then:

```bash
python experiments/realdata_case.py                    # faulty bearing (headline)
python experiments/realdata_case.py --mat data/100.mat # healthy baseline (robustness)
```

The script prints a clear pointer and exits cleanly if the data is absent; the rest
of the pipeline does not depend on it.

## No leakage / fair comparison

Both methods receive the same moment budget of `X` (`S=3`); PATP additionally uses
the off-lattice empirical moments `M_X(p_i(α)k+1)` — the structural advantage of the
fractional basis, costing one scalar `α`, not extra data. The `α`-optimiser sees
only the surrogate's fit residual to the response surface, never the ground-truth
output moments. See the paper §7 (and §8 for the real-data calibration choice) for
the full protocol.

## What is *not* claimed here

- The manufactured benchmark is **deterministic** — manufactured targets, quadrature
  ground truth, fixed fitting grid, deterministic optimiser. It reproduces exactly
  rather than in distribution, and there are no seeds to average over. Sampling
  variability enters only in the real-data study, which is bootstrapped (`B=300`).
- The only baseline actually implemented is **polynomial MUET**, at an identical
  coefficient budget. The fractional-moment MaxEnt line, the complex-fractional-moment
  reconstruction line, and polynomial-chaos surrogates are positioned in the paper
  but are not run as numerical competitors here.
- `heavy_tail_admissibility.py` verifies an **admissibility condition and one Pareto
  instance**. It is not a comparative heavy-tail benchmark, and the paper does not
  report one.
- `exp(-x)` is a genuine **negative result**: polynomial MUET wins on every moment
  there. It is in Table 3 and in `Results-RQ3.md`, not hidden.

## Environment

See [`SESSION_INFO.md`](SESSION_INFO.md) for tested versions.

## License & citation

MIT (see [`LICENSE`](LICENSE)). If you use this code, please cite the paper and the
software — see [`CITATION.cff`](CITATION.cff).
