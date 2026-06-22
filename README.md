# Ku-Mellin-PATP — Reproducibility Repository

Verification and reproducibility code for the paper:

> **A Stochastic-Polynomial Bridge: PATP Extension of Moment-Based Uncertainty
> Evaluation via the Mellin Transform**
> Serhii V. Zabolotnii, 2026.

**Repository:** <https://github.com/SZabolotnii/Ku-Mellin-PATP-code-supplement>
(`git clone https://github.com/SZabolotnii/Ku-Mellin-PATP-code-supplement.git`)

**One-line summary.** Classical moment-based uncertainty evaluation (MUET) queries
the Mellin transform `M_X(s) = E[X^{s-1}]` of the input density on the *integer
lattice* `s = k+1`. The signed-parity fractional-power **PATP** basis queries the
*same* `M_X` at *real* arguments `s = p_i(α)·k + 1` — an off-lattice continuation —
yielding a closed-form moment-propagation formula for fractional-power response
surfaces that polynomial MUET cannot reach. This repository reproduces every
numerical claim in the paper.

---

## What this repository reproduces

| Script | Paper artifact |
|---|---|
| `verification/cas/check_patp.py` | CAS cross-check of T1 (corner values), T2 (positivity), T3 (Mellin power rule), and the closed-form §6 multinomial formula vs Monte-Carlo |
| `verification/cas/region_of_validity.py` | **Table 1** — quadratic structure of `p_i(α)` (discriminant, vertex, minimum) and per-`k` admissibility |
| `verification/cas/mellin_table.py` | **Table 2** — closed-form Mellin transforms of the metrology distributions (uniform, beta, triangular, truncated normal), validated numerically |
| `verification/cas/multinomial_validity.py` | **§5** — 765-pair scan establishing that the multinomial validity minimum reduces to the single-index minimum |
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

## Formal verification (Lean 4)

The four supporting lemmas (corner values, positivity on `[0,1]`, the Mellin
power-substitution rule, and the `α=1` reduction) are formalized in **Lean 4 over
Mathlib v4.26.0** with no `sorry`/`axiom`. Those proofs live in the companion
`Ku_PATP` project (`PATP/Lean/`: `Param.lean`, `Positivity.lean`, `MellinPower.lean`,
`AlphaOneReduction.lean`) and are not duplicated here — this repository covers the
numerical (Python) layer.

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
Table 4 / §8, download `105.mat` (and `100.mat`) into `data/` — see
[`data/README.md`](data/README.md) — then:

```bash
python experiments/realdata_case.py                    # faulty bearing (headline)
python experiments/realdata_case.py --mat data/100.mat # healthy baseline (robustness)
```

The script reports a clear pointer and exits cleanly if the data is absent; the
rest of the pipeline does not depend on it.

## No leakage / fair comparison

Both methods receive the same moment budget of `X` (`S=3`); PATP additionally uses
the off-lattice empirical moments `M_X(p_i(α)k+1)` — the structural advantage of the
fractional basis, costing one scalar `α`, not extra data. The `α`-optimiser sees
only the surrogate's fit residual to the response surface, never the ground-truth
output moments. See the paper §7 (and §8 for the real-data calibration choice) for
the full protocol.

## Environment

See [`SESSION_INFO.md`](SESSION_INFO.md) for tested versions (Python 3.13;
numpy / scipy / matplotlib / sympy).

## License & citation

MIT (see [`LICENSE`](LICENSE)). If you use this code, please cite the paper and the
software — see [`CITATION.cff`](CITATION.cff).
