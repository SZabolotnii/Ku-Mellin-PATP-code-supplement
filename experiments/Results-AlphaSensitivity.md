# Results — alpha-optimizer sensitivity (reviewer Q7, 2026-06-21)

Script: `alpha_sensitivity.py`. Two independent knobs over `X ~ U[0,1]`, `S=3`,
`i in {2,3,4}`:
- scipy `minimize_scalar` (bounded) `xatol ∈ {1e-4, 1e-6, 1e-8}`
- deterministic grid + local refine, density `∈ {51, 101, 201, 401}`

## Headline

`alpha*` and the propagated relative error are **insensitive** to optimizer tolerance
and grid design: across all 7 settings, `alpha*` spread `≤ 4e-4` and `rel_err(j=2)`
spread `< 3%` (relative).

| target | alpha* range | alpha* spread | rel_err(j=2) range |
|---|---|---|---|
| M1 0.6x^0.5+0.4x^1.5 | 0.45080–0.45120 | 4.0e-4 | [3.43e-6, 3.53e-6] |
| M4 √x(1+x²/3)        | 0.52000–0.52030 | 3.0e-4 | [1.150e-5, 1.166e-5] |
| M6 ln(1+x)           | 0.62000–0.62005 | 5.0e-5 | [1.547e-6, 1.553e-6] |

The two independent optimizers (scipy Brent-bounded vs deterministic grid) agree to
`~10^-4` in `alpha*`, so the result is not an artifact of a particular optimizer.
