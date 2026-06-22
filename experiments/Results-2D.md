# Results — 2D separable propagation (reviewer Q4, 2026-06-21)

Script: `twod_demo.py`. `X1, X2 ~ U[0,1]` independent; separable response
`g(x1,x2)=g1(x1)·g2(x2)`. Moments factor: `E[(g1 g2)^j] = E[g1^j]·E[g2^j]`, each factor
propagated by the EXISTING univariate PATP-MUET (one `alpha`-fit per axis) and multiplied.
Ground truth = product of 1-D adaptive quadratures; cross-checked by `N=4e6` Monte Carlo.

## Headline

The closed form **factorises exactly** under independence + separability:

- **Case A** (PATP-exact factors `g1=x^0.5`, `g2=x^0.7`): PATP-MUET 2D reaches
  `rel ≈ 10^{-11}..10^{-14}` (optimizer floor) vs polynomial-MUET 2D `~3–5×10^{-4}`.
- **Case B** (non-trivial `g1=√x(1+0.4x)`, `g2=ln(1+3x→1+x)`): PATP-MUET 2D
  `rel ≈ 2×10^{-6}..2×10^{-5}` vs polynomial 2D `~2×10^{-5}..3×10^{-4}` — PATP wins
  (or ties at machine-noise level on j=3). MC 95% CI brackets the exact value at all j.

```
Case A  alpha1*=0.250  alpha2*=0.371
 j   exact          PATP-MUET 2D    poly 2D       rel_patp   rel_poly
 1   0.39215686     0.39215686      0.39199396    2.4e-14    4.2e-4
 2   0.20833333     0.20833333      0.20821942    1.8e-11    5.5e-4
 4   0.08771930     0.08771930      0.08774713    9.4e-12    3.2e-4

Case B  alpha1*=0.433  alpha2*=0.620
 j   exact          PATP-MUET 2D    poly 2D       rel_patp   rel_poly
 1   0.31933667     0.31933119      0.31926032    1.7e-5     2.4e-4
 2   0.15190929     0.15190999      0.15186835    4.6e-6     2.7e-4
 4   0.05565715     0.05565602      0.05565964    2.0e-5     4.5e-5
```

## Validity constraint (folded into §5 + §9)

Exact factorisation requires **independent components and a separable response**; the
univariate machinery and validity table apply per axis. General (non-separable or
dependent) multivariate propagation needs the joint Mellin transform and is future work.
