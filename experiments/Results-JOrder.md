# Results — j-order stress + numerical conditioning (reviewer revision, 2026-06-21)

Script: `jorder_stress.py`. Regime: `X ~ U[0,1]`, `S=3`, `i in {2,3,4}`, `j = 1..8`.
Extended-precision column = the Theorem-3 multinomial sum evaluated in 50-digit
arithmetic (mpmath); double-precision diagnostics: `cond(A)` of the PATP fit design
matrix and `max|k_i|`.

## Headline

The closed form (Theorem 3) is **exact at all orders**: `rel_patp(mp50)` tracks the
true moment to `O(10^-3)` or better through `j=8` on every fractional-friendly target.
Double precision degrades only when the fit is **ill-conditioned** (clustered exponents
`p_i(alpha*)` => near-collinear basis => large alternating coefficients => catastrophic
cancellation in the multinomial sum). Extended precision removes the degradation.

| target | alpha* | cond(A) | max\|k_i\| | double accurate to | mp50 rel @ j=8 | poly rel @ j=8 |
|---|---|---|---|---|---|---|
| M1 0.6x^0.5+0.4x^1.5 | 0.4509 | 1.69e4 | 78  | j≈6 | 7.9e-4 | 5.8e-3 |
| M4 √x(1+x²/3)        | 0.5203 | 9.66e4 | 1220 | j≈4 | 2.9e-3 | 1.0e-2 |
| M6 ln(1+x)           | 0.6200 | 2.79e3 | 5.2  | j=8  | 9.6e-5 | 2.0e-4 |
| M7 exp(-x) (neg.)    | 0.6216 | 2.72e3 | 4.3  | j=8  | 1.67e-4 | 1.68e-4 |

In extended precision PATP-MUET **retains its advantage over polynomial MUET through j=8
on all three fractional-friendly targets** (M1, M4, M6: `rel_patp(mp50) < rel_poly` at
every order); M7 stays the honest negative (polynomial-friendly target), unchanged.

## Diagnostic (double vs mp50), M4 (worst conditioning)

```
 j  rel_poly  rel_patp(dbl)  rel_patp(mp50)
 4  5.17e-04   5.97e-04        1.52e-04
 5  1.37e-03   2.24e-01        4.67e-04
 6  3.12e-03   7.17e+02        1.02e-03
 7  5.98e-03   1.84e+06        1.81e-03
 8  9.98e-03   3.33e+07        2.85e-03
```

Cross-check: direct quadrature of the SAME surrogate, ∫₀¹ ĝ(x)^j dx, agrees with the
mp50 closed form to full precision (so the blow-up is purely finite-precision summation,
not a property of Theorem 3).

## Practitioner guidance (folded into §5 + §9)

- Operating range in **double precision**: `j ≤ 6` for well-conditioned fits;
  monitor `cond(A)` / `max|k_i|` (degradation tracks `cond(A) ≳ 10^4`).
- For higher orders or ill-conditioned (clustered-exponent) fits, evaluate the closed
  form in **extended / compensated precision** — exactness is then recovered (demonstrated
  to `j=8`). Regularising the fractional-LS fit (penalising `||k||`) is an alternative.
