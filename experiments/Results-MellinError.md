# Results — empirical Mellin estimator error (reviewer Q5, 2026-06-21)

Script: `mellin_error.py`. Real CWRU `105.mat`, `N=473` windows on `(0,1]`,
`B=3000` resamples.

## Analytical facts (demonstrated numerically)

`M̂_X(s) = (1/N) Σ_n X_n^{s-1}` is **unbiased** for `M_X(s)=E[X^{s-1}]`, with
`Var[M̂_X(s)] = Var(X^{s-1})/N`, **finite iff `E[X^{2(s-1)}] < ∞`**, i.e. iff `2s-1`
lies in the fundamental strip. For bounded support (strip `Re s > 0`) that means
`s > 1/2` — HALF the mean-existence range. PATP-MUET queries at `s = p_i(alpha)k+1 ≥ 1`
are comfortably inside the variance-finite zone.

## Bias & variance vs s

```
   s   2s-1   mean M̂     ref M̂     bias      Var       Var*N
0.45  -0.10  192.06    190.24   +1.8e0   3.5e4    1.7e7   <- 2s-1<=0 (var ill-defined)
0.50   0.00   69.75     68.57   +1.2e0   4.5e3    2.1e6   <- strip edge
0.55   0.10   25.61     25.34   +2.7e-1  5.7e2    2.7e5
0.60   0.20    9.75      9.95   -2.0e-1  6.8e1    3.2e4
0.80   0.60    1.36      1.36   -1.0e-3  1.8e-2   8.5
1.00   1.00    1.00      1.00   +0.0     0.0      0.0    (X^0=1 deterministic)
1.50   2.00    0.619     0.619  +4.5e-5  4.7e-5   2.2e-2
2.00   3.00    0.406     0.406  +1.1e-4  6.9e-5   3.3e-2
3.00   5.00    0.198     0.198  +1.8e-4  6.1e-5   2.9e-2
```

Variance is small and stable for `s ≳ 0.8` and **explodes as `s → 1/2+`** (the strip
edge for the second moment): `Var*N` grows from `~10^-2` (s=2) to `~10^7` (s=0.45).

## 1/N rate (s=2, safe)

```
   n      Var       Var*n
  59   5.39e-4   3.18e-2
 118   2.90e-4   3.42e-2
 236   1.39e-4   3.28e-2
 473   6.86e-5   3.24e-2
```

`Var*n ≈ const` ⇒ the `1/N` rate holds. Guidance (folded into §8): keep
`2(p_i(alpha)k+1) − 1` inside the strip; the off-lattice queries (`s ≥ 1`) satisfy this
automatically, so empirical PATP-MUET inherits the `1/N` accuracy of the input sample.
