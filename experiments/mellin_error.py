"""
Empirical Mellin estimator error analysis (reviewer Q5).

The empirical Mellin transform  M̂_X(s) = (1/N) sum_n X_n^{s-1}  is UNBIASED for
M_X(s) = E[X^{s-1}], with  Var[M̂_X(s)] = Var(X^{s-1}) / N,  finite iff
E[X^{2(s-1)}] < inf, i.e. iff 2s-1 lies in the fundamental strip. For bounded-
support inputs (strip Re s > 0) that means s > 1/2 -- HALF the mean-existence
range. The PATP-MUET off-lattice queries are at s = p_i(alpha) k + 1 >= 1,
comfortably inside this variance-finite zone.

Demonstrated on the real CWRU sample:
  (a) ~zero bias of M̂_X vs the full-sample reference,
  (b) the 1/N variance rate (subsampling at a safe s),
  (c) variance blow-up as s -> 1/2+ (the strip edge for the SECOND moment).

The CWRU bearing data is NOT bundled (size + source licence). Place 105.mat in
data/ (see data/README.md) or pass --mat <path>.

Run: python mellin_error.py [--mat <path>]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from realdata_case import load_real_input, emp_mx_factory


def main() -> int:
    ap = argparse.ArgumentParser()
    default_mat = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               os.pardir, "data", "105.mat")
    ap.add_argument("--mat", default=default_mat)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    if not os.path.exists(args.mat):
        print(f"Input file not found: {args.mat}\n"
              f"The CWRU bearing data is NOT bundled (size + source licence). "
              f"See data/README.md to download 105.mat into the data/ folder, "
              f"or pass --mat <path>.", file=sys.stderr)
        return 2

    x, key, nsig = load_real_input(args.mat, args.window)
    N = len(x)
    rng = np.random.default_rng(args.seed)
    full = emp_mx_factory(x)  # full-sample reference

    print(f"CWRU {os.path.basename(args.mat)}::{key}  N={N} windows on (0,1]")
    print(f"Empirical M̂_X(s): unbiasedness + variance vs s "
          f"(B={args.reps} resamples of size N)")
    print("  variance finite iff 2s-1 > 0  <=>  s > 0.5 (strip edge for 2nd moment)\n")
    print(f"  {'s':>5} {'2s-1':>6} {'mean M̂':>12} {'ref M̂':>12} "
          f"{'bias':>11} {'Var':>11} {'Var*N':>11}")
    for s in [0.45, 0.50, 0.55, 0.60, 0.80, 1.00, 1.50, 2.00, 3.00]:
        ests = np.array([np.mean(x[rng.integers(0, N, N)] ** (s - 1.0))
                         for _ in range(args.reps)])
        ref = full(s)
        flag = "" if (2 * s - 1) > 0 else "  <- 2s-1<=0 (var ill-defined)"
        print(f"  {s:>5.2f} {2 * s - 1:>6.2f} {ests.mean():>12.4f} {ref:>12.4f} "
              f"{ests.mean() - ref:>+11.2e} {ests.var():>11.3e} {ests.var() * N:>11.3e}{flag}")

    s = 2.0
    print(f"\n1/N variance rate at s={s} (Var should ~halve as N doubles; Var*n ~ const):")
    print(f"  {'n':>6} {'Var':>12} {'Var*n':>12}")
    for frac in [0.125, 0.25, 0.5, 1.0]:
        n = max(8, int(N * frac))
        ests = np.array([np.mean(rng.choice(x, n, replace=True) ** (s - 1.0))
                         for _ in range(args.reps)])
        print(f"  {n:>6} {ests.var():>12.3e} {ests.var() * n:>12.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
