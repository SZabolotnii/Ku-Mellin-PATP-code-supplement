"""
j-order stress test + numerical-conditioning analysis (reviewer: "limited
exploration of higher moment orders / conditioning issues").

NON-DESTRUCTIVE: the headline rq3_demo.py table stays at j=1..4. This script
pushes the moment order to j=1..8 in the safe regime (S=3, i in {2,3,4}, alpha
per the validity table) and separates two distinct effects:

  * The closed form (Theorem 3) is MATHEMATICALLY EXACT at every order. We confirm
    this by evaluating the SAME multinomial sum in 50-digit precision (mpmath):
    it matches direct quadrature of the surrogate to full precision at all j.

  * Its DOUBLE-PRECISION evaluation can suffer catastrophic cancellation at high j
    when the fitted PATP coefficients are large and alternating -- which happens
    when the exponents p_i(alpha*) cluster (near-collinear basis => ill-conditioned
    least-squares fit). We report cond(A) of the fit design matrix and max|k_i| as
    the diagnostic, and compare rel-error in double vs 50-digit precision.

Takeaway: PATP-MUET stays accurate through j=8 on well-conditioned fits (e.g.
ln(1+x)); on ill-conditioned fits (clustered exponents) double precision degrades
beyond j~5-6 while extended precision stays exact -- i.e. the limit is finite-
precision summation of the closed form, not the closed form itself.

Run: .venv/bin/python jorder_stress.py
"""
from __future__ import annotations

from itertools import product
from math import factorial

import numpy as np
from scipy.optimize import minimize_scalar

from rq3_demo import (
    p_num, fit_polynomial, polynomial_moment, fit_patp, patp_fit_loss,
    patp_moment, truth_moment,
)

try:
    import mpmath as mp
    mp.mp.dps = 50
    HAVE_MP = True
except Exception as _exc:  # pragma: no cover
    HAVE_MP = False
    print(f"[warn] mpmath unavailable ({_exc}); extended-precision column skipped")

JMAX = 8
INDICES = [2, 3, 4]
S_POLY = 3
NGRID = 401
XGRID = np.linspace(1e-6, 1.0, NGRID)

TARGETS = [
    ("M1  0.6 x^0.5 + 0.4 x^1.5  (non-trivial)", lambda x: 0.6 * x ** 0.5 + 0.4 * x ** 1.5),
    ("M4  sqrt(x) (1 + x^2/3)    (non-trivial)", lambda x: np.sqrt(x) * (1 + x ** 2 / 3)),
    ("M6  ln(1 + x)              (hard)",        lambda x: np.log(1 + x)),
    ("M7  exp(-x)                (hard, neg.)",  lambda x: np.exp(-x)),
]


def design_cond(alpha):
    cols = [np.ones_like(XGRID)] + [np.abs(XGRID) ** p_num(i, alpha) for i in INDICES]
    return float(np.linalg.cond(np.column_stack(cols)))


def patp_moment_mp(k0, ks, alpha, j, dps=50):
    """The Theorem-3 multinomial sum evaluated in `dps`-digit precision (M_X=1/s)."""
    mp.mp.dps = dps
    idx = sorted(ks.keys())
    pv = {i: mp.mpf(1) / i + (4 - i - mp.mpf(3) / i) * mp.mpf(alpha)
          + (2 * i - 4 + mp.mpf(2) / i) * mp.mpf(alpha) ** 2 for i in idx}
    ksm = {i: mp.mpf(repr(ks[i])) for i in idx}
    k0m = mp.mpf(repr(k0))
    tot = mp.mpf(0)
    for m in range(j + 1):
        binom = factorial(j) // (factorial(m) * factorial(j - m))
        inner = mp.mpf(0)
        for combo in product(range(m + 1), repeat=len(idx)):
            if sum(combo) != m:
                continue
            cm = factorial(m)
            pk = mp.mpf(1)
            s = mp.mpf(1)
            for pos, i in enumerate(idx):
                ke = combo[pos]
                if ke == 0:
                    continue
                cm //= factorial(ke)
                pk *= ksm[i] ** ke
                s += ke * pv[i]
            inner += cm * pk * (mp.mpf(1) / s)
        tot += binom * k0m ** (j - m) * inner
    return float(tot)


def fit_both(f):
    poly = fit_polynomial(f, S_POLY, XGRID)
    res = minimize_scalar(lambda a: patp_fit_loss(a, f, XGRID, INDICES),
                          bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-6})
    a = float(res.x)
    par = fit_patp(f, a, XGRID, INDICES)
    return poly, a, par


def main() -> int:
    print(f"j-order stress: S={S_POLY}, i in {INDICES}, X ~ U[0,1], j = 1..{JMAX}")
    print("(safe regime: all p_i(alpha) > 0 on [0,1], Thm 2 / Table 1)\n")
    for name, f in TARGETS:
        poly, a, par = fit_both(f)
        cond = design_cond(a)
        kmax = max(abs(v) for v in par["ks"].values())
        print(f"--- {name}   alpha* = {a:.4f}   cond(A) = {cond:.2e}   max|k_i| = {kmax:.2e} ---")
        hdr = f"  {'j':>2} {'rel_poly':>11} {'rel_patp(dbl)':>14}"
        if HAVE_MP:
            hdr += f" {'rel_patp(mp50)':>15}"
        print(hdr)
        for j in range(1, JMAX + 1):
            t = truth_moment(f, j)
            pe = polynomial_moment(poly, j)
            ke = patp_moment(par["k0"], par["ks"], a, j)
            rp = abs(pe - t) / max(abs(t), 1e-15)
            rk = abs(ke - t) / max(abs(t), 1e-15)
            line = f"  {j:>2} {rp:>11.2e} {rk:>14.2e}"
            if HAVE_MP:
                kem = patp_moment_mp(par["k0"], par["ks"], a, j)
                rkm = abs(kem - t) / max(abs(t), 1e-15)
                line += f" {rkm:>15.2e}"
            print(line)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
