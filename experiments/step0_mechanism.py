"""
Step 0, post-hoc diagnostics (NOT pre-registered; run after step0_capacity_matched.py).

1. Reproducibility of tab:rq3 M4, j = 4. The PATP-opt fit on M4 sits at
   alpha* = 0.520, next to the alpha = 1/2 degeneracy, with coefficients of
   order 1e3 and alternating signs (cond(A) ~ 1e5). The float64 multinomial sum
   of eq:patp-muet then cancels, and the j = 4 moment depends on the 7th digit
   of the coefficients. This prints the float64 value (rq3_demo.patp_moment),
   the 50-digit value of the same surrogate, and adaptive quadrature of it.

2. Mechanism. Near alpha = 1/2 the three exponents cluster around a centre c,
   and a large alternating combination of x^{p_i} is a finite difference in
   the exponent, i.e. it spans approximately {x^c, x^c ln x, x^c ln^2 x}.
   The script fits that basis {1, x^c, x^c ln x, x^c ln^2 x} with c optimised
   (also five parameters) and compares its grid L2 with PATP-opt.

Run:  ../verification/cas/.venv/bin/python step0_mechanism.py
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

import rq3_demo
import step0_capacity_matched as s

X, L = s.X_GRID, np.log(s.X_GRID)


def reproducibility_m4() -> None:
    f = s.TARGETS[1][2]
    p = s.patp_opt(f)
    k0, ks = float(p.coefs[0]), {i: float(c) for i, c in zip((2, 3, 4), p.coefs[1:])}
    poly3 = s.poly(f, 3)
    print("1. M4 moment errors: float64 closed form vs 50-digit closed form vs quadrature of the surrogate")
    print(f"   alpha* = {p.alpha:.10f}, coefs = {[round(float(c), 6) for c in p.coefs]}")
    print(f"   {'j':>2}{'float64 rel':>14}{'50-digit rel':>14}{'quad rel':>12}{'poly3 rel':>12}{'impr (50-digit)':>18}")
    for j in s.J:
        t = s.truth(f, j)
        fl = rq3_demo.patp_moment(k0, ks, p.alpha, j)
        hp = p.moment(j)
        qd = quad(lambda x: p(x) ** j, 1e-12, 1.0, limit=400, epsabs=1e-15)[0]
        rp = abs(poly3.moment(j) - t) / t
        print(f"   {j:>2}{abs(fl - t) / t:>14.2e}{abs(hp - t) / t:>14.2e}{abs(qd - t) / t:>12.2e}{rp:>12.2e}"
              f"{(rp - abs(hp - t) / t) / rp * 100:>+17.1f}%")
    print("   published tab:rq3 (results/rq3.txt): j=4 PATP rel 5.97e-04, improvement -15.6%, row avg +61.3%")


def xc_log_l2(c: float, f) -> float:
    B = np.column_stack([np.ones_like(X), X ** c, X ** c * L, X ** c * L ** 2])
    k, *_ = np.linalg.lstsq(B, f(X), rcond=None)
    return float(np.sqrt(np.mean((B @ k - f(X)) ** 2)))


def mechanism() -> None:
    print("\n2. PATP-opt vs the basis {1, x^c, x^c ln x, x^c ln^2 x}, c optimised (5 parameters each)")
    print(f"   {'target':<8}{'alpha*':>8}{'mean p':>9}{'spread':>9}{'L2 PATP':>11}{'c*':>8}{'L2 xc-log':>11}")
    for tag, _, f in s.TARGETS[:3] + s.TARGETS[4:]:
        p = s.patp_opt(f)
        r = minimize_scalar(lambda c: xc_log_l2(c, f), bounds=(0.05, 3.0), method="bounded",
                            options={"xatol": 1e-6})
        ex = p.exps[1:]
        print(f"   {tag:<8}{p.alpha:>8.3f}{np.mean(ex):>9.3f}{max(ex) - min(ex):>9.3f}{p.l2(f):>11.2e}"
              f"{r.x:>8.3f}{r.fun:>11.2e}")


def main() -> int:
    reproducibility_m4()
    mechanism()
    return 0


if __name__ == "__main__":
    sys.exit(main())
