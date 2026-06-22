"""
2D separable propagation (reviewer Q4): independent components, separable
fractional-power response.

For X = (X1, X2) independent and a separable response g(x1,x2) = g1(x1)*g2(x2),
the moments factor:
        E[(g1 g2)^j] = E[g1(X1)^j] * E[g2(X2)^j].
Each factor is propagated by the EXISTING univariate PATP-MUET (one alpha-fit per
axis) and multiplied. This is the regime where the joint Mellin transform
factorises; it is the additional validity constraint the multivariate case carries
(independence + separability).

  Case A: PATP-exact separable factors  -> 2D moment to machine precision.
  Case B: non-trivial separable factors -> PATP-MUET 2D beats polynomial-MUET 2D;
          a large-N Monte-Carlo estimate brackets the exact value.

Reuses the univariate fit + propagation from rq3_demo.

Run: .venv/bin/python twod_demo.py
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import quad

from rq3_demo import (
    fit_polynomial, polynomial_moment, fit_patp, patp_fit_loss, patp_moment,
)

INDICES = [2, 3, 4]
S_POLY = 3
NGRID = 401
XGRID = np.linspace(1e-6, 1.0, NGRID)


def fit_axis(f):
    poly = fit_polynomial(f, S_POLY, XGRID)
    res = minimize_scalar(lambda a: patp_fit_loss(a, f, XGRID, INDICES),
                          bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-8})
    a = float(res.x)
    par = fit_patp(f, a, XGRID, INDICES)
    return poly, a, par


def exact_axis_moment(f, j):  # E[f(X)^j], X ~ U[0,1]
    v, _ = quad(lambda x: f(x) ** j, 0.0, 1.0, limit=400)
    return v


CASES = [
    ("A  g1=x^0.5, g2=x^0.7            (PATP-exact factors)",
     lambda x: x ** 0.5, lambda x: x ** 0.7),
    ("B  g1=sqrt(x)(1+0.4x), g2=ln(1+x) (non-trivial)",
     lambda x: np.sqrt(x) * (1 + 0.4 * x), lambda x: np.log(1 + x)),
]


def main() -> int:
    rng = np.random.default_rng(2026)
    nmc = 4_000_000
    print(f"2D separable propagation, X1,X2 ~ U[0,1] independent, MC N={nmc:,}\n")
    for name, g1, g2 in CASES:
        p1, a1, par1 = fit_axis(g1)
        p2, a2, par2 = fit_axis(g2)
        x1 = rng.random(nmc)
        x2 = rng.random(nmc)
        gprod = g1(x1) * g2(x2)
        print(f"--- Case {name} ---")
        print(f"    alpha1* = {a1:.4f}   alpha2* = {a2:.4f}")
        print(f"  {'j':>2} {'exact(prod-quad)':>17} {'PATP-MUET 2D':>14} {'poly-MUET 2D':>14} "
              f"{'rel_patp':>10} {'rel_poly':>10} {'MC 95% CI':>27}")
        for j in (1, 2, 3, 4):
            ex = exact_axis_moment(g1, j) * exact_axis_moment(g2, j)
            kp = patp_moment(par1["k0"], par1["ks"], a1, j) * patp_moment(par2["k0"], par2["ks"], a2, j)
            pp = polynomial_moment(p1, j) * polynomial_moment(p2, j)
            rk = abs(kp - ex) / max(abs(ex), 1e-15)
            rp = abs(pp - ex) / max(abs(ex), 1e-15)
            gj = gprod ** j
            mc = gj.mean()
            se = gj.std(ddof=1) / np.sqrt(nmc)
            print(f"  {j:>2} {ex:>17.8f} {kp:>14.8f} {pp:>14.8f} {rk:>10.2e} {rp:>10.2e} "
                  f"[{mc - 1.96 * se:>11.7f},{mc + 1.96 * se:>11.7f}]")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
