"""
alpha-optimizer sensitivity (reviewer Q7): how stable is alpha* (and the
resulting propagated-moment error) under changes to the optimizer tolerance and
the grid design?

Two independent knobs, reusing both alpha-optimizers in the codebase:
  * scipy minimize_scalar (bounded) xatol in {1e-4, 1e-6, 1e-8}   (rq3_demo style)
  * deterministic grid + local refine, density in {51,101,201,401} (realdata style)

For each target we report alpha* under every setting, the alpha* spread, and the
spread of the j=2 propagated relative error.

Run: .venv/bin/python alpha_sensitivity.py
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from rq3_demo import fit_patp, patp_fit_loss, patp_moment, truth_moment

INDICES = [2, 3, 4]
NGRID = 401
XGRID = np.linspace(1e-6, 1.0, NGRID)

TARGETS = [
    ("M1  0.6 x^0.5 + 0.4 x^1.5", lambda x: 0.6 * x ** 0.5 + 0.4 * x ** 1.5),
    ("M4  sqrt(x) (1 + x^2/3)",   lambda x: np.sqrt(x) * (1 + x ** 2 / 3)),
    ("M6  ln(1 + x)",             lambda x: np.log(1 + x)),
]


def alpha_scipy(f, xatol):
    res = minimize_scalar(lambda a: patp_fit_loss(a, f, XGRID, INDICES),
                          bounds=(0.0, 1.0), method="bounded", options={"xatol": xatol})
    return float(res.x)


def alpha_grid(f, npts):
    g = np.linspace(0.0, 1.0, npts)
    a0 = g[int(np.argmin([patp_fit_loss(a, f, XGRID, INDICES) for a in g]))]
    step = 1.0 / (npts - 1)
    lo, hi = max(0.0, a0 - step), min(1.0, a0 + step)
    gf = np.linspace(lo, hi, npts)
    return float(gf[int(np.argmin([patp_fit_loss(a, f, XGRID, INDICES) for a in gf]))])


def rel_j2(f, a):
    par = fit_patp(f, a, XGRID, INDICES)
    t = truth_moment(f, 2)
    ke = patp_moment(par["k0"], par["ks"], a, 2)
    return abs(ke - t) / max(abs(t), 1e-15)


def main() -> int:
    xatols = [1e-4, 1e-6, 1e-8]
    npts_list = [51, 101, 201, 401]
    print("alpha* sensitivity to optimizer tolerance and grid design (S=3, i in {2,3,4})\n")
    for name, f in TARGETS:
        print(f"--- {name} ---")
        avals = []
        print("  scipy minimize_scalar (bounded):")
        for xa in xatols:
            a = alpha_scipy(f, xa)
            avals.append(a)
            print(f"    xatol={xa:.0e}: alpha*={a:.6f}  rel_err(j=2)={rel_j2(f, a):.3e}")
        print("  deterministic grid + local refine:")
        for n in npts_list:
            a = alpha_grid(f, n)
            avals.append(a)
            print(f"    grid={n:>3}: alpha*={a:.6f}  rel_err(j=2)={rel_j2(f, a):.3e}")
        re = [rel_j2(f, a) for a in avals]
        print(f"  ==> alpha* spread (all 7 settings): {max(avals) - min(avals):.2e}  "
              f"(min={min(avals):.5f}, max={max(avals):.5f})")
        print(f"  ==> rel_err(j=2) spread: [{min(re):.3e}, {max(re):.3e}]\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
