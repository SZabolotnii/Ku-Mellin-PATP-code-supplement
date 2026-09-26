"""
Step 0 of the revised analysis: does the PATP advantage survive a capacity-matched
comparison?

The question is whether the tab:rq3 gains come from surrogate fit rather than from the
propagation formula, given that the original comparison was not capacity-matched
(PATP: 4 linear coefficients + alpha; cubic polynomial: 4 coefficients).
This script re-runs the manufactured benchmark of rq3_demo.py (same targets,
same 401-point grid on [1e-6, 1], same OLS fit, same alpha optimiser, same
X ~ U[0,1] and quadrature truth) against capacity-matched baselines.

Competitors (every one propagates exactly on its own surrogate):
  poly3      degree-3 polynomial, 4 coefficients           (paper's baseline)
  PATP-opt   PATP S=3, alpha optimised on the fit residual (paper's method)
  poly4      degree-4 polynomial, 5 coefficients           (parameter-count match)
  PATP-a0    PATP S=3, alpha = 0 fixed                     (no shape optimisation)
  PATP-a25   PATP S=3, alpha = 0.25 fixed                  (no shape optimisation)
  sqrt3      {1, x^1/2, x, x^3/2}, 4 coefficients, fixed   (generic fractional basis)
  poly-dm    smallest-degree polynomial whose grid L2 <= PATP-opt's L2 (accuracy match)

The two fixed alphas and the sqrt3 basis were chosen before running and are
not tuned per target. sqrt3 contains M1 exactly; that row is flagged, not used.

Metric: per-target geometric mean over j = 1..4 of the relative moment error
(robust to one j dominating), plus the paper's arithmetic avg improvement.
Cost: number of distinct Mellin arguments needed for j <= 4.
Propagation check: |closed form - quadrature of the same surrogate|, which must
be at quadrature level if the closed form is exact.

PRE-REGISTERED DECISION (written before the first run, targets M1/M4/M6):
  K1 capacity   PATP-opt beats poly4 (lower geo-mean error) on >= 2 of 3 targets.
                If K1 FAILS, the empirical headline collapses to the sanity rows
                and the paper must be reframed as a propagation calculus with
                validity theory, without a numerical-advantage claim.
  K2 alpha      PATP-opt beats both fixed-alpha PATP variants on >= 2 of 3.
                If K2 fails, alpha optimisation adds nothing and is dropped.
  K3 basis      PATP-opt beats sqrt3 on M4 and M6 (M1 excluded: sqrt3 exact).
                If K3 fails, a generic fractional basis does as well as PATP.
  K4 accuracy   report the degree poly-dm needs; d <= 4 means polynomial MUET
                reaches PATP accuracy with at most one extra coefficient.
M5 (x^3) and M7 (exp(-x)) are run as controls and do not enter K1-K4.

Run:  ../verification/cas/.venv/bin/python step0_capacity_matched.py
"""

from __future__ import annotations

import sys
from itertools import combinations_with_replacement
from math import factorial

import mpmath as mp
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

mp.mp.dps = 50
J = (1, 2, 3, 4)
X_GRID = np.linspace(1e-6, 1.0, 401)


def p_num(i: int, alpha: float) -> float:
    return 1.0 / i + (4 - i - 3.0 / i) * alpha + (2 * i - 4 + 2.0 / i) * alpha ** 2


# ---------------------------------------------------------------------------
# One surrogate type: g(x) = sum_k c_k x^{e_k}, e_0 = 0. On X ~ U[0,1],
# E[g^j] = sum_{|kappa|=j} multinom * prod c^kappa / (sum kappa e + 1).
# Evaluated in 50-digit arithmetic so that the reported error is the
# surrogate's, not float cancellation's.
# ---------------------------------------------------------------------------

class PowerSurrogate:
    def __init__(self, name: str, exps: list[float], coefs: np.ndarray, n_params: int):
        self.name, self.exps, self.coefs, self.n_params = name, list(exps), np.asarray(coefs), n_params

    def __call__(self, x):
        return sum(c * x ** e for c, e in zip(self.coefs, self.exps))

    def moment(self, j: int) -> float:
        n = len(self.exps)
        total = mp.mpf(0)
        for combo in combinations_with_replacement(range(n), j):
            kappa = [combo.count(k) for k in range(n)]
            multi = mp.mpf(factorial(j))
            prod, s = mp.mpf(1), mp.mpf(1)
            for k, ke in enumerate(kappa):
                if ke:
                    multi /= factorial(ke)
                    prod *= mp.mpf(float(self.coefs[k])) ** ke
                    s += ke * mp.mpf(float(self.exps[k]))
            total += multi * prod / s
        return float(total)

    def n_mellin_args(self, jmax: int = 4) -> int:
        nonconst = [e for e in self.exps if e != 0]
        args = {0.0}
        for j in range(1, jmax + 1):
            for combo in combinations_with_replacement(nonconst, j):
                args.add(round(sum(combo), 12))
        return len(args)

    def l2(self, f) -> float:
        return float(np.sqrt(np.mean((self(X_GRID) - f(X_GRID)) ** 2)))


def ols(f, exps) -> np.ndarray:
    A = np.column_stack([X_GRID ** e for e in exps])
    coefs, *_ = np.linalg.lstsq(A, f(X_GRID), rcond=None)
    return coefs


def poly(f, d: int, name: str | None = None) -> PowerSurrogate:
    exps = list(range(d + 1))
    return PowerSurrogate(name or f"poly{d}", exps, ols(f, exps), d + 1)


def patp(f, alpha: float, name: str, n_params: int) -> PowerSurrogate:
    exps = [0.0] + [p_num(i, alpha) for i in (2, 3, 4)]
    s = PowerSurrogate(name, exps, ols(f, exps), n_params)
    s.alpha = alpha
    return s


def patp_opt(f) -> PowerSurrogate:
    def loss(a):
        exps = [0.0] + [p_num(i, a) for i in (2, 3, 4)]
        A = np.column_stack([X_GRID ** e for e in exps])
        c, *_ = np.linalg.lstsq(A, f(X_GRID), rcond=None)
        return float(np.mean((A @ c - f(X_GRID)) ** 2))
    res = minimize_scalar(loss, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-6})
    return patp(f, res.x, "PATP-opt", 5)


def truth(f, j: int) -> float:
    return quad(lambda x: f(x) ** j, 0.0, 1.0, limit=400)[0]


def rel_errors(s: PowerSurrogate, f) -> list[float]:
    return [abs(s.moment(j) - truth(f, j)) / abs(truth(f, j)) for j in J]


def geo(v) -> float:
    return float(np.exp(np.mean(np.log(np.maximum(v, 1e-300)))))


TARGETS = [
    ("M1", "0.6 x^0.5 + 0.4 x^1.5", lambda x: 0.6 * x ** 0.5 + 0.4 * x ** 1.5),
    ("M4", "sqrt(x)(1 + x^2/3)", lambda x: np.sqrt(x) * (1 + x ** 2 / 3)),
    ("M6", "ln(1 + x)", lambda x: np.log(1 + x)),
    ("M5", "x^3 (control)", lambda x: x ** 3),
    ("M7", "exp(-x) (control)", lambda x: np.exp(-x)),
]


def run_target(tag, label, f):
    popt = patp_opt(f)
    target_l2 = popt.l2(f)
    dm = None
    for d in range(1, 13):
        cand = poly(f, d)
        if cand.l2(f) <= target_l2:
            dm = poly(f, d, name=f"poly-dm(d={d})")
            break
    methods = [poly(f, 3), popt, poly(f, 4),
               patp(f, 0.0, "PATP-a0", 4), patp(f, 0.25, "PATP-a25", 4),
               PowerSurrogate("sqrt3", [0, 0.5, 1.0, 1.5], ols(f, [0, 0.5, 1.0, 1.5]), 4)]
    if dm is not None:
        methods.append(dm)

    print(f"\n=== {tag}: f(x) = {label} ===")
    print(f"PATP-opt alpha* = {popt.alpha:.4f}, exponents = "
          f"{[round(e, 4) for e in popt.exps[1:]]}, coefs = {[round(float(c), 3) for c in popt.coefs]}")
    A = np.column_stack([X_GRID ** e for e in popt.exps])
    print(f"PATP-opt cond(A) = {np.linalg.cond(A):.2e}")
    if dm is None:
        print("poly-dm: no degree <= 12 reaches PATP-opt's L2")
    hdr = f"{'method':<14}{'params':>7}{'mellin':>7}{'L2':>11}" + "".join(f"{'rel j=' + str(j):>11}" for j in J) \
        + f"{'geo':>11}{'prop.err':>11}"
    print(hdr)
    rows = {}
    for s in methods:
        r = rel_errors(s, f)
        prop = max(abs(s.moment(j) - quad(lambda x: s(x) ** j, 1e-12, 1.0, limit=400, epsabs=1e-15)[0]) for j in J)
        rows[s.name.split("(")[0]] = {"geo": geo(r), "rel": r, "l2": s.l2(f)}
        print(f"{s.name:<14}{s.n_params:>7}{s.n_mellin_args():>7}{s.l2(f):>11.2e}"
              + "".join(f"{e:>11.2e}" for e in r) + f"{geo(r):>11.2e}{prop:>11.1e}")
    base = rows["poly3"]["rel"]
    for name, row in rows.items():
        if name == "poly3":
            continue
        avg = np.mean([(b - e) / b * 100 for b, e in zip(base, row["rel"])])
        print(f"  avg improvement vs poly3 ({name}): {avg:+.1f}%")
    return rows, dm


def main() -> int:
    results = {}
    for tag, label, f in TARGETS:
        results[tag] = run_target(tag, label, f)

    core = ["M1", "M4", "M6"]
    print("\n" + "=" * 78)
    print("PRE-REGISTERED VERDICT (targets M1, M4, M6)")
    print("=" * 78)

    def beats(tag, a, b):
        return results[tag][0][a]["geo"] < results[tag][0][b]["geo"]

    k1 = [t for t in core if beats(t, "PATP-opt", "poly4")]
    print(f"K1 PATP-opt < poly4 on {k1}  -> {'PASS' if len(k1) >= 2 else 'FAIL'} ({len(k1)}/3)")
    k2 = [t for t in core if beats(t, "PATP-opt", "PATP-a0") and beats(t, "PATP-opt", "PATP-a25")]
    print(f"K2 PATP-opt < both fixed-alpha on {k2}  -> {'PASS' if len(k2) >= 2 else 'FAIL'} ({len(k2)}/3)")
    k3 = [t for t in ["M4", "M6"] if beats(t, "PATP-opt", "sqrt3")]
    print(f"K3 PATP-opt < sqrt3 on {k3} of [M4, M6]  -> {'PASS' if len(k3) == 2 else 'FAIL'}")
    for t in core:
        dm = results[t][1]
        print(f"K4 {t}: accuracy-matched polynomial degree = {dm.n_params - 1 if dm else '>12'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
