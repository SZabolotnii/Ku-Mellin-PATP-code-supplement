"""
P1 of the revised analysis: the
alpha = 1/2 degeneracy of the PATP basis, treated through its confluent
(log-power) limit.

Background (step 0: experiments/results/step0.txt, step0_mechanism.txt).
  p_i(alpha) = 1/i + (4 - i - 3/i) alpha + (2i - 4 + 2/i) alpha^2; for S = 3 the
  basis is {1, x^{p_2}, x^{p_3}, x^{p_4}}. At alpha = 1/2 every p_i = 1 and the
  design matrix loses rank. With eps = alpha - 1/2,
      p_i = 1 + (i - 1/i) eps + (2i - 4 + 2/i) eps^2 ,
  so near 1/2 the exponents cluster and the least-squares fit uses large
  alternating coefficients. Divided differences of q -> x^q converge to x^c,
  x^c ln x, x^c ln^2 x / 2, so the PATP span should have a removable
  singularity at 1/2 with limit span{1, x, x ln x, x ln^2 x}; and the basis
  {1, x^c, x^c ln x, x^c ln^2 x} propagates moments in closed form through
  derivatives of the Mellin transform, E[X^a (ln X)^b] = (d/ds)^b M_X (a + 1).
  For X ~ U[0,1]: M_X(s) = 1/s and (d/ds)^b M_X(s) = (-1)^b b! / s^(b+1).

Common setup (identical to step0_capacity_matched.py and imported from it):
  401-point grid X_GRID on [1e-6, 1]; OLS fits on that grid; targets M1, M4, M6
  of step0.TARGETS; PATP-opt = step0.patp_opt (bounded Brent on [0, 1], xatol
  1e-6); truth = step0.truth (adaptive quadrature of f^j); geo = geometric mean
  over j = 1..4 of the relative moment error; X ~ U[0,1].

Bases.
  RAW  (a) {1, x^{q1}, x^{q2}, x^{q3}}, q_k = p_{k+1}(alpha)  (PATP as published).
  DD   (b) {1, D0, D1, D2}: Newton divided differences in the exponent over the
           same nodes, D0 = x^{q1}, D1 = x^[q1,q2], D2 = x^[q1,q2,q3]. Same span as
           RAW for alpha != 1/2; at alpha = 1/2 it is {1, x, x ln x, x ln^2 x / 2}
           (Hermite-Genocchi limit).
  CONF (c) {1, x^c, x^c ln x, x^c ln^2 x}, c optimised on the grid residual only:
           scan c in [0.05, 3] with step 0.005, then bounded Brent within +-0.005
           of the best scan point, xatol 1e-9.

cond(A) := sigma_max / sigma_min of the unscaled 401 x 4 design matrix. The
  reported value comes from the 4 x 4 Gram matrix A^T A, formed and diagonalised
  in 80-digit arithmetic from the float64 grid points and float64 exponents (as
  the fitting code produces them) taken as exact: cond = sqrt(l_max / l_min).
  This stays exact where a float64 SVD saturates near alpha = 1/2; the float64
  np.linalg.cond value is printed beside it. DD columns are formed in 80 digits.

PRE-REGISTERED QUESTIONS (fixed before the first run; not edited afterwards)

Q1 conditioning. cond(A) of RAW as a function of alpha on [0, 1] (uniform step
   0.005, plus 1/2 +- 10^k for k in [-8, -1], 8 points per decade), of DD on the
   same grid and at alpha = 1/2 itself, and of CONF at c*; at the optimum of
   each target M1, M4, M6 (RAW and DD at alpha*, CONF at c*).
Q2 float64 stability. Closed-form moments j = 1..8 of the fitted surrogate
   (coefficients as stored in float64), evaluated (i) in float64 with the
   structure of eq:patp-muet (binomial in k0, multinomial in the remaining
   coefficients), (ii) in 50-digit arithmetic (the exact moment of that
   surrogate), (iii) by scipy adaptive quadrature of the float64 surrogate on
   [0, 1] (epsabs 1e-15, epsrel 1e-13, limit 500); for RAW PATP-opt and CONF, on
   M1, M4, M6. Reported: relative deviation of (i) and of (iii) from (ii).
   Implementation check: (iii) must agree with (ii) to 1e-9 relative in every
   cell, otherwise Q2 is reported INVALID rather than PASS/FAIL.
Q3 accuracy parity. Grid L2 (RMS residual) and geo (j = 1..4; surrogate moments
   in 50 digits against step0.truth) of CONF versus PATP-opt on M1 and M4. M6 is
   printed for information only.
Q4 continuity. For alpha = 1/2 +- eps, eps = 1e-1, 1e-2, ..., 1e-6, the RAW
   least-squares fit is computed in extended precision (normal equations at 120
   digits, repeated at 160 digits; the difference is the noise estimate) on the
   same grid data (float64 target values taken as exact), and compared with the
   least-squares fit on {1, x, x ln x, x ln^2 x} at the same precision.
   Distances: |L2(eps) - L2(0)|; RMS grid distance between the two fitted
   functions; relative moment differences for j = 1..4 (closed form in the
   working precision) -- six distances per target and sign. A float64 lstsq fit
   is printed alongside to show where double precision breaks down.

DECISION RULE (pre-registered)
  ADOPT the confluent treatment in the paper if all four hold:
   (Q1) CONF cond(A) at c* <= 1/10 of RAW PATP-opt cond(A) at alpha*, on both M1
        and M4;
   (Q2) CONF float64 relative deviation from the exact moment <= 1e-10 for all
        j <= 8 on M1, M4 and M6;
   (Q3) CONF geo <= 1.5 x PATP-opt geo on M1 and on M4;
   (Q4) for each of M1, M4, M6, each sign of eps and each of the six distances,
        the sequence over eps = 1e-1 ... 1e-6 is non-increasing, a step being
        accepted as noise when the later value is <= 10 x its noise estimate;
        and the distance at eps = 1e-6 is <= 1e-3 x the distance at eps = 1e-1
        (or at noise level).
  Otherwise FALLBACK: excluded neighbourhood |alpha - 1/2| < delta plus
  extended-precision evaluation; the failing criteria are reported.

Descriptive items (pre-registered, no role in the decision):
  D1 number of Mellin evaluations for j <= 4: distinct arguments for RAW,
     distinct (argument, derivative order) pairs for CONF.
  D2 float64 accuracy of the DD columns over the alpha grid: max normwise
     relative column error against the 80-digit columns (same float64 nodes),
     for naive divided differences and for a stable evaluation (D1 through
     expm1; D2 through Hermite-Genocchi Gauss quadrature on the simplex where
     |ln x| * spread <= 2, through expm1-based differences elsewhere).
  D3 empirical convergence order in Q4: log10 of consecutive-decade distance
     ratios.
  D4 least-squares slope of log10 cond(RAW) against log10 |alpha - 1/2| over
     0 < |alpha - 1/2| <= 1e-3 (both sides); the lemma predicts -2.

Anything else is POST-HOC and is printed under a separate POST-HOC header.

Run from experiments/:
  PYTHONDONTWRITEBYTECODE=1 ../verification/cas/.venv/bin/python \
      p1_confluent_limit.py | tee results/p1_confluent.txt
Writes results/p1_cond_vs_alpha.csv and ../paper/fig_cond_alpha.pdf.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from itertools import combinations_with_replacement
from math import comb, factorial

import matplotlib
import mpmath as mp
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

import step0_capacity_matched as s0
from step0_capacity_matched import TARGETS, X_GRID, geo, p_num, patp_opt, truth

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logging.getLogger("fontTools").setLevel(logging.ERROR)   # font 'head' timestamp chatter on PDF save

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "results", "p1_cond_vs_alpha.csv")
FIG_PATH = os.path.normpath(os.path.join(HERE, "..", "outputs", "fig_cond_alpha.pdf"))  # supplement: outputs/, not ../paper/

CORE = ("M1", "M4", "M6")
TGT = {tag: (label, f) for tag, label, f in TARGETS}
IDX = (2, 3, 4)
J4 = (1, 2, 3, 4)
J8 = tuple(range(1, 9))
EPS_Q4 = [1, 2, 3, 4, 5, 6]          # eps = 10^-k
DPS_COND, DPS_Q4, DPS_Q4_CHECK = 80, 120, 160
LOG_X = np.log(X_GRID)


# ---------------------------------------------------------------------------
# Closed-form moments of g(x) = k0 + sum_k c_k x^{a_k} (ln x)^{b_k}, X ~ U[0,1],
# with the structure of eq:patp-muet: binomial in k0, multinomial in the rest.
# E[X^A (ln X)^W] = (d/ds)^W (1/s) at s = A + 1 = (-1)^W W! / (A + 1)^(W + 1).
# ---------------------------------------------------------------------------

def _mp(v):
    return v if isinstance(v, mp.mpf) else mp.mpf(float(v))


def cf_moment(k0, coefs, terms, j: int, arith: str):
    if arith == "mp":
        k0, coefs = _mp(k0), [_mp(c) for c in coefs]
        terms = [(_mp(a), b) for a, b in terms]
        zero, one, fact = mp.mpf(0), mp.mpf(1), mp.factorial
    else:
        k0, coefs = float(k0), [float(c) for c in coefs]
        terms = [(float(a), b) for a, b in terms]
        zero, one, fact = 0.0, 1.0, (lambda n: float(factorial(n)))
    n = len(terms)
    total = zero
    for m in range(j + 1):
        inner = zero
        for combo in combinations_with_replacement(range(n), m):
            kappa = [combo.count(k) for k in range(n)]
            multi = factorial(m)
            prod, A, W = one, zero, 0
            for k, ke in enumerate(kappa):
                if ke:
                    multi //= factorial(ke)
                    prod *= coefs[k] ** ke
                    A += ke * terms[k][0]
                    W += ke * terms[k][1]
            inner += multi * prod * ((-1) ** W * fact(W) / (A + 1) ** (W + 1))
        total += comb(j, m) * k0 ** (j - m) * inner
    return total


class Surr:
    """g(x) = k0 + sum_k coefs[k] * x^{a_k} (ln x)^{b_k}."""

    def __init__(self, name, k0, coefs, terms):
        self.name, self.k0 = name, float(k0)
        self.coefs = [float(c) for c in coefs]
        self.terms = [(float(a), int(b)) for a, b in terms]

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        pos = x > 0
        xs = np.where(pos, x, 1.0)
        L = np.log(xs)
        out = np.full_like(xs, self.k0)
        for c, (a, b) in zip(self.coefs, self.terms):
            out = out + np.where(pos, c * xs ** a * L ** b, 0.0)
        return out if out.ndim else float(out)

    def moment(self, j, arith="mp"):
        return cf_moment(self.k0, self.coefs, self.terms, j, arith)

    def l2(self, f):
        return float(np.sqrt(np.mean((self(X_GRID) - f(X_GRID)) ** 2)))

    def quad_moment(self, j):
        return quad(lambda x: self(x) ** j, 0.0, 1.0, epsabs=1e-15, epsrel=1e-13, limit=500)[0]

    def n_mellin(self, jmax=4):
        pairs = {(0.0, 0)}
        for m in range(1, jmax + 1):
            for combo in combinations_with_replacement(range(len(self.terms)), m):
                A = sum(self.terms[k][0] for k in combo)
                W = sum(self.terms[k][1] for k in combo)
                pairs.add((round(A, 12), W))
        return len(pairs)


def raw_surr(popt) -> Surr:
    return Surr("PATP-opt", popt.coefs[0], popt.coefs[1:], [(e, 0) for e in popt.exps[1:]])


def conf_design(c):
    xc = X_GRID ** c
    return np.column_stack([np.ones_like(X_GRID), xc, xc * LOG_X, xc * LOG_X ** 2])


def conf_l2(f, c):
    B = conf_design(c)
    k, *_ = np.linalg.lstsq(B, f(X_GRID), rcond=None)
    return float(np.sqrt(np.mean((B @ k - f(X_GRID)) ** 2)))


def conf_opt(f):
    scan = np.arange(0.05, 3.0 + 1e-12, 0.005)
    vals = np.array([conf_l2(f, c) for c in scan])
    cb = float(scan[int(np.argmin(vals))])
    r = minimize_scalar(lambda c: conf_l2(f, c), bounds=(max(0.05, cb - 0.005), min(3.0, cb + 0.005)),
                        method="bounded", options={"xatol": 1e-9})
    c = float(r.x) if r.fun <= vals.min() else cb
    k, *_ = np.linalg.lstsq(conf_design(c), f(X_GRID), rcond=None)
    return c, Surr("CONF", k[0], k[1:], [(c, 0), (c, 1), (c, 2)])


# ---------------------------------------------------------------------------
# Extended-precision design matrices and condition numbers
# ---------------------------------------------------------------------------

_LOGS: dict[int, list] = {}


def grid_logs():
    d = mp.mp.dps
    if d not in _LOGS:
        _LOGS[d] = [mp.log(mp.mpf(float(x))) for x in X_GRID]
    return _LOGS[d]


def ones_col():
    return [mp.mpf(1)] * len(X_GRID)


def pow_col(q):
    q = _mp(q)
    return [mp.exp(q * L) for L in grid_logs()]


def dd_from_raw(q, cols):
    q1, q2, q3 = (_mp(v) for v in q)
    c1, c2, c3 = cols
    d1 = [(b - a) / (q2 - q1) for a, b in zip(c1, c2)]
    d23 = [(b - a) / (q3 - q2) for a, b in zip(c2, c3)]
    d2 = [(e - d) / (q3 - q1) for d, e in zip(d1, d23)]
    return [c1, d1, d2]


def dd_limit(c=1):
    """Confluent DD columns at coincident nodes c: x^c, x^c ln x, x^c ln^2 x / 2."""
    L = grid_logs()
    xc = [mp.exp(_mp(c) * l) for l in L]
    return [xc, [a * l for a, l in zip(xc, L)], [a * l * l / 2 for a, l in zip(xc, L)]]


def gram(cols):
    n = len(cols)
    G = mp.matrix(n, n)
    for a in range(n):
        for b in range(a, n):
            G[a, b] = G[b, a] = mp.fsum(u * v for u, v in zip(cols[a], cols[b]))
    return G


def cond_mp(cols_nonconst):
    E = mp.eigsy(gram([ones_col()] + list(cols_nonconst)), eigvals_only=True)
    lo, hi = min(E), max(E)
    return mp.inf if lo <= 0 else mp.sqrt(hi / lo)


def raw_f64(q):
    return np.column_stack([np.ones_like(X_GRID)] + [X_GRID ** v for v in q])


# D2: float64 DD columns, naive and stable.
_GL_T, _GL_W = np.polynomial.legendre.leggauss(10)
_GL_T, _GL_W = (_GL_T + 1) / 2, _GL_W / 2


def dd_naive_f64(q):
    q1, q2, q3 = q
    c1, c2, c3 = X_GRID ** q1, X_GRID ** q2, X_GRID ** q3
    d1 = (c2 - c1) / (q2 - q1)
    d2 = ((c3 - c2) / (q3 - q2) - d1) / (q3 - q1)
    return [c1, d1, d2]


def _dd1_stable(qa, qb):
    h = qb - qa
    if h == 0:
        return X_GRID ** qa * LOG_X
    return X_GRID ** qa * np.expm1(h * LOG_X) / h


def dd_stable_f64(q):
    q1, q2, q3 = q
    d0, d1 = X_GRID ** q1, _dd1_stable(q1, q2)
    spread = max(q) - min(q)
    small = np.abs(LOG_X) * spread <= 2.0
    d2 = np.empty_like(X_GRID)
    if (~small).any():
        d2[~small] = ((_dd1_stable(q2, q3) - _dd1_stable(q1, q2)) / (q3 - q1))[~small]
    if small.any():
        U, V = np.meshgrid(_GL_T, _GL_T, indexing="ij")
        WU, WV = np.meshgrid(_GL_W, _GL_W, indexing="ij")
        t1, t2 = U, (1 - U) * V                      # Duffy map of the unit simplex
        tau = ((1 - t1 - t2) * q1 + t1 * q2 + t2 * q3).ravel()
        w = (WU * WV * (1 - U)).ravel()
        Ls = LOG_X[small]
        d2[small] = Ls ** 2 * (np.exp(np.outer(Ls, tau)) @ w)
    return [d0, d1, d2]


def col_err(cols_f64, cols_mp):
    worst = 0.0
    for cf, cm in zip(cols_f64, cols_mp):
        ref = np.array([float(v) for v in cm])
        scale = np.max(np.abs(ref))
        worst = max(worst, float(np.max(np.abs(cf - ref)) / scale))
    return worst


# ---------------------------------------------------------------------------
# Q1
# ---------------------------------------------------------------------------

def alpha_grid():
    uni = [round(0.005 * k, 12) for k in range(201)]
    near = []
    for e in np.logspace(-8, -1, 57):
        near += [0.5 - float(e), 0.5 + float(e)]
    return sorted(set(uni) | set(near))


def q1_curve():
    rows = []
    with mp.workdps(DPS_COND):
        for a in alpha_grid():
            if a == 0.5:
                dd = dd_limit(1)
                rows.append(dict(alpha=a, amh=0.0, cond_raw=mp.inf, cond_raw_f64=float("inf"),
                                 cond_dd=cond_mp(dd),
                                 err_naive=float("nan"),
                                 err_stable=col_err(dd_stable_f64([1.0, 1.0, 1.0]), dd)))
                continue
            q = [p_num(i, a) for i in IDX]
            raw = [pow_col(v) for v in q]
            dd = dd_from_raw(q, raw)
            rows.append(dict(alpha=a, amh=a - 0.5, cond_raw=cond_mp(raw),
                             cond_raw_f64=float(np.linalg.cond(raw_f64(q))),
                             cond_dd=cond_mp(dd),
                             err_naive=col_err(dd_naive_f64(q), dd),
                             err_stable=col_err(dd_stable_f64(q), dd)))
    return rows


def cond_at(alpha=None, c=None):
    with mp.workdps(DPS_COND):
        if c is not None:
            cols = [pow_col(c)]
            L = grid_logs()
            cols += [[a * l for a, l in zip(cols[0], L)], [a * l * l for a, l in zip(cols[0], L)]]
            return cond_mp(cols), float(np.linalg.cond(conf_design(c)))
        q = [p_num(i, alpha) for i in IDX]
        raw = [pow_col(v) for v in q]
        return cond_mp(raw), float(np.linalg.cond(raw_f64(q))), cond_mp(dd_from_raw(q, raw))


# ---------------------------------------------------------------------------
# Q4
# ---------------------------------------------------------------------------

def p_mp(i, alpha):
    i = mp.mpf(i)
    return 1 / i + (4 - i - 3 / i) * alpha + (2 * i - 4 + 2 / i) * alpha ** 2


def ls_mp(cols, y):
    n = len(cols)
    G = gram(cols)
    r = mp.matrix(n, 1)
    for a in range(n):
        r[a] = mp.fsum(u * v for u, v in zip(cols[a], y))
    k = mp.lu_solve(G, r)
    k = [k[i] for i in range(n)]
    fitted = [mp.fsum(k[a] * cols[a][t] for a in range(n)) for t in range(len(y))]
    return k, fitted


def rms(u, v):
    return mp.sqrt(mp.fsum((a - b) ** 2 for a, b in zip(u, v)) / len(u))


def q4_target(f, dps):
    out = {}
    with mp.workdps(dps):
        L = grid_logs()
        xs = [mp.mpf(float(x)) for x in X_GRID]
        y = [mp.mpf(float(v)) for v in f(X_GRID)]
        one = ones_col()
        lim_cols = [one, xs, [x * l for x, l in zip(xs, L)], [x * l * l for x, l in zip(xs, L)]]
        k, g0 = ls_mp(lim_cols, y)
        l2_0 = rms(g0, y)
        m0 = [cf_moment(k[0], k[1:], [(1, 0), (1, 1), (1, 2)], j, "mp") for j in J4]
        out["limit"] = dict(l2=l2_0, m=m0)
        for sign in (-1, 1):
            for e in EPS_Q4:
                alpha = mp.mpf(1) / 2 + sign * mp.mpf(10) ** (-e)
                q = [p_mp(i, alpha) for i in IDX]
                kk, g = ls_mp([one] + [pow_col(v) for v in q], y)
                l2 = rms(g, y)
                m = [cf_moment(kk[0], kk[1:], [(v, 0) for v in q], j, "mp") for j in J4]
                d = [abs(l2 - l2_0), rms(g, g0)] + [abs(mj - m0j) / abs(m0j) for mj, m0j in zip(m, m0)]
                out[(sign, e)] = dict(l2=l2, m=m, d=d, kmax=max(abs(v) for v in kk))
    return out


def q4_float64(f, sign, e):
    a = 0.5 + sign * 10.0 ** (-e)
    q = [p_num(i, a) for i in IDX]
    A = raw_f64(q)
    k, *_ = np.linalg.lstsq(A, f(X_GRID), rcond=None)
    l2 = float(np.sqrt(np.mean((A @ k - f(X_GRID)) ** 2)))
    m = [cf_moment(k[0], k[1:], [(v, 0) for v in q], j, "f64") for j in J4]
    return l2, m, float(np.max(np.abs(k)))


DNAMES = ["|dL2|", "fit-dist", "dm1", "dm2", "dm3", "dm4"]


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(rows, opt):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "axes.unicode_minus": False,
        "mathtext.fontset": "cm", "font.size": 8.5,
        "axes.labelsize": 9.5, "legend.fontsize": 7, "axes.linewidth": 0.6,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "lines.linewidth": 1.1,
        "pdf.fonttype": 42, "axes.formatter.use_mathtext": True,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.75), constrained_layout=True)
    a = np.array([r["alpha"] for r in rows])
    craw = np.array([float(r["cond_raw"]) for r in rows])
    cdd = np.array([float(r["cond_dd"]) for r in rows])
    fin = np.isfinite(craw)
    ax1.semilogy(a[fin], craw[fin], color="black", ls="-", label="raw PATP")
    ax1.semilogy(a, cdd, color="0.45", ls="--", label="divided differences")
    for tag in CORE:
        o = opt[tag]
        ax1.plot(o["alpha"], o["cond_raw"], "o", mfc="white", mec="black", ms=4.5, mew=0.9)
        ax1.plot(o["alpha"], o["cond_conf"], "s", mfc="0.55", mec="black", ms=4.5, mew=0.6)
        off = (-15, 2) if o["alpha"] < 0.5 else (5, 2)
        ax1.annotate(tag, (o["alpha"], o["cond_raw"]), xytext=off, textcoords="offset points", fontsize=7)
    ax1.plot([], [], "o", mfc="white", mec="black", ms=4.5, mew=0.9, label=r"raw PATP at $\alpha^\star$")
    ax1.plot([], [], "s", mfc="0.55", mec="black", ms=4.5, mew=0.6, label=r"confluent at $c^\star$")
    ax1.axvline(0.5, color="0.7", lw=0.5, ls=":")
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_ylabel(r"$\mathrm{cond}_2(A)$")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(10, 1e8)
    ax1.legend(loc="upper left", ncol=1, frameon=False, handlelength=2.2)

    amh = np.array([r["amh"] for r in rows])
    left, right = (amh < 0) & fin, (amh > 0) & fin
    ax2.loglog(-amh[left], craw[left], color="black", ls="-", label=r"raw, $\alpha<1/2$")
    ax2.loglog(amh[right], craw[right], color="black", ls="-.", label=r"raw, $\alpha>1/2$")
    ax2.loglog(-amh[amh < 0], cdd[amh < 0], color="0.45", ls="--", label="divided differences")
    ax2.loglog(amh[amh > 0], cdd[amh > 0], color="0.45", ls="--")
    dref = np.logspace(-8, -2, 20)
    i0 = int(np.argmin(np.abs(amh[right] - 1e-4)))
    cref = 30 * craw[right][i0] * (dref / amh[right][i0]) ** -2
    ax2.loglog(dref, cref, color="0.55", ls=":", lw=1.0, label=r"$\propto|\alpha-1/2|^{-2}$ (offset)")
    ax2.set_xlabel(r"$|\alpha-1/2|$")
    ax2.set_ylabel(r"$\mathrm{cond}_2(A)$")
    ax2.legend(loc="upper right", frameon=False, handlelength=2.2)
    fig.savefig(FIG_PATH)
    plt.close(fig)


# ---------------------------------------------------------------------------

def fmt(v, p=2):
    return mp.nstr(v, p + 1, min_fixed=1, max_fixed=0) if isinstance(v, mp.mpf) else f"{v:.{p}e}"


def main() -> int:
    t0 = time.time()
    print("P1 -- alpha = 1/2 degeneracy of PATP via the confluent (log-power) limit")
    print(f"numpy {np.__version__}, scipy {scipy.__version__}, mpmath {mp.__version__}, "
          f"matplotlib {matplotlib.__version__}, python {sys.version.split()[0]}")
    print(f"grid: {len(X_GRID)} points on [{X_GRID[0]:g}, {X_GRID[-1]:g}]; X ~ U[0,1]")

    fits = {}
    for tag in CORE:
        f = TGT[tag][1]
        popt = patp_opt(f)
        c, conf = conf_opt(f)
        fits[tag] = dict(popt=popt, raw=raw_surr(popt), c=c, conf=conf)

    # ------------------------------------------------------------------ Q1
    print("\n" + "=" * 78 + "\nQ1 conditioning (cond_2 of the 401 x 4 design matrix; 80-digit Gram)\n" + "=" * 78)
    rows = q1_curve()
    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["alpha", "alpha_minus_half", "cond_raw_mp80", "cond_raw_float64", "cond_dd_mp80",
                    "dd_naive_float64_col_err", "dd_stable_float64_col_err"])
        for r in rows:
            w.writerow([f"{r['alpha']:.17g}", f"{r['amh']:.17g}",
                        "inf" if r["cond_raw"] == mp.inf else mp.nstr(r["cond_raw"], 12),
                        f"{r['cond_raw_f64']:.6e}", mp.nstr(r["cond_dd"], 12),
                        f"{r['err_naive']:.3e}", f"{r['err_stable']:.3e}"])
    print(f"curve: {len(rows)} alpha values -> {os.path.relpath(CSV_PATH, HERE)}")
    show = [0.0, 0.25, 0.4, 0.45, 0.49, 0.5 - 1e-3, 0.5 - 1e-5, 0.5 - 1e-8, 0.5, 0.5 + 1e-8, 0.5 + 1e-5,
            0.5 + 1e-3, 0.51, 0.55, 0.6, 0.75, 1.0]
    print(f"{'alpha':>22}{'cond RAW':>12}{'RAW f64':>11}{'cond DD':>10}{'DD naive err':>14}{'DD stable err':>15}")
    for tgt in show:
        r = min(rows, key=lambda r: abs(r["alpha"] - tgt))
        print(f"{r['alpha']:>22.12f}{fmt(r['cond_raw']):>12}{r['cond_raw_f64']:>11.2e}{fmt(r['cond_dd']):>10}"
              f"{r['err_naive']:>14.2e}{r['err_stable']:>15.2e}")
    dd_all = [float(r["cond_dd"]) for r in rows]
    print(f"DD over the whole grid: min cond {min(dd_all):.3e}, max cond {max(dd_all):.3e}")
    print(f"D2 max DD column error over the grid: naive {np.nanmax([r['err_naive'] for r in rows]):.2e}, "
          f"stable {max(r['err_stable'] for r in rows):.2e}")
    near = [r for r in rows if 0 < abs(r["amh"]) <= 1e-3]
    sl = np.polyfit(np.log10([abs(r["amh"]) for r in near]), np.log10([float(r["cond_raw"]) for r in near]), 1)[0]
    print(f"D4 slope of log10 cond(RAW) vs log10|alpha-1/2| on (0, 1e-3]: {sl:.4f} (lemma: -2)")

    opt = {}
    print(f"\n{'target':<7}{'alpha*':>9}{'cond RAW':>11}{'RAW f64':>10}{'cond DD':>10}{'c*':>9}"
          f"{'cond CONF':>11}{'CONF f64':>10}{'CONF/RAW':>10}")
    for tag in CORE:
        fr = fits[tag]
        cr, crf, cd = cond_at(alpha=fr["popt"].alpha)
        cc, ccf = cond_at(c=fr["c"])
        opt[tag] = dict(alpha=fr["popt"].alpha, cond_raw=float(cr), cond_dd=float(cd), cond_conf=float(cc))
        print(f"{tag:<7}{fr['popt'].alpha:>9.4f}{fmt(cr):>11}{crf:>10.2e}{fmt(cd):>10}{fr['c']:>9.5f}"
              f"{fmt(cc):>11}{ccf:>10.2e}{float(cc / cr):>10.3f}")
    q1_pass = all(opt[t]["cond_conf"] <= opt[t]["cond_raw"] / 10 for t in ("M1", "M4"))
    print(f"Q1 criterion: CONF cond <= RAW cond / 10 on M1 and M4 -> {'PASS' if q1_pass else 'FAIL'}")

    # ------------------------------------------------------------------ Q2
    print("\n" + "=" * 78 + "\nQ2 float64 stability of the closed form, j = 1..8 "
          "(relative deviation from the 50-digit value)\n" + "=" * 78)
    q2_pass, q2_valid = True, True
    q2data = {}
    for tag in CORE:
        fr = fits[tag]
        print(f"\n{tag}: RAW alpha* = {fr['popt'].alpha:.6f}, coefs = {[float(f'{c:.6g}') for c in fr['popt'].coefs]}")
        print(f"{tag}: CONF c* = {fr['c']:.6f}, coefs = {[float(f'{c:.6g}') for c in [fr['conf'].k0] + fr['conf'].coefs]}")
        print(f"{'j':>3}{'exact RAW':>14}{'RAW f64':>11}{'RAW quad':>11}{'exact CONF':>14}{'CONF f64':>11}{'CONF quad':>11}")
        for j in J8:
            line = f"{j:>3}"
            for key in ("raw", "conf"):
                sgt = fr[key]
                ex = sgt.moment(j, "mp")
                d64 = float(abs(mp.mpf(sgt.moment(j, "f64")) - ex) / abs(ex))
                dq = float(abs(mp.mpf(sgt.quad_moment(j)) - ex) / abs(ex))
                line += f"{float(ex):>14.6e}{d64:>11.2e}{dq:>11.2e}"
                q2data[(tag, key, j)] = (d64, dq)
                if dq > 1e-9:
                    q2_valid = False
                if key == "conf" and d64 > 1e-10:
                    q2_pass = False
            print(line)
    q2 = "INVALID" if not q2_valid else ("PASS" if q2_pass else "FAIL")
    print(f"Q2 criterion: CONF float64 deviation <= 1e-10 for all j <= 8 on M1, M4, M6 "
          f"(quadrature check {'ok' if q2_valid else 'FAILED'}) -> {q2}")

    # ------------------------------------------------------------------ Q3
    print("\n" + "=" * 78 + "\nQ3 accuracy parity (moments against quadrature truth of f^j)\n" + "=" * 78)
    print(f"{'target':<7}{'method':<10}{'params':>7}{'L2':>11}" + "".join(f"{'rel j=' + str(j):>11}" for j in J4)
          + f"{'geo':>11}")
    q3 = {}
    for tag in CORE:
        f, fr = TGT[tag][1], fits[tag]
        tr = [truth(f, j) for j in J4]
        for key, name in (("raw", "PATP-opt"), ("conf", "CONF")):
            sgt = fr[key]
            rel = [float(abs(sgt.moment(j, "mp") - t) / abs(t)) for j, t in zip(J4, tr)]
            q3[(tag, key)] = dict(l2=sgt.l2(f), geo=geo(rel))
            print(f"{tag:<7}{name:<10}{5:>7}{sgt.l2(f):>11.3e}" + "".join(f"{e:>11.2e}" for e in rel)
                  + f"{geo(rel):>11.3e}")
        print(f"{'':<7}ratio geo CONF / PATP-opt = {q3[(tag, 'conf')]['geo'] / q3[(tag, 'raw')]['geo']:.3f}"
              f"   (step0 PowerSurrogate check of PATP-opt geo: "
              f"{geo([abs(fr['popt'].moment(j) - t) / abs(t) for j, t in zip(J4, tr)]):.3e})")
    q3_pass = all(q3[(t, "conf")]["geo"] <= 1.5 * q3[(t, "raw")]["geo"] for t in ("M1", "M4"))
    print(f"Q3 criterion: CONF geo <= 1.5 x PATP-opt geo on M1 and M4 -> {'PASS' if q3_pass else 'FAIL'}")

    print("\nD1 Mellin evaluations for j <= 4 (distinct arguments / (argument, derivative order) pairs):")
    for tag in CORE:
        fr = fits[tag]
        print(f"  {tag}: PATP-opt {fr['raw'].n_mellin()}, CONF {fr['conf'].n_mellin()} "
              f"(at {len({round(fr['c'] * m, 12) for m in range(5)})} distinct arguments), "
              f"poly3 {s0.poly(TGT[tag][1], 3).n_mellin_args()}, poly4 {s0.poly(TGT[tag][1], 4).n_mellin_args()}")

    # ------------------------------------------------------------------ Q4
    print("\n" + "=" * 78 + f"\nQ4 continuity as alpha -> 1/2 (LS in {DPS_Q4} digits, noise = |{DPS_Q4} - "
          f"{DPS_Q4_CHECK} digits|)\n" + "=" * 78)
    q4_pass, fails = True, []
    q4data, q4f64, q4fail_keys = {}, {}, []
    for tag in CORE:
        f = TGT[tag][1]
        A, B = q4_target(f, DPS_Q4), q4_target(f, DPS_Q4_CHECK)
        q4data[tag] = (A, B)
        lim = B["limit"]
        print(f"\n{tag}: limit fit {{1, x, x ln x, x ln^2 x}}: L2 = {fmt(lim['l2'], 6)}, moments j=1..4 = "
              f"{[mp.nstr(v, 10) for v in lim['m']]}")
        print(f"{'sgn':>4}{'eps':>7}{'L2':>12}" + "".join(f"{n:>10}" for n in DNAMES) + f"{'max noise':>11}"
              f"{'max|k|':>10}{'f64 dL2':>10}{'f64 dm':>10}")
        for sign in (-1, 1):
            seqs = [[] for _ in DNAMES]
            noises = [[] for _ in DNAMES]
            for e in EPS_Q4:
                ra, rb = A[(sign, e)], B[(sign, e)]
                nz = [abs(x - y) for x, y in zip(ra["d"], rb["d"])]
                for i in range(len(DNAMES)):
                    seqs[i].append(rb["d"][i])
                    noises[i].append(nz[i])
                l2f, mf, _ = q4_float64(f, sign, e)
                dl2f = abs(l2f - float(rb["l2"])) / float(rb["l2"])
                dmf = max(float(abs(mp.mpf(x) - y) / abs(y)) for x, y in zip(mf, rb["m"]))
                q4f64[(tag, sign, e)] = (dl2f, dmf)
                print(f"{'+' if sign > 0 else '-':>4}{'1e-' + str(e):>7}{fmt(rb['l2'], 4):>12}"
                      + "".join(f"{fmt(v, 1):>10}" for v in rb["d"]) + f"{fmt(max(nz), 0):>11}"
                      f"{fmt(rb['kmax'], 1):>10}{dl2f:>10.1e}{dmf:>10.1e}")
            for i, name in enumerate(DNAMES):
                d, nz = seqs[i], noises[i]
                steps = all(d[k + 1] <= d[k] or d[k + 1] <= 10 * nz[k + 1] for k in range(len(d) - 1))
                conv = d[-1] <= mp.mpf("1e-3") * d[0] or d[-1] <= 10 * nz[-1]
                if not (steps and conv):
                    q4_pass = False
                    q4fail_keys.append((tag, sign, i))
                    fails.append(f"{tag} sign {sign:+d} {name}: steps {'ok' if steps else 'NOT monotone'}, "
                                 f"final/first {mp.nstr(d[-1] / d[0], 3)}")
            orders = [[float(mp.log10(seqs[i][k] / seqs[i][k + 1])) for k in range(len(EPS_Q4) - 1)]
                      for i in range(len(DNAMES))]
            print(f"  D3 orders per decade (sign {sign:+d}): "
                  + "; ".join(f"{n} " + ",".join(f"{o:.2f}" for o in od) for n, od in zip(DNAMES, orders)))
    for line in fails:
        print("  Q4 failure: " + line)
    print(f"Q4 criterion: monotone convergence within noise, all targets/signs/distances -> "
          f"{'PASS' if q4_pass else 'FAIL'}")

    # ------------------------------------------------------------------ verdict
    make_figure(rows, opt)
    print(f"\nfigure -> {os.path.relpath(FIG_PATH, HERE)}")
    print("\n" + "=" * 78 + "\nPRE-REGISTERED VERDICT\n" + "=" * 78)
    verdict = {"Q1": "PASS" if q1_pass else "FAIL", "Q2": q2, "Q3": "PASS" if q3_pass else "FAIL",
               "Q4": "PASS" if q4_pass else "FAIL"}
    for k, v in verdict.items():
        print(f"{k}: {v}")
    failed = [k for k, v in verdict.items() if v != "PASS"]
    print("DECISION: " + ("ADOPT the confluent treatment" if not failed else
                          f"FALLBACK (excluded neighbourhood + extended precision); failed: {', '.join(failed)}"))

    print("\n" + "=" * 78 + "\nPOST-HOC (not pre-registered; added after the first run, no bearing on the "
          "decision above)\n" + "=" * 78)
    post_hoc(fits, rows, q2data, q4data, q4f64, q4fail_keys)
    print(f"\nelapsed {time.time() - t0:.0f} s")
    return 0


# ---------------------------------------------------------------------------
# POST-HOC (written after the first run, 2026-09-26; it does not alter any
# pre-registered question, threshold or verdict above).
#  PH1 Q4 monotonicity restricted to the asymptotic range eps = 1e-2 ... 1e-6,
#      same noise rule, convergence d(1e-6) <= 1e-3 d(1e-2).
#  PH2 the sequences that failed Q4: signed differences at eps = 1e-1, 1e-2.
#  PH3 raw-coefficient blow-up predicted by the corollary: the coefficient of
#      x^{p_4} behaves as (96/117) e2 / eps^2, e2 = coefficient of x ln^2 x in
#      the limit fit; checked at eps = +-1e-6 in 120 digits.
#  PH4 summary maxima quoted in the draft's practical paragraph.
#  PH5 cost of the 50-digit closed form (RAW, j = 1..8, M4).
# ---------------------------------------------------------------------------

def post_hoc(fits, rows, q2data, q4data, q4f64, q4fail_keys):
    print("PH1 Q4 criterion restricted to eps = 1e-2 ... 1e-6:")
    n_ok = n_all = 0
    for tag in CORE:
        A, B = q4data[tag]
        for sign in (-1, 1):
            for i, name in enumerate(DNAMES):
                d = [B[(sign, e)]["d"][i] for e in EPS_Q4[1:]]
                nz = [abs(A[(sign, e)]["d"][i] - B[(sign, e)]["d"][i]) for e in EPS_Q4[1:]]
                steps = all(d[k + 1] <= d[k] or d[k + 1] <= 10 * nz[k + 1] for k in range(len(d) - 1))
                conv = d[-1] <= mp.mpf("1e-3") * d[0] or d[-1] <= 10 * nz[-1]
                n_all += 1
                n_ok += steps and conv
    print(f"  {n_ok}/{n_all} sequences monotone and convergent on eps = 1e-2 ... 1e-6")

    print("PH2 sequences that failed Q4 -- signed difference (quantity(eps) - quantity(limit)):")
    for tag, sign, i in q4fail_keys:
        B = q4data[tag][1]
        lim = B["limit"]
        vals = []
        for e in (1, 2):
            r = B[(sign, e)]
            if i == 0:
                v = r["l2"] - lim["l2"]
            elif i == 1:
                v = r["d"][1]
            else:
                v = (r["m"][i - 2] - lim["m"][i - 2]) / abs(lim["m"][i - 2])
            vals.append(v)
        print(f"  {tag} sign {sign:+d} {DNAMES[i]}: alpha = {0.5 + sign * 0.1:.2f}: {mp.nstr(vals[0], 4)}; "
              f"alpha = {0.5 + sign * 0.01:.2f}: {mp.nstr(vals[1], 4)}; alpha* = {fits[tag]['popt'].alpha:.4f}")

    print("PH3 coefficient of x^{p_4} times eps^2 versus (96/117) e2 (120 digits):")
    with mp.workdps(DPS_Q4):
        L = grid_logs()
        xs = [mp.mpf(float(x)) for x in X_GRID]
        one = ones_col()
        for tag in CORE:
            y = [mp.mpf(float(v)) for v in TGT[tag][1](X_GRID)]
            kl, _ = ls_mp([one, xs, [x * l for x, l in zip(xs, L)], [x * l * l for x, l in zip(xs, L)]], y)
            pred = mp.mpf(96) / 117 * kl[3]
            for sign in (-1, 1):
                eps = sign * mp.mpf(10) ** -6
                q = [p_mp(i, mp.mpf(1) / 2 + eps) for i in IDX]
                kk, _ = ls_mp([one] + [pow_col(v) for v in q], y)
                print(f"  {tag} eps = {mp.nstr(eps, 2)}: k4 eps^2 = {mp.nstr(kk[3] * eps ** 2, 8)}, "
                      f"(96/117) e2 = {mp.nstr(pred, 8)}, ratio = {mp.nstr(kk[3] * eps ** 2 / pred, 8)}")

    print("PH4 summary maxima:")
    dq_max = max(v[1] for v in q2data.values())
    print(f"  Q2 max |quadrature - 50-digit| / |50-digit| over all {len(q2data)} cells: {dq_max:.2e}")
    conf64 = max(v[0] for k, v in q2data.items() if k[1] == "conf")
    print(f"  Q2 max CONF float64 deviation over j <= 8 and M1, M4, M6: {conf64:.2e}")
    for tag in CORE:
        ok = [j for j in J8 if q2data[(tag, "raw", j)][0] <= 1e-10]
        first_bad = next((j for j in J8 if q2data[(tag, "raw", j)][0] > 1e-10), None)
        print(f"  Q2 RAW float64 deviation <= 1e-10 for j in {ok} on {tag}; first j above 1e-10: {first_bad}")
    f64_1e3 = max(q4f64[(t, s, 3)][0] for t in CORE for s in (-1, 1))
    c1e3 = [float(r["cond_raw"]) for r in rows if abs(abs(r["amh"]) - 1e-3) < 1e-12]
    print(f"  Q4 float64 relative L2 deviation at |alpha - 1/2| = 1e-3, max over targets and signs: {f64_1e3:.2e}")
    print(f"  cond(RAW) at |alpha - 1/2| = 1e-3 (both sides): {', '.join(f'{c:.3e}' for c in c1e3)}")

    sgt = fits["M4"]["raw"]
    best = float("inf")
    for _ in range(5):
        t1 = time.perf_counter()
        for j in J8:
            sgt.moment(j, "mp")
        best = min(best, time.perf_counter() - t1)
    print(f"PH5 50-digit closed form, RAW PATP-opt on M4, all j = 1..8: {best * 1e3:.1f} ms (min of 5)")


if __name__ == "__main__":
    sys.exit(main())
