"""
P4 of the revised analysis: heavy-tail benchmark.

QUESTION
  Proposition `prop:heavytail` shows that PATP-MUET with alpha in [0, 1/2] needs a finite
  moment of X of order j*P_S(alpha) <= j for the j-th output moment, whereas degree-S
  polynomial MUET needs order j*S. Question: does this enlarged admissible region yield
  ACCURATE engineering quantities rather than only finite formal moments? If not, the
  heavy-tail discussion stays an admissibility result.
  The crux is extrapolation: a surrogate is fitted where g was evaluated, on a bounded
  design range [0, x_hi], but the moments integrate over the unbounded support.

======================================================================================
PRE-REGISTERED DESIGN, HYPOTHESES AND DECISION RULE
(written before the first run and not changed afterwards; anything added after the
first run lives below the marker "POST-HOC" at the end of this file)
======================================================================================

INPUT LAW
  Primary family: Lomax (Pareto type II), f(x) = (a/lam) (1 + x/lam)^-(a+1), x >= 0,
  tail indices a in {3, 4, 6}. E[X^r] < inf iff -1 < r < a, so the index b - 1 of
  Prop. heavytail equals a. Mellin transform: t = x/(lam + x) maps X to Beta(1, a)
  (x = lam t/(1-t), f(x) dx = a (1-t)^(a-1) dt), hence
      M_X(s) = E[X^(s-1)] = a lam^(s-1) B(s, a+1-s)
             = lam^(s-1) Gamma(s) Gamma(a+1-s) / Gamma(a),     0 < Re s < a+1,
  checked against direct quadrature in block V1.
  Scale lam = a - 1, so E[X] = 1: X is the random load in units of the static preload of
  R1, and the mean load equals the preload. The bulk of X then spans the transition from
  the linear regime (X << 1) to the Hertzian regime (X >> 1) of R1.
  Secondary family (sensitivity S4 only): log-logistic, F(x) = 1/(1 + (x/sig)^-beta),
  beta in {3, 4, 6}, M_X(s) = sig^(s-1) Gamma(1 + (s-1)/beta) Gamma(1 - (s-1)/beta),
  1 - beta < Re s < 1 + beta, sig = sin(pi/beta)/(pi/beta) so that E[X] = 1. Its density
  vanishes at 0, so the bulk sits away from the origin. Tail index b - 1 = beta.

RESPONSES
  R1 (physical; the only response in the decision rule): preloaded Hertzian point
     contact. A sphere-on-flat contact under normal force F has approach delta = C F^(2/3)
     (Hertz theory). With a static preload F0 and a random additional load X F0, the
     additional approach in units of C F0^(2/3) is
         g1(x) = (1 + x)^(2/3) - 1.
     Linear at 0 (g1 ~ 2x/3), ~ x^(2/3) as x -> inf; not in the span of any tested basis
     (integer powers or a finite set of pure fractional powers). E[g1(X)^j] < inf iff
     2j/3 < a, i.e. j = 1..4 are finite for every a in {3, 4, 6}.
  R2 (control; analytic at 0, sub-linear, NO power-law growth): g2(x) = ln(1 + x); every
     moment is finite for every a. It tests whether a PATP advantage, if any, depends on
     R1's power-law tail. (R1 is itself analytic at 0: its fractional structure is
     asymptotic, which is exactly where a heavy tail lives; the control differs there.)

DESIGN (the surrogate sees g only at design nodes in [0, x_hi])
  x_hi = F^-1(q), q = 0.999 (primary). Design D1: N = 401 nodes at the input quantiles,
  x_k = F^-1(u_k), u_k = linspace(0, q, N) (so x_1 = 0, x_N = x_hi). For X ~ U[0,1] this
  is exactly the uniform grid of rq3/step0, i.e. the paper's convention generalized to a
  non-uniform input; it discretizes the input-weighted L2 fit on [0, x_hi]. OLS with
  equal weights on the nodes, as in step0.

COMPETITORS (the same nodes for every surrogate; moments of every surrogate evaluated in
50-digit arithmetic by the closed form -- Thm patp-muet for PATP, the multinomial formula
for polynomials -- with the Mellin values of the input law)
  poly3-full   degree-3 polynomial propagated with the full input law. Needs E[X^(3j)]:
               "undef" when 3j >= a.
  poly3-trunc  the same surrogate with the input replaced by the truncated law
               X | X <= x_hi (renormalized); always finite, biased.
  poly1        degree-1 polynomial, full input law; admissible for j < a.
  PATP-r       PATP, S = 3 (basis i = 2, 3, 4), alpha restricted to [0, 1/2 - delta],
               delta = 0.05, i.e. [0, 0.45]; admissible for j P_3(alpha) < a, so j = 1, 2
               are always admissible for a >= 3 (P_3 <= 1 on [0, 1/2]).
               Why delta = 0.05: at alpha = 1/2 all p_i = 1 and the design matrix loses
               rank; near it p_i ~ 1 - (i - 1/i)(1/2 - alpha), so the exponent spread is
               ~ 2.25 (1/2 - alpha). delta = 0.05 keeps the spread >= ~0.1 and excludes
               the near-degenerate cluster where step 0 found ill-conditioned fits
               (M4: alpha* = 0.520, cond(A) = 9.7e4). Admissibility itself holds on all
               of [0, 1/2]; delta is a conditioning choice only.
  PATP-opt     PATP, alpha over all of [0, 1] (the paper's current optimizer range),
               admissible or not; reported either way.
  GJ-8, GJ-16  direct Gauss quadrature of g with respect to the input law, n = 8, 16.
               A Golub-Welsch rule from the moments of X does not exist here: an n-point
               rule needs E[X^k] for k <= 2n - 1, and the Lomax law has them only for
               k < a (n <= 2 for a = 3, 4; n <= 3 for a = 6). Substitution rule instead:
               t = X/(lam + X) ~ Beta(1, a), then the n-point Gauss-Jacobi rule of the
               weight (1-t)^(a-1) on [0, 1] (log-logistic: t = F(X) ~ U(0,1),
               Gauss-Legendre). This reference evaluates g at n nodes anywhere on
               [0, inf), beyond x_hi included: it is the "g is evaluable" reference.
  g-trunc      descriptive row, not a competitor: the EXACT g with the truncated input,
               i.e. the floor of any truncated-input method.

  alpha search (both PATP variants): mean squared residual on the nodes over a grid of
  step 0.0025 on the admissible interval, then bounded Brent (xatol 1e-8) inside the
  bracket of the best grid point; boundary solutions are accepted as they are. The
  optimizer never sees any moment of g(X).
  Truth: E[g(X)^j] by tanh-sinh quadrature (mpmath) in the t-variable over the whole
  support, cross-checked against scipy adaptive quadrature over [0, x_hi] and
  [x_hi, inf) (block V2). The closed-form surrogate moments are checked against direct
  quadrature of the same surrogate (block V3).

METRICS
  Relative error of E[g^j], j = 1..4 (where finite); relative error of the mean (j = 1)
  and of the standard deviation sqrt(E[g^2] - E[g]^2).
  Descriptive (pre-registered, not in any hypothesis): RMS fit residual on the nodes,
  cond(A) of the PATP design matrix, number of g evaluations, number of distinct Mellin
  values for j <= 4, the growth exponent of each surrogate, the extrapolation ratio
  g_hat(x)/g(x) at x in {x_hi, 3 x_hi, 10 x_hi}, the share of E[g^2] carried by
  X > x_hi, and for j = 2 the split of each surrogate's error into the parts from
  X <= x_hi and X > x_hi.

HYPOTHESES (response R1, primary configuration, evaluated separately per a)
  H1  PATP-r returns a finite mean and a finite standard deviation -- a case where
      poly3-full is undefined, since the standard deviation needs E[X^6] = inf for every
      a <= 6 -- and both relative errors are <= 5 %.
  H2  PATP-r's relative error of the standard deviation is strictly smaller than that of
      poly3-trunc and that of poly1.
  H3  No directional prediction against GJ-8 / GJ-16; reported.

DECISION RULE
  If H1 and H2 both hold for R1 in at least 2 of the 3 tail indices (primary
  configuration), the paper may state that the enlarged admissible region yields
  accurate moments for sub-linear responses to heavy-tailed inputs, with the scope:
    - control R2: if H1 and H2 also hold for R2 in >= 2 of 3, the scope sentence covers
      sub-linear responses in general; otherwise it is limited to responses with
      power-law growth such as R1, and the paper says that PATP-r does not win on the
      logarithmic control;
    - sensitivity S1-S4: every configuration in which the R1 verdict (H1 and H2 in >= 2
      of 3) differs from the primary one is reported by name as a limitation.
  Otherwise the heavy-tail result stays an admissibility statement and the paper says so.

PRE-REGISTERED SENSITIVITY (one factor at a time from the primary; reported, not decisive)
  S1  q = 0.9999                    (design range farther into the tail)
  S2  design uniform in x on [0, x_hi], N = 401  (operating-range DoE)
  S3  N = 21 quantile nodes         (small design, "g is expensive")
  S4  log-logistic input family, primary design

Run (from experiments/):
  ../verification/cas/.venv/bin/python p4_heavy_tail.py | tee results/p4_heavy_tail.txt
"""

from __future__ import annotations

import sys
import time
import warnings
from itertools import combinations_with_replacement
from math import factorial

import mpmath as mp
import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import minimize_scalar
from scipy.special import roots_jacobi, roots_legendre

mp.mp.dps = 50
QUAD_DPS = 30

J = (1, 2, 3, 4)
TAILS = (3, 4, 6)
DELTA = 0.05
ALPHA_R = (0.0, 0.5 - DELTA)
ALPHA_F = (0.0, 1.0)
GRID_STEP = 0.0025
THRESH = 0.05
GJ_N = (8, 16)

PRIMARY = dict(family="lomax", design="quantile", N=401, q=0.999)
SENSITIVITY = [
    ("S1 q=0.9999", dict(PRIMARY, q=0.9999)),
    ("S2 uniform-x design", dict(PRIMARY, design="uniform")),
    ("S3 N=21 nodes", dict(PRIMARY, N=21)),
    ("S4 log-logistic input", dict(PRIMARY, family="loglogistic")),
]


def p_num(i: int, alpha: float) -> float:
    return 1.0 / i + (4 - i - 3.0 / i) * alpha + (2 * i - 4 + 2.0 / i) * alpha ** 2


def P3(alpha: float) -> float:
    return max(p_num(i, alpha) for i in (2, 3, 4))


# ---------------------------------------------------------------------------
# Input laws. Each exposes: ppf (numpy), a map t <-> x with the density of t
# (w_t) for quadrature, the Mellin value E[X^e] (None outside the strip), the
# partial moment E[X^e; X <= c], and its substitution Gauss rule.
# ---------------------------------------------------------------------------

class Lomax:
    family = "Lomax"

    def __init__(self, a: int):
        self.a, self.lam = float(a), float(a - 1)
        self.A, self.L = mp.mpf(a), mp.mpf(a - 1)
        self.label = f"Lomax(a={a}, lambda={a - 1})"
        self.rule = "Gauss-Jacobi in t = X/(lambda+X), weight (1-t)^(a-1)"

    def ppf(self, u):
        return self.lam * ((1.0 - np.asarray(u, float)) ** (-1.0 / self.a) - 1.0)

    def cdf(self, x):
        return 1 - (1 + mp.mpf(x) / self.L) ** (-self.A)

    def pdf_np(self, x):
        return (self.a / self.lam) * (1.0 + x / self.lam) ** (-(self.a + 1.0))

    def pdf_mp(self, x):
        return (self.A / self.L) * (1 + x / self.L) ** (-(self.A + 1))

    def t_of_x(self, x):
        x = mp.mpf(x)
        return x / (self.L + x)

    def x_of_t(self, t):
        return self.L * t / (1 - t)

    def w_t(self, t):
        return self.A * (1 - t) ** (self.A - 1)

    # v = 1 - t, used by the quadrature so that the singular end t -> 1 sits at v -> 0
    def v_of_x(self, x):
        x = mp.mpf(x)
        return self.L / (self.L + x)

    def x_of_v(self, v):
        return self.L * (1 - v) / v

    def w_v(self, v):
        return self.A * v ** (self.A - 1)

    def mellin(self, e):
        e = mp.mpf(e)
        if e >= self.A or e <= -1:
            return None
        return self.L ** e * mp.gamma(e + 1) * mp.gamma(self.A - e) / mp.gamma(self.A)

    def partial_moment(self, e, c):
        e = mp.mpf(e)
        return self.A * self.L ** e * mp.betainc(e + 1, self.A - e, 0, self.t_of_x(c))

    def gauss(self, n: int):
        xi, w = roots_jacobi(n, self.a - 1.0, 0.0)
        t = (1.0 + xi) / 2.0
        return self.lam * t / (1.0 - t), w * self.a * 2.0 ** (-self.a)


class LogLogistic:
    family = "log-logistic"

    def __init__(self, beta: int):
        self.a = float(beta)
        self.sig = float(np.sin(np.pi / beta) / (np.pi / beta))
        self.A = mp.mpf(beta)
        self.S = mp.sin(mp.pi / self.A) / (mp.pi / self.A)
        self.label = f"log-logistic(beta={beta}, sigma={self.sig:.4f})"
        self.rule = "Gauss-Legendre in t = F(X)"

    def ppf(self, u):
        u = np.asarray(u, float)
        return self.sig * (u / (1.0 - u)) ** (1.0 / self.a)

    def cdf(self, x):
        z = (mp.mpf(x) / self.S) ** self.A
        return z / (1 + z)

    def pdf_np(self, x):
        z = (x / self.sig) ** self.a
        return (self.a / self.sig) * (x / self.sig) ** (self.a - 1.0) / (1.0 + z) ** 2

    def pdf_mp(self, x):
        z = (x / self.S) ** self.A
        return (self.A / self.S) * (x / self.S) ** (self.A - 1) / (1 + z) ** 2

    def t_of_x(self, x):
        return self.cdf(x)

    def x_of_t(self, t):
        return self.S * (t / (1 - t)) ** (1 / self.A)

    def w_t(self, t):
        return mp.mpf(1)

    def v_of_x(self, x):
        return 1 / (1 + (mp.mpf(x) / self.S) ** self.A)

    def x_of_v(self, v):
        return self.S * ((1 - v) / v) ** (1 / self.A)

    def w_v(self, v):
        return mp.mpf(1)

    def mellin(self, e):
        e = mp.mpf(e)
        if e >= self.A or e <= -self.A:
            return None
        return self.S ** e * mp.gamma(1 + e / self.A) * mp.gamma(1 - e / self.A)

    def partial_moment(self, e, c):
        e = mp.mpf(e)
        return self.S ** e * mp.betainc(1 + e / self.A, 1 - e / self.A, 0, self.cdf(c))

    def gauss(self, n: int):
        xi, w = roots_legendre(n)
        t = (1.0 + xi) / 2.0
        return self.sig * (t / (1.0 - t)) ** (1.0 / self.a), w / 2.0


def make_law(family: str, a: int):
    return Lomax(a) if family == "lomax" else LogLogistic(a)


def expect(law, h, x_lo=0.0, x_hi=None):
    """E[h(X); x_lo < X <= x_hi] by tanh-sinh quadrature in the t-variable.

    Implemented in the reflected variable v = 1 - t: the singular end t -> 1 (x -> inf)
    then sits at v -> 0, where tanh-sinh nodes keep full relative precision (in t they
    round to 1 and x_of_t divides by zero).
    """
    with mp.workdps(QUAD_DPS):
        v_lo = law.v_of_x(x_lo)
        v_hi = mp.mpf(0) if x_hi is None else law.v_of_x(x_hi)
        pts = [v_hi]
        for u in (0.9999, 0.999, 0.99, 0.9, 0.5):
            b = float(law.ppf(u))
            if b > x_lo and (x_hi is None or b < x_hi):
                pts.append(law.v_of_x(b))
        pts.append(v_lo)
        return mp.quad(lambda v: h(law.x_of_v(v)) * law.w_v(v), pts)


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

RESPONSES = {
    "R1": dict(label="Hertz contact with preload, (1+x)^(2/3) - 1", growth=2.0 / 3.0,
               np=lambda x: np.expm1((2.0 / 3.0) * np.log1p(x)),
               mp=lambda x: mp.expm1(mp.mpf(2) / 3 * mp.log1p(x))),
    "R2": dict(label="control ln(1+x)", growth=0.0,
               np=np.log1p, mp=mp.log1p),
}


# ---------------------------------------------------------------------------
# Surrogates g_hat(x) = sum_k c_k x^{e_k}, e_0 = 0, and their closed-form moments
# E[g_hat^j] = sum_{|kappa|=j} multinom(kappa) prod c^kappa E[X^{sum kappa e}].
# This is Thm patp-muet for the PATP basis (the binomial split in k_0 followed by
# the multinomial over i >= 2 is the same expansion) and classical MUET for the
# polynomial basis. None = some required Mellin value is infinite.
# ---------------------------------------------------------------------------

class Surr:
    def __init__(self, name, exps, coefs, **meta):
        self.name = name
        self.exps = [float(e) for e in exps]
        self.coefs = np.asarray(coefs, float)
        self.meta = meta
        self._E = [mp.mpf(e) for e in self.exps]
        self._C = [mp.mpf(float(c)) for c in self.coefs]

    def np_eval(self, x):
        x = np.asarray(x, float)
        return sum(c * (np.ones_like(x) if e == 0 else x ** e) for c, e in zip(self.coefs, self.exps))

    def mp_eval(self, x):
        return mp.fsum(c * (1 if e == 0 else x ** e) for c, e in zip(self._C, self._E))

    def moment(self, j, mellin):
        total = mp.mpf(0)
        n = len(self._E)
        for combo in combinations_with_replacement(range(n), j):
            kap = [combo.count(k) for k in range(n)]
            e = mp.fsum(kk * E for kk, E in zip(kap, self._E))
            M = mellin(e)
            if M is None:
                return None
            mult, prod = mp.mpf(factorial(j)), mp.mpf(1)
            for kk, C in zip(kap, self._C):
                if kk:
                    mult /= factorial(kk)
                    prod *= C ** kk
            total += mult * prod * M
        return total

    def growth(self):
        return max(e for e, c in zip(self.exps, self.coefs) if c != 0.0)

    def n_mellin(self, jmax=4):
        nonconst = [e for e in self.exps if e != 0]
        args = {0.0}
        for j in range(1, jmax + 1):
            for combo in combinations_with_replacement(nonconst, j):
                args.add(round(sum(combo), 12))
        return len(args)


def ols(x, y, exps):
    A = np.column_stack([np.ones_like(x) if e == 0 else x ** e for e in exps])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c, A


def fit_poly(x, y, d, name):
    exps = list(range(d + 1))
    c, A = ols(x, y, exps)
    return Surr(name, exps, c, rms=float(np.sqrt(np.mean((A @ c - y) ** 2))))


def patp_exps(alpha):
    return [0.0] + [p_num(i, alpha) for i in (2, 3, 4)]


def fit_patp(x, y, lo, hi, name):
    def loss(a):
        c, A = ols(x, y, patp_exps(a))
        return float(np.mean((A @ c - y) ** 2))

    n = int(round((hi - lo) / GRID_STEP)) + 1
    grid = np.linspace(lo, hi, n)
    losses = np.array([loss(a) for a in grid])
    k = int(np.argmin(losses))
    br = (grid[max(k - 1, 0)], grid[min(k + 1, n - 1)])
    res = minimize_scalar(loss, bounds=br, method="bounded", options={"xatol": 1e-8})
    alpha = float(res.x) if res.fun < losses[k] else float(grid[k])
    exps = patp_exps(alpha)
    c, A = ols(x, y, exps)
    return Surr(name, exps, c, alpha=alpha, cond=float(np.linalg.cond(A)),
                rms=float(np.sqrt(np.mean((A @ c - y) ** 2))),
                at_bound=bool(abs(alpha - hi) < 1e-6 or abs(alpha - lo) < 1e-6))


def design(law, cfg):
    x_hi = float(law.ppf(cfg["q"]))
    if cfg["design"] == "quantile":
        x = law.ppf(np.linspace(0.0, cfg["q"], cfg["N"]))
    else:
        x = np.linspace(0.0, x_hi, cfg["N"])
    return x, x_hi


# ---------------------------------------------------------------------------
# Caches for truth and truncated-law quantities (they do not depend on design)
# ---------------------------------------------------------------------------

_TRUTH, _TRUNC_TRUTH, _V2 = {}, {}, []


def truth(law, rkey, j):
    key = (law.label, rkey, j)
    if key not in _TRUTH:
        r = RESPONSES[rkey]
        if r["growth"] * j >= law.a:
            _TRUTH[key] = None
        else:
            g = r["mp"]
            _TRUTH[key] = expect(law, lambda x: g(x) ** j)
    return _TRUTH[key]


def truth_scipy(law, rkey, j, x_hi):
    g = RESPONSES[rkey]["np"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        f = lambda x: g(x) ** j * law.pdf_np(x)
        return quad(f, 0.0, x_hi, limit=500)[0] + quad(f, x_hi, np.inf, limit=500)[0]


def trunc_truth(law, rkey, j, x_hi):
    key = (law.label, rkey, j, round(x_hi, 12))
    if key not in _TRUNC_TRUTH:
        g = RESPONSES[rkey]["mp"]
        _TRUNC_TRUTH[key] = expect(law, lambda x: g(x) ** j, 0.0, x_hi) / law.cdf(x_hi)
    return _TRUNC_TRUTH[key]


def trunc_mellin(law, x_hi):
    Fc = law.cdf(x_hi)
    cache = {}

    def m(e):
        k = mp.nstr(e, 40)
        if k not in cache:
            cache[k] = law.partial_moment(e, x_hi) / Fc
        return cache[k]
    return m


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def mean_std(mom):
    m1, m2 = mom.get(1), mom.get(2)
    if m1 is None or m2 is None:
        return m1, None
    var = m2 - m1 ** 2
    return m1, (mp.sqrt(var) if var > 0 else None)


def errors(mom, tru):
    rel = {}
    for j in J:
        m, t = mom.get(j), tru.get(j)
        rel[j] = None if (m is None or t is None) else float(abs(m - t) / abs(t))
    mh, sh = mean_std(mom)
    mt, st = mean_std(tru)
    mean_err = None if mh is None else float(abs(mh - mt) / abs(mt))
    std_err = None if sh is None else float(abs(sh - st) / abs(st))
    return dict(rel=rel, mean_err=mean_err, std_err=std_err)


def fmt(v, w=10):
    return f"{'undef':>{w}}" if v is None else f"{v:>{w}.2e}"


def run_case(law, cfg, rkey, verbose):
    x, x_hi = design(law, cfg)
    gnp, gmp = RESPONSES[rkey]["np"], RESPONSES[rkey]["mp"]
    y = gnp(x)
    surr = {
        "poly3": fit_poly(x, y, 3, "poly3"),
        "poly1": fit_poly(x, y, 1, "poly1"),
        "PATP-r": fit_patp(x, y, *ALPHA_R, "PATP-r"),
        "PATP-opt": fit_patp(x, y, *ALPHA_F, "PATP-opt"),
    }
    tru = {j: truth(law, rkey, j) for j in J}
    tm = trunc_mellin(law, x_hi)
    moms = {
        "poly3-full": {j: surr["poly3"].moment(j, law.mellin) for j in J},
        "poly3-trunc": {j: surr["poly3"].moment(j, tm) for j in J},
        "poly1": {j: surr["poly1"].moment(j, law.mellin) for j in J},
        "PATP-r": {j: surr["PATP-r"].moment(j, law.mellin) for j in J},
        "PATP-opt": {j: surr["PATP-opt"].moment(j, law.mellin) for j in J},
    }
    gj_max = {}
    for n in GJ_N:
        xs, ws = law.gauss(n)
        gj_max[n] = float(xs.max())
        moms[f"GJ-{n}"] = {j: mp.fsum(mp.mpf(float(w)) * gmp(mp.mpf(float(xx))) ** j
                                      for xx, w in zip(xs, ws)) for j in J}
    moms["g-trunc"] = {j: trunc_truth(law, rkey, j, x_hi) for j in J}
    res = {k: errors(v, tru) for k, v in moms.items()}

    if verbose:
        print(f"\n--- {law.label}, {rkey}: {RESPONSES[rkey]['label']} ---")
        tail_share = 1 - expect(law, lambda z: gmp(z) ** 2, 0.0, x_hi) / tru[2]
        print(f"x_hi = F^-1({cfg['q']}) = {x_hi:.4f};  share of E[g^2] from X > x_hi = "
              f"{float(tail_share):.3e};  truth E[g] = {float(tru[1]):.10f}, "
              f"std = {float(mean_std(tru)[1]):.10f}")
        for k in ("PATP-r", "PATP-opt"):
            s = surr[k]
            adm = [j for j in J if j * P3(s.meta['alpha']) < law.a]
            print(f"{k}: alpha* = {s.meta['alpha']:.6f}{' (at bound)' if s.meta['at_bound'] else ''}, "
                  f"exponents = {[round(e, 4) for e in s.exps[1:]]}, "
                  f"coefs = {[float(f'{c:.4g}') for c in s.coefs]}, cond(A) = {s.meta['cond']:.2e}, "
                  f"P_3 = {P3(s.meta['alpha']):.4f}, admissible j = {adm}")
        print(f"poly3 coefs = {[float(f'{c:.4g}') for c in surr['poly3'].coefs]};  "
              f"poly1 coefs = {[float(f'{c:.4g}') for c in surr['poly1'].coefs]}")
        print(f"GJ largest node: n=8 -> {gj_max[8]:.1f}, n=16 -> {gj_max[16]:.1f}  ({law.rule})")
        hdr = (f"{'method':<12}{'n_g':>5}{'n_M':>5}{'rms':>10}" + "".join(f"{'j=' + str(j):>10}" for j in J)
               + f"{'mean':>10}{'std':>10}")
        print(hdr)
        info = {"poly3-full": (cfg["N"], surr["poly3"]), "poly3-trunc": (cfg["N"], surr["poly3"]),
                "poly1": (cfg["N"], surr["poly1"]), "PATP-r": (cfg["N"], surr["PATP-r"]),
                "PATP-opt": (cfg["N"], surr["PATP-opt"])}
        for k, r in res.items():
            if k in info:
                n_g, s = info[k]
                lead = f"{k:<12}{n_g:>5}{s.n_mellin():>5}{s.meta['rms']:>10.2e}"
            elif k.startswith("GJ"):
                lead = f"{k:<12}{int(k[3:]):>5}{0:>5}{'-':>10}"
            else:
                lead = f"{k:<12}{'-':>5}{'-':>5}{'-':>10}"
            print(lead + "".join(fmt(r["rel"][j]) for j in J) + fmt(r["mean_err"]) + fmt(r["std_err"]))

        # extrapolation ratio and growth exponent
        print("extrapolation g_hat(x)/g(x) at x = x_hi, 3 x_hi, 10 x_hi; growth exponent of g_hat:")
        xs = np.array([x_hi, 3 * x_hi, 10 * x_hi])
        for k in ("poly3", "poly1", "PATP-r", "PATP-opt"):
            s = surr[k]
            ratio = s.np_eval(xs) / gnp(xs)
            print(f"  {k:<9} " + "  ".join(f"{v:8.4f}" for v in ratio) + f"   growth {s.growth():.4f}")
        print(f"  {'truth':<9} growth {RESPONSES[rkey]['growth']:.4f}" + ("  (logarithmic)" if rkey == "R2" else ""))

        # V3: closed form vs direct quadrature of the same surrogate (j = 1, 2)
        worst = 0.0
        for k, s, lawm in (("poly1", surr["poly1"], None), ("PATP-r", surr["PATP-r"], None),
                           ("PATP-opt", surr["PATP-opt"], None), ("poly3-trunc", surr["poly3"], "trunc")):
            for j in (1, 2):
                cf = moms[k][j]
                if cf is None:
                    continue
                if lawm == "trunc":
                    q_ = expect(law, lambda z: s.mp_eval(z) ** j, 0.0, x_hi) / law.cdf(x_hi)
                else:
                    q_ = expect(law, lambda z: s.mp_eval(z) ** j)
                worst = max(worst, float(abs(cf - q_) / abs(q_)))
        print(f"V3 closed form vs quadrature of the same surrogate, j=1,2: max rel diff = {worst:.1e}")

        # error split for j = 2
        print("error of E[g^2], split by region (relative to E[g^2]):")
        for k in ("poly1", "PATP-r", "PATP-opt"):
            s = surr[k]
            if moms[k][2] is None:
                print(f"  {k:<11} undefined")
                continue
            bulk = expect(law, lambda z: s.mp_eval(z) ** 2 - gmp(z) ** 2, 0.0, x_hi)
            tail = expect(law, lambda z: s.mp_eval(z) ** 2 - gmp(z) ** 2, x_hi, None)
            print(f"  {k:<11} X<=x_hi {float(bulk / tru[2]):+.3e}   X>x_hi {float(tail / tru[2]):+.3e}")
        s = surr["poly3"]
        fit_part = moms["poly3-trunc"][2] - moms["g-trunc"][2]
        bias = moms["g-trunc"][2] - tru[2]
        print(f"  {'poly3-trunc':<11} fit on truncated law {float(fit_part / tru[2]):+.3e}   "
              f"truncation bias {float(bias / tru[2]):+.3e}")
    return res, surr


def h_flags(res):
    pr = res["PATP-r"]
    h1 = (pr["mean_err"] is not None and pr["std_err"] is not None
          and pr["mean_err"] <= THRESH and pr["std_err"] <= THRESH)
    comps = [res["poly3-trunc"]["std_err"], res["poly1"]["std_err"]]
    h2 = pr["std_err"] is not None and all(c is None or pr["std_err"] < c for c in comps)
    return h1, h2


def run_config(cfg, verbose):
    out = {}
    for a in TAILS:
        law = make_law(cfg["family"], a)
        for rkey in ("R1", "R2"):
            out[(a, rkey)] = run_case(law, cfg, rkey, verbose)
    return out


def summary(label, cfg, out):
    print(f"\n[{label}]  family={cfg['family']}, design={cfg['design']}, N={cfg['N']}, q={cfg['q']}")
    print(f"{'a':>2} {'resp':<4} {'alpha_r':>8} {'alpha_o':>8} | rel. error of std: "
          f"{'PATP-r':>9}{'PATP-opt':>9}{'p3-trunc':>9}{'poly1':>9}{'GJ-8':>9}{'GJ-16':>9} | "
          f"mean: {'PATP-r':>9}{'p3-trunc':>9}{'poly1':>9} | H1 H2")
    verdict = {}
    for (a, rkey), (res, surr) in out.items():
        h1, h2 = h_flags(res)
        verdict[(a, rkey)] = (h1, h2)
        std = [res[k]["std_err"] for k in ("PATP-r", "PATP-opt", "poly3-trunc", "poly1", "GJ-8", "GJ-16")]
        mean = [res[k]["mean_err"] for k in ("PATP-r", "poly3-trunc", "poly1")]
        print(f"{a:>2} {rkey:<4} {surr['PATP-r'].meta['alpha']:>8.4f} {surr['PATP-opt'].meta['alpha']:>8.4f} | "
              f"{'':>19}" + "".join(fmt(v, 9) for v in std) + " | "
              + "      " + "".join(fmt(v, 9) for v in mean)
              + f" | {'Y' if h1 else 'n'}  {'Y' if h2 else 'n'}")
    return verdict


def r1_passes(verdict, rkey="R1"):
    ok = [a for a in TAILS if all(verdict[(a, rkey)])]
    return ok, len(ok) >= 2


def verify_mellin():
    print("V1 Mellin closed form vs direct quadrature in x (mpmath, independent of the t-map):")
    worst = 0.0
    for fam in ("lomax", "loglogistic"):
        for a in TAILS:
            law = make_law(fam, a)
            for e in (-0.6, 0.0, 0.5, 1.0, 1.7, 2.0, a - 0.4):
                cf = law.mellin(e)
                with mp.workdps(QUAD_DPS):
                    qv = mp.quad(lambda z: z ** e * law.pdf_mp(z), [0, 0.5, 1, 4, mp.inf])
                worst = max(worst, float(abs(cf - qv) / abs(qv)))
    print(f"   max rel diff over 2 families x 3 tails x 7 orders = {worst:.1e}")
    print("V4 truncated moments E[X^e; X<=c] (incomplete beta) vs quadrature, e up to 12:")
    worst = 0.0
    for fam in ("lomax", "loglogistic"):
        for a in TAILS:
            law = make_law(fam, a)
            c = float(law.ppf(0.999))
            for e in (0.0, 0.7, 1.0, 3.0, 6.0, 12.0):
                cf = law.partial_moment(e, c)
                with mp.workdps(QUAD_DPS):
                    qv = mp.quad(lambda z: z ** e * law.pdf_mp(z), [0, 0.5, 1, c])
                worst = max(worst, float(abs(cf - qv) / abs(qv)))
    print(f"   max rel diff = {worst:.1e}")


def verify_truth(q=0.999):
    worst = 0.0
    for fam in ("lomax", "loglogistic"):
        for a in TAILS:
            law = make_law(fam, a)
            x_hi = float(law.ppf(q))
            for rkey in ("R1", "R2"):
                for j in J:
                    t = truth(law, rkey, j)
                    if t is None:
                        continue
                    s = truth_scipy(law, rkey, j, x_hi)
                    worst = max(worst, abs(float(t) - s) / abs(float(t)))
    print(f"V2 truth: mpmath tanh-sinh vs scipy adaptive quadrature, all cases, j=1..4: "
          f"max rel diff = {worst:.1e}")


def main() -> int:
    t0 = time.time()
    print("P4 heavy-tail benchmark")
    print(f"PATP S=3 (i=2,3,4); PATP-r alpha in [{ALPHA_R[0]}, {ALPHA_R[1]}]; PATP-opt alpha in [0, 1]; "
          f"threshold {THRESH:.0%}; mp.dps={mp.mp.dps}")
    print()
    verify_mellin()
    verify_truth()

    print("\n" + "=" * 100)
    print("PRIMARY CONFIGURATION: Lomax input, quantile design D1, N = 401, q = 0.999")
    print("rel. error per moment j, of the mean and of the std; 'undef' = a required Mellin value is infinite")
    print("n_g = evaluations of g, n_M = distinct Mellin values for j <= 4")
    print("=" * 100)
    prim = run_config(PRIMARY, verbose=True)

    print("\n" + "=" * 100)
    print("SUMMARY TABLES")
    print("=" * 100)
    verdicts = {"PRIMARY": summary("PRIMARY", PRIMARY, prim)}
    for label, cfg in SENSITIVITY:
        verdicts[label] = summary(label, cfg, run_config(cfg, verbose=False))

    print("\n" + "=" * 100)
    print("PRE-REGISTERED VERDICT (response R1, primary configuration)")
    print("=" * 100)
    v = verdicts["PRIMARY"]
    for a in TAILS:
        h1, h2 = v[(a, "R1")]
        res = prim[(a, "R1")][0]
        pr = res["PATP-r"]
        print(f"a = {a}: H1 {'PASS' if h1 else 'FAIL'} (PATP-r mean err {fmt(pr['mean_err'], 0)}, "
              f"std err {fmt(pr['std_err'], 0)}, threshold {THRESH:.0%}; poly3-full std: "
              f"{'undef' if res['poly3-full']['std_err'] is None else fmt(res['poly3-full']['std_err'], 0)}) | "
              f"H2 {'PASS' if h2 else 'FAIL'} (std err poly3-trunc {fmt(res['poly3-trunc']['std_err'], 0)}, "
              f"poly1 {fmt(res['poly1']['std_err'], 0)})")
        print(f"        H3 (report): std err GJ-8 {fmt(res['GJ-8']['std_err'], 0)}, GJ-16 "
              f"{fmt(res['GJ-16']['std_err'], 0)}; mean err GJ-8 {fmt(res['GJ-8']['mean_err'], 0)}, "
              f"GJ-16 {fmt(res['GJ-16']['mean_err'], 0)}")
    ok, passed = r1_passes(v)
    print(f"R1: H1 and H2 hold for a in {ok} -> {len(ok)}/3 -> "
          f"{'ACCURACY CLAIM ALLOWED' if passed else 'ADMISSIBILITY ONLY'}")
    ok2, passed2 = r1_passes(v, "R2")
    print(f"R2 control: H1 and H2 hold for a in {ok2} -> {len(ok2)}/3 -> scope "
          f"{'sub-linear responses in general' if passed2 else 'limited to power-law growth (R1-type)'}"
          f"{'' if passed else ' (moot: no accuracy claim)'}")
    for label, _ in SENSITIVITY:
        oks, ps = r1_passes(verdicts[label])
        flag = "same as primary" if ps == passed else "DIFFERS from primary -> report as limitation"
        print(f"  {label:<24} R1 H1&H2 for a in {oks} ({len(oks)}/3): {flag}")
    posthoc(prim)
    print(f"\nwall time {time.time() - t0:.0f} s")
    return 0


# ======================================================================================
# POST-HOC
# Added after the first run (2026-09-26). Not pre-registered; nothing here feeds H1-H3
# or the decision rule above. Purpose: (P1) an independent check of the truth values,
# (P2) the origin of the largest V3 discrepancy (5.3e-6, a = 3, R1), (P3) where the
# restricted PATP surrogate changes sign beyond the design range, (P4) how PATP-r's
# error grows as the moment order approaches the admissibility edge j P_3 = a.
# ======================================================================================

def lomax_power_moment(a: int, c):
    """E[(1+X)^c] for X ~ Lomax(a, lam = a-1), c < a, via the Euler integral:
    u = X/(lam+X) ~ Beta(1, a) gives a int_0^1 (1 + (lam-1)u)^c (1-u)^(a-1-c) du
    = a/(a-c) 2F1(-c, 1; a-c+1; 1-lam)."""
    A, lam = mp.mpf(a), mp.mpf(a - 1)
    return A / (A - c) * mp.hyp2f1(-c, 1, A - c + 1, 1 - lam)


def lomax_truth_closed(a: int, rkey: str, j: int):
    if rkey == "R1":   # (1+x)^(2/3) - 1 raised to j: binomial in (1+x)^(2m/3)
        return mp.fsum(mp.binomial(j, m) * (-1) ** (j - m) * lomax_power_moment(a, mp.mpf(2 * m) / 3)
                       for m in range(j + 1))
    return mp.diff(lambda c: lomax_power_moment(a, c), 0, j)   # E[ln(1+X)^j]


def quad_x(law, h, x_lo=0.0, x_hi=None, dps=45):
    """Second, independent quadrature: tanh-sinh directly in x with extra breakpoints."""
    with mp.workdps(dps):
        pts = [mp.mpf(x_lo)] + [mp.mpf(float(law.ppf(u))) for u in (0.5, 0.9, 0.99, 0.999, 0.9999)
                                if float(law.ppf(u)) > x_lo and (x_hi is None or float(law.ppf(u)) < x_hi)]
        pts += [mp.mpf(x_hi)] if x_hi is not None else [mp.mpf(10) * pts[-1], mp.inf]
        return mp.quad(lambda z: h(z) * law.pdf_mp(z), pts, maxdegree=10)


def posthoc(prim):
    print("\n" + "=" * 100)
    print("POST-HOC (added after the first run; not pre-registered; does not enter H1-H3)")
    print("=" * 100)

    # P1: truth
    worst_cf, worst_x = 0.0, 0.0
    for a in TAILS:
        law = make_law("lomax", a)
        for rkey in ("R1", "R2"):
            for j in J:
                t = truth(law, rkey, j)
                worst_cf = max(worst_cf, float(abs(t - lomax_truth_closed(a, rkey, j)) / abs(t)))
    wx = {}
    for fam in ("lomax", "loglogistic"):
        for a in TAILS:
            law = make_law(fam, a)
            for rkey in ("R1", "R2"):
                g = RESPONSES[rkey]["mp"]
                for j in J:
                    t = truth(law, rkey, j)
                    d = float(abs(t - quad_x(law, lambda z: g(z) ** j)) / abs(t))
                    grp = (fam, "j<=2" if j <= 2 else "j=3,4")
                    if d >= wx.get(grp, (0.0, None))[0]:
                        wx[grp] = (d, (law.label, rkey, j))
    print(f"P1 truth, Lomax: tanh-sinh (v-variable) vs hypergeometric closed form "
          f"E[(1+X)^c] = a/(a-c) 2F1(-c,1;a-c+1;1-lambda), R1 and R2, j=1..4: max rel diff = {worst_cf:.1e}")
    print("P1 truth, v-variable (30 digits) vs x-variable (45 digits) tanh-sinh, max rel diff:")
    for grp in sorted(wx):
        print(f"   {grp[0]:<12} {grp[1]:<6} {wx[grp][0]:.1e}  at {wx[grp][1]}")
    worst = (0.0, None)
    for fam in ("lomax", "loglogistic"):
        for a in TAILS:
            law = make_law(fam, a)
            x_hi = float(law.ppf(0.999))
            for rkey in ("R1", "R2"):
                for j in J:
                    t = float(truth(law, rkey, j))
                    d = abs(t - truth_scipy(law, rkey, j, x_hi)) / abs(t)
                    if d > worst[0]:
                        worst = (d, (law.label, rkey, j))
    print(f"P1 location of the V2 maximum (mpmath vs scipy, {worst[0]:.1e}): {worst[1]}")

    # P2: V3 discrepancy, a = 3, R1
    law = make_law("lomax", 3)
    res, surr = prim[(3, "R1")]
    print("P2 V3 detail, Lomax a=3, R1: closed form vs quadrature of the same surrogate")
    for k in ("poly1", "PATP-r", "PATP-opt"):
        s = surr[k]
        for j in (1, 2):
            cf = s.moment(j, law.mellin)
            if cf is None:
                continue
            q30 = expect(law, lambda z: s.mp_eval(z) ** j)
            q45 = quad_x(law, lambda z: s.mp_eval(z) ** j)
            top = j * s.growth()
            print(f"   {k:<9} j={j}  tail integrand ~ x^({top:.3f} - {law.a + 1:.0f}):  "
                  f"|cf - quad30|/|cf| = {float(abs(cf - q30) / abs(cf)):.1e},  "
                  f"|cf - quad45|/|cf| = {float(abs(cf - q45) / abs(cf)):.1e}")

    # P3: sign change of the PATP-r surrogate beyond x_hi
    print("P3 PATP-r surrogate beyond the design range (primary): leading coefficient, zero crossing x0,"
          " P(X > x0)")
    for (a, rkey), (res, surr) in prim.items():
        s = surr["PATP-r"]
        law = make_law("lomax", a)
        x_hi = float(law.ppf(0.999))
        lead = s.coefs[int(np.argmax(s.exps))]
        xs = np.geomspace(x_hi, 1e15, 4000)
        vals = s.np_eval(xs)
        idx = np.where(np.sign(vals[1:]) != np.sign(vals[:-1]))[0]
        if len(idx):
            lo, hi = xs[idx[0]], xs[idx[0] + 1]
            for _ in range(200):
                mid = np.sqrt(lo * hi)
                if np.sign(s.np_eval(np.array([mid]))[0]) == np.sign(s.np_eval(np.array([lo]))[0]):
                    lo = mid
                else:
                    hi = mid
            x0 = np.sqrt(lo * hi)
            p0 = (1.0 + x0 / law.lam) ** (-law.a)
            print(f"   a={a} {rkey}: leading coef {lead:+.3f} on x^{max(s.exps):.4f};  x0 = {x0:.4g} "
                  f"= {x0 / x_hi:.3g} x_hi;  P(X > x0) = {p0:.2e}")
        else:
            print(f"   a={a} {rkey}: leading coef {lead:+.3f}; no sign change up to 1e15")

    # P4: error vs distance to the admissibility edge
    print("P4 PATP-r relative error of E[g^j] vs j P_3(alpha*)/a (primary; 1 = admissibility edge)")
    for (a, rkey), (res, surr) in prim.items():
        s = surr["PATP-r"]
        pa = P3(s.meta["alpha"])
        cells = []
        for j in J:
            e = res["PATP-r"]["rel"][j]
            cells.append(f"j={j}: {j * pa / a:5.3f} -> {fmt(e, 0) if e is not None else 'undef'}")
        print(f"   a={a} {rkey}: " + ";  ".join(cells))

    # P5: degree-2 polynomial, full support (not among the pre-registered competitors;
    # admissible for the std only when 4 < a, i.e. a = 6 here)
    print("P5 degree-2 polynomial on the primary design, full input law (admissible for j < a/2):")
    for (a, rkey), (res, surr) in prim.items():
        law = make_law("lomax", a)
        x, x_hi = design(law, PRIMARY)
        s2 = fit_poly(x, RESPONSES[rkey]["np"](x), 2, "poly2")
        tru = {j: truth(law, rkey, j) for j in J}
        e2 = errors({j: s2.moment(j, law.mellin) for j in J}, tru)
        print(f"   a={a} {rkey}: coefs {[float(f'{c:.4g}') for c in s2.coefs]};  mean err {fmt(e2['mean_err'], 0)}, "
              f"std err {fmt(e2['std_err'], 0)}  (PATP-r: {fmt(res['PATP-r']['mean_err'], 0)}, "
              f"{fmt(res['PATP-r']['std_err'], 0)})")


if __name__ == "__main__":
    sys.exit(main())
