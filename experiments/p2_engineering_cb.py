"""
P2 of the PEM revision (PREM-D-26-00488): an engineering example that can fail.

Both reviewers ask for an engineering / system example (R1-M5b) with an independent
validation and comparisons that control for model complexity (R2). Step 0 showed that
PATP wins at matched capacity only where the response is non-analytic at the edge of the
input support (sqrt-type behaviour at 0) and loses on analytic responses. This script
takes one physical instance of that class, chosen from heat-transfer physics and not
tuned to the method, and compares PATP-MUET against surrogate and quadrature
competitors at an equal number of evaluations of the response.

=========================================================================================
PRE-REGISTERED DESIGN (written before the first run; not to be edited after it)
=========================================================================================

Physical model. Convective heat loss per unit length of a bare cylinder in cross-flow,
the forced-convection term of the overhead-conductor heat balance used for
(dynamic) line rating:
    q_c(V) = pi * k_air * dT * Nu(Re),   Re = V D / nu,
    Nu = 0.3 + 0.62 Re^{1/2} Pr^{1/3} / [1 + (0.4/Pr)^{2/3}]^{1/4}
               * [1 + (Re/282000)^{5/8}]^{4/5}              (Churchill-Bernstein 1977).
Constants: D = 28.14 mm (795 kcmil 26/7 "Drake" ACSR); conductor surface 75 C, ambient
25 C, so dT = 50 K and film temperature T_f = 323.15 K; dry air at 101.325 kPa with
mu and k_air from Sutherland's law (mu_0 = 1.716e-5 Pa s, S = 110.4 K; k_0 = 0.0241
W/(m K), S_k = 194.4 K; reference 273.15 K), rho from the ideal gas law (R = 287.05
J/(kg K)), c_p = 1007 J/(kg K); nu = mu/rho, Pr = mu c_p / k_air. Wind perpendicular to
the conductor. Natural convection (which dominates at wind speeds of a few tenths of
m/s and which IEEE Std 738 handles separately) is not modelled; the response is the
forced-convection correlation over the whole wind range, including its Re^{1/2}
(laminar boundary-layer) behaviour at V -> 0. k_air * dT is a pure scale factor and
cancels in every relative error below.

Input law. Wind speed V ~ Weibull(shape k, scale lam). Baseline climate k = 2,
lam = 6 m/s. Normalised input X = V / v_hi, v_hi = lam * (ln 1e6)^{1/k} (the
1 - 1e-6 quantile). Mellin transform of X over its FULL support [0, inf):
    M_X(s) = (lam/v_hi)^{s-1} Gamma(1 + (s-1)/k),  Re s > 0,
and E[X^a ln^b X] = M_X^{(b)}(a+1) (b-th derivative, computed from the polygamma
functions by the Leibniz recursion on M' = (ln M)' M, 50-digit arithmetic).

Truth. E[q_c(V)^j], j = 1..4, by adaptive tanh-sinh quadrature (mpmath, 30 digits) of
q_c(V)^j against the Weibull density over [0, inf), cross-checked by a second
quadrature in u = (V/lam)^k against e^{-u}.

--- Question 1 (baseline climate, equal number n of evaluations of g = q_c) ---
Budgets n in {6, 8, 12, 16, 24}. Surrogates are fitted by ordinary least squares on the
n Chebyshev-Gauss-Lobatto nodes x_i = (1 - cos(pi i/(n-1)))/2 on [0, 1] (V = 0 and
V = v_hi are nodes), in x = V/v_hi, and propagated over the FULL Weibull support in
closed form (50-digit multinomial sum with the scaled-Weibull M_X; exact for the fitted
surrogate). Primary competitors:
  poly3      degree-3 polynomial, 4 coefficients
  poly4      degree-4 polynomial, 5 coefficients
  PATP-opt   PATP S = 3, exponents {0, p_2(a), p_3(a), p_4(a)}, 4 coefficients + a;
             a chosen on the node residual only: best of the grid a = i/1000,
             i = 0..1000 with the exact degenerate point a = 1/2 removed, then bounded
             Brent refinement on [a_grid - 1e-3, a_grid + 1e-3] (xatol 1e-10); the
             refined value is kept only if its residual is lower.
  sqrt3      fixed basis {1, x^{1/2}, x, x^{3/2}}, 4 coefficients
  conf       confluent basis {1, x^c, x^c ln x, x^c ln^2 x}, 4 coefficients + c;
             c chosen on the node residual: grid c = i/1000, i = 1..3000, then the same
             Brent refinement; moments via derivatives of M_X.
  GL         n-point Gauss quadrature with respect to the Weibull law, realised as
             n-point Gauss-Laguerre in u = (V/lam)^k: E[g^j] ~ sum_i w_i g(lam u_i^{1/k})^j;
             the same n evaluations serve every j.
Secondary competitors (reported, NOT part of H1-H3 or of the decision rule, except
through the qualifiers below):
  GaussV     n-point Gauss rule for the Weibull density in the variable V itself
             (Golub-Welsch from the exact moments lam^m Gamma(1 + m/k), 150 digits).
  GaussT     n-point Gauss rule for the law of T = (V/lam)^{1/2} (moments
             Gamma(1 + m/(2k))), evaluated at V_i = lam t_i^2. This is the quadrature
             that, like sqrt3, is told about the square-root behaviour at V = 0.
  polyI      degree-(n-1) polynomial interpolant at the n nodes (Chebyshev form,
             converted exactly to the monomial basis, moments in 200-digit arithmetic):
             polynomial MUET given the whole design. Question 1 only.

Metrics, per method and n: relative error e_j = |m_hat_j - m_j| / m_j of E[q_c^j],
j = 1..4; their geometric mean geo = exp(mean_j ln e_j) (primary); relative errors of
the mean and of the standard deviation of q_c; for surrogates also the optimised a* or
c*, cond(A) of the n x p design matrix, the RMS residual at the nodes and the RMS and
maximum error on a dense validation grid of 2001 uniform points in x in [0, 1] (W/m),
as the overfitting check; cost = evaluations of g and number of distinct Mellin values
(for j <= 4). Machinery checks: derivative recursion vs mpmath.diff; closed-form
surrogate moment vs quadrature of the same surrogate over the full support (Question 1);
Golub-Welsch exactness on moments 0..2n-1; the two truth quadratures against each other.

--- Question 2 (parametric sweep of climates) ---
20 climates: k in {1.6, 2.0, 2.4, 2.8} x lam in {4, 5, 6, 7, 8} m/s. Surrogates are
fitted ONCE on N Chebyshev-Gauss-Lobatto nodes spanning the union range [0, v_hi^U],
v_hi^U = max over the 20 climates of the 1 - 1e-6 quantile, with X = V / v_hi^U; the
moments for each climate then follow in closed form with zero new evaluations of g
(M_X with lam/v_hi^U). Per-climate quadratures (GL, GaussV, GaussT) need m new
evaluations per climate, 20 m in total. Per method and budget: worst case (max over the
20 climates) and median of geo.
  (a) Equal per-climate n: surrogate with N = n nodes in total vs quadrature with m = n
      per climate (20 n in total), n in {6, 8, 12, 16, 24}.
  (b) Equal TOTAL evaluations N_tot = 20 m: surrogate with N_tot nodes vs quadrature with
      m = N_tot/20 per climate, m in {1, 2, 3, 4, 6, 8, 12, 16, 24}, i.e.
      N_tot in {20, 40, 60, 80, 120, 160, 240, 320, 480}. The budgets m in {6, 8, 12,
      16, 24} (N_tot >= 120) enter H3; m in {1, 2, 3, 4} (the "surrogate with n nodes vs
      quadrature with n/20 per climate" regime) are reported descriptively only.

--- Hypotheses ---
H1  In Question 1, PATP-opt has a lower geo than BOTH poly3 and poly4 at every
    n in {8, 12, 16, 24}. PASS only if all four hold.
H2  In Question 1, PATP-opt vs GL at the same n: no directional prediction; the winner
    and the ratio are reported at every n.
H3  In Question 2 at equal total evaluations, PATP-opt (N_tot = 20 m nodes) has a lower
    worst-case geo than GL (m per climate). Evaluated at m in {6, 8, 12, 16, 24}:
    PASS if it holds at all five, FAIL if at none, MIXED otherwise.

--- Decision rule ---
Let Q1-GL denote "GL beats PATP-opt (lower geo) at every n in {8, 12, 16, 24}".
  If Q1-GL AND H3 = FAIL: the paper must NOT claim a practical advantage of PATP-MUET
  over quadrature when g can be evaluated, and positions PATP-MUET as a propagation
  calculus for a given surrogate.
  Otherwise: the paper states precisely the regime (the n of Question 1 and the budgets
  of Question 2) in which PATP-opt beats GL, and claims nothing outside it.
Qualifiers (pre-registered; they restrict wording, they do not flip the rule):
  Q-a  If GaussV or GaussT beats PATP-opt at every n in {8, 12, 16, 24} in Question 1,
       the positioning must state that a quadrature exists that removes the advantage,
       whatever the rule above allows.
  Q-b  If sqrt3 or conf beats PATP-opt at a majority of n in {8, 12, 16, 24} in
       Question 1, any advantage is attributed to fractional-power (resp. log-power)
       surrogates in general, not to the PATP family.
Figure (pre-registered content): geo vs n in the baseline climate for the six primary
competitors plus GaussV and GaussT; polyI is tabulated only.

Run:  ../verification/cas/.venv/bin/python p2_engineering_cb.py | tee results/p2_engineering.txt
"""

from __future__ import annotations

import sys
import time
from itertools import combinations_with_replacement
from math import comb, factorial

import mpmath as mp
import numpy as np
from numpy.polynomial import chebyshev as npcheb
from numpy.polynomial.laguerre import laggauss
from scipy.optimize import minimize_scalar

from step0_capacity_matched import geo, p_num

mp.mp.dps = 50
J = (1, 2, 3, 4)
NS = (6, 8, 12, 16, 24)
H_NS = (8, 12, 16, 24)
TAIL = 1e-6

# ---------------------------------------------------------------------------
# Physical model
# ---------------------------------------------------------------------------
D_COND = 0.02814           # m, Drake ACSR outside diameter
T_S, T_A = 348.15, 298.15  # K
DT = T_S - T_A             # 50 K
T_F = 0.5 * (T_S + T_A)    # 323.15 K
P_ATM = 101325.0
MU = 1.716e-5 * (T_F / 273.15) ** 1.5 * (273.15 + 110.4) / (T_F + 110.4)
K_AIR = 0.0241 * (T_F / 273.15) ** 1.5 * (273.15 + 194.4) / (T_F + 194.4)
RHO = P_ATM / (287.05 * T_F)
NU = MU / RHO
PR = MU * 1007.0 / K_AIR
CB = 0.62 * PR ** (1.0 / 3.0) / (1.0 + (0.4 / PR) ** (2.0 / 3.0)) ** 0.25


def q_c(V):
    V = np.asarray(V, dtype=float)
    Re = V * D_COND / NU
    Nu = 0.3 + CB * np.sqrt(Re) * (1.0 + (Re / 282000.0) ** 0.625) ** 0.8
    return np.pi * K_AIR * DT * Nu


_mp_c = {}


def q_c_mp(V):
    if not _mp_c:
        _mp_c.update(D=mp.mpf(D_COND), nu=mp.mpf(NU), cb=mp.mpf(CB),
                     scale=mp.pi * mp.mpf(K_AIR) * mp.mpf(DT))
    Re = V * _mp_c["D"] / _mp_c["nu"]
    Nu = mp.mpf("0.3") + _mp_c["cb"] * mp.sqrt(Re) * (1 + (Re / 282000) ** mp.mpf("0.625")) ** mp.mpf("0.8")
    return _mp_c["scale"] * Nu


def v_quantile(k, lam, tail=TAIL):
    return lam * (-np.log(tail)) ** (1.0 / k)


# ---------------------------------------------------------------------------
# Weibull Mellin transform of X = V / vnorm, with derivatives in s
# ---------------------------------------------------------------------------
class WeibullMellin:
    def __init__(self, k, lam, vnorm):
        self.k = mp.mpf(k)
        self.r = mp.mpf(lam) / mp.mpf(vnorm)
        self.lnr = mp.log(self.r)
        self.cache = {}

    def M(self, s):
        return mp.exp((s - 1) * self.lnr + mp.loggamma(1 + (s - 1) / self.k))

    def D(self, s, b):
        """b-th derivative of M_X at s, i.e. E[X^{s-1} ln^b X]."""
        key = (s, b)
        if key in self.cache:
            return self.cache[key]
        z = 1 + (s - 1) / self.k
        L = [None, self.lnr + mp.digamma(z) / self.k]
        for m in range(2, b + 1):
            L.append(mp.polygamma(m - 1, z) / self.k ** m)
        Ms = [self.M(s)]
        for nn in range(1, b + 1):
            Ms.append(mp.fsum(comb(nn - 1, i) * L[i + 1] * Ms[nn - 1 - i] for i in range(nn)))
        self.cache[key] = Ms[b]
        return Ms[b]


# ---------------------------------------------------------------------------
# Surrogates: g_hat(x) = sum_t c_t x^{a_t} ln^{b_t} x
# ---------------------------------------------------------------------------
def basis_matrix(x, terms):
    x = np.asarray(x, dtype=float)
    lx = np.log(np.where(x > 0, x, 1.0))
    cols = []
    for a, b in terms:
        if a == 0 and b == 0:
            cols.append(np.ones_like(x))
        else:
            cols.append(np.where(x > 0, x ** a * lx ** b, 0.0))
    return np.column_stack(cols)


class Surrogate:
    kind = "power"

    def __init__(self, name, terms, coefs, n_params, A, y, shape=None):
        self.name, self.terms, self.coefs, self.n_params = name, list(terms), np.asarray(coefs, float), n_params
        self.shape = shape
        self.cond = float(np.linalg.cond(A))
        self.node_rms = float(np.sqrt(np.mean((A @ self.coefs - y) ** 2)))

    def __call__(self, x):
        return basis_matrix(x, self.terms) @ self.coefs

    def moment(self, j, wm: WeibullMellin):
        n = len(self.terms)
        cm = [mp.mpf(float(c)) for c in self.coefs]
        am = [mp.mpf(float(a)) for a, _ in self.terms]
        bm = [b for _, b in self.terms]
        total = []
        for combo in combinations_with_replacement(range(n), j):
            kap = [combo.count(t) for t in range(n)]
            multi = factorial(j)
            prod, a, b = mp.mpf(1), mp.mpf(0), 0
            for t, kt in enumerate(kap):
                if kt:
                    multi //= factorial(kt)
                    prod *= cm[t] ** kt
                    a += kt * am[t]
                    b += kt * bm[t]
            total.append(multi * prod * wm.D(a + 1, b))
        return mp.fsum(total)

    def n_mellin(self, jmax=4):
        n = len(self.terms)
        args = set()
        for j in range(0, jmax + 1):
            for combo in combinations_with_replacement(range(n), j):
                a = round(sum(self.terms[t][0] for t in combo), 12)
                b = sum(self.terms[t][1] for t in combo)
                args.add((a, b))
        return len(args)


class ChebInterp:
    """Degree-(n-1) interpolant at the n CGL nodes; moments in exact monomial form."""
    kind = "interp"

    def __init__(self, x, y):
        self.name = "polyI"
        self.n_params = len(x)
        self.cheb = npcheb.chebfit(2 * x - 1, y, len(x) - 1)
        V = npcheb.chebvander(2 * x - 1, len(x) - 1)
        self.cond = float(np.linalg.cond(V))
        self.node_rms = float(np.sqrt(np.mean((V @ self.cheb - y) ** 2)))
        self.shape = None
        with mp.workdps(200):
            deg = len(x) - 1
            T = [[mp.mpf(1)], [mp.mpf(-1), mp.mpf(2)]]           # T_m(2x-1) in powers of x
            for m in range(1, deg):
                nxt = [mp.mpf(0)] * (m + 2)
                for i, c in enumerate(T[m]):                    # 2 (2x-1) T_m
                    nxt[i] += -2 * c
                    nxt[i + 1] += 4 * c
                for i, c in enumerate(T[m - 1]):
                    nxt[i] -= c
                T.append(nxt)
            coef = [mp.mpf(0)] * (deg + 1)
            for m in range(deg + 1):
                cm = mp.mpf(float(self.cheb[m]))
                for i, c in enumerate(T[m]):
                    coef[i] += cm * c
            self.mono = coef

    def __call__(self, x):
        return npcheb.chebval(2 * np.asarray(x, float) - 1, self.cheb)

    def moment(self, j, wm: WeibullMellin):
        with mp.workdps(200):
            p = [mp.mpf(1)]
            for _ in range(j):
                q = [mp.mpf(0)] * (len(p) + len(self.mono) - 1)
                for i, a in enumerate(p):
                    for l, b in enumerate(self.mono):
                        q[i + l] += a * b
                p = q
            k, r = mp.mpf(wm.k), mp.mpf(wm.r)
            val = mp.fsum(c * r ** m * mp.gamma(1 + mp.mpf(m) / k) for m, c in enumerate(p))
        return +val

    def n_mellin(self, jmax=4):
        return (self.n_params - 1) * jmax + 1


def fit_fixed(name, x, y, terms, n_params=None, shape=None):
    A = basis_matrix(x, terms)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return Surrogate(name, terms, c, n_params or len(terms), A, y, shape)


def _sse(x, y, terms):
    A = basis_matrix(x, terms)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.sum((A @ c - y) ** 2))


def _grid_then_brent(loss, grid, lo, hi):
    vals = np.array([loss(g) for g in grid])
    best = float(grid[int(np.argmin(vals))])
    fbest = float(vals.min())
    res = minimize_scalar(loss, bounds=(max(lo, best - 1e-3), min(hi, best + 1e-3)),
                          method="bounded", options={"xatol": 1e-10})
    return (float(res.x), float(res.fun)) if res.fun < fbest else (best, fbest)


def patp_terms(a):
    return [(0.0, 0)] + [(p_num(i, a), 0) for i in (2, 3, 4)]


A_GRID = np.delete(np.arange(1001) / 1000.0, 500)
C_GRID = np.arange(1, 3001) / 1000.0


def fit_patp_opt(x, y):
    a, _ = _grid_then_brent(lambda a: _sse(x, y, patp_terms(a)), A_GRID, 0.0, 1.0)
    return fit_fixed("PATP-opt", x, y, patp_terms(a), 5, shape=a)


def conf_terms(c):
    return [(0.0, 0), (c, 0), (c, 1), (c, 2)]


def fit_conf(x, y):
    c, _ = _grid_then_brent(lambda c: _sse(x, y, conf_terms(c)), C_GRID, 1e-3, 3.0)
    return fit_fixed("conf", x, y, conf_terms(c), 5, shape=c)


def cgl(n):
    return (1.0 - np.cos(np.pi * np.arange(n) / (n - 1))) / 2.0


def fit_all_surrogates(n, vnorm, with_interp=False):
    x = cgl(n)
    y = q_c(x * vnorm)
    out = [fit_fixed("poly3", x, y, [(float(d), 0) for d in range(4)]),
           fit_fixed("poly4", x, y, [(float(d), 0) for d in range(5)]),
           fit_patp_opt(x, y),
           fit_fixed("sqrt3", x, y, [(0.0, 0), (0.5, 0), (1.0, 0), (1.5, 0)]),
           fit_conf(x, y)]
    if with_interp:
        out.append(ChebInterp(x, y))
    return out


# ---------------------------------------------------------------------------
# Quadratures
# ---------------------------------------------------------------------------
def gauss_from_moments(mu_fn, n, dps=150):
    """Golub-Welsch: nodes/weights of the n-point Gauss rule from moments mu_0..mu_2n."""
    with mp.workdps(dps):
        mu = [mu_fn(m) for m in range(2 * n + 1)]
        H = mp.matrix(n + 1, n + 1)
        for i in range(n + 1):
            for l in range(n + 1):
                H[i, l] = mu[i + l]
        R = mp.cholesky(H).T
        al = [R[i, i + 1] / R[i, i] - (R[i - 1, i] / R[i - 1, i - 1] if i > 0 else 0) for i in range(n)]
        be = [R[i + 1, i + 1] / R[i, i] for i in range(n - 1)]
        Jm = mp.matrix(n, n)
        for i in range(n):
            Jm[i, i] = al[i]
            if i < n - 1:
                Jm[i, i + 1] = Jm[i + 1, i] = be[i]
        E, Q = mp.eigsy(Jm)
        nodes = [E[i] for i in range(n)]
        w = [mu[0] * Q[0, i] ** 2 for i in range(n)]
        exact = max(abs(mp.fsum(wi * xi ** m for wi, xi in zip(w, nodes)) - mu[m]) / mu[m]
                    for m in range(2 * n))
        return np.array([float(v) for v in nodes]), np.array([float(v) for v in w]), float(exact)


_rule_cache = {}


def quad_rule(kind, n, k):
    """Returns (V/lam nodes, weights, exactness) for the law with Weibull shape k."""
    key = (kind, n, k)
    if key in _rule_cache:
        return _rule_cache[key]
    if kind == "GL":
        u, w = laggauss(n)
        out = (u ** (1.0 / k), w, 0.0)
    elif kind == "GaussV":
        kk = mp.mpf(k)
        y, w, ex = gauss_from_moments(lambda m: mp.gamma(1 + mp.mpf(m) / kk), n)
        out = (y, w, ex)
    elif kind == "GaussT":
        kk = mp.mpf(k)
        t, w, ex = gauss_from_moments(lambda m: mp.gamma(1 + mp.mpf(m) / (2 * kk)), n)
        out = (t ** 2, w, ex)
    else:
        raise ValueError(kind)
    _rule_cache[key] = out
    return out


def quad_moments(kind, n, k, lam):
    ynodes, w, _ = quad_rule(kind, n, k)
    g = q_c(lam * ynodes)
    return [float(np.sum(w * g ** j)) for j in J]


QUADS = ("GL", "GaussV", "GaussT")

# ---------------------------------------------------------------------------
# Truth
# ---------------------------------------------------------------------------
_truth_cache = {}


def truth(k, lam):
    key = (k, lam)
    if key in _truth_cache:
        return _truth_cache[key]
    with mp.workdps(30):
        kk, ll = mp.mpf(k), mp.mpf(lam)

        def pdf(V):
            return kk / ll * (V / ll) ** (kk - 1) * mp.exp(-(V / ll) ** kk)
        pts = [0, ll / 4, ll, 2 * ll, 4 * ll, mp.inf]
        upts = [0, mp.mpf("0.5"), 2, 8, 32, mp.inf]
        tv, tu = [], []
        for j in J:
            tv.append(mp.quad(lambda V: q_c_mp(V) ** j * pdf(V), pts))
            tu.append(mp.quad(lambda u: q_c_mp(ll * u ** (1 / kk)) ** j * mp.exp(-u), upts))
        disc = max(float(abs(a - b) / a) for a, b in zip(tv, tu))
    out = ([mp.mpf(v) for v in tv], disc)
    _truth_cache[key] = out
    return out


def errors(mhat, mtrue):
    e = [float(abs(mp.mpf(a) - b) / b) for a, b in zip(mhat, mtrue)]
    sd_hat = mp.sqrt(mp.mpf(mhat[1]) - mp.mpf(mhat[0]) ** 2) if mp.mpf(mhat[1]) > mp.mpf(mhat[0]) ** 2 else mp.nan
    sd_true = mp.sqrt(mtrue[1] - mtrue[0] ** 2)
    e_sd = float(abs(sd_hat - sd_true) / sd_true) if sd_hat == sd_hat else float("nan")
    return e, geo(e), e_sd


def surrogate_prop_check(s, k, lam, vnorm, moms):
    """|closed form - quadrature of the same surrogate over [0, inf)| / closed form."""
    # Code fix after the first run (design unchanged): that run crashed here with a float
    # overflow of g_hat(x)^j for polyI at n = 24 far in the tail, so the integrand is now
    # evaluated in log space, sign(g)^j exp(j ln|g| + ln f_X).
    from scipy.integrate import quad
    r = lam / vnorm

    def integrand(x, j):
        if x <= 0.0:
            return float(s(np.array([0.0]))[0]) ** j * 0.0
        g = float(s(np.array([x]))[0])
        if g == 0.0:
            return 0.0
        lf = np.log(k / r) + (k - 1) * np.log(x / r) - (x / r) ** k
        with np.errstate(over="ignore"):
            return float(np.sign(g) ** j * np.exp(j * np.log(abs(g)) + lf))
    pts = sorted({r / 4, r, 2 * r, 1.0})
    worst = 0.0
    for j, m in zip(J, moms):
        tot, lo = 0.0, 0.0
        for hi in pts + [np.inf]:
            tot += quad(integrand, lo, hi, args=(j,), limit=500, epsabs=0, epsrel=1e-13)[0]
            lo = hi
        worst = max(worst, abs(tot - float(m)) / abs(float(m)))
    return worst


# ---------------------------------------------------------------------------
# Question 1
# ---------------------------------------------------------------------------
def question1(k=2.0, lam=6.0):
    vhi = v_quantile(k, lam)
    wm = WeibullMellin(k, lam, vhi)
    mt, disc = truth(k, lam)
    mean_t = mt[0]
    sd_t = mp.sqrt(mt[1] - mt[0] ** 2)
    print(f"\nQUESTION 1 -- baseline climate k = {k}, lam = {lam} m/s, v_hi = {vhi:.4f} m/s "
          f"(Re at v_hi = {vhi * D_COND / NU:.0f})")
    print(f"truth: E[q_c] = {mp.nstr(mean_t, 12)} W/m, SD[q_c] = {mp.nstr(sd_t, 12)} W/m, "
          f"E[q_c^j] j=1..4 = {[mp.nstr(v, 12) for v in mt]}")
    print(f"truth cross-check (V-space vs u-space quadrature): max rel diff = {disc:.1e}")
    print(f"q_c(0) = {float(q_c(0.0)):.4f} W/m, q_c(v_hi) = {float(q_c(vhi)):.3f} W/m")

    res = {}
    for n in NS:
        print(f"\n--- n = {n} evaluations of q_c ---")
        hdr = (f"{'method':<9}{'p':>4}{'shape':>8}{'cond(A)':>10}{'nodeRMS':>10}{'valRMS':>10}{'valMax':>10}"
               f"{'mellin':>7}" + "".join(f"{'e_j=' + str(j):>10}" for j in J) + f"{'geo':>10}{'e_sd':>10}{'prop':>9}")
        print(hdr)
        xv = np.linspace(0.0, 1.0, 2001)
        gv = q_c(xv * vhi)
        for s in fit_all_surrogates(n, vhi, with_interp=True):
            moms = [s.moment(j, wm) for j in J]
            e, g, esd = errors(moms, mt)
            err = s(xv) - gv
            prop = surrogate_prop_check(s, k, lam, vhi, moms)
            shape = f"{s.shape:.4f}" if s.shape is not None else "-"
            print(f"{s.name:<9}{s.n_params:>4}{shape:>8}{s.cond:>10.2e}{s.node_rms:>10.2e}"
                  f"{np.sqrt(np.mean(err ** 2)):>10.2e}{np.max(np.abs(err)):>10.2e}{s.n_mellin():>7}"
                  + "".join(f"{v:>10.2e}" for v in e) + f"{g:>10.2e}{esd:>10.2e}{prop:>9.1e}")
            res[(s.name, n)] = dict(e=e, geo=g, esd=esd, shape=s.shape, cond=s.cond)
        for qk in QUADS:
            moms = quad_moments(qk, n, k, lam)
            e, g, esd = errors(moms, mt)
            ex = quad_rule(qk, n, k)[2]
            print(f"{qk:<9}{n:>4}{'-':>8}{'-':>10}{'-':>10}{'-':>10}{'-':>10}{'-':>7}"
                  + "".join(f"{v:>10.2e}" for v in e) + f"{g:>10.2e}{esd:>10.2e}"
                  + (f"{'GW ' + format(ex, '.0e'):>9}" if qk != "GL" else f"{'-':>9}"))
            res[(qk, n)] = dict(e=e, geo=g, esd=esd)
    return res


# ---------------------------------------------------------------------------
# Question 2
# ---------------------------------------------------------------------------
KS = (1.6, 2.0, 2.4, 2.8)
LAMS = (4.0, 5.0, 6.0, 7.0, 8.0)
CLIMATES = [(k, l) for k in KS for l in LAMS]
M_TOT = (1, 2, 3, 4, 6, 8, 12, 16, 24)
SURR_NAMES = ("poly3", "poly4", "PATP-opt", "sqrt3", "conf")


def question2():
    vU = max(v_quantile(k, l) for k, l in CLIMATES)
    print(f"\nQUESTION 2 -- 20 climates, union range v_hi^U = {vU:.4f} m/s "
          f"(Re = {vU * D_COND / NU:.0f})")
    truths = {}
    worst_disc = 0.0
    for c in CLIMATES:
        mt, d = truth(*c)
        truths[c] = mt
        worst_disc = max(worst_disc, d)
    print(f"truth cross-check over 20 climates: max rel diff V-space vs u-space = {worst_disc:.1e}")

    def surr_geos(N):
        out, shapes = {}, {}
        for s in fit_all_surrogates(N, vU):
            shapes[s.name] = s.shape
            out[s.name] = {}
            for c in CLIMATES:
                wm = WeibullMellin(c[0], c[1], vU)
                out[s.name][c] = errors([s.moment(j, wm) for j in J], truths[c])[1]
        return out, shapes

    def quad_geos(m):
        return {qk: {c: errors(quad_moments(qk, m, c[0], c[1]), truths[c])[1] for c in CLIMATES}
                for qk in QUADS}

    def summ(d):
        v = np.array(list(d.values()))
        worst_c = max(d, key=d.get)
        return float(v.max()), float(np.median(v)), worst_c

    qg = {m: quad_geos(m) for m in sorted(set(M_TOT) | set(NS))}
    for m in sorted(qg):
        for qk in ("GaussV", "GaussT"):
            ex = max(quad_rule(qk, m, k)[2] for k in KS)
            if ex > 1e-20:
                print(f"  WARNING Golub-Welsch exactness {qk} m={m}: {ex:.1e}")

    print("\n(a) equal per-climate n: surrogate with N = n nodes in total, quadrature n per climate (20 n total)")
    print(f"{'n':>4}{'method':>10}{'g evals':>9}{'shape':>8}{'worst geo':>11}{'median geo':>12}  worst climate")
    res_a = {}
    for n in NS:
        sg, shp = surr_geos(n)
        for name in SURR_NAMES:
            w, med, wc = summ(sg[name])
            sh = f"{shp[name]:.4f}" if shp[name] is not None else "-"
            print(f"{n:>4}{name:>10}{n:>9}{sh:>8}{w:>11.2e}{med:>12.2e}  k={wc[0]}, lam={wc[1]}")
            res_a[(name, n)] = (w, med)
        for qk in QUADS:
            w, med, wc = summ(qg[n][qk])
            print(f"{n:>4}{qk:>10}{20 * n:>9}{'-':>8}{w:>11.2e}{med:>12.2e}  k={wc[0]}, lam={wc[1]}")
            res_a[(qk, n)] = (w, med)

    print("\n(b) equal TOTAL evaluations N_tot = 20 m: surrogate with N_tot nodes, quadrature m per climate")
    print(f"{'m':>4}{'N_tot':>7}{'method':>10}{'shape':>8}{'worst geo':>11}{'median geo':>12}  worst climate")
    res_b, per_climate = {}, {}
    for m in M_TOT:
        N = 20 * m
        sg, shp = surr_geos(N)
        for name in SURR_NAMES:
            w, med, wc = summ(sg[name])
            sh = f"{shp[name]:.4f}" if shp[name] is not None else "-"
            print(f"{m:>4}{N:>7}{name:>10}{sh:>8}{w:>11.2e}{med:>12.2e}  k={wc[0]}, lam={wc[1]}")
            res_b[(name, m)] = (w, med)
        for qk in QUADS:
            w, med, wc = summ(qg[m][qk])
            print(f"{m:>4}{N:>7}{qk:>10}{'-':>8}{w:>11.2e}{med:>12.2e}  k={wc[0]}, lam={wc[1]}")
            res_b[(qk, m)] = (w, med)
        per_climate[m] = (sg["PATP-opt"], qg[m]["GL"])

    print("\nper-climate geo at equal total evaluations, PATP-opt (N_tot = 20 m nodes) / GL (m per climate)")
    print(f"{'k':>5}{'lam':>5}" + "".join(f"{'m=' + str(m):>20}" for m in (6, 8, 12, 16, 24)))
    for c in CLIMATES:
        print(f"{c[0]:>5}{c[1]:>5}" + "".join(
            f"{per_climate[m][0][c]:>10.1e}{per_climate[m][1][c]:>10.1e}" for m in (6, 8, 12, 16, 24)))
    return res_a, res_b


# ---------------------------------------------------------------------------
# Verdicts and figure
# ---------------------------------------------------------------------------
def verdicts(r1, res_a, res_b):
    print("\n" + "=" * 90)
    print("PRE-REGISTERED VERDICT")
    print("=" * 90)
    h1 = {n: r1[("PATP-opt", n)]["geo"] < min(r1[("poly3", n)]["geo"], r1[("poly4", n)]["geo"]) for n in H_NS}
    print("H1 PATP-opt < poly3 and poly4 (geo), baseline:")
    for n in H_NS:
        print(f"   n={n:>2}: PATP-opt {r1[('PATP-opt', n)]['geo']:.2e}  poly3 {r1[('poly3', n)]['geo']:.2e}  "
              f"poly4 {r1[('poly4', n)]['geo']:.2e}  -> {'yes' if h1[n] else 'no'}")
    print(f"   H1: {'PASS' if all(h1.values()) else 'FAIL'}")

    print("H2 PATP-opt vs GL at equal n (no directional prediction):")
    for n in NS:
        a, b = r1[("PATP-opt", n)]["geo"], r1[("GL", n)]["geo"]
        print(f"   n={n:>2}: PATP-opt {a:.2e}  GL {b:.2e}  winner {'PATP-opt' if a < b else 'GL'}  "
              f"ratio GL/PATP = {b / a:.2e}")
    q1_gl = all(r1[("GL", n)]["geo"] < r1[("PATP-opt", n)]["geo"] for n in H_NS)
    print(f"   Q1-GL (GL beats PATP-opt at every n in {H_NS}): {q1_gl}")

    print("H3 worst-case geo at equal total evaluations, PATP-opt (20 m nodes) vs GL (m per climate):")
    h3 = {}
    for m in (6, 8, 12, 16, 24):
        a, b = res_b[("PATP-opt", m)][0], res_b[("GL", m)][0]
        h3[m] = a < b
        print(f"   m={m:>2} (N_tot={20 * m:>3}): PATP-opt {a:.2e}  GL {b:.2e}  -> {'PATP-opt' if h3[m] else 'GL'}")
    nh3 = sum(h3.values())
    h3v = "PASS" if nh3 == 5 else ("FAIL" if nh3 == 0 else "MIXED")
    print(f"   H3: {h3v} ({nh3}/5)")
    print("   descriptive, m in {1,2,3,4}: " + "; ".join(
        f"m={m}: PATP-opt {res_b[('PATP-opt', m)][0]:.1e} vs GL {res_b[('GL', m)][0]:.1e}" for m in (1, 2, 3, 4)))

    print("DECISION RULE:")
    if q1_gl and h3v == "FAIL":
        print("   Q1-GL and H3 FAIL -> NO practical-advantage claim over quadrature when g can be evaluated;")
        print("   PATP-MUET is positioned as a propagation calculus for a given surrogate.")
    else:
        wins_q1 = [n for n in NS if r1[("PATP-opt", n)]["geo"] < r1[("GL", n)]["geo"]]
        wins_q2 = [m for m in (6, 8, 12, 16, 24) if h3[m]]
        print(f"   advantage regime -> Question 1: PATP-opt beats GL at n = {wins_q1}; "
              f"Question 2 (equal total): at m = {wins_q2}.")
    qa = [qk for qk in ("GaussV", "GaussT") if all(r1[(qk, n)]["geo"] < r1[("PATP-opt", n)]["geo"] for n in H_NS)]
    print(f"   Q-a: secondary quadratures beating PATP-opt at every n in {H_NS}: {qa or 'none'}")
    qb = {nm: sum(r1[(nm, n)]["geo"] < r1[("PATP-opt", n)]["geo"] for n in H_NS) for nm in ("sqrt3", "conf")}
    print(f"   Q-b: number of n in {H_NS} where the basis beats PATP-opt: {qb} "
          f"-> {'triggered' if any(v >= 3 for v in qb.values()) else 'not triggered'}")


def figure(r1, path):
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
                         "axes.linewidth": 0.6, "pdf.fonttype": 42, "legend.fontsize": 7.5})
    style = {
        "PATP-opt": dict(color="black", ls="-", marker="o", mfc="black", lw=1.6, label="PATP-opt ($S=3$, $\\alpha^\\star$)"),
        "poly3": dict(color="0.45", ls="--", marker="s", mfc="white", lw=1.0, label="poly3"),
        "poly4": dict(color="0.45", ls="-.", marker="^", mfc="white", lw=1.0, label="poly4"),
        "sqrt3": dict(color="black", ls=":", marker="D", mfc="white", lw=1.2, label="$\\{1,x^{1/2},x,x^{3/2}\\}$"),
        "conf": dict(color="black", ls="--", marker="v", mfc="0.6", lw=1.0, label="confluent $x^c\\ln^b x$"),
        "GL": dict(color="black", ls="-", marker="x", lw=1.0, label="Gauss–Laguerre in $u$"),
        "GaussV": dict(color="0.6", ls="-", marker="+", lw=0.8, label="Gauss in $V$ (secondary)"),
        "GaussT": dict(color="0.6", ls="--", marker="*", lw=0.8, label="Gauss in $\\sqrt{V}$ (secondary)"),
    }
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for name, st in style.items():
        ax.semilogy(NS, [r1[(name, n)]["geo"] for n in NS], ms=4.5, **st)
    ax.set_xlabel("evaluations of $q_c$, $n$")
    ax.set_ylabel("geometric-mean relative error, $j=1..4$")
    ax.set_xticks(NS)
    ax.grid(True, which="major", color="0.85", lw=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    print(f"\nfigure written: {path}")


def machinery_checks():
    print("MACHINERY CHECKS")
    wm = WeibullMellin(2.0, 6.0, v_quantile(2.0, 6.0))
    s0 = mp.mpf("2.37")
    for b in (1, 3, 6):
        d1 = wm.D(s0, b)
        d2 = mp.diff(wm.M, s0, b)
        print(f"  d^{b}M_X/ds^{b} at s = 2.37: recursion {mp.nstr(d1, 15)}  mpmath.diff {mp.nstr(d2, 15)}  "
              f"rel diff {float(abs(d1 - d2) / abs(d2)):.1e}")
    x = mp.mpf("1.7")
    with mp.workdps(30):
        kk, r = mp.mpf(2), wm.r
        f = lambda t: t ** (x - 1) * mp.log(t) ** 2 * kk / r * (t / r) ** (kk - 1) * mp.exp(-(t / r) ** kk)
        qv = mp.quad(f, [0, r, 4 * r, mp.inf])
    print(f"  E[X^0.7 ln^2 X]: closed form {mp.nstr(wm.D(x, 2), 15)}  quadrature {mp.nstr(qv, 15)}")
    for kind in ("GaussV", "GaussT"):
        worst = max(quad_rule(kind, n, k)[2] for n in NS for k in KS)
        print(f"  Golub-Welsch {kind}: max rel exactness error on moments 0..2n-1 (n in {NS}, k in {KS}) = {worst:.1e}")


# =========================================================================================
# POST-HOC (NOT pre-registered; added after the first complete run had been read).
# Nothing below changes a hypothesis, a verdict or the decision rule printed above.
#   PH1  PATP exponents at a*, distance of a* from the degenerate point 1/2, and the
#        confluent c* against the mean PATP exponent (the step-0 mechanism check).
#   PH2  Question 1 extended to n in {32, 48} for the surrogates and GL, to locate the
#        crossover suggested by the shrinking GL/PATP ratio (1.09 at n = 24).
#   PH3  Descriptive reading of sweep (b) for the secondary quadratures: the smallest
#        per-climate m at which each quadrature's worst case is below PATP-opt's at equal
#        total evaluations.
# =========================================================================================
def posthoc(r1, res_b, k=2.0, lam=6.0):
    print("\n" + "=" * 90)
    print("POST-HOC (not pre-registered)")
    print("=" * 90)
    print("PH1 PATP exponents at a* (baseline climate)")
    print(f"{'n':>4}{'a*':>9}{'|a*-1/2|':>10}{'p2':>8}{'p3':>8}{'p4':>8}{'mean p':>8}{'spread':>8}{'conf c*':>9}")
    for n in NS:
        a = r1[("PATP-opt", n)]["shape"]
        ps = [p_num(i, a) for i in (2, 3, 4)]
        print(f"{n:>4}{a:>9.4f}{abs(a - 0.5):>10.4f}" + "".join(f"{p:>8.4f}" for p in ps)
              + f"{np.mean(ps):>8.4f}{max(ps) - min(ps):>8.4f}{r1[('conf', n)]['shape']:>9.4f}")

    print("\nPH2 Question 1 extended to n in {32, 48} (baseline climate)")
    vhi = v_quantile(k, lam)
    wm = WeibullMellin(k, lam, vhi)
    mt, _ = truth(k, lam)
    print(f"{'n':>4}{'method':>10}{'shape':>8}{'geo':>11}")
    for n in (32, 48):
        for s in fit_all_surrogates(n, vhi):
            g = errors([s.moment(j, wm) for j in J], mt)[1]
            print(f"{n:>4}{s.name:>10}{s.shape if s.shape is None else round(s.shape, 4)!s:>8}{g:>11.2e}")
        g = errors(quad_moments("GL", n, k, lam), mt)[1]
        print(f"{n:>4}{'GL':>10}{'-':>8}{g:>11.2e}")

    print("\nPH3 smallest per-climate m (N_tot = 20 m) at which a quadrature's worst-case geo is below PATP-opt's")
    for qk in QUADS:
        beat = [m for m in M_TOT if res_b[(qk, m)][0] < res_b[("PATP-opt", m)][0]]
        print(f"   {qk:<7}: beats PATP-opt at m = {beat} "
              + "; ".join(f"m={m}: {res_b[(qk, m)][0]:.1e} vs {res_b[('PATP-opt', m)][0]:.1e}" for m in (1, 2, 3, 4)))


def main() -> int:
    t0 = time.time()
    print("P2 engineering example: Churchill-Bernstein convective cooling, Weibull wind")
    print(f"air at T_f = {T_F} K: mu = {MU:.5e} Pa s, rho = {RHO:.5f} kg/m^3, nu = {NU:.5e} m^2/s, "
          f"k_air = {K_AIR:.5f} W/(m K), Pr = {PR:.4f}")
    print(f"D = {D_COND} m, dT = {DT} K, pi k_air dT = {np.pi * K_AIR * DT:.4f} W/m, "
          f"CB prefactor 0.62 Pr^1/3 / [1+(0.4/Pr)^2/3]^1/4 = {CB:.5f}")
    machinery_checks()
    r1 = question1()
    res_a, res_b = question2()
    verdicts(r1, res_a, res_b)
    figure(r1, "../outputs/fig_engineering.pdf")  # supplement: outputs/, not ../paper/ (run from experiments/)
    posthoc(r1, res_b)
    print(f"\nwall time {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
