"""
Mellin transform of the truncated normal (PEM revision, item R1-M4; replaces the unbacked
"< 1e-10" sentence of the submitted Appendix A).

X ~ N(mu, sigma^2) truncated to [a, b], 0 <= a < b <= inf, density
    f(x) = phi((x - mu)/sigma) / (sigma Z),   Z = Phi((b - mu)/sigma) - Phi((a - mu)/sigma),
    M_X(s) = E[X^(s-1)] = int_a^b x^(s-1) f(x) dx.

Claims checked (each is quoted in the revised Appendix A of paper/main.tex):

  T1  mu = 0: with u = x^2/(2 sigma^2),
          M_X(s) = sigma^(s-1) 2^((s-2)/2) / (sqrt(2 pi) Z)
                   * [ gamma(s/2, b^2/(2 sigma^2)) - gamma(s/2, a^2/(2 sigma^2)) ],
      gamma = lower incomplete gamma; strip Re s > 0 if a = 0, entire in s if a > 0.
      Checked against 30-digit adaptive (tanh-sinh) quadrature, closed form also at 30 digits.
  T2  the same closed form in double precision (scipy.special.gammainc * gamma) against the
      30-digit reference -- what a practitioner's implementation achieves.
  T3  signed support, mu = 0, a < 0 < b: the bilateral pair of Lemma lem:bilateral,
          M_|X|(s)     = K(s) [ gamma(s/2, a^2/(2 sigma^2)) + gamma(s/2, b^2/(2 sigma^2)) ],
          M_sgn.X(s)   = K(s) [ gamma(s/2, b^2/(2 sigma^2)) - gamma(s/2, a^2/(2 sigma^2)) ],
      K(s) = sigma^(s-1) 2^((s-2)/2) / (sqrt(2 pi) Z), against 30-digit quadrature.
  T4  mu != 0 on a finite [a, b]: no elementary closed form is claimed; M_X is evaluated by
      double-precision adaptive quadrature (scipy.integrate.quad, QUADPACK; x^(s-1) passed as
      an algebraic endpoint weight when a = 0) and compared with an independent 30-digit mpmath
      tanh-sinh quadrature. That reference is itself cross-checked against a method with no
      quadrature at all: expanding exp(mu x / sigma^2) gives the absolutely convergent series
          M_X(s) = exp(-mu^2/(2 sigma^2)) / (sigma Z sqrt(2 pi)) * sum_n (mu/sigma^2)^n / n! * I0(s+n),
      I0(r) = sigma^r 2^((r-2)/2) [gamma(r/2, b^2/(2 sigma^2)) - gamma(r/2, a^2/(2 sigma^2))],
      i.e. an infinite (not finite) incomplete-gamma expansion.
  T5  mu != 0 on [0, inf): parabolic-cylinder form from DLMF 12.5.1 with D_nu(z) = U(-nu-1/2, z)
      (DLMF 12.1):
          int_0^inf x^(s-1) exp(-(x-mu)^2/(2 sigma^2)) dx = sigma^s Gamma(s) exp(-mu^2/(4 sigma^2)) D_(-s)(-mu/sigma),
      so M_X(s) = sigma^(s-1) Gamma(s) exp(-mu^2/(4 sigma^2)) D_(-s)(-mu/sigma) / (sqrt(2 pi) Phi(mu/sigma)),
      Re s > 0; checked against 30-digit quadrature on [0, inf).
  T6  why translation fails: for integer order n = s-1 the binomial theorem reduces
      E[X^n] to finitely many truncated moments of the standardised variable (checked for
      mu != 0 against quadrature); for non-integer s-1 the binomial series of
      (mu + sigma z)^(s-1) is infinite, so no finite incomplete-gamma combination follows.
  T7  companion check for the other Table tab:mellin-distros row changed in the revision: the
      isosceles triangle on [0, b], M_X(s) = 2 b^(s-1) (2 - 2^(1-s)) / (s (s+1)), agrees with
      quadrature for -1 < s <= 0 as well, and its singularity at s = 0 is removable
      (M_X(0) = 4 ln 2 / b), so the strip is Re s > -1.

Run:
    verification/cas/.venv/bin/python verification/cas/truncnormal_mellin.py
Output (also printed): verification/cas/truncnormal_mellin.txt
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy
from scipy import integrate, special

mp.mp.dps = 30
OUT = Path(__file__).with_suffix(".txt")
LINES: list[str] = []
FAILS: list[str] = []


def out(s: str = "") -> None:
    LINES.append(s)
    print(s)


def check(name: str, cond: bool) -> None:
    out(f"  [{'OK' if cond else 'FAIL'}] {name}")
    if not cond:
        FAILS.append(name)


def banner(t: str) -> None:
    out()
    out("=" * 78)
    out(t)
    out("=" * 78)


# ------------------------------------------------------------------ 30-digit reference
def Z_mp(mu, sigma, a, b):
    lo = mp.ncdf((mp.mpf(a) - mu) / sigma)
    hi = mp.mpf(1) if b == mp.inf else mp.ncdf((mp.mpf(b) - mu) / sigma)
    return hi - lo


def mellin_quad_mp(s, mu, sigma, a, b, signed=False):
    """int_a^b |x|^(s-1) [sgn x] f(x) dx at 30 digits (tanh-sinh), breakpoints at 0 and mu.

    On a segment that ends at the origin the factor |x|^(s-1) is singular for s < 1; there the
    substitution |x| = t^(1/s) (dx = t^(1/s-1)/s dt) removes it exactly:
        int_0^c x^(s-1) w(x) dx = (1/s) int_0^(c^s) w(t^(1/s)) dt.
    """
    s, mu, sigma = mp.mpf(s), mp.mpf(mu), mp.mpf(sigma)
    Z = Z_mp(mu, sigma, a, b)

    def w(x):
        return mp.exp(-((x - mu) / sigma) ** 2 / 2) / (sigma * mp.sqrt(2 * mp.pi) * Z)

    lo = mp.mpf(a)
    hi = mp.inf if b == mp.inf else mp.mpf(b)
    cuts = sorted({lo, hi} | {c for c in (mp.mpf(0), mu) if lo < c < hi})
    total = mp.mpf(0)
    for l, r in zip(cuts[:-1], cuts[1:]):
        sign = (mp.mpf(-1) if r <= 0 else mp.mpf(1)) if signed else mp.mpf(1)
        if l == 0:            # [0, r]: x = t^(1/s)
            val = mp.quad(lambda t: w(t ** (1 / s)), [0, r ** s if r != mp.inf else mp.inf]) / s
        elif r == 0:          # [l, 0]: x = -t^(1/s)
            val = mp.quad(lambda t: w(-(t ** (1 / s))), [0, abs(l) ** s]) / s
        else:
            val = mp.quad(lambda x: abs(x) ** (s - 1) * w(x), [l, r])
        total += sign * val
    return total


def series_mp(s, mu, sigma, a, b, tol=mp.mpf(10) ** -34):
    """Independent evaluation for mu != 0, 0 <= a < b < inf: expand exp(mu x / sigma^2),
        M_X(s) = exp(-mu^2/(2 sigma^2)) / (sigma Z sqrt(2 pi))
                 * sum_n (mu/sigma^2)^n / n! * I0(s + n),
        I0(r) = int_a^b x^(r-1) exp(-x^2/(2 sigma^2)) dx
              = sigma^r 2^((r-2)/2) [gamma(r/2, b^2/(2 sigma^2)) - gamma(r/2, a^2/(2 sigma^2))].
    """
    s, mu, sigma = mp.mpf(s), mp.mpf(mu), mp.mpf(sigma)
    Z = Z_mp(mu, sigma, a, b)
    A2, B2 = mp.mpf(a) ** 2 / (2 * sigma ** 2), mp.mpf(b) ** 2 / (2 * sigma ** 2)

    def I0(r):
        return sigma ** r * mp.power(2, (r - 2) / 2) * mp.gammainc(r / 2, A2, B2)

    acc, n, c = mp.mpf(0), 0, mp.mpf(1)
    while True:
        term = c * I0(s + n)
        acc += term
        if n > 10 and abs(term) < tol * abs(acc):
            break
        n += 1
        c *= (mu / sigma ** 2) / n
    return mp.exp(-mu ** 2 / (2 * sigma ** 2)) * acc / (sigma * Z * mp.sqrt(2 * mp.pi)), n


def K_mp(s, sigma, Z):
    return sigma ** (s - 1) * mp.power(2, (s - 2) / 2) / (mp.sqrt(2 * mp.pi) * Z)


def lower_gamma_mp(z, x):
    return mp.gammainc(z, 0, x)  # int_0^x t^(z-1) e^(-t) dt


def closed_mu0_mp(s, sigma, a, b):
    s, sigma = mp.mpf(s), mp.mpf(sigma)
    Z = Z_mp(0, sigma, a, b)
    ga = lower_gamma_mp(s / 2, mp.mpf(a) ** 2 / (2 * sigma ** 2))
    gb = lower_gamma_mp(s / 2, mp.mpf(b) ** 2 / (2 * sigma ** 2))
    return K_mp(s, sigma, Z) * (gb - ga)


def closed_mu0_signed_mp(s, sigma, a, b):
    """a < 0 < b: (M_|X|, M_sgn.X)."""
    s, sigma = mp.mpf(s), mp.mpf(sigma)
    Z = Z_mp(0, sigma, a, b)
    ga = lower_gamma_mp(s / 2, mp.mpf(a) ** 2 / (2 * sigma ** 2))
    gb = lower_gamma_mp(s / 2, mp.mpf(b) ** 2 / (2 * sigma ** 2))
    K = K_mp(s, sigma, Z)
    return K * (ga + gb), K * (gb - ga)


def pcf_halfline_mp(s, mu, sigma):
    """M_X(s) for N(mu, sigma^2) truncated to [0, inf), via D_(-s) (DLMF 12.5.1, 12.1)."""
    s, mu, sigma = mp.mpf(s), mp.mpf(mu), mp.mpf(sigma)
    Z = mp.ncdf(mu / sigma)
    D = mp.pcfd(-s, -mu / sigma)
    return sigma ** (s - 1) * mp.gamma(s) * mp.exp(-mu ** 2 / (4 * sigma ** 2)) * D / (mp.sqrt(2 * mp.pi) * Z)


# ------------------------------------------------------------------ double precision
def closed_mu0_f64(s, sigma, a, b):
    Z = special.ndtr(b / sigma) - special.ndtr(a / sigma)
    G = special.gamma(s / 2)
    ga = special.gammainc(s / 2, a * a / (2 * sigma * sigma)) * G
    gb = special.gammainc(s / 2, b * b / (2 * sigma * sigma)) * G
    K = sigma ** (s - 1) * 2 ** ((s - 2) / 2) / (math.sqrt(2 * math.pi) * Z)
    return K * (gb - ga)


def mellin_quad_f64(s, mu, sigma, a, b):
    """Double-precision QUADPACK. For a = 0 the factor x^(s-1) is passed as an algebraic
    endpoint weight (QAWS, weight='alg'), the standard treatment of x^(s-1) with s < 1."""
    Z = special.ndtr((b - mu) / sigma) - special.ndtr((a - mu) / sigma)

    def w(x):
        return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi) * Z)

    if a == 0:
        val, err = integrate.quad(w, 0.0, b, weight="alg", wvar=(s - 1, 0.0), epsabs=0.0, epsrel=1e-13, limit=500)
        return val, err
    pts = [mu] if a < mu < b else None
    val, err = integrate.quad(lambda x: x ** (s - 1) * w(x), a, b, points=pts, epsabs=0.0, epsrel=1e-13, limit=500)
    return val, err


def rel(x, y) -> float:
    x, y = mp.mpf(x), mp.mpf(y)
    return float(abs(x - y) / abs(y))


def main() -> int:
    out("truncnormal_mellin.py -- Mellin transform of the truncated normal")
    out(f"mpmath {mp.__version__} (mp.dps = {mp.mp.dps}), scipy {scipy.__version__}, "
        f"numpy {np.__version__}, python {sys.version.split()[0]}")

    S_GRID = [0.25, 0.5, 1.0, 1.37, 2.0, 2.9, 4.5, 7.3]

    # ------------------------------------------------------------------ T1 / T2
    banner("T1/T2  mu = 0, 0 <= a < b: lower-incomplete-gamma closed form")
    cases_mu0 = [(1.0, 0.0, 2.0), (0.5, 0.0, 1.0), (2.0, 0.3, 5.0), (1.0, 1.0, 3.0)]
    worst_t1 = 0.0
    worst_t2 = 0.0
    n_t1 = 0
    out(f"  {'sigma':>5} {'a':>4} {'b':>4} {'s':>5} {'closed form (30 dig.)':>34} {'rel vs quad30':>13} {'rel f64 vs ref':>14}")
    for sigma, a, b in cases_mu0:
        for s in S_GRID:
            ref = mellin_quad_mp(s, 0, sigma, a, b)
            cf = closed_mu0_mp(s, sigma, a, b)
            e1 = rel(cf, ref)
            e2 = rel(closed_mu0_f64(s, sigma, a, b), ref)
            worst_t1, worst_t2 = max(worst_t1, e1), max(worst_t2, e2)
            n_t1 += 1
            out(f"  {sigma:>5} {a:>4} {b:>4} {s:>5} {mp.nstr(cf, 28):>34} {e1:>13.2e} {e2:>14.2e}")
    out(f"  T1: {n_t1} points, max relative difference closed form vs 30-digit quadrature = {worst_t1:.2e}")
    out(f"  T2: max relative error of the double-precision closed form vs 30-digit reference = {worst_t2:.2e}")
    check("T1: closed form = quadrature to < 1e-25 on every point", worst_t1 < 1e-25)
    check("T2: double-precision closed form accurate to < 1e-13", worst_t2 < 1e-13)
    # normalisation M_X(1) = 1 is an identity of the formula
    norm = max(abs(closed_mu0_mp(1, sg, a, b) - 1) for sg, a, b in cases_mu0)
    check(f"T1: closed form at s = 1 equals 1 (max |M_X(1) - 1| = {mp.nstr(norm, 3)})", norm < mp.mpf(10) ** -25)
    # the factor: at a = 0, b = inf the formula must give the half-normal moment 2^((s-1)/2) Gamma(s/2)/sqrt(pi)
    hn = max(rel(closed_mu0_mp(s, 1, 0, mp.inf), mp.power(2, (mp.mpf(s) - 1) / 2) * mp.gamma(mp.mpf(s) / 2) / mp.sqrt(mp.pi))
             for s in S_GRID)
    check(f"T1: a = 0, b = inf, sigma = 1 reproduces E|N(0,1)|^(s-1) = 2^((s-1)/2) Gamma(s/2)/sqrt(pi) (max rel {hn:.1e})",
          hn < 1e-25)

    # ------------------------------------------------------------------ T3
    banner("T3  mu = 0, signed support a < 0 < b: bilateral pair (M_|X|, M_sgn.X)")
    worst_t3 = 0.0
    n_t3 = 0
    for sigma, a, b in [(1.0, -1.5, 2.0), (0.7, -0.4, 1.2), (2.0, -3.0, 1.0)]:
        for s in S_GRID:
            m_abs, m_sgn = closed_mu0_signed_mp(s, sigma, a, b)
            r_abs = mellin_quad_mp(s, 0, sigma, a, b, signed=False)
            r_sgn = mellin_quad_mp(s, 0, sigma, a, b, signed=True)
            e = max(rel(m_abs, r_abs), rel(m_sgn, r_sgn))
            worst_t3 = max(worst_t3, e)
            n_t3 += 1
        out(f"  sigma={sigma}, [a,b]=[{a},{b}]: s=1.37 -> M_|X| = {mp.nstr(closed_mu0_signed_mp(1.37, sigma, a, b)[0], 20)}, "
            f"M_sgn.X = {mp.nstr(closed_mu0_signed_mp(1.37, sigma, a, b)[1], 20)}")
    out(f"  T3: {n_t3} points x 2 transforms, max relative difference vs 30-digit quadrature = {worst_t3:.2e}")
    check("T3: bilateral closed forms = quadrature to < 1e-25", worst_t3 < 1e-25)

    # ------------------------------------------------------------------ T4
    banner("T4  mu != 0, finite [a, b]: double-precision adaptive quadrature vs two 30-digit references")
    out("  reference 1: 30-digit tanh-sinh quadrature (singular endpoint removed by x = t^(1/s));")
    out("  reference 2: the incomplete-gamma series of series_mp (exp(mu x/sigma^2) expanded), no quadrature.")
    cases = [(0.5, 1.0, 0.0, 3.0), (2.0, 1.0, 0.1, 4.0), (-1.0, 1.5, 0.0, 2.5),
             (2.0, 0.5, 1.0, 6.0), (0.8, 0.3, 0.0, 1.0), (3.0, 1.2, 0.5, 5.0)]
    s_t4 = [0.25, 0.5, 1.37, 2.5, 4.5, 7.3]
    worst_t4 = 0.0
    worst_ref = 0.0
    n_t4 = 0
    max_terms = 0
    out(f"  {'mu':>4} {'sigma':>5} {'a':>4} {'b':>4} {'s':>5} {'M_X(s) (30 dig.)':>32} {'rel f64 quad':>13} {'quad vs series':>14}")
    for mu, sigma, a, b in cases:
        for s in s_t4:
            ref = mellin_quad_mp(s, mu, sigma, a, b)
            ref2, nterms = series_mp(s, mu, sigma, a, b)
            max_terms = max(max_terms, nterms)
            val, _ = mellin_quad_f64(s, mu, sigma, a, b)
            e = rel(val, ref)
            eref = rel(ref2, ref)
            worst_t4, worst_ref = max(worst_t4, e), max(worst_ref, eref)
            n_t4 += 1
            out(f"  {mu:>4} {sigma:>5} {a:>4} {b:>4} {s:>5} {mp.nstr(ref, 26):>32} {e:>13.2e} {eref:>14.1e}")
    out(f"  T4: {n_t4} points, max relative error of double-precision quad vs 30-digit reference = {worst_t4:.2e}")
    out(f"      the two 30-digit references (quadrature, incomplete-gamma series) agree to {worst_ref:.1e};")
    out(f"      the series needed at most {max_terms} terms")
    check("T4: the two 30-digit references agree to < 1e-25", worst_ref < 1e-25)
    check("T4: double-precision adaptive quadrature accurate to < 1e-12", worst_t4 < 1e-12)

    # ------------------------------------------------------------------ T5
    banner("T5  mu != 0 on [0, inf): parabolic-cylinder closed form (DLMF 12.5.1 with D_nu = U(-nu-1/2, .))")
    worst_t5 = 0.0
    n_t5 = 0
    for mu, sigma in [(-1.0, 0.7), (0.5, 1.3), (2.0, 1.0), (0.0, 1.0)]:
        for s in [0.5, 1.0, 1.37, 2.5, 4.5]:
            cf = pcf_halfline_mp(s, mu, sigma)
            ref = mellin_quad_mp(s, mu, sigma, 0, mp.inf)
            e = rel(cf, ref)
            worst_t5 = max(worst_t5, e)
            n_t5 += 1
        out(f"  mu={mu:>4}, sigma={sigma}: M_X(1.37) = {mp.nstr(pcf_halfline_mp(1.37, mu, sigma), 22)}")
    out(f"  T5: {n_t5} points, max relative difference vs 30-digit quadrature on [0, inf) = {worst_t5:.2e}")
    check("T5: parabolic-cylinder form = quadrature to < 1e-25", worst_t5 < 1e-25)
    m1 = max(abs(pcf_halfline_mp(1, mu, sg) - 1) for mu, sg in [(-1.0, 0.7), (0.5, 1.3), (2.0, 1.0)])
    check(f"T5: the form gives M_X(1) = 1 (max |M_X(1)-1| = {mp.nstr(m1, 3)})", m1 < mp.mpf(10) ** -25)

    # ------------------------------------------------------------------ T6
    banner("T6  Translation works for integer orders only (binomial theorem), mu != 0")
    worst_t6 = 0.0
    for mu, sigma, a, b in [(2.0, 1.0, 0.1, 4.0), (0.5, 1.0, 0.0, 3.0)]:
        al, be = (mp.mpf(a) - mu) / sigma, (mp.mpf(b) - mu) / sigma
        Z = mp.ncdf(be) - mp.ncdf(al)

        def trunc_moment_std(k):  # int_al^be z^k phi(z) dz / Z, split at 0 into half-line incomplete gammas
            def half(x):  # int_0^|x| t^k phi(t) dt = 2^((k-1)/2) gamma((k+1)/2, x^2/2) / sqrt(2 pi)
                return mp.power(2, (mp.mpf(k) - 1) / 2) * lower_gamma_mp((mp.mpf(k) + 1) / 2, x * x / 2) / mp.sqrt(2 * mp.pi)
            if al >= 0:
                v = half(be) - half(al)
            elif be <= 0:
                v = (-1) ** k * (half(al) - half(be))
            else:
                v = half(be) + (-1) ** k * half(al)
            return v / Z

        for n in range(0, 5):
            binom = sum(mp.binomial(n, k) * mp.mpf(mu) ** (n - k) * mp.mpf(sigma) ** k * trunc_moment_std(k)
                        for k in range(n + 1))
            ref = mellin_quad_mp(n + 1, mu, sigma, a, b)
            e = rel(binom, ref)
            worst_t6 = max(worst_t6, e)
        out(f"  mu={mu}, sigma={sigma}, [a,b]=[{a},{b}]: E[X^n], n=0..4, via binomial + incomplete gamma")
    out(f"  T6: max relative difference vs 30-digit quadrature = {worst_t6:.2e}")
    check("T6: integer orders reduce to finitely many incomplete-gamma terms (< 1e-25)", worst_t6 < 1e-25)
    out("  For non-integer s-1 the binomial series of (mu + sigma z)^(s-1) does not terminate, and it converges")
    out("  only where |sigma z| < |mu|, i.e. on part of the support; no finite incomplete-gamma form follows.")

    # ------------------------------------------------------------------ T7
    banner("T7  Same table, isosceles-triangular row: strip Re s > -1 (pole at s = 0 cancels)")
    bb = mp.mpf(3)

    def iso_closed(s):
        s = mp.mpf(s)
        return 2 * bb ** (s - 1) * (2 - mp.power(2, 1 - s)) / (s * (s + 1))

    def iso_quad(s):
        s = mp.mpf(s)
        # f = 4x/b^2 on [0, b/2]: int_0^(b/2) x^(s-1) 4x/b^2 dx in closed form is what we test,
        # so integrate numerically with the substitution x = t^(1/(s+1)) on the singular piece
        left = mp.quad(lambda t: 4 / bb ** 2, [0, (bb / 2) ** (s + 1)]) / (s + 1)
        right = mp.quad(lambda x: x ** (s - 1) * 4 * (bb - x) / bb ** 2, [bb / 2, bb])
        return left + right

    worst_t7 = 0.0
    for s in [-0.9, -0.5, -0.2, 0.3, 1.5, 2.3]:
        e = rel(iso_closed(s), iso_quad(s))
        worst_t7 = max(worst_t7, e)
        out(f"  b=3, s={s:>5}: closed form {mp.nstr(iso_closed(s), 22):>26}   rel vs quadrature {e:.1e}")
    lim0 = mp.limit(iso_closed, 0)
    q0 = iso_quad(mp.mpf(0))
    out(f"  s -> 0: limit of the closed form = {mp.nstr(lim0, 22)}; 4 ln 2 / b = {mp.nstr(4 * mp.log(2) / bb, 22)}; "
        f"quadrature at s = 0: {mp.nstr(q0, 22)}")
    check(f"T7: closed form = quadrature on s in (-1, 3), including -1 < s <= 0 (max rel {worst_t7:.1e})", worst_t7 < 1e-25)
    check("T7: the singularity at s = 0 is removable, M_X(0) = 4 ln 2 / b",
          abs(lim0 - 4 * mp.log(2) / bb) < mp.mpf(10) ** -20 and abs(q0 - 4 * mp.log(2) / bb) < mp.mpf(10) ** -25)

    banner("SUMMARY (numbers quoted in the revised Appendix A)")
    out(f"  T1 mu=0 closed form vs 30-digit quadrature: {n_t1} points, max rel diff {worst_t1:.1e}")
    out(f"  T2 mu=0 closed form in double precision:     max rel err {worst_t2:.1e}")
    out(f"  T3 mu=0 signed bilateral pair:               {n_t3} points, max rel diff {worst_t3:.1e}")
    out(f"  T4 mu!=0 double-precision quad vs 30 digits: {n_t4} points, max rel err {worst_t4:.1e}")
    out(f"  T5 mu!=0 [0,inf) parabolic-cylinder form:    {n_t5} points, max rel diff {worst_t5:.1e}")
    out(f"  T6 integer orders via binomial theorem:      max rel diff {worst_t6:.1e}")
    out(f"  T7 isosceles triangle, -1 < s <= 0 included: max rel diff {worst_t7:.1e}")
    if FAILS:
        out(f"  {len(FAILS)} FAILED checks:")
        for f_ in FAILS:
            out(f"    - {f_}")
    else:
        out("  ALL CHECKS PASSED.")
    OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
