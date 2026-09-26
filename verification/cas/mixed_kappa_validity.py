"""
Mixed-multi-index validity of the PATP-MUET closed form (PEM revision, items R1-M3b/c, R2 scope).

Every claim of the revised Section "Region of validity" (paper/main.tex) that carries a
number is checked here, in exact rational arithmetic (fractions / sympy), with a float
grid only as an independent cross-check:

  V1  table tab:validity: A_i, B_i, C_i of p_i(alpha) = C_i + B_i alpha + A_i alpha^2,
      discriminant D_i, vertex alpha_i*, vertex value, and the minimum m_i over [0,1];
  V2  the per-index multiplicity columns kappa_i = 1..5 (condition kappa_i m_i + 1 > 0);
  V3  min_alpha sum_i kappa_i p_i(alpha) >= sum_i kappa_i m_i (the direction of the
      inequality before the proposition) on every mixed pair of the scan;
  V4  the counterexample kappa_10 = 1, kappa_9 = 2: sum of minima, the exact minimum of the
      mixed sum over [0,1], its minimiser, and the alpha-interval where the Mellin argument
      leaves the strip Re s > 0;
  V5  scan over mixed pairs (i1 < i2) in {2..10}, kappa_1, kappa_2 >= 1, |kappa| <= 5:
      where the per-index ceilings pass but the exact mixed condition fails, and where the
      sufficient condition sum kappa_i m_i + 1 > 0 is conservative;
  V6  feasibility check at a fixed alpha*: min over |kappa| <= j of sum kappa_i p_i(alpha*)
      equals j * min(0, min_i p_i(alpha*)), and the max equals j * P_S(alpha*) (brute force);
  V7  exponent facts used by the heavy-tail paragraph and the alpha = 1 remark:
      p_2 >= 1/2 on [0,1]; 1/2 <= P_S(alpha) < 1 on [0,1/2); P_3(1/4) = 11/16;
      p_i(alpha) = 1 only at alpha = 1/2 and alpha = -1/(i-1); p_i(1) = i.

Run:
    verification/cas/.venv/bin/python verification/cas/mixed_kappa_validity.py
Output (also printed): verification/cas/mixed_kappa_validity.txt
"""

from __future__ import annotations

import itertools
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import sympy as sp

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


# ---------------------------------------------------------------- exact exponent map
def coeffs(i: int) -> tuple[F, F, F]:
    """p_i(alpha) = C + B alpha + A alpha^2 (eq. patp-exponent)."""
    C = F(1, i)
    B = F(4 - i) - F(3, i)
    A = F(2 * i - 4) + F(2, i)
    return A, B, C


def p(i: int, a: F) -> F:
    A, B, C = coeffs(i)
    return C + B * a + A * a * a


def quad_min_on_01(A: F, B: F, C: F) -> tuple[F, F]:
    """Exact minimum of C + B a + A a^2 (A > 0) over a in [0,1]; returns (min, argmin)."""
    v = -B / (2 * A)
    cands = [F(0), F(1)]
    if 0 <= v <= 1:
        cands.append(v)
    vals = [(C + B * a + A * a * a, a) for a in cands]
    return min(vals)


def m(i: int) -> F:
    return quad_min_on_01(*coeffs(i))[0]


def mixed_coeffs(idx_kappa: dict[int, int]) -> tuple[F, F, F]:
    A = sum((k * coeffs(i)[0] for i, k in idx_kappa.items()), F(0))
    B = sum((k * coeffs(i)[1] for i, k in idx_kappa.items()), F(0))
    C = sum((k * coeffs(i)[2] for i, k in idx_kappa.items()), F(0))
    return A, B, C


def fr(x: F) -> str:
    return f"{x.numerator}/{x.denominator}" if x.denominator != 1 else f"{x.numerator}"


GRID = np.linspace(0.0, 1.0, 200001)


def p_float(i: int, a):
    A, B, C = (float(c) for c in coeffs(i))
    return C + B * a + A * a * a


def main() -> int:
    out("mixed_kappa_validity.py -- exact validity analysis of the PATP multinomial argument")
    out(f"sympy {sp.__version__}, numpy {np.__version__}, python {sys.version.split()[0]}")

    # ------------------------------------------------------------------ V1
    banner("V1  Table tab:validity: quadratic structure of p_i(alpha), i = 2..10 (exact)")
    paper = {  # values printed in the submitted table: D_i, alpha_i*, min p_i
        2: (F(-7, 4), F(-1, 4), F(7, 16)),
        3: (F(-32, 9), F(0), F(1, 3)),
        4: (F(-63, 16), F(1, 12), F(7, 32)),
        5: (F(-64, 25), F(1, 8), F(1, 10)),
        6: (F(25, 36), F(3, 20), F(-1, 48)),
        7: (F(288, 49), F(1, 6), F(-1, 7)),
        8: (F(833, 64), F(5, 28), F(-17, 64)),
        9: (F(1792, 81), F(3, 16), F(-7, 18)),
        10: (F(3321, 100), F(7, 36), F(-41, 80)),
    }
    out(f"  {'i':>3} {'A_i':>8} {'B_i':>8} {'C_i':>6} {'D_i':>10} {'vertex':>8} {'p_i(vertex)':>12}"
        f" {'m_i=min_[0,1]':>14} {'argmin':>7}")
    for i in range(2, 11):
        A, B, C = coeffs(i)
        D = B * B - 4 * A * C
        v = -B / (2 * A)
        pv = C - B * B / (4 * A)
        mi, am = quad_min_on_01(A, B, C)
        out(f"  {i:>3} {fr(A):>8} {fr(B):>8} {fr(C):>6} {fr(D):>10} {fr(v):>8} {fr(pv):>12}"
            f" {fr(mi):>14} {fr(am):>7}")
        Dp, vp, mp_ = paper[i]
        check(f"i={i}: D_i, vertex and vertex value match the submitted table", (D, v, pv) == (Dp, vp, mp_))
        # grid cross-check of the minimum over [0,1]
        g = float(np.min(p_float(i, GRID)))
        check(f"i={i}: float grid min over [0,1] = {g:+.10f} agrees with exact m_i", abs(g - float(mi)) < 1e-9)
    out()
    out("  NOTE: the submitted column 'min p_i' is the vertex value. For i = 2 the vertex -1/4 lies")
    out("  outside [0,1]; there m_2 = min_[0,1] p_2 = p_2(0) = 1/2. For i >= 3 the vertex lies in [0,1]")
    out("  and m_i equals the vertex value.")
    check("m_2 = 1/2 attained at alpha = 0", quad_min_on_01(*coeffs(2)) == (F(1, 2), F(0)))
    check("m_i = vertex value for i = 3..10", all(m(i) == paper[i][2] for i in range(3, 11)))
    check("m_i > 0 for i = 2..5 and m_i < 0 for i = 6..10",
          all(m(i) > 0 for i in range(2, 6)) and all(m(i) < 0 for i in range(6, 11)))
    out(f"  min over i<=5 of m_i = {fr(min(m(i) for i in range(2, 6)))} (i = 5)")

    # ------------------------------------------------------------------ V2
    banner("V2  Per-index multiplicity columns: kappa_i m_i + 1 > 0, kappa_i = 1..5 (strip Re s > 0)")
    paper_marks = {i: "yyyyy" for i in range(2, 8)}
    paper_marks.update({8: "yyynn", 9: "yynnn", 10: "ynnnn"})
    for i in range(2, 11):
        row = "".join("y" if k * m(i) + 1 > 0 else "n" for k in range(1, 6))
        margins = " ".join(f"{fr(k * m(i) + 1):>8}" for k in range(1, 6))
        out(f"  i={i:>2}: kappa_i m_i + 1 for kappa_i=1..5: {margins}   pass pattern {row}")
        check(f"i={i}: pass pattern equals the submitted table ({paper_marks[i]})", row == paper_marks[i])
    out("  For a single index these columns are exact: min_alpha kappa_i p_i(alpha) = kappa_i m_i.")

    # ------------------------------------------------------------------ V4 (before the scan, it is the headline)
    banner("V4  Counterexample: kappa_10 = 1, kappa_9 = 2 (both single-index rows pass)")
    k10, k9 = 1, 2
    out(f"  row i=10, kappa=1: kappa m_10 + 1 = {fr(k10 * m(10) + 1)} > 0 -> passes")
    out(f"  row i=9,  kappa=2: kappa m_9  + 1 = {fr(k9 * m(9) + 1)} > 0 -> passes")
    check("both single-index rows pass", k10 * m(10) + 1 > 0 and k9 * m(9) + 1 > 0)
    sum_min = k10 * m(10) + k9 * m(9)
    out(f"  sum of per-index minima: 1*(-41/80) + 2*(-7/18) = {fr(sum_min)} = {float(sum_min):+.10f}")
    check("sum of per-index minima = -929/720 < -1", sum_min == F(-929, 720) and sum_min < -1)
    A, B, C = mixed_coeffs({10: k10, 9: k9})
    out(f"  q(alpha) = p_10(alpha) + 2 p_9(alpha) = {fr(C)} + ({fr(B)}) alpha + ({fr(A)}) alpha^2")
    qmin, qarg = quad_min_on_01(A, B, C)
    out(f"  exact min over [0,1] of q: {fr(qmin)} = {float(qmin):+.10f} at alpha = {fr(qarg)} = {float(qarg):.10f}")
    out(f"  per-index minimisers: alpha_9* = 3/16 = {3/16:.6f}, alpha_10* = 7/36 = {7/36:.6f}")
    out(f"  gap (exact min) - (sum of minima) = {fr(qmin - sum_min)} = {float(qmin - sum_min):+.3e}  (>= 0, as it must be)")
    check("exact min >= sum of minima (V3 direction)", qmin >= sum_min)
    check("exact min < -1: the mixed argument leaves Re s > 0", qmin < -1)
    out(f"  lowest Mellin argument q(alpha)+1 = {fr(qmin + 1)} = {float(qmin + 1):+.10f}")
    # interval of alpha where q(alpha) + 1 <= 0
    al = sp.Symbol("alpha", real=True)
    qs = sp.Rational(C.numerator, C.denominator) + sp.Rational(B.numerator, B.denominator) * al \
        + sp.Rational(A.numerator, A.denominator) * al ** 2
    roots = sorted(sp.solve(sp.Eq(qs + 1, 0), al), key=lambda r: float(r))
    out(f"  q(alpha) + 1 = 0 at alpha = {', '.join(str(sp.nsimplify(r)) for r in roots)}")
    out(f"                         ~ {', '.join(f'{float(r):.10f}' for r in roots)}")
    out(f"  so the argument is <= 0 (outside the strip) on [{float(roots[0]):.4f}, {float(roots[1]):.4f}] "
        f"of width {float(roots[1] - roots[0]):.4f}")
    check("both roots lie inside [0,1]", all(0 <= float(r) <= 1 for r in roots))
    g = float(np.min(p_float(10, GRID) + 2 * p_float(9, GRID)))
    check(f"float grid (200001 pts) min of q = {g:+.10f} agrees with the exact value", abs(g - float(qmin)) < 1e-9)
    # divergence for the uniform input: E[X^c] = int_0^1 x^c dx = 1/(c+1) for c > -1, infinite for c <= -1
    c = float(qmin)
    out(f"  uniform input: E[X^c] = int_0^1 x^c dx is finite iff c > -1; here c = {c:+.6f}, so the")
    out("  term kappa = (kappa_9, kappa_10) = (2, 1) of the j >= 3 moment diverges at alpha near 0.19.")
    for eps in (1e-2, 1e-4, 1e-6, 1e-8):
        partial = (1 - eps ** (c + 1)) / (c + 1)
        out(f"    int_eps^1 x^c dx at eps = {eps:.0e}: {partial:.6e}")

    # ------------------------------------------------------------------ V3 + V5 scan
    banner("V5  Scan: mixed pairs i1 < i2 in {2..10}, kappa_1, kappa_2 >= 1, kappa_1 + kappa_2 <= 5")
    out("  per-index = both single-index ceilings pass (kappa_1 m_i1 + 1 > 0 and kappa_2 m_i2 + 1 > 0)")
    out("  suff      = kappa_1 m_i1 + kappa_2 m_i2 + 1 > 0             (sufficient, uniform in alpha)")
    out("  exact     = min_[0,1] (kappa_1 p_i1 + kappa_2 p_i2) + 1 > 0 (exact, uniform in alpha)")
    total = 0
    v3_viol = 0
    suff_not_exact = 0
    perindex_pass_exact_fail = []
    suff_fail_exact_pass = []
    max_gap = (F(0), None)
    small_pairs_ok = True
    for i1, i2 in itertools.combinations(range(2, 11), 2):
        for k1 in range(1, 5):
            for k2 in range(1, 6 - k1):
                total += 1
                A, B, C = mixed_coeffs({i1: k1, i2: k2})
                emin, earg = quad_min_on_01(A, B, C)
                smin = k1 * m(i1) + k2 * m(i2)
                if emin < smin:
                    v3_viol += 1
                gap = emin - smin
                if gap > max_gap[0]:
                    max_gap = (gap, (i1, i2, k1, k2, earg))
                per = (k1 * m(i1) + 1 > 0) and (k2 * m(i2) + 1 > 0)
                suff = smin + 1 > 0
                exact = emin + 1 > 0
                if suff and not exact:
                    suff_not_exact += 1
                if per and not exact:
                    perindex_pass_exact_fail.append((i1, i2, k1, k2, smin, emin, earg))
                if (not suff) and exact:
                    suff_fail_exact_pass.append((i1, i2, k1, k2, smin, emin, earg))
                if i2 <= 5 and not (emin >= F(1, 10) * (k1 + k2)):
                    small_pairs_ok = False
    out(f"  tuples scanned: {total}  (36 pairs x 10 multiplicity splits)")
    check(f"V3: min of the mixed sum >= sum of minima on all {total} tuples (violations: {v3_viol})", v3_viol == 0)
    check(f"sufficient condition never passes where the exact one fails (cases: {suff_not_exact})", suff_not_exact == 0)
    check("i1, i2 <= 5: min of the mixed sum >= |kappa|/10 on every tuple", small_pairs_ok)
    g, t = max_gap
    out(f"  largest gap (exact min - sum of minima): {fr(g)} = {float(g):.4e} at (i1,i2,k1,k2) = {t[:4]}")
    out()
    out(f"  per-index ceilings pass BUT exact mixed condition fails: {len(perindex_pass_exact_fail)} tuples")
    out(f"    {'i1':>3} {'i2':>3} {'k1':>3} {'k2':>3} {'sum of minima':>14} {'exact min':>12} {'at alpha':>9}")
    for i1, i2, k1, k2, smin, emin, earg in perindex_pass_exact_fail:
        out(f"    {i1:>3} {i2:>3} {k1:>3} {k2:>3} {float(smin):>+14.6f} {float(emin):>+12.6f} {float(earg):>9.5f}")
    check("the (i1,i2,k1,k2) = (9,10,2,1) counterexample is in this list",
          any(r[:4] == (9, 10, 2, 1) for r in perindex_pass_exact_fail))
    out()
    out(f"  sufficient condition fails BUT exact condition passes (conservatism): "
        f"{len(suff_fail_exact_pass)} tuples")
    for i1, i2, k1, k2, smin, emin, earg in suff_fail_exact_pass:
        out(f"    {i1:>3} {i2:>3} {k1:>3} {k2:>3} {float(smin):>+14.6f} {float(emin):>+12.6f} {float(earg):>9.5f}")

    # ------------------------------------------------------------------ V6
    banner("V6  Feasibility check at a fixed alpha*: extreme arguments over |kappa| <= j (brute force)")
    alphas = [F(0), F(1, 10), F(3, 20), F(19, 100), F(1, 4), F(9, 20), F(1, 2), F(13, 20), F(1)]
    index_sets = [tuple(range(2, 5)), tuple(range(2, 6)), tuple(range(2, 8)), tuple(range(2, 11)), (6, 9, 10)]
    ok_all = True
    n_cases = 0
    for I in index_sets:
        for a in alphas:
            ps = [p(i, a) for i in I]
            for j in range(1, 6):
                lo = hi = F(0)
                for kap in itertools.product(range(j + 1), repeat=len(I)):
                    if sum(kap) > j:
                        continue
                    val = sum((k * q for k, q in zip(kap, ps)), F(0))
                    lo, hi = min(lo, val), max(hi, val)
                n_cases += 1
                pred_lo = j * min(F(0), min(ps))
                pred_hi = j * max(F(0), max(ps))
                if lo != pred_lo or hi != pred_hi:
                    ok_all = False
                    out(f"    mismatch I={I} alpha={fr(a)} j={j}: lo={lo} pred={pred_lo}, hi={hi} pred={pred_hi}")
    check(f"min_{{|kappa|<=j}} sum kappa_i p_i(a) = j*min(0, min_i p_i(a)) and max = j*max_i p_i(a) "
          f"on {n_cases} (I, alpha, j) cases", ok_all)
    check("max_i p_i(alpha) > 0 always (p_2 > 0), so the upper extreme is j P_S(alpha)",
          all(p(2, a) > 0 for a in alphas))
    out("  So the practical check is two scalar inequalities:")
    out("    1 + j*min(0, min_i p_i(alpha*)) > a   and   1 + j*P_S(alpha*) < b.")
    # worked example: S = 3 fit
    for a in (F(451, 1000), F(520, 1000)):
        ps = {i: p(i, a) for i in (2, 3, 4)}
        out(f"  e.g. S=3, alpha*={float(a):.3f}: p_2,p_3,p_4 = "
            f"{', '.join(f'{float(v):.4f}' for v in ps.values())}; lower edge 1 (all p_i > 0), "
            f"P_3 = {float(max(ps.values())):.4f}")

    # ------------------------------------------------------------------ V7
    banner("V7  Exponent facts for the alpha = 1 remark and the heavy-tail paragraph")
    check("p_i(1) = i for i = 2..10", all(p(i, F(1)) == i for i in range(2, 11)))
    check("p_i(1/2) = 1 for i = 2..10", all(p(i, F(1, 2)) == 1 for i in range(2, 11)))
    for i in range(2, 11):
        A, B, C = coeffs(i)
        a_ = sp.Symbol("a")
        sol = sp.solve(sp.Rational(C.numerator, C.denominator) - 1
                       + sp.Rational(B.numerator, B.denominator) * a_
                       + sp.Rational(A.numerator, A.denominator) * a_ ** 2, a_)
        sol = sorted(sol)
        good = set(sol) == {sp.Rational(1, 2), sp.Rational(-1, i - 1)}
        if not good:
            out(f"    i={i}: roots of p_i = 1: {sol}")
        check(f"i={i}: p_i(alpha) = 1 only at alpha in {{-1/{i-1}, 1/2}} (so in [0,1] only at 1/2)", good)
    out("  At alpha = 1 the PATP span on X >= 0 is {1, x^2, ..., x^(S+1)}; for S = 3: exponents "
        f"{[int(p(i, F(1))) for i in (2, 3, 4)]}, versus {{1, x, x^2, x^3}} for degree-3 MUET.")
    check("p_2(alpha) = 1/2 + alpha/2 + alpha^2 (so p_2 >= 1/2 on [0,1])", coeffs(2) == (F(1), F(1, 2), F(1, 2)))
    half = GRID[GRID < 0.5]
    for S in range(1, 10):
        P = np.max(np.vstack([p_float(i, half) for i in range(2, S + 2)]), axis=0)
        check(f"S={S}: 1/2 <= P_S(alpha) < 1 on the grid over [0, 1/2)  (min {P.min():.6f}, max {P.max():.10f})",
              P.min() >= 0.5 - 1e-15 and P.max() < 1)
    P3q = max(p(i, F(1, 4)) for i in (2, 3, 4))
    out(f"  P_3(1/4) = max(p_2, p_3, p_4)(1/4) = max({fr(p(2, F(1, 4)))}, {fr(p(3, F(1, 4)))}, "
        f"{fr(p(4, F(1, 4)))}) = {fr(P3q)} = {float(P3q)}")
    check("P_3(1/4) = 11/16 = 0.6875", P3q == F(11, 16))
    out(f"  widening factor S / P_S(alpha) at S = 3: alpha=0 -> {3 / 0.5:.1f}; alpha=1/4 -> "
        f"{3 / float(P3q):.4f}; alpha -> 1/2 -> 3")

    banner("SUMMARY")
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
