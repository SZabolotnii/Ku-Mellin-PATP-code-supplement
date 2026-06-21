"""
CAS cross-check for Derivation-A1-Mellin-PATP.md.

Етап 1 (PLAN.md): чотири перевірки кандидат-виводу перед тим, як витрачати
час на Lean-формалізацію.

  C1. T1 — кутові тотожності PATP-експоненти pᵢ(α) для i ∈ {2,…,10}.
  C2. T2 — знаковість pᵢ(α) на [0,1]: i∈{2..5} додатна; i=6 має від'ємний
       проміжок з мінімумом ≈ −0.021 (Derivation-A1 §9).
  C3. T3 — правило Мелліна для степеня M_{X^c}(s) = M_X(c(s−1)+1) на
       Uniform[0,1] і Beta(2,3); порівняння символьне.
  C4. §6 — повна формула моментної пропагації на Uniform[0,1] vs Monte
       Carlo для двох α (0 — дробовий базис; ½ — лінійна вироджена форма).

Запуск:
    source .venv/bin/activate
    python check_patp.py
"""

from __future__ import annotations

import sys
from fractions import Fraction
from math import factorial

import numpy as np
import sympy as sp


PASS = "[OK]"
FAIL = "[FAIL]"


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# PATP exponent
# ---------------------------------------------------------------------------

def p_sym(i: int, alpha: sp.Symbol) -> sp.Expr:
    """Symbolic PATP exponent pᵢ(α) = 1/i + (4−i−3/i)·α + (2i−4+2/i)·α²."""
    I = sp.Integer(i)
    return sp.Rational(1, i) + (4 - I - sp.Rational(3, i)) * alpha + (2 * I - 4 + sp.Rational(2, i)) * alpha ** 2


def p_num(i: int, alpha: float) -> float:
    return 1.0 / i + (4 - i - 3.0 / i) * alpha + (2 * i - 4 + 2.0 / i) * alpha ** 2


# ---------------------------------------------------------------------------
# C1. T1 corners
# ---------------------------------------------------------------------------

def check_T1_corners() -> bool:
    banner("C1. T1 — pᵢ(0)=1/i, pᵢ(½)=1, pᵢ(1)=i  (i = 2..10)")
    alpha = sp.Symbol("alpha", positive=True)
    ok = True
    for i in range(2, 11):
        expr = p_sym(i, alpha)
        v0 = sp.simplify(expr.subs(alpha, 0))
        v_half = sp.simplify(expr.subs(alpha, sp.Rational(1, 2)))
        v1 = sp.simplify(expr.subs(alpha, 1))
        exp0 = sp.Rational(1, i)
        exp_half = sp.Integer(1)
        exp1 = sp.Integer(i)
        row_ok = (v0 == exp0) and (v_half == exp_half) and (v1 == exp1)
        ok &= row_ok
        tag = PASS if row_ok else FAIL
        print(f"  i={i:2d}: p(0)={v0}  p(½)={v_half}  p(1)={v1}  {tag}")
    return ok


# ---------------------------------------------------------------------------
# C2. T2 sign analysis
# ---------------------------------------------------------------------------

def check_T2_signs() -> bool:
    banner("C2. T2 — знаковість pᵢ(α) на [0,1]")
    alpha = sp.Symbol("alpha", real=True)
    ok = True

    # (a) Discriminant test: A=1/i, B=4−i−3/i, C=2i−4+2/i; D = B² − 4AC
    print("  Дискримінант D_i = B² − 4AC:")
    for i in range(2, 11):
        A = sp.Rational(1, i)
        B = sp.Rational(4) - sp.Integer(i) - sp.Rational(3, i)
        C = sp.Rational(2 * i) - sp.Rational(4) + sp.Rational(2, i)
        D = sp.simplify(B ** 2 - 4 * A * C)
        sign_D = sp.sign(D)
        sign_str = "<0" if sign_D == -1 else (">0" if sign_D == 1 else "=0")
        expected = "<0" if i in {2, 3, 4, 5} else ">0"
        row_ok = sign_str == expected
        ok &= row_ok
        tag = PASS if row_ok else FAIL
        print(f"    i={i:2d}: D = {D}  ({sign_str}, очік. {expected})  {tag}")

    # (b) Numerical grid: i=2..5 → all positive; i=6 → min on [0,1]
    print("\n  Сітковий тест pᵢ(α) на 1001 точці α ∈ [0,1]:")
    grid = np.linspace(0.0, 1.0, 1001)
    for i in range(2, 7):
        vals = np.array([p_num(i, a) for a in grid])
        vmin = vals.min()
        vmax = vals.max()
        if i in {2, 3, 4, 5}:
            row_ok = vmin > 0
            tag = PASS if row_ok else FAIL
            print(f"    i={i}: min={vmin:+.6f}, max={vmax:+.6f}  (очік. min>0)  {tag}")
            ok &= row_ok
        else:  # i == 6 — від'ємний проміжок
            row_ok = vmin < 0 and abs(vmin - (-0.021)) < 0.01
            tag = PASS if row_ok else FAIL
            alpha_min = grid[vals.argmin()]
            # знайти інтервал від'ємності
            neg_mask = vals < 0
            if neg_mask.any():
                neg_alphas = grid[neg_mask]
                a_lo, a_hi = neg_alphas[0], neg_alphas[-1]
                print(f"    i={i}: min={vmin:+.6f} @ α={alpha_min:.3f}; pᵢ<0 на α∈[{a_lo:.3f},{a_hi:.3f}]  (очік. min≈−0.021)  {tag}")
            ok &= row_ok

    return ok


# ---------------------------------------------------------------------------
# C3. T3 power rule
# ---------------------------------------------------------------------------

def check_T3_power_rule() -> bool:
    """
    T3: M_{X^c}(s) = M_X(c(s−1)+1) for c > 0, X ≥ 0.

    Перевірка повністю чисельна (scipy.integrate.quad), щоб уникнути
    важкого символьного інтегрування з дробовою підстановкою. Це чесно:
    скорочення Якобіана `(1/c)·c = 1` — алгебраїчне, перевіряється тривіально;
    нетривіальне твердження — що інтеграли рівні чисельно, що ми й тестуємо.
    """
    from scipy.integrate import quad
    from scipy.special import beta as B_func

    banner("C3. T3 — M_{X^c}(s) = M_X(c(s−1)+1)  (numerical)")

    c = 7.0 / 5.0  # дробове c

    distros = [
        # (name, density f_X(x), support, closed-form M_X(s))
        ("Uniform[0,1]",
         lambda x: 1.0,
         (0.0, 1.0),
         lambda s: 1.0 / s),
        ("Beta(2,3)",
         lambda x: x * (1 - x) ** 2 / B_func(2.0, 3.0),
         (0.0, 1.0),
         lambda s: B_func(s + 1.0, 3.0) / B_func(2.0, 3.0)),
        ("Beta(0.7,1.3)",
         lambda x: x ** (0.7 - 1) * (1 - x) ** (1.3 - 1) / B_func(0.7, 1.3),
         (0.0, 1.0),
         lambda s: B_func(s - 1.0 + 0.7, 1.3) / B_func(0.7, 1.3)),
    ]

    s_test = [1.5, 2.3, 5.0]
    ok = True

    for name, f_X, (a, b), M_X in distros:
        print(f"\n  {name}, c = {c}:")
        # density f_Y for Y = X^c:  f_Y(y) = f_X(y^{1/c}) · (1/c) · y^{1/c − 1}
        def f_Y(y, f_X=f_X, c=c):
            if y <= 0:
                return 0.0
            return f_X(y ** (1.0 / c)) * (1.0 / c) * y ** (1.0 / c - 1.0)

        a_Y, b_Y = a ** c, b ** c

        for s in s_test:
            # LHS: M_Y(s) via direct numeric integration
            lhs, _ = quad(lambda y, s=s: y ** (s - 1) * f_Y(y), a_Y, b_Y, limit=200)
            # RHS: rule M_X(c(s-1)+1)
            rhs = M_X(c * (s - 1.0) + 1.0)
            err = abs(lhs - rhs)
            rel = err / max(abs(lhs), abs(rhs), 1e-15)
            row_ok = rel < 1e-8
            ok &= row_ok
            tag = PASS if row_ok else FAIL
            print(f"    s={s}: LHS={lhs:.10f}  RHS={rhs:.10f}  rel_err={rel:.2e}  {tag}")

    return ok


# ---------------------------------------------------------------------------
# C4. §6 full formula vs Monte Carlo (Uniform[0,1], n=3)
# ---------------------------------------------------------------------------

def patp_moment_uniform01(k0: float, ks: dict[int, float], alpha: float, j: int, *, N_mc: int = 2_000_000, seed: int = 0) -> tuple[float, float]:
    """
    Обчислює j-й момент g(X;α) = k0 + Σᵢ kᵢ · sgn(X)|X|^{pᵢ(α)} двома способами:
    (формула §6 — закритий вираз через M_X) і Monte Carlo. Повертає (formula, mc).

    X ~ Uniform[0,1]; M_X(s) = 1/s; sgn ≡ 1 на додатній підтримці, тож
    парність-селекція колапсує: M_X^{(m)} = M_X.
    """

    def M_X(s: float) -> float:
        # для Uniform[0,1]: ∫₀¹ x^{s−1} dx = 1/s, s>0
        return 1.0 / s

    indices = sorted(ks.keys())  # i = 2..S+1
    pvals = {i: p_num(i, alpha) for i in indices}

    # Mультиномне розкладання Σ_{|κ|=m} (m!/Πκᵢ!) Πᵢ kᵢ^{κᵢ} M_X( Σκᵢpᵢ + 1 )
    from itertools import product

    def expand_kappa(m: int):
        # перебір невід'ємних цілих векторів довжини len(indices) з сумою m
        n = len(indices)
        for combo in product(range(m + 1), repeat=n):
            if sum(combo) == m:
                yield combo

    moment = 0.0
    for m in range(j + 1):
        binom = factorial(j) // (factorial(m) * factorial(j - m))
        outer = binom * k0 ** (j - m)
        inner = 0.0
        for kappa in expand_kappa(m):
            coef_multi = factorial(m)
            prod_k = 1.0
            s_arg = 1.0  # +1
            denom_mult = 1
            for idx_pos, ki in enumerate(indices):
                ke = kappa[idx_pos]
                if ke == 0:
                    continue
                coef_multi //= factorial(ke)
                prod_k *= ks[ki] ** ke
                s_arg += ke * pvals[ki]
            # for ke=0 multinomial factor factorial(0)=1, нічого не змінює
            inner += coef_multi * prod_k * M_X(s_arg)
        moment += outer * inner

    # Monte Carlo
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=N_mc)
    g = k0 + sum(ks[i] * np.sign(X) * np.abs(X) ** pvals[i] for i in indices)
    mc_moment = float(np.mean(g ** j))
    return moment, mc_moment


def check_C4_full_formula() -> bool:
    banner("C4. §6 — повна формула моментної пропагації vs Monte Carlo (Uniform[0,1])")
    ks = {2: 0.7, 3: -0.4, 4: 0.3}  # n=3: i ∈ {2,3,4}
    k0 = 0.5
    scenarios = [
        ("α=0   (дробовий базис, pᵢ=1/i)", 0.0),
        ("α=¼   (проміжний α)", 0.25),
        ("α=½   (вироджена лінійна форма, pᵢ=1)", 0.5),
        ("α=1   (цілостеповий, pᵢ=i)", 1.0),
    ]
    ok = True
    print(f"  k0={k0}, k={ks}")
    for name, alpha in scenarios:
        print(f"\n  {name}")
        for j in (1, 2, 3, 4):
            formula, mc = patp_moment_uniform01(k0, ks, alpha, j)
            # MC noise: для N=2e6 std≈O(σ/√N); толеранс 5e-3 відносної
            denom = max(abs(formula), abs(mc), 1e-12)
            rel_err = abs(formula - mc) / denom
            row_ok = rel_err < 5e-3
            tag = PASS if row_ok else FAIL
            print(f"    j={j}: formula={formula:+.6f}  mc={mc:+.6f}  rel_err={rel_err:.2e}  {tag}")
            ok &= row_ok
    return ok


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    results = {
        "C1 T1 corners": check_T1_corners(),
        "C2 T2 sign":    check_T2_signs(),
        "C3 T3 power":   check_T3_power_rule(),
        "C4 §6 vs MC":   check_C4_full_formula(),
    }
    banner("ПІДСУМОК")
    all_ok = True
    for name, val in results.items():
        tag = PASS if val else FAIL
        print(f"  {name:20s}  {tag}")
        all_ok &= val
    print()
    if all_ok:
        print("ALL CHECKS PASSED — Derivation-A1 кандидат-вивід узгоджений з CAS+MC.")
        return 0
    print("FAILURES DETECTED — повернутися до Derivation-A1, не йти в Lean.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
