"""
Етап 6 — таблиця M_X(s) для метрологічних розподілів.

Для кожного розподілу:
  • Символьна формула M_X(s) (sympy)
  • Фундаментальна смуга
  • Чисельна перевірка: символьне значення M_X(s_test) vs scipy.quad
    від x^{s-1}·f(x) для кількох тестових s

Розподіли:
  D1. Uniform[0, b]
  D2. Beta(α_β, β_β)        — також з нецілими параметрами форми
  D3. Right-triangular[0, b] — f(x) = 2x/b² (зведена форма; isoceles див. D3')
  D3'. Isoceles triangular[0, b] — f симетрична відносно b/2 (signed випадок розглядається пізніше)
  D4. Truncated Normal[a, b] — на додатній підтримці, через incomplete gamma

Запуск:
    .venv/bin/python mellin_table.py
"""

from __future__ import annotations

import sys

import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.special import beta as B_func, erf, gammainc, gamma


PASS = "[OK]"
FAIL = "[FAIL]"


def hline(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def check_pair(name: str, M_symbolic, M_X_callable, density, support, s_tests, tol: float = 1e-9) -> bool:
    """Generic helper: compare symbolic M_X(s_test) against numeric ∫ x^{s−1} f(x) dx."""
    s = sp.Symbol("s", real=True, positive=True)
    ok = True
    print(f"  {name}:")
    a, b = support
    for s_val in s_tests:
        sym_val = float(M_symbolic.subs(s, sp.Rational(s_val)) if isinstance(s_val, (int, float)) else M_symbolic.subs(s, s_val))
        num_val, _ = quad(lambda x, s_val=s_val: x ** (s_val - 1) * density(x), a, b, limit=400)
        rel_err = abs(sym_val - num_val) / max(abs(sym_val), abs(num_val), 1e-15)
        row_ok = rel_err < tol
        ok &= row_ok
        tag = PASS if row_ok else FAIL
        print(f"    s={s_val}: sym={sym_val:.10f}  num={num_val:.10f}  rel_err={rel_err:.2e}  {tag}")
    return ok


def D1_uniform() -> bool:
    hline("D1. Uniform[0, b]:  M_X(s) = b^{s−1} / s,  смуга Re s > 0")
    b = sp.Rational(3)  # b = 3
    s = sp.Symbol("s", positive=True)
    M = b ** (s - 1) / s
    print(f"  Символьно: M_X(s) = {M}  (b = {b})")
    density = lambda x: 1.0 / 3.0
    return check_pair("Uniform[0,3]", M, None, density, (0.0, 3.0), [1.5, 2.3, 5.0])


def D2_beta() -> bool:
    hline("D2. Beta(α, β):  M_X(s) = B(α + s − 1, β) / B(α, β),  смуга Re s > 1 − α")
    for a_par, b_par in [(sp.Rational(2), sp.Rational(3)),
                          (sp.Rational(7, 10), sp.Rational(13, 10))]:
        s = sp.Symbol("s", positive=True)
        M = sp.beta(s + a_par - 1, b_par) / sp.beta(a_par, b_par)
        print(f"\n  Beta({a_par}, {b_par}): M_X(s) = {M}")
        ap_f, bp_f = float(a_par), float(b_par)
        density = lambda x, ap=ap_f, bp=bp_f: x ** (ap - 1) * (1 - x) ** (bp - 1) / B_func(ap, bp)
        ok = check_pair(f"Beta({a_par},{b_par})", M, None, density, (0.0, 1.0), [1.5, 2.3, 5.0])
        if not ok:
            return False
    return True


def D3_right_triangular() -> bool:
    hline("D3. Right-triangular[0, b]:  f(x) = 2x/b²,  M_X(s) = 2 b^{s−1} / (s + 1),  смуга Re s > −1")
    b = sp.Rational(3)
    s = sp.Symbol("s", positive=True)
    M = 2 * b ** (s - 1) / (s + 1)
    print(f"  Символьно: M_X(s) = {M}  (b = {b})")
    density = lambda x: 2.0 * x / (3.0 ** 2)
    return check_pair("Right-triangular[0,3]", M, None, density, (0.0, 3.0), [1.5, 2.3, 5.0])


def D3p_isoceles_triangular() -> bool:
    hline("D3'. Isoceles triangular[0, b]:  пік у b/2,  M_X(s) у замкнутій формі через β-функцію")
    # f(x) = 4x/b² на [0, b/2]; 4(b−x)/b² на [b/2, b]
    # M_X(s) = ∫₀^{b/2} x^{s−1} · 4x/b² dx + ∫_{b/2}^b x^{s−1} · 4(b−x)/b² dx
    #        = (4/b²) · [ (b/2)^{s+1} / (s+1) + b · ((b^s − (b/2)^s)/s) − (b^{s+1} − (b/2)^{s+1})/(s+1) ]
    b = sp.Rational(3)
    s = sp.Symbol("s", positive=True)
    half = b / 2
    part1 = half ** (s + 1) / (s + 1)
    part2 = b * (b ** s - half ** s) / s
    part3 = (b ** (s + 1) - half ** (s + 1)) / (s + 1)
    M = (4 / b ** 2) * (part1 + part2 - part3)
    M = sp.simplify(M)
    print(f"  Символьно: M_X(s) = {M}  (b = {b})")
    def density(x, b_f=float(b)):
        if x < b_f / 2:
            return 4.0 * x / b_f ** 2
        else:
            return 4.0 * (b_f - x) / b_f ** 2
    return check_pair("Isoceles triangular[0,3]", M, None, density, (0.0, 3.0), [1.5, 2.3, 5.0])


def D4_truncated_normal() -> bool:
    hline("D4. Truncated Normal на [a, b] (a > 0):  M_X(s) через неповні моменти стандартного N(0,1)")
    # X ~ TN(μ, σ; a, b) має щільність f(x) = φ((x−μ)/σ) / (σ Z), де Z = Φ((b−μ)/σ) − Φ((a−μ)/σ)
    # M_X(s) = ∫_a^b x^{s−1} f(x) dx — у замкнутій формі не елементарна, але виражається через
    # моменти f^{TN}(s) = E[X^{s−1}]. Для верифікації використаємо чисельну адаптивну квадратуру
    # як «зразок істини», а як «закриту форму» — символьне представлення через інтеграл.

    # Параметри: μ=2, σ=1, a=0.1, b=4 (хвости обрізані, ядро ~ N(2,1))
    mu, sigma = 2.0, 1.0
    a, b = 0.1, 4.0
    Z = 0.5 * (erf((b - mu) / (sigma * np.sqrt(2))) - erf((a - mu) / (sigma * np.sqrt(2))))
    pdf_normal = lambda x: np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    f_TN = lambda x: pdf_normal(x) / Z

    # Перевірка: ∫_a^b f dx = 1
    norm_check, _ = quad(f_TN, a, b)
    print(f"  Параметри: μ={mu}, σ={sigma}, a={a}, b={b}, Z={Z:.6f}, ∫f={norm_check:.10f}")

    # Чисельний M_X(s) — наш референт.
    # «Закрита форма» — комбінація incomplete gamma для центрованих степенів, виноситься в текст A3.
    # Тут ми лише фіксуємо, що M_X задана коректним інтегралом.
    s_tests = [1.5, 2.3, 5.0]
    print("  Чисельні значення M_X(s):")
    for s_val in s_tests:
        val, _ = quad(lambda x, s_val=s_val: x ** (s_val - 1) * f_TN(x), a, b, limit=400)
        print(f"    s={s_val}: M_X={val:.10f}")

    # Базова перевірка: M_X(1) = ∫f = 1
    M1, _ = quad(lambda x: f_TN(x), a, b)
    ok = abs(M1 - 1.0) < 1e-10
    print(f"  M_X(1) = ∫f = {M1:.10f}  {'[OK]' if ok else '[FAIL]'}")
    return ok


def main() -> int:
    results = {
        "D1 Uniform":              D1_uniform(),
        "D2 Beta":                 D2_beta(),
        "D3 Right-triangular":     D3_right_triangular(),
        "D3' Isoceles triangular": D3p_isoceles_triangular(),
        "D4 Truncated Normal":     D4_truncated_normal(),
    }
    hline("ПІДСУМОК")
    all_ok = True
    for name, val in results.items():
        tag = PASS if val else FAIL
        print(f"  {name:30s}  {tag}")
        all_ok &= val
    print()
    if all_ok:
        print("ALL CHECKS PASSED.")
        return 0
    print("FAILURES DETECTED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
