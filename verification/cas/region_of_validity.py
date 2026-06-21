"""
Етап 5 — детальний аналіз області чинності PATP-MUET.

Для кожного i ∈ {2, …, 10}:
  • Виводить квадратичну форму p_i(α) = A_i + B_i·α + C_i·α²
  • Дискримінант D_i = B_i² − 4·A_i·C_i (точне раціональне)
  • Координати вершини (α*, p_i(α*))
  • Якщо D_i > 0: дійсні корені α₁ < α₂; знак-від'ємний інтервал на (α₁, α₂) ∩ [0,1]

Для кожної пари (i, k) з k ∈ {1, …, 5}:
  • Чи виконується p_i(α)·k + 1 > 0 на всьому [0,1]?
  • Якщо ні — таблиця α-інтервалів, де порушується (інтегровність біля 0)

Запускається після check_patp.py і пише `region_table.txt` для прямого
включення в Derivation-A2-Region-of-Validity.md.
"""

from __future__ import annotations

import sys
from fractions import Fraction

import sympy as sp


def p_sym(i: int, alpha: sp.Symbol) -> sp.Expr:
    I = sp.Integer(i)
    return sp.Rational(1, i) + (4 - I - sp.Rational(3, i)) * alpha + (2 * I - 4 + sp.Rational(2, i)) * alpha ** 2


def quadratic_coeffs(i: int) -> tuple[sp.Rational, sp.Rational, sp.Rational]:
    A = sp.Rational(1, i)
    B = sp.Rational(4) - sp.Integer(i) - sp.Rational(3, i)
    C = sp.Rational(2 * i) - sp.Rational(4) + sp.Rational(2, i)
    return A, B, C


def analyze_pi() -> str:
    """Таблиця 1: знаковість p_i(α) для i = 2..10."""
    out = []
    out.append("## Таблиця R1 — Квадратична форма pᵢ(α) і дискримінант\n")
    out.append("| i | A | B | C | D = B² − 4AC | Знак D | Вершина α\\* | min pᵢ |")
    out.append("|---|---|---|---|---|---|---|---|")
    alpha = sp.Symbol("alpha", real=True)
    for i in range(2, 11):
        A, B, C = quadratic_coeffs(i)
        D = sp.simplify(B ** 2 - 4 * A * C)
        # вершина квадратичної A + Bα + Cα² (C > 0): α* = −B/(2C)
        alpha_star = sp.simplify(-B / (2 * C))
        p_min = sp.simplify(p_sym(i, alpha).subs(alpha, alpha_star))
        sign_D = "<0" if D < 0 else (">0" if D > 0 else "=0")
        out.append(f"| {i} | {A} | {B} | {C} | {D} | {sign_D} | {alpha_star} | {p_min} |")
    out.append("")
    return "\n".join(out)


def analyze_negative_intervals() -> str:
    """Таблиця 2: для i з D>0 — корені p_i = 0 і від'ємний інтервал на [0,1]."""
    out = []
    out.append("## Таблиця R2 — Інтервали від'ємності pᵢ(α) ∩ [0,1]\n")
    out.append("| i | Корінь α₁ | Корінь α₂ | (α₁, α₂) ∩ [0,1] | pᵢ в мінімумі |")
    out.append("|---|---|---|---|---|")
    alpha = sp.Symbol("alpha", real=True)
    for i in range(6, 11):
        A, B, C = quadratic_coeffs(i)
        D = sp.simplify(B ** 2 - 4 * A * C)
        if D <= 0:
            continue
        sqrtD = sp.sqrt(D)
        alpha1 = sp.simplify((-B - sqrtD) / (2 * C))
        alpha2 = sp.simplify((-B + sqrtD) / (2 * C))
        # вершина
        alpha_star = sp.simplify(-B / (2 * C))
        p_min = sp.simplify(p_sym(i, alpha).subs(alpha, alpha_star))
        # обчислимо чисельно для надійності
        alpha1_f = float(alpha1)
        alpha2_f = float(alpha2)
        lo = max(0.0, alpha1_f)
        hi = min(1.0, alpha2_f)
        interval_str = f"[{lo:.4f}, {hi:.4f}]" if lo < hi else "пусто"
        p_min_f = float(p_min)
        out.append(f"| {i} | {alpha1_f:.4f} | {alpha2_f:.4f} | {interval_str} | {p_min_f:+.6f} |")
    out.append("")
    return "\n".join(out)


def analyze_integrability_uniform() -> str:
    """
    Таблиця 3: для Uniform[0,b] (фундаментальна смуга Re s > 0),
    умова pᵢ(α)·k + 1 > 0 — тобто pᵢ(α) > −1/k.

    Для kᵐᵃˣ ∈ {1..5}: чи виконано на всьому [0,1]?
    """
    out = []
    out.append("## Таблиця R3 — Інтегровність pᵢ(α)·k + 1 > 0 для Uniform[0,b]\n")
    out.append("Фундаментальна смуга M_X: Re s > 0, тобто s = pᵢ(α)·k + 1 > 0 ⇔ pᵢ(α) > −1/k.\n")
    out.append("| i | min pᵢ на [0,1] | Достатня умова | k=1 | k=2 | k=3 | k=4 | k=5 |")
    out.append("|---|---|---|---|---|---|---|---|")
    alpha = sp.Symbol("alpha", real=True)
    for i in range(2, 11):
        A, B, C = quadratic_coeffs(i)
        alpha_star = sp.simplify(-B / (2 * C))
        # обмежити вершину до [0,1] (бо мінімум на [0,1] може бути в кутах)
        alpha_star_f = float(alpha_star)
        candidates_alpha = [0.0, 1.0]
        if 0.0 <= alpha_star_f <= 1.0:
            candidates_alpha.append(alpha_star_f)
        # обчислити p_i на цих точках
        vals = [float(p_sym(i, alpha).subs(alpha, sp.Float(a))) for a in candidates_alpha]
        p_min_01 = min(vals)
        row = [str(i), f"{p_min_01:+.6f}"]
        # достатня умова: p_min > -1/k
        # для k=1..5
        if p_min_01 > 0:
            row.append("безумовно (p>0)")
        else:
            # знайти найменше k, для якого p_min ≤ -1/k, тобто k ≥ -1/p_min (мінімальне таке k)
            min_k_fail = int((-1.0 / p_min_01)) + 1 if p_min_01 < 0 else 999
            row.append(f"k < {-1.0/p_min_01:.3f} (мін. фейл k={min_k_fail})")
        for k in range(1, 6):
            ok = p_min_01 * k + 1.0 > 0
            row.append("✓" if ok else "✗")
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    out.append("**Прочитання.** «✓» означає: для всіх α ∈ [0,1] аргумент Мелліна s = pᵢ(α)k+1 потрапляє у фундаментальну смугу Uniform[0,b].")
    return "\n".join(out)


def main() -> int:
    sections = [
        analyze_pi(),
        analyze_negative_intervals(),
        analyze_integrability_uniform(),
    ]
    text = "\n".join(sections)
    print(text)
    out_path = "region_table.md"
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\n[written] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
