"""
Умова допустимості на необмеженому носії (Proposition `prop:heavytail`).

Для обмеженого носія зв'язує лише НИЖНЯ межа смуги (`Re s > 0`) — це
покривають `region_of_validity.py` та `multinomial_validity.py`. Для
необмеженого носія зв'язує ВЕРХНЯ межа: якщо `b − 1 = sup{r ≥ 0 : E[X^r] < ∞}`
(момент-індекс хвоста), то найбільший аргумент, який запитує замкнена форма
для `j`-го моменту, дорівнює `j·P_S(α) + 1`, де

    P_S(α) := max_{2 ≤ i ≤ S+1} p_i(α).

Скрипт перевіряє два твердження:

  (1) межі `P_S`: `P_S(0) = 1/2`, `P_S(1/2) = 1`, `P_S(1) = S+1`, і
      `P_S(α) ≤ 1` на всьому `[0, 1/2]` (опуклість `p_i`);
  (2) сама замкнена форма на важкому хвості: для `X ~ Pareto(ν)` на `[1,∞)`
      з `ν = 4` і `S = 3` поліноміальний маршрут при `j = 2` вимагає
      `E[X^6] = ∞`, тоді як PATP-MUET при `α = 0.25` (`j·P₃ = 1.375 < 4`)
      повертає другий момент у замкненій формі — звіряємо з адаптивною
      квадратурою.

Запуск:
    python verification/cas/heavy_tail_admissibility.py
"""

from __future__ import annotations

import itertools
import math
import sys

import numpy as np
from scipy import integrate

TOL_ENDPOINT = 1e-12
TOL_CLOSED_FORM = 1e-9


def p(i: int, a: float) -> float:
    """Показник PATP: p_i(α) = 1/i + (4 − i − 3/i)α + (2i − 4 + 2/i)α²."""
    return 1 / i + (4 - i - 3 / i) * a + (2 * i - 4 + 2 / i) * a**2


def P_S(S: int, a: float) -> float:
    """Верхній показник базису: max_{2 ≤ i ≤ S+1} p_i(α)."""
    return max(p(i, a) for i in range(2, S + 2))


# ---------------------------------------------------------------- (1) межі P_S


def check_bounds(S_values=(3, 4, 5), n_grid: int = 200_001) -> bool:
    ok = True
    grid = np.linspace(0.0, 1.0, n_grid)
    half = grid[grid <= 0.5]
    for S in S_values:
        at0, athalf, at1 = P_S(S, 0.0), P_S(S, 0.5), P_S(S, 1.0)
        sup_half = max(P_S(S, a) for a in half)
        sup_all = max(P_S(S, a) for a in grid)
        rows = [
            ("P_S(0) = 1/2", abs(at0 - 0.5) < TOL_ENDPOINT, at0),
            ("P_S(1/2) = 1", abs(athalf - 1.0) < TOL_ENDPOINT, athalf),
            ("P_S(1) = S+1", abs(at1 - (S + 1)) < TOL_ENDPOINT, at1),
            ("P_S <= 1 on [0,1/2]", sup_half <= 1.0 + TOL_ENDPOINT, sup_half),
            ("sup_[0,1] P_S = S+1", abs(sup_all - (S + 1)) < TOL_ENDPOINT, sup_all),
        ]
        print(f"S = {S}  (i = 2..{S + 1})")
        for name, passed, value in rows:
            print(f"  [{'ok' if passed else 'FAIL'}] {name:<24} value = {value:.10f}")
            ok &= passed
        # опуклість: старший коефіцієнт 2i − 4 + 2/i > 0 для i ≥ 2
        for i in range(2, S + 2):
            lead = 2 * i - 4 + 2 / i
            if lead <= 0:
                print(f"  [FAIL] p_{i} не опукла: старший коеф. = {lead}")
                ok = False
    return ok


# ------------------------------------------------- (2) замкнена форма на Pareto


def mellin_pareto(s: float, nu: float) -> float:
    """M_X(s) = E[X^{s−1}] для Pareto(ν) на [1,∞); ∞ поза смугою."""
    r = s - 1.0
    return math.inf if r >= nu else nu / (nu - r)


def closed_form(k0: float, k: dict[int, float], alpha: float, j: int, nu: float) -> float:
    total = 0.0
    idx = sorted(k)
    for m in range(j + 1):
        inner = 0.0
        for kap in itertools.product(range(m + 1), repeat=len(idx)):
            if sum(kap) != m:
                continue
            coef = math.factorial(m) / math.prod(math.factorial(t) for t in kap)
            prod = math.prod(k[i] ** t for i, t in zip(idx, kap))
            arg = sum(t * p(i, alpha) for i, t in zip(idx, kap)) + 1.0
            inner += coef * prod * mellin_pareto(arg, nu)
        total += math.comb(j, m) * k0 ** (j - m) * inner
    return total


def check_pareto(nu: float = 4.0, S: int = 3, alpha: float = 0.25, j: int = 2) -> bool:
    k0 = 0.3
    k = {2: 1.0, 3: -0.5, 4: 0.25}
    top = P_S(S, alpha)
    print(f"\nPareto(ν = {nu}) на [1,∞), S = {S}, α = {alpha}, j = {j}")
    print(f"  P_S(α)        = {top:.6f}")
    print(f"  j·P_S(α)      = {j * top:.6f}   {'<' if j * top < nu else '>='} b − 1 = {nu}")
    print(f"  поліном j·S   = {j * S}          {'<' if j * S < nu else '>='} b − 1 = {nu}"
          f"   -> E[X^{j * S}] = {mellin_pareto(j * S + 1, nu)}")

    cf = closed_form(k0, k, alpha, j, nu)

    def g(x: float) -> float:
        return k0 + sum(k[i] * x ** p(i, alpha) for i in k)

    quad, _ = integrate.quad(lambda x: g(x) ** j * nu * x ** (-nu - 1), 1.0, np.inf, limit=400)
    rel = abs(cf - quad) / abs(quad)
    passed = rel < TOL_CLOSED_FORM
    print(f"  замкнена форма = {cf:.12f}")
    print(f"  квадратура     = {quad:.12f}")
    print(f"  [{'ok' if passed else 'FAIL'}] відносна різниця = {rel:.3e}")
    return passed


def main() -> int:
    print("=== (1) межі верхнього показника P_S ===")
    ok = check_bounds()
    print("\n=== (2) замкнена форма на необмеженому носії ===")
    ok &= check_pareto()
    print("\nПІДСУМОК:", "усі перевірки пройдено" if ok else "Є ПОМИЛКИ")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
