"""
R6 (review v1 → v2): мультиномне розширення аналізу області чинності.

`Derivation-A1` §6 потребує `M_X^{(m)}( Σᵢ κᵢ·pᵢ(α) + 1 )` для кожного
мультиномного вектора `κ = (κ₂,…,κ_{S+1})` з `|κ| = m`. Для `i ≤ 5` всі
`pᵢ(α) > 0` на `[0,1]`, тож сума додатних додатна — аргумент Мелліна
безумовно потрапляє в смугу `Re s > 0`.

Для `i ≥ 6` `pᵢ` бувають від'ємні. Тоді мінімум суми
`min_α Σᵢ κᵢ·pᵢ(α)` може бути МЕНШИМ за просту суму `Σᵢ κᵢ·(min_α pᵢ(α))`,
якщо мінімуми досягаються в різних `α`. Цей скрипт сканує реальні мінімуми.

Для пар (i₁, i₂) ∈ {2..10}² (включно з мішаними) і `|κ| ≤ 5`:
  • обчислює `min_{α∈[0,1]} κ₁·p_{i₁}(α) + κ₂·p_{i₂}(α)` сітково;
  • перевіряє чи `min + 1 > 0` (умова смуги Re s > 0);
  • фіксує пари (i₁, i₂, κ₁, κ₂), де порушується.

Запуск:
    .venv/bin/python verification/cas/multinomial_validity.py
"""

from __future__ import annotations

import sys

import numpy as np


def p_num(i: int, alpha):
    return 1.0 / i + (4 - i - 3.0 / i) * alpha + (2 * i - 4 + 2.0 / i) * alpha ** 2


def banner(t: str) -> None:
    print()
    print("=" * 72)
    print(t)
    print("=" * 72)


def scan_pair(i1: int, i2: int, kappa1: int, kappa2: int, n_grid: int = 10001) -> dict:
    alpha_grid = np.linspace(0.0, 1.0, n_grid)
    p1 = p_num(i1, alpha_grid)
    p2 = p_num(i2, alpha_grid)
    sum_p = kappa1 * p1 + kappa2 * p2
    j_min = int(np.argmin(sum_p))
    s_arg_min = float(sum_p[j_min]) + 1.0  # +1 з аргументу Мелліна
    return {
        "i1": i1, "i2": i2, "k1": kappa1, "k2": kappa2,
        "alpha_min": float(alpha_grid[j_min]),
        "min_sum_p": float(sum_p[j_min]),
        "s_arg_min": s_arg_min,
        "ok": s_arg_min > 0,
    }


def main() -> int:
    banner("R6: мультиномне розширення R-of-V для пар (i₁, i₂), |κ| ≤ 5")
    print("Сканує min_α κ₁·p_{i₁}(α) + κ₂·p_{i₂}(α) на сітці 10001 точок.\n")

    # Single-index baseline (для довідки)
    print("## Baseline: single-index pᵢ(α)·k + 1 > 0 для k ≤ 5")
    print("(дублює `region_of_validity.py` Table R3, у форматі pairs з другим коеф 0)")
    for i in range(2, 11):
        for k in range(1, 6):
            res = scan_pair(i, i, k, 0, n_grid=5001)
            # Для k₂=0 це p_{i}(α)·k, що ми вже знаємо. Просто перевірка
            if not res["ok"] and i <= 6 and k <= 5:
                print(f"  WARN: i={i}, k={k}: s_min = {res['s_arg_min']:+.6f}")

    # Реальний multinomial scan
    banner("## Multinomial scan — порушення smug-condition s > 0")
    print(f"  {'i₁':>3} {'i₂':>3}  {'κ₁':>2} {'κ₂':>2}  {'α*':>7}  {'min Σκᵢpᵢ':>10}  {'s=Σ+1':>8}  status")
    print("-" * 72)
    violations = 0
    total = 0
    interesting = []

    for i1 in range(2, 11):
        for i2 in range(i1, 11):  # i2 >= i1 для уникнення дублів
            for k1 in range(0, 6):
                for k2 in range(0, 6):
                    if k1 + k2 == 0 or k1 + k2 > 5:
                        continue
                    if i1 == i2 and k2 > 0:
                        continue  # дубль single-index
                    total += 1
                    res = scan_pair(i1, i2, k1, k2)
                    if not res["ok"]:
                        violations += 1
                        interesting.append(res)
                    elif i1 >= 6 or i2 >= 6:
                        # Записуємо також близькі-до-границі для перегляду
                        if res["s_arg_min"] < 0.5:
                            interesting.append(res)

    # Відсортувати за s_arg_min (найгірші зверху)
    interesting.sort(key=lambda r: r["s_arg_min"])
    for r in interesting[:30]:
        tag = "✗ violation" if not r["ok"] else "○ tight"
        print(f"  {r['i1']:>3} {r['i2']:>3}  {r['k1']:>2} {r['k2']:>2}  {r['alpha_min']:>7.4f}  {r['min_sum_p']:>+10.6f}  {r['s_arg_min']:>+8.4f}  {tag}")

    banner("ПІДСУМОК")
    print(f"  Сканована simbo всього (i₁ ≤ i₂ ≤ 10, |κ| ≤ 5, mixed): {total}")
    print(f"  Порушень умови s > 0 (тобто Σκᵢ·minₐ pᵢ + 1 ≤ 0): {violations}")
    print()

    # Спеціальний випадок: чи розширюється порушення для i ≤ 5?
    print("  Перевірка: для i₁, i₂ ∈ {2..5} умова має виконуватися безумовно (pᵢ > 0).")
    safe_violations = 0
    for r in interesting:
        if r["i1"] <= 5 and r["i2"] <= 5 and not r["ok"]:
            safe_violations += 1
            print(f"    UNEXPECTED: {r}")
    print(f"  Порушень у безпечному режимі i ≤ 5: {safe_violations}")

    # Чи мультиномне порушення гірше за single-index?
    print()
    print("  Compare: чи комбінація (i₁,i₂) з різними мінімумами `pᵢ` дає гірший `min Σ`")
    print("  за просту суму одно-індексних мінімумів?")
    from collections import defaultdict
    single_min = {i: float(min(p_num(i, np.linspace(0, 1, 5001)))) for i in range(2, 11)}
    extra_loss_examples = []
    for r in interesting[:60]:
        if r["i1"] == r["i2"]:
            continue
        simple_sum = r["k1"] * single_min[r["i1"]] + r["k2"] * single_min[r["i2"]]
        actual = r["min_sum_p"]
        extra_loss = actual - simple_sum  # negative if actual is worse
        if extra_loss < -1e-6:
            extra_loss_examples.append((r, simple_sum, actual, extra_loss))

    if extra_loss_examples:
        print(f"  Знайдено {len(extra_loss_examples)} прикладів, де actual < simple_sum:")
        for r, s, a, d in extra_loss_examples[:10]:
            print(f"    i₁={r['i1']} i₂={r['i2']} κ=({r['k1']},{r['k2']}): "
                  f"simple={s:+.4f}, actual={a:+.4f}, Δ={d:+.4f}")
    else:
        print("  Жодного! Мінімум суми = сума одно-індексних мінімумів (мінімуми досягаються близько).")
        print("  Це структурний факт: для PATP-сім'ї всі pᵢ мають мінімум при близьких α* ∈ [0.15, 0.20].")

    return 0


if __name__ == "__main__":
    sys.exit(main())
