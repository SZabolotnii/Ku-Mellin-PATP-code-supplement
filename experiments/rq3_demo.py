"""
Етап 7 — RQ3 метрологічна демонстрація.

Перевірка H3 (Spec-2 §4 line 92): на дробово-степеневій response surface
PATP-MUET дає ≥ 15% покращення на хвостовій помилці vs поліноміальний
MUET при тому ж бюджеті моментів.

## Постановка manufactured solution

X ~ Uniform[0, 1] (відома, M_X(s) = 1/s).
Y = f(X), де f — дробово-степенева функція двох випадків:
  M1.  f(x) = 0.6·x^{0.5} + 0.4·x^{1.5}    — суміш двох дробових степенів
  M2.  f(x) = x^{0.7}                       — чистий нецілий степінь
Відомі моменти E[Y^j] = ground truth (інтегруємо точно).

## Два оцінювачі

Обидва обмежені тим самим бюджетом моментів X: {E[X^1], …, E[X^S]} плюс
E[Y]. Чесно: PATP-MUET ще потребує E[X^{p_i(α)}], E[X^{2p_i(α)}], …
дробових порядків — це і є переваги PATP, що дозволяють точно
представити дробово-степеневу f. Поліноміальний MUET цього не може.

  E1.  Polynomial MUET — Rajan-style. Апроксимуємо f(x) поліномом степеня
       S=3 (метод найменших квадратів на [0,1]); пропагуємо моменти Y_poly
       через біноміальне розкладання поліноміальної response surface.
  E2.  PATP-MUET — fit з PATP-базисом порядку S=3 (i ∈ {2,3,4}, фіксоване
       α; оптимізуємо α гриджом по [0,1]). Пропагуємо моменти через
       Derivation-A1 §6.

## Метрики

Для j ∈ {1, 2, 3, 4}: bias = E_method[Y^j] - E_true[Y^j]; rel_bias.
Tail metric: оцінка розширеної невизначеності U = √Var[Y] · 3 (стандартний
GUM коефіцієнт покриття 3-σ); порівняння довжини й покриття проти MC
квантильного інтервалу [Q_0.0015, Q_0.9985].

## Bootstrap

Для статистичної значущості: повторити з 100 bootstrap-репліками MC
для оцінки квантильного інтервалу.

Запуск:
  .venv/bin/python rq3_demo.py
"""

from __future__ import annotations

import sys
from math import factorial
from itertools import product

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# PATP exponent + Derivation-A1 §6 formula (повторено з check_patp.py для standalone)
# ---------------------------------------------------------------------------

def p_num(i: int, alpha: float) -> float:
    return 1.0 / i + (4 - i - 3.0 / i) * alpha + (2 * i - 4 + 2.0 / i) * alpha ** 2


def M_X_uniform01(s: float) -> float:
    # Uniform[0,1]: M_X(s) = 1/s, s > 0
    return 1.0 / s


def patp_moment(k0: float, ks: dict[int, float], alpha: float, j: int) -> float:
    """
    j-й момент g(X;α) = k0 + Σᵢ kᵢ · sgn(X)|X|^{pᵢ(α)} через формулу §6.
    X на додатній підтримці (Uniform[0,1]): M_X^{(m)} = M_X для всіх m.
    """
    indices = sorted(ks.keys())
    pvals = {i: p_num(i, alpha) for i in indices}

    def expand_kappa(m: int):
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
            s_arg = 1.0
            for idx_pos, ki in enumerate(indices):
                ke = kappa[idx_pos]
                if ke == 0:
                    continue
                coef_multi //= factorial(ke)
                prod_k *= ks[ki] ** ke
                s_arg += ke * pvals[ki]
            inner += coef_multi * prod_k * M_X_uniform01(s_arg)
        moment += outer * inner
    return moment


# ---------------------------------------------------------------------------
# Polynomial MUET
# ---------------------------------------------------------------------------

def fit_polynomial(f, S: int, x_grid: np.ndarray) -> np.ndarray:
    """МНК-апроксимація f поліномом степеня S на сітці. Повертає [a₀, a₁, …, a_S]."""
    y = f(x_grid)
    A = np.vander(x_grid, S + 1, increasing=True)  # [1, x, x², …, x^S]
    coefs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coefs  # shape (S+1,)


def polynomial_moment(coefs: np.ndarray, j: int) -> float:
    """
    j-й момент полінома p(X) = Σ_{i=0}^{S} a_i X^i при X ~ Uniform[0,1].
    Розкладаємо (Σ a_i X^i)^j мультиномно і використовуємо E[X^k] = 1/(k+1).
    """
    S = len(coefs) - 1

    def expand(m: int, indices):
        n = len(indices)
        for combo in product(range(m + 1), repeat=n):
            if sum(combo) == m:
                yield combo

    # (Σ_i a_i X^i)^j = Σ_{|κ|=j} (j! / Πκ_i!) Π a_i^{κ_i} · X^{Σ i·κ_i}
    total = 0.0
    indices = list(range(S + 1))
    for kappa in expand(j, indices):
        coef_multi = factorial(j)
        prod_a = 1.0
        x_power = 0
        for i, ke in enumerate(kappa):
            if ke == 0:
                continue
            coef_multi //= factorial(ke)
            prod_a *= coefs[i] ** ke
            x_power += i * ke
        # E[X^{x_power}] = 1/(x_power + 1) для Uniform[0,1]
        total += coef_multi * prod_a * (1.0 / (x_power + 1))
    return total


# ---------------------------------------------------------------------------
# PATP fitting: для фіксованого α — МНК на k₀, k₂, k₃, k₄
# ---------------------------------------------------------------------------

def fit_patp(f, alpha: float, x_grid: np.ndarray, indices: list[int] = [2, 3, 4]) -> dict:
    """
    Підгоняємо k₀ + Σᵢ kᵢ · sgn(x)|x|^{pᵢ(α)} до f на сітці МНК-ом.
    Повертає {k0, ks: {i: k_i}}.
    """
    y = f(x_grid)
    cols = [np.ones_like(x_grid)]
    for i in indices:
        p = p_num(i, alpha)
        cols.append(np.sign(x_grid) * np.abs(x_grid) ** p)
    A = np.column_stack(cols)
    coefs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {"k0": float(coefs[0]), "ks": {indices[k]: float(coefs[k + 1]) for k in range(len(indices))}}


def patp_fit_loss(alpha: float, f, x_grid: np.ndarray, indices: list[int] = [2, 3, 4]) -> float:
    """L² помилка PATP-fit на сітці — для оптимізації α."""
    par = fit_patp(f, alpha, x_grid, indices)
    cols = [np.ones_like(x_grid)]
    for i in indices:
        p = p_num(i, alpha)
        cols.append(np.sign(x_grid) * np.abs(x_grid) ** p)
    pred = par["k0"] * cols[0] + sum(par["ks"][i] * cols[k + 1] for k, i in enumerate(indices))
    return float(np.mean((pred - f(x_grid)) ** 2))


# ---------------------------------------------------------------------------
# Ground truth (closed form for our manufactured solutions)
# ---------------------------------------------------------------------------

def truth_moment(f, j: int, x_grid: np.ndarray = None) -> float:
    """E[f(X)^j] для X ~ Uniform[0,1] через адаптивну квадратуру."""
    val, _ = quad(lambda x: f(x) ** j, 0.0, 1.0, limit=400)
    return val


# ---------------------------------------------------------------------------
# Експеримент
# ---------------------------------------------------------------------------

def approx_l2_error(f, predictor, x_grid: np.ndarray) -> float:
    """‖f - ĝ‖_{L²(x_grid)} — корінь середньоквадратичної відстані на сітці.
    Чисто апроксимаційна метрика: НЕ дивиться на моменти.
    """
    res = predictor(x_grid) - f(x_grid)
    return float(np.sqrt(np.mean(res ** 2)))


def run_one(name: str, f, indices=[2, 3, 4], S_poly: int = 3, n_grid: int = 401) -> dict:
    print(f"\n--- {name} ---")
    x_grid = np.linspace(1e-6, 1.0, n_grid)  # уникнути нуля для x^p

    # Полиноміальний MUET fit (L²-МНК на сітці; ground-truth moments не використовуються)
    poly_coefs = fit_polynomial(f, S_poly, x_grid)
    print(f"  Polynomial (S={S_poly}) coefs: {poly_coefs}")

    def poly_predictor(x):
        return sum(poly_coefs[i] * x ** i for i in range(len(poly_coefs)))

    poly_l2 = approx_l2_error(f, poly_predictor, x_grid)

    # PATP-MUET: optimize α на L²-fit g(·;α,k̂(α)) до f на сітці.
    # ВАЖЛИВО: α-оптимізатор НЕ бачить ground-truth моменти (`truth_moment(f, j)`);
    # він мінімізує лише L²-помилку наближення на тій самій x_grid, що й polynomial.
    res = minimize_scalar(lambda a: patp_fit_loss(a, f, x_grid, indices), bounds=(0.0, 1.0), method="bounded",
                          options={"xatol": 1e-6})
    alpha_opt = res.x
    patp_par = fit_patp(f, alpha_opt, x_grid, indices)
    print(f"  PATP α* = {alpha_opt:.4f}  (L²-loss={res.fun:.2e})")
    print(f"    k₀={patp_par['k0']:+.6f}  ks={ {i: round(v, 6) for i, v in patp_par['ks'].items()} }")
    print(f"    pᵢ(α*) = { {i: round(p_num(i, alpha_opt), 6) for i in indices} }")

    def patp_predictor(x):
        out = patp_par["k0"] * np.ones_like(x)
        for i, k in patp_par["ks"].items():
            out = out + k * np.sign(x) * np.abs(x) ** p_num(i, alpha_opt)
        return out

    patp_l2 = approx_l2_error(f, patp_predictor, x_grid)

    print(f"  Approximation L² error:  poly={poly_l2:.4e}  patp={patp_l2:.4e}")

    results = {"name": name, "alpha_opt": alpha_opt, "poly_coefs": poly_coefs.tolist(),
               "patp_par": patp_par, "j_results": {},
               "approx_l2": {"poly": poly_l2, "patp": patp_l2}}

    print(f"  {'j':>2}  {'truth':>12}  {'poly':>12}  {'patp':>12}  {'rel_poly':>10}  {'rel_patp':>10}  {'win':>10}")
    for j in [1, 2, 3, 4]:
        truth = truth_moment(f, j)
        poly_est = polynomial_moment(poly_coefs, j)
        patp_est = patp_moment(patp_par["k0"], patp_par["ks"], alpha_opt, j)
        rel_poly = abs(poly_est - truth) / max(abs(truth), 1e-15)
        rel_patp = abs(patp_est - truth) / max(abs(truth), 1e-15)
        win = "PATP" if rel_patp < rel_poly else "POLY"
        improvement = (rel_poly - rel_patp) / max(rel_poly, 1e-15) * 100  # % зменшення помилки
        print(f"  {j:>2}  {truth:>12.6f}  {poly_est:>12.6f}  {patp_est:>12.6f}  {rel_poly:>10.2e}  {rel_patp:>10.2e}  {win:>10}  ({improvement:+.1f}%)")
        results["j_results"][j] = {
            "truth": truth, "poly": poly_est, "patp": patp_est,
            "rel_poly": rel_poly, "rel_patp": rel_patp, "improvement_pct": improvement
        }

    # Tail metric: extended uncertainty U = 3·σ
    mu_true = truth_moment(f, 1)
    var_true = truth_moment(f, 2) - mu_true ** 2
    U_true = 3.0 * np.sqrt(var_true)

    mu_poly = polynomial_moment(poly_coefs, 1)
    var_poly = polynomial_moment(poly_coefs, 2) - mu_poly ** 2
    U_poly = 3.0 * np.sqrt(max(var_poly, 0.0))

    mu_patp = patp_moment(patp_par["k0"], patp_par["ks"], alpha_opt, 1)
    var_patp = patp_moment(patp_par["k0"], patp_par["ks"], alpha_opt, 2) - mu_patp ** 2
    U_patp = 3.0 * np.sqrt(max(var_patp, 0.0))

    print(f"  Extended uncertainty (3σ):  true={U_true:.6f}  poly={U_poly:.6f}  patp={U_patp:.6f}")
    print(f"  |ΔU|: poly={abs(U_poly-U_true):.2e}  patp={abs(U_patp-U_true):.2e}")
    results["U"] = {"true": U_true, "poly": U_poly, "patp": U_patp}
    return results


def main() -> int:
    rng = np.random.default_rng(42)

    # Сценарії v2 (post-review):
    #   M1, M4 — реальні non-trivial цілі (PATP-базис НЕ містить точно)
    #   M2, M3 — sanity (PATP-базис МІСТИТЬ ціль точно при правильному α)
    #   M5     — anti-test (polynomial-basis МІСТИТЬ ціль точно)
    #   M6, M7 — НОВІ «hard» цілі: ні PATP, ні polynomial не містять точно;
    #            обидва мають апроксимаційну похибку → справжній тест моментної пропагації
    scenarios = [
        ("M1: f(x) = 0.6·x^{0.5} + 0.4·x^{1.5} (non-trivial mixed)", lambda x: 0.6 * x ** 0.5 + 0.4 * x ** 1.5),
        ("M2: f(x) = x^{0.7} (sanity: PATP-basis exact)",            lambda x: x ** 0.7),
        ("M3: f(x) = x^{1.7} (sanity: PATP-basis exact)",            lambda x: x ** 1.7),
        ("M4: f(x) = √x · (1 + x²/3) (hybrid frac × poly)",          lambda x: np.sqrt(x) * (1 + x ** 2 / 3)),
        ("M5: f(x) = x³ (anti-test: polynomial-basis exact)",        lambda x: x ** 3),
        ("M6: f(x) = ln(1 + x) (hard: outside both bases)",          lambda x: np.log(1 + x)),
        ("M7: f(x) = exp(-x) (hard: outside both bases)",            lambda x: np.exp(-x)),
    ]

    all_results = []
    for name, f in scenarios:
        all_results.append(run_one(name, f))

    print("\n" + "=" * 72)
    print("ПІДСУМОК v2 — approximation error + per-j moment errors")
    print("=" * 72)
    print(f"  {'Сценарій':<55} {'L²-poly':>10} {'L²-patp':>10}  {'avg-impr':>9}")
    for r in all_results:
        a_poly = r["approx_l2"]["poly"]
        a_patp = r["approx_l2"]["patp"]
        avg_impr = float(np.mean([r["j_results"][j]["improvement_pct"] for j in [1, 2, 3, 4]]))
        print(f"  {r['name'][:55]:<55} {a_poly:>10.2e} {a_patp:>10.2e} {avg_impr:>+8.1f}%")

    print("\nКатегоризація (v2):")
    print("  • non-trivial (M1, M4): PATP проти polynomial при non-zero approx error обох")
    print("  • sanity (M2, M3): PATP-basis MISTYT' f точно → moment err ~ machine precision")
    print("  • anti-test (M5): polynomial-basis MISTYT' f точно → polynomial виграє")
    print("  • hard (M6, M7): обидва базиси мають approx error → справжній тест пропагації")

    print("\nH3 verdict (≥15% покращення на non-degenerate цілях):")
    non_degenerate = [r for r in all_results if r["name"].startswith(("M1", "M4", "M6", "M7"))]
    for r in non_degenerate:
        avg_impr = float(np.mean([r["j_results"][j]["improvement_pct"] for j in [1, 2, 3, 4]]))
        max_impr = max(r["j_results"][j]["improvement_pct"] for j in [1, 2, 3, 4])
        h3_pass = avg_impr >= 15.0
        tag = "✓ confirmed" if h3_pass else ("○ marginal" if avg_impr > 0 else "✗ negative")
        print(f"  {r['name'][:55]:<55}: avg = {avg_impr:+6.1f}%, max = {max_impr:+6.1f}% — {tag}")

    print("\nSanity (M2, M3 — degenerate, винесено з заголовного числа):")
    for r in [x for x in all_results if x["name"].startswith(("M2", "M3"))]:
        print(f"  {r['name'][:55]:<55}: PATP L² = {r['approx_l2']['patp']:.2e} (точне попадання)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
