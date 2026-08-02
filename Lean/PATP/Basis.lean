import Mathlib.Tactic
import Mathlib.Data.Real.Sign
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import PATP.Param

/-!
# PATP.Basis — знакозбережна базова сім'я $\varphi_i(\xi; \alpha)$

> **VENDORED COPY.** Канонічне джерело — `github.com/SZabolotnii/Ku_PATP/Lean/PATP/Basis.lean`.
> Цей файл — byte-identical копія для замикання графу імпортів цього окремого проєкту
> (`Ku-Mellin-PATP`). При зміні в Ku_PATP — синхронізувати тут вручну.
> Sync date: 2026-05-18.

Формалізує signed-parity (Form B) сім'ю базисних функцій PATP з §3.1
статті: $\varphi_i(\xi; \alpha) = \mathrm{sign}(\xi) \cdot |\xi|^{p_i(\alpha)}$
для $i \geq 2$.

Доведено три структурні теореми про граничні випадки базису (§3.3):

* `patpBasis_at_zero`:  $\varphi_i(\xi; 0)   = \mathrm{sign}(\xi)|\xi|^{1/i}$
* `patpBasis_at_half`:  $\varphi_i(\xi; 1/2) = \xi$ (вироджений лінійний)
* `patpBasis_at_one`:   $\varphi_i(\xi; 1)   = \mathrm{sign}(\xi)|\xi|^i$

Перша і третя теореми --- лише rewriting через `piAlpha_at_zero/one`;
друга вимагає тотожності $\mathrm{sign}(\xi) \cdot |\xi| = \xi$ для всіх $\xi \in \mathbb{R}$.
-/

namespace PATP

/-- PATP знакозбережна (signed-parity) базисна функція індексу $i \geq 2$:
$$\varphi_i(\xi; \alpha) := \mathrm{sign}(\xi) \cdot |\xi|^{p_i(\alpha)},$$
де $p_i(\alpha)$ задається у `piAlpha` (§3.2 paper, Lagrange interpolation).

Перша базисна функція (для $i = 1$) у конструкції PATP завжди тотожна
$\varphi_1(\xi) \equiv \xi$ (не залежить від $\alpha$), тому її не
параметризовано і не означено тут. -/
noncomputable def patpBasis (i : ℕ) (α : ℝ) (ξ : ℝ) : ℝ :=
  ξ.sign * |ξ| ^ piAlpha i α

/-- Допоміжна тотожність: $\mathrm{sign}(\xi) \cdot |\xi| = \xi$ для всіх $\xi \in \mathbb{R}$. -/
theorem sign_mul_abs_eq_self (ξ : ℝ) : ξ.sign * |ξ| = ξ := by
  rcases lt_trichotomy ξ 0 with h | h | h
  · rw [Real.sign_of_neg h, abs_of_neg h]; ring
  · subst h; simp
  · rw [Real.sign_of_pos h, abs_of_pos h]; ring

/-- **Граничний випадок $\alpha = 0$ (фрактальний режим):**
$\varphi_i(\xi; 0) = \mathrm{sign}(\xi) \cdot |\xi|^{1/i}$ для $i \neq 0$.

При $\alpha = 0$ маємо $p_i(0) = 1/i$, тому базисна функція стає
дробно-степеневою з показником $1/i$, придатною для розподілів з
важкими хвостами (де цілі моменти $\E[\xi^i]$ можуть бути нескінченні,
а $\E[|\xi|^{1/i}]$ скінченні). -/
theorem patpBasis_at_zero (i : ℕ) (ξ : ℝ) :
    patpBasis i 0 ξ = ξ.sign * |ξ| ^ (1 / (i : ℝ)) := by
  unfold patpBasis
  rw [piAlpha_at_zero]

/-- **Граничний випадок $\alpha = 1/2$ (вироджений лінійний режим):**
$\varphi_i(\xi; 1/2) = \xi$ для всіх $i \neq 0$.

При $\alpha = 1/2$ маємо $p_i(1/2) = 1$, тому
$\mathrm{sign}(\xi) \cdot |\xi|^1 = \mathrm{sign}(\xi) \cdot |\xi| = \xi$.
Усі базисні функції $\{\varphi_2(\,\cdot\,; 1/2), \ldots, \varphi_S(\,\cdot\,; 1/2)\}$
збігаються з $\varphi_1(\xi) \equiv \xi$, що приводить до виродженості
матриці $\mathbf{F}_S(1/2)$ (див. §3.4 paper). -/
theorem patpBasis_at_half (i : ℕ) (hi : i ≠ 0) (ξ : ℝ) :
    patpBasis i (1 / 2) ξ = ξ := by
  unfold patpBasis
  rw [piAlpha_at_half i hi, Real.rpow_one]
  exact sign_mul_abs_eq_self ξ

/-- **Граничний випадок $\alpha = 1$ (signed-parity integer-polynomial режим):**
$\varphi_i(\xi; 1) = \mathrm{sign}(\xi) \cdot |\xi|^i$ для $i \neq 0$.

При $\alpha = 1$ маємо $p_i(1) = i$. Для **непарних** $i$ ця функція
тотожна $\xi^i$ (точне відтворення класичного степеневого базису
Кунченка). Для **парних** $i$ маємо $\mathrm{sign}(\xi)|\xi|^i \neq \xi^i$:
це signed-parity варіація класичного базису, що приводить до
дещо іншого оцінювача порівняно з канонічним PMM Кунченка
(детальний аналіз --- у §4 paper). -/
theorem patpBasis_at_one (i : ℕ) (hi : i ≠ 0) (ξ : ℝ) :
    patpBasis i 1 ξ = ξ.sign * |ξ| ^ (i : ℝ) := by
  unfold patpBasis
  rw [piAlpha_at_one i hi]

/-- **Непарність базису:** $\varphi_i(-\xi; \alpha) = -\varphi_i(\xi; \alpha)$
для всіх $i \in \mathbb{N}$, $\alpha \in \mathbb{R}$, $\xi \in \mathbb{R}$.

Це структурна властивість signed-parity конструкції: PATP-базис є
\emph{непарною} функцією $\xi$ за будь-якого $\alpha$, незалежно від
парності показника $p_i(\alpha)$. Саме ця властивість виокремлює Form~B
(residual-basis з signed-parity) серед інших можливих формулювань і
забезпечує одержання нетривіальних кореляцій $F_{ij}$ для асиметричних
розподілів. -/
theorem patpBasis_odd (i : ℕ) (α : ℝ) (ξ : ℝ) :
    patpBasis i α (-ξ) = -(patpBasis i α ξ) := by
  unfold patpBasis
  rw [abs_neg, Real.sign_neg]
  ring

/-- **Базис у нулі:** $\varphi_i(0; \alpha) = 0$ для всіх $i$, $\alpha$ з $p_i(\alpha) > 0$. -/
theorem patpBasis_at_zero_input (i : ℕ) (α : ℝ) (hp : 0 < piAlpha i α) :
    patpBasis i α 0 = 0 := by
  unfold patpBasis
  rw [Real.sign_zero, abs_zero, Real.zero_rpow (ne_of_gt hp)]
  ring

/-- **Конкретний приклад:** $\varphi_2(\xi; 1/2) = \xi$ (вироджений випадок). -/
example (ξ : ℝ) : patpBasis 2 (1 / 2) ξ = ξ := patpBasis_at_half 2 (by decide) ξ

/-- **Конкретний приклад:** $\varphi_2(\xi; 0) = \mathrm{sign}(\xi) |\xi|^{1/2}$ (фрактальний). -/
example (ξ : ℝ) : patpBasis 2 0 ξ = ξ.sign * |ξ| ^ (1 / (2 : ℝ)) := by
  have := patpBasis_at_zero 2 ξ
  norm_num at this
  exact this

/-- **Конкретний приклад непарності:** $\varphi_2(-3; 0) = -\varphi_2(3; 0)$. -/
example : patpBasis 2 0 (-3) = -(patpBasis 2 0 3) := patpBasis_odd 2 0 3

end PATP
