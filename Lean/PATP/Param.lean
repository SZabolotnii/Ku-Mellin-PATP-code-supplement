import Mathlib.Tactic

/-!
# PATP.Param — поліноміальна параметризація показників $p_i(\alpha)$

> **VENDORED COPY.** Канонічне джерело — `github.com/SZabolotnii/Ku_PATP/Lean/PATP/Param.lean`.
> Цей файл — byte-identical копія для замикання графу імпортів цього окремого проєкту
> (`Ku-Mellin-PATP`). При зміні в Ku_PATP — синхронізувати тут вручну.
> Sync date: 2026-05-18.

Формалізує квадратичну параметризацію показників PATP-базису з §3.2
статті: $p_i(\alpha) = 1/i + (4 - i - 3/i)\alpha + (2i - 4 + 2/i)\alpha^2$.

Доведено три структурні теореми про граничні випадки (§3.3):

* `pi_alpha_at_zero`: $p_i(0) = 1/i$ (фрактальний режим)
* `pi_alpha_at_half`: $p_i(1/2) = 1$ (вироджений лінійний режим)
* `pi_alpha_at_one`: $p_i(1) = i$ (класичний степеневий режим Кунченка)

Усі три теореми вимагають $i \neq 0$, бо параметризація містить $1/i$.
-/

namespace PATP

/-- Параметризація показника $p_i(\alpha)$ у PATP-сім'ї (§3.2 paper):
$$p_i(\alpha) = \frac{1}{i} + \left(4 - i - \frac{3}{i}\right)\alpha
              + \left(2i - 4 + \frac{2}{i}\right)\alpha^2.$$

Виведена через лагранжеву інтерполяцію в точках $\{0, 1/2, 1\}$
зі значеннями $\{1/i, 1, i\}$ відповідно (див. §3.2 paper, рівн. (3.3)).
-/
noncomputable def piAlpha (i : ℕ) (α : ℝ) : ℝ :=
  1 / (i : ℝ) + (4 - (i : ℝ) - 3 / (i : ℝ)) * α
            + (2 * (i : ℝ) - 4 + 2 / (i : ℝ)) * α ^ 2

/-- **Граничний випадок $\alpha = 0$ (фрактальний режим):** $p_i(0) = 1/i$. -/
theorem piAlpha_at_zero (i : ℕ) : piAlpha i 0 = 1 / (i : ℝ) := by
  unfold piAlpha
  ring

/-- **Граничний випадок $\alpha = 1/2$ (вироджений лінійний режим):**
$p_i(1/2) = 1$ для всіх $i \neq 0$.

У цій точці всі базисні функції $\varphi_i(\xi; 1/2)$ колапсують до
лінійної $\xi$, матриця $\mathbf{F}_S(1/2)$ вироджена (ранг 1), і
PATP-оцінювач зводиться до МНК. Доведення цієї виродженості —
аналітичне обґрунтування §3.3 paper. -/
theorem piAlpha_at_half (i : ℕ) (hi : i ≠ 0) : piAlpha i (1 / 2) = 1 := by
  unfold piAlpha
  have hi' : (i : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hi
  field_simp
  ring

/-- **Граничний випадок $\alpha = 1$ (класичний степеневий режим):**
$p_i(1) = i$ для всіх $i \neq 0$.

При $\alpha = 1$ PATP-сім'я відтворює канонічний степеневий базис
Кунченка $\{\xi, \mathrm{sign}(\xi)|\xi|^2, \mathrm{sign}(\xi)|\xi|^3,
\ldots, \mathrm{sign}(\xi)|\xi|^S\}$. -/
theorem piAlpha_at_one (i : ℕ) (hi : i ≠ 0) : piAlpha i 1 = (i : ℝ) := by
  unfold piAlpha
  have hi' : (i : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hi
  field_simp
  ring

/-- **Конкретний приклад:** $p_2(1/2) = 1$. -/
example : piAlpha 2 (1 / 2) = 1 := piAlpha_at_half 2 (by decide)

/-- **Конкретний приклад:** $p_2(1) = 2$ (класичний $\xi^2$-член Кунченка). -/
example : piAlpha 2 1 = 2 := by
  have := piAlpha_at_one 2 (by decide)
  norm_num at this
  exact this

/-- **Конкретний приклад:** $p_3(0) = 1/3$ (фрактальний $|\xi|^{1/3}$). -/
example : piAlpha 3 0 = 1 / 3 := by
  have := piAlpha_at_zero 3
  norm_num at this
  exact this

end PATP
