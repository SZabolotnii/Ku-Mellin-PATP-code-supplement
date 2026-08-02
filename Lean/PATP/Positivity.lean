import Mathlib.Tactic
import PATP.Param

/-!
# PATP.Positivity — знаковість показників $p_i(\alpha)$ на $[0,1]$

T2 з `Derivation-A1-Mellin-PATP.md` §11 (проєкт Ku-Mellin-PATP).
Закриває аналіз області §9 цього ж документа: load-bearing передумова
того, що PATP-MUET в режимах $i \le 5$, $\alpha \in [0,1]$ безумовно
чинна без додаткових обмежень.

## Структура

* Для $i \in \{2, 3, 4, 5\}$: $p_i(\alpha) > 0$ на всьому $[0,1]$.
  Доведення — через подання $p_i(\alpha) = A_i \cdot (\alpha - \alpha_i^\*)^2 + r_i$
  з $A_i > 0$ і $r_i > 0$ (від'ємний дискримінант $\Rightarrow$ квадратна
  форма не змінює знаку; знак узгоджений із знаком сталого члена).

* Для $i = 6$: пред'явлено явний свідок $\alpha = 3/20$, де
  $p_6(3/20) = -1/48 < 0$. Це підтверджує `Derivation-A1` §9:
  «для $i=6$: $p_i(\alpha)<0$ при $\alpha \in (0.10, 0.20)$,
  мінімум $\approx -0.021$» ($-1/48 \approx -0.02083$).

Незалежність від Mathlib-перекриття Мелліна: усі доведення —
поліноміальна алгебра, тактики `ring` / `nlinarith` / `norm_num`.
-/

namespace PATP

/-! ## 1. Підготовчі обчислення -/

/-- Розгорнута квадратична форма $p_2(\alpha) = 1/2 + (1/2)\alpha + \alpha^2$. -/
theorem piAlpha_two_expand (α : ℝ) :
    piAlpha 2 α = 1/2 + (1/2) * α + α^2 := by
  unfold piAlpha
  push_cast
  ring

/-- $p_3(\alpha) = 1/3 + (8/3)\alpha^2$ (середній коефіцієнт нульовий). -/
theorem piAlpha_three_expand (α : ℝ) :
    piAlpha 3 α = 1/3 + (8/3) * α^2 := by
  unfold piAlpha
  push_cast
  ring

/-- $p_4(\alpha) = 1/4 - (3/4)\alpha + (9/2)\alpha^2$. -/
theorem piAlpha_four_expand (α : ℝ) :
    piAlpha 4 α = 1/4 - (3/4) * α + (9/2) * α^2 := by
  unfold piAlpha
  push_cast
  ring

/-- $p_5(\alpha) = 1/5 - (8/5)\alpha + (32/5)\alpha^2$. -/
theorem piAlpha_five_expand (α : ℝ) :
    piAlpha 5 α = 1/5 - (8/5) * α + (32/5) * α^2 := by
  unfold piAlpha
  push_cast
  ring

/-- $p_6(\alpha) = 1/6 - (5/2)\alpha + (25/3)\alpha^2$. -/
theorem piAlpha_six_expand (α : ℝ) :
    piAlpha 6 α = 1/6 - (5/2) * α + (25/3) * α^2 := by
  unfold piAlpha
  push_cast
  ring

/-! ## 2. T2 — додатність для $i \in \{2, 3, 4, 5\}$

Для кожного $i$ підхід однаковий: розгортаємо `piAlpha`, передаємо
`nlinarith` квадрат-додатну підказку `sq_nonneg ((α - α_i^*))` зі
зсувом до вершини параболи. Оскільки дискримінант від'ємний (див.
`Derivation-A1` §9), мінімум параболи додатний, і `nlinarith` закриває.
-/

/-- $\forall α \in [0,1]:\; p_2(α) > 0$. -/
theorem piAlpha_pos_two (α : ℝ) (h0 : 0 ≤ α) (h1 : α ≤ 1) :
    0 < piAlpha 2 α := by
  rw [piAlpha_two_expand]
  -- p_2(α) = (α + 1/4)^2 + 7/16
  nlinarith [sq_nonneg (α + (1:ℝ)/4), sq_nonneg α]

/-- $\forall α \in [0,1]:\; p_3(α) > 0$. Тривіально: $1/3 + (8/3)α^2 > 0$. -/
theorem piAlpha_pos_three (α : ℝ) (_h0 : 0 ≤ α) (_h1 : α ≤ 1) :
    0 < piAlpha 3 α := by
  rw [piAlpha_three_expand]
  nlinarith [sq_nonneg α]

/-- $\forall α \in [0,1]:\; p_4(α) > 0$. Вершина: $α^* = 3/(2 \cdot 9) = 1/12$. -/
theorem piAlpha_pos_four (α : ℝ) (h0 : 0 ≤ α) (h1 : α ≤ 1) :
    0 < piAlpha 4 α := by
  rw [piAlpha_four_expand]
  -- p_4(α) = (9/2)(α - 1/12)^2 + (1/4 - (9/2)/144) = ... > 0
  nlinarith [sq_nonneg (α - (1:ℝ)/12), sq_nonneg α]

/-- $\forall α \in [0,1]:\; p_5(α) > 0$. Вершина: $α^* = (8/5)/(2 \cdot 32/5) = 1/8$. -/
theorem piAlpha_pos_five (α : ℝ) (h0 : 0 ≤ α) (h1 : α ≤ 1) :
    0 < piAlpha 5 α := by
  rw [piAlpha_five_expand]
  nlinarith [sq_nonneg (α - (1:ℝ)/8), sq_nonneg α]

/-! ## 3. T2 — від'ємність для $i = 6$ (контр-приклад)

Свідок $\alpha = 3/20$ дає точне раціональне значення
$p_6(3/20) = -1/48 < 0$. Це падає в інтервал
$(0.10, 0.20)$ зі сітки CAS (`verification/cas/REPORT.md`, C2):
від'ємний інтервал виявлено на $\alpha \in [0.101, 0.199]$.
-/

/-- $p_6(3/20) = -1/48$. Точне раціональне значення. -/
theorem piAlpha_six_at_witness : piAlpha 6 (3/20) = -1/48 := by
  rw [piAlpha_six_expand]
  norm_num

/-- $\exists \alpha \in (0, 1):\; p_6(\alpha) < 0$. Свідок: $\alpha = 3/20$. -/
theorem piAlpha_six_negative_witness :
    ∃ α : ℝ, 0 < α ∧ α < 1 ∧ piAlpha 6 α < 0 := by
  refine ⟨3/20, ?_, ?_, ?_⟩
  · norm_num
  · norm_num
  · rw [piAlpha_six_at_witness]; norm_num

/-! ## 4. Конкретні приклади -/

/-- **Приклад:** $p_2(0.5) > 0$ (вироджений лінійний режим). -/
example : 0 < piAlpha 2 (1/2) :=
  piAlpha_pos_two (1/2) (by norm_num) (by norm_num)

/-- **Приклад:** $p_5(1) > 0$ (цілостеповий, $p_5(1) = 5$). -/
example : 0 < piAlpha 5 1 :=
  piAlpha_pos_five 1 (by norm_num) (by norm_num)

end PATP
