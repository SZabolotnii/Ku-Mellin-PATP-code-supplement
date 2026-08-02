import Mathlib.Analysis.MellinTransform
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Pow.Complex

/-!
# PATP.MellinPower — правило Мелліна для степеневого перетворення

T3 з `Derivation-A1-Mellin-PATP.md` §3, §11 (проєкт Ku-Mellin-PATP).

## Структура

T3 формалізується у двох виглядах:

* `mellin_power_rule_abstract` — пряме переформулювання Mathlib-ової
  `mellin_comp_rpow` у запис, що використовує `c > 0` замість `a ∈ ℝ`:
  $$\mellin (f \circ (\cdot^{1/c}))(s) = c \cdot \mellin f (c s).$$
  Доказ — `mellin_comp_rpow` з `a = 1/c`; `|1/c|^{-1} = c` і `s / (1/c) = c s`.

* `patp_mellin_density_substitution` — повна формула з §3 для
  щільності $Y = X^c$ (з Якобіаном):
  $$\mellin f_Y(s) = \mellin f_X(c(s-1) + 1),$$
  де $f_Y(y) = (1/c) \cdot y^{1/c - 1} \cdot f_X(y^{1/c})$.
  Цей запис ідентичний `Derivation-A1` §3.

Оскільки PATP-MUET (Derivation-A1 §6) використовує **моментну форму**
`E[Y^{s-1}] = E[X^{c(s-1)}] = mellin f_X (c(s-1)+1)`, а не власне
mellin щільності `f_Y`, друга лема — це строгий аналог формули,
використаної в §6 при ототожненні `M_X^{(m)}( pᵢ(α)k + 1 )` =
`M_X( pᵢ(α)·k + 1 )` на додатній підтримці.
-/

namespace PATP

open Real Complex MeasureTheory

/-- **T3 — абстрактне правило Мелліна для степеня.**
$$\mellin (f \circ (\cdot^{1/c}))(s) = c \cdot \mellin f (c s),\quad c > 0.$$

Безпосередній наслідок `mellin_comp_rpow` з `a = 1/c`: на додатних `c` маємо
`|1/c|⁻¹ = c` і `s / (1/c) = c·s`. -/
theorem mellin_power_rule_abstract
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℂ E]
    (f : ℝ → E) (s : ℂ) {c : ℝ} (hc : 0 < c) :
    mellin (fun t => f (t ^ (1 / c))) s = (c : ℂ) • mellin f (c * s) := by
  -- Mathlib: mellin (fun t => f (t ^ a)) s = |a|⁻¹ • mellin f (s / a)
  rw [mellin_comp_rpow f s (1 / c)]
  have h1c : (1 : ℝ) / c > 0 := by positivity
  have habs : |(1 : ℝ) / c| = 1 / c := abs_of_pos h1c
  rw [habs]
  -- Goal: ((1/c)⁻¹ : ℝ) • mellin f (s / (1/c : ℝ)) = (c : ℂ) • mellin f (c * s)
  have hinv : ((1 : ℝ) / c)⁻¹ = c := by field_simp
  rw [hinv]
  -- s / (↑(1/c)) = c * s; congr 1 splits into smul-scalar and mellin-arg parts
  congr 1
  push_cast
  field_simp

/-- **T3 — повна форма для щільності `Y = X^c`** (`Derivation-A1` §3).
Якщо $f_X$ — щільність на $(0, \infty)$ і $Y = X^c$ при $c > 0$, то щільність $Y$ —
$f_Y(y) = (1/c) \cdot y^{1/c - 1} \cdot f_X(y^{1/c})$, і
$$\mellin f_Y(s) = \mellin f_X\bigl(c(s-1) + 1\bigr).$$

Доказ — композиція `mellin_cpow_smul` (на множнику `(1/c) · y^{1/c-1}`) та
`mellin_power_rule_abstract` (на `f_X ∘ (·^{1/c})`).
-/
theorem patp_mellin_density_substitution
    (f_X : ℝ → ℂ) (s : ℂ) {c : ℝ} (hc : 0 < c) :
    mellin (fun y => ((1 / (c : ℂ)) * (y : ℂ) ^ ((1 / c : ℂ) - 1)) * f_X (y ^ (1 / c))) s
      = mellin f_X (c * (s - 1) + 1) := by
  have hc' : (c : ℂ) ≠ 0 := by exact_mod_cast hc.ne'
  -- Перепишемо інтегранд як  (1/c) • ( (y : ℂ)^{1/c-1} • f_X(y^{1/c}) )
  have heq :
      (fun y : ℝ => ((1 / (c : ℂ)) * (y : ℂ) ^ ((1 / c : ℂ) - 1)) * f_X (y ^ (1 / c)))
        = fun y : ℝ => (1 / (c : ℂ)) • ((y : ℂ) ^ ((1 / c : ℂ) - 1) • f_X (y ^ (1 / c))) := by
    funext y
    simp [smul_eq_mul, mul_assoc]
  rw [heq]
  -- mellin (fun y => (1/c) • g y) s = (1/c) • mellin g s (через mellin_const_smul із 𝕜 = ℂ)
  rw [mellin_const_smul (fun y => ((y : ℂ) ^ ((1 / c : ℂ) - 1) • f_X (y ^ (1 / c)))) s
        (1 / (c : ℂ))]
  -- mellin (fun y => y^a • g y) s = mellin g (s + a)   (Mathlib: mellin_cpow_smul)
  rw [mellin_cpow_smul (fun y => f_X (y ^ (1 / c))) s ((1 / c : ℂ) - 1)]
  -- Тепер застосовуємо абстрактну степеневу леммy
  rw [mellin_power_rule_abstract f_X (s + ((1 / c : ℂ) - 1)) hc]
  -- Залишаємось з: (1/c) • (c : ℂ) • mellin f_X (c * (s + 1/c - 1)) = mellin f_X (c*(s-1)+1)
  rw [smul_smul]
  have h1 : (1 / (c : ℂ)) * (c : ℂ) = 1 := by field_simp
  rw [h1, one_smul]
  -- arg simplification: c*(s + 1/c - 1) = c*(s - 1) + 1
  congr 1
  field_simp
  ring

end PATP
