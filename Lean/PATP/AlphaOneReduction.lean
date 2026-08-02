import Mathlib.Tactic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import PATP.Param
import PATP.Basis
import PATP.MellinPower

/-!
# PATP.AlphaOneReduction — зведення формули §6 при α=1

T4 з `Derivation-A1-Mellin-PATP.md` §7, §11.

## Зміст

При `α = 1` маємо `pᵢ(1) = i` (Param.piAlpha_at_one), і
`φᵢ(ξ; 1) = sign(ξ) · |ξ|^i` (Basis.patpBasis_at_one). Для будь-якого
натурального степеня `j`:

  `(φᵢ(ξ; 1))^j = (sign(ξ))^j · |ξ|^{i·j}`.

Для `j` парного: `(sign ξ)^j = 1`, отже `(φᵢ)^j = |ξ|^{i·j}`.
Для `j` непарного: `(sign ξ)^j = sign ξ`, отже `(φᵢ)^j = sign(ξ) · |ξ|^{i·j}`.

Це **точна межа зведення** PATP-MUET до класичного raw-monomial MUET
на додатній підтримці: для `X ≥ 0` обидві форми тотожні; для signed `X`
PATP-MUET зберігає знак, тоді як raw-monomial MUET його втрачає на парних
ступенях (`Derivation-A1` §7, §10).
-/

namespace PATP

open Real

/-- **T4 — алгебраїчна форма зведення `α = 1`.**
Для будь-якого `i ≠ 0` і `ξ : ℝ`:
$$\varphi_i(\xi; 1) = \mathrm{sign}(\xi) \cdot |\xi|^i.$$

Прямий висновок `patpBasis_at_one` — наведено тут для повноти T-послідовності. -/
theorem alpha_one_reduction (i : ℕ) (hi : i ≠ 0) (ξ : ℝ) :
    patpBasis i 1 ξ = ξ.sign * |ξ| ^ (i : ℝ) :=
  patpBasis_at_one i hi ξ

/-- **T4 — j-та степінь PATP-базису при `α = 1`.**
$$\bigl(\varphi_i(\xi; 1)\bigr)^j = (\mathrm{sign}\,\xi)^j \cdot |\xi|^{i \cdot j}.$$

Це й є «знакозберігаюча цілостепеневa пропагація» зі Spec-2 §7 та
`Derivation-A1` §7: для парного `j` знакова частина зникає, для непарного —
зберігається. Перехід до моменту `E[(sgn ξ)^j · |ξ|^{ij}]` — це парність-
селекція бістороннього перетворення Мелліна (`Derivation-A1` §4.2, §5). -/
theorem alpha_one_pow (i : ℕ) (hi : i ≠ 0) (ξ : ℝ) (j : ℕ) (hξ : ξ ≠ 0) :
    (patpBasis i 1 ξ) ^ j = (ξ.sign) ^ j * |ξ| ^ ((i : ℝ) * j) := by
  rw [patpBasis_at_one i hi ξ]
  rw [mul_pow]
  congr 1
  rw [← Real.rpow_natCast (|ξ| ^ (i : ℝ)) j]
  rw [← Real.rpow_mul (abs_nonneg ξ)]

/-- **Конкретний приклад T4:** $\varphi_3(\xi; 1)^2 = \xi^2 \cdot |\xi|^4$
(для `ξ ≠ 0`; раціональне $\xi^2 = (\mathrm{sign}\,\xi)^2 \cdot |\xi|^2$). -/
example (ξ : ℝ) (hξ : ξ ≠ 0) :
    (patpBasis 3 1 ξ) ^ 2 = (ξ.sign) ^ 2 * |ξ| ^ ((3 : ℝ) * 2) :=
  alpha_one_pow 3 (by decide) ξ 2 hξ

end PATP
