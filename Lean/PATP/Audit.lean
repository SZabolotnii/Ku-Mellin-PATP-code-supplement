import PATP.Param
import PATP.Basis
import PATP.Positivity
import PATP.MellinPower
import PATP.AlphaOneReduction

/-!
# Аудит аксіом

`#print axioms` для кожної декларації верифікаційного шару. Очікуваний вивід —
**лише** стандартні аксіоми Lean/Mathlib (`propext`, `Classical.choice`,
`Quot.sound`): жодного `sorryAx`, жодної власної аксіоми.

Без цього файлу збірка мовчить про залежності: `lake build PATP` проходить
зелено і тоді, коли теорема спирається на щось стороннє, бо `sorry`-вільність
і чистота аксіом — це різні твердження. Саме цей файл перетворює «зібралось»
на «засвідчено», і саме на нього можна послатися в рукописі.

Покриття: 26 декларацій — T2 (`Positivity`), T3 (`MellinPower`),
T4 (`AlphaOneReduction`) з `Derivation-A1-Mellin-PATP.md` §11, плюс
vendored-база `Param` / `Basis` з Ku_PATP.
-/

open PATP

/-! ## База: показники `p_i(α)` та знакозбережний базис `φ_i(ξ; α)`
(vendored з Ku_PATP — первинне джерело `Ku_PATP/Lean/PATP/{Param,Basis}.lean`) -/

#print axioms piAlpha
#print axioms piAlpha_at_zero
#print axioms piAlpha_at_half
#print axioms piAlpha_at_one

#print axioms patpBasis
#print axioms sign_mul_abs_eq_self
#print axioms patpBasis_at_zero
#print axioms patpBasis_at_half
#print axioms patpBasis_at_one
#print axioms patpBasis_odd
#print axioms patpBasis_at_zero_input

/-! ## T2 — знаковий аналіз `p_i(α)` на `[0,1]`

Розклади `piAlpha_*_expand` тримають додатність для `i = 2…5`; свідок
`piAlpha 6 (3/20) = -1/48` показує, що при `i = 6` вона вже хибна — це і є
межа області, а не технічна прогалина. -/

#print axioms piAlpha_two_expand
#print axioms piAlpha_three_expand
#print axioms piAlpha_four_expand
#print axioms piAlpha_five_expand
#print axioms piAlpha_six_expand

#print axioms piAlpha_pos_two
#print axioms piAlpha_pos_three
#print axioms piAlpha_pos_four
#print axioms piAlpha_pos_five

#print axioms piAlpha_six_at_witness
#print axioms piAlpha_six_negative_witness

/-! ## T3 — правило Мелліна для `Y = X^c` -/

#print axioms mellin_power_rule_abstract
#print axioms patp_mellin_density_substitution

/-! ## T4 — степені PATP-базису при `α = 1` -/

#print axioms alpha_one_reduction
#print axioms alpha_one_pow
