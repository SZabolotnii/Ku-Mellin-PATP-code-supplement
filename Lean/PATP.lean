import PATP.Param
import PATP.Basis
import PATP.Positivity
import PATP.MellinPower
import PATP.AlphaOneReduction
import PATP.Audit

/-!
# PATP — Lean verification layer for Ku-Mellin-PATP

Цей entry-point покриває T2/T3/T4 з `Derivation-A1-Mellin-PATP.md` §11
(PLAN.md Phase 3-4 проєкту Ku-Mellin-PATP).

## Зміст

* `PATP.Param` — функція показників `p_i(α)` та теореми про три
  граничні випадки (vendored copy з Ku_PATP; первинне джерело —
  `Ku_PATP/Lean/PATP/Param.lean`).
* `PATP.Basis` — знакозбережна сім'я `φ_i(ξ; α)` (vendored copy
  з Ku_PATP).
* `PATP.Positivity` — **T2**: знаковий аналіз `p_i(α)` на $[0,1]$.
* `PATP.MellinPower` — **T3**: правило Мелліна для $Y = X^c$.
* `PATP.AlphaOneReduction` — **T4**: степеня PATP-базису при $α=1$.
* `PATP.Audit` — `#print axioms` для всіх 26 декларацій: збірка друкує
  залежності, тож «зелено» стає перевірюваним твердженням, а не мовчанням.

Аудит покриття Mathlib4: `Lean/PATP/MELLIN_AUDIT.md`.

## Залежність від Ku_PATP

`Param.lean` і `Basis.lean` — це **vendored copies** для замикання
графу імпортів. Канонічний first-party реліз цих файлів — у репозиторії
[`Ku_PATP`](https://github.com/SZabolotnii/Ku_PATP), де вони цитуються
як load-bearing у `paper_cstat` Appendix. При змінах у Ku_PATP — оновити
тут синхронно (стабільні алгебраїчні факти, рідко змінюються).
-/
