# MELLIN_AUDIT — покриття перетворення Мелліна в Mathlib4

**Дата.** 2026-05-17
**Етап PLAN.md.** 3 (Ku-Mellin-PATP/PLAN.md)
**Mathlib pin.** `v4.26.0` (з `lake-manifest.json`, rev `2df2f015…`)
**Джерело.** `Mathlib.Analysis.MellinTransform` (модуль mathlib4_docs)

## Вердикт

**(A) — повне покриття.** Mathlib має означення `mellin`, теореми збіжності (одно- і двостороннє), голоморфність на фундаментальній смузі та — критично — `mellin_comp_rpow`, з якого T3 виводиться короткою композицією. Lean-обсяг для T3+T4 (Етап 4 PLAN.md) **розблоковано**.

## Перелік релевантних лем

### Означення

| Mathlib | Семантика |
|---|---|
| `mellin` | `mellin f s = ∫_{(0,∞)} t^{s−1} • f t` (`s : ℂ`) |
| `mellinInv` | Обернене перетворення Мелліна |

### Збіжність / інтегрованість

| Mathlib | Зміст |
|---|---|
| `MellinConvergent` | Предикат «інтеграл коректно визначений» |
| `mellin_convergent_iff_norm` | Векторне ↔ скалярне |
| `mellin_convergent_top_of_isBigO` | `f = O(x^{−a})` на `+∞`, `s.re < a` ⇒ збіжність на хвості |
| `mellin_convergent_zero_of_isBigO` | `f = O(x^{−b})` біля 0, `b < s.re` ⇒ збіжність біля нуля |
| `mellin_convergent_of_isBigO_scalar` | Комбінація: степеневі оцінки в обох кінцях |
| `mellinConvergent_of_isBigO_rpow` | Векторна версія для `b < s.re < a` |
| `mellinConvergent_of_isBigO_rpow_exp` | Експоненційне спадання на нескінченності |

### Перетворення (ключове для T3)

| Mathlib | Формула |
|---|---|
| **`mellin_comp_rpow`** | `mellin (fun t => f (t^a)) s = \|a\|⁻¹ • mellin f (s / a)` |
| `mellin_cpow_smul` | `mellin (fun t => t^a • f t) s = mellin f (s + a)` |
| `mellin_comp_mul_left`, `mellin_comp_mul_right`, `mellin_comp_inv` | Зсуви/інверсія аргументу |

### Голоморфність на смузі

| Mathlib | Зміст |
|---|---|
| `mellin_differentiableAt_of_isBigO_rpow` | Голоморфність на `b < s.re < a` |
| `mellin_differentiableAt_of_isBigO_rpow_exp` | Те саме для експоненційного спадання |

## Як T3 виводиться з `mellin_comp_rpow`

Постановка PATP-MUET (Derivation-A1 §3): для `c > 0` і `Y = X^c` з густиною `f_X` на `(0, ∞)`,
```
M_{f_Y}(s) = M_{f_X}(c·(s − 1) + 1).
```
де `f_Y(y) = f_X(y^{1/c}) · (1/c) · y^{1/c−1}` — формула щільності образу. Розпис:

```
∫₀^∞ y^{s−1} · f_X(y^{1/c}) · (1/c) · y^{1/c−1} dy
  [підстановка u = y^{1/c}, y = u^c, dy = c·u^{c−1} du]
= ∫₀^∞ u^{c(s−1)} · f_X(u) · (1/c) · u^{1−c} · c · u^{c−1} du
= ∫₀^∞ u^{(c(s−1)+1)−1} · f_X(u) du
= mellin f_X (c(s−1)+1). ∎
```

У Mathlib-нотації це **подвоєний крок**: 

1. Покажемо, що `f_Y t = (1/c) · t^{1/c − 1} · f_X(t^{1/c})`.
2. Застосуємо `mellin_cpow_smul` до `t^{1/c−1}`-множника.
3. Застосуємо `mellin_comp_rpow` до `f_X(t^{1/c})` з `a = 1/c`.
4. Зведемо алгебру `s + (1/c − 1)` поділене на `1/c` → `c(s−1)+1`; `(1/c) · |1/c|⁻¹` для `c>0` дає `1` (Якобіан скорочується).

Доведення в Lean — порядку 30–50 рядків, без `sorry`.

## Рекомендація для Етапу 4 PLAN.md

- **T3 (`mellin_power_rule`)** — реалізувати як `PATP/MellinPower.lean`. Базис: `mellin_comp_rpow` + `mellin_cpow_smul`. Складність — алгебра комплексних степенів і `(1/c)·|1/c|⁻¹ = 1`. Очікувано закривається `simp [mellin_comp_rpow, mellin_cpow_smul]; field_simp; ring_nf` після підготовки рівностей типів.
- **T4 (`patp_alpha_one_reduction`)** — реалізувати як `PATP/AlphaOneReduction.lean`. Базується на `piAlpha_at_one` (вже доведено) і §6-формулі (потрібно ввести `def patpMomentSum`). Алгебраїчна перевірка.

## Що з обмеженнями

- `mellin` визначене для `f : ℝ → E` (скаляр або банахів простір) на `Ioi 0` — додатна підтримка. **Знаковий випадок** (`Derivation-A1` §4.2) потребує двосторонньої пари `(M_{|X|}, M_{sgn·X})`. У Mathlib цього як готового конструктора **немає** — треба або означити поверх `mellin` (`mellin (fun t => f t + f (-t))` тощо), або зробити окремий `bilateralMellin`. **Це не блокує T3**: T3 формулюється для `c > 0` на `Ioi 0`; знакову розширення кладеться в окремий розділ (Етап 4, опційно).
- `mellin` має сигнатуру `s : ℂ`; наше PATP-формулювання — `s : ℝ`. Потрібен явний coercion `(s : ℂ)` у Lean-формулюваннях.
- `MeasureTheory.IsLocallyIntegrable` / `IntegrableOn` — стандартні передумови; для обмежених вимірюваних розподілів (uniform, beta, triangular, truncated normal) перевіряються одноразово.

## Стан задачі

| T-ціль | Стан після аудиту |
|---|---|
| T1 | ✓ доведено (`PATP/Param.lean`, `piAlpha_at_zero/half/one`) |
| T2 pos | ✓ доведено (`PATP/Positivity.lean`, `piAlpha_pos_{two,three,four,five}`) |
| T2 neg | ✓ доведено (`PATP/Positivity.lean`, `piAlpha_six_negative_witness`) |
| T3     | готове до реалізації, А-гілка; план — вище |
| T4     | готове до реалізації, після T3 і `def patpMomentSum` |

## Блокер для запуску

`elan` + `lake` локально не встановлено (див. `Ku-Mellin-PATP/ENV.md`). Сам аудит — read-only через документи Mathlib. Перший локальний `lake build` (з кешем mathlib) — 5–10 хв; повне холодне будування Mathlib — 1–2 год.
