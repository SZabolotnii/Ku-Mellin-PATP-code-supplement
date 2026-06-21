"""
Real-data case study — PATP-MUET vs polynomial MUET on a REAL measured input
distribution (for the MATCOM / computational-statistics journal version).

WHY THIS EXISTS
---------------
The arXiv v2 benchmark (`rq3_demo.py`) propagates moments through an *analytic*
input, X ~ Uniform[0,1] (M_X(s)=1/s). A Q1 simulation/computational-statistics
reviewer will ask for a demonstration on a REAL, non-textbook input distribution
whose Mellin transform has no closed form. This script supplies exactly that:

  * Input X  — REAL measured data: windowed vibration POWER (mean-square of the
    CWRU drive-end accelerometer signal), min-max normalised to (0,1].
    A genuinely non-Gaussian, positively-skewed empirical distribution; its
    Mellin transform M_X(s)=E[X^{s-1}] is replaced by the *empirical* Mellin
    transform  M̂_X(s) = (1/N) Σ_n X_n^{s-1}  (integer AND off-lattice s).

  * Response surface Y=g(X) — standard fractional-power MEASUREMENT models from
    the amplitude<->power family (amplitude ∝ √power is the canonical √-law in
    vibration metrology / true-RMS instrumentation), plus an honest integer
    counter-case and a hard log case:
        C1 (headline, non-trivial): Y = √X·(1+0.4 X)  — √-dominant with a mild
            power-dependent nonlinearity (sensor crest/saturation correction).
            Neither basis contains it exactly.
        C2 (reference, near-pure √):  Y = √X            — amplitude-from-power.
        C3 (honest counter, integer): Y = X^2          — polynomial basis exact.
        C4 (hard, outside both):      Y = ln(1+3X).

GROUND TRUTH is the direct empirical push-forward  Ê[Y^j] = (1/N) Σ_n g(X_n)^j
(the real answer: every measured sample pushed through the physical law).

FAIR BUDGET / NO LEAKAGE
------------------------
Both methods receive only the SAME moment budget of X (S=3): polynomial uses the
integer empirical moments Ê[X^k]=M̂_X(k+1); PATP additionally uses off-lattice
empirical moments Ê[X^{p_i(α)k}]=M̂_X(p_i(α)k+1) — the structural advantage of the
fractional PATP basis, costing one extra scalar α, not extra data. The α-optimiser
minimises only the L² fit residual of the surrogate to g on the measured support;
it never sees the ground-truth output moments.

BOOTSTRAP
---------
Because ground truth now comes from a finite real sample, we bootstrap the whole
pipeline (resample the N measured windows with replacement, B times) and report a
95% CI on the per-moment improvement — the real-data robustness check flagged as
future work in Results-RQ3.md.

Run:
  .venv/bin/python realdata_case.py [--mat <path>] [--window 256] [--boot 300]
"""

from __future__ import annotations

import argparse
import os
import sys
from math import factorial
from itertools import product

import numpy as np

try:
    import scipy.io as sio
except Exception as exc:  # pragma: no cover
    print(f"need scipy: {exc}", file=sys.stderr)
    raise


# ---------------------------------------------------------------------------
# PATP exponent (Derivation-A1) and the §6 closed-form, generalised to an
# ARBITRARY (here empirical) Mellin transform mx(s) = E[X^{s-1}].
# ---------------------------------------------------------------------------

def p_num(i: int, alpha: float) -> float:
    return 1.0 / i + (4 - i - 3.0 / i) * alpha + (2 * i - 4 + 2.0 / i) * alpha ** 2


def _kappas(m: int, n: int):
    for combo in product(range(m + 1), repeat=n):
        if sum(combo) == m:
            yield combo


def patp_moment(k0: float, ks: dict, alpha: float, j: int, mx) -> float:
    """j-th moment of g(X;α)=k0+Σ_i k_i·X^{p_i(α)} on positive support, via §6,
    with M_X queried through the callable mx(s) (off-lattice allowed)."""
    idx = sorted(ks.keys())
    pv = {i: p_num(i, alpha) for i in idx}
    out = 0.0
    for m in range(j + 1):
        binom = factorial(j) // (factorial(m) * factorial(j - m))
        inner = 0.0
        for kappa in _kappas(m, len(idx)):
            cm = factorial(m)
            prod_k = 1.0
            s_arg = 1.0
            for pos, i in enumerate(idx):
                ke = kappa[pos]
                if ke == 0:
                    continue
                cm //= factorial(ke)
                prod_k *= ks[i] ** ke
                s_arg += ke * pv[i]
            inner += cm * prod_k * mx(s_arg)
        out += binom * k0 ** (j - m) * inner
    return out


def polynomial_moment(coefs: np.ndarray, j: int, mx) -> float:
    """j-th moment of p(X)=Σ_{i=0}^{S} a_i X^i, M_X via mx(s); Ê[X^k]=mx(k+1)."""
    S = len(coefs) - 1
    total = 0.0
    for kappa in _kappas(j, S + 1):
        cm = factorial(j)
        prod_a = 1.0
        xp = 0
        for i, ke in enumerate(kappa):
            if ke == 0:
                continue
            cm //= factorial(ke)
            prod_a *= coefs[i] ** ke
            xp += i * ke
        total += cm * prod_a * mx(xp + 1)   # E[X^{xp}] = M_X(xp+1)
    return total


# ---------------------------------------------------------------------------
# Surrogate fitting (LS on the measured support; no ground-truth leakage)
# ---------------------------------------------------------------------------

def fit_polynomial(f, S, x):
    A = np.vander(x, S + 1, increasing=True)
    c, *_ = np.linalg.lstsq(A, f(x), rcond=None)
    return c


def fit_patp(f, alpha, x, indices):
    cols = [np.ones_like(x)] + [np.abs(x) ** p_num(i, alpha) for i in indices]
    A = np.column_stack(cols)
    c, *_ = np.linalg.lstsq(A, f(x), rcond=None)
    return {"k0": float(c[0]), "ks": {indices[k]: float(c[k + 1]) for k in range(len(indices))}}


def patp_fit_loss(alpha, f, x, indices):
    par = fit_patp(f, alpha, x, indices)
    pred = par["k0"] + sum(par["ks"][i] * np.abs(x) ** p_num(i, alpha) for i in indices)
    return float(np.mean((pred - f(x)) ** 2))


def opt_alpha(f, x, indices):
    # coarse grid + local refine (avoids importing scipy.optimize; deterministic)
    grid = np.linspace(0.0, 1.0, 201)
    losses = [patp_fit_loss(a, f, x, indices) for a in grid]
    a0 = grid[int(np.argmin(losses))]
    lo, hi = max(0.0, a0 - 0.01), min(1.0, a0 + 0.01)
    fine = np.linspace(lo, hi, 201)
    a1 = fine[int(np.argmin([patp_fit_loss(a, f, x, indices) for a in fine]))]
    return float(a1)


# Surrogate calibration grid: uniform over the (normalised) operating range [0,1].
# Decoupled from the propagation measure so EVERY moment j=1..4 is informative
# (fitting on the propagation sample would force j=1 exact for both methods by
# least-squares orthogonality of the intercept — a trivial artefact, not a result).
CAL_GRID = np.linspace(1e-6, 1.0, 401)


def calibrate(f, indices=(2, 3, 4), S_poly=3):
    """Calibrate both surrogates to the physical law f over the operating range
    (grid LS); returns the frozen surrogates + α*. No input-sample / ground-truth
    information enters here."""
    indices = list(indices)
    poly_c = fit_polynomial(f, S_poly, CAL_GRID)
    a_star = opt_alpha(f, CAL_GRID, indices)
    patp = fit_patp(f, a_star, CAL_GRID, indices)
    return dict(poly=poly_c, alpha=a_star, patp=patp, indices=indices)


# ---------------------------------------------------------------------------
# Real measured input  X  (windowed vibration power, min-max normalised)
# ---------------------------------------------------------------------------

def load_real_input(mat_path, window):
    m = sio.loadmat(mat_path)
    de = [k for k in m if k.endswith("DE_time")]
    if not de:
        raise SystemExit(f"no *_DE_time variable in {mat_path}: keys={list(m)}")
    sig = m[de[0]].ravel().astype(float)
    n = len(sig) // window
    seg = sig[:n * window].reshape(n, window)
    ms = (seg ** 2).mean(axis=1)            # mean-square = vibration power per window
    x = (ms - ms.min()) / (ms.max() - ms.min())   # min-max -> support [0,1]
    # nudge the single exact-zero off 0 so fractional powers are finite for all n
    x = np.clip(x, 1e-9, 1.0)
    return x, de[0], len(sig)


# response surfaces (real measurement models)
SURFACES = [
    ("C1 √-dominant hybrid  Y=√X·(1+0.4X)   [non-trivial]", lambda x: np.sqrt(x) * (1 + 0.4 * x)),
    ("C2 amplitude<-power   Y=√X            [near-pure √]",  lambda x: np.sqrt(x)),
    ("C3 quadratic transd.  Y=X^2           [integer counter]", lambda x: x ** 2),
    ("C4 log compression    Y=ln(1+3X)      [hard, outside both]", lambda x: np.log(1 + 3 * x)),
]


def emp_mx_factory(x):
    """Empirical Mellin transform  M̂_X(s)=mean(X^{s-1})  for X>0."""
    return lambda s: float(np.mean(x ** (s - 1.0)))


def evaluate(x, f, surr):
    """Given a frozen surrogate `surr` (calibrated to f over the range), propagate
    its moments through the EMPIRICAL Mellin transform of the real sample x and
    compare to the empirical push-forward ground truth Ê[f(X)^j]."""
    mx = emp_mx_factory(x)
    poly_c, a_star, patp = surr["poly"], surr["alpha"], surr["patp"]
    out = {"alpha": a_star, "j": {}}
    for j in (1, 2, 3, 4):
        truth = float(np.mean(f(x) ** j))                       # push-forward GT
        pe = polynomial_moment(poly_c, j, mx)
        ke = patp_moment(patp["k0"], patp["ks"], a_star, j, mx)
        rp = abs(pe - truth) / max(abs(truth), 1e-15)
        rk = abs(ke - truth) / max(abs(truth), 1e-15)
        out["j"][j] = dict(truth=truth, poly=pe, patp=ke, rel_poly=rp, rel_patp=rk,
                           impr=(rp - rk) / max(rp, 1e-15) * 100.0)
    # extended uncertainty U = 3σ
    mu_t, m2_t = np.mean(f(x)), np.mean(f(x) ** 2)
    out["U_true"] = 3.0 * np.sqrt(max(m2_t - mu_t ** 2, 0.0))
    mp = polynomial_moment(poly_c, 1, mx); vp = polynomial_moment(poly_c, 2, mx) - mp ** 2
    mk = patp_moment(patp["k0"], patp["ks"], a_star, 1, mx)
    vk = patp_moment(patp["k0"], patp["ks"], a_star, 2, mx) - mk ** 2
    out["U_poly"] = 3.0 * np.sqrt(max(vp, 0.0))
    out["U_patp"] = 3.0 * np.sqrt(max(vk, 0.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    # CWRU file expected in ../data/ (not bundled — see data/README.md for the
    # one-line download instructions for 105.mat / 100.mat).
    default_mat = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data", "105.mat")
    ap.add_argument("--mat", default=default_mat)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if not os.path.exists(args.mat):
        print(f"Input file not found: {args.mat}\n"
              f"The CWRU bearing data is NOT bundled (size + source licence). "
              f"See data/README.md to download 105.mat (and 100.mat) into the data/ folder, "
              f"or pass --mat <path>.", file=sys.stderr)
        return 2

    x, key, nsig = load_real_input(args.mat, args.window)
    sk = float(((x - x.mean()) ** 3).mean() / x.std() ** 3)
    print(f"REAL INPUT  {os.path.basename(args.mat)}::{key}  raw_samples={nsig}")
    print(f"  windowed vibration power, min-max normalised -> N={len(x)} windows on (0,1]")
    print(f"  mean={x.mean():.4f}  sd={x.std():.4f}  skew(Fisher)={sk:+.3f}  "
          f"min={x.min():.2e}  q05={np.quantile(x,0.05):.3f}  max={x.max():.3f}")
    print(f"  empirical M_X is NON-analytic (heavy/skewed); both methods share the same S=3 moment budget\n")

    rng = np.random.default_rng(args.seed)
    N = len(x)

    print(f"{'surface':<48}{'α*':>7}  {'j':>2}  {'rel_poly':>9}  {'rel_patp':>9}  {'impr%':>7}  {'95% CI impr%':>18}")
    summary = []
    for name, f in SURFACES:
        surr = calibrate(f)                       # frozen surrogate (range-calibrated)
        base = evaluate(x, f, surr)
        # bootstrap the real-input sampling uncertainty (surrogate held fixed)
        boot = {j: [] for j in (1, 2, 3, 4)}
        for _ in range(args.boot):
            xb = x[rng.integers(0, N, N)]
            rb = evaluate(xb, f, surr)
            for j in (1, 2, 3, 4):
                boot[j].append(rb["j"][j]["impr"])
        first = True
        avg_impr = float(np.mean([base["j"][j]["impr"] for j in (1, 2, 3, 4)]))
        for j in (1, 2, 3, 4):
            r = base["j"][j]
            lo, hi = np.percentile(boot[j], [2.5, 97.5])
            head = f"{name:<48}{base['alpha']:>7.3f}" if first else " " * 55
            print(f"{head}  {j:>2}  {r['rel_poly']:>9.2e}  {r['rel_patp']:>9.2e}  "
                  f"{r['impr']:>+6.1f}  [{lo:>+6.1f}, {hi:>+6.1f}]")
            first = False
        print(f"{'  -> U=3σ: true=%.4f poly=%.4f patp=%.4f  |ΔU|poly=%.2e |ΔU|patp=%.2e' % (base['U_true'], base['U_poly'], base['U_patp'], abs(base['U_poly']-base['U_true']), abs(base['U_patp']-base['U_true']))}")
        print(f"{'  -> avg improvement (j=1..4): %+.1f%%' % avg_impr}\n")
        # machine-precision floor: if BOTH methods are exact (both bases contain f),
        # the improvement ratio is meaningless noise -> report parity, not a "loss".
        worst = max(max(base["j"][j]["rel_poly"], base["j"][j]["rel_patp"]) for j in (1, 2, 3, 4))
        summary.append((name, base['alpha'], avg_impr, worst))

    print("=" * 78)
    print("SUMMARY (real CWRU input, push-forward ground truth, B=%d bootstrap)" % args.boot)
    print("=" * 78)
    for name, a, ai, worst in summary:
        if worst < 1e-9:
            verdict = "parity (both bases exact, machine precision)"
        elif ai > 15:
            verdict = "PATP wins"
        elif ai > -15:
            verdict = "≈ parity"
        else:
            verdict = "POLY wins (honest)"
        print(f"  {name:<48} α*={a:.3f}  avg_impr={ai:>+7.1f}%   {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
