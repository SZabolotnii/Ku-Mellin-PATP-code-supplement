"""
realdata_split.py -- out-of-sample real-data case study for the PEM revision
(PREM-D-26-00488, task R). It replaces the in-sample design of realdata_case.py,
which is kept unchanged as the record of the submitted version.

===== BEGIN PRE-REGISTRATION (fixed 2026-09-26, before the first run that computes any method error) =====

WHY. Reviewer 2 (Q2, Q3, Q5, Q7): the submitted case study used one vibration sample both to
estimate the input Mellin transform and to build the output reference, so it was in-sample; the
reviewer asks for a split or an independent record, the full bootstrap protocol, sample sizes,
preprocessing, intervals for every moment order, and whether the responses are calibrated
physical models. Reviewer 1 (minor 4): justify U = 3 sigma or drop it. A further point from our
own analysis (ANALYSIS_2026-09-26.md): with the empirical Mellin transform
M_hat(s) = mean(X^(s-1)), the closed form of PATP-MUET is a finite sum and equals
(1/N) sum_n g_hat(X_n)^j exactly. A surrogate route can therefore only ADD surrogate error to the
plug-in estimate (1/N) sum_n g(X_n)^j with the true g. When g is known and cheap, that plug-in
(called ORACLE here) is the estimator to use; a surrogate is relevant only when g is expensive.
This script measures how much each surrogate adds to the unavoidable sampling error when the
moments of a HELD-OUT segment are predicted.

DATA. CWRU Bearing Data Center, "12k Drive End Bearing Fault Data", 12,000 samples/s.
  105.mat: inner-race fault 0.007 in, motor load 0 hp, approx. 1797 rpm (file variable
           X105RPM = 1797); channel X105_DE_time (drive-end accelerometer), 121,265 samples.
  106.mat: same fault, load 1 hp, approx. 1772 rpm (X106RPM = 1772); X106_DE_time, 121,991 samples.
  sha256 of both files is printed and must match ~/Project/datasets/store/cwru-bearing/MANIFEST.sha256.

PREPROCESSING. No filtering, no detrending, no DC removal (the channel mean of 105 is 0.013 and
its square is 0.2 % of the mean window power). The signal is cut into consecutive NON-OVERLAPPING
windows of W = 256 samples (21.3 ms); the incomplete tail is dropped (177 samples for 105). This
gives N = 473 windows for 105 and N = 476 for 106. Per-window power P_n = mean of the squared
samples of window n.

PRIMARY VALIDATION (temporal split, record 105). TRAIN = windows 0..235 (the first
n_tr = N // 2 = 236), TEST = windows 236..472 (n_te = 237). The normalisation constants come
from TRAIN only: X = (P - min_train P) / (max_train P - min_train P). Train values lie in [0, 1]
and the train minimum maps to exactly 0, with the conventions 0^0 := 1, 0^p := 0 for p > 0 and
0^a ln^b 0 := 0 for a > 0. TEST values are mapped with the same constants and may leave [0, 1].
Values above 1 are kept. Values below 0 are power below the train minimum; the input-only check
found 3 such windows in the full-data split. They are set to 0 before the response is applied,
so the reference uses g(max(X, 0)), because sqrt and the fractional bases are undefined for
negative arguments. The surrogates are never evaluated at test points: every prediction uses the
train sample only, through its empirical Mellin transform. The out-of-range rule therefore
touches the reference only.

SECONDARY VALIDATION (cross-record, distribution shift). TRAIN = all 473 windows of 105, with the
constants taken from 105. TEST = all 476 windows of 106, mapped with 105's constants. This is
reported as a robustness check under a change of motor load, not as the headline.

RESPONSES. These are illustrative measurement-model transformations on the normalised power
scale, NOT calibrated physical models:
  g1 = sqrt(x) (1 + 0.4 x),  g2 = sqrt(x),  g3 = x^2,  g4 = ln(1 + 3 x).
Amplitude proportional to sqrt(power) is physical. The factor (1 + 0.4 x), the quadratic and the
log compression are illustrative.

SURROGATES. Each is fitted by ordinary least squares on a design grid of 401 equispaced points on
[0, 1], which is the train operating range after normalisation. g is evaluated only at the grid
points, and no fit sees the test data or any moment.
  poly3   {1, x, x^2, x^3}                                  4 parameters
  poly4   {1, ..., x^4}                                     5 parameters (parameter-matched)
  PATP    {1, x^p2(a), x^p3(a), x^p4(a)} with
          p_i(a) = 1/i + (4 - i - 3/i) a + (2i - 4 + 2/i) a^2;
          a in [0, 1] minimises the grid residual: 101-point grid in a, then bounded Brent
          (scipy minimize_scalar) between the grid neighbours of the best point,
          xatol = 1e-10                                     5 parameters
  CONF    {1, x^c, x^c ln x, x^c ln^2 x}, the confluent (a -> 1/2) limit of the PATP span;
          c in [0.05, 3] by a 60-point grid and then Brent, as for a        5 parameters.
          It is propagated through derivatives of the same transform:
          E[X^a ln^b X] = M^(b)(a + 1), estimated by mean(X^a (ln X)^b).
  ORACLE  plug-in (1/n_tr) sum g(X_n)^j with the TRUE g on the train sample. This is the
          sampling-error floor: every surrogate prediction equals the oracle plus that
          surrogate's deviation.

PROPAGATION. For j = 1..4, m_hat_j = E_hat_train[g_hat(X)^j] through the train empirical Mellin
transform M_hat(s) = mean_train X^(s-1). The argument s is an integer for poly,
s = sum_i kappa_i p_i(a) + 1 for PATP, and a derivative of M_hat for CONF. On the original data
each closed form is evaluated in 50-digit arithmetic (mpmath) and compared with the direct finite
sum mean_train g_hat(X)^j, which it must equal; the script prints this identity check. Inside
bootstrap replicates the script uses the direct float64 finite sum. It is identical by that
identity and numerically stable, whereas the float64 closed-form sum cancels near a = 1/2.

REFERENCE. Test empirical moments r_j = mean_test g(max(X, 0))^j with the true g.

METRICS.
  e_{m,j} = |m_hat_{m,j} - r_j| / r_j, the out-of-sample relative moment error; e_bar_m is its
    mean over j = 1..4.
  sigma = sqrt(m2 - m1^2), and the GUM expanded uncertainty U = k sigma with k = 2
    (JCGM 100:2008, Sect. 6.2-6.3 and G.6.6: approx. 95 % coverage for an approximately normal
    output). The relative error of U equals that of sigma. U = 3 sigma from the submitted version
    is dropped, because no coverage argument for k = 3 applies to these outputs.
  dev_{m,j} = |m_hat_{m,j} - m_hat_{ORACLE,j}| / r_j, the surrogate's own error under the train
    empirical measure; dev_bar_m is its mean over j.
  The grid L2 error of each surrogate is also reported.

UNCERTAINTY. Moving-block bootstrap (Kunsch 1989) over windows, with block length L = 22,
B = 1000 replicates and seed 20260926 (analysis k uses default_rng(20260926 + k): 0 = primary,
1 = S1, 2/3/4 = S2 with L = 1/11/44, 5 = S3). Resampling is stratified by the split. Each
replicate draws ceil(n / L) blocks of L consecutive windows, with uniform start positions, from
the TRAIN segment and, independently, from the TEST segment, then concatenates and truncates to
n_tr and n_te. No original window can appear in both segments of one replicate. Inside every
replicate the WHOLE pipeline is repeated: normalisation constants from the replicate's train
windows, surrogate fits, the a and c optimisation, the empirical Mellin moments, the reference
and the metrics. Under min-max normalisation the design grid is always [0, 1] in normalised
units, so the fitted surrogates are invariant across replicates by construction; the script
checks this and prints the range of a* and c*. Variability then enters through the train
empirical Mellin transform and the test reference. Intervals are percentile 95 % intervals (the
2.5 and 97.5 percentiles of the B replicate values), reported for every method, response and j,
for U, and for the decision statistics.
Block-length justification. The following input-only diagnostics were run before registration;
no response or surrogate was involved, and the script reprints them.
  - The window-power series of 105 is phase-locked to shaft rotation. On the train half its ACF
    is -0.41 at lag 1 and peaks at lag 11 (0.64). Lag 11 is where the window start phase
    realigns with the shaft: 11 x 256 samples = 7.03 revolutions at 1797 rpm and 12 kHz.
  - The ACF does not decay within 66 lags (peaks 0.61 at lag 47 and 0.53 at lag 58).
  - L = 22 covers two realignment periods. It also matches the Politis-White (2004;
    Patton-Politis-White 2009) automatic block length for the moving-block bootstrap, which is
    21.4 on the train half (14.7 on the test half).
  - The dominant short-lag correlations are negative, so the Bartlett long-run-variance ratio of
    the train series is 0.28-0.35 (truncation 11-66). An iid bootstrap would therefore OVERSTATE
    the variability of window means here, not understate it. The block bootstrap is used because
    it is the valid choice for a dependent series, whichever way the correction goes.

DECISION RULE (temporal split only). For each response and each baseline b in {poly3, poly4}:
  D_b = e_bar_b - e_bar_PATP (out-of-sample error difference) and
  Ddev_b = dev_bar_b - dev_bar_PATP (surrogate-deviation difference).
The paper may state that PATP-MUET REDUCES the out-of-sample moment error relative to poly3 and
poly4 for a response ONLY IF, for BOTH baselines,
  (A) the 95 % percentile interval of D_b lies entirely above tau = 1e-9, and
  (B) the 95 % percentile interval of Ddev_b lies entirely above tau.
Condition A is the requested rule. Condition B is a further necessary condition added before the
run: it stops a reduction that comes from a baseline bias which happens to offset the shared
train-test shift from being credited to PATP. tau is a numerical-parity floor, so that
differences at rounding level (the exact-basis case g3 = x^2) cannot pass.
"Baseline b better" requires both intervals to lie entirely below -tau. Anything else is reported
as PARITY for that response. The oracle's errors are reported as the floor, and the text must say
that when g is cheap and known the plug-in is preferable. Per-j intervals of D_b are reported
descriptively and do not change the verdict. D_CONF = e_bar_CONF - e_bar_PATP is also reported
descriptively; it tests the step-0 finding that PATP-opt near a = 1/2 behaves like the confluent
x^c ln^b x basis.

SECONDARY ANALYSES (pre-registered; reported, but they do not enter the verdict).
  S1  cross-record 105 -> 106 as above: same metrics, B = 1000, L = 22.
  S2  block-length sensitivity on the primary split with L in {1 (iid), 11, 44}, B = 1000 each:
      intervals of D_b and Ddev_b and the verdict each L would give.
  S3  normalisation sensitivity on the primary split: scale-only X = P / max_train P (no offset,
      so X > 0 and sqrt(X) is the physical amplitude ratio for g2). The design grid is
      [min_train X, 1], the train operating range. L = 22, B = 1000, all metrics and D_b.

FIGURE. paper/fig_realdata.pdf, written from the primary point estimate:
  (a) the empirical input distribution (train histogram, test outline);
  (b) the fitted poly3 and PATP surfaces against the true g1 over the observed range;
  (c) their residuals, with poly4.

===== END PRE-REGISTRATION =====

CHANGELOG after registration: bug fixes only, none of which changes the design.
  2026-09-26, after the first run: figure styling only. Computer Modern fonts; panel (c) shows
  |g_hat - g1| on a log scale, because a linear axis was dominated by the poly3 residual at
  x = 0; legends moved off the data. No statistic changes, and the pre-registered part of the
  output is byte-identical to the first run's (checked by diff).

POST-HOC: no post-hoc analysis at registration time. Any later addition goes below this line,
marked POST-HOC, and does not alter the pre-registered part above, whose sha256 is printed at
run time.
  POST-HOC 1 (added 2026-09-26 after the first run; descriptive, does not change the verdict).
  The percentile intervals of the primary split have long upper tails, the oracle floor
  included. The hypothesis checked: min-max normalisation is anchored on the two most extreme
  train windows, so a replicate that omits the minimum window shifts the scale and pushes test
  windows below 0, where they are clipped. The block prints how often clipping happens and the
  oracle's error with and without it. It reuses the stored primary replicates and draws no new
  random numbers. Outcome (see the output): clipping occurs in almost every replicate, and the
  oracle's error distribution is about the same with and without it. The hypothesis is
  therefore NOT supported: the tails are the ordinary variability of a difference between two
  segment moments, which grows with j.
  TABLE ROWS (added 2026-09-26 after the first run; formatting only). They print the
  pre-registered statistics as LaTeX rows for revision/drafts/realdata.tex, so that every number
  in the draft is copied from this output rather than rounded by hand.

Run (from experiments/):
  ../verification/cas/.venv/bin/python realdata_split.py | tee results/realdata_split.txt
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import sys
from itertools import combinations_with_replacement
from math import comb, factorial

import matplotlib
import mpmath as mp
import numpy as np
import scipy
import scipy.io as sio
from scipy.optimize import minimize_scalar

mp.mp.dps = 50

# supplement: data from ../data/ (or $CWRU_DATA_DIR), hashes from the committed data/MANIFEST.sha256, figure to
# ../outputs/ (relative path, so the printed "figure written:" line carries no machine-specific directory)
_SUPP = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_DIR = os.environ.get("CWRU_DATA_DIR", os.path.join(_SUPP, "data"))
MANIFEST = os.path.join(_SUPP, "data", "MANIFEST.sha256")
FIG_DEFAULT = os.path.relpath(os.path.join(_SUPP, "outputs", "fig_realdata.pdf"))

W = 256
J = (1, 2, 3, 4)
N_GRID = 401
L_BLOCK = 22
B_DEFAULT = 1000
SEED = 20260926
TAU = 1e-9
K_COV = 2.0
BASELINES = ("poly3", "poly4")
METHODS = ("poly3", "poly4", "PATP", "CONF", "ORACLE")
N_PAR = {"poly3": 4, "poly4": 5, "PATP": 5, "CONF": 5, "ORACLE": 0}

RESPONSES = [
    ("g1", "sqrt(x)(1+0.4x)", lambda x: np.sqrt(x) * (1.0 + 0.4 * x)),
    ("g2", "sqrt(x)", lambda x: np.sqrt(x)),
    ("g3", "x^2", lambda x: x ** 2),
    ("g4", "ln(1+3x)", lambda x: np.log1p(3.0 * x)),
]


# ---------------------------------------------------------------------------
# Bases (x >= 0; 0^0 = 1, 0^p = 0 for p > 0, 0^a ln^b 0 = 0 for a > 0)
# ---------------------------------------------------------------------------

def p_num(i: int, a: float) -> float:
    return 1.0 / i + (4 - i - 3.0 / i) * a + (2 * i - 4 + 2.0 / i) * a ** 2


def p_mp(i: int, a) -> mp.mpf:
    a = mp.mpf(a)
    return mp.mpf(1) / i + (4 - i - mp.mpf(3) / i) * a + (2 * i - 4 + mp.mpf(2) / i) * a ** 2


def xlogpow(x: np.ndarray, c: float, b: int) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    pos = x > 0
    out[pos] = x[pos] ** c * np.log(x[pos]) ** b
    if b == 0 and c == 0:
        out[~pos] = 1.0
    return out


def basis(kind: str, x: np.ndarray, param) -> np.ndarray:
    if kind == "poly":
        return np.column_stack([x ** k for k in range(int(param) + 1)])
    if kind == "patp":
        return np.column_stack([np.ones_like(x)] + [x ** p_num(i, param) for i in (2, 3, 4)])
    if kind == "conf":
        return np.column_stack([np.ones_like(x)] + [xlogpow(x, param, b) for b in (0, 1, 2)])
    raise ValueError(kind)


class Surrogate:
    def __init__(self, name: str, kind: str, param, coefs: np.ndarray, l2: float):
        self.name, self.kind, self.param, self.coefs, self.l2 = name, kind, param, np.asarray(coefs), l2

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return basis(self.kind, x, self.param) @ self.coefs

    def moments_direct(self, x: np.ndarray) -> np.ndarray:
        v = self(x)
        return np.array([np.mean(v ** j) for j in J])

    # -- closed forms through the empirical Mellin transform --------------
    def power_exps_mp(self):
        if self.kind == "poly":
            return [mp.mpf(k) for k in range(int(self.param) + 1)]
        return [mp.mpf(0)] + [p_mp(i, self.param) for i in (2, 3, 4)]

    def moments_closed_mp(self, x: np.ndarray) -> list:
        xs = [mp.mpf(float(v)) for v in x]
        if self.kind in ("poly", "patp"):
            return [power_sum_moment_mp(self.power_exps_mp(), self.coefs, j, xs) for j in J]
        return [conf_moment_mp(self.param, self.coefs, j, xs) for j in J]

    def moments_closed_f64(self, x: np.ndarray) -> np.ndarray:
        """Same closed form as moments_closed_mp in float64 (to expose cancellation)."""
        exps = [float(e) for e in self.power_exps_mp()]
        out = []
        for j in J:
            tot = 0.0
            for combo in combinations_with_replacement(range(len(exps)), j):
                kap = [combo.count(k) for k in range(len(exps))]
                mult = factorial(j)
                prod, s = 1.0, 1.0
                for k, ke in enumerate(kap):
                    if ke:
                        mult //= factorial(ke)
                        prod *= float(self.coefs[k]) ** ke
                        s += ke * exps[k]
                tot += mult * prod * float(np.mean(x ** (s - 1.0)))
            out.append(tot)
        return np.array(out)


def emp_mellin_mp(xs: list, e) -> mp.mpf:
    """M_hat(e + 1) = mean X^e, with 0^0 = 1 and 0^e = 0 for e > 0."""
    tot = mp.mpf(0)
    for v in xs:
        if v == 0:
            tot += 1 if e == 0 else 0
        else:
            tot += v ** e
    return tot / len(xs)


def emp_mellin_log_mp(xs: list, a, b: int) -> mp.mpf:
    """M_hat^(b)(a + 1) = mean X^a (ln X)^b, with 0^a ln^b 0 = 0 for a > 0."""
    tot = mp.mpf(0)
    for v in xs:
        if v == 0:
            tot += 1 if (a == 0 and b == 0) else 0
        else:
            tot += v ** a * mp.log(v) ** b
    return tot / len(xs)


def power_sum_moment_mp(exps: list, coefs, j: int, xs: list) -> mp.mpf:
    """E[(sum_k c_k X^{e_k})^j] = sum_{|kappa|=j} multinom * prod c^kappa * M_hat(sum kappa e + 1).
    Equivalent to eq. (patp-muet) (binomial in k_0, multinomial in the rest)."""
    n = len(exps)
    total = mp.mpf(0)
    for combo in combinations_with_replacement(range(n), j):
        kap = [combo.count(k) for k in range(n)]
        mult = mp.mpf(factorial(j))
        prod, e = mp.mpf(1), mp.mpf(0)
        for k, ke in enumerate(kap):
            if ke:
                mult /= factorial(ke)
                prod *= mp.mpf(float(coefs[k])) ** ke
                e += ke * exps[k]
        total += mult * prod * emp_mellin_mp(xs, e)
    return total


def conf_moment_mp(c: float, coefs, j: int, xs: list) -> mp.mpf:
    """g = k0 + X^c Q(ln X), Q(L) = k1 + k2 L + k3 L^2:
    E[g^j] = sum_m C(j,m) k0^{j-m} sum_b q_{m,b} M_hat^{(b)}(m c + 1)."""
    k0, k1, k2, k3 = (mp.mpf(float(v)) for v in coefs)
    cm = mp.mpf(c)
    total = mp.mpf(0)
    q = [mp.mpf(1)]                              # Q^0
    for m in range(j + 1):
        if m > 0:                                 # q <- q * (k1 + k2 L + k3 L^2)
            new = [mp.mpf(0)] * (len(q) + 2)
            for b, v in enumerate(q):
                new[b] += v * k1
                new[b + 1] += v * k2
                new[b + 2] += v * k3
            q = new
        inner = mp.mpf(0)
        for b, v in enumerate(q):
            inner += v * emp_mellin_log_mp(xs, m * cm, b)
        total += comb(j, m) * k0 ** (j - m) * inner
    return total


N_MELLIN = {"poly3": 13, "poly4": 17, "PATP": 35, "CONF": 25}   # distinct (argument, derivative) pairs, j <= 4


def n_mellin_check():
    patp = len({tuple(sorted(c)) for jj in range(5) for c in combinations_with_replacement(range(3), jj)})
    conf = sum(2 * m + 1 for m in range(5))
    return {"poly3": 4 * 3 + 1, "poly4": 4 * 4 + 1, "PATP": patp, "CONF": conf}


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def ols(A: np.ndarray, y: np.ndarray):
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c, float(np.sqrt(np.mean((A @ c - y) ** 2)))


def opt_shape(kind: str, grid: np.ndarray, y: np.ndarray, lo: float, hi: float, ngrid: int) -> float:
    def loss(t):
        return ols(basis(kind, grid, t), y)[1]
    cand = np.linspace(lo, hi, ngrid)
    losses = np.array([loss(t) for t in cand])
    i0 = int(np.argmin(losses))
    a, b = cand[max(i0 - 1, 0)], cand[min(i0 + 1, ngrid - 1)]
    res = minimize_scalar(loss, bounds=(a, b), method="bounded", options={"xatol": 1e-10})
    return float(res.x) if res.fun <= losses[i0] else float(cand[i0])


def fit_all(grid: np.ndarray, y: np.ndarray) -> dict:
    out = {}
    for d in (3, 4):
        c, l2 = ols(basis("poly", grid, d), y)
        out[f"poly{d}"] = Surrogate(f"poly{d}", "poly", d, c, l2)
    a = opt_shape("patp", grid, y, 0.0, 1.0, 101)
    c, l2 = ols(basis("patp", grid, a), y)
    out["PATP"] = Surrogate("PATP", "patp", a, c, l2)
    cc = opt_shape("conf", grid, y, 0.05, 3.0, 60)
    c, l2 = ols(basis("conf", grid, cc), y)
    out["CONF"] = Surrogate("CONF", "conf", cc, c, l2)
    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_hash(fname: str) -> str | None:
    try:
        with open(MANIFEST) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) == 2 and parts[1] == fname:
                    return parts[0]
    except OSError:
        return None
    return None


def load_power(rec: int) -> tuple[np.ndarray, dict]:
    path = os.path.join(DATA_DIR, f"{rec}.mat")
    m = sio.loadmat(path)
    key = f"X{rec:03d}_DE_time"
    sig = m[key].ravel().astype(float)
    n = len(sig) // W
    P = (sig[: n * W].reshape(n, W) ** 2).mean(axis=1)
    rpm_key = f"X{rec:03d}RPM"
    meta = dict(rec=rec, key=key, nsamp=len(sig), nwin=n, tail=len(sig) - n * W,
                rpm=float(m[rpm_key].ravel()[0]) if rpm_key in m else None,
                dc=float(sig.mean()), sha=sha256(path), sha_manifest=manifest_hash(f"{rec}.mat"))
    return P, meta


# ---------------------------------------------------------------------------
# Input-only diagnostics (block length)
# ---------------------------------------------------------------------------

def acov(x: np.ndarray, K: int) -> np.ndarray:
    x = x - x.mean()
    n = len(x)
    return np.array([np.dot(x[: n - k], x[k:]) / n for k in range(K + 1)])


def politis_white(x: np.ndarray):
    """Automatic block length (Politis & White 2004, corrected by Patton, Politis & White 2009).
    Returns (m_hat, M, b_stationary, b_circular/moving)."""
    n = len(x)
    R = acov(x, n // 2)
    rho = R / R[0]
    cthr = 2.0 * np.sqrt(np.log10(n) / n)
    KN = max(5, int(np.ceil(np.sqrt(np.log10(n)))))
    mh = None
    for mm in range(1, n // 2 - KN):
        if np.all(np.abs(rho[mm + 1: mm + 1 + KN]) < cthr):
            mh = mm
            break
    if mh is None:
        return None
    M = min(2 * mh, n // 2 - 1)
    k = np.arange(-M, M + 1)
    t = np.abs(k) / M
    lam = np.where(t <= 0.5, 1.0, 2.0 * (1.0 - t))
    Rk = R[np.abs(k)]
    G = np.sum(lam * np.abs(k) * Rk)
    g0 = np.sum(lam * Rk)
    return mh, M, (2 * G ** 2 / (2 * g0 ** 2)) ** (1 / 3) * n ** (1 / 3), \
        (2 * G ** 2 / (4 / 3 * g0 ** 2)) ** (1 / 3) * n ** (1 / 3)


def bartlett_lrv_ratio(x: np.ndarray, K: int) -> float:
    R = acov(x, K)
    rho = R / R[0]
    k = np.arange(1, K + 1)
    return float(1 + 2 * np.sum((1 - k / (K + 1)) * rho[1:]))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def normalise(P_tr: np.ndarray, P_te: np.ndarray, norm: str):
    if norm == "minmax":
        lo, hi = P_tr.min(), P_tr.max()
        return (P_tr - lo) / (hi - lo), (P_te - lo) / (hi - lo), 0.0
    if norm == "scale":
        hi = P_tr.max()
        x_tr = P_tr / hi
        return x_tr, P_te / hi, float(x_tr.min())
    raise ValueError(norm)


def uexp(m: np.ndarray) -> float:
    return K_COV * float(np.sqrt(max(m[1] - m[0] ** 2, 0.0)))


def evaluate(P_tr: np.ndarray, P_te: np.ndarray, norm: str, keep: bool = False):
    x_tr, x_te, g_lo = normalise(P_tr, P_te, norm)
    grid = np.linspace(g_lo, 1.0, N_GRID)
    x_te_c = np.maximum(x_te, 0.0)
    st = {("n_low",): float((x_te < 0).sum()), ("n_high",): float((x_te > 1).sum())}
    kept = {"x_tr": x_tr, "x_te": x_te, "grid": grid, "surr": {}, "pred": {}, "ref": {}}
    for r, _, g in RESPONSES:
        surr = fit_all(grid, g(grid))
        ref = np.array([np.mean(g(x_te_c) ** j) for j in J])
        pred = {m: surr[m].moments_direct(x_tr) for m in surr}
        pred["ORACLE"] = np.array([np.mean(g(x_tr) ** j) for j in J])
        U_ref = uexp(ref)
        st[(r, "REF", "U")] = U_ref
        for jj, j in enumerate(J):
            st[(r, "REF", "m", j)] = ref[jj]
        ebar, dbar = {}, {}
        for m in METHODS:
            e = np.abs(pred[m] - ref) / ref
            dev = np.abs(pred[m] - pred["ORACLE"]) / ref
            for jj, j in enumerate(J):
                st[(r, m, "pred", j)] = pred[m][jj]
                st[(r, m, "e", j)] = e[jj]
                st[(r, m, "dev", j)] = dev[jj]
            ebar[m], dbar[m] = float(e.mean()), float(dev.mean())
            st[(r, m, "ebar")] = ebar[m]
            st[(r, m, "devbar")] = dbar[m]
            U = uexp(pred[m])
            st[(r, m, "U")] = U
            st[(r, m, "eU")] = abs(U - U_ref) / U_ref
            if m in surr:
                st[(r, m, "l2")] = surr[m].l2
                if m in ("PATP", "CONF"):
                    st[(r, m, "param")] = surr[m].param
        for b in BASELINES:
            st[(r, "D", b)] = ebar[b] - ebar["PATP"]
            st[(r, "Ddev", b)] = dbar[b] - dbar["PATP"]
            for j in J:
                st[(r, "Dj", b, j)] = st[(r, b, "e", j)] - st[(r, "PATP", "e", j)]
        st[(r, "D", "CONF")] = ebar["CONF"] - ebar["PATP"]
        st[(r, "D", "ORACLE")] = ebar["ORACLE"] - ebar["PATP"]
        if keep:
            kept["surr"][r], kept["pred"][r], kept["ref"][r] = surr, pred, ref
    return st, kept


def mbb_indices(n: int, L: int, rng: np.random.Generator) -> np.ndarray:
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, nb)
    return (starts[:, None] + np.arange(L)[None, :]).ravel()[:n]


def bootstrap(P_tr, P_te, norm, L, B, seed):
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(B):
        itr = mbb_indices(len(P_tr), L, rng)
        ite = mbb_indices(len(P_te), L, rng)
        reps.append(evaluate(P_tr[itr], P_te[ite], norm)[0])
    return {k: np.array([rep[k] for rep in reps]) for k in reps[0]}


def ci(a: np.ndarray):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def verdict(boot: dict, r: str):
    per = {}
    for b in BASELINES:
        dlo, dhi = ci(boot[(r, "D", b)])
        vlo, vhi = ci(boot[(r, "Ddev", b)])
        if dlo > TAU and vlo > TAU:
            per[b] = "PATP<b"
        elif dhi < -TAU and vhi < -TAU:
            per[b] = "PATP>b"
        else:
            per[b] = "n.s."
    if all(per[b] == "PATP<b" for b in BASELINES):
        v = "PATP REDUCES the out-of-sample error vs poly3 AND poly4"
    elif any(per[b] == "PATP>b" for b in BASELINES):
        v = "BASELINE BETTER: " + ", ".join(b for b in BASELINES if per[b] == "PATP>b")
    else:
        v = "PARITY"
    return v, per


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt_ci(point, arr):
    lo, hi = ci(arr)
    return f"{point:.3e} [{lo:.3e}, {hi:.3e}]"


def report_full(title, st, boot, kept, B, L):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(f"n_tr = {len(kept['x_tr'])}, n_te = {len(kept['x_te'])}; test windows below the train minimum "
          f"(set to 0 for the reference): {int(st[('n_low',)])}; above the train maximum: {int(st[('n_high',)])}")
    print(f"design grid: {N_GRID} points on [{kept['grid'][0]:.4f}, 1]; MBB block L = {L}, B = {B}, percentile 95% CI")
    for r, label, _ in RESPONSES:
        print(f"\n--- {r}: g(x) = {label} ---")
        refs = "  ".join(f"r_{j} = {st[(r, 'REF', 'm', j)]:.6e}" for j in J)
        print(f"  test reference moments: {refs}")
        print(f"  reference U (k=2) = {fmt_ci(st[(r, 'REF', 'U')], boot[(r, 'REF', 'U')])}")
        for m in METHODS:
            extra = ""
            if (r, m, "param") in st:
                pa = boot[(r, m, "param")]
                nm = "alpha*" if m == "PATP" else "c*"
                extra = f"  {nm} = {st[(r, m, 'param')]:.6f} (replicates {pa.min():.6f}..{pa.max():.6f})"
            l2 = f"  grid L2 = {st[(r, m, 'l2')]:.3e}" if (r, m, "l2") in st else ""
            print(f"  [{m}] params = {N_PAR[m]}{l2}{extra}")
            for j in J:
                print(f"      j={j}  pred = {st[(r, m, 'pred', j)]:.6e}   e = {fmt_ci(st[(r, m, 'e', j)], boot[(r, m, 'e', j)])}"
                      f"   dev = {fmt_ci(st[(r, m, 'dev', j)], boot[(r, m, 'dev', j)])}")
            print(f"      mean e (j=1..4) = {fmt_ci(st[(r, m, 'ebar')], boot[(r, m, 'ebar')])}"
                  f"   mean dev = {fmt_ci(st[(r, m, 'devbar')], boot[(r, m, 'devbar')])}")
            print(f"      U (k=2) = {fmt_ci(st[(r, m, 'U')], boot[(r, m, 'U')])}"
                  f"   rel. error of U = {fmt_ci(st[(r, m, 'eU')], boot[(r, m, 'eU')])}")
        print("  decision statistics (baseline minus PATP; > 0 favours PATP):")
        for b in BASELINES:
            print(f"      D_{b:<6} = {fmt_ci(st[(r, 'D', b)], boot[(r, 'D', b)])}"
                  f"   Ddev_{b:<6} = {fmt_ci(st[(r, 'Ddev', b)], boot[(r, 'Ddev', b)])}")
            print("        per j: " + "  ".join(f"j={j}: {fmt_ci(st[(r, 'Dj', b, j)], boot[(r, 'Dj', b, j)])}" for j in J))
        print(f"      D_CONF   = {fmt_ci(st[(r, 'D', 'CONF')], boot[(r, 'D', 'CONF')])}   (CONF minus PATP, descriptive)")
        print(f"      D_ORACLE = {fmt_ci(st[(r, 'D', 'ORACLE')], boot[(r, 'D', 'ORACLE')])}   (ORACLE minus PATP, descriptive)")
        v, per = verdict(boot, r)
        print(f"  rule outcome for this analysis: {v}   (per baseline: {per})")


def closed_form_check(kept):
    print("\nIdentity check on the original data: closed form through the empirical Mellin transform")
    print("(50-digit mpmath) vs the direct finite sum mean_train g_hat(X)^j (float64); max over j of the")
    print("relative discrepancy. PATP float64 closed form shown to expose cancellation.")
    x_tr = kept["x_tr"]
    worst = 0.0
    for r, _, _ in RESPONSES:
        row = []
        for m in ("poly3", "poly4", "PATP", "CONF"):
            s = kept["surr"][r][m]
            cf = s.moments_closed_mp(x_tr)
            dr = s.moments_direct(x_tr)
            d = max(abs(float(cf[k]) - dr[k]) / abs(dr[k]) for k in range(len(J)))
            worst = max(worst, d)
            row.append(f"{m}: {d:.1e}")
        s = kept["surr"][r]["PATP"]
        f64 = s.moments_closed_f64(x_tr)
        dr = s.moments_direct(x_tr)
        d64 = max(abs(f64[k] - dr[k]) / abs(dr[k]) for k in range(len(J)))
        cond = np.linalg.cond(basis("patp", kept["grid"], s.param))
        print(f"  {r}: " + "  ".join(row) + f"   | PATP float64 closed form: {d64:.1e}"
              f" (alpha* = {s.param:.4f}, cond(A_grid) = {cond:.2e}, max|k| = {np.max(np.abs(s.coefs)):.3e})")
    print(f"  worst 50-digit discrepancy: {worst:.1e}")


def make_figure(kept, path):
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif", "font.serif": ["cmr10"],
        "axes.formatter.use_mathtext": True, "axes.unicode_minus": False, "font.size": 8.5,
        "mathtext.fontset": "cm", "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "legend.frameon": False,
    })
    g = RESPONSES[0][2]
    x_tr, x_te = kept["x_tr"], kept["x_te"]
    surr = kept["surr"]["g1"]
    lo_obs = min(0.0, float(x_te.min()))
    hi_obs = max(1.0, float(x_te.max()))
    xx = np.linspace(0.0, 1.0, 2001)
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.35), constrained_layout=True)

    bins = np.linspace(lo_obs, hi_obs, 31)
    ax[0].hist(x_tr, bins=bins, density=True, color="0.78", edgecolor="0.45", linewidth=0.4,
               label=f"train ($n={len(x_tr)}$)")
    ax[0].hist(x_te, bins=bins, density=True, histtype="step", color="black", linewidth=0.9,
               linestyle="--", label=f"test ($n={len(x_te)}$)")
    ax[0].set_xlabel("normalised window power $x$")
    ax[0].set_ylabel("density")
    ax[0].set_title("(a) input distribution", fontsize=8.5)
    ax[0].set_ylim(0, 1.45 * ax[0].get_ylim()[1])
    ax[0].legend(loc="upper right", fontsize=7)

    ax[1].plot(xx, g(xx), color="black", lw=1.4, label=r"true $g_1$")
    ax[1].plot(xx, surr["poly3"](xx), color="0.35", lw=1.0, ls="--", label="poly3")
    ax[1].plot(xx, surr["PATP"](xx), color="black", lw=0.9, ls=":", label=rf"PATP ($\alpha^\star={surr['PATP'].param:.3f}$)")
    ax[1].set_xlabel("$x$")
    ax[1].set_ylabel(r"$g_1(x)=\sqrt{x}\,(1+0.4x)$")
    ax[1].set_title("(b) response and surrogates", fontsize=8.5)
    ax[1].legend(loc="lower right", fontsize=7)
    ax[1].set_xlim(0, 1)

    xr = xx[1:]
    ax[2].semilogy(xr, np.abs(surr["poly3"](xr) - g(xr)), color="0.35", lw=1.0, ls="--", label="poly3")
    ax[2].semilogy(xr, np.abs(surr["poly4"](xr) - g(xr)), color="0.55", lw=1.0, ls="-.", label="poly4")
    ax[2].semilogy(xr, np.abs(surr["PATP"](xr) - g(xr)), color="black", lw=0.9, ls="-", label="PATP")
    q = np.quantile(x_tr, [0.025, 0.975])
    ax[2].axvspan(q[0], q[1], color="0.90", zorder=0, lw=0, label="central 95% of train")
    ax[2].set_xlabel("$x$")
    ax[2].set_ylabel(r"$|\hat g(x) - g_1(x)|$")
    ax[2].set_title("(c) absolute surrogate residual", fontsize=8.5)
    ax[2].set_xlim(0, 1)
    ax[2].set_ylim(1e-7, 50.0)
    ax[2].legend(loc="upper right", fontsize=6.5, ncol=2)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------

def prereg_hash() -> str:
    doc = __doc__
    a = doc.index("===== BEGIN PRE-REGISTRATION")
    b = doc.index("===== END PRE-REGISTRATION =====") + len("===== END PRE-REGISTRATION =====")
    return hashlib.sha256(doc[a:b].encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    ap.add_argument("--fig", default=FIG_DEFAULT)
    args = ap.parse_args()
    B = args.B
    # supplement: the CWRU files are not bundled; exit cleanly with a pointer if they are absent
    missing = [f"{r}.mat" for r in (105, 106) if not os.path.exists(os.path.join(DATA_DIR, f"{r}.mat"))]
    if missing:
        print(f"CWRU file(s) {missing} not found in {DATA_DIR}. The data is NOT bundled: see "
              f"data/README.md, or set CWRU_DATA_DIR to the folder that holds 105.mat and 106.mat.",
              file=sys.stderr)
        return 2

    print("realdata_split.py -- out-of-sample CWRU case study (PEM revision, task R)")
    print(f"pre-registration block sha256: {prereg_hash()}")
    print(f"Python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}, "
          f"mpmath {mp.__version__}, matplotlib {matplotlib.__version__}; mpmath dps = {mp.mp.dps}")
    print(f"B = {B}, L = {L_BLOCK}, seed = {SEED}, tau = {TAU:g}, k = {K_COV:g}, window W = {W} (non-overlapping)")
    nm = n_mellin_check()
    print(f"distinct Mellin (argument, derivative) evaluations for j <= 4: {nm}")
    assert nm == N_MELLIN

    P5, m5 = load_power(105)
    P6, m6 = load_power(106)
    for m in (m5, m6):
        ok = "MATCH" if m["sha"] == m["sha_manifest"] else "MISMATCH"
        print(f"record {m['rec']}: {m['key']}, {m['nsamp']} samples, {m['nwin']} windows "
              f"(tail {m['tail']} dropped), RPM variable = {m['rpm']}, channel mean = {m['dc']:.5f}")
        print(f"    sha256 {m['sha']}  manifest: {ok}")
    for rec, P in ((105, P5), (106, P6)):
        print(f"    window power {rec}: mean {P.mean():.5f}, sd {P.std():.5f}, min {P.min():.5f}, max {P.max():.5f}")

    n_tr = len(P5) // 2
    P_tr, P_te = P5[:n_tr], P5[n_tr:]

    print("\nInput-only diagnostics of the window-power series (block length):")
    for name, x in (("105 train", P_tr), ("105 test", P_te)):
        rho = acov(x, 66)
        rho = rho / rho[0]
        pw = politis_white(x)
        pk = sorted(range(12, 67), key=lambda k: -rho[k])[:2]
        print(f"  {name} (n={len(x)}): ACF lags 1..12 = {np.round(rho[1:13], 3).tolist()}")
        print(f"      largest ACF beyond lag 11: lag {pk[0]} ({rho[pk[0]]:.3f}), lag {pk[1]} ({rho[pk[1]]:.3f})")
        print(f"      Bartlett LRV ratio K=11/22/66: {bartlett_lrv_ratio(x, 11):.3f} / "
              f"{bartlett_lrv_ratio(x, 22):.3f} / {bartlett_lrv_ratio(x, 66):.3f}")
        if pw:
            print(f"      Politis-White: m_hat = {pw[0]}, M = {pw[1]}, b_SB = {pw[2]:.1f}, b_MBB/CBB = {pw[3]:.1f}")
    rev = 11 * W / (12000.0 * 60.0 / m5["rpm"])
    print(f"  lag 11 = {11 * W} samples = {rev:.3f} shaft revolutions at {m5['rpm']:.0f} rpm, 12 kHz")

    # ---------------- PRIMARY ----------------
    st, kept = evaluate(P_tr, P_te, "minmax", keep=True)
    closed_form_check(kept)
    boot = bootstrap(P_tr, P_te, "minmax", L_BLOCK, B, SEED + 0)
    report_full("PRIMARY: temporal split of record 105 (first 50% train / last 50% test), min-max with train constants",
                st, boot, kept, B, L_BLOCK)
    make_figure(kept, args.fig)
    print(f"\nfigure written: {os.path.normpath(args.fig)}")

    # ---------------- S1: cross-record ----------------
    st1, kept1 = evaluate(P5, P6, "minmax", keep=True)
    boot1 = bootstrap(P5, P6, "minmax", L_BLOCK, B, SEED + 1)
    report_full("S1 (secondary): cross-record, train = all of 105 (0 hp), test = all of 106 (1 hp), 105 constants",
                st1, boot1, kept1, B, L_BLOCK)

    # ---------------- S2: block length ----------------
    print("\n" + "=" * 100)
    print("S2 (secondary): block-length sensitivity, primary split; decision statistics only")
    print("=" * 100)
    s2 = {}
    for k, L in ((2, 1), (3, 11), (4, 44)):
        s2[L] = bootstrap(P_tr, P_te, "minmax", L, B, SEED + k)
    for r, label, _ in RESPONSES:
        print(f"--- {r}: {label}")
        for L in (1, 11, L_BLOCK, 44):
            bb = boot if L == L_BLOCK else s2[L]
            parts = [f"D_{b} [{ci(bb[(r, 'D', b)])[0]:+.2e}, {ci(bb[(r, 'D', b)])[1]:+.2e}]"
                     f" Ddev_{b} [{ci(bb[(r, 'Ddev', b)])[0]:+.2e}, {ci(bb[(r, 'Ddev', b)])[1]:+.2e}]"
                     for b in BASELINES]
            print(f"   L={L:>2}: " + "  ".join(parts) + f"  -> {verdict(bb, r)[0]}")
        print(f"   oracle mean-e CI by L: " + "  ".join(
            f"L={L}: [{ci((boot if L == L_BLOCK else s2[L])[(r, 'ORACLE', 'ebar')])[0]:.2e}, "
            f"{ci((boot if L == L_BLOCK else s2[L])[(r, 'ORACLE', 'ebar')])[1]:.2e}]" for L in (1, 11, L_BLOCK, 44)))

    # ---------------- S3: normalisation ----------------
    st3, kept3 = evaluate(P_tr, P_te, "scale", keep=True)
    boot3 = bootstrap(P_tr, P_te, "scale", L_BLOCK, B, SEED + 5)
    report_full("S3 (secondary): scale-only normalisation X = P / max_train P, primary split",
                st3, boot3, kept3, B, L_BLOCK)

    # ---------------- SUMMARY ----------------
    print("\n" + "=" * 100)
    print("SUMMARY: mean out-of-sample relative moment error over j = 1..4, point [95% percentile CI]")
    print("=" * 100)
    for title, s_, b_ in (("PRIMARY temporal 105", st, boot), ("S1 cross-record 105->106", st1, boot1),
                          ("S3 scale-normalised 105", st3, boot3)):
        print(f"{title}:")
        for r, label, _ in RESPONSES:
            cells = "  ".join(f"{m} {s_[(r, m, 'ebar')]:.2e} [{ci(b_[(r, m, 'ebar')])[0]:.2e}, {ci(b_[(r, m, 'ebar')])[1]:.2e}]"
                              for m in METHODS)
            print(f"  {r}: {cells}")
            print(f"      rule: {verdict(b_, r)[0]}")
    print("\nPRE-REGISTERED VERDICT (primary temporal split only):")
    for r, label, _ in RESPONSES:
        v, per = verdict(boot, r)
        print(f"  {r} = {label:<16} {v}   {per}")

    # ================= POST-HOC 1 (added after the first run; not pre-registered) =================
    print("\n" + "=" * 100)
    print("POST-HOC 1 (added after the first run; descriptive, does not change the verdict)")
    print("=" * 100)
    nl = boot[("n_low",)]
    print(f"primary, L = {L_BLOCK}: test windows below the replicate's train minimum (clipped to 0):")
    print(f"  replicates with >= 1 clipped window: {int(np.sum(nl > 0))} of {len(nl)} ({np.mean(nl > 0):.3f}); "
          f"median {np.median(nl):.0f}, 95th percentile {np.percentile(nl, 95):.0f}, max {nl.max():.0f}")
    for r, label, _ in RESPONSES:
        e = boot[(r, "ORACLE", "ebar")]
        if np.sum(nl == 0) == 0 or np.sum(nl > 0) == 0:
            print(f"  {r}: one of the two groups is empty; comparison skipped")
            continue
        print(f"  {r}: oracle mean e, median over replicates with 0 clipped = {np.median(e[nl == 0]):.2e}, "
              f"97.5th percentile = {np.percentile(e[nl == 0], 97.5):.2e};  with >= 1 clipped: median = "
              f"{np.median(e[nl > 0]):.2e}, 97.5th percentile = {np.percentile(e[nl > 0], 97.5):.2e}")

    print_table_rows(st, boot, st1, boot1, st3, boot3, s2)
    return 0


# ---------------------------------------------------------------------------
# TABLE ROWS (formatting only; added after the first run)
# ---------------------------------------------------------------------------

def tex_num(v: float, n: int) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    d = n - 1 - int(np.floor(np.log10(a)))
    r = round(a, d)
    if r != 0 and n - 1 - int(np.floor(np.log10(r))) < d:
        d = n - 1 - int(np.floor(np.log10(r)))
        r = round(a, d)
    s = f"{r:.{max(d, 0)}f}"
    return ("$-$" if v < 0 else "") + s


def tex_cell(point: float, arr: np.ndarray, scale: float = 1e3) -> str:
    lo, hi = ci(arr)
    if max(abs(point), abs(lo), abs(hi)) < 1e-12:
        return "exact"
    return f"{tex_num(point * scale, 3)}\\,[{tex_num(lo * scale, 2)}, {tex_num(hi * scale, 2)}]"


def tex_sci(v: float) -> str:
    if v < 1e-12:
        return "exact"
    m, e = f"{v:.1e}".split("e")
    return f"${m}{{\\cdot}}10^{{{int(e)}}}$"


RULE_TEX = {"PARITY": "parity"}


def print_table_rows(st, boot, st1, boot1, st3, boot3, s2):
    lab = {"g1": r"$g_1$", "g2": r"$g_2$", "g3": r"$g_3$", "g4": r"$g_4$"}
    print("\n" + "=" * 100)
    print("TABLE ROWS for revision/drafts/realdata.tex (formatting only; relative errors x 1e3, 95% percentile CI)")
    print("=" * 100)
    print("% Table A (primary): response & method & grid L2 & e_1 & e_2 & e_3 & e_4 & rel. err. U(k=2)")
    for r, _, _ in RESPONSES:
        for i, m in enumerate(METHODS):
            head = lab[r] if i == 0 else ""
            l2 = tex_sci(st[(r, m, "l2")]) if (r, m, "l2") in st else "--"
            cells = " & ".join(tex_cell(st[(r, m, "e", j)], boot[(r, m, "e", j)]) for j in J)
            print(f"{head} & {m} & {l2} & {cells} & {tex_cell(st[(r, m, 'eU')], boot[(r, m, 'eU')])} \\\\")
        print("\\midrule" if r != RESPONSES[-1][0] else "\\bottomrule")
    print("% Table B (primary decision): response & U_ref(k=2) & ORACLE mean e & mean dev poly3 & poly4 & PATP"
          " & D_poly3 & D_poly4 & rule")
    for r, _, _ in RESPONSES:
        lo, hi = ci(boot[(r, "REF", "U")])
        uref = f"{tex_num(st[(r, 'REF', 'U')], 3)}\\,[{tex_num(lo, 2)}, {tex_num(hi, 2)}]"
        devs = " & ".join(tex_cell(st[(r, m, "devbar")], boot[(r, m, "devbar")]) for m in ("poly3", "poly4", "PATP"))
        ds = " & ".join(tex_cell(st[(r, "D", b)], boot[(r, "D", b)]) for b in BASELINES)
        v = verdict(boot, r)[0]
        print(f"{lab[r]} & {uref} & {tex_cell(st[(r, 'ORACLE', 'ebar')], boot[(r, 'ORACLE', 'ebar')])} & {devs} & {ds}"
              f" & {RULE_TEX.get(v, v)} \\\\")
    print("% condition A / B per baseline (primary), Ddev intervals x 1e3:")
    for r, _, _ in RESPONSES:
        parts = []
        for b in BASELINES:
            dlo = ci(boot[(r, "D", b)])[0]
            vlo, vhi = ci(boot[(r, "Ddev", b)])
            parts.append(f"{b}: A={'yes' if dlo > TAU else 'no'} B={'yes' if vlo > TAU else 'no'} "
                         f"Ddev={tex_cell(st[(r, 'Ddev', b)], boot[(r, 'Ddev', b)])}")
        print(f"%   {r}: " + "; ".join(parts))
    print("% Table C (secondary): analysis & response & ORACLE mean e & PATP mean e & poly3 mean e & D_poly3 & D_poly4 & rule")
    for name, s_, b_ in (("S1 cross-record", st1, boot1), ("S3 scale-only", st3, boot3)):
        for i, (r, _, _) in enumerate(RESPONSES):
            head = name if i == 0 else ""
            es = " & ".join(tex_cell(s_[(r, m, "ebar")], b_[(r, m, "ebar")]) for m in ("ORACLE", "PATP", "poly3"))
            ds = " & ".join(tex_cell(s_[(r, "D", b)], b_[(r, "D", b)]) for b in BASELINES)
            v = verdict(b_, r)[0]
            print(f"{head} & {lab[r]} & {es} & {ds} & {RULE_TEX.get(v, v)} \\\\")
        print("\\midrule" if name.startswith("S1") else "\\bottomrule")
    print("% S2 rule outcome by block length L = 1 / 11 / 22 / 44:")
    for r, _, _ in RESPONSES:
        outs = [verdict(boot if L == L_BLOCK else s2[L], r)[0] for L in (1, 11, L_BLOCK, 44)]
        print(f"%   {r}: " + " / ".join(outs))


if __name__ == "__main__":
    sys.exit(main())
