"""
P3 Part II -- balanced benchmark over two classes of response surfaces.

Revised analysis. Tests a broader and more balanced set of response types than the
fractional-power-dominated targets, reports model complexity, fitted shape parameter,
approximation error and propagated-moment error for each basis, and separates surrogate
approximation error, propagation error relative to direct integration of the fitted
surrogate, and input-information cost at matched degrees of freedom.

=====================================================================================
PRE-REGISTRATION -- written 2026-09-26 before the first run of this script.
The target list, design, prediction and decision rule below are frozen. Anything
computed after the first run that is not listed here is printed under a header
"POST-HOC" and does not enter the verdict. Implementation bugs found after the first
run may be fixed; every such fix is recorded in the CHANGELOG at the end of this
docstring, with a statement of whether it changed any verdict.
=====================================================================================

1. Targets (fixed; none is dropped or swapped after the first run)

  Class E -- non-analytic at the support edge x = 0 (an algebraic branch point there),
  analytic on (0, 1]:
    E1  sqrt(x) exp(-x)
    E2  x^(1/3) (1 + x)^(1/2)
    E3  sqrt(x (2 - x))
    E4  x^0.7 / (1 + x)
    E5  sqrt(x) (1 + x^2/3)            (= M4 of tab:rq3; step-0 outcome already known)

  Class A -- analytic on a neighbourhood of [0, 1]:
    A1  exp(-x)                         (= M7; step-0 outcome already known)
    A2  ln(1 + x)                       (= M6; step-0 outcome already known)
    A3  sin(pi x / 2)
    A4  1 / (1 + x)
    A5  tanh(3 x)

  Exact containment. Every E target is x^gamma times a function analytic at 0 whose
  Taylor series does not terminate, except E5 = x^(1/2) + x^(5/2)/3, whose two exponents
  are never simultaneously values of p_2, p_3, p_4 at a single alpha (p_2 = 1/2 only at
  alpha = 0, where p_3 = 1/3 and p_4 = 1/4). The A targets are non-polynomial entire or
  meromorphic functions. So no target lies in the finite span of any tested basis. The
  script checks this numerically before any moment is computed: every fixed-capacity
  method (all methods except poly-dm) must have grid L2 > 1e-9 on every target (the
  sanity rows of tab:rq3, where containment is exact, sit at 1e-10 .. 1e-11). If the check
  fails, the run stops without a verdict.

2. Inputs
  primary    X ~ U[0,1]      M_X(s) = 1/s
  secondary  X ~ Beta(2,5)   M_X(s) = B(1+s, 5)/B(2, 5) = 720 / prod_{k=1..5} (s + k)
  The log-power basis needs derivatives of M_X; both are exact: U: (-1)^b b!/s^(b+1);
  Beta: partial fractions of the rational M_X, 30/(s+1) - 120/(s+2) + 180/(s+3)
  - 120/(s+4) + 30/(s+5). A self-test at start-up checks them against mp.beta, mp.diff
  and mp.quad.

3. Surrogates -- fitted once per target on the step-0 grid (401 points,
  linspace(1e-6, 1, 401)) by unweighted OLS (np.linalg.lstsq, rcond=None), independent
  of the input law, exactly as in step 0 and in the paper.
    poly3      {1, x, x^2, x^3}                                  4 params
    poly4      {1, x, ..., x^4}                                  5 params
    PATP-opt   {1, x^p2(a), x^p3(a), x^p4(a)}, a in [0,1] optimised     5 params (4 + a)
    PATP-a25   same basis with a = 0.25 fixed                    4 params
    sqrt3      {1, x^(1/2), x, x^(3/2)} fixed                    4 params
    conf-opt   {1, x^c, x^c ln x, x^c ln^2 x}, c in [0.05, 3] optimised  5 params (4 + c)
    poly-dm    smallest degree d <= 12 with grid L2 <= L2(PATP-opt)    d + 1 params
  The shape parameter (a or c) minimises the grid mean squared residual only; no moment
  of X or of f enters the fit. Two searches are run and the lower residual is kept:
  (i) bounded Brent on the whole interval, xatol = 1e-6 (the paper's optimiser);
  (ii) a 201-point scan followed by bounded Brent (xatol = 1e-6) on the bracketing cell.
  Both candidates are printed.

4. Reported per target and method
  params; Mellin cost for j <= 4 = number of distinct values the closed form consumes
  (distinct points s, and for conf-opt the distinct (s, derivative order) pairs);
  a* or c*; cond(A) of the grid design matrix; grid L2 (RMS residual on the grid);
  relative moment errors |E[g^j] - E[f^j]| / E[f^j], j = 1..4; their geometric mean
  (geo); propagation error = max_j |closed form - quadrature of the same surrogate| /
  |quadrature|, once for the closed form in 50-digit arithmetic (prop50) and once for
  the closed form in float64 (prop64).
  Moments of a surrogate are evaluated in 50-digit arithmetic (mpmath) from its float64
  coefficients and exponents, so the surrogate is exactly the fitted function; the truth
  E[f(X)^j] and the quadrature of the surrogate are mpmath tanh-sinh at 50 digits.

5. PRIMARY PREDICTION AND DECISION RULE (X ~ U[0,1])
  P-E  geo(PATP-opt) < geo(poly4) on >= 4 of the 5 Class E targets.
  P-A  geo(poly4) < geo(PATP-opt) on >= 4 of the 5 Class A targets.
  If P-E and P-A both hold, the paper may state the selection rule
    "PATP-MUET pays off when the response is non-analytic at the edge of the input
     support; for analytic responses a polynomial of one degree more is better."
  If either fails, the script prints which targets break the rule, and the paper must
  not state it as a rule.
  E5, A1 and A2 have step-0 outcomes already known (PATP-opt < poly4 on M4; poly4 <
  PATP-opt on M6 and M7). The counts restricted to the seven new targets (E1-E4, A3-A5)
  are printed as well; the verdict uses all ten.

6. SECONDARY PREDICTIONS (reported; they do not change the verdict)
  S1  Beta(2,5) down-weights the edge x = 0, so the PATP advantage on Class E shrinks:
      the median over E1-E5 of log10(geo(poly4) / geo(PATP-opt)) is smaller under
      Beta(2,5) than under U[0,1]. The Class E and Class A win counts under Beta(2,5)
      are printed too.
  S2  (mechanism, from step 0) on every target with |a* - 1/2| < 0.1, conf-opt reaches
      grid L2 <= 1.1 * L2(PATP-opt).
  S3  (shape optimisation matters, from step 0) geo(PATP-opt) < geo(PATP-a25) on all ten
      targets under U[0,1].

Run (from experiments/):
  ../verification/cas/.venv/bin/python p3_balanced_benchmark.py | tee results/p3_balanced.txt

The module is also imported by p3_reverify.py for its surrogate / moment machinery.

CHANGELOG (after the first run)
  - Before the first run, the machinery (not the benchmark targets) was exercised on the
    step-0 targets M1, M4, M6, M7 to confirm that it reproduces results/step0.txt.
  - 2026-09-26, after the first run: a POST-HOC section (H1-H5) was appended to main():
    Bernstein-ellipse parameters of the Class A singularities, the least-squares
    polynomial degree ladder, PATP with the Brent-only optimiser, the worst propagation
    error per method, and per-target geo ratios quoted in the draft text. No
    pre-registered code path was changed; the pre-registered part of the output was
    checked to be identical to the first run.
"""

from __future__ import annotations

import sys
from itertools import combinations_with_replacement
from math import factorial

import mpmath as mp
import numpy as np
import scipy
from scipy.optimize import minimize_scalar

import step0_capacity_matched as s0  # same grid and exponent map as step 0

mp.mp.dps = 50
J = (1, 2, 3, 4)
X_GRID = s0.X_GRID
p_num = s0.p_num
CONTAIN_TOL = 1e-9
DM_MAX = 12
METHODS = ["poly3", "poly4", "PATP-opt", "PATP-a25", "sqrt3", "conf-opt", "poly-dm"]
FIXED_CAPACITY = METHODS[:-1]


# ---------------------------------------------------------------------------
# Input laws: b-th derivative of M_X(s) = E[X^(s-1)], in 50 digits and in float64
# ---------------------------------------------------------------------------

class Uniform01:
    name = "U[0,1]"

    def mellin(self, s, b: int):
        s = mp.mpf(s)
        return (-1) ** b * mp.factorial(b) / s ** (b + 1)

    def mellin_f64(self, s: float, b: int) -> float:
        return (-1) ** b * factorial(b) / s ** (b + 1)

    def density(self, x):
        return mp.mpf(1)


class Beta25:
    name = "Beta(2,5)"
    PF = {1: 30, 2: -120, 3: 180, 4: -120, 5: 30}  # 720/prod(s+k) = sum PF_k/(s+k)

    def mellin(self, s, b: int):
        s = mp.mpf(s)
        if b == 0:
            return mp.mpf(720) / mp.fprod([s + k for k in range(1, 6)])
        return mp.fsum(a * (-1) ** b * mp.factorial(b) / (s + k) ** (b + 1) for k, a in self.PF.items())

    def mellin_f64(self, s: float, b: int) -> float:
        if b == 0:
            return 720.0 / float(np.prod([s + k for k in range(1, 6)]))
        return sum(a * (-1) ** b * factorial(b) / (s + k) ** (b + 1) for k, a in self.PF.items())

    def density(self, x):
        return 30 * x * (1 - x) ** 4


UNIFORM, BETA25 = Uniform01(), Beta25()


# ---------------------------------------------------------------------------
# Surrogate g(x) = sum_k c_k x^(e_k) (ln x)^(b_k); first term is the constant (0, 0).
# E[g^j] = sum_{|kappa| = j} multinom(kappa) prod c^kappa M_X^{(B)}(A + 1),
# A = sum kappa_k e_k, B = sum kappa_k b_k   (E[X^a ln^b X] = M_X^{(b)}(a + 1)).
# ---------------------------------------------------------------------------

def design(x: np.ndarray, terms) -> np.ndarray:
    lx = np.log(x)
    return np.column_stack([x ** e * lx ** b for e, b in terms])


def ols(f, terms) -> np.ndarray:
    coefs, *_ = np.linalg.lstsq(design(X_GRID, terms), f(X_GRID), rcond=None)
    return coefs


def grid_mse(f, terms) -> float:
    A = design(X_GRID, terms)
    c, *_ = np.linalg.lstsq(A, f(X_GRID), rcond=None)
    return float(np.mean((A @ c - f(X_GRID)) ** 2))


class Surrogate:
    def __init__(self, name, terms, coefs, n_params, shape=None, opt=None):
        self.name = name
        self.terms = [(float(e), int(b)) for e, b in terms]
        self.coefs = np.asarray(coefs, dtype=float)
        self.n_params = n_params
        self.shape = shape  # (symbol, value) or None
        self.opt = opt      # optimiser record or None
        self._mt = [(mp.mpf(e), b) for e, b in self.terms]
        self._mc = [mp.mpf(float(c)) for c in self.coefs]

    def __call__(self, x):
        return design(x, self.terms) @ self.coefs

    def eval_mp(self, x):
        lx = mp.log(x)
        return mp.fsum(c * x ** e * lx ** b for c, (e, b) in zip(self._mc, self._mt))

    def _multisets(self, j: int):
        n = len(self.terms)
        for combo in combinations_with_replacement(range(n), j):
            yield [combo.count(k) for k in range(n)]

    def moment_mp(self, j: int, law):
        total = mp.mpf(0)
        for kappa in self._multisets(j):
            multi, prod, A, B = mp.mpf(factorial(j)), mp.mpf(1), mp.mpf(0), 0
            for k, ke in enumerate(kappa):
                if ke:
                    multi /= factorial(ke)
                    prod *= self._mc[k] ** ke
                    A += ke * self._mt[k][0]
                    B += ke * self._mt[k][1]
            total += multi * prod * law.mellin(A + 1, B)
        return total

    def moment_f64(self, j: int, law) -> float:
        total = 0.0
        for kappa in self._multisets(j):
            multi, prod, A, B = factorial(j), 1.0, 0.0, 0
            for k, ke in enumerate(kappa):
                if ke:
                    multi //= factorial(ke)
                    prod *= float(self.coefs[k]) ** ke
                    A += ke * self.terms[k][0]
                    B += ke * self.terms[k][1]
            total += multi * prod * law.mellin_f64(A + 1.0, B)
        return total

    def quad_moment(self, j: int, law):
        return mp.quad(lambda x: self.eval_mp(x) ** j * law.density(x), [0, 0.5, 1])

    def mellin_cost(self, jmax: int = 4) -> tuple[int, int]:
        pts, vals = set(), set()
        for kappa in self._multisets(jmax):  # the constant term fills lower orders
            A = round(sum(ke * e for ke, (e, _) in zip(kappa, self.terms)), 10)
            B = sum(ke * b for ke, (_, b) in zip(kappa, self.terms))
            pts.add(A)
            vals.add((A, B))
        return len(pts), len(vals)

    def l2(self, f) -> float:
        return float(np.sqrt(np.mean((self(X_GRID) - f(X_GRID)) ** 2)))

    def cond(self) -> float:
        return float(np.linalg.cond(design(X_GRID, self.terms)))


def poly_terms(d: int):
    return [(k, 0) for k in range(d + 1)]


def patp_terms(a: float):
    return [(0.0, 0)] + [(p_num(i, a), 0) for i in (2, 3, 4)]


SQRT3_TERMS = [(0.0, 0), (0.5, 0), (1.0, 0), (1.5, 0)]


def conf_terms(c: float):
    return [(0.0, 0), (c, 0), (c, 1), (c, 2)]


def optimise(loss, lo: float, hi: float):
    r1 = minimize_scalar(loss, bounds=(lo, hi), method="bounded", options={"xatol": 1e-6})
    grid = np.linspace(lo, hi, 201)
    vals = np.array([loss(a) for a in grid])
    k = int(np.argmin(vals))
    r2 = minimize_scalar(loss, bounds=(grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]),
                         method="bounded", options={"xatol": 1e-6})
    cands = [(float(r1.fun), float(r1.x), "brent"), (float(r2.fun), float(r2.x), "scan+brent"),
             (float(vals[k]), float(grid[k]), "scan")]
    best = min(cands, key=lambda t: t[0])
    return best[1], {"brent": (float(r1.x), float(r1.fun)), "scan+brent": (float(r2.x), float(r2.fun)),
                     "chosen": best[2]}


def poly_sur(f, d: int, name: str | None = None) -> Surrogate:
    t = poly_terms(d)
    return Surrogate(name or f"poly{d}", t, ols(f, t), d + 1)


def fit_methods(f) -> list[Surrogate]:
    """All seven methods of the design, fitted on the grid residual only."""
    out = [poly_sur(f, 3), poly_sur(f, 4)]
    a, ainfo = optimise(lambda a: grid_mse(f, patp_terms(a)), 0.0, 1.0)
    popt = Surrogate("PATP-opt", patp_terms(a), ols(f, patp_terms(a)), 5, ("a", a), ainfo)
    out.append(popt)
    out.append(Surrogate("PATP-a25", patp_terms(0.25), ols(f, patp_terms(0.25)), 4, ("a", 0.25)))
    out.append(Surrogate("sqrt3", SQRT3_TERMS, ols(f, SQRT3_TERMS), 4))
    c, cinfo = optimise(lambda c: grid_mse(f, conf_terms(c)), 0.05, 3.0)
    out.append(Surrogate("conf-opt", conf_terms(c), ols(f, conf_terms(c)), 5, ("c", c), cinfo))
    target = popt.l2(f)
    for d in range(1, DM_MAX + 1):
        cand = poly_sur(f, d)
        if cand.l2(f) <= target:
            cand.name = "poly-dm"
            cand.dm_degree = d
            out.append(cand)
            break
    return out


def by_name(surs, name):
    for s in surs:
        if s.name == name:
            return s
    return None


def truth_moment(f_mp, law, j: int):
    return mp.quad(lambda x: f_mp(x) ** j * law.density(x), [0, 0.5, 1])


def geo(v) -> float:
    return float(np.exp(np.mean(np.log(np.maximum(np.asarray(v, dtype=float), 1e-300)))))


def evaluate(surs, f_mp, law, jset=J):
    """Relative moment errors, geo, and propagation errors of every surrogate under `law`."""
    truth = {j: truth_moment(f_mp, law, j) for j in jset}
    res = {}
    for s in surs:
        rel, p50, p64, m50s, m64s = [], 0.0, 0.0, {}, {}
        for j in jset:
            m50 = s.moment_mp(j, law)
            m64 = s.moment_f64(j, law)
            q = s.quad_moment(j, law)
            m50s[j], m64s[j] = m50, m64
            rel.append(float(abs(m50 - truth[j]) / abs(truth[j])))
            p50 = max(p50, float(abs(m50 - q) / abs(q)))
            p64 = max(p64, float(abs(mp.mpf(m64) - q) / abs(q)))
        res[s.name] = {"rel": rel, "geo": geo(rel), "p50": p50, "p64": p64, "m50": m50s, "m64": m64s}
    return truth, res


# ---------------------------------------------------------------------------
# Targets (numpy version for fitting, mpmath version for truth)
# ---------------------------------------------------------------------------

E_TARGETS = [
    ("E1", "sqrt(x) exp(-x)", lambda x: np.sqrt(x) * np.exp(-x), lambda x: mp.sqrt(x) * mp.exp(-x)),
    ("E2", "x^(1/3) (1+x)^(1/2)", lambda x: np.cbrt(x) * np.sqrt(1 + x), lambda x: mp.cbrt(x) * mp.sqrt(1 + x)),
    ("E3", "sqrt(x (2-x))", lambda x: np.sqrt(x * (2 - x)), lambda x: mp.sqrt(x * (2 - x))),
    ("E4", "x^0.7 / (1+x)", lambda x: x ** 0.7 / (1 + x), lambda x: x ** mp.mpf("0.7") / (1 + x)),
    ("E5", "sqrt(x) (1 + x^2/3)  [=M4]", lambda x: np.sqrt(x) * (1 + x ** 2 / 3),
     lambda x: mp.sqrt(x) * (1 + x ** 2 / 3)),
]
A_TARGETS = [
    ("A1", "exp(-x)  [=M7]", lambda x: np.exp(-x), lambda x: mp.exp(-x)),
    ("A2", "ln(1+x)  [=M6]", lambda x: np.log(1 + x), lambda x: mp.log(1 + x)),
    ("A3", "sin(pi x/2)", lambda x: np.sin(np.pi * x / 2), lambda x: mp.sin(mp.pi * x / 2)),
    ("A4", "1/(1+x)", lambda x: 1 / (1 + x), lambda x: 1 / (1 + x)),
    ("A5", "tanh(3x)", lambda x: np.tanh(3 * x), lambda x: mp.tanh(3 * x)),
]
KNOWN = {"E5", "A1", "A2"}


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def shape_str(s: Surrogate) -> str:
    if s.shape is None:
        return "-"
    return f"{s.shape[0]}*={s.shape[1]:.6f}"


def print_fit_block(tag, label, cls, surs, f):
    print(f"\n=== {tag} [{cls}]  f(x) = {label} ===")
    print(f"  {'method':<10}{'params':>7}{'M-pts':>7}{'M-vals':>8}{'shape':>16}{'cond(A)':>11}"
          f"{'grid L2':>11}{'max|c|':>10}")
    for s in surs:
        pts, vals = s.mellin_cost()
        name = s.name + (f"(d={s.dm_degree})" if s.name == "poly-dm" else "")
        print(f"  {name:<10}{s.n_params:>7}{pts:>7}{vals:>8}{shape_str(s):>16}{s.cond():>11.2e}"
              f"{s.l2(f):>11.2e}{np.max(np.abs(s.coefs)):>10.2e}")
    if by_name(surs, "poly-dm") is None:
        print(f"  poly-dm: no degree <= {DM_MAX} reaches L2(PATP-opt)")
    for s in surs:
        if s.opt is not None:
            b, sb = s.opt["brent"], s.opt["scan+brent"]
            print(f"  {s.name} optimiser: brent {s.shape[0]}={b[0]:.6f} (mse {b[1]:.3e}); "
                  f"scan+brent {s.shape[0]}={sb[0]:.6f} (mse {sb[1]:.3e}); chosen: {s.opt['chosen']}")
    popt = by_name(surs, "PATP-opt")
    print(f"  PATP-opt exponents = {[round(e, 5) for e, _ in popt.terms[1:]]}, "
          f"coefs = {[round(float(c), 4) for c in popt.coefs]}")


def print_moment_block(law, truth, res, surs):
    print(f"  [{law.name}] truth E[f^j] = " + ", ".join(f"{mp.nstr(truth[j], 12)}" for j in J))
    print(f"  {'method':<10}" + "".join(f"{'rel j=' + str(j):>11}" for j in J)
          + f"{'geo':>11}{'prop50':>11}{'prop64':>11}")
    for s in surs:
        r = res[s.name]
        print(f"  {s.name:<10}" + "".join(f"{e:>11.2e}" for e in r["rel"])
              + f"{r['geo']:>11.2e}{r['p50']:>11.1e}{r['p64']:>11.1e}")


def self_test() -> bool:
    ok = True
    print("Self-test of the Mellin derivatives")
    for s in (mp.mpf("1"), mp.mpf("1.5"), mp.mpf("2.7")):
        ref = mp.beta(1 + s, 5) / mp.beta(2, 5)
        d0 = abs(BETA25.mellin(s, 0) - ref) / ref
        d0f = abs(mp.mpf(BETA25.mellin_f64(float(s), 0)) - ref) / ref
        good = d0 < mp.mpf("1e-45") and d0f < 1e-14
        ok &= bool(good)
        print(f"  Beta(2,5) M_X({mp.nstr(s, 3)}) vs mp.beta: 50d {mp.nstr(d0, 3)}, f64 {mp.nstr(d0f, 3)}"
              f"  [{'ok' if good else 'FAIL'}]")
        for b in (1, 2, 3):
            num = mp.diff(lambda t: mp.beta(1 + t, 5) / mp.beta(2, 5), s, b)
            db = abs(BETA25.mellin(s, b) - num) / abs(num)
            good = db < mp.mpf("1e-30")
            ok &= bool(good)
            print(f"    derivative b={b}: vs mp.diff {mp.nstr(db, 3)}  [{'ok' if good else 'FAIL'}]")
    for law in (UNIFORM, BETA25):
        for a, b in ((mp.mpf("0.37"), 2), (mp.mpf("1.2"), 3)):
            q = mp.quad(lambda x: x ** a * mp.log(x) ** b * law.density(x), [0, 0.5, 1])
            d = abs(law.mellin(a + 1, b) - q) / abs(q)
            good = d < mp.mpf("1e-35")
            ok &= bool(good)
            print(f"  {law.name}: E[X^{mp.nstr(a, 3)} ln^{b} X] closed vs quad {mp.nstr(d, 3)}"
                  f"  [{'ok' if good else 'FAIL'}]")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("P3 balanced benchmark (pre-registered design in the module docstring)")
    print(f"python {sys.version.split()[0]}, numpy {np.__version__}, scipy {scipy.__version__}, "
          f"mpmath {mp.__version__}, mp.dps = {mp.mp.dps}")
    print(f"grid: {len(X_GRID)} points on [{X_GRID[0]:g}, {X_GRID[-1]:g}]; OLS; truth and surrogate "
          f"quadrature: mpmath tanh-sinh, 50 digits")
    if not self_test():
        print("SELF-TEST FAILED -- no benchmark run")
        return 1

    targets = [(t, l, fn, fm, "E") for t, l, fn, fm in E_TARGETS] + \
              [(t, l, fn, fm, "A") for t, l, fn, fm in A_TARGETS]

    # 1. Fit everything and check exact containment before any moment is computed.
    fits = {}
    bad = []
    for tag, label, fn, fm, cls in targets:
        surs = fit_methods(fn)
        fits[tag] = surs
        for s in surs:
            if s.name in FIXED_CAPACITY and s.l2(fn) <= CONTAIN_TOL:
                bad.append((tag, s.name, s.l2(fn)))
    print("\nContainment check (fixed-capacity methods, grid L2 > 1e-9 required):")
    for tag, label, fn, fm, cls in targets:
        mn = min((s.l2(fn), s.name) for s in fits[tag] if s.name in FIXED_CAPACITY)
        print(f"  {tag}: smallest fixed-capacity grid L2 = {mn[0]:.2e} ({mn[1]})")
    if bad:
        print(f"CONTAINMENT CHECK FAILED: {bad} -- run stops without a verdict")
        return 2
    print("  passed: no target lies in a tested basis")

    # 2. Per-target blocks under both laws.
    results = {UNIFORM.name: {}, BETA25.name: {}}
    for tag, label, fn, fm, cls in targets:
        surs = fits[tag]
        print_fit_block(tag, label, "class " + cls, surs, fn)
        for law in (UNIFORM, BETA25):
            truth, res = evaluate(surs, fm, law)
            results[law.name][tag] = res
            print_moment_block(law, truth, res, surs)

    # 3. Summary matrices.
    for law in (UNIFORM, BETA25):
        print("\n" + "=" * 100)
        print(f"GEO-MEAN RELATIVE MOMENT ERROR, j = 1..4, X ~ {law.name}")
        print("=" * 100)
        print(f"  {'target':<8}" + "".join(f"{m:>11}" for m in METHODS) + f"{'poly4/PATP':>12}")
        for tag, *_ in targets:
            r = results[law.name][tag]
            cells = "".join(f"{r[m]['geo']:>11.2e}" if m in r else f"{'-':>11}" for m in METHODS)
            print(f"  {tag:<8}{cells}{r['poly4']['geo'] / r['PATP-opt']['geo']:>12.2f}")

    # 4. Pre-registered verdict.
    U = results[UNIFORM.name]
    print("\n" + "=" * 100)
    print("PRE-REGISTERED VERDICT (X ~ U[0,1])")
    print("=" * 100)
    e_tags = [t for t, *_ in E_TARGETS]
    a_tags = [t for t, *_ in A_TARGETS]
    e_win = [t for t in e_tags if U[t]["PATP-opt"]["geo"] < U[t]["poly4"]["geo"]]
    a_win = [t for t in a_tags if U[t]["poly4"]["geo"] < U[t]["PATP-opt"]["geo"]]
    pe, pa = len(e_win) >= 4, len(a_win) >= 4
    print(f"P-E  PATP-opt < poly4 on Class E: {e_win}  -> {len(e_win)}/5  {'HOLDS' if pe else 'FAILS'}")
    print(f"P-A  poly4 < PATP-opt on Class A: {a_win}  -> {len(a_win)}/5  {'HOLDS' if pa else 'FAILS'}")
    new_e = [t for t in e_win if t not in KNOWN]
    new_a = [t for t in a_win if t not in KNOWN]
    print(f"     restricted to the new targets: Class E {len(new_e)}/4 {new_e}, Class A {len(new_a)}/3 {new_a}")
    if pe and pa:
        print("DECISION: both hold -> the selection rule may be stated in the paper.")
    else:
        breakers = [t for t in e_tags if t not in e_win] if not pe else []
        breakers += [t for t in a_tags if t not in a_win] if not pa else []
        print(f"DECISION: the rule FAILS -> must not be stated as a rule. Breaking targets: {breakers}")
    print("Targets against the rule (all, whether or not the count threshold is met): "
          f"{[t for t in e_tags if t not in e_win] + [t for t in a_tags if t not in a_win]}")

    # 5. Secondary predictions.
    B = results[BETA25.name]
    print("\n" + "=" * 100)
    print("SECONDARY PREDICTIONS (pre-registered, not part of the decision)")
    print("=" * 100)
    lr_u = [np.log10(U[t]["poly4"]["geo"] / U[t]["PATP-opt"]["geo"]) for t in e_tags]
    lr_b = [np.log10(B[t]["poly4"]["geo"] / B[t]["PATP-opt"]["geo"]) for t in e_tags]
    print("S1  log10(geo(poly4)/geo(PATP-opt)) on Class E:")
    for t, u, b in zip(e_tags, lr_u, lr_b):
        print(f"      {t}: U[0,1] {u:+.3f}   Beta(2,5) {b:+.3f}")
    s1 = float(np.median(lr_b)) < float(np.median(lr_u))
    print(f"    median U[0,1] {np.median(lr_u):+.3f}, median Beta(2,5) {np.median(lr_b):+.3f}"
          f"  -> S1 {'HOLDS (advantage shrinks)' if s1 else 'FAILS'}")
    eb = [t for t in e_tags if B[t]["PATP-opt"]["geo"] < B[t]["poly4"]["geo"]]
    ab = [t for t in a_tags if B[t]["poly4"]["geo"] < B[t]["PATP-opt"]["geo"]]
    print(f"    Beta(2,5) win counts: PATP-opt < poly4 on Class E {len(eb)}/5 {eb}; "
          f"poly4 < PATP-opt on Class A {len(ab)}/5 {ab}")
    print("S2  conf-opt L2 vs PATP-opt L2 where |a* - 1/2| < 0.1:")
    s2_rows = []
    for tag, label, fn, fm, cls in targets:
        popt, conf = by_name(fits[tag], "PATP-opt"), by_name(fits[tag], "conf-opt")
        a = popt.shape[1]
        ratio = conf.l2(fn) / popt.l2(fn)
        near = abs(a - 0.5) < 0.1
        if near:
            s2_rows.append(ratio <= 1.1)
        print(f"      {tag}: a* = {a:.4f} {'(near 1/2)' if near else '          '}  "
              f"L2 conf/PATP = {ratio:.3f}  c* = {conf.shape[1]:.4f}")
    if s2_rows:
        print(f"    -> S2 {'HOLDS' if all(s2_rows) else 'FAILS'} ({sum(s2_rows)}/{len(s2_rows)} near-1/2 targets)")
    else:
        print("    -> S2 not testable (no target with |a* - 1/2| < 0.1)")
    s3 = [t for t, *_ in targets if U[t]["PATP-opt"]["geo"] < U[t]["PATP-a25"]["geo"]]
    print(f"S3  PATP-opt < PATP-a25 on {len(s3)}/10: {s3}  -> S3 {'HOLDS' if len(s3) == 10 else 'FAILS'}")
    post_hoc(fits, results, targets)
    return 0


# ---------------------------------------------------------------------------
# POST-HOC -- added after the first run; not pre-registered; does not enter the verdict
# ---------------------------------------------------------------------------

# Nearest singularity of each Class A target in the complex x-plane (None = entire).
NEAREST_SINGULARITY = {"A1": None, "A2": -1.0 + 0j, "A3": None, "A4": -1.0 + 0j, "A5": 1j * np.pi / 6}


def bernstein_rho(z: complex) -> float:
    """Parameter rho > 1 of the Bernstein ellipse for [0, 1] passing through z."""
    t = 2 * z - 1
    w = t + np.sqrt(t * t - 1 + 0j)
    r = abs(w)
    return max(r, 1 / r)


def post_hoc(fits, results, targets) -> None:
    U = results[UNIFORM.name]
    print("\n" + "=" * 100)
    print("POST-HOC (added after the first run; not pre-registered; does not change the verdict)")
    print("=" * 100)

    print("H1  Class A: Bernstein-ellipse parameter rho of the nearest singularity (interval [0,1]);")
    print("    polynomial best approximation converges like rho^(-n)")
    for tag, z in NEAREST_SINGULARITY.items():
        if z is None:
            print(f"      {tag}: entire (rho = infinity)")
        else:
            print(f"      {tag}: singularity at x = {z.real:+.4f}{z.imag:+.4f}i  ->  rho = {bernstein_rho(z):.3f}")

    print("H2  grid L2 of the least-squares polynomial of degree d (algebraic vs geometric decay)")
    degs = (2, 4, 6, 8, 10, 12)
    print(f"      {'target':<8}" + "".join(f"{'d=' + str(d):>11}" for d in degs) + f"{'L2(4)/L2(12)':>14}")
    for tag, label, fn, fm, cls in targets:
        l2s = [poly_sur(fn, d).l2(fn) for d in degs]
        print(f"      {tag:<8}" + "".join(f"{v:>11.2e}" for v in l2s) + f"{l2s[1] / l2s[-1]:>14.2e}")

    print("H3  PATP with the paper's optimiser alone (bounded Brent on [0,1], xatol 1e-6), U[0,1],")
    print("    on targets where it lands on a different alpha than the pre-registered two-search optimiser")
    any_diff = False
    for tag, label, fn, fm, cls in targets:
        popt = by_name(fits[tag], "PATP-opt")
        ab = popt.opt["brent"][0]
        if abs(ab - popt.shape[1]) > 1e-3:
            any_diff = True
            sb = Surrogate("PATP-brent", patp_terms(ab), ols(fn, patp_terms(ab)), 5, ("a", ab))
            _, r = evaluate([sb], fm, UNIFORM)
            g = r["PATP-brent"]["geo"]
            print(f"      {tag}: brent a = {ab:.6f} (L2 {sb.l2(fn):.2e}), chosen a = {popt.shape[1]:.6f} "
                  f"(L2 {popt.l2(fn):.2e}); geo brent-only {g:.2e} vs poly4 {U[tag]['poly4']['geo']:.2e} "
                  f"-> {'PATP still ahead' if g < U[tag]['poly4']['geo'] else 'poly4 ahead'}")
    if not any_diff:
        print("      none")

    print("H4  closed form vs quadrature of the same surrogate: largest prop50 and prop64 over the ten targets,")
    print("    per method (U[0,1])")
    for m in METHODS:
        v50 = max((U[t][m]["p50"], t) for t, *_ in targets if m in U[t])
        v64 = max((U[t][m]["p64"], t) for t, *_ in targets if m in U[t])
        print(f"      {m:<9} max prop50 = {v50[0]:.1e} ({v50[1]})   max prop64 = {v64[0]:.1e} ({v64[1]})")

    print("H5  geo ratios per target (for the text): PATP-opt/poly4 and best method among the 4-5 parameter ones")
    B = results[BETA25.name]
    small = ["poly3", "poly4", "PATP-opt", "PATP-a25", "sqrt3", "conf-opt"]
    for tag, *_ in targets:
        best = min(small, key=lambda m: U[tag][m]["geo"])
        print(f"      {tag}: U[0,1] PATP/poly4 = {U[tag]['PATP-opt']['geo'] / U[tag]['poly4']['geo']:.3g}, "
              f"poly4/PATP = {U[tag]['poly4']['geo'] / U[tag]['PATP-opt']['geo']:.3g}; "
              f"Beta(2,5) poly4/PATP = {B[tag]['poly4']['geo'] / B[tag]['PATP-opt']['geo']:.3g}; "
              f"best 4-5 param method (U) = {best} ({U[tag][best]['geo']:.2e}); "
              f"PATP-a25/PATP-opt = {U[tag]['PATP-a25']['geo'] / U[tag]['PATP-opt']['geo']:.3g}")


if __name__ == "__main__":
    sys.exit(main())
