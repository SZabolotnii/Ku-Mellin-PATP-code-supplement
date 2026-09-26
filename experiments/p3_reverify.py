"""
P3 Part I -- re-verification of every number printed in the manufactured-benchmark part
of paper/main.tex (PEM revision PREM-D-26-00488).

Scope (paper/main.tex):
  Sec. "Manufactured-solution experiments": the set-up text, the target list, tab:rq3
  (every cell), the "Outcome" paragraph, "Higher moment orders and conditioning",
  "Stability of the alpha-optimizer", "Determinism and replication", the separable 2-D
  subsection and tab:twod; plus the Pareto example of Sec. "Unbounded support and heavy
  tails" (P_3 = 0.6875, j P_3 = 1.375, agreement 1.8e-12).

Method. Every surrogate is fitted exactly as the paper's scripts fit it (rq3_demo.py,
jorder_stress.py, alpha_sensitivity.py, twod_demo.py, verification/cas/
heavy_tail_admissibility.py are imported, not modified). Its moments are then evaluated
(a) with the paper's float64 closed form, (b) with the same closed form in 50-digit
arithmetic (mpmath) from the same float64 coefficients and exponents, and (c) by 50-digit
tanh-sinh quadrature of the same surrogate; the truth E[f(X)^j] is 50-digit quadrature
of f. The recomputed value of a moment-based number is the 50-digit one (b), which (c)
confirms; (a) is printed wherever the paper's statement concerns float64 behaviour.

Verdict rules (fixed before the run):
  number        FLAG if the sign flips, a ranking flips, or
                |recomputed - published| > 0.05 |published| + (half a unit of the last
                printed digit)   -- i.e. > 5 % relative, allowing for the printed rounding.
  floor number  (|published| < 1e-9: optimiser/precision floors) FLAG unless
                0.1 <= recomputed / published <= 10 (same decade band).
  "exact"       holds if every relative error < 1e-9.
  "~1e-16" / "machine precision"   holds if < 100 eps = 2.2e-14.
  "~1e-k"       holds if 10^(-k-1) < value < 10^(-k+1).
  "a few times 1e-k"   holds if 1e-k <= value < 1e-(k-1).
  claims        stated TRUE/FALSE with the evidence printed; FALSE -> FLAG.
  "accurate in double precision" at order j: |float64 - exact| <= 0.1 |exact - truth|
                (float64 rounding changes the reported error by at most 10 %).

Run (from experiments/):
  ../verification/cas/.venv/bin/python p3_reverify.py | tee results/p3_reverify.txt
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy
from scipy import integrate
from scipy.linalg import solve_triangular
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "verification" / "cas"))

import alpha_sensitivity as asens  # noqa: E402
import heavy_tail_admissibility as hta  # noqa: E402
import p3_balanced_benchmark as pb  # noqa: E402
import rq3_demo as rq  # noqa: E402
import twod_demo as td  # noqa: E402

mp.mp.dps = 50
X = np.linspace(1e-6, 1.0, 401)
IDX = [2, 3, 4]
EPS100 = 100 * np.finfo(float).eps
U = pb.UNIFORM

TARGETS = {
    "M1": ("0.6 x^0.5 + 0.4 x^1.5", lambda x: 0.6 * x ** 0.5 + 0.4 * x ** 1.5,
           lambda x: mp.mpf("0.6") * mp.sqrt(x) + mp.mpf("0.4") * x ** mp.mpf("1.5")),
    "M2": ("x^0.7", lambda x: x ** 0.7, lambda x: x ** mp.mpf("0.7")),
    "M3": ("x^1.7", lambda x: x ** 1.7, lambda x: x ** mp.mpf("1.7")),
    "M4": ("sqrt(x)(1 + x^2/3)", lambda x: np.sqrt(x) * (1 + x ** 2 / 3), lambda x: mp.sqrt(x) * (1 + x ** 2 / 3)),
    "M5": ("x^3", lambda x: x ** 3, lambda x: x ** 3),
    "M6": ("ln(1 + x)", lambda x: np.log(1 + x), lambda x: mp.log(1 + x)),
    "M7": ("exp(-x)", lambda x: np.exp(-x), lambda x: mp.exp(-x)),
}

RECORDS: list[dict] = []
_TRUTH: dict = {}


def truth(tag: str, j: int):
    key = (tag, j)
    if key not in _TRUTH:
        _TRUTH[key] = pb.truth_moment(TARGETS[tag][2], U, j)
    return _TRUTH[key]


# ---------------------------------------------------------------------------
# Record keeping
# ---------------------------------------------------------------------------

def parse_pub(s: str) -> tuple[float, float]:
    s = s.strip().replace(",", "")
    mant, ex = (s.split("e") + ["0"])[:2] if "e" in s else (s, "0")
    dec = len(mant.split(".")[1]) if "." in mant else 0
    return float(s), 10.0 ** (int(ex) - dec)


def _emit(rid, loc, quote, pub, rec, rel, ok, note):
    RECORDS.append({"id": rid, "loc": loc, "quote": quote, "pub": pub, "rec": rec, "rel": rel,
                    "ok": ok, "note": note})
    print(f"[{rid}] {loc}")
    print(f"      quote: \"{quote}\"")
    print(f"      published: {pub:<22} recomputed: {rec:<26} rel.diff: {rel:<10} -> {'OK' if ok else 'FLAG'}"
          + (f"   ({note})" if note else ""))


def check_num(rid, loc, quote, pub_s, rec, note="", fmt="{:.3g}", ranking_ok=True):
    pub, ulp = parse_pub(pub_s)
    rel = abs(rec - pub) / abs(pub)
    if abs(pub) < 1e-9:
        ok = rec != 0 and 0.1 <= rec / pub <= 10
        note = ("floor quantity: decade rule; " + note).strip("; ")
    elif pub * rec < 0:
        ok = False
        note = ("SIGN FLIP; " + note).strip("; ")
    else:
        ok = abs(rec - pub) <= 0.05 * abs(pub) + 0.5 * ulp
    if not ranking_ok:
        ok = False
        note = ("RANKING FLIP; " + note).strip("; ")
    _emit(rid, loc, quote, pub_s, fmt.format(rec), f"{rel:.1%}", ok, note)
    return ok


def check_rule(rid, loc, quote, pub_s, values, rule, note=""):
    vals = list(values)
    if rule == "exact":
        ok = all(v < 1e-9 for v in vals)
    elif rule == "machine":
        ok = all(v < EPS100 for v in vals)
    elif rule.startswith("decade:"):
        k = float(rule.split(":")[1])
        ok = all(10 ** (k - 1) < v < 10 ** (k + 1) for v in vals)
    elif rule.startswith("few:"):
        k = float(rule.split(":")[1])
        ok = all(10 ** k <= v < 10 ** (k + 1) for v in vals)
    elif rule.startswith("upper:"):
        ok = all(v < float(rule.split(":")[1]) for v in vals)
    else:
        raise ValueError(rule)
    rec = ", ".join(f"{v:.2e}" for v in vals)
    _emit(rid, loc, quote, pub_s, rec, "-", ok, (f"rule {rule}; " + note).strip("; "))
    return ok


def check_claim(rid, loc, quote, pub_s, rec_s, ok, note=""):
    _emit(rid, loc, quote, pub_s, rec_s, "-", ok, note)
    return ok


# ---------------------------------------------------------------------------
# The paper's fits, turned into exact-arithmetic surrogates
# ---------------------------------------------------------------------------

def paper_fit(f, xatol=1e-6):
    poly = rq.fit_polynomial(f, 3, X)
    res = minimize_scalar(lambda a: rq.patp_fit_loss(a, f, X, IDX), bounds=(0.0, 1.0), method="bounded",
                          options={"xatol": xatol})
    a = float(res.x)
    par = rq.fit_patp(f, a, X, IDX)
    return poly, a, par, float(res.fun)


def sur_poly(coefs):
    return pb.Surrogate("poly3", pb.poly_terms(len(coefs) - 1), coefs, len(coefs))


def sur_patp(a, par):
    return pb.Surrogate("PATP", pb.patp_terms(a), [par["k0"]] + [par["ks"][i] for i in IDX], 5, ("a", a))


def rel(m, t) -> float:
    return float(abs(mp.mpf(m) - t) / abs(t))


def patp_f64(par, a, j):
    return rq.patp_moment(par["k0"], par["ks"], a, j)


def design_cond(a) -> float:
    return float(np.linalg.cond(np.column_stack([np.ones_like(X)] + [X ** rq.p_num(i, a) for i in IDX])))


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

LOC_RQ3 = "Sec. Manufactured-solution experiments, Table tab:rq3"
PUB_RQ3 = {
    "M1": ("6.9e-3", "3.7e-4", ["+92", "+98", "+75", "+2"], "+67"),
    "M4": ("1.2e-2", "2.0e-3", ["+79", "+97", "+85", "-16"], "+61"),
    "M6": ("2.7e-4", "1.0e-4", ["+37", "+77", "+32", "+78"], "+56"),
    "M7": ("1.2e-4", "1.6e-4", ["-57", "-168", "-342", "-12755"], "neg"),
    "M2": ("5.4e-3", "4.2e-11", "exact", "sanity"),
    "M3": ("8.4e-4", "1.7e-10", "exact", "sanity"),
    "M5": ("1.3e-16", "4.5e-11", "~1e-16", "poly exact"),
}


def section_rq3():
    print("\n" + "=" * 100)
    print("1. tab:rq3 -- the paper's pipeline (rq3_demo: S = 3, 401-point grid, Brent xatol 1e-6)")
    print("=" * 100)
    fits = {}
    for tag in ["M1", "M4", "M6", "M7", "M2", "M3", "M5"]:
        label, fn, fm = TARGETS[tag]
        poly, a, par, mse = paper_fit(fn)
        sp, sk = sur_poly(poly), sur_patp(a, par)
        l2p = rq.approx_l2_error(fn, lambda x: sum(poly[i] * x ** i for i in range(4)), X)
        l2k = sk.l2(fn)
        row = {"poly": poly, "a": a, "par": par, "mse": mse, "sp": sp, "sk": sk, "l2p": l2p, "l2k": l2k,
               "rp": [], "rk": [], "rp64": [], "rk64": [], "impr": [], "impr64": [], "qdiff": 0.0}
        print(f"\n--- {tag}: f = {label}   alpha* = {a:.10f}   grid MSE = {mse:.3e}   "
              f"L2-poly = {l2p:.4e}   L2-PATP = {l2k:.4e}")
        print(f"    PATP coefs k0,k2,k3,k4 = {[par['k0']] + [par['ks'][i] for i in IDX]}")
        print(f"    {'j':>2}{'truth (50d)':>22}{'rel poly f64':>14}{'rel poly 50d':>14}{'rel PATP f64':>14}"
              f"{'rel PATP 50d':>14}{'PATP 50d-quad':>15}{'impr f64':>11}{'impr 50d':>11}")
        for j in (1, 2, 3, 4):
            t = truth(tag, j)
            rp64, rk64 = rel(rq.polynomial_moment(poly, j), t), rel(patp_f64(par, a, j), t)
            mk = sk.moment_mp(j, U)
            rp, rk = rel(sp.moment_mp(j, U), t), rel(mk, t)
            qd = float(abs(mk - sk.quad_moment(j, U)) / abs(mk))
            row["qdiff"] = max(row["qdiff"], qd)
            i64 = (rp64 - rk64) / rp64 * 100
            ix = (rp - rk) / rp * 100
            for k, v in (("rp", rp), ("rk", rk), ("rp64", rp64), ("rk64", rk64), ("impr", ix), ("impr64", i64)):
                row[k].append(v)
            print(f"    {j:>2}{mp.nstr(t, 16):>22}{rp64:>14.3e}{rp:>14.3e}{rk64:>14.3e}{rk:>14.3e}{qd:>15.1e}"
                  f"{i64:>+10.1f}%{ix:>+10.1f}%")
        print(f"    avg improvement: float64 {np.mean(row['impr64']):+.1f}%   50-digit {np.mean(row['impr']):+.1f}%")
        fits[tag] = row

    print("\n--- checks against the printed tab:rq3 ---")
    for tag, (l2p_s, l2k_s, cells, avg_s) in PUB_RQ3.items():
        r = fits[tag]
        label = TARGETS[tag][0]
        pub_rank = parse_pub(l2p_s)[0] < parse_pub(l2k_s)[0]
        rec_rank = r["l2p"] < r["l2k"]
        loc = f"{LOC_RQ3}, row {tag} f = {label}"
        if tag == "M5":
            check_rule(f"RQ3-{tag}-L2poly", loc, f"L2-poly {l2p_s}", l2p_s, [r["l2p"]], "machine")
        else:
            check_num(f"RQ3-{tag}-L2poly", loc, f"L2-poly {l2p_s}", l2p_s, r["l2p"], ranking_ok=pub_rank == rec_rank)
        check_num(f"RQ3-{tag}-L2PATP", loc, f"L2-PATP {l2k_s}", l2k_s, r["l2k"], ranking_ok=pub_rank == rec_rank)
        if cells == "exact":
            check_rule(f"RQ3-{tag}-cells", loc, "j=1..4: exact", "exact", r["rk"], "exact",
                       "PATP rel. errors, 50-digit")
        elif cells == "~1e-16":
            check_rule(f"RQ3-{tag}-cells", loc, "j=1..4: approx 1e-16 (poly exact)", "~1e-16", r["rp"], "machine",
                       "poly3 rel. errors, 50-digit")
        else:
            for j, c in enumerate(cells, start=1):
                check_num(f"RQ3-{tag}-j{j}", loc, f"improvement j={j}: {c}%", c, r["impr"][j - 1],
                          note=f"float64 today {r['impr64'][j - 1]:+.1f}%", fmt="{:+.1f}")
            avg = float(np.mean(r["impr"]))
            if avg_s == "neg":
                check_claim(f"RQ3-{tag}-avg", loc, "avg: neg", "negative", f"{avg:+.1f}%", avg < 0)
            else:
                check_num(f"RQ3-{tag}-avg", loc, f"avg {avg_s}%", avg_s, avg,
                          note=f"float64 today {np.mean(r['impr64']):+.1f}%", fmt="{:+.1f}")
    return fits


def section_setup_and_outcome(fits):
    print("\n" + "=" * 100)
    print("2. Set-up text, target list and the Outcome paragraph")
    print("=" * 100)
    loc_setup = "Sec. Manufactured-solution experiments, first paragraph"
    sp, sk = fits["M1"]["sp"], fits["M1"]["sk"]
    check_claim("SET-budget", loc_setup, "both at S = 3, identical coefficient budget",
                "identical budget", f"fitted params poly3 {sp.n_params} vs PATP {sk.n_params} (4 coefs + alpha)",
                True, "literally true for the 4 linear coefficients; misleading as a capacity statement (R1-M2)")
    pp, _ = sp.mellin_cost()
    kp, _ = sk.mellin_cost()
    check_claim("SET-moments", "Sec. Manufactured-solution experiments, alpha-optimizer paragraph",
                "costs one real scalar, not additional moments of X", "no additional moments",
                f"Mellin values for j<=4: PATP {kp} vs poly3 {pp}", False,
                "PATP-MUET consumes 35 fractional moments of X where poly3 consumes 13 integer ones")

    loc_t = "Sec. Manufactured-solution experiments, 'Targets, categorized' (anti-test bullet)"
    a5 = fits["M5"]["a"]
    check_claim("TGT-M5-anti", loc_t, "anti-test (M5): f = x^3; polynomial basis contains f exactly",
                "poly basis exact", f"true; the PATP span contains x^3 too: p_4({a5:.6f}) = {rq.p_num(4, a5):.8f}, "
                f"and at alpha = 1 the span is {{1, x^2, x^3, x^4}}", True,
                "M5 is a sanity row for both bases, not an anti-test for PATP")
    # How far is the paper's scipy-quad reference from the 50-digit truth? (explains floor-level cells)
    print("    reference check: rq3_demo.truth_moment (scipy quad) vs 50-digit truth, max rel. error over j = 1..4")
    for tag in ("M1", "M4", "M6", "M7", "M2", "M3", "M5"):
        fn = TARGETS[tag][1]
        err = max(float(abs(rq.truth_moment(fn, j) - truth(tag, j)) / truth(tag, j)) for j in (1, 2, 3, 4))
        print(f"      {tag}: {err:.1e}")
    check_rule("TGT-M5-PATP", loc_t, "PATP carries the optimizer noise (a few times 10^{-12})", "few x 1e-12",
               fits["M5"]["rk"], "few:-12", "M5 PATP rel. errors j=1..4, 50-digit")
    check_rule("TGT-M5-poly", loc_t, "polynomial reaches machine precision (~10^{-16})", "~1e-16",
               fits["M5"]["rp"], "machine", "M5 poly3 rel. errors j=1..4, 50-digit")

    loc = "Sec. Manufactured-solution experiments, 'Outcome' paragraph"
    avgs = {t: float(np.mean(fits[t]["impr"])) for t in ("M1", "M4", "M6")}
    check_num("OUT-range-lo", loc, "PATP-MUET delivers 56%--67% average relative-error reduction (lower end)", "56",
              min(avgs.values()), note=f"avg M1 {avgs['M1']:+.1f}, M4 {avgs['M4']:+.1f}, M6 {avgs['M6']:+.1f}",
              fmt="{:+.1f}")
    check_num("OUT-range-hi", loc, "56%--67% (upper end)", "67", max(avgs.values()), fmt="{:+.1f}")
    cells = [(fits[t]["impr"][j], t, j + 1) for t in ("M1", "M4", "M6") for j in range(4)]
    lo, hi = min(cells), max(cells)
    check_num("OUT-min", loc, "per-moment reductions ... from -16% (M4, j = 4)", "-16", lo[0],
              note=f"minimum now at {lo[1]}, j = {lo[2]}", fmt="{:+.1f}")
    check_num("OUT-max", loc, "to +98% (M1, j = 2)", "+98", hi[0], note=f"maximum at {hi[1]}, j = {hi[2]}",
              fmt="{:+.1f}")
    r7 = fits["M7"]
    rank = r7["l2p"] < r7["l2k"]
    check_num("OUT-M7-L2poly", loc, "M7: L2-poly 1.2e-4", "1.2e-4", r7["l2p"], ranking_ok=rank)
    check_num("OUT-M7-L2PATP", loc, "vs L2-PATP 1.6e-4", "1.6e-4", r7["l2k"], ranking_ok=rank)
    check_claim("OUT-M7-loses", loc, "PATP loses on every moment (M7)", "PATP worse at j=1..4",
                ", ".join(f"{v:+.0f}%" for v in r7["impr"]), all(v < 0 for v in r7["impr"]))
    check_rule("OUT-sanity-res", loc, "The sanity rows M2/M3 ... show 10^{-10}-level residual", "~1e-10",
               [fits["M2"]["l2k"], fits["M3"]["l2k"]], "decade:-10", "L2-PATP of M2, M3")
    check_rule("OUT-M5-poly", loc, "polynomial at machine precision ~10^{-16}", "~1e-16", fits["M5"]["rp"], "machine")
    check_rule("OUT-M5-PATP", loc, "PATP at ~10^{-12}, the level of the alpha-optimizer tolerance", "~1e-12",
               fits["M5"]["rk"], "decade:-12")

    # The "two precision floors" explanation.
    third = mp.mpf(1) / 3
    a_exact = {"M2": mp.sqrt((mp.mpf("0.7") - third) * 3 / 8), "M3": mp.sqrt((mp.mpf("1.7") - third) * 3 / 8),
               "M5": (mp.mpf("0.75") + mp.sqrt(mp.mpf("0.5625") + 18 * mp.mpf("2.75"))) / 9}
    hit = {"M2": 3, "M3": 3, "M5": 4}
    print("\n    diagnostics for the 'two precision floors' sentence:")
    print(f"    {'tag':<4}{'alpha*':>14}{'alpha exact':>14}{'|d alpha|':>11}{'grid MSE':>11}{'max rel PATP':>14}"
          f"{'k_hit':>10}{'max|other k|':>14}")
    for t in ("M2", "M3", "M5"):
        r = fits[t]
        ks = [r["par"]["k0"]] + [r["par"]["ks"][i] for i in IDX]
        other = max(abs(k) for i, k in zip([0] + IDX, ks) if i != hit[t])
        print(f"    {t:<4}{r['a']:>14.10f}{float(a_exact[t]):>14.10f}{abs(r['a'] - float(a_exact[t])):>11.1e}"
              f"{r['mse']:>11.2e}{max(r['rk']):>14.2e}{r['par']['ks'][hit[t]]:>10.6f}{other:>14.1e}")
    check_rule("OUT-floor-21", loc, "two precision floors (10^{-21} in the LS residual of M2/M3", "~1e-21",
               [fits["M2"]["mse"], fits["M3"]["mse"]], "decade:-21", "grid MSE (the optimiser's loss) of M2, M3")
    same_floor = 0.1 <= fits["M5"]["mse"] / fits["M2"]["mse"] <= 10
    moments_same = max(max(fits["M2"]["rk"]), max(fits["M3"]["rk"])) > 1e-12
    check_claim("OUT-floor-expl", loc,
                "vs 10^{-12} in M5 PATP) arise because in M2/M3 the optimizer drives one k_i to exactly 1 ... "
                "whereas in M5 the optimizer must search for alpha", "two different floors",
                f"M5 MSE {fits['M5']['mse']:.1e} (same band as M2 {fits['M2']['mse']:.1e}); "
                f"M2/M3 moment errors up to {max(max(fits['M2']['rk']), max(fits['M3']['rk'])):.1e}",
                not (same_floor and moments_same),
                "M2, M3 and M5 all reach p_i(alpha*) = c by the same alpha search, all have one k ~ 1 and the "
                "rest ~ 0; the sentence compares a squared residual with a moment error")


def section_jorder():
    print("\n" + "=" * 100)
    print("3. 'Higher moment orders and conditioning' (jorder_stress: j = 1..8)")
    print("=" * 100)
    loc = "Sec. Manufactured-solution experiments, 'Higher moment orders and conditioning'"
    info = {}
    for tag in ("M1", "M4", "M6", "M7"):
        label, fn, fm = TARGETS[tag]
        poly, a, par, _ = paper_fit(fn)
        sp, sk = sur_poly(poly), sur_patp(a, par)
        cond = design_cond(a)
        kmax = max(abs(v) for v in par["ks"].values())
        rows = []
        print(f"\n--- {tag}: alpha* = {a:.4f}, cond(A) = {cond:.3e}, max|k_i| = {kmax:.3e}")
        print(f"    {'j':>2}{'rel poly 50d':>14}{'rel PATP f64':>14}{'rel PATP 50d':>14}{'50d vs quad':>13}"
              f"{'|f64-50d|/err':>15}")
        for j in range(1, 9):
            t = truth(tag, j)
            mk = sk.moment_mp(j, U)
            m64 = patp_f64(par, a, j)
            q = sk.quad_moment(j, U)
            rk = rel(mk, t)
            contam = float(abs(mp.mpf(m64) - mk) / abs(mk - t))
            rows.append({"j": j, "rp": rel(sp.moment_mp(j, U), t), "rk64": rel(m64, t), "rk": rk,
                         "qd": float(abs(mk - q) / abs(q)), "contam": contam})
            r = rows[-1]
            print(f"    {j:>2}{r['rp']:>14.3e}{r['rk64']:>14.3e}{r['rk']:>14.3e}{r['qd']:>13.1e}{contam:>15.2e}")
        safe = 0
        for r in rows:
            if r["contam"] <= 0.1:
                safe = r["j"]
            else:
                break
        print(f"    largest j with float64 accurate at every order <= j: {safe}")
        info[tag] = {"rows": rows, "cond": cond, "kmax": kmax, "safe": safe}

    qd = max(r["qd"] for t in ("M1", "M4", "M6") for r in info[t]["rows"])
    check_rule("HJ-exact", loc, "a 50-digit evaluation matches direct quadrature of the surrogate to full precision",
               "full precision", [qd], "upper:1e-15",
               "max over M1, M4, M6, j<=8; 'full precision' read as full double precision -- with coefficients "
               "~1e3 the 50-digit sum itself keeps ~25 digits at j = 8")
    behind = [(t, r["j"], r["rk"], r["rp"]) for t in ("M1", "M4", "M6") for r in info[t]["rows"] if r["rk"] >= r["rp"]]
    margins = sorted((r["rp"] / r["rk"], t, r["j"]) for t in ("M1", "M4", "M6") for r in info[t]["rows"])
    check_claim("HJ-ahead", loc, "keeps PATP-MUET ahead of polynomial MUET at every order", "ahead at all 24 (target, j)",
                f"behind at {len(behind)} of 24; narrowest margin x{margins[0][0]:.3f} at {margins[0][1]} j={margins[0][2]}",
                len(behind) == 0)
    r8 = info["M4"]["rows"][7]
    check_num("HJ-M4-j8", loc, "M4: relative error 2.9e-3 at j = 8", "2.9e-3", r8["rk"])
    check_num("HJ-M4-j8poly", loc, "vs polynomial 1.0e-2", "1.0e-2", r8["rp"])
    check_num("HJ-M4-cond", loc, "M4: cond(A) approx 9.7e4", "9.7e4", info["M4"]["cond"])
    check_num("HJ-M4-kmax", loc, "max_i |k_i| approx 1.2e3", "1.2e3", info["M4"]["kmax"])
    check_num("HJ-M6-cond", loc, "M6, ln(1+x): cond(A) approx 2.8e3", "2.8e3", info["M6"]["cond"])
    m6c = max(r["contam"] for r in info["M6"]["rows"])
    check_claim("HJ-M6-double", loc, "Well-conditioned fits stay accurate through j = 8 in double precision (M6)",
                "M6 float64 accurate to j=8", f"max |f64-50d|/err = {m6c:.1e} over j<=8", m6c <= 0.1)
    m4 = info["M4"]["rows"]
    check_claim("HJ-M4-cancel", loc, "In double precision ... the multinomial sum can cancel catastrophically at high j "
                "(M4)", "cancellation", f"M4 float64 rel err j=4..8: " +
                ", ".join(f"{r['rk64']:.1e}" for r in m4[3:]), m4[4]["rk64"] > 100 * m4[4]["rk"])
    safes = {t: info[t]["safe"] for t in info}
    check_claim("HJ-rule-j6", loc, "The practical rule is to keep j <= 6 in double precision", "j<=6 safe",
                "largest safe j: " + ", ".join(f"{t} {s}" for t, s in safes.items()), min(safes.values()) >= 6,
                f"M4 float64 rel. error today: j=4 {m4[3]['rk64']:.2e} (exact {m4[3]['rk']:.2e}), "
                f"j=5 {m4[4]['rk64']:.2e} (exact {m4[4]['rk']:.2e}); M1 cond(A) {info['M1']['cond']:.1e} "
                f"is off from j = {safes['M1'] + 1}")
    return info


def section_alpha():
    print("\n" + "=" * 100)
    print("4. 'Stability of the alpha-optimizer' (alpha_sensitivity settings, all j = 1..4, 50-digit and float64)")
    print("=" * 100)
    loc = "Sec. Manufactured-solution experiments, 'Stability of the alpha-optimizer'"
    worst_a, worst_rel, worst_rel64 = 0.0, (0.0, "", 0), (0.0, "", 0)
    for tag in ("M1", "M4", "M6"):
        label, fn, fm = TARGETS[tag]
        settings = [(f"xatol={x:.0e}", asens.alpha_scipy(fn, x)) for x in (1e-4, 1e-6, 1e-8)] + \
                   [(f"grid={n}", asens.alpha_grid(fn, n)) for n in (51, 101, 201, 401)]
        errs, errs64 = {j: [] for j in (1, 2, 3, 4)}, {j: [] for j in (1, 2, 3, 4)}
        print(f"\n--- {tag}")
        for name, a in settings:
            par = rq.fit_patp(fn, a, X, IDX)
            sk = sur_patp(a, par)
            line = f"    {name:<12} alpha* = {a:.6f}  rel 50d:"
            for j in (1, 2, 3, 4):
                t = truth(tag, j)
                errs[j].append(rel(sk.moment_mp(j, U), t))
                errs64[j].append(rel(patp_f64(par, a, j), t))
                line += f" {errs[j][-1]:.3e}"
            print(line + "   f64 j=4: " + f"{errs64[4][-1]:.3e}")
        al = [a for _, a in settings]
        spread_a = max(al) - min(al)
        worst_a = max(worst_a, spread_a)
        print(f"    alpha* spread {spread_a:.2e}")
        for j in (1, 2, 3, 4):
            s = (max(errs[j]) - min(errs[j])) / min(errs[j])
            s64 = (max(errs64[j]) - min(errs64[j])) / min(errs64[j])
            print(f"    j={j}: relative spread of the error, 50-digit {s:.1%}, float64 {s64:.1%}")
            if s > worst_rel[0]:
                worst_rel = (s, tag, j)
            if s64 > worst_rel64[0]:
                worst_rel64 = (s64, tag, j)
    check_rule("ALPHA-spread", loc, "alpha* varies by <= 4e-4", "<= 4e-4", [worst_a], "upper:4.5e-4",
               "max over M1, M4, M6; an upper bound, so the printed 4e-4 covers values < 4.5e-4")
    check_claim("ALPHA-err", loc, "and the propagated relative error by under 3% on M1/M4/M6", "< 3%",
                f"max spread 50-digit {worst_rel[0]:.1%} ({worst_rel[1]} j={worst_rel[2]}); float64 "
                f"{worst_rel64[0]:.1%} ({worst_rel64[1]} j={worst_rel64[2]})", worst_rel[0] < 0.03,
                "the published check (alpha_sensitivity.py) looked at j = 2 only; 'grid densities' there are "
                "alpha-grid densities of the deterministic optimiser, the fitting grid stays at 401 points")


def section_determinism(fits):
    print("\n" + "=" * 100)
    print("5. 'Determinism and replication' -- does tab:rq3 reproduce exactly? (M4, j = 4)")
    print("=" * 100)
    loc = "Sec. Manufactured-solution experiments, 'Determinism and replication'"
    label, fn, fm = TARGETS["M4"]
    r = fits["M4"]
    a = r["a"]
    t4 = truth("M4", 4)
    exps = [0.0] + [rq.p_num(i, a) for i in IDX]
    A = np.column_stack([X ** e for e in exps])
    y = fn(X)
    sol = {}
    sol["lstsq (paper)"] = np.linalg.lstsq(A, y, rcond=None)[0]
    Q, R = np.linalg.qr(A)
    sol["QR"] = solve_triangular(R, Q.T @ y)
    D = np.linalg.norm(A, axis=0)
    sol["scaled lstsq"] = np.linalg.lstsq(A / D, y, rcond=None)[0] / D
    Am = mp.matrix([[mp.mpf(float(v)) for v in row] for row in A])
    ym = mp.matrix([mp.mpf(float(v)) for v in y])
    cm = mp.lu_solve(Am.T * Am, Am.T * ym)
    sol["normal eq. 50d"] = np.array([float(c) for c in cm])
    print(f"    alpha* = {a:.10f}; truth E[f^4] = {mp.nstr(t4, 16)}")
    print(f"    {'solver':<16}{'k2':>16}{'k3':>18}{'k4':>16}{'rel j=4 f64':>13}{'rel j=4 50d':>13}")
    exact_vals, f64_vals = [], []
    for name, c in sol.items():
        par = {"k0": float(c[0]), "ks": {i: float(v) for i, v in zip(IDX, c[1:])}}
        s = sur_patp(a, par) if name != "normal eq. 50d" else None
        if s is None:
            s = pb.Surrogate("PATP", pb.patp_terms(a), c, 5)
            s._mc = [cm[i] for i in range(4)]  # keep the 50-digit LS solution itself
        m50 = s.moment_mp(4, U)
        m64 = patp_f64(par, a, 4)
        exact_vals.append(rel(m50, t4))
        f64_vals.append(rel(m64, t4))
        print(f"    {name:<16}{c[1]:>16.9f}{c[2]:>18.9f}{c[3]:>16.9f}{f64_vals[-1]:>13.3e}{exact_vals[-1]:>13.6e}")
    print(f"    published (results/rq3.txt, Aug 2026): rel PATP j=4 = 5.97e-04, improvement -15.6%")
    print(f"    spread of the exact j=4 error across solvers: {(max(exact_vals) - min(exact_vals)) / min(exact_vals):.1e}"
          f" (relative); float64 values range {min(f64_vals):.2e} .. {max(f64_vals):.2e}")
    check_num("DET-M4-j4", loc, "tab:rq3 M4 j=4 PATP rel. error (results/rq3.txt 5.97e-4, printed as -16%)",
              "5.97e-4", r["rk"][3], note=f"float64 today {r['rk64'][3]:.2e}")
    check_claim("DET-exact", loc, "The table therefore reproduces exactly rather than in distribution",
                "reproduces exactly", f"M4 j=4 float64: Aug 5.97e-4, today {r['rk64'][3]:.2e}, exact {r['rk'][3]:.2e}",
                False, "the float64 cell depends on the last digits of the LS coefficients (solver/BLAS)")


LOC_2D = "Sec. Separable multivariate propagation, Table tab:twod"
PUB_2D = {"A": {"PATP": ["2.4e-14", "1.8e-11", "2.1e-11", "9.4e-12"], "poly": ["4.2e-4", "5.5e-4", "2.1e-4", "3.2e-4"]},
          "B": {"PATP": ["1.7e-5", "4.6e-6", "3.0e-6", "2.0e-5"], "poly": ["2.4e-4", "2.7e-4", "1.8e-5", "4.5e-5"]}}
MP_2D = {"A": (lambda x: mp.sqrt(x), lambda x: x ** mp.mpf("0.7")),
         "B": (lambda x: mp.sqrt(x) * (1 + mp.mpf("0.4") * x), lambda x: mp.log(1 + x))}


def section_twod():
    print("\n" + "=" * 100)
    print("6. Separable 2-D propagation, tab:twod (twod_demo: Brent xatol 1e-8)")
    print("=" * 100)
    loc_txt = "Sec. Separable multivariate propagation, text"
    res = {}
    for (case, (name, g1, g2)) in zip(("A", "B"), td.CASES):
        m1, m2 = MP_2D[case]
        axes = []
        for g, gm in ((g1, m1), (g2, m2)):
            poly, a, par = td.fit_axis(g)
            axes.append((sur_poly(poly), sur_patp(a, par), a, par, poly, gm))
        out = {"PATP": [], "poly": [], "PATP64": [], "poly64": [], "exact": []}
        print(f"\n--- Case {name.strip()}   alpha1* = {axes[0][2]:.6f}, alpha2* = {axes[1][2]:.6f}")
        print(f"    {'j':>2}{'exact (50d)':>20}{'rel PATP 50d':>14}{'rel PATP f64':>14}{'rel poly 50d':>14}{'rel poly f64':>14}")
        for j in (1, 2, 3, 4):
            ex = mp.mpf(1)
            kp, pp, kp64, pp64 = mp.mpf(1), mp.mpf(1), 1.0, 1.0
            for sp, sk, a, par, poly, gm in axes:
                ex *= pb.truth_moment(gm, U, j)
                kp *= sk.moment_mp(j, U)
                pp *= sp.moment_mp(j, U)
                kp64 *= patp_f64(par, a, j)
                pp64 *= rq.polynomial_moment(poly, j)
            out["exact"].append(ex)
            out["PATP"].append(rel(kp, ex))
            out["poly"].append(rel(pp, ex))
            out["PATP64"].append(rel(kp64, ex))
            out["poly64"].append(rel(pp64, ex))
            print(f"    {j:>2}{mp.nstr(ex, 12):>20}{out['PATP'][-1]:>14.3e}{out['PATP64'][-1]:>14.3e}"
                  f"{out['poly'][-1]:>14.3e}{out['poly64'][-1]:>14.3e}")
        res[case] = out
        for meth in ("PATP", "poly"):
            for j, pub in enumerate(PUB_2D[case][meth], start=1):
                check_num(f"2D-{case}-{meth}-j{j}", f"{LOC_2D}, case {case}, {meth}", f"j={j}: {pub}", pub,
                          out[meth][j - 1], note=f"float64 {out[meth + '64'][j - 1]:.2e}")
    # Where do the published case-A values (~1e-11) come from? twod_demo's reference is scipy quad.
    print("\n    diagnosis of case A: the reference E[g1^j] E[g2^j] of twod_demo.py is scipy quad")
    name, g1, g2 = td.CASES[0]
    ref_err = []
    for j in (1, 2, 3, 4):
        e1 = abs(td.exact_axis_moment(g1, j) - pb.truth_moment(MP_2D["A"][0], U, j)) / pb.truth_moment(MP_2D["A"][0], U, j)
        e2 = abs(td.exact_axis_moment(g2, j) - pb.truth_moment(MP_2D["A"][1], U, j)) / pb.truth_moment(MP_2D["A"][1], U, j)
        (p1, a1, par1), (p2, a2, par2) = td.fit_axis(g1), td.fit_axis(g2)
        ex_sc = td.exact_axis_moment(g1, j) * td.exact_axis_moment(g2, j)
        kp = patp_f64(par1, a1, j) * patp_f64(par2, a2, j)
        ref_err.append(float(e1 + e2))
        print(f"    j={j}: scipy-reference rel. error g1 {float(e1):.1e}, g2 {float(e2):.1e};  "
              f"twod_demo today (float64 closed form vs scipy reference) {abs(kp - ex_sc) / ex_sc:.2e}")
    check_claim("2D-A-floor", loc_txt, "With PATP-exact factors ... the 2-D moments reach the optimizer floor (~10^{-11})",
                "PATP error ~1e-11", "PATP 50-digit vs exact reference: " +
                ", ".join(f"{v:.1e}" for v in res["A"]["PATP"]) + "; scipy-reference error: " +
                ", ".join(f"{v:.1e}" for v in ref_err), False,
                "the printed 1.8e-11, 2.1e-11, 9.4e-12 (j=2..4) are the error of the scipy-quad reference (mostly "
                "the x^0.7 factor), not of PATP-MUET; against an exact reference PATP is at 1e-15..1e-13")
    check_rule("2D-A-poly", loc_txt, "against polynomial MUET's ~10^{-4}", "~1e-4", res["A"]["poly"], "decade:-4")
    ratios = [p / k for p, k in zip(res["B"]["poly"], res["B"]["PATP"])]
    check_claim("2D-B-lead", loc_txt, "with non-trivial factors ... PATP-MUET leads by one to two orders of magnitude",
                "poly/PATP in [10, 100]", "ratios j=1..4: " + ", ".join(f"{r:.1f}" for r in ratios),
                all(r >= 10 for r in ratios))

    # Monte Carlo, replicated with twod_demo's seed and draw order.
    rng = np.random.default_rng(2026)
    nmc = 4_000_000
    inside = []
    print("\n    Monte-Carlo replication (seed 2026, N = 4e6, same draw order as twod_demo.py)")
    for (case, (name, g1, g2)) in zip(("A", "B"), td.CASES):
        x1, x2 = rng.random(nmc), rng.random(nmc)
        gp = g1(x1) * g2(x2)
        for j in (1, 2, 3, 4):
            gj = gp ** j
            mc, se = gj.mean(), gj.std(ddof=1) / np.sqrt(nmc)
            ex = float(res[case]["exact"][j - 1])
            ok = mc - 1.96 * se <= ex <= mc + 1.96 * se
            inside.append(ok)
            print(f"    case {case} j={j}: exact {ex:.8f}  CI [{mc - 1.96 * se:.7f}, {mc + 1.96 * se:.7f}]  "
                  f"{'inside' if ok else 'OUTSIDE'}")
    check_claim("2D-MC", loc_txt, "an N = 4e6 Monte-Carlo interval brackets the exact value at every order",
                "8/8 inside", f"{sum(inside)}/8 inside", all(inside))


def section_pareto():
    print("\n" + "=" * 100)
    print("7. Pareto example, Sec. 'Unbounded support and heavy tails', 'Reach on heavy tails'")
    print("=" * 100)
    loc = "Sec. Unbounded support and heavy tails, 'Reach on heavy tails'"
    a = Fraction(1, 4)
    pf = {i: Fraction(1, i) + (4 - i - Fraction(3, i)) * a + (2 * i - 4 + Fraction(2, i)) * a * a for i in IDX}
    P3 = max(pf.values())
    print(f"    exact p_i(1/4): {', '.join(f'p_{i} = {v}' for i, v in pf.items())};  P_3 = {P3} = {float(P3)}")
    check_num("PAR-P3", loc, "P_3 = 0.6875", "0.6875", float(P3), fmt="{:.6g}")
    check_num("PAR-jP3", loc, "so j P_3 = 1.375 < 4", "1.375", float(2 * P3), fmt="{:.6g}")
    nu, S, j = 4.0, 3, 2
    check_claim("PAR-poly", loc, "polynomial MUET already fails at j = 2 (it requests E[X^6] = infinity)",
                "jS = 6 >= b-1 = 4", f"jS = {j * S}, M_X({j * S + 1}) = {hta.mellin_pareto(j * S + 1, nu)}",
                j * S >= nu, "S = 3 here is the degree-3 polynomial; the reviewer's R1-M1 point on S vs S+1 applies")
    k0, k = 0.3, {2: 1.0, 3: -0.5, 4: 0.25}  # the coefficients hard-coded in hta.check_pareto
    cf = hta.closed_form(k0, k, 0.25, j, nu)

    def g(x):
        return k0 + sum(k[i] * x ** hta.p(i, 0.25) for i in k)

    qd, _ = integrate.quad(lambda x: g(x) ** j * nu * x ** (-nu - 1), 1.0, np.inf, limit=400)
    rel64 = abs(cf - qd) / abs(qd)
    pm = {i: mp.mpf(v.numerator) / v.denominator for i, v in pf.items()}
    km = {i: mp.mpf(v) for i, v in k.items()}
    k0m, num = mp.mpf("0.3"), mp.mpf(nu)

    def mellin(s):
        return num / (num - (s - 1))

    cfm = mp.mpf(0)
    terms = [(mp.mpf(0), k0m)] + [(pm[i], km[i]) for i in IDX]
    for c1 in range(4):
        for c2 in range(4):
            e = terms[c1][0] + terms[c2][0]
            cfm += terms[c1][1] * terms[c2][1] * mellin(e + 1)
    qm = mp.quad(lambda x: (k0m + mp.fsum(km[i] * x ** pm[i] for i in IDX)) ** 2 * num * x ** (-num - 1), [1, 2, mp.inf])
    rel50 = abs(cfm - qm) / abs(qm)
    print(f"    float64 closed form {cf:.15f}, scipy quad {qd:.15f}, rel {rel64:.2e}")
    print(f"    50-digit closed form {mp.nstr(cfm, 30)}, mp.quad {mp.nstr(qm, 30)}, rel {mp.nstr(rel50, 3)}")
    print(f"    float64 closed form vs 50-digit closed form: rel {float(abs(mp.mpf(cf) - cfm) / cfm):.2e}")
    check_num("PAR-agree", loc, "agreeing with adaptive quadrature to a relative 1.8e-12", "1.8e-12", rel64,
              note=f"50-digit closed form vs 50-digit quadrature: {mp.nstr(rel50, 2)} -- 1.8e-12 is the "
                   f"float64/quad floor; the surrogate coefficients (0.3; 1, -0.5, 0.25) are arbitrary, not fitted")


def section_revised():
    print("\n" + "=" * 100)
    print("8. Revised tab:rq3 data -- M1..M7, all methods of the P3 design (p3_balanced_benchmark.fit_methods),")
    print("   exact arithmetic, X ~ U[0,1]. Feeds revision/drafts/p3_rq3_table.tex.")
    print("=" * 100)
    rows = {}
    for tag in ("M1", "M4", "M6", "M7", "M2", "M3", "M5"):
        label, fn, fm = TARGETS[tag]
        surs = pb.fit_methods(fn)
        pb.print_fit_block(tag, label, "tab:rq3", surs, fn)
        tr, res = pb.evaluate(surs, fm, U)
        pb.print_moment_block(U, tr, res, surs)
        rows[tag] = (surs, res)
    print("\n  GEO-MEAN RELATIVE MOMENT ERROR, j = 1..4 (M1..M7)")
    print(f"  {'target':<8}" + "".join(f"{m:>11}" for m in pb.METHODS) + f"{'dm degree':>11}")
    for tag, (surs, res) in rows.items():
        dm = pb.by_name(surs, "poly-dm")
        print(f"  {tag:<8}" + "".join(f"{res[m]['geo']:>11.2e}" if m in res else f"{'-':>11}" for m in pb.METHODS)
              + f"{(dm.dm_degree if dm else '>12'):>11}")
    print("\n  geo ratios (for the text)")
    print(f"  {'target':<8}{'poly3/PATP':>12}{'poly4/PATP':>12}{'a25/PATP':>12}{'conf/PATP':>12}{'L2 conf/PATP':>14}")
    for tag, (surs, res) in rows.items():
        g = {m: res[m]["geo"] for m in res}
        lc = pb.by_name(surs, "conf-opt").l2(TARGETS[tag][1]) / pb.by_name(surs, "PATP-opt").l2(TARGETS[tag][1])
        print(f"  {tag:<8}{g['poly3'] / g['PATP-opt']:>12.3g}{g['poly4'] / g['PATP-opt']:>12.3g}"
              f"{g['PATP-a25'] / g['PATP-opt']:>12.3g}{g['conf-opt'] / g['PATP-opt']:>12.3g}{lc:>14.3f}")
    print("\n  largest propagation errors over M1..M7 (closed form vs quadrature of the same surrogate)")
    for m in pb.METHODS:
        v50 = max((res[m]["p50"], tag) for tag, (surs, res) in rows.items() if m in res)
        v64 = max((res[m]["p64"], tag) for tag, (surs, res) in rows.items() if m in res)
        print(f"  {m:<9} prop50 {v50[0]:.1e} ({v50[1]})   prop64 {v64[0]:.1e} ({v64[1]})")
    return rows


def summary():
    flags = [r for r in RECORDS if not r["ok"]]
    print("\n" + "=" * 100)
    print(f"SUMMARY: {len(RECORDS)} checks, {len(flags)} FLAG")
    print("=" * 100)
    for r in flags:
        print(f"  FLAG [{r['id']}] {r['loc']}")
        print(f"       \"{r['quote']}\": published {r['pub']} -> recomputed {r['rec']}"
              + (f"  ({r['note']})" if r["note"] else ""))


def main() -> int:
    print("P3 Part I -- re-verification of the manufactured-benchmark numbers of paper/main.tex")
    print(f"python {sys.version.split()[0]}, numpy {np.__version__}, scipy {scipy.__version__}, "
          f"mpmath {mp.__version__}, mp.dps = {mp.mp.dps}")
    fits = section_rq3()
    section_setup_and_outcome(fits)
    section_jorder()
    section_alpha()
    section_determinism(fits)
    section_twod()
    section_pareto()
    section_revised()
    summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
