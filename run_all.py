#!/usr/bin/env python3
"""
Run every reproducibility script and report which paper artifact each produces.

    python run_all.py                   # revised manuscript, then the submitted-version scripts
    python run_all.py --revision-only   # only the scripts behind the revised manuscript

Every script runs from its own directory (experiments/ or verification/cas/), as its
header says. Output convention for the revised manuscript (the paths of its Table
tab:repro):
  * experiments/results/<name>.txt  -- the script's stdout, captured here; the file is
    replaced only when the step exits 0;
  * verification/cas/<name>.txt     -- written by the CAS script itself.
Both are committed, so after a run `git diff` shows whether anything moved; only the
wall-clock lines ("elapsed", "wall time", "ms (min of 5)") are expected to change.
Figures go to outputs/ (git-ignored).

The real-data steps need the CWRU records in data/ (not bundled; see data/README.md).
They exit with code 2 and a pointer when the files are absent; nothing else depends on
them.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# (script, captured stdout -> file, manuscript artifact). Table/figure numbers follow the
# revised draft of 2026-09-26; the LaTeX labels are the stable reference.
REVISION = [
    ("verification/cas/mixed_kappa_validity.py", None,
     "Table 1 (tab:validity) + mixed-kappa counterexample, Sec. 5 -> verification/cas/mixed_kappa_validity.txt"),
    ("verification/cas/truncnormal_mellin.py", None,
     "Table 2 (tab:mellin-distros), App. A (app:tn) -> verification/cas/truncnormal_mellin.txt"),
    ("verification/cas/heavy_tail_admissibility.py", None,
     "Pareto example, Sec. 5.2 (sec:heavytail) -> console"),
    ("experiments/step0_capacity_matched.py", "experiments/results/step0.txt",
     "revision step 0: capacity-matched re-run (background; module imported by p1/p2/p3_balanced)"),
    ("experiments/step0_mechanism.py", "experiments/results/step0_mechanism.txt",
     "revision step 0: post-hoc mechanism diagnostic (background to Sec. 5.3)"),
    ("experiments/p1_confluent_limit.py", "experiments/results/p1_confluent.txt",
     "Corollary 10 (cor:removable), Figure 2 (fig:cond-alpha), Sec. 5.3 (sec:confluent)"
     " -> also results/p1_cond_vs_alpha.csv, outputs/fig_cond_alpha.pdf"),
    ("experiments/p3_reverify.py", "experiments/results/p3_reverify.txt",
     "Tables 3, 4, 6 (tab:rq3, tab:rq3-baselines, tab:twod), Sec. 6.4 (sec:precision)"),
    ("experiments/p3_balanced_benchmark.py", "experiments/results/p3_balanced.txt",
     "Table 5 (tab:balanced), Sec. 6.3"),
    ("experiments/p4_heavy_tail.py", "experiments/results/p4_heavy_tail.txt",
     "Table 7 (tab:heavytail-bench), Sec. 6.6"),
    ("experiments/p2_engineering_cb.py", "experiments/results/p2_engineering.txt",
     "Tables 8, 9 (tab:eng-q1, tab:eng-sweep), Figure 3 (fig:engineering), Sec. 7"
     " -> also outputs/fig_engineering.pdf"),
    ("experiments/realdata_split.py", "experiments/results/realdata_split.txt",
     "Sec. 8 (sec:realdata): Table 10, Figure 4; App. E: Tables 12, 13"
     " -> also outputs/fig_realdata.pdf; needs data/105.mat, data/106.mat (~3 min)"),
]

# Scripts of the submitted version, kept unchanged as its record; the revised manuscript
# takes none of its numbers from them (p3_reverify.py imports rq3_demo, twod_demo and
# alpha_sensitivity as modules and re-evaluates their surrogates in 50 digits).
SUBMITTED = [
    ("verification/cas/check_patp.py", None,
     "submitted: T1/T2/T3 + closed-form CAS cross-check (supporting; still valid)"),
    ("verification/cas/region_of_validity.py", None,
     "submitted: Table 1 (superseded by mixed_kappa_validity.py)"),
    ("verification/cas/mellin_table.py", None,
     "submitted: Table 2 closed-form rows (uniform, Beta, triangular); supporting"),
    ("verification/cas/multinomial_validity.py", None,
     "submitted: Sec. 5 multinomial scan (superseded by mixed_kappa_validity.py)"),
    ("experiments/rq3_demo.py", None, "submitted: Table 3 (superseded by p3_reverify.py)"),
    ("experiments/make_figure.py", None, "submitted: Figure 1 -> outputs/fig_rq3.pdf (not in the revision)"),
    ("experiments/twod_demo.py", None, "submitted: Table 4 (superseded by p3_reverify.py)"),
    ("experiments/jorder_stress.py", None, "submitted: Sec. 7 j-order stress (superseded by p3_reverify.py)"),
    ("experiments/alpha_sensitivity.py", None, "submitted: Sec. 7 alpha stability (superseded by p3_reverify.py)"),
    ("experiments/realdata_case.py", None,
     "submitted: Table 5 / Sec. 8, in-sample design (superseded by realdata_split.py; needs data/105.mat)"),
    ("experiments/mellin_error.py", None,
     "submitted: Sec. 8 empirical-Mellin error (not in the revision; needs data/105.mat)"),
]


def run_step(rel, capture, desc):
    path = os.path.join(HERE, rel)
    print(f"\n{'=' * 72}\n[RUN] {rel}\n      -> {desc}\n{'=' * 72}", flush=True)
    t0 = time.time()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    cwd = os.path.dirname(path)
    if capture is None:
        rc = subprocess.call([PY, path], cwd=cwd, env=env)
    else:
        out = os.path.join(HERE, capture)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".partial"
        with open(tmp, "w", encoding="utf-8") as fh:
            proc = subprocess.Popen([PY, path], cwd=cwd, env=env, stdout=subprocess.PIPE,
                                    text=True, encoding="utf-8")
            for line in proc.stdout:
                sys.stdout.write(line)
                fh.write(line)
            rc = proc.wait()
        if rc == 0:
            os.replace(tmp, out)
            print(f"[saved] {capture}")
        else:
            os.remove(tmp)
            print(f"[kept]  {capture} unchanged (step failed)")
    dt = time.time() - t0
    if rc != 0:
        print(f"[note] {rel} exited with code {rc}")
    return rc, dt


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--revision-only", action="store_true",
                    help="skip the scripts of the submitted version")
    args = ap.parse_args()
    os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)

    steps = REVISION + ([] if args.revision_only else SUBMITTED)
    t_all = time.time()
    results = [(rel,) + run_step(rel, cap, desc) for rel, cap, desc in steps]

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    for rel, rc, dt in results:
        tag = "ok" if rc == 0 else f"exit {rc}"
        print(f"  [{tag:>7}] {dt:7.1f} s  {rel}")
    print(f"  total {time.time() - t_all:.0f} s")
    bad = [r for r, c, _ in results if c != 0]
    print("\nAll steps completed." if not bad
          else f"\nCompleted; non-zero exits (missing real data gives exit 2): {bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
