#!/usr/bin/env python3
"""
Run every reproducibility script and report which paper artifact each produces.

    python run_all.py

The real-data step (experiments/realdata_case.py) needs the CWRU bearing file in
data/ (not bundled — see data/README.md). It exits cleanly with a pointer if the
file is absent; the rest of the pipeline does not depend on it.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

STEPS = [
    ("verification/cas/check_patp.py",          "T1/T2/T3 + Eq.(closed form) CAS cross-check"),
    ("verification/cas/region_of_validity.py",  "Table 1  (region of validity)"),
    ("verification/cas/mellin_table.py",        "Table 2  (closed-form Mellin transforms)"),
    ("verification/cas/multinomial_validity.py","Sec. 5   (multinomial-reduction scan, 765 pairs)"),
    ("experiments/rq3_demo.py",                 "Table 3  (manufactured-solution benchmark)"),
    ("experiments/make_figure.py",              "Figure 1 (-> outputs/fig_rq3.pdf)"),
    ("experiments/realdata_case.py",            "Table 4 / Sec. 8 (real data; needs data/105.mat)"),
]


def main():
    results = []
    for rel, desc in STEPS:
        path = os.path.join(HERE, rel)
        print(f"\n{'=' * 72}\n[RUN] {rel}\n      -> {desc}\n{'=' * 72}")
        rc = subprocess.call([PY, path], cwd=HERE)
        results.append((rel, rc))
        if rc != 0:
            print(f"[note] {rel} exited with code {rc}")

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    for rel, rc in results:
        tag = "ok" if rc == 0 else f"exit {rc}"
        print(f"  [{tag:>7}]  {rel}")
    bad = [r for r, c in results if c != 0]
    print("\nAll steps completed." if not bad
          else f"\nCompleted; non-zero exits (e.g. missing real data is expected): {bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
