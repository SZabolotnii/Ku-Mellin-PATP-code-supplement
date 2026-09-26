# Session / environment info

## Version 2.0.0 (revised manuscript)

Reproduced **2026-09-26** with `python run_all.py` from this repository, using the
pinned versions of `requirements.txt`:

| Component | Version |
|---|---|
| Python | 3.14.5 |
| numpy | 2.4.5 |
| scipy | 1.17.1 |
| mpmath | 1.3.0 |
| sympy | 1.14.0 |
| matplotlib | 3.10.9 |

Platform: macOS on Apple silicon (arm64), CPU only.

All 22 steps exited `ok` with the CWRU records 105 and 106 present in `data/` (both hashes
`MATCH`). Wall time 240 s in total: 234 s for the eleven revision steps, of which 183 s is
`realdata_split.py` (the `B = 1000` moving-block bootstrap), and 6 s for the eleven
submitted-version steps.

Every text output of the revision steps was compared line by line with the output that
produced the manuscript's numbers. The numerical content is identical. The only lines
that differ are:

- wall-clock timings (`wall time … s` in `p2_engineering.txt`, `… ms (min of 5)` in
  `p1_confluent.txt`);
- figure paths, because figures now go to `outputs/` instead of the manuscript folder
  (`p1_confluent.txt`, `p2_engineering.txt`, `realdata_split.txt`);
- eight `fontTools` log lines (`'created' timestamp seems very low …`) that the original
  capture of `realdata_split.txt` took from stderr; `run_all.py` captures stdout only.

The pre-registration hash printed by `realdata_split.py`
(`634c2102947bc2d5b8ac8b2ed9e4aa8efaab77234632599661948ac63a6bfe94`) is unchanged: the
supplement edits only code below the docstring (the data, manifest and figure paths, and
a guard that exits with a pointer when the data is absent).

No network is needed except a one-time download of the CWRU records (see
`data/README.md`).

## Version 1.2.0 (submitted version)

Last reproduced **2026-08-02** in a clean virtualenv with Python 3.13.12, numpy 2.5.1,
scipy 1.18.0, matplotlib 3.11.1 and sympy 1.14.0 (unpinned requirements at the time); the
paper's numbers were produced with Python 3.14.5. With the pinned versions above the
submitted-version scripts still run; their float64 values on ill-conditioned sums differ
from the 2026-08-02 outputs in the last digits, and the `j = 4` cell of M4 in
`rq3_demo.py` changes sign (see README, "Scripts of the submitted version").

## Lean layer

| Component | Version |
|---|---|
| Lean | `leanprover/lean4:v4.26.0` (see `lean-toolchain`) |
| Mathlib | `v4.26.0` (pinned in `lake-manifest.json`) |

```bash
lake build PATP        # 3123 jobs on a cold cache
./check_no_sorry.sh    # OK: Lean/ is free of sorry/axiom/admit/native_decide/#exit
```

Reproduce the Python environment:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```
