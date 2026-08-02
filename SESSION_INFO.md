# Session / environment info

Last reproduced **2026-08-02** in a clean virtualenv (fresh `python -m venv`,
`pip install -r requirements.txt`, `python run_all.py`) with:

| Component | Version |
|---|---|
| Python | 3.13.12 |
| numpy | 2.5.1 |
| scipy | 1.18.0 |
| matplotlib | 3.11.1 |
| sympy | 1.14.0 |

All ten offline steps exited `ok`; the two real-data steps exit `2` with a pointer
to `data/README.md` when the CWRU file is absent, which is the intended behaviour.
With `105.mat` present, the real-data study reproduces Table 5 of the paper
(`+97.0%` / `+100%` / parity / `+97.9%`).

Platform: macOS (Darwin), CPU only. No network is needed except a one-time download
of the CWRU bearing data for the real-data case study (see `data/README.md`).

The paper's numbers were produced with Python 3.14.5; the versions above are the
clean-room re-check. `requirements.txt` is deliberately unpinned — the results are
analytic and do not depend on minor library versions, as the two runs confirm.

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
