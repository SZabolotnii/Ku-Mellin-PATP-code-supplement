# Session / environment info

Last reproduced **2026-06-22** with:

| Component | Version |
|---|---|
| Python | 3.13.12 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| matplotlib | 3.10.9 |
| sympy | 1.14.0 |

Platform: macOS (Darwin). CPU only; no network needed except a one-time download
of the CWRU bearing data for the real-data case study (see `data/README.md`).

Reproduce the environment:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```
