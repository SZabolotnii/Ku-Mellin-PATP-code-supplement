# data/ — real-data input (not bundled)

The real-data case study (`experiments/realdata_case.py`, Table 5 / Section 8 of
the paper) uses vibration signals from the **Case Western Reserve University (CWRU)
Bearing Data Center**. That data is **not redistributed here** (size + source terms);
download it from the official, freely available source and drop the `.mat` files
into this folder.

## What to download

| File | CWRU record | Used as |
|---|---|---|
| `105.mat` | 12k Drive-End, Inner-Race fault 0.007″, load 1 hp | headline (faulty bearing) |
| `100.mat` | 12k Drive-End, Normal baseline | robustness check |

Official source: CWRU Bearing Data Center
<https://engineering.case.edu/bearingdatacenter/download-data-file>

Place the files so that:

```
data/105.mat
data/100.mat
```

## Run

```bash
# headline (faulty bearing)
python experiments/realdata_case.py
# robustness (healthy baseline)
python experiments/realdata_case.py --mat data/100.mat
```

Each script reads the drive-end accelerometer channel (`X1xx_DE_time`), forms the
per-window vibration **power** (mean square, window 256), min–max normalises it to
`(0,1]`, and propagates that real empirical distribution through the fractional-power
measurement models. No other preprocessing; the loader auto-detects the `*_DE_time`
variable.

The `.mat` files are git-ignored (see top-level `.gitignore`) so they are never
accidentally committed.
