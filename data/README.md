# data/ — real-data input (not bundled)

The real-data case study of the revised paper (`experiments/realdata_split.py`, §8 and
Appendix E) uses vibration records from the **Case Western Reserve University (CWRU)
Bearing Data Center**, "12k Drive End Bearing Fault Data". That data is **not
redistributed here** (size + source terms); download it from the official, freely
available source and put the `.mat` files in this folder.

Official source: CWRU Bearing Data Center
<https://engineering.case.edu/bearingdatacenter/download-data-file>
Dataset citation: W. A. Smith, R. B. Randall, *Rolling element bearing diagnostics using
the Case Western Reserve University data: a benchmark study*, Mechanical Systems and
Signal Processing 64–65 (2015) 100–131, doi:10.1016/j.ymssp.2015.04.021.

## What to download

| File | CWRU label | Fault | Motor load | Speed | Channel used | Samples | Used by |
|---|---|---|---|---|---|---|---|
| `105.mat` | IR007_0 | inner race, 0.007 in | 0 hp | 1797 rpm | `X105_DE_time` | 121,265 | revised §8: train/test split (primary); train record of the cross-record check |
| `106.mat` | IR007_1 | inner race, 0.007 in | 1 hp | 1772 rpm | `X106_DE_time` | 121,991 | revised §8: test record of the cross-record check |
| `100.mat` | normal baseline | none (healthy) | 3 hp | 1725 rpm (`X100RPM`) | `X100_DE_time` | 485,643 | submitted version only (`realdata_case.py --mat data/100.mat`) |

Both revision records are sampled at 12,000 samples/s; only the drive-end accelerometer
channel is used.

## Integrity (SHA-256)

```
f80b0ea04fd06b372a0eaec7c056543ea37e4bb4727a5b173d2a5bacd2aa9cab  105.mat
e5cec7cdd138e6cd1deb9ed8634e5aaa9bc1bd7094ddc075bff606580eb6e883  106.mat
```

These two lines are [`MANIFEST.sha256`](MANIFEST.sha256). `realdata_split.py` hashes both
files on every run and prints `manifest: MATCH` (or `MISMATCH`) against it. To check by
hand:

```bash
cd data && shasum -a 256 -c MANIFEST.sha256     # or: sha256sum -c MANIFEST.sha256
```

Place the files so that:

```
data/105.mat
data/106.mat
```

or set `CWRU_DATA_DIR` to the folder that holds them.

## Run

```bash
cd experiments
python realdata_split.py                                  # revised §8 (~3 min)
CWRU_DATA_DIR=/path/to/cwru python realdata_split.py      # files kept elsewhere
```

`realdata_split.py` reads `X<rec>_DE_time`, cuts it into non-overlapping windows of 256
samples (the incomplete tail is dropped: 473 windows for 105, 476 for 106), and forms the
per-window power (mean square). No filtering, detrending, DC removal or resampling. The
primary analysis splits record 105 in time (windows 0–235 train, 236–472 test) and
min–max normalises with training constants only; the cross-record check trains on all of
105 and tests on all of 106. The full protocol is in the script's pre-registration
docstring.

The submitted version's scripts (`realdata_case.py`, `mellin_error.py`) read
`data/105.mat` by default and accept `--mat <path>`.

The `.mat` files are git-ignored (see top-level `.gitignore`) so they are never
accidentally committed.
