# Data Download Guide

## Expected Raw Data Directory

Place raw public dataset files in:

```text
data/raw/
```

## Preferred Files

Preferred raw files:

- `metr-la.h5`
- `pems-bay.h5`

Expected placement:

```text
data/raw/metr-la.h5
data/raw/pems-bay.h5
```

## Alternative Zenodo Files

If the preferred `.h5` files are not available, use the CSV and adjacency files:

- `METR-LA.csv`
- `PEMS-BAY.csv`
- `adj_mx_METR-LA.pkl`
- `adj_mx_PEMS-BAY.pkl`

Expected CSV placement:

```text
data/raw/METR-LA.csv
data/raw/PEMS-BAY.csv
```

Expected adjacency placement:

```text
data/raw/adj_mx_METR-LA.pkl
data/raw/adj_mx_PEMS-BAY.pkl
```

## Integrity Notes

- Do not mark data as downloaded until the files actually exist in `data/raw/`.
- Do not claim real-data preprocessing or experiments have run unless output CSV, JSON, logs, or processed files exist.
- Synthetic smoke outputs are only for code validation.
- Synthetic smoke outputs must not be used as paper evidence, main results, or reproduced benchmark results.
- Any paper table value must be traceable to saved CSV, JSON, or logs generated from real experiment runs.
