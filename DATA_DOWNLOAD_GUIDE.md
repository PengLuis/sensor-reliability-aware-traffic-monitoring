# Data Download Guide

METR-LA and PEMS-BAY are public third-party datasets and are **not uploaded or redistributed in this repository**. Download them from their public sources and place the files under `data/raw/`.

## Public Sources

The speed datasets and the PEMS-BAY adjacency file are available from the public Zenodo record:

- METR-LA and PEMS-BAY traffic speed data: https://zenodo.org/records/5146275
- Original DCRNN repository and graph-generation instructions: https://github.com/liyaguang/DCRNN

The Zenodo record provides the commonly used files `metr-la.h5`, `pems-bay.h5`, and `adj_mx_bay.pkl`. Follow the source repository's terms and citation guidance.

## Required Placement

Create the raw-data directory:

```text
data/raw/
```

Place the downloaded speed files as:

```text
data/raw/metr-la.h5
data/raw/pems-bay.h5
```

Place the PEMS-BAY adjacency file as:

```text
data/raw/adj_mx_bay.pkl
```

For METR-LA spatial experiments, generate or obtain the public DCRNN-compatible adjacency pickle and place it as:

```text
data/raw/adj_mx_METR-LA.pkl
```

The preprocessing utilities also accept documented capitalization and underscore variants. The filenames above are recommended because they are discovered directly by the supplied scripts.

## Preprocessing

From the repository root, run:

```bash
python scripts/preprocess_data.py --dataset METR-LA --raw-dir data/raw --processed-dir data/processed
python scripts/preprocess_pems_bay.py --raw-dir data/raw --processed-dir data/processed
```

Processed outputs are written under `data/processed/` and are excluded from Git.

## Data Integrity Rules

- Do not commit raw or processed dataset files to this repository.
- Do not mark a dataset as downloaded until the expected files exist under `data/raw/`.
- Do not claim preprocessing or experiments have completed unless the corresponding output files exist.
- Synthetic smoke data may be used only for code checks, never as paper evidence.
- Every reported numerical result must remain traceable to saved CSV, JSON, or log artifacts.
