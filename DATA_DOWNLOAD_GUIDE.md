# Data Download Guide

METR-LA and PEMS-BAY are public third-party datasets and are **not uploaded or redistributed in this repository**. Download them from their public sources and place the required files under `data/raw/`.

## Public Sources

The manuscript uses the traffic-speed data distributed through Zenodo record `5146275`:

- METR-LA and PEMS-BAY traffic-speed data: `https://zenodo.org/records/5146275`
- Original DCRNN repository and graph-generation instructions: `https://github.com/liyaguang/DCRNN`

The cited Zenodo record is the CSV distribution and provides the speed files `METR-LA.csv` and `PEMS-BAY.csv`, together with the PEMS-BAY adjacency file `adj_mx_bay.pkl`. Follow the source repositories' licensing and citation guidance.

The METR-LA adjacency pickle is not supplied by that Zenodo CSV record. Generate or obtain the public DCRNN-compatible METR-LA adjacency file using the DCRNN graph-generation procedure.

## Required Placement

Create the raw-data directory:

```text
data/raw/
```

Place the files as follows:

```text
data/raw/METR-LA.csv
data/raw/PEMS-BAY.csv
data/raw/adj_mx_METR-LA.pkl
data/raw/adj_mx_bay.pkl
```

The supplied preprocessing utilities also accept documented capitalization, hyphen, and underscore variants. The filenames above are recommended because they are discovered directly by the public scripts.

Alternative HDF5 mirrors can be read by the generic loader, but they are not the files distributed by the manuscript-cited Zenodo CSV record. Do not silently mix dataset versions or file sources when reproducing the reported results.

## Preprocessing

From the repository root, run:

```bash
python scripts/preprocess_data.py --dataset METR-LA --raw-dir data/raw --processed-dir data/processed
python scripts/preprocess_pems_bay.py --raw-dir data/raw --processed-dir data/processed
```

Processed outputs are written under `data/processed/` and are excluded from Git.

After preprocessing, verify the generated metadata files before starting formal experiments. Use the exact output paths printed by the preprocessing scripts as the source of truth.

## Data Integrity Rules

- Do not commit raw or processed dataset files.
- Do not mark a dataset as downloaded until the expected speed and adjacency files exist under `data/raw/`.
- Do not claim preprocessing or experiments have completed unless the corresponding output files exist.
- Do not substitute a newer or differently packaged dataset version without recording the change.
- Synthetic smoke data may be used only for code checks, never as manuscript evidence.
- Every reported numerical result must remain traceable to saved CSV, JSON, or log artifacts.
