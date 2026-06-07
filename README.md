# Lightweight Sensor-Reliability-Aware Traffic Monitoring

This repository contains the public code package for the paper project:

**A Lightweight Sensor-Reliability-Aware Framework for Robust Traffic Monitoring under Missing and Faulty Sensor Observations**

The project is positioned around sensor networks, IoT monitoring, intelligent sensing, and robust traffic sensor observation under missing, noisy, drifting, and faulty inputs.

## What Is Included

- `src/`: reusable Python implementation for datasets, simulated faults, metrics, baselines, SRAF models, seeding, and profiling.
- `scripts/`: preprocessing, training, evaluation, ablation, auditing, and table-generation entry points.
- `configs/`: default model, experiment, and fault protocol configurations.
- `results/`: small paper-facing CSV/MD artifacts used as traceable table, figure, and audit sources.
- `DATA_DOWNLOAD_GUIDE.md`: expected placement of public METR-LA and PEMS-BAY data files.

## What Is Not Included

The repository intentionally excludes raw datasets, processed datasets, model checkpoints, large per-run logs, Word/PDF manuscript drafts, and private project notes.

Raw public datasets should be downloaded by users and placed under:

```text
data/raw/
```

See `DATA_DOWNLOAD_GUIDE.md` for accepted file names.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The formal experiment scripts may require additional optional packages depending on which baseline adapters are enabled, such as `matplotlib`, `scikit-learn`, or PyPOTS-related dependencies.

## Quick Smoke Check

Use the synthetic smoke workflow only to check that the code runs. Synthetic outputs must not be used as paper evidence.

```bash
python scripts/test_faults.py
python scripts/run_sraf.py --dataset synthetic_smoke --data-dir data/processed/synthetic_smoke --epochs 1 --output-dir experiments/synthetic_smoke_public
```

## Reproduction Outline

1. Download public METR-LA and PEMS-BAY files into `data/raw/`.
2. Preprocess the datasets with the provided preprocessing scripts.
3. Run baseline and SRAF experiment scripts using the configs in `configs/`.
4. Regenerate summaries, tables, figures, and audits with the scripts under `scripts/`.
5. Compare regenerated artifacts against the CSV/MD files in `results/`.

The paper-facing result values in this package are traceable to saved CSV/JSON/MD artifacts under `results/`. Do not treat missing files as completed evidence.

## Main Entry Points

- `scripts/preprocess_data.py`
- `scripts/preprocess_pems_bay.py`
- `scripts/run_baseline.py`
- `scripts/run_sraf.py`
- `scripts/run_experiment_matrix.py`
- `scripts/run_sraf_v2_formal_10seed_matrix.py`
- `scripts/summarize_results.py`
- `scripts/plot_results.py`
- `scripts/profile_model.py`

## Integrity Notes

- Faults are applied to model inputs only, not to target `Y`.
- Random seeds are set in the training and experiment scripts.
- Public data only; no real disaster labels are required.
- Disaster or emergency settings are used only as application motivation for faulty sensor observations.
- If an experiment artifact is missing, mark it as missing rather than inventing values.

## Citation

A formal citation entry can be added after acceptance or DOI assignment.
