# A Lightweight Sensor-Reliability-Aware Framework for Robust Traffic Monitoring under Missing and Faulty Sensor Observations

This repository is the public code and result-artifact package for the paper:

**A Lightweight Sensor-Reliability-Aware Framework for Robust Traffic Monitoring under Missing and Faulty Sensor Observations**

The work is positioned as robust sensor-network and IoT monitoring. It studies traffic forecasting when historical sensor observations are missing, noisy, drifting, continuously unavailable, or stuck at stale values.

## Included in v1.0.0

- `src/`: reusable dataset, fault-injection, metric, baseline, SRAF, seeding, and profiling code.
- `scripts/`: preprocessing, training, evaluation, ablation, audit, and paper-artifact entry points.
- `configs/`: model, experiment-matrix, and controlled-fault configurations.
- `results/`: compact CSV/JSON/Markdown artifacts supporting the reported tables, figures, and integrity checks.
- `DATA_DOWNLOAD_GUIDE.md`: public download sources, accepted filenames, and required placement for METR-LA and PEMS-BAY.

## Excluded from v1.0.0

- Raw or processed METR-LA and PEMS-BAY datasets.
- Model checkpoints, large per-seed outputs, full training logs, and temporary experiment directories.
- Manuscript DOCX/PDF files, author information, and private project notes.
- Third-party baseline source code. The optional PyPOTS-SAITS adapter requires the packages listed in `requirements-optional.txt`.

These exclusions keep the release small and avoid redistributing externally hosted datasets or third-party software.

## Installation

Python 3.9 or later is recommended. Create an isolated environment and install the core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the optional PyPOTS-SAITS comparison:

```bash
python -m pip install -r requirements-optional.txt
```

## Reproduction Outline

1. Download METR-LA and PEMS-BAY from the public sources in `DATA_DOWNLOAD_GUIDE.md` and place the files under `data/raw/`.
2. Preprocess METR-LA with `scripts/preprocess_data.py` and PEMS-BAY with `scripts/preprocess_pems_bay.py`.
3. Run the same-backbone baselines and SRAF-ID experiments with the configurations under `configs/`.
4. Run the formal multi-seed matrix with `scripts/run_sraf_v2_formal_10seed_matrix.py`; install `requirements-optional.txt` when reproducing the PyPOTS-SAITS baseline.
5. Regenerate summaries and paper-facing artifacts with the audit, summary, plotting, and packaging scripts under `scripts/`.
6. Compare regenerated outputs with the traceable files under `results/`.

The formal configuration snapshot is stored in `results/configs/formal_runner_config_snapshot.json`. Existing paper-facing values are included only where supporting CSV, JSON, or Markdown artifacts are present.

## Quick Code Check

The synthetic smoke path checks code execution only and must not be used as paper evidence:

```bash
python scripts/test_faults.py
python scripts/preprocess_data.py --synthetic-smoke --processed-dir data/processed
python scripts/run_sraf.py --dataset synthetic_smoke --data-dir data/processed/synthetic_smoke --epochs 1 --output-dir experiments/synthetic_smoke_public
```

## Integrity Notes

- Controlled faults are applied to model inputs, not forecast targets.
- Formal experiments use fixed random seeds and saved configurations.
- Only public datasets are used; no real disaster labels are required.
- Emergency or disaster conditions are application motivation only, not modeled physical processes.
- Missing evidence must be reported as missing rather than reconstructed or invented.

## License

The repository code is released under the MIT License. Dataset files and third-party packages remain subject to their original terms.

## Citation

A formal bibliographic entry will be added after publication metadata or a DOI becomes available.
