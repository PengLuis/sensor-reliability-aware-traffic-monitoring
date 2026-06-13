# SRAF-ID: A Low-Parameter Sensor-Reliability-Aware Framework for Robust Traffic Speed Forecasting under Missing and Faulty Sensor Observations

This repository contains the public code and compact result artifacts for the SRAF-ID Sensors manuscript. The study evaluates traffic speed forecasting under controlled missing, outage, noise, drift, and stuck-value corruption of historical sensor observations. It does not claim online detection of unknown faults or universal robustness to real sensor failures.

## Paper Model Mapping

- Paper name: `SRAF-ID`
- Implementation: `SRAFOfficialStyleSTIDWrapperFactorAblation` with `SRAFRepairFactorAblation`
- Internal formal-run name: `A3_mlp_only_softmax_no_profile`
- Fusion: learned two-way softmax over temporal and spatial repair candidates
- Additional learned reliability gate: none
- Observed-input blend: fixed at `0.5`
- Model-size overhead: about `0.17%` relative to matched ID-MLP-CA
- Full-test inference latency: `1.906x` on METR-LA and `2.184x` on PEMS-BAY in the saved formal artifacts

The framework is low-parameter, not low-latency.

## Recorded Execution Environment

- Python `3.9.23`
- PyTorch `2.4.1+cu121` with CUDA 12.1 runtime
- NVIDIA GeForce RTX 4060 Laptop GPU
- 13th Gen Intel Core i9-13900HX CPU
- 64-bit Microsoft Windows 11

## Included

- `src/`: models, faults, metrics, data utilities, and shared protocol loader.
- `scripts/`: preprocessing, matched training, evaluation, audit, and artifact-generation entry points.
- `configs/matched_baseline_formal.yaml`: the single source for the matched formal protocol.
- `results/paper_ready_tables/`: compact S1-S6 and manuscript-facing CSV artifacts.
- `results/audits/`: machine-readable and Markdown consistency reports.
- `DATA_DOWNLOAD_GUIDE.md`: public dataset sources and placement instructions.

## Excluded

- Raw and processed METR-LA or PEMS-BAY data.
- Model checkpoints, large per-run logs, and temporary experiment directories.
- Third-party source code and dataset files.
- Real sensor-fault logs or disaster labels.

## Installation

Python 3.9 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install optional PyPOTS-SAITS dependencies only when reproducing that comparison:

```bash
python -m pip install -r requirements-optional.txt
```

## Data Preparation

Download the public datasets described in `DATA_DOWNLOAD_GUIDE.md` and place them under `data/raw/`. Then run:

```bash
python scripts/preprocess_data.py --dataset METR-LA --raw-dir data/raw --processed-dir data/processed
python scripts/preprocess_pems_bay.py --raw-dir data/raw --processed-dir data/processed
```

## Matched Formal Protocol

- Datasets: METR-LA and PEMS-BAY.
- Seeds: `42-51`.
- Training rotation, in its saved formal-run order: CO24, RM40, GN-high, LD-high, SV-high.
- RM20 is test-only.
- Both ID-MLP-CA and SRAF-ID share split, batch order, fault sequence and seeds, optimizer, learning rate, weight decay, clipping, validation sets, early stopping, seed schedule, and test corruption generator.
- SRAF-ID alone adds the explicit repair front-end and auxiliary repair loss.

Inspect the planned matrix without launching the 10-seed run:

```bash
python scripts/run_matched_baseline_formal_10seed.py --dry-run
python scripts/run_sraf_id_final_figure_table_package.py --dry-run
```

Formal execution, after data preparation and resource review:

```bash
python scripts/run_matched_baseline_formal_10seed.py --device cuda
python scripts/run_sraf_id_final_figure_table_package.py --device cuda --skip-existing
python scripts/audit_seed_level_paired_statistics.py
python scripts/audit_manuscript_result_consistency.py --evidence-root .
python scripts/generate_manuscript_summary_figures.py
```

The final audit command regenerates the paper-facing Table 3 summary and Supplementary Tables S1-S6 when the complete saved artifacts are present. The public repository excludes large historical imputation, ablation, checkpoint, and per-run directories; therefore, a clean clone can verify the shipped compact artifacts but cannot reconstruct every comparison table without those excluded artifacts. Figure and table packaging for the formal SRAF-ID run is performed by `run_sraf_id_final_figure_table_package.py`.

After preprocessing, a reduced execution smoke test can be run without the historical comparison package:

```bash
python scripts/run_matched_baseline_formal_10seed.py --datasets METR-LA --seeds 42 --device cpu --epochs 1 --patience 1 --train-limit 128 --val-limit 64 --test-limit 64 --output-dir experiments/smoke_matched
python scripts/run_sraf_id_final_figure_table_package.py --datasets METR-LA --seeds 42 --device cpu --epochs 1 --patience 1 --train-limit 128 --val-limit 64 --test-limit 64 --train-only --output-dir experiments/smoke_sraf
```

These commands validate execution only and must not be used as manuscript evidence.

## Reproducibility Boundary

- Raw data, processed data, checkpoints, and large logs are intentionally not uploaded.
- Compact paper-facing CSV, JSON, and Markdown artifacts are uploaded.
- Synthetic smoke outputs check execution only and are not manuscript evidence.
- Controlled fault-location masks are used for auxiliary repair supervision and controlled-oracle imputation baselines.
- For finite-valued noise, drift, and stuck-value faults, fault-location masks are not inference inputs to SRAF-ID; the observation-availability mask remains one at finite positions.

## Quick Checks

```bash
python -m compileall src scripts
python scripts/test_faults.py
pytest -q tests/test_matched_protocol.py
```

## License and Citation

Code is released under the MIT License. Dataset and third-party package terms remain with their respective providers. Use `CITATION.cff` for the current manuscript citation metadata; publication DOI and final venue metadata remain unset until available.
