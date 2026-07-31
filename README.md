# SRAF-ID revised-manuscript evidence release v1.1.0

This release corresponds to the revised Sensors manuscript and supersedes v1.0.2 for revised-manuscript provenance. Release v1.0.2 is retained unchanged as the record of the original submission.

## Frozen model

- Variant: `sraf_id_forecast_only` (source-run name: `sraf_id_lambda000`)
- Objective: `total_loss = forecast_loss`
- `repair_loss_weight = 0.0`
- `observed_input_blend = 0.5` (fixed a priori; not claimed optimal)
- Controlled fault-location labels are not inputs to final training or inference.

## Contents

- `code/`: model, experiment, diagnostic, audit, and figure scripts.
- `configs/`: traceable forecast-only reference configurations for both datasets.
- `evidence/tables/`: final Tables 3-5 and Supplementary Tables S1-S9 source data.
- `evidence/figures/`: final figure exports and source-data-derived plots.
- `evidence/per_seed/`: ten-seed configurations and compact raw metric summaries.
- `evidence/audit/`: information-boundary, completeness, environment, and consistency checks.
- `evidence/summary/`: final run manifests.

## Reproduction boundary

Datasets, checkpoints, and large logs are intentionally excluded. Place the public METR-LA and PEMS-BAY processed arrays as documented by the project scripts, install the environment described in `evidence/audit/environment.json`, and run the relevant scripts under `code/tools/revision/`. The seed schedule is 42-51. Failed exploratory attempts are not accepted as evidence.

## Integrity check

Verify the files in this release against `RELEASE_MANIFEST_SHA256.csv`. Do not overwrite or relabel v1.0.2; it documents the original submission.
