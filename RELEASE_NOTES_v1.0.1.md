# v1.0.1 - Matched-Baseline Manuscript-Consistent Release

Planned release notes. This version has not been published.

## Protocol

- Matched ID-MLP-CA and SRAF-ID training on METR-LA and PEMS-BAY.
- Formal seeds `42-51`.
- Shared five-fault training rotation: CO24, RM40, GN-high, LD-high, and SV-high.
- RM20 retained as a test-only condition.
- Shared data splits, batch order, corruption seeds, optimization settings, validation protocol, early stopping, and test generator.

## Paper Model

- Public name: SRAF-ID.
- Internal formal implementation: `A3_mlp_only_softmax_no_profile`.
- Two-way softmax temporal-spatial fusion.
- No additional learned reliability gate.
- The saved formal artifacts show limited parameter growth (`0.17%`) but measurable latency overhead (`1.906x` and `2.184x`).

## Included

- Matched protocol config and runner.
- Protocol parity tests.
- Paper-facing S1-S6 CSV tables and consistency reports.
- Corrected Figure 4/6 source CSVs, 600-dpi PNGs, and a deterministic regeneration script.
- Dataset download and preprocessing instructions.

## Recorded Environment

- Python `3.9.23` and PyTorch `2.4.1+cu121`.
- CUDA 12.1 runtime on an NVIDIA GeForce RTX 4060 Laptop GPU.
- 13th Gen Intel Core i9-13900HX CPU and 64-bit Microsoft Windows 11.

## Excluded and Limitations

- Raw/processed datasets, checkpoints, and large logs are excluded.
- No real sensor-fault logs or disaster labels are included.
- Faults are controlled synthetic stress tests on historical observations.
- Evaluation covers one identity-enhanced forecasting backbone and two public traffic datasets.
- The release shows limited parameter growth but measurable latency overhead.
