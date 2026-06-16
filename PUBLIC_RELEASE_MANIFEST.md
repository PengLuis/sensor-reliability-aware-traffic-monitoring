# Public Release Manifest

The repository is public and licensed under the MIT License.

## Version Status

- `v1.0.0`: preserved initial public package for provenance only; it is not the manuscript-aligned evidence version.
- `v1.0.1`: current manuscript-consistent matched-baseline release aligned with the 2026-06-16 Sensors submission files.

## Included

- Stable paper-facing model entry: `src/models/sraf_id.py`.
- Core source under `src/`.
- Experiment and audit scripts under `scripts/`.
- Formal matched protocol in `configs/matched_baseline_formal.yaml`.
- Corrected CSV dataset placement instructions in `DATA_DOWNLOAD_GUIDE.md`.
- Compact paper-facing CSV, JSON, and Markdown artifacts under `results/`, limited to the 2026-06-16 evidence chain.
- Corrected 600-dpi manuscript summary figures and their source CSVs under `results/paper_ready_figures/`.
- MIT `LICENSE` and `CITATION.cff`.

## Excluded

- Raw and processed datasets.
- Model checkpoints, binary masks, large per-run logs, and temporary experiments.
- Manuscript DOCX/PDF files and private author records.
- Third-party source code.
- Real-world sensor-fault logs or disaster labels.
