"""Regenerate revised Figures 4, 5, and 8 with publication-facing labels."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "revision_final_20260730"
DATASET_LABELS = {"metr_la": "METR-LA", "pems_bay": "PEMS-BAY"}
VARIANT_LABELS = {
    "sraf_id_forecast_only": "SRAF-ID",
    "temporal_only_forecast_only": "Temporal-only",
    "spatial_only_forecast_only": "Spatial-only",
    "fixed_fusion_forecast_only": "Fixed fusion",
    "gated_fusion_forecast_only": "Gated fusion",
}


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    for suffix, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(OUT / "figures" / f"{name}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    overall = pd.read_csv(OUT / "tables" / "revised_main_results.csv")
    overall = overall[overall.input_setting == "faulty_average"].copy()
    overall["dataset_label"] = overall.dataset.map(DATASET_LABELS)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(overall)); width = 0.34
    ax.bar(x - width / 2, overall.reference_mean_mae, width, yerr=overall.reference_sample_sd, label="ID-MLP-CA", color="#8C8C8C", capsize=3)
    ax.bar(x + width / 2, overall.candidate_mean_mae, width, yerr=overall.candidate_sample_sd, label="SRAF-ID", color="#2878B5", capsize=3)
    ax.set_xticks(x, overall.dataset_label)
    ax.set_ylabel("Faulty-average MAE (mean ± sample SD)")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax.legend(frameon=False)
    save(fig, "revised_overall_mae")

    faultwise = pd.read_csv(OUT / "tables" / "revised_faultwise_results.csv")
    labels = faultwise.dataset.map(DATASET_LABELS) + " / " + faultwise.fault
    colors = np.where(faultwise.relative_reduction_percent >= 0, "#2878B5", "#C44E52")
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    ax.bar(np.arange(len(faultwise)), faultwise.relative_reduction_percent, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(faultwise)), labels, rotation=55, ha="right")
    ax.set_ylabel("Relative MAE reduction (%)\npositive = SRAF-ID better")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    save(fig, "revised_faultwise_gain")

    ablation = pd.read_csv(OUT / "tables" / "revised_architecture_ablation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, dataset in zip(axes, ("metr_la", "pems_bay")):
        part = ablation[ablation.dataset == dataset].copy()
        part["variant_label"] = part.variant.map(VARIANT_LABELS)
        colors = np.where(part.difference_vs_full >= 0, "#C44E52", "#2878B5")
        ax.barh(part.variant_label, part.difference_vs_full, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("MAE difference vs SRAF-ID\n(positive = worse)")
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    save(fig, "revised_architecture_ablation")


if __name__ == "__main__":
    main()
