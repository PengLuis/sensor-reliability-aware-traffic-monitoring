"""Regenerate the corrected manuscript Figures 4 and 6 at 600 dpi."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "paper_ready_tables"
OUTPUT = ROOT / "results" / "paper_ready_figures"
COLORS = {
    "ID-MLP-CA": "#EAA300",
    "SRAF-ID": "#087DB7",
    "KNN+ID-MLP": "#B8B8B8",
    "PPCA-lite+ID-MLP": "#8EA2CF",
    "PyPOTS-SAITS+ID-MLP": "#80C9C0",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#DDDDDD",
        }
    )


def figure4() -> None:
    data = pd.read_csv(TABLES / "table3_ten_seed_faulty_summary.csv")
    data.to_csv(OUTPUT / "figure4_average_faulty_mae_seed_summary.csv", index=False)
    indexed = data.set_index(["dataset", "model"])
    datasets = ["METR-LA", "PEMS-BAY"]
    models = ["ID-MLP-CA", "SRAF-ID"]
    x = np.arange(2)
    width = 0.32
    fig, ax = plt.subplots(figsize=(6.15, 3.65))
    for offset, model in zip((-width / 2, width / 2), models):
        means = [indexed.loc[(dataset, model), "faulty_mae_mean"] for dataset in datasets]
        stds = [indexed.loc[(dataset, model), "faulty_mae_std"] for dataset in datasets]
        ax.bar(x + offset, means, width, yerr=stds, capsize=4, color=COLORS[model], label=model)
    for idx, dataset in enumerate(datasets):
        baseline = indexed.loc[(dataset, "ID-MLP-CA"), "faulty_mae_mean"]
        sraf = indexed.loc[(dataset, "SRAF-ID"), "faulty_mae_mean"]
        gain = (baseline - sraf) / baseline * 100
        ax.text(idx, max(baseline, sraf) + 0.18, f"{gain:.3f}%", ha="center", fontsize=10)
    ax.set_ylabel("Average faulty-input MAE")
    ax.set_xticks(x, datasets)
    ax.set_ylim(0, 5.75)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure4_average_faulty_mae_vnext.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def figure6() -> None:
    data = pd.read_csv(TABLES / "supplementary_table_s5_imputation_forecasting.csv")
    data.to_csv(OUTPUT / "figure6_imputation_forecasting_seed_summary.csv", index=False)
    models = ["KNN+ID-MLP", "PPCA-lite+ID-MLP", "PyPOTS-SAITS+ID-MLP", "SRAF-ID"]
    short = ["KNN", "PPCA", "SAITS", "SRAF-ID"]
    fig, axes = plt.subplots(1, 2, figsize=(8.15, 3.05))
    for panel, (ax, dataset) in enumerate(zip(axes, ["METR-LA", "PEMS-BAY"])):
        subset = data[data["Dataset"] == dataset].set_index("Model").loc[models]
        means = subset["Average faulty-input MAE"].to_numpy()
        stds = subset["Standard deviation"].to_numpy()
        bars = ax.bar(
            np.arange(4),
            means,
            yerr=stds,
            capsize=3,
            color=[COLORS[model] for model in models],
            edgecolor="#555555",
        )
        ax.set_xticks(np.arange(4), short)
        ax.set_ylabel("Average faulty-input MAE")
        ax.set_title(f"({'a' if panel == 0 else 'b'}) {dataset}", loc="left", fontweight="bold")
        ax.set_ylim(0, max(means + stds) * 1.18)
        for bar, value in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, value + max(means) * 0.025, f"{value:.2f}", ha="center", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[model], ec="#555555") for model in models]
    fig.legend(handles, models, loc="upper center", ncol=4, frameon=True, fontsize=8, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=2.0)
    fig.savefig(OUTPUT / "figure6_imputation_forecasting_vnext.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure4()
    figure6()
    print(OUTPUT)


if __name__ == "__main__":
    main()
