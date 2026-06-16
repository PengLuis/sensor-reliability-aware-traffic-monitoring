"""Audit manuscript-facing values and generate traceable S1-S6 source tables."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FAULTS = [
    "random_missing_20",
    "random_missing_40",
    "continuous_outage_24",
    "gaussian_noise_high",
    "linear_drift_high",
    "stuck_at_last_value_high",
]
FAULT_LABELS = {
    "random_missing_20": "RM20",
    "random_missing_40": "RM40",
    "continuous_outage_24": "CO24",
    "gaussian_noise_high": "GN-high",
    "linear_drift_high": "LD-high",
    "stuck_at_last_value_high": "SV-high",
}
EXPECTED = {
    "METR-LA ID-MLP-CA faulty average": 5.1415,
    "METR-LA SRAF-ID faulty average": 4.8576,
    "METR-LA relative reduction": 5.521,
    "PEMS-BAY ID-MLP-CA faulty average": 1.9910,
    "PEMS-BAY SRAF-ID faulty average": 1.9542,
    "PEMS-BAY relative reduction": 1.849,
    "positive fault-wise MAE comparisons": 11,
    "positive horizon-12 comparisons": 11,
    "METR-LA parameter increase": 0.178,
    "PEMS-BAY parameter increase": 0.173,
    "METR-LA latency ratio": 1.906,
    "PEMS-BAY latency ratio": 2.184,
}


def gain(reference: float, value: float) -> float:
    return (reference - value) / reference * 100.0


def population_std(values: pd.Series) -> float:
    return float(np.std(values.to_numpy(dtype=float), ddof=0))


def load_per_seed(evidence_root: Path) -> pd.DataFrame:
    sraf_path = evidence_root / "experiments/sraf_id_final_figure_table_package/sraf_id_softmax_formal_per_seed_metrics.csv"
    id_path = evidence_root / "experiments/id_mlp_ca_matched_fault_distribution_10seed/aggregate/formal_10seed_per_seed_metrics.csv"
    if not sraf_path.exists() or not id_path.exists():
        raise FileNotFoundError("Public fallback required: internal per-seed experiment exports are not present.")
    frames = [pd.read_csv(sraf_path), pd.read_csv(id_path)]
    rows = pd.concat(frames, ignore_index=True, sort=False)
    required = {"dataset", "seed", "model", "fault", "mae", "rmse", "h12_mae"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Missing per-seed columns: {missing}")
    return rows


def seed_summary(rows: pd.DataFrame) -> pd.DataFrame:
    faulty = rows[rows["fault"].isin(FAULTS)].copy()
    by_seed = faulty.groupby(["dataset", "model", "seed"], as_index=False).agg(seed_faulty_mae=("mae", "mean"))
    return by_seed.groupby(["dataset", "model"], as_index=False).agg(
        faulty_mae_mean=("seed_faulty_mae", "mean"),
        faulty_mae_std=("seed_faulty_mae", population_std),
        seeds=("seed", "nunique"),
    )


def fault_table(rows: pd.DataFrame, metric: str, value_name: str) -> pd.DataFrame:
    selected = rows[rows["fault"].isin(FAULTS)]
    agg = selected.groupby(["dataset", "fault", "model"], as_index=False).agg(
        mean=(metric, "mean"), std=(metric, population_std), seeds=("seed", "nunique")
    )
    pivot = agg.pivot(index=["dataset", "fault"], columns="model", values=["mean", "std", "seeds"])
    output: list[dict[str, Any]] = []
    for (dataset, fault), row in pivot.iterrows():
        ref = float(row[("mean", "ID-MLP-CA")])
        val = float(row[("mean", "SRAF-ID")])
        output.append(
            {
                "Dataset": dataset,
                "Fault": FAULT_LABELS[fault],
                f"ID-MLP-CA {value_name} mean": ref,
                f"ID-MLP-CA {value_name} std": float(row[("std", "ID-MLP-CA")]),
                f"SRAF-ID {value_name} mean": val,
                f"SRAF-ID {value_name} std": float(row[("std", "SRAF-ID")]),
                "Relative gain (%)": gain(ref, val),
                "Seeds": int(row[("seeds", "SRAF-ID")]),
            }
        )
    return pd.DataFrame(output)


def horizon_table(rows: pd.DataFrame) -> pd.DataFrame:
    table = fault_table(rows, "h12_mae", "horizon-12 MAE")
    paired = rows[rows["fault"].isin(FAULTS)].pivot(
        index=["dataset", "fault", "seed"], columns="model", values="h12_mae"
    ).reset_index()
    wins = paired.assign(win=paired["SRAF-ID"] < paired["ID-MLP-CA"]).groupby(["dataset", "fault"])["win"].sum()
    table["Seed wins"] = [int(wins.loc[(row.Dataset, next(k for k, v in FAULT_LABELS.items() if v == row.Fault))]) for row in table.itertuples()]
    return table


def imputation_table(evidence_root: Path, sraf_seed_summary: pd.DataFrame) -> pd.DataFrame:
    files = sorted((evidence_root / "experiments/sraf_v2_main_formal_10seed_run/per_run").glob("*/metrics.csv"))
    old = pd.concat([pd.read_csv(path) for path in files], ignore_index=True, sort=False)
    models = ["KNN+ID-MLP", "PPCA-lite+ID-MLP", "PyPOTS-SAITS+ID-MLP"]
    old = old[old["model"].isin(models) & old["fault"].isin(FAULTS)]
    by_seed = old.groupby(["dataset", "model", "seed"], as_index=False).agg(seed_faulty_mae=("mae", "mean"))
    summary = by_seed.groupby(["dataset", "model"], as_index=False).agg(
        average_faulty_mae=("seed_faulty_mae", "mean"),
        standard_deviation=("seed_faulty_mae", population_std),
        seeds=("seed", "nunique"),
    )
    sraf = sraf_seed_summary[sraf_seed_summary["model"] == "SRAF-ID"].rename(
        columns={"faulty_mae_mean": "average_faulty_mae", "faulty_mae_std": "standard_deviation"}
    )[["dataset", "model", "average_faulty_mae", "standard_deviation", "seeds"]]
    summary = pd.concat([summary, sraf], ignore_index=True)
    sraf_means = sraf.set_index("dataset")["average_faulty_mae"]
    summary["relative_gain_of_sraf_id_pct"] = summary.apply(
        lambda row: 0.0 if row["model"] == "SRAF-ID" else gain(row["average_faulty_mae"], sraf_means.loc[row["dataset"]]), axis=1
    )
    summary["known_fault_location_use"] = summary["model"].map(
        {
            "KNN+ID-MLP": "Yes; controlled-oracle mask for imputation",
            "PPCA-lite+ID-MLP": "Yes; controlled-oracle mask for imputation",
            "PyPOTS-SAITS+ID-MLP": "Yes; controlled-oracle mask for imputation",
            "SRAF-ID": "Training repair loss only; not used for finite-valued-fault inference",
        }
    )
    return summary.rename(
        columns={
            "dataset": "Dataset",
            "model": "Model",
            "average_faulty_mae": "Average faulty-input MAE",
            "standard_deviation": "Standard deviation",
            "relative_gain_of_sraf_id_pct": "Relative gain of SRAF-ID (%)",
            "seeds": "Seeds",
            "known_fault_location_use": "Known fault-location use",
        }
    ).sort_values(["Dataset", "Model"])


def complexity_table(evidence_root: Path) -> pd.DataFrame:
    comp = pd.read_csv(evidence_root / "experiments/sraf_id_final_figure_table_package/combined_complexity_latency.csv")
    rows: list[dict[str, Any]] = []
    for dataset in ("METR-LA", "PEMS-BAY"):
        subset = comp[comp["dataset"] == dataset].set_index("model")
        ref = subset.loc["ID-MLP-CA"]
        val = subset.loc["SRAF-ID"]
        rows.append(
            {
                "Dataset": dataset,
                "ID-MLP-CA parameters": int(round(ref["parameter_count_mean"])),
                "SRAF-ID parameters": int(round(val["parameter_count_mean"])),
                "Parameter increase (%)": (val["parameter_count_mean"] / ref["parameter_count_mean"] - 1.0) * 100.0,
                "ID-MLP-CA latency (s)": ref["latency_mean_sec"],
                "SRAF-ID latency (s)": val["latency_mean_sec"],
                "Latency ratio": val["latency_mean_sec"] / ref["latency_mean_sec"],
                "Seeds": 10,
            }
        )
    return pd.DataFrame(rows)


def load_public_ready_tables(table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(table_dir / "table3_ten_seed_faulty_summary.csv")
    s2 = pd.read_csv(table_dir / "supplementary_table_s2_rmse.csv")
    s3 = pd.read_csv(table_dir / "supplementary_table_s3_faultwise_mae.csv")
    s4 = pd.read_csv(table_dir / "supplementary_table_s4_complexity_latency.csv")
    s5 = pd.read_csv(table_dir / "supplementary_table_s5_imputation_forecasting.csv")
    s6 = pd.read_csv(table_dir / "supplementary_table_s6_horizon12.csv")
    return summary, s2, s3, s4, s5, s6


def add_check(checks: list[dict[str, Any]], name: str, actual: float, source: str, decimals: int = 4) -> None:
    expected = EXPECTED[name]
    passed = round(float(actual), decimals) == round(float(expected), decimals)
    checks.append({"name": name, "expected": expected, "actual": float(actual), "passed": passed, "source": source})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/audits"))
    parser.add_argument("--table-dir", type=Path, default=Path("results/paper_ready_tables"))
    args = parser.parse_args()
    evidence_root = args.evidence_root.resolve()
    output_dir = args.output_dir.resolve()
    table_dir = args.table_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    try:
        rows = load_per_seed(evidence_root)
        summary = seed_summary(rows)
        summary.to_csv(table_dir / "table3_ten_seed_faulty_summary.csv", index=False)
        s2 = fault_table(rows, "rmse", "RMSE")
        s3 = fault_table(rows, "mae", "MAE")
        s4 = complexity_table(evidence_root)
        s5 = imputation_table(evidence_root, summary)
        s6 = horizon_table(rows)
        s2.to_csv(table_dir / "supplementary_table_s2_rmse.csv", index=False)
        s3.to_csv(table_dir / "supplementary_table_s3_faultwise_mae.csv", index=False)
        s4.to_csv(table_dir / "supplementary_table_s4_complexity_latency.csv", index=False)
        s5.to_csv(table_dir / "supplementary_table_s5_imputation_forecasting.csv", index=False)
        s6.to_csv(table_dir / "supplementary_table_s6_horizon12.csv", index=False)
        source = "formal per-seed metrics for matched ID-MLP-CA and SRAF-ID"
    except FileNotFoundError:
        summary, s2, s3, s4, s5, s6 = load_public_ready_tables(table_dir)
        source = "public paper-ready tables bundled in results/paper_ready_tables"

    s1_source = evidence_root / "seed_level_paired_statistics.csv"
    if s1_source.exists():
        shutil.copy2(s1_source, table_dir / "supplementary_table_s1_seed_level_paired_statistics.csv")

    checks: list[dict[str, Any]] = []
    summary_norm = summary.rename(columns={"faulty_mae_mean": "faulty_mae_mean", "faulty_mae_std": "faulty_mae_std"})
    index = summary_norm.set_index(["dataset", "model"])
    for dataset in ("METR-LA", "PEMS-BAY"):
        ref = float(index.loc[(dataset, "ID-MLP-CA"), "faulty_mae_mean"])
        val = float(index.loc[(dataset, "SRAF-ID"), "faulty_mae_mean"])
        add_check(checks, f"{dataset} ID-MLP-CA faulty average", ref, source, 4)
        add_check(checks, f"{dataset} SRAF-ID faulty average", val, source, 4)
        add_check(checks, f"{dataset} relative reduction", gain(ref, val), source, 3)
    add_check(checks, "positive fault-wise MAE comparisons", int((s3["Relative gain (%)"] > 0).sum()), "results/paper_ready_tables/supplementary_table_s3_faultwise_mae.csv", 0)
    add_check(checks, "positive horizon-12 comparisons", int((s6["Relative gain (%)"] > 0).sum()), "results/paper_ready_tables/supplementary_table_s6_horizon12.csv", 0)
    for _, row in s4.iterrows():
        add_check(checks, f"{row['Dataset']} parameter increase", row["Parameter increase (%)"], "results/paper_ready_tables/supplementary_table_s4_complexity_latency.csv", 3)
        add_check(checks, f"{row['Dataset']} latency ratio", row["Latency ratio"], "results/paper_ready_tables/supplementary_table_s4_complexity_latency.csv", 3)

    status = "PASS" if all(item["passed"] for item in checks) else "FAIL"
    report = {
        "status": status,
        "evidence_root": ".",
        "checks": checks,
        "generated_tables": sorted(str(path.relative_to(evidence_root)).replace("\\", "/") for path in table_dir.glob("*.csv")),
    }
    (output_dir / "MANUSCRIPT_RESULT_CONSISTENCY_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Manuscript Result Consistency Report", "", f"Status: **{status}**", "", "| Check | Expected | Actual | Pass | Source |", "|---|---:|---:|:---:|---|"]
    for item in checks:
        lines.append(f"| {item['name']} | {item['expected']} | {item['actual']:.6f} | {'PASS' if item['passed'] else 'FAIL'} | `{item['source']}` |")
    lines.extend(["", "All gains use `(baseline - SRAF-ID) / baseline * 100`. Faulty-input standard deviations are computed over ten seed-level six-fault averages (population SD, ddof=0), not over 60 pooled fault-seed observations."])
    (output_dir / "MANUSCRIPT_RESULT_CONSISTENCY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "checks": len(checks), "table_dir": str(table_dir)}, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
