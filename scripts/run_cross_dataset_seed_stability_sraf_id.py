"""Run cross-dataset seed stability for SRAF-ID.

This script reuses seed-42 full-confirmation artifacts and trains only
ID-MLP-CA and SRAF-ID for additional seeds. Evaluation faults use a fixed
fault seed so the measured variation is training-seed variation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_full_no_reliability_gate_ablation import (  # noqa: E402
    FAULTS,
    FAULTY,
    SEVERE,
    compute_rdr,
    display_table,
    fmt6,
    load_dataset_payload,
    make_fault_inputs,
    safe_metrics,
    source_metric_row,
    write_csv,
    write_latex_table,
    write_markdown_table,
)
from scripts.run_metr_la_sraf_stid_same_backbone_gain import (  # noqa: E402
    FAULT_SETTINGS,
    build_official_stid,
    build_sraf_stid,
    model_param_count,
    predict_model,
    train_official_stid_ca,
    train_sraf_stid,
)
from scripts.run_metr_la_strong_clean_backbone_integration import resolve_device  # noqa: E402


SOURCE_MODELS = {
    "METR-LA": {
        "ID-MLP-CA": "OfficialStyleSTID-corruption-aware full-train",
        "SRAF-ID": "SRAF-OfficialStyleSTID-full full-train",
    },
    "PEMS-BAY": {"ID-MLP-CA": "ID-MLP-CA", "SRAF-ID": "SRAF-ID"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["metr-la", "pems-bay"], default=["metr-la", "pems-bay"])
    parser.add_argument("--metr-la-data-dir", default="data/processed/metr-la")
    parser.add_argument("--pems-bay-data-dir", default="data/processed/pems-bay")
    parser.add_argument("--metr-la-seed42-artifact-dir", default="experiments/metr-la-sraf-stid-full-training-confirmation")
    parser.add_argument("--pems-bay-seed42-artifact-dir", default="experiments/pems-bay-sraf-id-full-confirmation")
    parser.add_argument("--output-dir", default="experiments/cross-dataset-seed-stability-sraf-id")
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44])
    parser.add_argument("--include-existing-seed42", action="store_true")
    parser.add_argument("--fixed-fault-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--loss", choices=["mae", "mse"], default="mae")
    parser.add_argument("--lambda-repair", type=float, default=0.05)
    parser.add_argument("--lambda-rel", type=float, default=0.01)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def write_table_family(df: pd.DataFrame, out_dir: Path, stem: str, note: str | None = None) -> None:
    csv_df = df.copy()
    for col in csv_df.columns:
        if col not in {"Dataset", "Fault", "Model", "Metric", "Status", "Note", "Wins"}:
            csv_df[col] = csv_df[col].map(lambda v: fmt6(v) if isinstance(v, (float, int)) and not isinstance(v, bool) else v)
    csv_df.to_csv(out_dir / f"{stem}.csv", index=False)
    display = display_table(df)
    write_markdown_table(display, out_dir / f"{stem}.md", note)
    write_latex_table(display, out_dir / f"{stem}.tex", note)


def train_args(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    out = copy.copy(args)
    out.seed = seed
    if args.smoke:
        out.epochs = 2
        out.patience = 1
    return out


def source_seed42_rows(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = payload["name"]
    metrics_df = pd.read_csv(payload["artifact_dir"] / "metrics_by_model_fault.csv")
    complexity_df = pd.read_csv(payload["artifact_dir"] / "complexity_metrics.csv")
    metrics_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    for alias, source_model in SOURCE_MODELS[dataset].items():
        for setting in FAULTS:
            row = source_metric_row(metrics_df, source_model, setting)
            fault_setting = next(s for s in FAULT_SETTINGS if s["label"] == setting)
            metrics_rows.append(
                {
                    "dataset": dataset,
                    "seed": 42,
                    "model": alias,
                    "fault": setting,
                    "fault_type": fault_setting["fault"],
                    "severity_group": fault_setting["severity_group"],
                    "metrics_scale": "original",
                    "mape_safe_denominator": 1.0 if dataset == "PEMS-BAY" else "source_artifact_not_explicit",
                    **row,
                }
            )
            horizon_rows.append(
                {
                    "dataset": dataset,
                    "seed": 42,
                    "model": alias,
                    "fault": setting,
                    "h3_mae": row["mae_h3"],
                    "h6_mae": row["mae_h6"],
                    "h12_mae": row["mae_h12"],
                }
            )
        comp = complexity_df[complexity_df["model"] == source_model]
        if not comp.empty:
            r = comp.iloc[0]
            avg_col = "average_fault_inference_time_sec" if "average_fault_inference_time_sec" in comp.columns else "average_inference_time_sec"
            complexity_rows.append(
                {
                    "dataset": dataset,
                    "seed": 42,
                    "model": alias,
                    "parameter_count": float(r["parameter_count"]),
                    "training_time_sec": float(r["training_time_sec"]) if pd.notna(r.get("training_time_sec", np.nan)) else "TODO",
                    "best_epoch": float(r["best_epoch"]) if pd.notna(r.get("best_epoch", np.nan)) else "TODO",
                    "clean_inference_time_sec": float(r["clean_inference_time_sec"]),
                    "average_fault_inference_time_sec": float(r[avg_col]),
                    "source": "seed42_full_confirmation",
                }
            )
    return metrics_rows, horizon_rows, complexity_rows


def evaluate_model(
    payload: dict[str, Any],
    seed: int,
    model_name: str,
    kind: str,
    model: torch.nn.Module,
    fault_inputs: dict[str, np.ndarray],
    observed: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    adjacency: torch.Tensor,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str, int], float]]:
    metrics_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    inference_times: dict[tuple[str, str, int], float] = {}
    for fault in FAULTS:
        if kind == "ca":
            pred, infer_time, _ = predict_model(model, fault_inputs[fault], args.batch_size, device, sraf=False)
        elif kind == "sraf":
            pred, infer_time, _ = predict_model(
                model,
                fault_inputs[fault],
                args.batch_size,
                device,
                sraf=True,
                observed_mask=observed[fault],
                adjacency=adjacency,
                return_components=False,
            )
        else:
            raise ValueError(kind)
        if not np.isfinite(pred).all():
            raise ValueError(f"Non-finite predictions for {payload['name']} seed={seed} model={model_name} fault={fault}")
        m = safe_metrics(payload["test_y"], pred, payload["mean"], payload["std"])
        setting = next(s for s in FAULT_SETTINGS if s["label"] == fault)
        metrics_rows.append(
            {
                "dataset": payload["name"],
                "seed": seed,
                "model": model_name,
                "fault": fault,
                "fault_type": setting["fault"],
                "severity_group": setting["severity_group"],
                "metrics_scale": "original",
                "mape_safe_denominator": 1.0,
                "mae": m["mae"],
                "rmse": m["rmse"],
                "mape": m["mape"],
                "mae_h3": m["mae_h3"],
                "mae_h6": m["mae_h6"],
                "mae_h12": m["mae_h12"],
                "inference_time_sec": infer_time,
            }
        )
        horizon_rows.append(
            {
                "dataset": payload["name"],
                "seed": seed,
                "model": model_name,
                "fault": fault,
                "h3_mae": m["mae_h3"],
                "h6_mae": m["mae_h6"],
                "h12_mae": m["mae_h12"],
            }
        )
        inference_times[(payload["name"], fault, seed)] = infer_time
    return metrics_rows, horizon_rows, inference_times


def run_seed(payload: dict[str, Any], seed: int, args: argparse.Namespace, out_dir: Path, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    targs = train_args(args, seed)
    train_x = payload["add_identity"](payload["train_x"], payload["starts"]["train"])
    val_x = payload["add_identity"](payload["val_x"], payload["starts"]["val"])
    fault_args = copy.copy(args)
    fault_args.seed = args.fixed_fault_seed
    fault_inputs, _, observed, mask_checks = make_fault_inputs(payload, fault_args, out_dir)
    adjacency = torch.from_numpy(payload["adjacency"]).to(device)
    metrics_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []

    ca = build_official_stid(payload["train_x"].shape[2], payload["train_x"].shape[1], payload["train_y"].shape[1])
    ca_dir = out_dir / "models" / f"seed_{seed}" / "ID-MLP-CA"
    ca_dir.mkdir(parents=True, exist_ok=True)
    ca_meta, ca_curves = train_official_stid_ca(ca, train_x, payload["train_y"], val_x, payload["val_y"], targs, ca_dir, device)
    for row in ca_curves:
        row["dataset"] = payload["name"]
        row["seed"] = seed
        row["model"] = "ID-MLP-CA"
    training_rows.extend(ca_curves)
    m_rows, h_rows, infer = evaluate_model(payload, seed, "ID-MLP-CA", "ca", ca, fault_inputs, observed, args, device, adjacency)
    metrics_rows.extend(m_rows)
    horizon_rows.extend(h_rows)
    fault_times = [v for (_, fault, _), v in infer.items() if fault != "clean"]
    complexity_rows.append(
        {
            "dataset": payload["name"],
            "seed": seed,
            "model": "ID-MLP-CA",
            "parameter_count": model_param_count(ca),
            "training_time_sec": ca_meta["training_time_sec"],
            "best_epoch": ca_meta["best_epoch"],
            "clean_inference_time_sec": infer[(payload["name"], "clean", seed)],
            "average_fault_inference_time_sec": float(np.mean(fault_times)),
            "source": "trained_seed_stability",
        }
    )

    sraf = build_sraf_stid(payload["train_x"].shape[2], payload["train_x"].shape[1], payload["train_y"].shape[1], use_reliability_gate=True)
    sraf_dir = out_dir / "models" / f"seed_{seed}" / "SRAF-ID"
    sraf_dir.mkdir(parents=True, exist_ok=True)
    sraf_meta, sraf_curves = train_sraf_stid(sraf, "SRAF-ID", train_x, payload["train_y"], val_x, payload["val_y"], targs, sraf_dir, device, adjacency)
    for row in sraf_curves:
        row["dataset"] = payload["name"]
        row["seed"] = seed
        row["model"] = "SRAF-ID"
    training_rows.extend(sraf_curves)
    m_rows, h_rows, infer = evaluate_model(payload, seed, "SRAF-ID", "sraf", sraf, fault_inputs, observed, args, device, adjacency)
    metrics_rows.extend(m_rows)
    horizon_rows.extend(h_rows)
    fault_times = [v for (_, fault, _), v in infer.items() if fault != "clean"]
    complexity_rows.append(
        {
            "dataset": payload["name"],
            "seed": seed,
            "model": "SRAF-ID",
            "parameter_count": model_param_count(sraf),
            "training_time_sec": sraf_meta["training_time_sec"],
            "best_epoch": sraf_meta["best_epoch"],
            "clean_inference_time_sec": infer[(payload["name"], "clean", seed)],
            "average_fault_inference_time_sec": float(np.mean(fault_times)),
            "source": "trained_seed_stability",
        }
    )
    return metrics_rows, horizon_rows, complexity_rows, training_rows


def same_backbone_gain(metrics: pd.DataFrame, rdr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, seed, fault), group in metrics.groupby(["dataset", "seed", "fault"]):
        if not {"ID-MLP-CA", "SRAF-ID"}.issubset(set(group["model"])):
            continue
        ca = group[group["model"] == "ID-MLP-CA"].iloc[0]
        sraf = group[group["model"] == "SRAF-ID"].iloc[0]
        ca_r = rdr[(rdr.dataset == dataset) & (rdr.seed == seed) & (rdr.model == "ID-MLP-CA") & (rdr.fault == fault)].iloc[0]
        sf_r = rdr[(rdr.dataset == dataset) & (rdr.seed == seed) & (rdr.model == "SRAF-ID") & (rdr.fault == fault)].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "fault": fault,
                "id_mlp_ca_mae": float(ca.mae),
                "sraf_id_mae": float(sraf.mae),
                "delta_sraf_minus_ca": float(sraf.mae) - float(ca.mae),
                "relative_gain": (float(ca.mae) - float(sraf.mae)) / float(ca.mae),
                "id_mlp_ca_h12": float(ca.mae_h12),
                "sraf_id_h12": float(sraf.mae_h12),
                "h12_delta_sraf_minus_ca": float(sraf.mae_h12) - float(ca.mae_h12),
                "id_mlp_ca_rdr": float(ca_r.rdr_mae),
                "sraf_id_rdr": float(sf_r.rdr_mae),
                "rdr_reduction": float(ca_r.rdr_mae) - float(sf_r.rdr_mae),
                "sraf_better": float(sraf.mae) < float(ca.mae),
                "h12_better": float(sraf.mae_h12) < float(ca.mae_h12),
            }
        )
    return pd.DataFrame(rows)


def compute_seed_rdr(metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = {
        (r["dataset"], int(r["seed"]), r["model"]): float(r["mae"])
        for r in metrics_rows
        if r["fault"] == "clean"
    }
    rows: list[dict[str, Any]] = []
    for r in metrics_rows:
        key = (r["dataset"], int(r["seed"]), r["model"])
        clean_mae = clean[key]
        fault_mae = float(r["mae"])
        rows.append(
            {
                "dataset": r["dataset"],
                "seed": int(r["seed"]),
                "model": r["model"],
                "fault": r["fault"],
                "fault_type": r["fault_type"],
                "severity_group": r["severity_group"],
                "clean_mae": clean_mae,
                "fault_mae": fault_mae,
                "rdr_mae": (fault_mae - clean_mae) / clean_mae if clean_mae else "TODO",
            }
        )
    return rows


def aggregate_tables(out_dir: Path, metrics: pd.DataFrame, horizon: pd.DataFrame, rdr: pd.DataFrame, gain: pd.DataFrame, complexity: pd.DataFrame) -> dict[str, Any]:
    faulty_gain = gain[gain.fault != "clean"].copy()
    summary_rows = []
    for dataset in sorted(metrics.dataset.unique()):
        for fault in FAULTY:
            subset = faulty_gain[(faulty_gain.dataset == dataset) & (faulty_gain.fault == fault)]
            summary_rows.append(
                {
                    "Dataset": dataset,
                    "Fault": fault,
                    "Seeds": int(subset.seed.nunique()),
                    "ID-MLP-CA MAE mean": float(subset.id_mlp_ca_mae.mean()),
                    "ID-MLP-CA MAE std": float(subset.id_mlp_ca_mae.std(ddof=0)),
                    "SRAF-ID MAE mean": float(subset.sraf_id_mae.mean()),
                    "SRAF-ID MAE std": float(subset.sraf_id_mae.std(ddof=0)),
                    "Relative gain mean": float(subset.relative_gain.mean()),
                    "Relative gain std": float(subset.relative_gain.std(ddof=0)),
                    "Win count": int(subset.sraf_better.sum()),
                    "Mean win": bool(subset.sraf_id_mae.mean() < subset.id_mlp_ca_mae.mean()),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    write_csv(out_dir / "seed_stability_summary.csv", summary_rows)
    write_table_family(summary_df, out_dir, "table_seed_stability_main", "Mean/std across available seeds; lower MAE is better.")

    h12_rows = []
    for dataset in sorted(metrics.dataset.unique()):
        for fault in FAULTY:
            subset = faulty_gain[(faulty_gain.dataset == dataset) & (faulty_gain.fault == fault)]
            h12_rows.append(
                {
                    "Dataset": dataset,
                    "Fault": fault,
                    "ID-MLP-CA h12 mean": float(subset.id_mlp_ca_h12.mean()),
                    "SRAF-ID h12 mean": float(subset.sraf_id_h12.mean()),
                    "h12 delta mean": float(subset.h12_delta_sraf_minus_ca.mean()),
                    "h12 win count": int(subset.h12_better.sum()),
                    "Mean h12 win": bool(subset.sraf_id_h12.mean() < subset.id_mlp_ca_h12.mean()),
                }
            )
    write_table_family(pd.DataFrame(h12_rows), out_dir, "table_seed_stability_h12", "Negative h12 delta means SRAF-ID improves.")

    rdr_rows = []
    for dataset in sorted(metrics.dataset.unique()):
        for fault in FAULTY:
            subset = faulty_gain[(faulty_gain.dataset == dataset) & (faulty_gain.fault == fault)]
            rdr_rows.append(
                {
                    "Dataset": dataset,
                    "Fault": fault,
                    "ID-MLP-CA RDR mean": float(subset.id_mlp_ca_rdr.mean()),
                    "SRAF-ID RDR mean": float(subset.sraf_id_rdr.mean()),
                    "RDR reduction mean": float(subset.rdr_reduction.mean()),
                }
            )
    write_table_family(pd.DataFrame(rdr_rows), out_dir, "table_seed_stability_rdr", "RDR depends on each model's own clean MAE.")

    lat_rows = []
    for (dataset, model), group in complexity.groupby(["dataset", "model"]):
        lat_rows.append(
            {
                "Dataset": dataset,
                "Model": model,
                "Seeds": int(group.seed.nunique()),
                "Params mean": float(group.parameter_count.mean()),
                "Training time mean": float(group.training_time_sec.mean()),
                "Clean latency mean": float(group.clean_inference_time_sec.mean()),
                "Avg fault latency mean": float(group.average_fault_inference_time_sec.mean()),
            }
        )
    write_table_family(pd.DataFrame(lat_rows), out_dir, "table_seed_stability_latency", "Latency measured in each run environment; compare cautiously.")

    make_figures(out_dir, summary_df, pd.DataFrame(h12_rows), faulty_gain)

    mean_wins = int(summary_df["Mean win"].sum())
    severe_wins = int(summary_df[summary_df["Fault"].isin(SEVERE)]["Mean win"].sum())
    h12_mean_wins = int(pd.DataFrame(h12_rows)["Mean h12 win"].sum())
    avg_by_dataset = {}
    for dataset in sorted(metrics.dataset.unique()):
        vals = {}
        for model in ["ID-MLP-CA", "SRAF-ID"]:
            per_seed = metrics[(metrics.dataset == dataset) & (metrics.model == model) & (metrics.fault != "clean")].groupby("seed")["mae"].mean()
            vals[model] = {"mean": float(per_seed.mean()), "std": float(per_seed.std(ddof=0))}
        vals["sraf_better"] = vals["SRAF-ID"]["mean"] < vals["ID-MLP-CA"]["mean"]
        avg_by_dataset[dataset] = vals
    status = "PASS" if mean_wins >= 9 and severe_wins >= 6 and h12_mean_wins >= 9 and all(v["sraf_better"] for v in avg_by_dataset.values()) else (
        "PARTIAL" if mean_wins > 0 and any(v["sraf_better"] for v in avg_by_dataset.values()) else "FAIL"
    )
    return {
        "status": status,
        "mean_fault_wins": f"{mean_wins}/12",
        "mean_severe_wins": f"{severe_wins}/8",
        "mean_h12_wins": f"{h12_mean_wins}/12",
        "average_faulty_mae_by_dataset": avg_by_dataset,
    }


def make_figures(out_dir: Path, summary: pd.DataFrame, h12: pd.DataFrame, gain: pd.DataFrame) -> None:
    summary.to_csv(out_dir / "figure_seed_stability_fault_mae_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(summary))
    ax.bar(x - 0.2, summary["ID-MLP-CA MAE mean"], width=0.4, label="ID-MLP-CA")
    ax.bar(x + 0.2, summary["SRAF-ID MAE mean"], width=0.4, label="SRAF-ID")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.Dataset}\n{r.Fault}" for r in summary.itertuples()], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Mean MAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figure_seed_stability_fault_mae.png", dpi=200)
    fig.savefig(out_dir / "figure_seed_stability_fault_mae.svg")
    plt.close(fig)

    gain.to_csv(out_dir / "figure_seed_stability_relative_gain_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    means = gain[gain.fault != "clean"].groupby(["dataset", "fault"])["relative_gain"].mean().reset_index()
    ax.bar(range(len(means)), means["relative_gain"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels([f"{r.dataset}\n{r.fault}" for r in means.itertuples()], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Mean relative gain")
    fig.tight_layout()
    fig.savefig(out_dir / "figure_seed_stability_relative_gain.png", dpi=200)
    fig.savefig(out_dir / "figure_seed_stability_relative_gain.svg")
    plt.close(fig)

    h12.to_csv(out_dir / "figure_seed_stability_h12_gain_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(h12)), h12["h12 delta mean"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(h12)))
    ax.set_xticklabels([f"{r.Dataset}\n{r.Fault}" for r in h12.itertuples()], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Mean h12 delta")
    fig.tight_layout()
    fig.savefig(out_dir / "figure_seed_stability_h12_gain.png", dpi=200)
    fig.savefig(out_dir / "figure_seed_stability_h12_gain.svg")
    plt.close(fig)


def write_reports(out_dir: Path, summary: dict[str, Any], seeds_available: dict[str, list[int]], failed: list[dict[str, Any]]) -> None:
    lines = [
        "# Seed Stability Diagnostics",
        "",
        f"- Gate status: **{summary['status']}**",
        f"- Mean fault wins: `{summary['mean_fault_wins']}`",
        f"- Mean severe wins: `{summary['mean_severe_wins']}`",
        f"- Mean h12 wins: `{summary['mean_h12_wins']}`",
        f"- Seeds available by dataset: `{seeds_available}`",
        "",
        "## Average Faulty MAE",
    ]
    for dataset, vals in summary["average_faulty_mae_by_dataset"].items():
        lines.append(
            f"- {dataset}: ID-MLP-CA `{vals['ID-MLP-CA']['mean']:.6f}+/-{vals['ID-MLP-CA']['std']:.6f}`, "
            f"SRAF-ID `{vals['SRAF-ID']['mean']:.6f}+/-{vals['SRAF-ID']['std']:.6f}`, SRAF better `{vals['sraf_better']}`."
        )
    if failed:
        lines.extend(["", "## Failed or Skipped", ""])
        for row in failed:
            lines.append(f"- {row}")
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- This is a three-seed stability check when seeds 42/43/44 are available.",
            "- PEMS-BAY linear drift must remain visible if it stays inconsistent.",
            "- Stuck reliability behavior is not reinterpreted as solved by seed stability.",
        ]
    )
    (out_dir / "seed_stability_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    claims = [
        "# Manuscript-Safe Seed Stability Claims",
        "",
        "## Allowed",
        "",
        f"- Across evaluated seeds, SRAF-ID improves mean same-backbone robustness on `{summary['mean_fault_wins']}` dataset-fault pairs.",
        f"- Across evaluated seeds, SRAF-ID improves mean severe-fault MAE on `{summary['mean_severe_wins']}` severe dataset-fault pairs.",
        f"- Across evaluated seeds, SRAF-ID improves mean h12 MAE on `{summary['mean_h12_wins']}` faulty dataset-fault pairs.",
        "- Report the exact seeds and do not imply exhaustive stability.",
        "",
        "## Forbidden",
        "",
        "- Do not claim exhaustive or all-random-seed stability.",
        "- Do not claim official STID reproduction.",
        "- Do not claim clean SOTA.",
        "- Do not hide PEMS-BAY linear drift regression or inconsistency.",
        "- Do not claim stuck reliability detection is solved.",
    ]
    (out_dir / "manuscript_safe_seed_stability_claims.md").write_text("\n".join(claims) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.metr_la_full_artifact_dir = args.metr_la_seed42_artifact_dir
    args.pems_bay_full_artifact_dir = args.pems_bay_seed42_artifact_dir
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    seeds_to_train = [args.seed] if args.smoke else list(args.seeds)
    all_metrics: list[dict[str, Any]] = []
    all_horizon: list[dict[str, Any]] = []
    all_complexity: list[dict[str, Any]] = []
    all_training: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    seed_coverage: dict[str, list[int]] = {}

    for dataset_key in args.datasets:
        payload = load_dataset_payload(dataset_key, args)
        ddir = out_dir / dataset_key
        ddir.mkdir(parents=True, exist_ok=True)
        if args.include_existing_seed42 and not args.smoke:
            m, h, c = source_seed42_rows(payload)
            all_metrics.extend(m)
            all_horizon.extend(h)
            all_complexity.extend(c)
            seed_coverage.setdefault(payload["name"], []).append(42)
        for seed in seeds_to_train:
            try:
                torch.manual_seed(seed)
                np.random.seed(seed)
                m, h, c, t = run_seed(payload, seed, args, ddir / f"seed_{seed}", device)
                all_metrics.extend(m)
                all_horizon.extend(h)
                all_complexity.extend(c)
                all_training.extend(t)
                seed_coverage.setdefault(payload["name"], []).append(seed)
            except Exception as exc:
                failed.append({"dataset": payload["name"], "seed": seed, "status": "failed", "reason": repr(exc)})
                if not args.smoke:
                    raise

    rdr_rows = compute_seed_rdr(all_metrics)
    metrics_df = pd.DataFrame(all_metrics)
    horizon_df = pd.DataFrame(all_horizon)
    rdr_df = pd.DataFrame(rdr_rows)
    complexity_df = pd.DataFrame(all_complexity)
    gain_df = same_backbone_gain(metrics_df, rdr_df)

    write_csv(out_dir / "metrics_by_dataset_seed_model_fault.csv", all_metrics)
    write_csv(out_dir / "horizon_metrics_by_dataset_seed.csv", all_horizon)
    write_csv(out_dir / "rdr_by_dataset_seed.csv", rdr_rows)
    write_csv(out_dir / "same_backbone_gain_by_dataset_seed.csv", gain_df.to_dict("records"))
    write_csv(out_dir / "failed_or_skipped_models.csv", failed)
    write_csv(out_dir / "training_curves.csv", all_training)
    write_csv(out_dir / "latency_complexity_by_dataset_seed.csv", all_complexity)

    summary = aggregate_tables(out_dir, metrics_df, horizon_df, rdr_df, gain_df, complexity_df)
    write_reports(out_dir, summary, seed_coverage, failed)
    manifest = {
        "stage": "CROSS_DATASET_SEED_STABILITY_GATE",
        "status": "SMOKE" if args.smoke else summary["status"],
        "created_at": "2026-05-22",
        "datasets": args.datasets,
        "seeds_requested": seeds_to_train,
        "include_existing_seed42": bool(args.include_existing_seed42),
        "seed_coverage": seed_coverage,
        "fixed_fault_seed": args.fixed_fault_seed,
        "device_requested": args.device,
        "device_resolved": str(device),
        "training": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "loss": args.loss,
            "lambda_repair": args.lambda_repair,
            "lambda_rel": args.lambda_rel,
        },
        "target_corrupted": False,
        "identity_features_modified_by_sraf": False,
        "models_trained_for_additional_seeds": ["ID-MLP-CA", "SRAF-ID"],
        "summary": summary,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
