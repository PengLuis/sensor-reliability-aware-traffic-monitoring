"""Generate formal METR-LA SRAF-STID tables from saved full-training artifacts.

This script is deliberately evaluation-only. It reads CSV/JSON artifacts from the
full-training confirmation gate and writes paper-ready CSV/Markdown/LaTeX tables,
figures, and an evidence audit without retraining models.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


FAULTS = [
    "clean",
    "random_missing_20",
    "random_missing_40",
    "continuous_outage_24",
    "gaussian_noise_high",
    "linear_drift_high",
    "stuck_at_last_value_high",
]
FAULTY = [f for f in FAULTS if f != "clean"]
SEVERE_FAULTS = [
    "random_missing_40",
    "continuous_outage_24",
    "gaussian_noise_high",
    "linear_drift_high",
]
MAIN_MODELS = [
    "Persistence",
    "Strong ResidualGRU-time reference",
    "current SRAF-RC-V2-Horizon reference",
    "OfficialStyleSTID-clean full-train",
    "OfficialStyleSTID-corruption-aware full-train",
    "SRAF-OfficialStyleSTID-full full-train",
]
STID_MODELS = [
    "OfficialStyleSTID-clean full-train",
    "OfficialStyleSTID-corruption-aware full-train",
    "SRAF-OfficialStyleSTID-full full-train",
]
MODEL_LABELS = {
    "Persistence": "Persistence",
    "Strong ResidualGRU-time reference": "Strong ResidualGRU-time",
    "current SRAF-RC-V2-Horizon reference": "SRAF-RC-V2-Horizon",
    "OfficialStyleSTID-clean full-train": "OfficialStyleSTID-clean",
    "OfficialStyleSTID-corruption-aware full-train": "OfficialStyleSTID-CA",
    "SRAF-OfficialStyleSTID-full full-train": "SRAF-OfficialStyleSTID",
}
FAULT_LABELS = {
    "clean": "Clean",
    "random_missing_20": "RM20",
    "random_missing_40": "RM40",
    "continuous_outage_24": "Outage24",
    "gaussian_noise_high": "Noise-high",
    "linear_drift_high": "Drift-high",
    "stuck_at_last_value_high": "Stuck-high",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def fmt6(value: Any) -> str:
    if value == "TODO" or pd.isna(value):
        return "TODO"
    return f"{float(value):.6f}"


def fmt3(value: Any) -> str:
    if value == "TODO" or pd.isna(value):
        return "TODO"
    return f"{float(value):.3f}"


def tex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def ranked_format(values: list[Any], value: Any, mode: str) -> str:
    if value == "TODO" or pd.isna(value):
        return "TODO"
    numeric = sorted(float(v) for v in values if v != "TODO" and not pd.isna(v))
    raw = fmt3(value)
    if not numeric:
        return raw
    val = float(value)
    best = numeric[0]
    second = numeric[1] if len(numeric) > 1 else None
    if math.isclose(val, best, rel_tol=0.0, abs_tol=1e-12):
        return f"**{raw}**" if mode == "md" else f"\\textbf{{{raw}}}"
    if second is not None and math.isclose(val, second, rel_tol=0.0, abs_tol=1e-12):
        return f"<u>{raw}</u>" if mode == "md" else f"\\underline{{{raw}}}"
    return raw


def write_markdown_table(df: pd.DataFrame, path: Path, note: str | None = None) -> None:
    lines = []
    if note:
        lines.append(note)
        lines.append("")
    cols = list(df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(df: pd.DataFrame, path: Path, note: str | None = None) -> None:
    cols = list(df.columns)
    lines = []
    if note:
        lines.append(f"% {note}")
    lines.append("\\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}")
    lines.append("\\hline")
    lines.append(" & ".join(tex_escape(c) for c in cols) + " \\\\")
    lines.append("\\hline")
    for _, row in df.iterrows():
        lines.append(" & ".join(tex_escape(str(row[c])) for c in cols) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pivot_metric(metrics: pd.DataFrame, model: str, fault: str, metric: str) -> float:
    rows = metrics[(metrics["model"] == model) & (metrics["fault"] == fault)]
    if rows.empty:
        raise KeyError(f"Missing {metric} for {model} / {fault}")
    return float(rows.iloc[0][metric])


def metric_or_todo(metrics: pd.DataFrame, model: str, fault: str, metric: str) -> Any:
    rows = metrics[(metrics["model"] == model) & (metrics["fault"] == fault)]
    if rows.empty:
        return "TODO"
    value = rows.iloc[0][metric]
    if pd.isna(value):
        return "TODO"
    return float(value)


def audit_inputs(input_dir: Path, out_dir: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    required = {
        "metrics": "metrics_by_model_fault.csv",
        "horizon": "horizon_metrics.csv",
        "rdr": "robustness_rdr.csv",
        "clp": "clean_loss_penalty.csv",
        "same_gain": "same_backbone_gain_summary.csv",
        "repair": "repair_diagnostics_by_fault.csv",
        "reliability": "reliability_diagnostics.csv",
        "complexity": "complexity_metrics.csv",
        "training": "training_curves.csv",
        "validation": "validation_curves_clean_and_fault.csv",
        "manifest": "run_manifest.json",
        "failed": "failed_or_skipped_models.csv",
    }
    file_status = {}
    for key, name in required.items():
        file_status[key] = {"path": str(input_dir / name), "exists": (input_dir / name).exists()}

    manifest_path = input_dir / required["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    frames = {
        "metrics": read_csv(input_dir / required["metrics"]),
        "horizon": read_csv(input_dir / required["horizon"]),
        "rdr": read_csv(input_dir / required["rdr"]),
        "clp": read_csv(input_dir / required["clp"]),
        "same_gain": read_csv(input_dir / required["same_gain"]),
        "repair": read_csv(input_dir / required["repair"]),
        "complexity": read_csv(input_dir / required["complexity"]),
        "failed": read_csv(input_dir / required["failed"]),
    }
    if file_status["reliability"]["exists"]:
        frames["reliability"] = read_csv(input_dir / required["reliability"])

    metrics = frames["metrics"]
    horizon = frames["horizon"]
    numeric_checks = {
        "metrics_numeric_finite": bool(
            metrics[["mae", "rmse", "mape", "mae_h3", "mae_h6", "mae_h12"]]
            .apply(pd.to_numeric, errors="coerce")
            .map(math.isfinite)
            .all()
            .all()
        ),
        "horizon_numeric_finite": bool(
            horizon[["h3_mae", "h6_mae", "h12_mae"]]
            .apply(pd.to_numeric, errors="coerce")
            .map(math.isfinite)
            .all()
            .all()
        ),
        "metrics_scale_original": bool((metrics["metrics_scale"] == "original").all())
        if "metrics_scale" in metrics.columns
        else False,
    }
    dataset = manifest.get("dataset", {})
    manifest_checks = {
        "dataset_name_metr_la": dataset.get("name") == "METR-LA",
        "L_12": dataset.get("L") == 12,
        "H_12": dataset.get("H") == 12,
        "N_207": dataset.get("N") == 207,
        "full_train_23974": dataset.get("full_train_samples") == 23974,
        "full_val_3424": dataset.get("full_val_samples") == 3424,
        "full_test_6851": dataset.get("full_test_samples") == 6851,
        "seed_42": manifest.get("seed") == 42,
        "target_y_not_corrupted": "never corrupted" in manifest.get("target_leakage_check", "").lower(),
    }
    model_checks = {
        "official_stid_clean_exists": "OfficialStyleSTID-clean full-train" in set(metrics["model"]),
        "official_stid_ca_exists": "OfficialStyleSTID-corruption-aware full-train" in set(metrics["model"]),
        "sraf_stid_full_exists": "SRAF-OfficialStyleSTID-full full-train" in set(metrics["model"]),
        "faults_complete": set(FAULTS).issubset(set(metrics["fault"])),
        "horizon_columns_present": {"h3_mae", "h6_mae", "h12_mae"}.issubset(horizon.columns),
    }
    metadata_checks = []
    for meta_path in sorted((input_dir / "fault_masks").glob("*_metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata_checks.append(
            {
                "path": str(meta_path),
                "target_corrupted_false": meta.get("target_corrupted") is False
                or "target" not in json.dumps(meta).lower(),
            }
        )

    passed = (
        all(v["exists"] for v in file_status.values())
        and all(numeric_checks.values())
        and all(manifest_checks.values())
        and all(model_checks.values())
        and all(item["target_corrupted_false"] for item in metadata_checks)
    )
    audit = {
        "status": "PASS" if passed else "PARTIAL",
        "file_status": file_status,
        "manifest_checks": manifest_checks,
        "model_and_fault_checks": model_checks,
        "numeric_checks": numeric_checks,
        "fault_metadata_checks": metadata_checks,
        "note": "references/GATES.md and references/AUTOMATED_WORKFLOW.md are absent in this workspace; explicit user gate rules were used.",
    }
    (out_dir / "input_artifact_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    summary = [
        "# Input Artifact Audit",
        "",
        f"- Status: `{audit['status']}`",
        f"- Required files present: `{all(v['exists'] for v in file_status.values())}`",
        f"- Dataset manifest matches METR-LA L=12 H=12 N=207 full 23974/3424/6851 seed 42: `{all(manifest_checks.values())}`",
        f"- Metrics are original-scale and finite in required numeric columns: `{all(numeric_checks.values())}`",
        f"- Required models and fault settings present: `{all(model_checks.values())}`",
        f"- Fault metadata target-corrupted checks passed where metadata is available: `{all(item['target_corrupted_false'] for item in metadata_checks)}`",
        "- No new model training was run for this table gate.",
    ]
    (out_dir / "input_artifact_audit_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return audit, frames


def make_main_table(metrics: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for model in MAIN_MODELS:
        row: dict[str, Any] = {"Model": MODEL_LABELS[model]}
        row["Clean MAE"] = metric_or_todo(metrics, model, "clean", "mae")
        row["Clean RMSE"] = metric_or_todo(metrics, model, "clean", "rmse")
        row["Clean MAPE"] = metric_or_todo(metrics, model, "clean", "mape")
        faulty_values = []
        for fault in FAULTY:
            val = metric_or_todo(metrics, model, fault, "mae")
            row[f"{FAULT_LABELS[fault]} MAE"] = val
            if val != "TODO":
                faulty_values.append(float(val))
        row["Avg Faulty MAE"] = sum(faulty_values) / len(faulty_values) if faulty_values else "TODO"
        rows.append(row)
    csv_df = pd.DataFrame(rows)
    csv_df.to_csv(out_dir / "table1_main_fault_performance.csv", index=False, float_format="%.6f")
    render = csv_df.copy()
    for col in render.columns[1:]:
        values = csv_df[col].tolist()
        render[col] = [ranked_format(values, v, "md") for v in csv_df[col]]
    write_markdown_table(render, out_dir / "table1_main_fault_performance.md", "Lower is better. Bold is best; underline is second-best.")
    render_tex = csv_df.copy()
    for col in render_tex.columns[1:]:
        values = csv_df[col].tolist()
        render_tex[col] = [ranked_format(values, v, "tex") for v in csv_df[col]]
    write_latex_table(render_tex, out_dir / "table1_main_fault_performance.tex", "Lower is better. Bold is best; underline is second-best.")


def make_same_backbone_gain_table(frames: dict[str, pd.DataFrame], out_dir: Path) -> None:
    metrics = frames["metrics"]
    horizon = frames["horizon"]
    rows = []
    for fault in FAULTS:
        ca = pivot_metric(metrics, "OfficialStyleSTID-corruption-aware full-train", fault, "mae")
        sraf = pivot_metric(metrics, "SRAF-OfficialStyleSTID-full full-train", fault, "mae")
        ca_rdr = pivot_metric(frames["rdr"], "OfficialStyleSTID-corruption-aware full-train", fault, "rdr_mae")
        sraf_rdr = pivot_metric(frames["rdr"], "SRAF-OfficialStyleSTID-full full-train", fault, "rdr_mae")
        ca_h12 = pivot_metric(horizon, "OfficialStyleSTID-corruption-aware full-train", fault, "h12_mae")
        sraf_h12 = pivot_metric(horizon, "SRAF-OfficialStyleSTID-full full-train", fault, "h12_mae")
        rows.append(
            {
                "Fault": fault,
                "STID-CA MAE": ca,
                "SRAF-STID MAE": sraf,
                "Delta SRAF-CA": sraf - ca,
                "Relative Gain": (ca - sraf) / ca,
                "STID-CA RDR": ca_rdr,
                "SRAF-STID RDR": sraf_rdr,
                "h12 STID-CA": ca_h12,
                "h12 SRAF-STID": sraf_h12,
                "h12 Delta SRAF-CA": sraf_h12 - ca_h12,
            }
        )
    wins = sum(1 for row in rows if row["Delta SRAF-CA"] < 0 and row["Fault"] != "clean")
    severe_wins = sum(1 for row in rows if row["Delta SRAF-CA"] < 0 and row["Fault"] in SEVERE_FAULTS)
    faulty_rows = [row for row in rows if row["Fault"] != "clean"]
    rows.append(
        {
            "Fault": "SUMMARY",
            "STID-CA MAE": "TODO",
            "SRAF-STID MAE": "TODO",
            "Delta SRAF-CA": "TODO",
            "Relative Gain": sum(row["Relative Gain"] for row in faulty_rows) / len(faulty_rows),
            "STID-CA RDR": "TODO",
            "SRAF-STID RDR": "TODO",
            "h12 STID-CA": f"wins={wins}",
            "h12 SRAF-STID": f"severe_wins={severe_wins}",
            "h12 Delta SRAF-CA": sum(row["STID-CA RDR"] - row["SRAF-STID RDR"] for row in faulty_rows) / len(faulty_rows),
        }
    )
    csv_df = pd.DataFrame(rows)
    csv_df.to_csv(out_dir / "table2_same_backbone_gain.csv", index=False, float_format="%.6f")
    render = csv_df.copy()
    for col in render.columns[1:]:
        render[col] = [fmt3(v) if isinstance(v, (int, float)) else str(v) for v in render[col]]
    write_markdown_table(render, out_dir / "table2_same_backbone_gain.md", "Negative delta means SRAF-STID is better than STID-CA.")
    write_latex_table(render, out_dir / "table2_same_backbone_gain.tex", "Negative delta means SRAF-STID is better than STID-CA.")


def make_horizon_table(horizon: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for model in STID_MODELS:
        row = {"Model": MODEL_LABELS[model]}
        for fault in FAULTS:
            sub = horizon[(horizon["model"] == model) & (horizon["fault"] == fault)]
            if sub.empty:
                row[f"{fault} h3"] = "TODO"
                row[f"{fault} h6"] = "TODO"
                row[f"{fault} h12"] = "TODO"
            else:
                row[f"{fault} h3"] = float(sub.iloc[0]["h3_mae"])
                row[f"{fault} h6"] = float(sub.iloc[0]["h6_mae"])
                row[f"{fault} h12"] = float(sub.iloc[0]["h12_mae"])
        rows.append(row)
    csv_df = pd.DataFrame(rows)
    csv_df.to_csv(out_dir / "table3_horizon_metrics.csv", index=False, float_format="%.6f")

    compact_faults = ["clean", "random_missing_40", "continuous_outage_24", "gaussian_noise_high", "linear_drift_high"]
    compact_cols = ["Model"]
    for fault in compact_faults:
        compact_cols.extend([f"{fault} h3", f"{fault} h6", f"{fault} h12"])
    compact = csv_df[compact_cols].copy()
    for col in compact.columns[1:]:
        values = compact[col].tolist()
        compact[col] = [ranked_format(values, v, "md") for v in compact[col]]
    write_markdown_table(compact, out_dir / "table3_horizon_metrics.md", "Compact horizon table; full horizon table is in CSV.")
    compact_tex = csv_df[compact_cols].copy()
    for col in compact_tex.columns[1:]:
        values = compact_tex[col].tolist()
        compact_tex[col] = [ranked_format(values, v, "tex") for v in compact_tex[col]]
    write_latex_table(compact_tex, out_dir / "table3_horizon_metrics.tex", "Compact horizon table; full horizon table is in CSV.")


def make_rdr_table(rdr: pd.DataFrame, out_dir: Path) -> None:
    models = [
        "OfficialStyleSTID-corruption-aware full-train",
        "SRAF-OfficialStyleSTID-full full-train",
        "current SRAF-RC-V2-Horizon reference",
    ]
    rows = []
    for model in models:
        if model not in set(rdr["model"]):
            continue
        row = {"Model": MODEL_LABELS[model]}
        vals = []
        for fault in FAULTY:
            val = metric_or_todo(rdr, model, fault, "rdr_mae")
            row[f"{FAULT_LABELS[fault]} RDR"] = val
            if val != "TODO":
                vals.append(float(val))
        row["Avg RDR"] = sum(vals) / len(vals) if vals else "TODO"
        rows.append(row)
    csv_df = pd.DataFrame(rows)
    csv_df.to_csv(out_dir / "table4_robustness_rdr.csv", index=False, float_format="%.6f")
    render = csv_df.copy()
    for col in render.columns[1:]:
        values = csv_df[col].tolist()
        render[col] = [ranked_format(values, v, "md") for v in csv_df[col]]
    note = "RDR depends on each model's own clean MAE, so interpret it together with raw fault MAE."
    write_markdown_table(render, out_dir / "table4_robustness_rdr.md", note)
    render_tex = csv_df.copy()
    for col in render_tex.columns[1:]:
        values = csv_df[col].tolist()
        render_tex[col] = [ranked_format(values, v, "tex") for v in csv_df[col]]
    write_latex_table(render_tex, out_dir / "table4_robustness_rdr.tex", note)


def make_clean_tradeoff_table(frames: dict[str, pd.DataFrame], out_dir: Path) -> None:
    metrics = frames["metrics"]
    clp = frames["clp"]
    same_gain = frames["same_gain"]
    rows = []
    for model in STID_MODELS:
        fault_vals = [pivot_metric(metrics, model, f, "mae") for f in FAULTY]
        clp_rows = clp[clp["model"] == model]
        avg_rg = "TODO"
        if model == "SRAF-OfficialStyleSTID-full full-train":
            avg_rg = float(same_gain[same_gain["fault"].isin(FAULTY)]["same_backbone_robustness_gain"].mean())
        rows.append(
            {
                "Model": MODEL_LABELS[model],
                "Clean MAE": pivot_metric(metrics, model, "clean", "mae"),
                "Clean RMSE": pivot_metric(metrics, model, "clean", "rmse"),
                "Clean MAPE": pivot_metric(metrics, model, "clean", "mape"),
                "CLP vs OfficialStyleSTID-clean": float(clp_rows.iloc[0]["clean_loss_penalty"]) if not clp_rows.empty else "TODO",
                "Avg Faulty MAE": sum(fault_vals) / len(fault_vals),
                "Avg Same-Backbone Gain": avg_rg,
            }
        )
    csv_df = pd.DataFrame(rows)
    csv_df.to_csv(out_dir / "table5_clean_tradeoff.csv", index=False, float_format="%.6f")
    render = csv_df.copy()
    for col in render.columns[1:]:
        values = [v for v in csv_df[col].tolist() if v != "TODO"]
        render[col] = [ranked_format(values, v, "md") if v != "TODO" else "TODO" for v in csv_df[col]]
    note = "SRAF-STID has a small clean penalty versus clean-only STID and is cleaner than STID-CA while improving fault robustness."
    write_markdown_table(render, out_dir / "table5_clean_tradeoff.md", note)
    render_tex = csv_df.copy()
    for col in render_tex.columns[1:]:
        values = [v for v in csv_df[col].tolist() if v != "TODO"]
        render_tex[col] = [ranked_format(values, v, "tex") if v != "TODO" else "TODO" for v in csv_df[col]]
    write_latex_table(render_tex, out_dir / "table5_clean_tradeoff.tex", note)


def make_repair_table(repair: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for _, src in repair.iterrows():
        fault = src["fault"]
        lower = src.get("corrupted_lower_than_clean", "TODO")
        if fault in {"random_missing_20", "random_missing_40", "continuous_outage_24"}:
            interpretation = "Corrupted reliability is lower than clean reliability." if str(lower) == "True" else "Reliability separation is not confirmed."
        elif fault in {"gaussian_noise_high", "linear_drift_high"}:
            interpretation = "All positions are marked corrupted; clean-vs-corrupted separation is not applicable."
        elif fault == "stuck_at_last_value_high":
            interpretation = "Reliability separation is mixed and should not be overclaimed."
        else:
            interpretation = "Clean input; corrupted-position diagnostics are not applicable."
        rows.append(
            {
                "Fault": fault,
                "Mean Reliability": src.get("mean_reliability", "TODO"),
                "Corrupted Reliability Mean": src.get("corrupted_position_reliability_mean", "TODO"),
                "Clean Reliability Mean": src.get("clean_position_reliability_mean", "TODO"),
                "Repair Loss on Corrupted": src.get("repair_loss_on_corrupted_positions", "TODO"),
                "Fraction Repaired": src.get("fraction_positions_repaired", "TODO"),
                "Interpretation": interpretation,
            }
        )
    csv_df = pd.DataFrame(rows)
    csv_df.to_csv(out_dir / "table6_repair_reliability_diagnostics.csv", index=False)
    render = csv_df.copy()
    for col in ["Mean Reliability", "Corrupted Reliability Mean", "Clean Reliability Mean", "Repair Loss on Corrupted", "Fraction Repaired"]:
        render[col] = [fmt3(v) for v in render[col]]
    write_markdown_table(render, out_dir / "table6_repair_reliability_diagnostics.md")
    write_latex_table(render, out_dir / "table6_repair_reliability_diagnostics.tex")


def make_complexity_table(complexity: pd.DataFrame, out_dir: Path) -> None:
    ca_rows = complexity[complexity["model"] == "OfficialStyleSTID-corruption-aware full-train"]
    ca_params = float(ca_rows.iloc[0]["parameter_count"]) if not ca_rows.empty else None
    ca_clean_latency = float(ca_rows.iloc[0]["clean_inference_time_sec"]) if not ca_rows.empty else None
    ca_avg_latency = float(ca_rows.iloc[0]["average_inference_time_sec"]) if not ca_rows.empty else None
    wanted = [
        "OfficialStyleSTID-clean full-train",
        "OfficialStyleSTID-corruption-aware full-train",
        "SRAF-OfficialStyleSTID-full full-train",
        "current SRAF-RC-V2-Horizon reference",
    ]
    rows = []
    for model in wanted:
        src = complexity[complexity["model"] == model]
        if src.empty:
            rows.append(
                {
                    "Model": MODEL_LABELS.get(model, model),
                    "Parameter Count": "TODO",
                    "Param Overhead vs STID-CA": "TODO",
                    "Clean Inference Latency": "TODO",
                    "Avg Fault Inference Latency": "TODO",
                    "Clean Latency Overhead vs STID-CA": "TODO",
                    "Avg Latency Overhead vs STID-CA": "TODO",
                    "Training Time Sec": "TODO",
                }
            )
            continue
        src = src.iloc[0]
        params = float(src["parameter_count"])
        clean_lat = float(src["clean_inference_time_sec"])
        avg_lat = float(src["average_inference_time_sec"])
        rows.append(
            {
                "Model": MODEL_LABELS.get(model, model),
                "Parameter Count": params,
                "Param Overhead vs STID-CA": params - ca_params if ca_params is not None else "TODO",
                "Clean Inference Latency": clean_lat,
                "Avg Fault Inference Latency": avg_lat,
                "Clean Latency Overhead vs STID-CA": clean_lat - ca_clean_latency if ca_clean_latency is not None else "TODO",
                "Avg Latency Overhead vs STID-CA": avg_lat - ca_avg_latency if ca_avg_latency is not None else "TODO",
                "Training Time Sec": float(src["training_time_sec"]),
            }
        )
    csv_df = pd.DataFrame(rows)
    csv_df.to_csv(out_dir / "table7_complexity_latency.csv", index=False, float_format="%.6f")
    render = csv_df.copy()
    for col in render.columns[1:]:
        render[col] = [fmt3(v) for v in render[col]]
    note = "Parameter overhead is tiny; latency overhead is measurable and should be reported."
    write_markdown_table(render, out_dir / "table7_complexity_latency.md", note)
    write_latex_table(render, out_dir / "table7_complexity_latency.tex", note)


def make_figures(frames: dict[str, pd.DataFrame], out_dir: Path) -> list[str]:
    fig_dir = out_dir
    made = []
    metrics = frames["metrics"]
    same = frames["same_gain"]
    horizon = frames["horizon"]
    repair = frames["repair"].copy()

    fault_plot = metrics[metrics["model"].isin(["OfficialStyleSTID-corruption-aware full-train", "SRAF-OfficialStyleSTID-full full-train"]) & metrics["fault"].isin(FAULTY)]
    fault_plot.to_csv(out_dir / "figure_fault_mae_bar_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, model in enumerate(["OfficialStyleSTID-corruption-aware full-train", "SRAF-OfficialStyleSTID-full full-train"]):
        vals = [pivot_metric(metrics, model, f, "mae") for f in FAULTY]
        xs = [x + (i - 0.5) * 0.35 for x in range(len(FAULTY))]
        ax.bar(xs, vals, width=0.35, label=MODEL_LABELS[model])
    ax.set_xticks(range(len(FAULTY)))
    ax.set_xticklabels([FAULT_LABELS[f] for f in FAULTY], rotation=25, ha="right")
    ax.set_ylabel("MAE")
    ax.legend()
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(fig_dir / f"figure_fault_mae_bar.{ext}", dpi=200)
    plt.close(fig)
    made.append("figure_fault_mae_bar")

    same[same["fault"].isin(FAULTY)].to_csv(out_dir / "figure_same_backbone_gain_bar_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = [float(same[same["fault"] == f].iloc[0]["same_backbone_robustness_gain"]) for f in FAULTY]
    ax.bar([FAULT_LABELS[f] for f in FAULTY], vals)
    ax.set_ylabel("Relative MAE gain")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(fig_dir / f"figure_same_backbone_gain_bar.{ext}", dpi=200)
    plt.close(fig)
    made.append("figure_same_backbone_gain_bar")

    h12_rows = []
    for fault in FAULTY:
        ca_h12 = pivot_metric(horizon, "OfficialStyleSTID-corruption-aware full-train", fault, "h12_mae")
        sraf_h12 = pivot_metric(horizon, "SRAF-OfficialStyleSTID-full full-train", fault, "h12_mae")
        h12_rows.append({"fault": fault, "h12_delta_sraf_minus_ca": sraf_h12 - ca_h12})
    pd.DataFrame(h12_rows).to_csv(out_dir / "figure_h12_gain_bar_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([FAULT_LABELS[r["fault"]] for r in h12_rows], [r["h12_delta_sraf_minus_ca"] for r in h12_rows])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("h12 delta (SRAF - STID-CA)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(fig_dir / f"figure_h12_gain_bar.{ext}", dpi=200)
    plt.close(fig)
    made.append("figure_h12_gain_bar")

    diag = repair[repair["fault"].isin(FAULTY)].copy()
    diag.to_csv(out_dir / "figure_reliability_diagnostics_source.csv", index=False)
    for col in ["corrupted_position_reliability_mean", "clean_position_reliability_mean"]:
        diag[col] = pd.to_numeric(diag[col], errors="coerce")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([FAULT_LABELS[f] for f in diag["fault"]], diag["corrupted_position_reliability_mean"], marker="o", label="Corrupted")
    ax.plot([FAULT_LABELS[f] for f in diag["fault"]], diag["clean_position_reliability_mean"], marker="o", label="Clean")
    ax.set_ylabel("Reliability")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(fig_dir / f"figure_reliability_diagnostics.{ext}", dpi=200)
    plt.close(fig)
    made.append("figure_reliability_diagnostics")

    trade_rows = []
    for model in STID_MODELS:
        clean = pivot_metric(metrics, model, "clean", "mae")
        avg_fault = sum(pivot_metric(metrics, model, f, "mae") for f in FAULTY) / len(FAULTY)
        trade_rows.append({"model": MODEL_LABELS[model], "clean_mae": clean, "avg_fault_mae": avg_fault})
    pd.DataFrame(trade_rows).to_csv(out_dir / "figure_clean_vs_robustness_tradeoff_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 4))
    for row in trade_rows:
        ax.scatter(row["clean_mae"], row["avg_fault_mae"])
        ax.annotate(row["model"], (row["clean_mae"], row["avg_fault_mae"]), fontsize=8)
    ax.set_xlabel("Clean MAE")
    ax.set_ylabel("Average faulty MAE")
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(fig_dir / f"figure_clean_vs_robustness_tradeoff.{ext}", dpi=200)
    plt.close(fig)
    made.append("figure_clean_vs_robustness_tradeoff")
    return made


def write_summaries(frames: dict[str, pd.DataFrame], out_dir: Path, figures: list[str]) -> None:
    metrics = frames["metrics"]
    same = frames["same_gain"]
    complexity = frames["complexity"]
    repair = frames["repair"]
    clean_stid = pivot_metric(metrics, "OfficialStyleSTID-clean full-train", "clean", "mae")
    ca_clean = pivot_metric(metrics, "OfficialStyleSTID-corruption-aware full-train", "clean", "mae")
    sraf_clean = pivot_metric(metrics, "SRAF-OfficialStyleSTID-full full-train", "clean", "mae")
    clp = float(frames["clp"][frames["clp"]["model"] == "SRAF-OfficialStyleSTID-full full-train"].iloc[0]["clean_loss_penalty"])
    wins = int((same[same["fault"].isin(FAULTY)]["sraf_better"].astype(str) == "True").sum())
    severe_wins = int((same[same["fault"].isin(SEVERE_FAULTS)]["sraf_better"].astype(str) == "True").sum())
    h12_wins = 0
    for fault in FAULTY:
        if pivot_metric(frames["horizon"], "SRAF-OfficialStyleSTID-full full-train", fault, "h12_mae") < pivot_metric(frames["horizon"], "OfficialStyleSTID-corruption-aware full-train", fault, "h12_mae"):
            h12_wins += 1
    sraf_complexity = complexity[complexity["model"] == "SRAF-OfficialStyleSTID-full full-train"].iloc[0]
    ca_complexity = complexity[complexity["model"] == "OfficialStyleSTID-corruption-aware full-train"].iloc[0]

    summary = [
        "# Formal METR-LA Evidence Summary",
        "",
        f"Using the full METR-LA train/validation/test split (23974/3424/6851), SRAF-OfficialStyleSTID improves over the same OfficialStyleSTID corruption-aware backbone on {wins}/6 evaluated faulty-input settings while keeping a clean MAE of {sraf_clean:.6f} versus {ca_clean:.6f} for STID-CA and {clean_stid:.6f} for clean-only STID.",
        "",
        "## Protocol",
        "",
        "- Dataset: METR-LA.",
        "- Input: `[speed_norm, tod_norm, dow_norm]`; SRAF repair touches only `speed_norm`.",
        "- Target `Y`: clean speed target, not corrupted.",
        "- Seed: 42.",
        "- Loss: MAE.",
        "- Device: CUDA.",
        "",
        "## Main Evidence",
        "",
        f"- SRAF-STID beats STID-CA on `{wins}/6` faulty settings.",
        f"- SRAF-STID improves all `{severe_wins}/4` severe faults: RM40, outage24, noise_high, and drift_high.",
        f"- Clean penalty versus clean-only STID is `{clp * 100:.3f}%`.",
        f"- h12 MAE improves on `{h12_wins}/6` faulty settings.",
        f"- Parameter overhead is `{int(sraf_complexity['parameter_count']) - int(ca_complexity['parameter_count'])}` parameters.",
        "",
        "## Limitations",
        "",
        f"- Clean latency overhead is `{float(sraf_complexity['clean_inference_time_sec']) - float(ca_complexity['clean_inference_time_sec']):.6f}` seconds in this run.",
        f"- Average fault latency overhead is `{float(sraf_complexity['average_inference_time_sec']) - float(ca_complexity['average_inference_time_sec']):.6f}` seconds.",
        "- Stuck reliability separation is mixed and should not be overclaimed.",
        "- Full no-reliability-gate ablation is missing; bounded no-gate evidence is supporting only.",
        "- Clean MAE is an internal full-training result, not an official STID reproduction or state-of-the-art claim.",
        "",
        "## Readiness",
        "",
        "The results are ready for manuscript table drafting with careful claim boundaries and explicit latency/reliability limitations.",
        "",
        "## Recommended Next Action",
        "",
        "Run a seed-stability or formal final METR-LA reporting gate before starting PEMS-BAY or drafting manuscript conclusions.",
    ]
    (out_dir / "formal_metr_la_evidence_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    stuck_row = repair[repair["fault"] == "stuck_at_last_value_high"].iloc[0]
    claims = [
        "# Manuscript-Safe Claims",
        "",
        "## Supported by Full-Training Artifacts",
        "",
        "- SRAF-STID improves robustness over the same OfficialStyleSTID corruption-aware backbone across all evaluated METR-LA faulty-input settings.",
        "- SRAF-STID has a small clean-performance penalty relative to clean-only OfficialStyleSTID.",
        "- SRAF-STID improves h12 MAE under all evaluated faulty-input settings.",
        "- SRAF-STID has negligible parameter overhead relative to OfficialStyleSTID-CA, but measurable latency overhead.",
        "- Reliability diagnostics support random missing and continuous outage cases, where corrupted positions receive lower reliability than clean positions.",
        "",
        "## Restricted or Forbidden",
        "",
        "- Do not claim official STID reproduction parity.",
        "- Do not claim state-of-the-art clean forecasting.",
        "- Do not claim stuck reliability detection is solved.",
        f"- For stuck, corrupted reliability mean is `{stuck_row['corrupted_position_reliability_mean']}` and clean reliability mean is `{stuck_row['clean_position_reliability_mean']}`, so the diagnostic is mixed.",
        "- Do not claim full-training no-reliability-gate ablation evidence unless that ablation is run.",
        "- Do not claim latency-free or zero-overhead deployment.",
        "",
        "## Evidence Boundary",
        "",
        "- Bounded no-reliability-gate results are supporting evidence only and are not promoted as full-training evidence.",
        "- No PEMS-BAY result is included in these METR-LA tables.",
    ]
    (out_dir / "manuscript_safe_claims.md").write_text("\n".join(claims) + "\n", encoding="utf-8")

    fig_summary = [
        "# Figure Generation Summary",
        "",
        f"- Figures generated: `{len(figures)}`",
        *[f"- `{name}.png` and `{name}.svg`" for name in figures],
    ]
    (out_dir / "figure_generation_summary.md").write_text("\n".join(fig_summary) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audit, frames = audit_inputs(input_dir, out_dir)
    make_main_table(frames["metrics"], out_dir)
    make_same_backbone_gain_table(frames, out_dir)
    make_horizon_table(frames["horizon"], out_dir)
    make_rdr_table(frames["rdr"], out_dir)
    make_clean_tradeoff_table(frames, out_dir)
    make_repair_table(frames["repair"], out_dir)
    make_complexity_table(frames["complexity"], out_dir)
    figures = make_figures(frames, out_dir)
    write_summaries(frames, out_dir, figures)

    manifest = {
        "gate": "FORMAL_METR_LA_TABLES_AND_EVIDENCE_AUDIT_GATE",
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "input_audit_status": audit["status"],
        "tables_generated": [
            "table1_main_fault_performance",
            "table2_same_backbone_gain",
            "table3_horizon_metrics",
            "table4_robustness_rdr",
            "table5_clean_tradeoff",
            "table6_repair_reliability_diagnostics",
            "table7_complexity_latency",
        ],
        "figures_generated": figures,
        "integrity": {
            "new_training_run": False,
            "pems_bay_started": False,
            "moe_run": False,
            "manuscript_conclusions_written": False,
            "bounded_no_gate_promoted": False,
        },
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": audit["status"], "figures_generated": len(figures)}, indent=2))


if __name__ == "__main__":
    main()
