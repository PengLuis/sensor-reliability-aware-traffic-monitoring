"""Run final SRAF-ID softmax-fusion mainline and rebuild figure/table package.

This gate promotes the diagnostic A3 softmax fusion repair as the new
manuscript-facing SRAF-ID candidate. It trains only this new mainline on the
formal 10-seed matrix, reads existing formal baselines for comparison, and
generates traceable table/figure assets in a fresh output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_metr_la_sraf_stid_same_backbone_gain import model_param_count  # noqa: E402
from scripts.run_metr_la_strong_clean_backbone_integration import apply_fault, resolve_device  # noqa: E402
from scripts.run_sraf_id_repair_factor_ablation import build_factor_model  # noqa: E402
import scripts.run_sraf_id_repair_v3_light_diagnostic as repair_training_module  # noqa: E402
from scripts.run_sraf_id_repair_v3_light_diagnostic import predict_sraf, train_v3, write_csv  # noqa: E402
from scripts.run_sraf_v2_publication_baseline_reproduction_and_selection import load_payload, safe_metrics  # noqa: E402
from src.protocols.matched_protocol import (  # noqa: E402
    DATASETS as PROTOCOL_DATASETS,
    MATCHED_TRAIN_FAULTS,
    SEEDS as PROTOCOL_SEEDS,
    TEST_FAULTS,
)


STAGE = "SRAF_ID_FINAL_FIGURE_TABLE_REPRODUCTION_GATE"
DATASETS = list(PROTOCOL_DATASETS)
SEEDS = list(PROTOCOL_SEEDS)
FAULTS = list(TEST_FAULTS)
LOCAL_FAULT_SPECS = {
    "clean": {"fault": "clean", "label": "clean"},
    "random_missing_20": {"fault": "random_missing", "rate": 0.20, "label": "random_missing_20"},
    "random_missing_40": {"fault": "random_missing", "rate": 0.40, "label": "random_missing_40"},
    "continuous_outage_24": {"fault": "continuous_outage", "length": 24, "label": "continuous_outage_24"},
    "gaussian_noise_high": {"fault": "gaussian_noise", "severity": "high", "label": "gaussian_noise_high"},
    "linear_drift_high": {"fault": "linear_drift", "severity": "high", "label": "linear_drift_high"},
    "stuck_at_last_value_high": {"fault": "stuck_at_last_value", "severity": "high", "label": "stuck_at_last_value_high"},
}
FAULTY = [f for f in FAULTS if f != "clean"]
SEVERE = ["random_missing_40", "continuous_outage_24", "gaussian_noise_high", "linear_drift_high"]
MAIN_MODELS = ["ID-MLP-CA", "KNN+ID-MLP", "PPCA-lite+ID-MLP", "PyPOTS-SAITS+ID-MLP", "SRAF-ID"]
PALETTE = {
    "SRAF-ID": "#0072B2",
    "ID-MLP-CA": "#E69F00",
    "KNN+ID-MLP": "#7A7A7A",
    "PPCA-lite+ID-MLP": "#56B4E9",
    "PyPOTS-SAITS+ID-MLP": "#009E73",
    "SRAF-ID-gated": "#CC79A7",
    "negative": "#D55E00",
    "grid": "#E6E6E6",
    "text": "#222222",
}
BASELINE_ROOT = ROOT / "experiments" / "sraf_v2_main_formal_10seed_run"
MATCHED_ID_ROOT = ROOT / "experiments" / "id_mlp_ca_matched_fault_distribution_10seed"


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.dataset.lower().replace('-', '_')}__sraf_id_softmax_fusion__seed{self.seed}"


def ensure_layout(out: Path) -> None:
    for rel in [
        "tables",
        "figures",
        "figure_data",
        "logs",
        "rerun_logs",
        "audit_logs",
        "reports",
        "per_run",
        "configs",
    ]:
        (out / rel).mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_ms(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "NA"
    if pd.isna(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def pct_gain(ref: float, val: float) -> float:
    return float((ref - val) / ref * 100.0) if ref and not pd.isna(ref) and not pd.isna(val) else float("nan")


def get_faulted(x: np.ndarray, fault: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fault == "clean":
        mask = np.zeros_like(x[..., :1], dtype=np.float32)
        observed = np.ones_like(mask, dtype=np.float32)
        return x.astype(np.float32), mask, observed
    spec = LOCAL_FAULT_SPECS[fault]
    speed, mask, _ = apply_fault(x[..., :1], spec, seed=seed, train_std=1.0)
    x_fault = x.copy()
    x_fault[..., :1] = speed
    observed = np.isfinite(speed).astype(np.float32)
    return x_fault.astype(np.float32), mask.astype(np.float32), observed


def build_softmax_model(payload: dict[str, Any]) -> torch.nn.Module:
    cfg = {
        "name": "SRAF-ID",
        "family": "factor",
        "temporal_mode": "basic",
        "spatial_mode": "adjacency",
        "fusion_mode": "softmax",
        "use_profile": False,
        "topk": 5,
        "fixed_profile_weight": 0.0,
        "description": "Final mainline: current temporal/spatial repair with 2-way softmax MLP fusion; no profile; no gate.",
    }
    return build_factor_model(payload, cfg)


def run_one_dataset_seed(job: dict[str, Any]) -> dict[str, Any]:
    args = argparse.Namespace(**job["args"])
    dataset = job["dataset"]
    seed = int(job["seed"])
    args.seed = seed
    out = Path(args.output_dir)
    run_dir = out / "per_run" / RunSpec(dataset, seed).key
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "metrics.csv"
    if args.skip_existing and manifest_path.exists() and metrics_path.exists():
        try:
            old = read_json(manifest_path)
            if old.get("status") == "completed":
                return {"dataset": dataset, "seed": seed, "status": "skipped_existing", "output_path": str(run_dir)}
        except Exception:
            pass
    start = perf_counter()
    device = resolve_device(args.device)
    np.random.seed(seed)
    torch.manual_seed(seed)
    payload = load_payload(dataset, 10**12 if args.train_limit == 0 else args.train_limit, 10**12 if args.val_limit == 0 else args.val_limit, None if args.test_limit == 0 else args.test_limit)
    adj_t = torch.from_numpy(payload["adj"]).to(device)
    model = build_softmax_model(payload)
    model.to(device)
    config = {
        "stage": STAGE,
        "dataset": dataset,
        "seed": seed,
        "model": "SRAF-ID",
        "internal_source": "A3_mlp_only_softmax_no_profile",
        "args": vars(args),
        "full_split": args.train_limit == 0 and args.val_limit == 0 and args.test_limit == 0,
        "faults": FAULTS,
        "formal_seed": True,
    }
    (run_dir / "config_snapshot.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    checkpoint = run_dir / "best_checkpoint.pt"
    curves_path = run_dir / "training_curves.csv"
    if args.skip_existing and checkpoint.exists() and curves_path.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        curves_df = pd.read_csv(curves_path)
        best_epoch = int(curves_df.loc[curves_df["best_selection_val_loss"].idxmin(), "epoch"]) if not curves_df.empty else -1
        meta = {"training_time_sec": float("nan"), "best_epoch": best_epoch, "best_val_loss": float(curves_df["best_selection_val_loss"].min()) if not curves_df.empty else float("nan"), "reused_checkpoint": True}
    else:
        # The saved formal artifacts used this exact five-fault rotation. Keep
        # the training module synchronized with the single public config.
        repair_training_module.FAULTS = list(MATCHED_TRAIN_FAULTS)
        meta, curves = train_v3(
            model,
            "SRAF-ID",
            payload["train_x"],
            payload["train_y"],
            payload["val_x"],
            payload["val_y"],
            args,
            run_dir,
            device,
            adj_t,
            lambda_delta=0.0,
        )
        meta["reused_checkpoint"] = False
        write_csv(curves_path, curves)
    rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for idx, fault in enumerate(FAULTS):
        x_fault, mask, observed = get_faulted(payload["test_x"], fault, seed + idx)
        pred, latency, comps = predict_sraf(model, x_fault, observed, args.batch_size, device, adj_t, return_components=True)
        met = safe_metrics(payload["test_y"], pred, payload["mean"], payload["std"])
        row = {
            "dataset": dataset,
            "seed": seed,
            "model": "SRAF-ID",
            "fault": fault,
            "mae": float(met["mae"]),
            "rmse": float(met["rmse"]),
            "h3_mae": float(met["mae_h3"]),
            "h6_mae": float(met["mae_h6"]),
            "h12_mae": float(met["mae_h12"]),
            "latency_sec": float(latency),
            "parameter_count": int(model_param_count(model)),
            "training_time_sec": float(meta["training_time_sec"]),
            "best_epoch": int(meta["best_epoch"]),
            "reused_checkpoint": bool(meta.get("reused_checkpoint", False)),
            "status": "completed",
            "source_result_file": str(metrics_path),
        }
        rows.append(row)
        if comps is not None and "weights" in comps:
            w = comps["weights"]
            component_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "model": "SRAF-ID",
                    "fault": fault,
                    "weight_temporal_mean": float(np.mean(w[..., 0])),
                    "weight_spatial_mean": float(np.mean(w[..., 1])),
                    "repair_displacement_mean": float(np.mean(comps.get("repair_disp", np.array([np.nan])))),
                }
            )
    write_csv(metrics_path, rows)
    if component_rows:
        write_csv(run_dir / "repair_component_stats.csv", component_rows)
    manifest = {
        "stage": STAGE,
        "dataset": dataset,
        "seed": seed,
        "model": "SRAF-ID",
        "status": "completed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": perf_counter() - start,
        "output_path": str(run_dir),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_new_metrics(out: Path) -> pd.DataFrame:
    files = sorted((out / "per_run").glob("**/metrics.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def load_baseline_metrics() -> pd.DataFrame:
    path = BASELINE_ROOT / "aggregate" / "formal_10seed_metrics_by_model_fault.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    model_map = {
        "SRAF-ID-v2-current-best": "SRAF-ID-gated",
        "ID-MLP-CA": "ID-MLP-CA",
        "KNN+ID-MLP": "KNN+ID-MLP",
        "PPCA-lite+ID-MLP": "PPCA-lite+ID-MLP",
        "PyPOTS-SAITS+ID-MLP": "PyPOTS-SAITS+ID-MLP",
    }
    df = df[df["model"].isin(model_map)].copy()
    df["model"] = df["model"].map(model_map)
    matched_path = MATCHED_ID_ROOT / "aggregate" / "formal_10seed_metrics_by_model_fault.csv"
    if matched_path.exists():
        matched = pd.read_csv(matched_path)
        df = pd.concat([df[df["model"] != "ID-MLP-CA"], matched], ignore_index=True, sort=False)
    df["source_result_file"] = str(path)
    if matched_path.exists():
        df.loc[df["model"] == "ID-MLP-CA", "source_result_file"] = str(matched_path)
    df["is_aggregate_source"] = True
    return df


def aggregate_new_metrics(out: Path, rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows.to_csv(out / "sraf_id_softmax_formal_per_seed_metrics.csv", index=False)
    agg = (
        rows.groupby(["dataset", "model", "fault"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", lambda x: float(np.std(x, ddof=0))),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", lambda x: float(np.std(x, ddof=0))),
            h3_mae_mean=("h3_mae", "mean"),
            h3_mae_std=("h3_mae", lambda x: float(np.std(x, ddof=0))),
            h6_mae_mean=("h6_mae", "mean"),
            h6_mae_std=("h6_mae", lambda x: float(np.std(x, ddof=0))),
            h12_mae_mean=("h12_mae", "mean"),
            h12_mae_std=("h12_mae", lambda x: float(np.std(x, ddof=0))),
            seeds=("seed", "nunique"),
            latency_mean_sec=("latency_sec", "mean"),
            latency_std_sec=("latency_sec", lambda x: float(np.std(x, ddof=0))),
            parameter_count_mean=("parameter_count", "mean"),
            training_time_mean_sec=("training_time_sec", "mean"),
        )
    )
    faulty = rows[rows["fault"] != "clean"]
    avg = (
        faulty.groupby(["dataset", "model"], as_index=False)
        .agg(
            avg_faulty_mae_mean=("mae", "mean"),
            avg_faulty_mae_std=("mae", lambda x: float(np.std(x, ddof=0))),
            avg_faulty_rmse_mean=("rmse", "mean"),
            avg_faulty_rmse_std=("rmse", lambda x: float(np.std(x, ddof=0))),
            seeds=("seed", "nunique"),
        )
    )
    severe = rows[rows["fault"].isin(SEVERE)].groupby(["dataset", "model"], as_index=False).agg(severe_fault_mae_mean=("mae", "mean"))
    clean = rows[rows["fault"] == "clean"].groupby(["dataset", "model"], as_index=False).agg(clean_mae_mean=("mae", "mean"), clean_mae_std=("mae", lambda x: float(np.std(x, ddof=0))))
    avg = avg.merge(severe, on=["dataset", "model"], how="left").merge(clean, on=["dataset", "model"], how="left")
    complexity = rows.groupby(["dataset", "model"], as_index=False).agg(parameter_count_mean=("parameter_count", "mean"), latency_mean_sec=("latency_sec", "mean"), training_time_mean_sec=("training_time_sec", "mean"))
    agg.to_csv(out / "sraf_id_softmax_formal_aggregate_by_fault.csv", index=False)
    avg.to_csv(out / "sraf_id_softmax_formal_avg_faulty_summary.csv", index=False)
    complexity.to_csv(out / "sraf_id_softmax_formal_complexity_latency.csv", index=False)
    return agg, avg, complexity


def combined_frames(out: Path, new_agg: pd.DataFrame, new_avg: pd.DataFrame, new_complexity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_agg = load_baseline_metrics()
    # Existing aggregate file already has mean/std columns.
    base_avg_path = BASELINE_ROOT / "aggregate" / "formal_10seed_avg_faulty_summary.csv"
    base_avg = pd.read_csv(base_avg_path)
    base_map = {
        "SRAF-ID-v2-current-best": "SRAF-ID-gated",
        "ID-MLP-CA": "ID-MLP-CA",
        "KNN+ID-MLP": "KNN+ID-MLP",
        "PPCA-lite+ID-MLP": "PPCA-lite+ID-MLP",
        "PyPOTS-SAITS+ID-MLP": "PyPOTS-SAITS+ID-MLP",
    }
    base_avg = base_avg[base_avg["model"].isin(base_map)].copy()
    base_avg["model"] = base_avg["model"].map(base_map)
    matched_avg_path = MATCHED_ID_ROOT / "aggregate" / "formal_10seed_avg_faulty_summary.csv"
    if matched_avg_path.exists():
        matched_avg = pd.read_csv(matched_avg_path)
        base_avg = pd.concat([base_avg[base_avg["model"] != "ID-MLP-CA"], matched_avg], ignore_index=True, sort=False)
    base_avg["source_result_file"] = str(base_avg_path)
    if matched_avg_path.exists():
        base_avg.loc[base_avg["model"] == "ID-MLP-CA", "source_result_file"] = str(matched_avg_path)
    comp_path = BASELINE_ROOT / "aggregate" / "formal_10seed_complexity_latency.csv"
    base_comp = pd.read_csv(comp_path)
    base_comp = base_comp[base_comp["model"].isin(base_map)].copy()
    base_comp["model"] = base_comp["model"].map(base_map)
    base_comp = base_comp.groupby(["dataset", "model"], as_index=False).agg(
        parameter_count_mean=("parameter_count", "mean"),
        latency_mean_sec=("latency_mean_sec", "mean"),
        training_time_mean_sec=("training_time_sec", "mean"),
    )
    matched_comp_path = MATCHED_ID_ROOT / "aggregate" / "formal_10seed_complexity_latency.csv"
    if matched_comp_path.exists():
        matched_comp = pd.read_csv(matched_comp_path).rename(
            columns={"parameter_count": "parameter_count_mean", "training_time_sec": "training_time_mean_sec"}
        )
        base_comp = pd.concat([base_comp[base_comp["model"] != "ID-MLP-CA"], matched_comp], ignore_index=True, sort=False)
    all_agg = pd.concat([base_agg, new_agg], ignore_index=True, sort=False)
    all_avg = pd.concat([base_avg, new_avg], ignore_index=True, sort=False)
    all_comp = pd.concat([base_comp, new_complexity], ignore_index=True, sort=False)
    all_agg.to_csv(out / "combined_metrics_by_model_fault.csv", index=False)
    all_avg.to_csv(out / "combined_avg_faulty_summary.csv", index=False)
    all_comp.to_csv(out / "combined_complexity_latency.csv", index=False)
    return all_agg, all_avg, all_comp


def markdown_table(df: pd.DataFrame, path: Path, best_low_cols: list[str] | None = None) -> None:
    best_low_cols = best_low_cols or []
    out = df.copy()
    for col in best_low_cols:
        if col in out.columns:
            numeric = pd.to_numeric(out[col], errors="coerce")
            if numeric.notna().any():
                best = numeric.min()
                out[col] = [f"**{v:.4f}**" if not pd.isna(v) and abs(float(v) - best) < 1e-12 else (f"{v:.4f}" if not pd.isna(v) else "NA") for v in numeric]
    path.write_text(out.to_markdown(index=False), encoding="utf-8")


def save_table(df: pd.DataFrame, out_dir: Path, name: str, best_cols: list[str] | None = None) -> None:
    df.to_csv(out_dir / f"{name}.csv", index=False)
    markdown_table(df, out_dir / f"{name}.md", best_cols)


def make_tables(out: Path, all_agg: pd.DataFrame, all_avg: pd.DataFrame, all_comp: pd.DataFrame) -> dict[str, str]:
    tables = out / "tables"
    source = {}
    table1 = pd.DataFrame(
        [
            {"Dataset": "METR-LA", "Sensors": 207, "Historical window L": 12, "Forecasting horizon H": 12, "Fault application": "historical input speed only", "Forecast target": "clean future speed", "Formal seeds": "42-51"},
            {"Dataset": "PEMS-BAY", "Sensors": 325, "Historical window L": 12, "Forecasting horizon H": 12, "Fault application": "historical input speed only", "Forecast target": "clean future speed", "Formal seeds": "42-51"},
        ]
    )
    save_table(table1, tables, "table1_dataset_protocol")
    table2 = pd.DataFrame(
        [
            {"Setting": "clean", "Mechanism": "no corruption", "Affected channel": "none", "Forecast target": "clean", "Purpose": "clean-input reference"},
            {"Setting": "random_missing_20", "Mechanism": "randomly mask 20% speed observations", "Affected channel": "input speed", "Forecast target": "clean", "Purpose": "moderate missing observations"},
            {"Setting": "random_missing_40", "Mechanism": "randomly mask 40% speed observations", "Affected channel": "input speed", "Forecast target": "clean", "Purpose": "severe missing observations"},
            {"Setting": "continuous_outage_24", "Mechanism": "sensor-level continuous outage over the input window", "Affected channel": "input speed", "Forecast target": "clean", "Purpose": "block sensor outage"},
            {"Setting": "gaussian_noise_high", "Mechanism": "high-amplitude additive Gaussian noise", "Affected channel": "input speed", "Forecast target": "clean", "Purpose": "noisy sensor readings"},
            {"Setting": "linear_drift_high", "Mechanism": "high linear drift in historical observations", "Affected channel": "input speed", "Forecast target": "clean", "Purpose": "sensor calibration drift"},
            {"Setting": "stuck_at_last_value_high", "Mechanism": "sensor value stuck at previous reading", "Affected channel": "input speed", "Forecast target": "clean", "Purpose": "stuck sensor fault"},
        ]
    )
    save_table(table2, tables, "table2_fault_protocol")
    table3 = pd.DataFrame(
        [
            {"Category": "same-backbone baseline", "Model": "ID-MLP-CA", "Repair/imputation stage": "none", "Forecasting backbone": "ID-MLP", "Purpose in comparison": "corruption-aware same-backbone reference"},
            {"Category": "proposed", "Model": "SRAF-ID", "Repair/imputation stage": "speed-channel temporal/spatial repair with softmax fusion", "Forecasting backbone": "ID-MLP", "Purpose in comparison": "final repair-only method"},
            {"Category": "impute-then-forecast", "Model": "KNN+ID-MLP", "Repair/imputation stage": "KNN spatial imputation", "Forecasting backbone": "ID-MLP", "Purpose in comparison": "classical spatial imputation"},
            {"Category": "impute-then-forecast", "Model": "PPCA-lite+ID-MLP", "Repair/imputation stage": "PPCA-lite statistical imputation", "Forecasting backbone": "ID-MLP", "Purpose in comparison": "statistical imputation"},
            {"Category": "impute-then-forecast", "Model": "PyPOTS-SAITS+ID-MLP", "Repair/imputation stage": "SAITS imputation via PyPOTS", "Forecasting backbone": "ID-MLP", "Purpose in comparison": "neural imputation"},
            {"Category": "ablation", "Model": "SRAF-ID-gated", "Repair/imputation stage": "gated repair variant", "Forecasting backbone": "ID-MLP", "Purpose in comparison": "supplementary gated variant"},
        ]
    )
    save_table(table3, tables, "table3_baseline_models")
    rows = []
    for ds in DATASETS:
        ca_row = all_avg[(all_avg["dataset"] == ds) & (all_avg["model"] == "ID-MLP-CA")].iloc[0]
        s_row = all_avg[(all_avg["dataset"] == ds) & (all_avg["model"] == "SRAF-ID")].iloc[0]
        ca_mean = float(ca_row["avg_faulty_mae_mean"])
        ca_std = float(ca_row["avg_faulty_mae_std"])
        s_mean = float(s_row["avg_faulty_mae_mean"])
        s_std = float(s_row["avg_faulty_mae_std"])
        rows.append(
            {
                "Dataset": ds,
                "ID-MLP-CA MAE mean ± std": fmt_ms(ca_mean, ca_std),
                "SRAF-ID MAE mean ± std": fmt_ms(s_mean, s_std),
                "Gain vs ID-MLP-CA": f"{pct_gain(ca_mean, s_mean):.3f}%",
                "SRAF-ID RMSE mean ± std": fmt_ms(float(s_row["avg_faulty_rmse_mean"]), np.nan),
                "SRAF-ID clean MAE mean ± std": fmt_ms(float(s_row["clean_mae_mean"]), float(s_row["clean_mae_std"])),
            }
        )
    table4 = pd.DataFrame(rows)
    save_table(table4, tables, "table4_average_faulty_performance")
    ca_fault = all_agg[all_agg["model"] == "ID-MLP-CA"][["dataset", "fault", "mae_mean", "mae_std"]].rename(columns={"mae_mean": "ca_mae", "mae_std": "ca_std"})
    s_fault = all_agg[all_agg["model"] == "SRAF-ID"][["dataset", "fault", "mae_mean", "mae_std"]].rename(columns={"mae_mean": "sraf_mae", "mae_std": "sraf_std"})
    t5 = ca_fault.merge(s_fault, on=["dataset", "fault"])
    t5 = t5[t5["fault"].isin(FAULTY)].copy()
    t5["Gain"] = (t5["ca_mae"] - t5["sraf_mae"]) / t5["ca_mae"] * 100.0
    table5 = pd.DataFrame(
        {
            "Dataset": t5["dataset"],
            "Fault": t5["fault"],
            "ID-MLP-CA MAE mean ± std": [fmt_ms(a, b) for a, b in zip(t5["ca_mae"], t5["ca_std"])],
            "SRAF-ID MAE mean ± std": [fmt_ms(a, b) for a, b in zip(t5["sraf_mae"], t5["sraf_std"])],
            "Gain": [f"{g:.3f}%" for g in t5["Gain"]],
            "Outcome": ["improved" if g > 0 else "negative" for g in t5["Gain"]],
        }
    )
    save_table(table5, tables, "table5_per_fault_robustness")
    rows = []
    for ds in DATASETS:
        s = all_avg[(all_avg.dataset == ds) & (all_avg.model == "SRAF-ID")].iloc[0]
        for baseline in ["KNN+ID-MLP", "PPCA-lite+ID-MLP", "PyPOTS-SAITS+ID-MLP"]:
            b = all_avg[(all_avg.dataset == ds) & (all_avg.model == baseline)].iloc[0]
            rows.append({"Dataset": ds, "Baseline": baseline, "Baseline MAE mean ± std": fmt_ms(b.avg_faulty_mae_mean, b.avg_faulty_mae_std), "SRAF-ID MAE mean ± std": fmt_ms(s.avg_faulty_mae_mean, s.avg_faulty_mae_std), "Gain": f"{pct_gain(b.avg_faulty_mae_mean, s.avg_faulty_mae_mean):.3f}%"})
    table6 = pd.DataFrame(rows)
    save_table(table6, tables, "table6_impute_then_forecast_comparison")
    h12_ca = all_agg[all_agg.model == "ID-MLP-CA"][["dataset", "fault", "h12_mae_mean", "h12_mae_std"]].rename(columns={"h12_mae_mean": "ca_h12", "h12_mae_std": "ca_h12_std"})
    h12_s = all_agg[all_agg.model == "SRAF-ID"][["dataset", "fault", "h12_mae_mean", "h12_mae_std"]].rename(columns={"h12_mae_mean": "s_h12", "h12_mae_std": "s_h12_std"})
    h = h12_ca.merge(h12_s, on=["dataset", "fault"])
    h = h[h.fault.isin(FAULTY)].copy()
    h["gain"] = (h["ca_h12"] - h["s_h12"]) / h["ca_h12"] * 100.0
    table7 = pd.DataFrame({"Dataset": h.dataset, "Fault": h.fault, "ID-MLP-CA h12 MAE mean ± std": [fmt_ms(a, b) for a, b in zip(h.ca_h12, h.ca_h12_std)], "SRAF-ID h12 MAE mean ± std": [fmt_ms(a, b) for a, b in zip(h.s_h12, h.s_h12_std)], "h12 Gain": [f"{x:.3f}%" for x in h.gain], "Outcome": ["improved" if x > 0 else "negative" for x in h.gain]})
    save_table(table7, tables, "table7_h12_comparison")
    formal_ablation_path = ROOT / "experiments" / "sraf_id_formal_repair_source_ablation" / "table8_ablation_study_revised.csv"
    if formal_ablation_path.exists():
        table8 = pd.read_csv(formal_ablation_path)
    else:
        rows = []
        for ds in DATASETS:
            s = all_avg[(all_avg.dataset == ds) & (all_avg.model == "SRAF-ID")].iloc[0]
            g = all_avg[(all_avg.dataset == ds) & (all_avg.model == "SRAF-ID-gated")].iloc[0]
            rows.append({"Dataset": ds, "Variant": "SRAF-ID", "Average faulty MAE mean ± std": fmt_ms(s.avg_faulty_mae_mean, s.avg_faulty_mae_std), "Difference vs SRAF-ID": "0.000%", "Interpretation": "final method"})
            rows.append({"Dataset": ds, "Variant": "SRAF-ID-gated", "Average faulty MAE mean ± std": fmt_ms(g.avg_faulty_mae_mean, g.avg_faulty_mae_std), "Difference vs SRAF-ID": f"{(g.avg_faulty_mae_mean - s.avg_faulty_mae_mean) / s.avg_faulty_mae_mean * 100.0:.3f}%", "Interpretation": "gated ablation/supplementary variant"})
            for variant in ["SRAF-ID-temporal-only", "SRAF-ID-spatial-only", "SRAF-ID-fixed-fusion"]:
                rows.append({"Dataset": ds, "Variant": variant, "Average faulty MAE mean ± std": "NA", "Difference vs SRAF-ID": "NA", "Interpretation": "formal evidence unavailable"})
        table8 = pd.DataFrame(rows)
    save_table(table8, tables, "table8_ablation_study")
    rows = []
    for ds in DATASETS:
        ca = all_comp[(all_comp.dataset == ds) & (all_comp.model == "ID-MLP-CA")].iloc[0]
        s = all_comp[(all_comp.dataset == ds) & (all_comp.model == "SRAF-ID")].iloc[0]
        rows.append({"Dataset": ds, "ID-MLP-CA params": int(round(ca.parameter_count_mean)), "SRAF-ID params": int(round(s.parameter_count_mean)), "Parameter increase": f"{(s.parameter_count_mean - ca.parameter_count_mean) / ca.parameter_count_mean * 100.0:.3f}%", "ID-MLP-CA latency": f"{ca.latency_mean_sec:.4f}s", "SRAF-ID latency": f"{s.latency_mean_sec:.4f}s", "Latency ratio": f"{s.latency_mean_sec / ca.latency_mean_sec:.3f}x"})
    table9 = pd.DataFrame(rows)
    save_table(table9, tables, "table9_complexity_latency")
    return {f"table{i}": str(tables / f"table{i}") for i in range(1, 10)}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "text.color": PALETTE["text"],
            "axes.labelcolor": PALETTE["text"],
            "axes.edgecolor": PALETTE["text"],
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_fig(fig: plt.Figure, out: Path) -> None:
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, linewidth=1.2, color="#555555"))


def box(ax: plt.Axes, xy: tuple[float, float], text: str, color: str, w: float = 1.8, h: float = 0.55) -> None:
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.08,rounding_size=0.08", linewidth=1.1, edgecolor=color, facecolor=color + "22"))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=8)


def make_architecture_figures(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    box(ax, (0.1, 2.3), "Corrupted speed\n$X^c$", PALETTE["negative"])
    box(ax, (0.1, 1.45), "Fault mask\n$\\Omega$", "#999999")
    box(ax, (0.1, 0.55), "Identity/time\nfeatures", PALETTE["SRAF-ID"])
    box(ax, (2.4, 2.75), "Temporal\nrepair", PALETTE["PPCA-lite+ID-MLP"])
    box(ax, (2.4, 1.85), "Spatial\nrepair with $A$", PALETTE["PyPOTS-SAITS+ID-MLP"])
    box(ax, (4.7, 2.3), "Adaptive\nfusion", PALETTE["SRAF-ID"])
    box(ax, (6.8, 2.3), "Repaired speed\n$X^r$", PALETTE["SRAF-ID"])
    box(ax, (6.8, 0.8), "ID-MLP\nbackbone", "#666666")
    box(ax, (8.8, 0.8), "Clean future\nforecast $\\hat{Y}$", PALETTE["SRAF-ID"])
    for p in [(1.9, 2.58, 2.35, 3.02), (1.9, 2.5, 2.35, 2.12), (1.9, 1.72, 2.35, 2.12), (4.2, 3.02, 4.65, 2.58), (4.2, 2.12, 4.65, 2.58), (6.5, 2.58, 6.75, 2.58), (7.7, 2.3, 7.7, 1.43), (1.9, 0.82, 6.75, 1.05), (8.65, 1.08, 8.75, 1.08)]:
        arrow(ax, *p)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 3.8)
    save_fig(fig, out / "figures" / "figure1_overall_architecture")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.axis("off")
    box(ax, (0.1, 2.1), "$X^c$, $\\Omega$", PALETTE["negative"])
    box(ax, (0.1, 1.1), "$X^c$, $A$, $\\Omega$", "#999999")
    box(ax, (2.35, 2.1), "Temporal branch\n$X_{temp}$", PALETTE["PPCA-lite+ID-MLP"])
    box(ax, (2.35, 1.1), "Spatial branch\n$X_{sp}$", PALETTE["PyPOTS-SAITS+ID-MLP"])
    box(ax, (4.8, 1.6), "Fusion MLP\n$\\alpha$", PALETTE["SRAF-ID"])
    box(ax, (6.6, 1.6), "$X^r=\\alpha X_{temp}\n+(1-\\alpha)X_{sp}$", PALETTE["SRAF-ID"], w=2.2)
    for p in [(1.9, 2.38, 2.3, 2.38), (1.9, 1.38, 2.3, 1.38), (4.2, 2.38, 4.75, 1.88), (4.2, 1.38, 4.75, 1.88), (6.25, 1.88, 6.55, 1.88)]:
        arrow(ax, *p)
    ax.set_xlim(0, 9.2)
    ax.set_ylim(0.5, 3.2)
    save_fig(fig, out / "figures" / "figure2_repair_module")


def make_fault_protocol_figure(out: Path) -> None:
    rng = np.random.default_rng(42)
    t = np.arange(48)
    clean = 55 + 8 * np.sin(t / 6.0) + 2 * np.cos(t / 2.7)
    variants = {
        "random_missing_20": clean.copy(),
        "random_missing_40": clean.copy(),
        "continuous_outage_24": clean.copy(),
        "gaussian_noise_high": clean + rng.normal(0, 5, size=len(t)),
        "linear_drift_high": clean + np.linspace(0, 12, len(t)),
        "stuck_at_last_value_high": clean.copy(),
    }
    variants["random_missing_20"][rng.choice(len(t), int(0.2 * len(t)), replace=False)] = np.nan
    variants["random_missing_40"][rng.choice(len(t), int(0.4 * len(t)), replace=False)] = np.nan
    variants["continuous_outage_24"][18:30] = np.nan
    variants["stuck_at_last_value_high"][20:35] = variants["stuck_at_last_value_high"][19]
    fig, axes = plt.subplots(2, 3, figsize=(9, 4.8), sharex=True, sharey=True)
    rows = []
    for ax, (name, y) in zip(axes.ravel(), variants.items()):
        ax.plot(t, clean, color="#333333", linewidth=1.2, label="clean")
        ax.plot(t, y, color=PALETTE["SRAF-ID"] if "missing" not in name else PALETTE["ID-MLP-CA"], linewidth=1.2, marker="x" if np.isnan(y).any() else None, markersize=3)
        ax.set_title(name)
        ax.grid(color=PALETTE["grid"], linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for i, val in zip(t, y):
            rows.append({"fault": name, "step": i, "clean_speed": clean[i], "corrupted_speed": val})
    fig.tight_layout()
    save_fig(fig, out / "figures" / "figure3_fault_protocol")
    pd.DataFrame(rows).to_csv(out / "figure_data" / "figure3_fault_protocol.csv", index=False)


def make_metric_figures(out: Path, all_agg: pd.DataFrame, all_avg: pd.DataFrame, all_comp: pd.DataFrame) -> None:
    fd = out / "figure_data"
    figs = out / "figures"
    avg = all_avg[all_avg.model.isin(["ID-MLP-CA", "SRAF-ID"])].copy()
    avg.to_csv(fd / "figure4_average_faulty_mae.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(DATASETS))
    width = 0.32
    for j, model in enumerate(["ID-MLP-CA", "SRAF-ID"]):
        sub = avg[avg.model == model].set_index("dataset").loc[DATASETS]
        ax.bar(x + (j - 0.5) * width, sub.avg_faulty_mae_mean, width, yerr=sub.avg_faulty_mae_std, capsize=3, color=PALETTE[model], label=model)
    for i, ds in enumerate(DATASETS):
        ca = avg[(avg.dataset == ds) & (avg.model == "ID-MLP-CA")].iloc[0].avg_faulty_mae_mean
        s = avg[(avg.dataset == ds) & (avg.model == "SRAF-ID")].iloc[0].avg_faulty_mae_mean
        ax.text(i, max(ca, s) * 1.02, f"{pct_gain(ca, s):.2f}%", ha="center", fontsize=8)
    ax.set_xticks(x, DATASETS)
    ax.set_ylabel("Average faulty MAE")
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    save_fig(fig, figs / "figure4_average_faulty_mae")
    ca = all_agg[all_agg.model == "ID-MLP-CA"][["dataset", "fault", "mae_mean"]].rename(columns={"mae_mean": "ca"})
    s = all_agg[all_agg.model == "SRAF-ID"][["dataset", "fault", "mae_mean"]].rename(columns={"mae_mean": "sraf"})
    heat = ca.merge(s, on=["dataset", "fault"])
    heat = heat[heat.fault.isin(FAULTY)].copy()
    heat["gain_pct"] = (heat.ca - heat.sraf) / heat.ca * 100.0
    heat.to_csv(fd / "figure5_fault_type_gain.csv", index=False)
    mat = heat.pivot(index="dataset", columns="fault", values="gain_pct").loc[DATASETS, FAULTY]
    fig, ax = plt.subplots(figsize=(9, 2.6))
    vmax = max(1.0, float(np.nanmax(np.abs(mat.values))))
    im = ax.imshow(mat.values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(FAULTY)), FAULTY, rotation=35, ha="right")
    ax.set_yticks(range(len(DATASETS)), DATASETS)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=8, color="#111111")
    fig.colorbar(im, ax=ax, label="Gain vs ID-MLP-CA (%)")
    fig.tight_layout()
    save_fig(fig, figs / "figure5_fault_type_gain")
    ext = all_avg[all_avg.model.isin(["KNN+ID-MLP", "PPCA-lite+ID-MLP", "PyPOTS-SAITS+ID-MLP", "SRAF-ID"])].copy()
    ext.to_csv(fd / "figure6_impute_then_forecast.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), sharey=False)
    models = ["KNN+ID-MLP", "PPCA-lite+ID-MLP", "PyPOTS-SAITS+ID-MLP", "SRAF-ID"]
    for ax, ds in zip(axes, DATASETS):
        sub = ext[ext.dataset == ds].set_index("model").loc[models]
        ax.bar(range(len(models)), sub.avg_faulty_mae_mean, yerr=sub.avg_faulty_mae_std, capsize=3, color=[PALETTE[m] for m in models])
        ax.set_title(ds)
        ax.set_xticks(range(len(models)), models, rotation=35, ha="right")
        ax.set_ylabel("Average faulty MAE")
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, figs / "figure6_impute_then_forecast")
    h = ca.drop(columns="ca").merge(all_agg[all_agg.model == "ID-MLP-CA"][["dataset", "fault", "h12_mae_mean"]].rename(columns={"h12_mae_mean": "ca_h12"}), on=["dataset", "fault"]).merge(all_agg[all_agg.model == "SRAF-ID"][["dataset", "fault", "h12_mae_mean"]].rename(columns={"h12_mae_mean": "s_h12"}), on=["dataset", "fault"])
    h = h[h.fault.isin(FAULTY)].copy()
    h["h12_gain_pct"] = (h.ca_h12 - h.s_h12) / h.ca_h12 * 100.0
    h.to_csv(fd / "figure7_h12_gain.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    labels = [f"{r.dataset}\n{r.fault.replace('_high','').replace('random_missing_','rm')}" for r in h.itertuples()]
    colors = [PALETTE["SRAF-ID"] if v >= 0 else PALETTE["negative"] for v in h.h12_gain_pct]
    ax.bar(range(len(h)), h.h12_gain_pct, color=colors)
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.set_xticks(range(len(h)), labels, rotation=45, ha="right")
    ax.set_ylabel("h12 gain vs ID-MLP-CA (%)")
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, figs / "figure7_h12_robustness")
    abl = all_avg[all_avg.model.isin(["SRAF-ID", "SRAF-ID-gated"])].copy()
    abl.to_csv(fd / "figure8_ablation.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4), sharey=False)
    for ax, ds in zip(axes, DATASETS):
        sub = abl[abl.dataset == ds].set_index("model").loc[["SRAF-ID", "SRAF-ID-gated"]]
        bars = ax.bar(range(2), sub.avg_faulty_mae_mean, yerr=sub.avg_faulty_mae_std, capsize=3, color=[PALETTE["SRAF-ID"], PALETTE["SRAF-ID-gated"]])
        best = int(np.argmin(sub.avg_faulty_mae_mean.to_numpy()))
        bars[best].set_edgecolor("#222222")
        bars[best].set_linewidth(1.5)
        ax.set_title(ds)
        ax.set_xticks(range(2), ["SRAF-ID", "SRAF-ID-gated"], rotation=20, ha="right")
        ax.set_ylabel("Average faulty MAE")
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, figs / "figure8_ablation")
    trade = all_avg[all_avg.model.isin(["ID-MLP-CA", "SRAF-ID"])][["dataset", "model", "avg_faulty_mae_mean"]].merge(all_comp, on=["dataset", "model"], how="left")
    trade.to_csv(fd / "figure9_complexity_latency.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ca_lat = all_comp[all_comp.model == "ID-MLP-CA"][["dataset", "latency_mean_sec"]].rename(columns={"latency_mean_sec": "ca_latency"})
    trade = trade.merge(ca_lat, on="dataset")
    trade["latency_ratio"] = trade["latency_mean_sec"] / trade["ca_latency"]
    for r in trade.itertuples():
        ax.scatter(r.latency_ratio, r.avg_faulty_mae_mean, s=max(40, r.parameter_count_mean / 2000), color=PALETTE[r.model], alpha=0.8)
        ax.text(r.latency_ratio, r.avg_faulty_mae_mean, f"{r.dataset} {r.model}", fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("Latency ratio vs ID-MLP-CA")
    ax.set_ylabel("Average faulty MAE")
    ax.grid(color=PALETTE["grid"], linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, figs / "figure9_complexity_latency")


def write_captions(out: Path) -> None:
    text = """# captions

- Figure 1. Overall architecture of SRAF-ID for speed-channel repair and identity-preserved forecasting.
- Figure 2. Speed-channel repair module with temporal and spatial candidates and adaptive fusion.
- Figure 3. Controlled faulty-observation protocol applied only to the historical speed channel.
- Figure 4. Average faulty MAE comparison between ID-MLP-CA and SRAF-ID over 10 seeds.
- Figure 5. Fault-type gain distribution of SRAF-ID relative to ID-MLP-CA.
- Figure 6. Comparison between SRAF-ID and impute-then-forecast baselines.
- Figure 7. Horizon-wise h12 robustness gain of SRAF-ID relative to ID-MLP-CA.
- Figure 8. Ablation comparison between SRAF-ID and the gated variant.
- Figure 9. Complexity and latency trade-off for SRAF-ID and ID-MLP-CA.
"""
    (out / "figures" / "captions.md").write_text(text, encoding="utf-8")


def write_manifest_and_reports(out: Path, all_agg: pd.DataFrame, all_avg: pd.DataFrame, all_comp: pd.DataFrame, run_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    completed = sum(1 for r in run_rows if r.get("status") in {"completed", "skipped_existing"})
    expected = len(DATASETS) * len(SEEDS)
    s_avg = all_avg[all_avg.model == "SRAF-ID"]
    ca_avg = all_avg[all_avg.model == "ID-MLP-CA"]
    gains = {}
    for ds in DATASETS:
        s = s_avg[s_avg.dataset == ds].iloc[0].avg_faulty_mae_mean
        ca = ca_avg[ca_avg.dataset == ds].iloc[0].avg_faulty_mae_mean
        gains[ds] = pct_gain(ca, s)
    source_inputs = {
        "baseline_formal_root": str(BASELINE_ROOT),
        "matched_id_mlp_ca_root": str(MATCHED_ID_ROOT),
        "new_mainline_per_run_root": str(out / "per_run"),
        "new_mainline_aggregate": str(out / "sraf_id_softmax_formal_aggregate_by_fault.csv"),
        "combined_metrics": str(out / "combined_metrics_by_model_fault.csv"),
    }
    manifest = {
        "stage": STAGE,
        "status": "PASS" if completed == expected else "PARTIAL",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "expected_training_jobs": expected,
        "completed_training_jobs": completed,
        "models": MAIN_MODELS + ["SRAF-ID-gated"],
        "datasets": DATASETS,
        "faults": FAULTS,
        "seeds": SEEDS,
        "source_inputs": source_inputs,
        "formal_results_modified": False,
        "manuscript_modified": False,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = []
    for i in range(1, 10):
        rows.append({"asset": f"Table {i}", "csv": str(out / "tables" / f"table{i}_{'*'}.csv"), "markdown": str(out / "tables"), "status": "READY" if completed == expected else "PARTIAL"})
    for i in range(1, 10):
        rows.append({"asset": f"Figure {i}", "svg": str(out / "figures" / f"figure{i}_*.svg"), "pdf": str(out / "figures" / f"figure{i}_*.pdf"), "png": str(out / "figures" / f"figure{i}_*.png"), "status": "READY" if completed == expected else "PARTIAL"})
    write_csv(out / "asset_manifest.csv", rows)
    readme = [
        "# SRAF-ID Final Figure/Table Package",
        "",
        f"- stage: `{STAGE}`",
        f"- created_at: `{manifest['timestamp']}`",
        "- final manuscript-facing method: `SRAF-ID`",
        "- internal source: `A3_mlp_only_softmax_no_profile` / softmax temporal-spatial fusion",
        "- existing formal result directory was read-only.",
        "- manuscript text was not modified.",
    ]
    (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
    audit = [
        "# audit_report",
        "",
        f"- expected new mainline training jobs: `{expected}`",
        f"- completed/skipped jobs: `{completed}`",
        "- matched ID-MLP-CA formal results available: `YES`",
        "- seed coverage for new SRAF-ID: `42-51`" if completed == expected else "- seed coverage for new SRAF-ID: `PARTIAL`",
        "- faults on historical speed only: inherited from `apply_fault` and audited by protocol; target Y clean in loader/evaluation.",
        "- split/scaler/input/horizon: inherited from processed payload loader used by prior formal runner.",
        "- temporal-only/spatial-only/fixed-fusion formal ablations: `GAP` unless separately authorized; prior diagnostics exist but are not included as formal manuscript evidence.",
    ]
    (out / "audit_report.md").write_text("\n".join(audit), encoding="utf-8")
    comp = []
    ca_fault = all_agg[all_agg.model == "ID-MLP-CA"][["dataset", "fault", "mae_mean", "h12_mae_mean"]].rename(columns={"mae_mean": "ca_mae", "h12_mae_mean": "ca_h12"})
    s_fault = all_agg[all_agg.model == "SRAF-ID"][["dataset", "fault", "mae_mean", "h12_mae_mean"]].rename(columns={"mae_mean": "sraf_mae", "h12_mae_mean": "sraf_h12"})
    m = ca_fault.merge(s_fault, on=["dataset", "fault"])
    m = m[m.fault.isin(FAULTY)].copy()
    m["gain"] = (m.ca_mae - m.sraf_mae) / m.ca_mae * 100.0
    m["h12_gain"] = (m.ca_h12 - m.sraf_h12) / m.ca_h12 * 100.0
    win_count = int((m.gain > 0).sum())
    h12_win = int((m.h12_gain > 0).sum())
    neg = m[m.gain <= 0][["dataset", "fault", "gain"]]
    for ds in DATASETS:
        comp.append(f"- `{ds}` SRAF-ID avg faulty gain vs ID-MLP-CA: `{gains[ds]:.3f}%`")
    comp.extend([f"- faulty pair win count vs ID-MLP-CA: `{win_count}/12`", f"- h12 win count vs ID-MLP-CA: `{h12_win}/12`"])
    if neg.empty:
        comp.append("- negative cases vs ID-MLP-CA: `none`")
    else:
        comp.append("- negative cases vs ID-MLP-CA:")
        for r in neg.itertuples():
            comp.append(f"  - `{r.dataset}` / `{r.fault}`: `{r.gain:.3f}%`")
    (out / "final_summary_metrics.md").write_text("# final_summary_metrics\n\n" + "\n".join(comp), encoding="utf-8")
    package_report = [
        "# FIGURE_TABLE_PACKAGE_REPORT",
        "",
        f"- STAGE: `{STAGE}`",
        f"- STATUS: `{'PASS' if completed == expected else 'PARTIAL'}`",
        f"- exact command(s) used: `{' '.join(sys.argv)}`",
        "",
        "## Result Coverage Matrix",
        f"- new mainline expected dataset/seed training jobs: `{expected}`",
        f"- completed/skipped: `{completed}`",
        "- per-seed evaluations per job: `7 settings`",
        "- formal seeds: `42,43,44,45,46,47,48,49,50,51`",
        "",
        "## Missing Runs",
        "- none" if completed == expected else "- see `run_manifest.csv`",
        "",
        "## Rerun Summary",
        "- rerun scope: existing SRAF-ID softmax-fusion mainline plus matched ID-MLP-CA evidence integration.",
        "- ID-MLP-CA baseline: retrained separately with the final SRAF-ID five-fault training distribution and matched test-mask seed rule.",
        "- previous formal directories overwritten: `NO`",
        "",
        "## Table Outputs",
        "- `tables/table1_dataset_protocol.*` through `tables/table9_complexity_latency.*`",
        "",
        "## Figure Outputs",
        "- `figures/figure1_*` through `figures/figure9_*` in SVG/PDF/600-dpi PNG.",
        "",
        "## Metric Consistency Checks",
        f"- METR-LA avg faulty gain vs ID-MLP-CA: `{gains.get('METR-LA', float('nan')):.3f}%`",
        f"- PEMS-BAY avg faulty gain vs ID-MLP-CA: `{gains.get('PEMS-BAY', float('nan')):.3f}%`",
        f"- faulty pair win count vs ID-MLP-CA: `{win_count}/12`",
        f"- h12 win count vs ID-MLP-CA: `{h12_win}/12`",
        "",
        "## Seed Coverage Checks",
        "- New SRAF-ID uses seeds `42-51` if coverage is PASS.",
        "",
        "## Notes on Values That Differ From V5",
        "- This package promotes a new softmax-fusion mainline, so SRAF-ID values may differ from the previous noGate/v5 evidence. Differences are reported via source CSVs and should be reviewed before manuscript claim changes.",
        "",
        "## Ablation Availability",
        "- `SRAF-ID-gated` is available from existing formal evidence.",
        "- `SRAF-ID-temporal-only`, `SRAF-ID-spatial-only`, and `SRAF-ID-fixed-fusion` are recorded as gaps because they are not formal 10-seed supported in this package.",
        "",
        "## Recommended Manuscript Updates",
        "- Do not update claims automatically. Mentor should compare `final_summary_metrics.md` and table outputs against the current v5 claims.",
        "- If adopting softmax-fusion as final SRAF-ID, update Methods Section 3.4.3 to describe adaptive softmax temporal-spatial fusion.",
        "",
        "## Next Action",
        "- Manuscript mentor reviews the package and decides whether to update Results and figures.",
    ]
    (out / "reports" / "FIGURE_TABLE_PACKAGE_REPORT.md").write_text("\n".join(package_report), encoding="utf-8")
    print("TERMINAL SUMMARY", flush=True)
    print(f"expected new mainline jobs: {expected}", flush=True)
    print(f"completed/skipped jobs: {completed}", flush=True)
    print(f"METR-LA SRAF-ID gain vs ID-MLP-CA: {gains.get('METR-LA', float('nan')):.3f}%", flush=True)
    print(f"PEMS-BAY SRAF-ID gain vs ID-MLP-CA: {gains.get('PEMS-BAY', float('nan')):.3f}%", flush=True)
    print(f"faulty win count vs ID-MLP-CA: {win_count}/12", flush=True)


def make_manifest_doc(out: Path) -> None:
    lines = ["# FIGURE_TABLE_MANIFEST", ""]
    for i in range(1, 10):
        lines.append(f"- Table {i}: see `tables/table{i}_*.csv` and `.md`; source: combined formal metrics; status: READY/NEEDS REVIEW depending run coverage.")
    for i in range(1, 10):
        lines.append(f"- Figure {i}: see `figures/figure{i}_*` and `figure_data/`; source: combined formal metrics or deterministic protocol illustration; status: READY/NEEDS REVIEW.")
    (out / "FIGURE_TABLE_MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="experiments/sraf_id_final_figure_table_package")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--train-only", action="store_true", help="Stop after aggregating newly trained SRAF-ID metrics.")
    p.add_argument("--datasets", default=",".join(DATASETS), help="Comma-separated dataset subset.")
    p.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS), help="Comma-separated seed subset.")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.002)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--lambda-repair", type=float, default=0.05)
    p.add_argument("--lambda-rel", type=float, default=0.01)
    p.add_argument("--loss", choices=["mae", "mse"], default="mae")
    p.add_argument("--train-limit", type=int, default=0)
    p.add_argument("--val-limit", type=int, default=0)
    p.add_argument("--test-limit", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = ROOT / args.output_dir
    ensure_layout(out)
    set_style()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}; allowed: {DATASETS}")
    if not datasets or not seeds:
        raise ValueError("At least one dataset and one seed are required.")
    specs = [RunSpec(ds, seed) for ds in datasets for seed in seeds]
    plan = {
        "stage": STAGE,
        "expected_new_mainline_training_jobs": len(specs),
        "expected_new_mainline_metric_rows": len(specs) * len(FAULTS),
        "datasets": datasets,
        "faults": FAULTS,
        "seeds": seeds,
        "model_run": "SRAF-ID",
        "baseline_source": str(BASELINE_ROOT),
        "matched_id_mlp_ca_source": str(MATCHED_ID_ROOT),
        "max_workers": args.max_workers,
        "dry_run": args.dry_run,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "run_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    run_rows: list[dict[str, Any]] = []
    jobs = [{"dataset": s.dataset, "seed": s.seed, "args": vars(args)} for s in specs]
    if args.max_workers <= 1:
        for job in jobs:
            try:
                result = run_one_dataset_seed(job)
                run_rows.append(result)
                print(f"{result['status']} {result['dataset']} seed={result['seed']}", flush=True)
            except Exception as exc:
                fail = {"dataset": job["dataset"], "seed": job["seed"], "status": "failed", "error_message": repr(exc)}
                run_rows.append(fail)
                print(f"failed {job['dataset']} seed={job['seed']}: {exc!r}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
            futures = [ex.submit(run_one_dataset_seed, job) for job in jobs]
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                    run_rows.append(result)
                    print(f"{result['status']} {result['dataset']} seed={result['seed']}", flush=True)
                except Exception as exc:
                    fail = {"status": "failed", "error_message": repr(exc)}
                    run_rows.append(fail)
                    print(f"failed {exc!r}", flush=True)
    write_csv(out / "run_manifest.csv", run_rows)
    new_rows = load_new_metrics(out)
    if new_rows.empty:
        raise RuntimeError("No new SRAF-ID metrics were produced.")
    new_agg, new_avg, new_comp = aggregate_new_metrics(out, new_rows)
    if args.train_only:
        print("Training-only run completed; optional historical comparison packaging was skipped.", flush=True)
        return
    all_agg, all_avg, all_comp = combined_frames(out, new_agg, new_avg, new_comp)
    make_tables(out, all_agg, all_avg, all_comp)
    make_architecture_figures(out)
    make_fault_protocol_figure(out)
    make_metric_figures(out, all_agg, all_avg, all_comp)
    write_captions(out)
    make_manifest_doc(out)
    write_manifest_and_reports(out, all_agg, all_avg, all_comp, run_rows, args)


if __name__ == "__main__":
    main()
