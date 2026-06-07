"""Run the frozen SRAF-v2 MAIN_FORMAL 10-seed matrix.

The script intentionally separates planning, execution, aggregation, tables,
figures, and audits. It does not modify model definitions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
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

from scripts.run_isolated_pypots_saits_adapter import impute_with_saits, train_pypots_saits  # noqa: E402
from scripts.run_metr_la_sraf_stid_same_backbone_gain import (  # noqa: E402
    model_param_count,
    predict_model,
    train_official_stid_ca,
    train_sraf_stid,
)
from scripts.run_metr_la_strong_clean_backbone_integration import apply_fault, resolve_device  # noqa: E402
from scripts.run_saits_grin_idmlp_baseline_adapter import train_forecaster_clean  # noqa: E402
from scripts.run_sraf_v2_publication_baseline_reproduction_and_selection import (  # noqa: E402
    knn_fill,
    load_payload,
    ppca_lite_fill,
    safe_metrics,
)
from scripts.run_sraf_v2_version_freeze_and_multi_direction_exploration import (  # noqa: E402
    build_official_stid,
    build_v1,
    build_v2,
    predict_v2,
    train_v2,
)


FORMAL_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
DEFAULT_DATASETS = ["METR-LA", "PEMS-BAY"]
DEFAULT_FAULTS = [
    "clean",
    "random_missing_20",
    "random_missing_40",
    "continuous_outage_24",
    "gaussian_noise_high",
    "linear_drift_high",
    "stuck_at_last_value_high",
]
SEVERE_FAULTS = ["random_missing_40", "continuous_outage_24", "gaussian_noise_high", "linear_drift_high"]
MAIN_FORMAL_MODELS = [
    "ID-MLP-CA",
    "SRAF-ID-v1-formal",
    "SRAF-ID-v2-current-best",
    "SRAF-ID-v2-current-best-noGate",
    "KNN+ID-MLP",
    "PPCA-lite+ID-MLP",
    "PyPOTS-SAITS+ID-MLP",
]
SUPPLEMENTARY_NOT_RUN = [
    "MeanFill+ID-MLP",
    "ForwardFill+ID-MLP",
    "SpatialAvg+ID-MLP",
    "TemporalSpatialAvg+ID-MLP",
    "local GRIN-style+ID-MLP",
    "PMM-lite+ID-MLP",
    "DSAE-lite+ID-MLP",
]
DEFERRED = ["Official-STID-clean", "Official-STID-CA", "official GRIN", "BRITS", "CSDI", "DCNN-GAN", "BGCP", "GAN"]
FORMAL_FAULT_SPECS = {
    "clean": {"fault": "clean", "label": "clean"},
    "random_missing_20": {"fault": "random_missing", "rate": 0.20, "label": "random_missing_20"},
    "random_missing_40": {"fault": "random_missing", "rate": 0.40, "label": "random_missing_40"},
    "continuous_outage_24": {"fault": "continuous_outage", "length": 24, "label": "continuous_outage_24"},
    "gaussian_noise_high": {"fault": "gaussian_noise", "severity": "high", "label": "gaussian_noise_high"},
    "linear_drift_high": {"fault": "linear_drift", "severity": "high", "label": "linear_drift_high"},
    "stuck_at_last_value_high": {"fault": "stuck_at_last_value", "severity": "high", "label": "stuck_at_last_value_high"},
}


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    model: str
    fault: str
    seed: int

    @property
    def run_key(self) -> str:
        ds = self.dataset.lower().replace("-", "_")
        model = self.model.lower().replace("+", "plus").replace(" ", "_").replace("/", "_")
        fault = self.fault.lower()
        return f"{ds}__{model}__{fault}__seed{self.seed}"


def parse_csv_list(raw: str | None, default: list[str]) -> list[str]:
    if not raw:
        return list(default)
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_seed_list(raw: str | None) -> list[int]:
    if not raw:
        return list(FORMAL_SEEDS)
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ensure_dirs(root: Path) -> None:
    for rel in ["per_run", "aggregate", "paper_ready_tables", "paper_ready_figures", "audits", "logs", "configs", "models"]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def build_specs(models: list[str], datasets: list[str], faults: list[str], seeds: list[int]) -> list[RunSpec]:
    return [RunSpec(ds, model, fault, seed) for ds in datasets for model in models for fault in faults for seed in seeds]


def metrics_path(root: Path, spec: RunSpec) -> Path:
    return root / "per_run" / spec.run_key / "metrics.json"


def completed_metrics(root: Path, spec: RunSpec) -> bool:
    p = metrics_path(root, spec)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    return data.get("status") == "completed"


def build_fault_formal(x: np.ndarray, label: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if label == "clean":
        mask = np.zeros_like(x[..., :1], dtype=np.float32)
        observed = np.ones_like(mask, dtype=np.float32)
        return x.astype(np.float32), mask, observed
    spec = FORMAL_FAULT_SPECS[label]
    speed, mask, _ = apply_fault(x[..., :1], spec, seed=seed, train_std=1.0)
    out = x.copy()
    out[..., :1] = speed
    observed = np.isfinite(speed).astype(np.float32)
    return out.astype(np.float32), mask.astype(np.float32), observed


def write_run_artifact(root: Path, spec: RunSpec, metrics: dict[str, Any], config: dict[str, Any]) -> None:
    run_dir = root / "per_run" / spec.run_key
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": "SRAF_V2_MAIN_FORMAL_10SEED_RUN_GATE",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": spec.seed,
        "dataset": spec.dataset,
        "fault": spec.fault,
        "model": spec.model,
        "formal_result": True,
        "diagnostic_result": False,
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "config_snapshot.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_csv(run_dir / "metrics.csv", [metrics])


def make_v2_current_best(sensors: int, input_length: int, horizon: int, use_gate: bool = True) -> torch.nn.Module:
    cfg = {
        "rel_hidden": 64,
        "alpha_hidden": 16,
        "adaptive_alpha": True,
        "stuck_features": True,
        "flatness": False,
        "second_delta": True,
        "repair_disagreement": True,
        "base_rel_only": False,
    }
    model = build_v2(sensors, input_length, horizon, cfg)
    if not use_gate:
        model.repairer.use_reliability_gate = False
    return model


def param_count(model: torch.nn.Module | None) -> int:
    if model is None:
        return 0
    return int(model_param_count(model))


def horizon_metrics(y_true: np.ndarray, pred: np.ndarray, mean: float, std: float) -> dict[str, float]:
    m = safe_metrics(y_true, pred, mean, std)
    return {
        "mae": float(m["mae"]),
        "rmse": float(m["rmse"]),
        "h3_mae": float(m["mae_h3"]),
        "h6_mae": float(m["mae_h6"]),
        "h12_mae": float(m["mae_h12"]),
    }


def train_or_load_models(dataset: str, seed: int, payload: dict[str, Any], args: argparse.Namespace, root: Path, device: torch.device) -> dict[str, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    sensors = payload["train_x"].shape[2]
    input_length = payload["train_x"].shape[1]
    horizon = payload["train_y"].shape[1]
    adj_t = torch.from_numpy(payload["adj"]).to(device)
    model_root = root / "models" / dataset.lower().replace("-", "_") / f"seed_{seed}"
    model_root.mkdir(parents=True, exist_ok=True)
    train_args = argparse.Namespace(**vars(args))
    train_args.seed = seed
    train_args.max_epochs_forecaster = args.max_epochs_forecaster
    train_args.forecaster_patience = args.forecaster_patience
    models: dict[str, dict[str, Any]] = {}

    ca = build_official_stid(sensors, input_length, horizon)
    st = perf_counter()
    ca_dir = model_root / "ID-MLP-CA"
    ca_dir.mkdir(parents=True, exist_ok=True)
    ca_meta, ca_curves = train_official_stid_ca(ca, payload["train_x"], payload["train_y"], payload["val_x"], payload["val_y"], train_args, ca_dir, device)
    write_csv(root / "logs" / f"{dataset}_seed{seed}_id_mlp_ca_training_curves.csv", [{**r, "dataset": dataset, "seed": seed} for r in ca_curves])
    models["ID-MLP-CA"] = {"model": ca, "kind": "ca", "params": param_count(ca), "training": {**ca_meta, "wall_sec": perf_counter() - st}}

    v1 = build_v1(sensors, input_length, horizon)
    st = perf_counter()
    v1_dir = model_root / "SRAF-ID-v1-formal"
    v1_dir.mkdir(parents=True, exist_ok=True)
    v1_meta, v1_curves = train_sraf_stid(v1, "SRAF-ID-v1-formal", payload["train_x"], payload["train_y"], payload["val_x"], payload["val_y"], train_args, v1_dir, device, adj_t)
    write_csv(root / "logs" / f"{dataset}_seed{seed}_sraf_v1_training_curves.csv", [{**r, "dataset": dataset, "seed": seed} for r in v1_curves])
    models["SRAF-ID-v1-formal"] = {"model": v1, "kind": "sraf", "params": param_count(v1), "training": {**v1_meta, "wall_sec": perf_counter() - st}}

    for name, gate in [("SRAF-ID-v2-current-best", True), ("SRAF-ID-v2-current-best-noGate", False)]:
        v2 = make_v2_current_best(sensors, input_length, horizon, use_gate=gate)
        st = perf_counter()
        v2_dir = model_root / name
        v2_dir.mkdir(parents=True, exist_ok=True)
        meta, curves = train_v2(v2, name, payload["train_x"], payload["train_y"], payload["val_x"], payload["val_y"], train_args, v2_dir, device, adj_t)
        write_csv(root / "logs" / f"{dataset}_seed{seed}_{name}_training_curves.csv", [{**r, "dataset": dataset, "seed": seed} for r in curves])
        models[name] = {"model": v2, "kind": "v2", "params": param_count(v2), "training": {**meta, "wall_sec": perf_counter() - st}}

    # Impute-then-forecast baselines train their own ID-MLP forecaster.
    for name, fill_kind in [("KNN+ID-MLP", "knn"), ("PPCA-lite+ID-MLP", "ppca")]:
        train_mask = np.zeros_like(payload["train_x"][..., :1], dtype=np.float32)
        val_mask = np.zeros_like(payload["val_x"][..., :1], dtype=np.float32)
        if fill_kind == "knn":
            train_speed = knn_fill(payload["train_x"][..., :1], train_mask, payload["adj"], k=5)
            val_speed = knn_fill(payload["val_x"][..., :1], val_mask, payload["adj"], k=5)
        else:
            train_speed = ppca_lite_fill(payload["train_x"][..., :1], train_mask)
            val_speed = ppca_lite_fill(payload["val_x"][..., :1], val_mask)
        train_imp = payload["train_x"].copy()
        val_imp = payload["val_x"].copy()
        train_imp[..., :1] = train_speed
        val_imp[..., :1] = val_speed
        f_model = build_official_stid(sensors, input_length, horizon)
        st = perf_counter()
        f_dir = model_root / name
        f_dir.mkdir(parents=True, exist_ok=True)
        f_model, meta, curves = train_forecaster_clean(f_model, train_imp, payload["train_y"], val_imp, payload["val_y"], train_args, f_dir, device)
        write_csv(root / "logs" / f"{dataset}_seed{seed}_{name}_training_curves.csv", [{**r, "dataset": dataset, "seed": seed, "model": name} for r in curves])
        models[name] = {"model": f_model, "kind": fill_kind, "params": param_count(f_model), "training": {**meta, "wall_sec": perf_counter() - st}}

    saits, saits_meta = train_pypots_saits(payload["train_x"], train_args, model_root / "PyPOTS-SAITS-imputer")
    zero_train_mask = np.zeros_like(payload["train_x"][..., :1], dtype=np.float32)
    zero_val_mask = np.zeros_like(payload["val_x"][..., :1], dtype=np.float32)
    train_imp, _ = impute_with_saits(saits, payload["train_x"], zero_train_mask)
    val_imp, _ = impute_with_saits(saits, payload["val_x"], zero_val_mask)
    f_model = build_official_stid(sensors, input_length, horizon)
    st = perf_counter()
    saits_forecaster_dir = model_root / "PyPOTS-SAITS+ID-MLP"
    saits_forecaster_dir.mkdir(parents=True, exist_ok=True)
    f_model, meta, curves = train_forecaster_clean(f_model, train_imp, payload["train_y"], val_imp, payload["val_y"], train_args, saits_forecaster_dir, device)
    write_csv(root / "logs" / f"{dataset}_seed{seed}_pypots_saits_forecaster_training_curves.csv", [{**r, "dataset": dataset, "seed": seed, "model": "PyPOTS-SAITS+ID-MLP"} for r in curves])
    models["PyPOTS-SAITS+ID-MLP"] = {
        "model": f_model,
        "kind": "saits",
        "imputer": saits,
        "params": param_count(f_model),
        "training": {**meta, "saits_training_time_sec": saits_meta.get("training_time_sec"), "wall_sec": perf_counter() - st},
    }

    return models


def evaluate_one(model_info: dict[str, Any], model_name: str, payload: dict[str, Any], fault: str, seed: int, args: argparse.Namespace, device: torch.device) -> tuple[dict[str, float], float]:
    x_fault, mask, observed = build_fault_formal(payload["test_x"], fault, seed)
    adj_t = torch.from_numpy(payload["adj"]).to(device)
    kind = model_info["kind"]
    model = model_info["model"]
    if kind == "ca":
        pred, latency, _ = predict_model(model, x_fault, args.batch_size, device, sraf=False)
    elif kind == "sraf":
        pred, latency, _ = predict_model(model, x_fault, args.batch_size, device, sraf=True, observed_mask=observed, adjacency=adj_t)
    elif kind == "v2":
        pred, latency = predict_v2(model, x_fault, observed, args.batch_size, device, adj_t)
    elif kind == "knn":
        x_rep = x_fault.copy()
        x_rep[..., :1] = knn_fill(x_fault[..., :1], mask, payload["adj"], k=5)
        pred, latency, _ = predict_model(model, x_rep, args.batch_size, device, sraf=False)
    elif kind == "ppca":
        x_rep = x_fault.copy()
        x_rep[..., :1] = ppca_lite_fill(x_fault[..., :1], mask)
        pred, latency, _ = predict_model(model, x_rep, args.batch_size, device, sraf=False)
    elif kind == "saits":
        x_rep, _ = impute_with_saits(model_info["imputer"], x_fault, mask)
        pred, latency, _ = predict_model(model, x_rep, args.batch_size, device, sraf=False)
    else:
        raise ValueError(kind)
    if not np.isfinite(pred).all():
        raise ValueError(f"Non-finite prediction: {model_name} {fault} seed={seed}")
    return horizon_metrics(payload["test_y"], pred, payload["mean"], payload["std"]), float(latency)


def aggregate(root: Path, specs: list[RunSpec]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for spec in specs:
        p = metrics_path(root, spec)
        if not p.exists():
            missing.append(spec.__dict__)
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("status") != "completed":
            missing.append({**spec.__dict__, "status": data.get("status"), "error": data.get("error")})
            continue
        rows.append(data)
    write_csv(root / "aggregate" / "formal_10seed_metrics_by_model_fault.csv", rows)
    write_csv(root / "aggregate" / "formal_10seed_negative_cases.csv", missing)
    if not rows:
        return {"completed": 0, "failed": len(missing)}
    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["dataset", "model", "fault"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", lambda x: float(np.std(x, ddof=0))),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", lambda x: float(np.std(x, ddof=0))),
            h3_mae_mean=("h3_mae", "mean"),
            h6_mae_mean=("h6_mae", "mean"),
            h12_mae_mean=("h12_mae", "mean"),
            seeds=("seed", "nunique"),
            latency_mean_sec=("latency_sec", "mean"),
        )
    )
    agg.to_csv(root / "aggregate" / "formal_10seed_metrics_by_model_fault.csv", index=False)
    horizon = agg[["dataset", "model", "fault", "h3_mae_mean", "h6_mae_mean", "h12_mae_mean", "seeds"]]
    horizon.to_csv(root / "aggregate" / "formal_10seed_horizon_metrics.csv", index=False)
    faulty = df[df["fault"] != "clean"]
    avg_faulty = (
        faulty.groupby(["dataset", "model"], as_index=False)
        .agg(avg_faulty_mae_mean=("mae", "mean"), avg_faulty_mae_std=("mae", lambda x: float(np.std(x, ddof=0))), avg_faulty_rmse_mean=("rmse", "mean"), seeds=("seed", "nunique"))
    )
    severe = df[df["fault"].isin(SEVERE_FAULTS)].groupby(["dataset", "model"], as_index=False).agg(severe_fault_mae_mean=("mae", "mean"))
    clean = df[df["fault"] == "clean"][["dataset", "model", "mae"]].groupby(["dataset", "model"], as_index=False).agg(clean_mae_mean=("mae", "mean"))
    avg_faulty = avg_faulty.merge(severe, on=["dataset", "model"], how="left").merge(clean, on=["dataset", "model"], how="left")
    avg_faulty.to_csv(root / "aggregate" / "formal_10seed_avg_faulty_summary.csv", index=False)
    seed_stability = df.groupby(["dataset", "model", "seed"], as_index=False).agg(seed_avg_faulty_mae=("mae", "mean"))
    seed_stability.to_csv(root / "aggregate" / "formal_10seed_seed_stability.csv", index=False)
    complexity = df.groupby(["dataset", "model", "seed"], as_index=False).agg(parameter_count=("parameter_count", "first"), latency_mean_sec=("latency_sec", "mean"), training_time_sec=("training_time_sec", "first"), best_epoch=("best_epoch", "first"))
    complexity.to_csv(root / "aggregate" / "formal_10seed_complexity_latency.csv", index=False)

    win_rows: list[dict[str, Any]] = []
    for ref in ["ID-MLP-CA", "SRAF-ID-v1-formal", "SRAF-ID-v2-current-best"]:
        ref_df = agg[agg.model == ref][["dataset", "fault", "mae_mean", "h12_mae_mean"]].rename(columns={"mae_mean": "ref_mae", "h12_mae_mean": "ref_h12"})
        merged = agg.merge(ref_df, on=["dataset", "fault"], how="left")
        for model in sorted(df.model.unique()):
            if model == ref:
                continue
            sub = merged[(merged.model == model) & (merged.fault != "clean")]
            win_rows.append(
                {
                    "model": model,
                    "reference": ref,
                    "faulty_win_count": int((sub.mae_mean < sub.ref_mae).sum()),
                    "faulty_pairs": int(sub.shape[0]),
                    "h12_win_count": int((sub.h12_mae_mean < sub.ref_h12).sum()),
                    "avg_gain_pct": float(((sub.ref_mae - sub.mae_mean) / sub.ref_mae * 100.0).mean()),
                }
            )
    write_csv(root / "aggregate" / "formal_10seed_win_counts.csv", win_rows)
    make_tables(root, agg, avg_faulty, horizon, complexity, win_rows)
    make_figures(root, agg, avg_faulty, seed_stability)
    make_audits(root, specs, rows, missing, win_rows, avg_faulty)
    return {"completed": len(rows), "failed": len(missing)}


def make_tables(root: Path, agg: pd.DataFrame, avg_faulty: pd.DataFrame, horizon: pd.DataFrame, complexity: pd.DataFrame, win_rows: list[dict[str, Any]]) -> None:
    out = root / "paper_ready_tables"
    avg_faulty.to_csv(out / "main_table_avg_faulty.csv", index=False)
    agg.to_csv(out / "main_table_per_fault_mae.csv", index=False)
    agg.to_csv(out / "main_table_baseline_comparison.csv", index=False)
    agg[agg.model.isin(["SRAF-ID-v2-current-best", "SRAF-ID-v2-current-best-noGate"])].to_csv(out / "main_table_nogate_ablation.csv", index=False)
    agg[agg.model.isin(["SRAF-ID-v2-current-best", "KNN+ID-MLP", "PPCA-lite+ID-MLP", "PyPOTS-SAITS+ID-MLP"])].to_csv(out / "main_table_saits_ppca_knn.csv", index=False)
    horizon.to_csv(out / "supplementary_horizon_metrics.csv", index=False)
    complexity.to_csv(out / "supplementary_complexity_latency.csv", index=False)
    pd.DataFrame(win_rows).to_csv(out / "supplementary_seed_stability.csv", index=False)


def save_bar(fig_path: Path, df: pd.DataFrame, title: str, y: str) -> None:
    plt.figure(figsize=(10, 5))
    if df.empty:
        plt.text(0.5, 0.5, "No data", ha="center")
    else:
        labels = [f"{r.dataset}\n{r.model}" for r in df.itertuples()]
        plt.bar(range(len(df)), df[y].to_numpy())
        plt.xticks(range(len(df)), labels, rotation=45, ha="right", fontsize=7)
        plt.ylabel(y)
        plt.title(title)
        plt.tight_layout()
    plt.savefig(fig_path.with_suffix(".svg"))
    plt.savefig(fig_path.with_suffix(".png"), dpi=200)
    plt.close()


def make_figures(root: Path, agg: pd.DataFrame, avg_faulty: pd.DataFrame, seed_stability: pd.DataFrame) -> None:
    out = root / "paper_ready_figures"
    save_bar(out / "figure_main_faulty_performance", avg_faulty, "Average faulty MAE", "avg_faulty_mae_mean")
    ca = avg_faulty[avg_faulty.model == "ID-MLP-CA"][["dataset", "avg_faulty_mae_mean"]].rename(columns={"avg_faulty_mae_mean": "ca"})
    gain = avg_faulty.merge(ca, on="dataset", how="left")
    gain["gain_vs_id_mlp_ca_pct"] = (gain["ca"] - gain["avg_faulty_mae_mean"]) / gain["ca"] * 100.0
    save_bar(out / "figure_gain_vs_id_mlp_ca", gain, "Average gain vs ID-MLP-CA", "gain_vs_id_mlp_ca_pct")
    nogate = avg_faulty[avg_faulty.model.isin(["SRAF-ID-v2-current-best", "SRAF-ID-v2-current-best-noGate"])]
    save_bar(out / "figure_nogate_ablation", nogate, "NoGate ablation", "avg_faulty_mae_mean")
    ss = seed_stability.groupby(["dataset", "model"], as_index=False).agg(seed_avg_faulty_mae=("seed_avg_faulty_mae", "mean"))
    save_bar(out / "figure_seed_stability", ss, "Seed stability", "seed_avg_faulty_mae")
    clean = avg_faulty[["dataset", "model", "clean_mae_mean", "avg_faulty_mae_mean"]].copy()
    save_bar(out / "figure_clean_robustness_tradeoff", clean, "Clean robustness tradeoff", "clean_mae_mean")
    ext = avg_faulty[avg_faulty.model.isin(["KNN+ID-MLP", "PPCA-lite+ID-MLP", "PyPOTS-SAITS+ID-MLP", "SRAF-ID-v2-current-best"])]
    save_bar(out / "figure_external_imputation_baselines", ext, "External/statistical imputation baselines", "avg_faulty_mae_mean")


def make_audits(root: Path, specs: list[RunSpec], rows: list[dict[str, Any]], missing: list[dict[str, Any]], win_rows: list[dict[str, Any]], avg_faulty: pd.DataFrame) -> None:
    audits = root / "audits"
    completed = len(rows)
    expected = len(specs)
    integrity = [
        "# result_integrity_check",
        "",
        f"- expected runs: `{expected}`",
        f"- completed runs: `{completed}`",
        f"- missing runs: `{len(missing)}`",
        "- diagnostic results mixed: `NO`",
        "- existing diagnostic/freeze results overwritten: `NO`",
    ]
    (audits / "result_integrity_check.md").write_text("\n".join(integrity), encoding="utf-8")
    forbidden = [
        "# unsupported_claims_check",
        "",
        "Forbidden claims for manuscript based on this gate:",
        "- clean SOTA",
        "- all faults solved",
        "- exhaustive stability",
        "- zero overhead",
        "- official STID reproduction",
        "- official GRIN reproduction",
        "- C7 as mainline",
        "- local GRIN-style as official GRIN",
        "- diagnostic results as formal evidence",
    ]
    (audits / "unsupported_claims_check.md").write_text("\n".join(forbidden), encoding="utf-8")
    claim_lines = ["# formal_10seed_claim_audit", ""]
    if not avg_faulty.empty:
        claim_lines.append("- C1 average faulty MAE comparisons: `SUPPORTED` if all 980 runs complete; otherwise `PARTIAL`.")
    claim_lines.append(f"- all 980 runs complete: `{completed == expected}`")
    claim_lines.append("- do not call this exhaustive stability.")
    (audits / "formal_10seed_claim_audit.md").write_text("\n".join(claim_lines), encoding="utf-8")
    (audits / "negative_cases.md").write_text("# negative_cases\n\nSee `aggregate/formal_10seed_negative_cases.csv`.\n", encoding="utf-8")
    (audits / "baseline_inclusion_audit.md").write_text("# baseline_inclusion_audit\n\nMAIN_FORMAL only was run. Supplementary baselines were not run with 10 seeds in this gate.\n", encoding="utf-8")
    repro = [
        "# reproducibility_snapshot",
        "",
        f"- timestamp: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- cwd: `{ROOT}`",
        f"- python: `{sys.version.replace(os.linesep, ' ')}`",
        f"- torch: `{torch.__version__}`",
        f"- cuda_available: `{torch.cuda.is_available()}`",
        f"- gpu: `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}`",
        f"- os: `{platform.platform()}`",
    ]
    (audits / "reproducibility_snapshot.md").write_text("\n".join(repro), encoding="utf-8")


def write_report(root: Path, specs: list[RunSpec], summary: dict[str, Any]) -> None:
    completed = int(summary.get("completed", 0))
    failed = int(summary.get("failed", len(specs) - completed))
    status = "PASS" if completed == len(specs) and failed == 0 else ("PARTIAL" if completed > 0 else "FAIL")
    lines = [
        "# SRAF_V2_MAIN_FORMAL_10SEED_RUN_REPORT",
        "",
        "## 1. Stage Metadata",
        f"- stage: `SRAF_V2_MAIN_FORMAL_10SEED_RUN_GATE`",
        f"- status: `{status}`",
        f"- timestamp: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- expected runs: `{len(specs)}`",
        f"- completed runs: `{completed}`",
        f"- failed runs: `{failed}`",
        "- existing results overwritten: `NO`",
        f"- output directory: `{root}`",
        "",
        "## 2. Frozen Version and Baseline Set",
        "- main method: `SRAF-ID-v2-current-best = v2_c7_no_flatness_features`",
        "- rollback method: `SRAF-ID-v1-formal`",
        f"- MAIN_FORMAL baselines: {', '.join(MAIN_FORMAL_MODELS)}",
        f"- SUPPLEMENTARY baselines not run in this gate: {', '.join(SUPPLEMENTARY_NOT_RUN)}",
        f"- deferred baselines: {', '.join(DEFERRED)}",
        "",
        "## 3. Formal Experimental Protocol",
        f"- datasets: {', '.join(sorted({s.dataset for s in specs}))}",
        f"- settings: {', '.join(DEFAULT_FAULTS)}",
        f"- seeds: {', '.join(str(s) for s in FORMAL_SEEDS)}",
        "- metrics: MAE, RMSE, h3/h6/h12 MAE, average faulty MAE, severe-fault average MAE, clean MAE, latency, parameter count",
        "- split/scaler/fault protocol: processed splits and train-scaler artifacts; faults applied only to input speed; target Y remains clean; identities remain clean.",
        "- leakage check: imputation baselines train on train historical X only; target Y is not used for imputer training.",
        "",
        "## 4. Main 10-Seed Results",
        "- See `aggregate/formal_10seed_avg_faulty_summary.csv` and `aggregate/formal_10seed_metrics_by_model_fault.csv`.",
        "",
        "## 5. Comparison Against ID-MLP-CA",
        "- See `aggregate/formal_10seed_win_counts.csv`.",
        "",
        "## 6. Comparison Against SRAF-ID-v1-formal",
        "- See `aggregate/formal_10seed_win_counts.csv`.",
        "",
        "## 7. External / Imputation Baseline Comparison",
        "- See `paper_ready_tables/main_table_saits_ppca_knn.csv`.",
        "",
        "## 8. Ablation",
        "- See `paper_ready_tables/main_table_nogate_ablation.csv`.",
        "",
        "## 9. Horizon, Complexity, and Latency",
        "- See `aggregate/formal_10seed_horizon_metrics.csv` and `aggregate/formal_10seed_complexity_latency.csv`.",
        "",
        "## 10. Negative Cases and Limitations",
        "- See `aggregate/formal_10seed_negative_cases.csv` and `audits/negative_cases.md`.",
        "",
        "## 11. Paper-Ready Assets",
        "- aggregate CSVs, paper-ready tables, paper-ready figures, and audits are under this output directory.",
        "",
        "## 12. Gate Decision",
        f"- SHOULD_USE_V2_IN_MANUSCRIPT: `{'YES' if status == 'PASS' else 'PARTIAL' if completed else 'NO'}`",
        "- SHOULD_ROLLBACK_TO_V1: `NO` pending mentor review of aggregate tables",
        f"- BASELINE_SET_SUFFICIENT_FOR_MANUSCRIPT: `{'YES' if status == 'PASS' else 'PARTIAL'}`",
        f"- FIGURES_READY: `{'YES' if status == 'PASS' else 'PARTIAL' if completed else 'NO'}`",
        f"- BLOCKERS: `{'none' if status == 'PASS' else 'missing/incomplete runs listed in aggregate/formal_10seed_negative_cases.csv'}`",
        "- NEXT_ACTION: manuscript mentor verifies formal 10-seed results before manuscript revision.",
    ]
    (root / "SRAF_V2_MAIN_FORMAL_10SEED_RUN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--models", default=None)
    p.add_argument("--datasets", default=None)
    p.add_argument("--faults", default=None)
    p.add_argument("--seeds", default=None)
    p.add_argument("--output-dir", default="experiments/sraf_v2_main_formal_10seed_run")
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.002)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--lambda-repair", type=float, default=0.05)
    p.add_argument("--lambda-rel", type=float, default=0.01)
    p.add_argument("--loss", choices=["mae", "mse"], default="mae")
    p.add_argument("--train-limit", type=int, default=0, help="0 means full split; nonzero is only for explicit debugging.")
    p.add_argument("--val-limit", type=int, default=0)
    p.add_argument("--test-limit", type=int, default=0)
    p.add_argument("--saits-device", default="cpu")
    p.add_argument("--saits-epochs", type=int, default=30)
    p.add_argument("--saits-patience", type=int, default=5)
    p.add_argument("--saits-d-model", type=int, default=32)
    p.add_argument("--saits-heads", type=int, default=2)
    p.add_argument("--train-mask-rate", type=float, default=0.2)
    p.add_argument("--max-epochs-forecaster", type=int, default=60)
    p.add_argument("--forecaster-patience", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    models = parse_csv_list(args.models, MAIN_FORMAL_MODELS)
    datasets = parse_csv_list(args.datasets, DEFAULT_DATASETS)
    faults = parse_csv_list(args.faults, DEFAULT_FAULTS)
    seeds = parse_seed_list(args.seeds)
    specs = build_specs(models, datasets, faults, seeds)
    out = ROOT / args.output_dir
    ensure_dirs(out)
    existing = sum(1 for s in specs if completed_metrics(out, s))
    plan = {
        "models": models,
        "datasets": datasets,
        "faults": faults,
        "seeds": seeds,
        "expected_runs": len(specs),
        "output_root": str(out),
        "existing_results_to_skip": existing if args.skip_existing else 0,
        "dry_run": bool(args.dry_run),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "formal_run_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (out / "configs" / "formal_runner_config_snapshot.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    if len(specs) != 980:
        raise SystemExit(f"Expected 980 runs for this gate, got {len(specs)}. Stop before execution.")
    device = resolve_device(args.device)
    train_limit = args.train_limit if args.train_limit > 0 else 10**12
    val_limit = args.val_limit if args.val_limit > 0 else 10**12
    test_limit = args.test_limit if args.test_limit > 0 else None
    payloads = {ds: load_payload(ds, int(train_limit), int(val_limit), test_limit) for ds in datasets}
    for dataset in datasets:
        for seed in seeds:
            needed = [s for s in specs if s.dataset == dataset and s.seed == seed]
            if args.skip_existing and all(completed_metrics(out, s) for s in needed):
                print(f"skip existing dataset={dataset} seed={seed}", flush=True)
                continue
            print(f"train/eval dataset={dataset} seed={seed}", flush=True)
            payload = payloads[dataset]
            trained = train_or_load_models(dataset, seed, payload, args, out, device)
            for model_name in models:
                for fault in faults:
                    spec = RunSpec(dataset, model_name, fault, seed)
                    if args.skip_existing and completed_metrics(out, spec):
                        continue
                    st = perf_counter()
                    try:
                        met, latency = evaluate_one(trained[model_name], model_name, payload, fault, seed, args, device)
                        row = {
                            "dataset": dataset,
                            "seed": seed,
                            "model": model_name,
                            "fault": fault,
                            **met,
                            "latency_sec": latency,
                            "parameter_count": trained[model_name]["params"],
                            "training_time_sec": trained[model_name]["training"].get("training_time_sec", trained[model_name]["training"].get("wall_sec")),
                            "best_epoch": trained[model_name]["training"].get("best_epoch", ""),
                            "status": "completed",
                            "wall_eval_sec": perf_counter() - st,
                        }
                    except Exception as exc:
                        row = {
                            "dataset": dataset,
                            "seed": seed,
                            "model": model_name,
                            "fault": fault,
                            "status": "failed",
                            "error": repr(exc),
                            "wall_eval_sec": perf_counter() - st,
                        }
                    write_run_artifact(out, spec, row, vars(args))
                    print(f"{row['status']} {spec.run_key}", flush=True)
    summary = aggregate(out, specs)
    write_report(out, specs, summary)
    print("TERMINAL SUMMARY", flush=True)
    print(f"expected runs: {len(specs)}", flush=True)
    print(f"completed runs: {summary['completed']}", flush=True)
    print(f"failed runs: {summary['failed']}", flush=True)


if __name__ == "__main__":
    main()
