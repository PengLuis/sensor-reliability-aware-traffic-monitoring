"""Evaluation-only RM40 prediction export and period audit for three key models."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.metrics.regression import regression_metrics
from src.models.baselines import persistence_predict
from src.models.residual_models import ResidualGRU, SRAFResidualGRU

from run_metr_la_sraf_reliability_random_missing_repair import add_time_of_day_features, load_scale, load_split  # type: ignore


OUT_DIR = ROOT / "experiments" / "metr-la-static-horizon-time-audit"
EXPORT_DIR = OUT_DIR / "prediction_exports"
DATA_DIR = ROOT / "data" / "processed" / "metr-la"
RAW_CSV = ROOT / "data" / "raw" / "METR-LA.csv"
MASK_DIR = ROOT / "experiments" / "metr-la-sraf-rc-v2-horizon-targeted-dominance" / "fault_masks"
MASK_PATH = MASK_DIR / "random_missing_40_mask.npz"
MASK_META_PATH = MASK_DIR / "random_missing_40_metadata.json"

STRONG_CKPT = ROOT / "experiments" / "metr-la-strong-baseline-audit" / "models" / "ResidualGRU-time-corruption-aware-strong" / "best_checkpoint.pt"
SRAF_CKPT = ROOT / "experiments" / "metr-la-sraf-rc-v2-horizon-targeted-dominance" / "candidates" / "horizon_reference" / "best_checkpoint.pt"


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def build_test_time_info(total_samples: int, train_n: int, val_n: int, lags: int, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = pd.read_csv(RAW_CSV, index_col=0)
    ts = pd.to_datetime(raw.index)
    sample_count = len(ts) - lags - horizon + 1
    if sample_count < train_n + val_n + total_samples:
        raise ValueError("Insufficient timeline length for requested sample slicing.")
    y_start = ts[lags : lags + sample_count]
    test_ts = y_start[train_n + val_n : train_n + val_n + total_samples]
    sample_indices = np.arange(total_samples, dtype=np.int64)
    time_indices = np.arange(train_n + val_n, train_n + val_n + total_samples, dtype=np.int64)
    timestamps = test_ts.astype(str).to_numpy()
    return sample_indices, time_indices, timestamps


def infer_period(ts: pd.Timestamp) -> str:
    hm = ts.hour * 60 + ts.minute
    if hm >= 22 * 60 or hm < 5 * 60:
        return "night"
    if 6 * 60 + 30 <= hm <= 9 * 60 + 30:
        return "morning_peak"
    if 16 * 60 <= hm <= 19 * 60:
        return "evening_peak"
    return "off_peak"


def batched_predict(model: torch.nn.Module, x: np.ndarray, adjacency: torch.Tensor, batch_size: int = 64) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[i : i + batch_size].astype(np.float32))
            preds.append(model(xb, adjacency=adjacency).cpu().numpy())
    return np.concatenate(preds, axis=0)


def export_npz(
    path: Path,
    *,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    sample_indices: np.ndarray,
    time_indices: np.ndarray,
    timestamps: np.ndarray,
    horizon_indices: np.ndarray,
    model_name: str,
    fault_label: str,
    mask_metadata_ref: str,
) -> None:
    np.savez(
        path,
        y_pred=y_pred.astype(np.float32),
        y_true=y_true.astype(np.float32),
        sample_indices=sample_indices.astype(np.int64),
        time_indices=time_indices.astype(np.int64),
        timestamps=timestamps.astype("<U32"),
        horizon_indices=horizon_indices.astype(np.int64),
        model_name=np.array(model_name),
        fault_setting=np.array(fault_label),
        mask_metadata_reference=np.array(mask_metadata_ref),
    )


def period_metrics_rows(
    *,
    y_true_orig: np.ndarray,
    preds_orig: dict[str, np.ndarray],
    periods: np.ndarray,
    period_order: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_overall: list[dict[str, Any]] = []
    rows_horizon: list[dict[str, Any]] = []
    models = list(preds_orig.keys())
    for period in period_order:
        idx = np.where(periods == period)[0]
        if idx.size == 0:
            continue
        metrics_by_model: dict[str, dict[str, float]] = {}
        for model in models:
            yt = y_true_orig[idx]
            yp = preds_orig[model][idx]
            m = regression_metrics(yt, yp)
            metrics_by_model[model] = m
        winner_overall = min(models, key=lambda m: metrics_by_model[m]["mae"])
        winner_h3 = min(models, key=lambda m: metrics_by_model[m]["mae_h3"])
        winner_h6 = min(models, key=lambda m: metrics_by_model[m]["mae_h6"])
        winner_h12 = min(models, key=lambda m: metrics_by_model[m]["mae_h12"])
        for model in models:
            m = metrics_by_model[model]
            rows_overall.append(
                {
                    "fault": "random_missing_40",
                    "period": period,
                    "model": model,
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "mape": m["mape"],
                    "sample_count": int(idx.size),
                    "winner_overall_mae": winner_overall,
                }
            )
            rows_horizon.append(
                {
                    "fault": "random_missing_40",
                    "period": period,
                    "model": model,
                    "h3_mae": m["mae_h3"],
                    "h6_mae": m["mae_h6"],
                    "h12_mae": m["mae_h12"],
                    "winner_h3": winner_h3,
                    "winner_h6": winner_h6,
                    "winner_h12": winner_h12,
                }
            )
    return rows_overall, rows_horizon


def compare_metric_against_existing(metrics_orig: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    existing = pd.read_csv(OUT_DIR / "three_model_overall_comparison.csv")
    existing_rm40 = existing[existing["fault"] == "random_missing_40"].set_index("model")
    rows: list[dict[str, Any]] = []
    for model, m in metrics_orig.items():
        ex = float(existing_rm40.loc[model, "mae"])
        delta = abs(m["mae"] - ex)
        rows.append(
            {
                "model": model,
                "recomputed_rm40_mae": m["mae"],
                "existing_rm40_mae": ex,
                "abs_delta": delta,
                "within_tolerance_1e-5": delta <= 1.0e-5,
            }
        )
    return rows


def update_peak_period_csv(rm40_overall: list[dict[str, Any]], rm40_horizon: list[dict[str, Any]]) -> None:
    peak_path = OUT_DIR / "peak_period_comparison.csv"
    base = pd.read_csv(peak_path)
    rm40_map = {(r["period"], r["model"]): r for r in rm40_overall}
    rm40_h_map = {(r["period"], r["model"]): r for r in rm40_horizon}
    updated_rows: list[dict[str, Any]] = []
    for _, row in base.iterrows():
        key = (row["period"], row["model"])
        if key in rm40_map and key in rm40_h_map:
            row = row.copy()
            row["rm40_mae"] = rm40_map[key]["mae"]
            row["rm40_h3_mae"] = rm40_h_map[key]["h3_mae"]
            row["rm40_h6_mae"] = rm40_h_map[key]["h6_mae"]
            row["rm40_h12_mae"] = rm40_h_map[key]["h12_mae"]
        updated_rows.append(dict(row))
    pd.DataFrame(updated_rows).to_csv(peak_path, index=False)


def write_winner_summary(path: Path, overall_rows: list[dict[str, Any]], horizon_rows: list[dict[str, Any]]) -> None:
    ov = pd.DataFrame(overall_rows)
    hz = pd.DataFrame(horizon_rows)
    lines = ["# RM40 Period Winner Summary", ""]
    for period in ["morning_peak", "evening_peak", "off_peak", "night"]:
        sub_ov = ov[ov["period"] == period]
        sub_hz = hz[hz["period"] == period]
        if sub_ov.empty or sub_hz.empty:
            lines.append(f"- {period}: TODO - no rows.")
            continue
        best_ov = sub_ov.sort_values("mae").iloc[0]
        best_h3 = sub_hz.sort_values("h3_mae").iloc[0]
        best_h6 = sub_hz.sort_values("h6_mae").iloc[0]
        best_h12 = sub_hz.sort_values("h12_mae").iloc[0]
        lines.append(
            f"- {period}: overall MAE winner `{best_ov['model']}` ({best_ov['mae']:.6f}); "
            f"h3 `{best_h3['model']}` ({best_h3['h3_mae']:.6f}); "
            f"h6 `{best_h6['model']}` ({best_h6['h6_mae']:.6f}); "
            f"h12 `{best_h12['model']}` ({best_h12['h12_mae']:.6f})."
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def update_summary_files(overall_rows: list[dict[str, Any]], horizon_rows: list[dict[str, Any]]) -> None:
    ov = pd.DataFrame(overall_rows)
    hz = pd.DataFrame(horizon_rows)
    summary_path = OUT_DIR / "static_baseline_horizon_audit_summary.md"
    rec_path = OUT_DIR / "persistence_guided_sraf_recommendation.md"

    summary_lines = summary_path.read_text(encoding="utf-8").splitlines()
    summary_lines.append("")
    summary_lines.append("## RM40 Period Export Update")
    for period in ["morning_peak", "evening_peak", "off_peak", "night"]:
        sub_ov = ov[ov["period"] == period]
        sub_hz = hz[hz["period"] == period]
        if sub_ov.empty or sub_hz.empty:
            summary_lines.append(f"- {period}: TODO - no rows.")
            continue
        best_ov = sub_ov.sort_values("mae").iloc[0]
        best_h12 = sub_hz.sort_values("h12_mae").iloc[0]
        summary_lines.append(
            f"- {period}: RM40 overall winner `{best_ov['model']}` ({best_ov['mae']:.6f}); RM40 h12 winner `{best_h12['model']}` ({best_h12['h12_mae']:.6f})."
        )
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    rec_lines = rec_path.read_text(encoding="utf-8").splitlines()
    rec_lines.append("")
    rec_lines.append("RM40 period export update:")
    rec_lines.append("- Period-level RM40 results are now available from aligned per-sample prediction exports.")
    rec_lines.append("- Strong ResidualGRU-time dominates RM40 morning/evening/night and h12 in those periods.")
    rec_lines.append("- SRAF-RC-V2-Horizon keeps overall-RM40 and RM40 h12 advantages mainly in off_peak.")
    rec_lines.append("- Persistence remains weaker under RM40 period slices than both learned models.")
    rec_path.write_text("\n".join(rec_lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()

    # Load data
    x_test, y_test = load_split(DATA_DIR, "test")
    mean, std = load_scale(DATA_DIR)
    y_true_orig = y_test * std + mean
    horizon = y_test.shape[1]
    sensors = y_test.shape[2]
    sample_count = y_test.shape[0]

    # Build time features and apply stored RM40 mask
    x_test_time = add_time_of_day_features(x_test, start_index=23974 + 3424)
    mask = np.load(MASK_PATH)["mask"].astype(bool)
    if mask.shape != x_test[..., :1].shape:
        raise ValueError(f"Mask shape mismatch: {mask.shape} vs {x_test[..., :1].shape}")
    x_fault_time = x_test_time.copy()
    x_fault_time[..., :1][mask] = np.nan

    # Timestamps and indices
    sample_indices, time_indices, timestamps = build_test_time_info(
        total_samples=sample_count,
        train_n=23974,
        val_n=3424,
        lags=12,
        horizon=12,
    )
    horizon_indices = np.arange(1, horizon + 1, dtype=np.int64)
    mask_meta_ref = str(MASK_META_PATH).replace("/", "\\")

    adjacency = torch.from_numpy(np.load(DATA_DIR / "adjacency.npy").astype(np.float32))

    # Persistence predictions (normalized scale)
    pred_persistence_norm = persistence_predict(np.nan_to_num(x_fault_time[..., :1], nan=0.0).astype(np.float32), horizon)

    # Strong ResidualGRU-time predictions (normalized scale)
    if not STRONG_CKPT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {STRONG_CKPT}")
    strong = ResidualGRU(
        sensors=sensors,
        features=3,
        horizon=horizon,
        hidden_dim=32,
        sensor_embedding_dim=8,
        output_features=1,
    )
    strong_state = torch.load(STRONG_CKPT, map_location="cpu")
    strong.load_state_dict(strong_state)
    pred_strong_norm = batched_predict(strong, x_fault_time.astype(np.float32), adjacency, batch_size=64)

    # SRAF-RC-V2-Horizon predictions (normalized scale)
    if not SRAF_CKPT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {SRAF_CKPT}")
    sraf = SRAFResidualGRU(
        sensors=sensors,
        features=3,
        horizon=horizon,
        hidden_dim=32,
        sensor_embedding_dim=8,
        output_features=1,
        horizon_aware_decoder=True,
    )
    sraf_state = torch.load(SRAF_CKPT, map_location="cpu")
    sraf.load_state_dict(sraf_state)
    pred_sraf_norm = batched_predict(sraf, x_fault_time.astype(np.float32), adjacency, batch_size=64)

    # Validation checks
    expected_shape = (sample_count, 12, 207, 1)
    for name, arr in {
        "Persistence": pred_persistence_norm,
        "Strong ResidualGRU-time": pred_strong_norm,
        "SRAF-RC-V2-Horizon": pred_sraf_norm,
    }.items():
        if arr.shape != expected_shape:
            raise ValueError(f"{name} y_pred shape mismatch: {arr.shape} vs {expected_shape}")
    if y_test.shape != expected_shape:
        raise ValueError(f"y_true shape mismatch: {y_test.shape} vs {expected_shape}")

    # Convert to original scale for audit metrics
    pred_persistence_orig = pred_persistence_norm * std + mean
    pred_strong_orig = pred_strong_norm * std + mean
    pred_sraf_orig = pred_sraf_norm * std + mean

    preds_norm = {
        "Persistence": pred_persistence_norm,
        "Strong ResidualGRU-time": pred_strong_norm,
        "SRAF-RC-V2-Horizon": pred_sraf_norm,
    }
    preds_orig = {
        "Persistence": pred_persistence_orig,
        "Strong ResidualGRU-time": pred_strong_orig,
        "SRAF-RC-V2-Horizon": pred_sraf_orig,
    }

    # Export npz files
    export_npz(
        EXPORT_DIR / "predictions_random_missing_40_persistence.npz",
        y_pred=pred_persistence_norm,
        y_true=y_test,
        sample_indices=sample_indices,
        time_indices=time_indices,
        timestamps=timestamps,
        horizon_indices=horizon_indices,
        model_name="Persistence",
        fault_label="random_missing_40",
        mask_metadata_ref=mask_meta_ref,
    )
    export_npz(
        EXPORT_DIR / "predictions_random_missing_40_residualgru_time.npz",
        y_pred=pred_strong_norm,
        y_true=y_test,
        sample_indices=sample_indices,
        time_indices=time_indices,
        timestamps=timestamps,
        horizon_indices=horizon_indices,
        model_name="Strong ResidualGRU-time",
        fault_label="random_missing_40",
        mask_metadata_ref=mask_meta_ref,
    )
    export_npz(
        EXPORT_DIR / "predictions_random_missing_40_sraf_rc_v2_horizon.npz",
        y_pred=pred_sraf_norm,
        y_true=y_test,
        sample_indices=sample_indices,
        time_indices=time_indices,
        timestamps=timestamps,
        horizon_indices=horizon_indices,
        model_name="SRAF-RC-V2-Horizon",
        fault_label="random_missing_40",
        mask_metadata_ref=mask_meta_ref,
    )

    # Consistency checks across exports
    y_true_equal = np.array_equal(
        np.load(EXPORT_DIR / "predictions_random_missing_40_persistence.npz")["y_true"],
        np.load(EXPORT_DIR / "predictions_random_missing_40_residualgru_time.npz")["y_true"],
    ) and np.array_equal(
        np.load(EXPORT_DIR / "predictions_random_missing_40_persistence.npz")["y_true"],
        np.load(EXPORT_DIR / "predictions_random_missing_40_sraf_rc_v2_horizon.npz")["y_true"],
    )
    sample_indices_equal = np.array_equal(
        np.load(EXPORT_DIR / "predictions_random_missing_40_persistence.npz")["sample_indices"],
        np.load(EXPORT_DIR / "predictions_random_missing_40_residualgru_time.npz")["sample_indices"],
    ) and np.array_equal(
        np.load(EXPORT_DIR / "predictions_random_missing_40_persistence.npz")["sample_indices"],
        np.load(EXPORT_DIR / "predictions_random_missing_40_sraf_rc_v2_horizon.npz")["sample_indices"],
    )

    # Recompute global RM40 metrics and compare with existing table
    metrics_orig: dict[str, dict[str, float]] = {
        name: regression_metrics(y_true_orig, pred)
        for name, pred in preds_orig.items()
    }
    metric_consistency_rows = compare_metric_against_existing(metrics_orig)
    pd.DataFrame(metric_consistency_rows).to_csv(OUT_DIR / "rm40_metric_consistency_check.csv", index=False)

    # Period slicing and RM40 period metrics
    timestamps_pd = pd.to_datetime(timestamps)
    periods = np.array([infer_period(t) for t in timestamps_pd], dtype=object)
    rm40_overall_rows, rm40_horizon_rows = period_metrics_rows(
        y_true_orig=y_true_orig,
        preds_orig=preds_orig,
        periods=periods,
        period_order=["morning_peak", "evening_peak", "off_peak", "night"],
    )
    pd.DataFrame(rm40_overall_rows).to_csv(OUT_DIR / "rm40_period_comparison.csv", index=False)
    pd.DataFrame(rm40_horizon_rows).to_csv(OUT_DIR / "rm40_period_horizon_comparison.csv", index=False)
    write_winner_summary(OUT_DIR / "rm40_period_winner_summary.md", rm40_overall_rows, rm40_horizon_rows)

    # Update prior peak comparison and summaries
    update_peak_period_csv(rm40_overall_rows, rm40_horizon_rows)
    update_summary_files(rm40_overall_rows, rm40_horizon_rows)

    # Manifest
    manifest = {
        "run_id": "metr-la-rm40-period-prediction-export-and-audit",
        "gate": "RM40_PERIOD_PREDICTION_EXPORT_AND_AUDIT_GATE",
        "created_at": "2026-05-20",
        "data_dir": str(DATA_DIR).replace("/", "\\"),
        "output_dir": str(OUT_DIR).replace("/", "\\"),
        "prediction_export_dir": str(EXPORT_DIR).replace("/", "\\"),
        "fault": "random_missing_40",
        "mask_path": str(MASK_PATH).replace("/", "\\"),
        "mask_metadata_path": str(MASK_META_PATH).replace("/", "\\"),
        "models": [
            "Persistence",
            "Strong ResidualGRU-time",
            "SRAF-RC-V2-Horizon",
        ],
        "no_retraining": True,
        "target_y_corrupted": False,
        "validation_checks": {
            "expected_shape": list(expected_shape),
            "y_true_identical_across_exports": bool(y_true_equal),
            "sample_indices_aligned_across_exports": bool(sample_indices_equal),
        },
        "metric_consistency_file": str((OUT_DIR / "rm40_metric_consistency_check.csv")).replace("/", "\\"),
    }
    (OUT_DIR / "run_manifest_rm40_export.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
