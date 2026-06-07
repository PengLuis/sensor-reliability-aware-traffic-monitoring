"""Run PEMS_BAY_SRAF_ID_TRANSFER_PRECHECK_AND_CONFIRMATION_GATE.

This script intentionally keeps the METR-LA SRAF-ID architecture unchanged and
adapts only the data loading, identity-feature audit, labels, and reporting to
PEMS-BAY.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_metr_la_sraf_stid_same_backbone_gain import (  # noqa: E402
    FAULT_SETTINGS,
    TRAIN_FAULTS,
    build_official_stid,
    build_sraf_stid,
    clean_input_for_backbone,
    corruption_aware_batch,
    eval_loss,
    fixed_corrupt_val_sets,
    iter_batches,
    make_loss,
    model_param_count,
    predict_model,
    reliability_stats,
    train_sraf_stid,
)
from scripts.run_metr_la_strong_clean_backbone_integration import apply_fault, resolve_device  # noqa: E402
from src.models.baselines import persistence_predict  # noqa: E402
from src.models.strong_backbones import OfficialStyleSTID  # noqa: E402


RUN_ID_PRECHECK = "pems-bay-sraf-id-transfer-precheck"
RUN_ID_FULL = "pems-bay-sraf-id-full-confirmation"
DATASET = "PEMS-BAY"


def inverse_scale(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return x * std + mean


def safe_metrics(y_true_norm: np.ndarray, y_pred_norm: np.ndarray, mean: float, std: float) -> dict[str, float]:
    y_true = inverse_scale(y_true_norm, mean, std)
    y_pred = inverse_scale(y_pred_norm, mean, std)
    diff = y_pred - y_true
    out = {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mape": float(np.mean(np.abs(diff) / np.maximum(np.abs(y_true), 1.0)) * 100.0),
    }
    for step in (3, 6, 12):
        out[f"mae_h{step}"] = float(np.mean(np.abs(y_pred[:, step - 1] - y_true[:, step - 1])))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/pems-bay")
    parser.add_argument("--output-dir", default="experiments/pems-bay-sraf-id-transfer-precheck")
    parser.add_argument("--train-limit", type=int, default=10000)
    parser.add_argument("--val-limit", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--loss", choices=["mae", "mse"], default="mae")
    parser.add_argument("--lambda-repair", type=float, default=0.05)
    parser.add_argument("--lambda-rel", type=float, default=0.01)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full-confirmation", action="store_true")
    return parser


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_npz_pair(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return data["x"].astype(np.float32), data["y"].astype(np.float32)


def load_split(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    return load_npz_pair(data_dir / f"{split}.npz")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_scale(data_dir: Path) -> tuple[float, float]:
    stats_path = data_dir / "dataset_stats.json"
    if not stats_path.exists():
        stats_path = data_dir / "metadata.json"
    stats = load_json(stats_path)
    if isinstance(stats.get("normalization"), dict):
        stats = stats["normalization"]
    return float(stats["mean"]), float(stats["std"])


def timestamp_for_index(start: pd.Timestamp, index: np.ndarray, sampling_minutes: int) -> pd.DatetimeIndex:
    flat = index.reshape(-1)
    offsets = pd.to_timedelta(flat.astype(np.int64) * sampling_minutes, unit="m")
    return pd.DatetimeIndex(start + offsets)


def add_pems_identity_features(
    x: np.ndarray,
    start_index: int,
    time_meta: dict[str, Any],
) -> np.ndarray:
    samples, length, sensors, _ = x.shape
    out = np.zeros((samples, length, sensors, 3), dtype=np.float32)
    out[..., 0:1] = x[..., 0:1]
    sample_idx = np.arange(samples, dtype=np.int64)[:, None]
    step_idx = np.arange(length, dtype=np.int64)[None, :]
    time_idx = start_index + sample_idx + step_idx

    sampling_minutes = int(time_meta.get("sampling_minutes", 5))
    timestamp_status = str(time_meta.get("timestamp_status", "inferred"))
    start_ts_raw = time_meta.get("start_timestamp")
    if timestamp_status == "real" and start_ts_raw:
        timestamps = timestamp_for_index(pd.Timestamp(start_ts_raw), time_idx, sampling_minutes)
        hours = timestamps.hour.to_numpy().reshape(samples, length)
        minutes = timestamps.minute.to_numpy().reshape(samples, length)
        tod_index = hours * (60 // sampling_minutes) + (minutes // sampling_minutes)
        dow_index = timestamps.dayofweek.to_numpy().reshape(samples, length)
    else:
        buckets_per_day = 24 * 60 // sampling_minutes
        tod_index = time_idx % buckets_per_day
        dow_index = (time_idx // buckets_per_day) % 7
    out[..., 1] = (tod_index.astype(np.float32) / 288.0)[:, :, None]
    out[..., 2] = (dow_index.astype(np.float32) / 7.0)[:, :, None]
    return out


def identity_audit_payload(x: np.ndarray, start_index: int, time_meta: dict[str, Any]) -> dict[str, Any]:
    samples, length, _, _ = x.shape
    sample_idx = np.arange(samples, dtype=np.int64)[:, None]
    step_idx = np.arange(length, dtype=np.int64)[None, :]
    time_idx = start_index + sample_idx + step_idx
    sampling_minutes = int(time_meta.get("sampling_minutes", 5))
    timestamp_status = str(time_meta.get("timestamp_status", "inferred"))
    start_ts_raw = time_meta.get("start_timestamp")
    if timestamp_status == "real" and start_ts_raw:
        timestamps = timestamp_for_index(pd.Timestamp(start_ts_raw), time_idx, sampling_minutes)
        tod = (timestamps.hour.to_numpy() * 12 + timestamps.minute.to_numpy() // 5).astype(np.int64)
        dow = timestamps.dayofweek.to_numpy().astype(np.int64)
    else:
        tod = (time_idx.reshape(-1) % 288).astype(np.int64)
        dow = ((time_idx.reshape(-1) // 288) % 7).astype(np.int64)
    return {
        "start_index": int(start_index),
        "source": "real_timestamp" if timestamp_status == "real" and start_ts_raw else "inferred_contiguous_5min",
        "tod_min": int(np.min(tod)),
        "tod_max": int(np.max(tod)),
        "tod_unique_count": int(np.unique(tod).size),
        "tod_first_50": [int(v) for v in tod[:50]],
        "dow_min": int(np.min(dow)),
        "dow_max": int(np.max(dow)),
        "dow_unique_count": int(np.unique(dow).size),
        "dow_first_50": [int(v) for v in dow[:50]],
    }


def write_input_audit(data_dir: Path, audit_dir: Path) -> dict[str, Any]:
    audit_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_json(data_dir / "metadata.json")
    stats = load_json(data_dir / "dataset_stats.json")
    time_meta = load_json(data_dir / "time_metadata.json")
    adjacency = np.load(data_dir / "adjacency.npy")
    train_x, train_y = load_split(data_dir, "train")
    val_x, val_y = load_split(data_dir, "val")
    test_x, test_y = load_split(data_dir, "test")
    speed_fault, _, _ = apply_fault(train_x[:8], {"fault": "random_missing", "rate": 0.40, "label": "audit"}, seed=42, train_std=1.0)
    split_offsets = time_meta.get(
        "split_start_indices",
        {"train": 0, "val": train_x.shape[0], "test": train_x.shape[0] + val_x.shape[0]},
    )
    clean_aug = add_pems_identity_features(train_x[:8], int(split_offsets["train"]), time_meta)
    fault_aug = add_pems_identity_features(speed_fault, int(split_offsets["train"]), time_meta)
    audit = {
        "dataset": DATASET,
        "data_dir": str(data_dir),
        "metadata_dataset": metadata.get("dataset"),
        "train_shape": list(train_x.shape),
        "val_shape": list(val_x.shape),
        "test_shape": list(test_x.shape),
        "train_y_shape": list(train_y.shape),
        "val_y_shape": list(val_y.shape),
        "test_y_shape": list(test_y.shape),
        "expected_train_samples": 36465,
        "expected_val_samples": 5209,
        "expected_test_samples": 10419,
        "counts_match_import_gate": bool(train_x.shape[0] == 36465 and val_x.shape[0] == 5209 and test_x.shape[0] == 10419),
        "sensor_count": int(train_x.shape[2]),
        "adjacency_shape": list(adjacency.shape),
        "adjacency_matches_sensor_count": bool(adjacency.shape == (train_x.shape[2], train_x.shape[2])),
        "scaler_source": stats.get("scaler_source") or metadata.get("normalization", {}).get("scaler_source"),
        "scaler_train_x_only": (stats.get("scaler_source") or metadata.get("normalization", {}).get("scaler_source")) == "train_x_only",
        "normalization_mean": stats.get("mean"),
        "normalization_std": stats.get("std"),
        "timestamp_status": time_meta.get("timestamp_status"),
        "real_timestamp_metadata_exists": bool(time_meta.get("real_timestamp_metadata_exists")),
        "identity_source": "real_timestamp" if time_meta.get("timestamp_status") == "real" else "inferred",
        "target_y_corrupted": bool(metadata.get("target_y_corrupted", False)),
        "speed_only_corruption_preserves_tod_dow": bool(np.array_equal(clean_aug[..., 1:], fault_aug[..., 1:])),
        "no_nan_inf_train_val_test": bool(
            np.isfinite(train_x).all()
            and np.isfinite(train_y).all()
            and np.isfinite(val_x).all()
            and np.isfinite(val_y).all()
            and np.isfinite(test_x).all()
            and np.isfinite(test_y).all()
        ),
        "identity_feature_audit": {
            "train": identity_audit_payload(train_x[:512], int(split_offsets["train"]), time_meta),
            "val": identity_audit_payload(val_x[:512], int(split_offsets["val"]), time_meta),
            "test": identity_audit_payload(test_x[:512], int(split_offsets["test"]), time_meta),
        },
    }
    (audit_dir / "input_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    summary = [
        "# PEMS-BAY SRAF-ID Input Audit",
        "",
        f"- Train/val/test shapes: `{audit['train_shape']}` / `{audit['val_shape']}` / `{audit['test_shape']}`",
        f"- Adjacency shape: `{audit['adjacency_shape']}`; matches N: `{audit['adjacency_matches_sensor_count']}`",
        f"- Scaler source: `{audit['scaler_source']}`; train-X-only: `{audit['scaler_train_x_only']}`",
        f"- Timestamp status: `{audit['timestamp_status']}`; identity source: `{audit['identity_source']}`",
        f"- Speed-only corruption preserves tod/dow: `{audit['speed_only_corruption_preserves_tod_dow']}`",
        f"- Target corrupted: `{audit['target_y_corrupted']}`",
    ]
    (audit_dir / "input_audit_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return audit


def train_id_mlp_clean(
    model: OfficialStyleSTID,
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    loss_fn = make_loss(args.loss)
    best_val = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0
    rows: list[dict[str, Any]] = []
    start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in iter_batches(train_x, train_y, args.batch_size, shuffle=True, seed=args.seed, epoch=epoch):
            xb_t = torch.from_numpy(clean_input_for_backbone(xb)).to(device)
            yb_t = torch.from_numpy(yb.astype(np.float32)).to(device)
            pred = model(xb_t)
            loss = loss_fn(pred, yb_t)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        clean_val = eval_loss(model, val_x, val_y, args.batch_size, device, loss_fn)
        scheduler.step(clean_val)
        improved = clean_val < best_val - 1.0e-6
        if improved:
            best_val = clean_val
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        rows.append(
            {
                "model": model_name,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "clean_val_loss": clean_val,
                "corruption_aware_val_loss": "not_applicable",
                "selection_val_loss": clean_val,
                "best_selection_val_loss": best_val,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "improved": improved,
                "early_stop_triggered": False,
            }
        )
        print(f"{model_name} epoch={epoch} train={np.mean(losses):.6f} clean_val={clean_val:.6f}", flush=True)
        if no_improve >= args.patience:
            rows[-1]["early_stop_triggered"] = True
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, run_dir / "best_checkpoint.pt")
    return {"training_time_sec": perf_counter() - start, "best_epoch": best_epoch, "best_val_loss": best_val}, rows


def train_id_mlp_ca(
    model: OfficialStyleSTID,
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    loss_fn = make_loss(args.loss)
    fixed_val = fixed_corrupt_val_sets(val_x, args.seed)
    best_val = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0
    rows: list[dict[str, Any]] = []
    start = perf_counter()
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in iter_batches(train_x, train_y, args.batch_size, shuffle=True, seed=args.seed, epoch=epoch):
            setting = TRAIN_FAULTS[step % len(TRAIN_FAULTS)]
            x_corrupt, _, _ = corruption_aware_batch(xb, setting, args.seed + step)
            xb_t = torch.from_numpy(clean_input_for_backbone(x_corrupt)).to(device)
            yb_t = torch.from_numpy(yb.astype(np.float32)).to(device)
            pred = model(xb_t)
            loss = loss_fn(pred, yb_t)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            step += 1
        clean_val = eval_loss(model, val_x, val_y, args.batch_size, device, loss_fn)
        corrupt_vals = [eval_loss(model, vx, val_y, args.batch_size, device, loss_fn) for vx, _, _ in fixed_val]
        corrupt_val = float(np.mean(corrupt_vals))
        selection_val = 0.5 * clean_val + 0.5 * corrupt_val
        scheduler.step(selection_val)
        improved = selection_val < best_val - 1.0e-6
        if improved:
            best_val = selection_val
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        rows.append(
            {
                "model": model_name,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "clean_val_loss": clean_val,
                "corruption_aware_val_loss": corrupt_val,
                "selection_val_loss": selection_val,
                "best_selection_val_loss": best_val,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "improved": improved,
                "early_stop_triggered": False,
            }
        )
        print(
            f"{model_name} epoch={epoch} train={np.mean(losses):.6f} "
            f"clean_val={clean_val:.6f} corrupt_val={corrupt_val:.6f}",
            flush=True,
        )
        if no_improve >= args.patience:
            rows[-1]["early_stop_triggered"] = True
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, run_dir / "best_checkpoint.pt")
    return {"training_time_sec": perf_counter() - start, "best_epoch": best_epoch, "best_val_loss": best_val}, rows


def save_run_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def format_float(value: Any, digits: int = 6) -> str:
    if isinstance(value, (float, int, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def main() -> None:
    args = build_parser().parse_args()
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.smoke:
        args.train_limit = 512
        args.val_limit = 128
        args.epochs = 2
        args.patience = 1
        test_limit: int | None = 256
        run_id = "pems-bay-sraf-id-transfer-smoke"
    elif args.full_confirmation:
        args.train_limit = None
        args.val_limit = None
        test_limit = None
        run_id = RUN_ID_FULL
    else:
        test_limit = None
        run_id = RUN_ID_PRECHECK

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "models"
    model_dir.mkdir(exist_ok=True)
    fault_dir = out_dir / "fault_masks"
    fault_dir.mkdir(exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    audit_dir = ROOT / "experiments" / "pems-bay-sraf-id-transfer"
    input_audit = write_input_audit(data_dir, audit_dir)
    (out_dir / "input_audit.json").write_text(json.dumps(input_audit, indent=2), encoding="utf-8")

    device = resolve_device(args.device)
    train_x_base_full, train_y_full = load_split(data_dir, "train")
    val_x_base_full, val_y_full = load_split(data_dir, "val")
    test_x_base_full, test_y_full = load_split(data_dir, "test")
    mean, std = load_scale(data_dir)
    time_meta = load_json(data_dir / "time_metadata.json")
    adjacency_np = np.load(data_dir / "adjacency.npy").astype(np.float32)
    adjacency = torch.from_numpy(adjacency_np).to(device)

    train_x_base = train_x_base_full if args.train_limit is None else train_x_base_full[: args.train_limit]
    train_y = train_y_full if args.train_limit is None else train_y_full[: args.train_limit]
    val_x_base = val_x_base_full if args.val_limit is None else val_x_base_full[: args.val_limit]
    val_y = val_y_full if args.val_limit is None else val_y_full[: args.val_limit]
    test_x_base = test_x_base_full if test_limit is None else test_x_base_full[:test_limit]
    test_y = test_y_full if test_limit is None else test_y_full[:test_limit]

    full_train_count = train_x_base_full.shape[0]
    full_val_count = val_x_base_full.shape[0]
    split_offsets = time_meta.get(
        "split_start_indices",
        {"train": 0, "val": full_train_count, "test": full_train_count + full_val_count},
    )
    train_x_id = add_pems_identity_features(train_x_base, int(split_offsets["train"]), time_meta)
    val_x_id = add_pems_identity_features(val_x_base, int(split_offsets["val"]), time_meta)
    test_start = int(split_offsets["test"])

    fault_inputs: dict[str, np.ndarray] = {}
    fault_masks: dict[str, np.ndarray] = {}
    observed_masks: dict[str, np.ndarray] = {}
    identity_preserved_by_fault = True
    clean_identity_test = add_pems_identity_features(test_x_base, test_start, time_meta)[..., 1:]
    for idx, setting in enumerate(FAULT_SETTINGS):
        label = setting["label"]
        speed_fault, mask, meta = apply_fault(test_x_base, setting, seed=args.seed + idx, train_std=1.0)
        fault_inputs[label] = add_pems_identity_features(speed_fault, test_start, time_meta)
        fault_masks[label] = mask.astype(bool)
        observed_masks[label] = np.isfinite(speed_fault).astype(np.float32)
        identity_preserved_by_fault = identity_preserved_by_fault and bool(np.array_equal(clean_identity_test, fault_inputs[label][..., 1:]))
        meta = {
            **setting,
            **meta,
            "dataset": DATASET,
            "label": label,
            "target_corrupted": False,
            "speed_channel_only_corruption": True,
            "tod_dow_unchanged": bool(np.array_equal(clean_identity_test, fault_inputs[label][..., 1:])),
            "mask_path": str(fault_dir / f"{label}_mask.npz"),
        }
        np.savez_compressed(fault_dir / f"{label}_mask.npz", mask=mask)
        (fault_dir / f"{label}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    failed_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []

    sensors = train_x_base.shape[2]
    input_length = train_x_base.shape[1]
    horizon = train_y.shape[1]

    models: dict[str, tuple[str, nn.Module | None]] = {"Persistence": ("persistence", None)}

    for name, trainer, corruption_aware in [
        ("ID-MLP-clean", train_id_mlp_clean, False),
        ("ID-MLP-CA", train_id_mlp_ca, True),
    ]:
        try:
            model = build_official_stid(sensors=sensors, input_length=input_length, horizon=horizon)
            run_model_dir = model_dir / name
            run_model_dir.mkdir(exist_ok=True)
            meta, curves = trainer(model, name, train_x_id, train_y, val_x_id, val_y, args, run_model_dir, device)
            training_rows.extend(curves)
            complexity_rows.append(
                {
                    "model": name,
                    "parameter_count": model_param_count(model),
                    "training_time_sec": meta["training_time_sec"],
                    "best_epoch": meta["best_epoch"],
                    "best_val_loss": meta["best_val_loss"],
                    "corruption_aware_training": corruption_aware,
                }
            )
            models[name] = ("id_mlp", model)
        except Exception as exc:
            failed_rows.append({"model": name, "status": "failed", "reason": repr(exc)})
            if name in {"ID-MLP-CA"}:
                raise

    try:
        sraf_model = build_sraf_stid(sensors=sensors, input_length=input_length, horizon=horizon, use_reliability_gate=True)
        sraf_dir = model_dir / "SRAF-ID"
        sraf_dir.mkdir(exist_ok=True)
        sraf_meta, sraf_curves = train_sraf_stid(
            sraf_model,
            "SRAF-ID",
            train_x_id,
            train_y,
            val_x_id,
            val_y,
            args,
            sraf_dir,
            device,
            adjacency,
        )
        training_rows.extend(sraf_curves)
        complexity_rows.append(
            {
                "model": "SRAF-ID",
                "parameter_count": model_param_count(sraf_model),
                "training_time_sec": sraf_meta["training_time_sec"],
                "best_epoch": sraf_meta["best_epoch"],
                "best_val_loss": sraf_meta["best_val_loss"],
                "corruption_aware_training": True,
                "lambda_repair": args.lambda_repair,
                "lambda_rel": args.lambda_rel,
            }
        )
        models["SRAF-ID"] = ("sraf_id", sraf_model)
    except Exception as exc:
        failed_rows.append({"model": "SRAF-ID", "status": "failed", "reason": repr(exc)})
        raise

    metrics_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    inference_times: dict[tuple[str, str], float] = {}

    for model_name, (kind, model) in models.items():
        for setting in FAULT_SETTINGS:
            label = setting["label"]
            if kind == "persistence":
                start = perf_counter()
                pred = persistence_predict(clean_input_for_backbone(fault_inputs[label])[..., :1], test_y.shape[1])
                infer_time = perf_counter() - start
                comps = None
            elif kind == "id_mlp":
                pred, infer_time, comps = predict_model(model, fault_inputs[label], args.batch_size, device, sraf=False)
            elif kind == "sraf_id":
                pred, infer_time, comps = predict_model(
                    model,
                    fault_inputs[label],
                    args.batch_size,
                    device,
                    sraf=True,
                    observed_mask=observed_masks[label],
                    adjacency=adjacency,
                    return_components=True,
                )
                if comps is not None:
                    diag = reliability_stats(comps["reliability"], fault_masks[label], comps["repaired_speed"], test_x_base[..., :1])
                    diag["all_positions_marked_corrupted"] = bool(np.all(fault_masks[label]))
                    repair_rows.append({"model": model_name, "fault": label, **diag})
                    reliability_rows.append({"model": model_name, "fault": label, **diag})
            else:
                raise ValueError(kind)
            if not np.isfinite(pred).all():
                raise ValueError(f"Non-finite predictions for {model_name} under {label}")
            inference_times[(model_name, label)] = infer_time
            m = safe_metrics(test_y, pred, mean, std)
            metrics_rows.append(
                {
                    "dataset": DATASET,
                    "run_id": run_id,
                    "metrics_scale": "original",
                    "model": model_name,
                    "fault": label,
                    "fault_type": setting["fault"],
                    "severity_group": setting["severity_group"],
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
                    "dataset": DATASET,
                    "run_id": run_id,
                    "model": model_name,
                    "fault": label,
                    "h3_mae": m["mae_h3"],
                    "h6_mae": m["mae_h6"],
                    "h12_mae": m["mae_h12"],
                }
            )
            if model_name in {"ID-MLP-CA", "SRAF-ID"} and label in {"clean", "random_missing_40"}:
                np.savez_compressed(pred_dir / f"{model_name}_{label}_predictions.npz", y_pred=pred, y_true=test_y)

    clean_by_model = {r["model"]: float(r["mae"]) for r in metrics_rows if r["fault"] == "clean"}
    rdr_rows: list[dict[str, Any]] = []
    for row in metrics_rows:
        clean_mae = clean_by_model.get(row["model"], math.nan)
        fault_mae = float(row["mae"])
        rdr_rows.append(
            {
                "dataset": DATASET,
                "run_id": run_id,
                "model": row["model"],
                "fault": row["fault"],
                "fault_type": row["fault_type"],
                "severity_group": row["severity_group"],
                "clean_mae": clean_mae,
                "fault_mae": fault_mae,
                "rdr_mae": (fault_mae - clean_mae) / clean_mae if clean_mae else "TODO",
            }
        )

    id_clean_mae = clean_by_model.get("ID-MLP-clean", math.nan)
    clp_rows = [
        {
            "model": model_name,
            "id_mlp_clean_mae": id_clean_mae,
            "model_clean_mae": clean_mae,
            "clean_loss_penalty": (clean_mae - id_clean_mae) / id_clean_mae if id_clean_mae else "TODO",
        }
        for model_name, clean_mae in clean_by_model.items()
    ]

    rg_rows: list[dict[str, Any]] = []
    same_gain_rows: list[dict[str, Any]] = []
    for setting in FAULT_SETTINGS:
        label = setting["label"]
        ca = next(r for r in metrics_rows if r["model"] == "ID-MLP-CA" and r["fault"] == label)
        sraf = next(r for r in metrics_rows if r["model"] == "SRAF-ID" and r["fault"] == label)
        ca_mae = float(ca["mae"])
        sraf_mae = float(sraf["mae"])
        rg = {
            "fault": label,
            "id_mlp_ca_mae": ca_mae,
            "sraf_id_mae": sraf_mae,
            "absolute_delta_sraf_minus_ca": sraf_mae - ca_mae,
            "same_backbone_robustness_gain": (ca_mae - sraf_mae) / ca_mae,
            "sraf_better": sraf_mae < ca_mae,
        }
        rg_rows.append(rg)
        ca_rdr = next(r for r in rdr_rows if r["model"] == "ID-MLP-CA" and r["fault"] == label)
        sraf_rdr = next(r for r in rdr_rows if r["model"] == "SRAF-ID" and r["fault"] == label)
        ca_h = next(r for r in horizon_rows if r["model"] == "ID-MLP-CA" and r["fault"] == label)
        sraf_h = next(r for r in horizon_rows if r["model"] == "SRAF-ID" and r["fault"] == label)
        same_gain_rows.append(
            {
                **rg,
                "id_mlp_ca_rdr": ca_rdr["rdr_mae"],
                "sraf_id_rdr": sraf_rdr["rdr_mae"],
                "id_mlp_ca_h12_mae": ca_h["h12_mae"],
                "sraf_id_h12_mae": sraf_h["h12_mae"],
                "h12_delta_sraf_minus_ca": float(sraf_h["h12_mae"]) - float(ca_h["h12_mae"]),
            }
        )

    for row in complexity_rows:
        model_name = row["model"]
        row["clean_inference_time_sec"] = inference_times.get((model_name, "clean"), "TODO")
        fault_latencies = [v for (m, f), v in inference_times.items() if m == model_name and f != "clean"]
        row["average_fault_inference_time_sec"] = float(np.mean(fault_latencies)) if fault_latencies else "TODO"
        ca_clean = inference_times.get(("ID-MLP-CA", "clean"))
        ca_faults = [v for (m, f), v in inference_times.items() if m == "ID-MLP-CA" and f != "clean"]
        ca_avg_fault = float(np.mean(ca_faults)) if ca_faults else None
        row["clean_latency_overhead_vs_id_mlp_ca"] = (
            row["clean_inference_time_sec"] - ca_clean
            if isinstance(row["clean_inference_time_sec"], float) and ca_clean is not None
            else "TODO"
        )
        row["avg_fault_latency_overhead_vs_id_mlp_ca"] = (
            row["average_fault_inference_time_sec"] - ca_avg_fault
            if isinstance(row["average_fault_inference_time_sec"], float) and ca_avg_fault is not None
            else "TODO"
        )

    if args.smoke:
        failed_rows.append({"model": "full_confirmation", "status": "skipped", "reason": "Smoke run only."})
    elif not args.full_confirmation:
        failed_rows.append({"model": "full_confirmation", "status": "not_run", "reason": "Bounded precheck gate stops before full confirmation unless explicitly launched."})

    write_csv(out_dir / "metrics_by_model_fault.csv", metrics_rows)
    write_csv(out_dir / "horizon_metrics.csv", horizon_rows)
    write_csv(out_dir / "robustness_rdr.csv", rdr_rows)
    write_csv(out_dir / "clean_loss_penalty.csv", clp_rows)
    write_csv(out_dir / "same_backbone_gain_summary.csv", same_gain_rows)
    write_csv(out_dir / "repair_diagnostics_by_fault.csv", repair_rows)
    write_csv(out_dir / "reliability_diagnostics.csv", reliability_rows)
    write_csv(out_dir / "complexity_metrics.csv", complexity_rows)
    write_csv(out_dir / "training_curves.csv", training_rows)
    write_csv(out_dir / "failed_or_skipped_models.csv", failed_rows)

    faulty = [r for r in rg_rows if r["fault"] != "clean"]
    improved_faults = sum(bool(r["sraf_better"]) for r in faulty)
    severe_labels = {"random_missing_40", "continuous_outage_24", "gaussian_noise_high", "linear_drift_high"}
    severe_improved = sum(bool(next(r for r in rg_rows if r["fault"] == label)["sraf_better"]) for label in severe_labels)
    h12_improved = sum(float(r["h12_delta_sraf_minus_ca"]) < 0.0 for r in same_gain_rows if r["fault"] != "clean")
    sraf_clp = float(next(r for r in clp_rows if r["model"] == "SRAF-ID")["clean_loss_penalty"])
    no_nan = all(np.isfinite(float(r["mae"])) for r in metrics_rows)
    if improved_faults >= 4 and severe_improved >= 3 and sraf_clp <= 0.15 and h12_improved >= 4 and no_nan:
        status = "PASS"
    elif improved_faults > 0 or severe_improved >= 2:
        status = "PARTIAL"
    else:
        status = "FAIL"

    clean_metrics = {r["model"]: r for r in metrics_rows if r["fault"] == "clean"}
    rm40_gain = next(r for r in same_gain_rows if r["fault"] == "random_missing_40")
    summary_lines = [
        "# PEMS-BAY SRAF-ID Transfer Summary",
        "",
        f"- Stage status: **{status}**",
        f"- Run ID: `{run_id}`",
        f"- Train/val/test used: `{train_x_base.shape[0]}` / `{val_x_base.shape[0]}` / `{test_x_base.shape[0]}`",
        f"- Device: `{device}`",
        f"- Fault identity preservation: `{identity_preserved_by_fault}`",
        f"- ID-MLP-clean clean MAE: `{format_float(clean_metrics['ID-MLP-clean']['mae'])}`",
        f"- ID-MLP-CA clean MAE: `{format_float(clean_metrics['ID-MLP-CA']['mae'])}`",
        f"- SRAF-ID clean MAE: `{format_float(clean_metrics['SRAF-ID']['mae'])}`",
        f"- SRAF-ID clean loss penalty vs ID-MLP-clean: `{sraf_clp:.6f}`",
        f"- SRAF-ID improved over ID-MLP-CA on `{improved_faults}/6` faulty settings.",
        f"- Severe-fault improvements: `{severe_improved}/4`.",
        f"- h12 improvements: `{h12_improved}/6` faulty settings.",
        f"- RM40 same-backbone gain: `{format_float(rm40_gain['same_backbone_robustness_gain'])}`",
        "",
        "Interpretation is bounded to this run. Persistence is a context baseline and is not the same-backbone proof target.",
    ]
    (out_dir / "candidate_selection_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    if args.full_confirmation:
        (out_dir / "pems_bay_sraf_id_full_confirmation_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    manifest = {
        "stage": "PEMS_BAY_SRAF_ID_TRANSFER_PRECHECK_AND_CONFIRMATION_GATE",
        "status": status,
        "run_id": run_id,
        "dataset": DATASET,
        "data_dir": str(data_dir),
        "output_dir": str(out_dir),
        "device": str(device),
        "seed": args.seed,
        "full_confirmation": bool(args.full_confirmation),
        "smoke": bool(args.smoke),
        "train_samples_full": int(train_x_base_full.shape[0]),
        "val_samples_full": int(val_x_base_full.shape[0]),
        "test_samples_full": int(test_x_base_full.shape[0]),
        "train_samples_used": int(train_x_base.shape[0]),
        "val_samples_used": int(val_x_base.shape[0]),
        "test_samples_used": int(test_x_base.shape[0]),
        "input_length": int(input_length),
        "horizon": int(horizon),
        "sensors": int(sensors),
        "input_features": "[speed_norm,tod_norm,dow_norm]",
        "target_corrupted": False,
        "speed_only_corruption": True,
        "tod_dow_modified_by_sraf": False,
        "identity_preserved_by_fault": bool(identity_preserved_by_fault),
        "metrics_scale": "original",
        "loss": args.loss,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lambda_repair": args.lambda_repair,
        "lambda_rel": args.lambda_rel,
        "decision": {
            "sraf_fault_wins_vs_id_mlp_ca": int(improved_faults),
            "sraf_severe_fault_wins_vs_id_mlp_ca": int(severe_improved),
            "sraf_h12_wins_vs_id_mlp_ca": int(h12_improved),
            "sraf_clean_loss_penalty": sraf_clp,
        },
    }
    save_run_manifest(out_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest["decision"], indent=2), flush=True)


if __name__ == "__main__":
    main()
