"""Run METR-LA formal-v4 time-aware SRAF fault matrix and ablation gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.corruptions.faults import (  # noqa: E402
    continuous_outage,
    gaussian_noise,
    linear_drift,
    random_missing,
    stuck_at_last_value,
)
from src.metrics.regression import regression_metrics  # noqa: E402
from src.models.baselines import historical_average_predict, persistence_predict  # noqa: E402
from src.models.residual_models import ResidualGRU, SRAFResidualGRU  # noqa: E402


FAULT_SETTINGS = [
    {"fault": "clean", "label": "clean"},
    {"fault": "random_missing", "rate": 0.10, "label": "random_missing_10", "severity_group": "low"},
    {"fault": "random_missing", "rate": 0.20, "label": "random_missing_20", "severity_group": "medium"},
    {"fault": "random_missing", "rate": 0.40, "label": "random_missing_40", "severity_group": "high"},
    {"fault": "continuous_outage", "length": 6, "label": "continuous_outage_6", "severity_group": "low"},
    {"fault": "continuous_outage", "length": 12, "label": "continuous_outage_12", "severity_group": "medium"},
    {"fault": "continuous_outage", "length": 24, "label": "continuous_outage_24", "severity_group": "high"},
    {"fault": "gaussian_noise", "severity": "low", "label": "gaussian_noise_low", "severity_group": "low"},
    {"fault": "gaussian_noise", "severity": "medium", "label": "gaussian_noise_medium", "severity_group": "medium"},
    {"fault": "gaussian_noise", "severity": "high", "label": "gaussian_noise_high", "severity_group": "high"},
    {"fault": "linear_drift", "severity": "low", "label": "linear_drift_low", "severity_group": "low"},
    {"fault": "linear_drift", "severity": "medium", "label": "linear_drift_medium", "severity_group": "medium"},
    {"fault": "linear_drift", "severity": "high", "label": "linear_drift_high", "severity_group": "high"},
    {"fault": "stuck_at_last_value", "severity": "low", "label": "stuck_at_last_value_low", "severity_group": "low"},
    {"fault": "stuck_at_last_value", "severity": "medium", "label": "stuck_at_last_value_medium", "severity_group": "medium"},
    {"fault": "stuck_at_last_value", "severity": "high", "label": "stuck_at_last_value_high", "severity_group": "high"},
]

ABLATION_FAULTS = {
    "clean",
    "random_missing_40",
    "continuous_outage_24",
    "gaussian_noise_high",
    "linear_drift_high",
    "stuck_at_last_value_high",
}

MODEL_SPECS = [
    {"model": "Persistence", "kind": "persistence", "role": "main", "eval": "full"},
    {"model": "HistoricalAverage", "kind": "historical_average", "role": "main", "eval": "full"},
    {
        "model": "ResidualGRU-time-clean",
        "kind": "residual",
        "role": "main",
        "train_protocol": "clean",
        "eval": "full",
    },
    {
        "model": "ResidualGRU-time-corruption-aware",
        "kind": "residual",
        "role": "main",
        "train_protocol": "corruption_aware",
        "eval": "full",
        "checkpoint": "experiments/metr-la-time-feature-parity/models/ResidualGRU-time-corruption-aware/best_checkpoint.pt",
        "source": "reused_time_feature_parity_checkpoint",
    },
    {
        "model": "SRAF-time-full",
        "kind": "sraf",
        "role": "main",
        "train_protocol": "corruption_aware",
        "eval": "full",
        "checkpoint": "experiments/metr-la-formal-v36-final-polish/configs/time_of_day_features_h32/best_checkpoint.pt",
        "source": "reused_v36_time_of_day_features_h32_checkpoint",
    },
    {
        "model": "SRAF-time-no-reliability-gate",
        "kind": "sraf",
        "role": "ablation",
        "train_protocol": "corruption_aware",
        "eval": "ablation",
        "sraf_kwargs": {"use_reliability_gate": False},
    },
    {
        "model": "SRAF-time-no-mask-aware-encoding",
        "kind": "sraf",
        "role": "ablation",
        "train_protocol": "corruption_aware",
        "eval": "ablation",
        "sraf_kwargs": {"use_mask_channel": False, "reliability_uses_mask": False},
    },
    {
        "model": "SRAF-time-no-temporal-repair",
        "kind": "sraf",
        "role": "ablation",
        "train_protocol": "corruption_aware",
        "eval": "ablation",
        "sraf_kwargs": {"use_temporal_repair": False},
    },
    {
        "model": "SRAF-time-no-spatial-repair",
        "kind": "sraf",
        "role": "ablation",
        "train_protocol": "corruption_aware",
        "eval": "ablation",
        "sraf_kwargs": {"use_spatial_repair": False},
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/metr-la")
    parser.add_argument("--output-dir", default="experiments/metr-la-formal-v4-time-ablation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--sensor-embedding-dim", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    return parser


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if path.name == "failed_or_skipped_runs.csv":
            path.write_text("model,status,reason\n", encoding="utf-8")
            return
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


def load_split(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(data_dir / f"{split}.npz")
    return data["x"].astype(np.float32), data["y"].astype(np.float32)


def load_scale(data_dir: Path) -> tuple[float, float]:
    stats = json.loads((data_dir / "dataset_stats.json").read_text(encoding="utf-8"))
    return float(stats["mean"]), float(stats["std"])


def inverse_scale(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return x * std + mean


def safe_metrics(y_true_norm: np.ndarray, y_pred_norm: np.ndarray, mean: float, std: float) -> dict[str, float]:
    return regression_metrics(inverse_scale(y_true_norm, mean, std), inverse_scale(y_pred_norm, mean, std))


def add_time_of_day_features(x: np.ndarray, start_index: int = 0) -> np.ndarray:
    samples, length, sensors, _ = x.shape
    offsets = (np.arange(samples)[:, None] + np.arange(length)[None, :] + start_index) % 288
    phase = 2.0 * np.pi * offsets.astype(np.float32) / 288.0
    sin = np.sin(phase)[:, :, None, None].repeat(sensors, axis=2)
    cos = np.cos(phase)[:, :, None, None].repeat(sensors, axis=2)
    return np.concatenate([x, sin.astype(np.float32), cos.astype(np.float32)], axis=-1)


def apply_fault(x: np.ndarray, setting: dict[str, Any], seed: int, train_std: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if setting["fault"] == "clean":
        return x.copy(), np.zeros_like(x, dtype=bool), {"fault": "clean", "seed": seed, "target_corrupted": False}
    if setting["fault"] == "random_missing":
        return random_missing(x, rate=setting["rate"], seed=seed)
    if setting["fault"] == "continuous_outage":
        return continuous_outage(x, length=setting["length"], seed=seed)
    if setting["fault"] == "gaussian_noise":
        return gaussian_noise(x, severity=setting["severity"], train_std=train_std, seed=seed)
    if setting["fault"] == "linear_drift":
        return linear_drift(x, severity=setting["severity"], train_std=train_std, seed=seed)
    if setting["fault"] == "stuck_at_last_value":
        return stuck_at_last_value(x, severity=setting["severity"], seed=seed)
    raise ValueError(f"Unknown fault setting: {setting}")


def iter_batches(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int, epoch: int) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(x.shape[0])
    if shuffle:
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(indices)
    return [(x[idx], y[idx]) for idx in np.array_split(indices, math.ceil(len(indices) / batch_size))]


def train_corruption_batch(x: np.ndarray, seed: int, step: int) -> np.ndarray:
    choices = [setting for setting in FAULT_SETTINGS if setting["fault"] != "clean"]
    setting = choices[(seed + step) % len(choices)]
    corrupted, _, _ = apply_fault(x, setting, seed + step, train_std=1.0)
    return corrupted


def model_param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def make_model(spec: dict[str, Any], sensors: int, horizon: int, args: argparse.Namespace) -> nn.Module:
    if spec["kind"] == "residual":
        return ResidualGRU(
            sensors=sensors,
            features=3,
            output_features=1,
            horizon=horizon,
            hidden_dim=args.hidden_dim,
            sensor_embedding_dim=args.sensor_embedding_dim,
        )
    if spec["kind"] == "sraf":
        kwargs = dict(spec.get("sraf_kwargs", {}))
        return SRAFResidualGRU(
            sensors=sensors,
            features=3,
            output_features=1,
            horizon=horizon,
            hidden_dim=args.hidden_dim,
            sensor_embedding_dim=args.sensor_embedding_dim,
            **kwargs,
        )
    raise ValueError(f"Spec does not require a torch model: {spec['model']}")


def evaluate_loss(model: nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int, adjacency: torch.Tensor) -> float:
    model.eval()
    loss_fn = nn.MSELoss()
    losses: list[float] = []
    with torch.no_grad():
        for xb, yb in iter_batches(x, y, batch_size, shuffle=False, seed=0, epoch=0):
            pred = model(torch.from_numpy(xb.astype(np.float32)), adjacency=adjacency)
            losses.append(float(loss_fn(pred, torch.from_numpy(yb.astype(np.float32))).detach().cpu()))
    return float(np.mean(losses))


def train_model(
    spec: dict[str, Any],
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    args: argparse.Namespace,
    run_dir: Path,
    adjacency: torch.Tensor,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    best_val = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0
    batch_step = 0
    rows: list[dict[str, Any]] = []
    start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in iter_batches(train_x, train_y, args.batch_size, shuffle=True, seed=args.seed, epoch=epoch):
            if spec.get("train_protocol") == "corruption_aware":
                xb = train_corruption_batch(xb, args.seed, batch_step)
            pred = model(torch.from_numpy(xb.astype(np.float32)), adjacency=adjacency)
            loss = loss_fn(pred, torch.from_numpy(yb.astype(np.float32)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            batch_step += 1
        train_loss = float(np.mean(losses))
        val_loss = evaluate_loss(model, val_x, val_y, args.batch_size, adjacency)
        improved = val_loss < best_val - 1.0e-6
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        row = {
            "run_id": "formal-v4-time-ablation",
            "model": spec["model"],
            "epoch": epoch,
            "train_protocol": spec.get("train_protocol", ""),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val,
            "improved": improved,
            "early_stop_triggered": False,
            "has_nan_or_inf": (not math.isfinite(train_loss)) or (not math.isfinite(val_loss)),
        }
        rows.append(row)
        print(f"{spec['model']} epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
        if no_improve >= args.patience:
            rows[-1]["early_stop_triggered"] = True
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, run_dir / "best_checkpoint.pt")
    (run_dir / "train_log.txt").write_text(
        "\n".join(
            f"epoch={r['epoch']},train_loss={r['train_loss']:.6f},val_loss={r['val_loss']:.6f},"
            f"best_val_loss={r['best_val_loss']:.6f},early_stop_triggered={r['early_stop_triggered']}"
            for r in rows
        ),
        encoding="utf-8",
    )
    return {"training_time_sec": perf_counter() - start, "best_epoch": best_epoch, "best_val_loss": best_val}, rows


def predict_model(model: nn.Module, x: np.ndarray, batch_size: int, adjacency: torch.Tensor) -> tuple[np.ndarray, float]:
    model.eval()
    preds = []
    start = perf_counter()
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[i : i + batch_size].astype(np.float32))
            preds.append(model(xb, adjacency=adjacency).cpu().numpy())
    return np.concatenate(preds, axis=0), perf_counter() - start


def timed_baseline_predict(kind: str, x: np.ndarray, horizon: int) -> tuple[np.ndarray, float]:
    start = perf_counter()
    x_filled = np.nan_to_num(x, nan=0.0).astype(np.float32)
    if kind == "persistence":
        pred = persistence_predict(x_filled[..., :1], horizon)
    elif kind == "historical_average":
        pred = historical_average_predict(x_filled[..., :1], horizon)
        pred = np.nan_to_num(pred, nan=0.0).astype(np.float32)
    else:
        raise ValueError(kind)
    return pred, perf_counter() - start


def load_training_meta_from_csv(path: Path, key_field: str, key_value: str) -> dict[str, float]:
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get(key_field) == key_value:
                    return {
                        "training_time_sec": float(row.get("training_time_sec", 0.0)),
                        "best_epoch": float(row.get("best_epoch", 0.0)),
                        "best_val_loss": float(row.get("best_val_loss", 0.0)),
                    }
    return {"training_time_sec": 0.0, "best_epoch": 0.0, "best_val_loss": 0.0}


def reused_training_meta(model_name: str) -> dict[str, float]:
    if model_name == "ResidualGRU-time-corruption-aware":
        return load_training_meta_from_csv(ROOT / "experiments/metr-la-time-feature-parity/complexity_metrics.csv", "model", model_name)
    if model_name == "SRAF-time-full":
        return load_training_meta_from_csv(ROOT / "experiments/metr-la-formal-v36-final-polish/complexity_metrics.csv", "config", "time_of_day_features_h32")
    return {"training_time_sec": 0.0, "best_epoch": 0.0, "best_val_loss": 0.0}


def training_meta_from_log(run_dir: Path) -> dict[str, Any]:
    log_path = run_dir / "train_log.txt"
    if not log_path.exists():
        return {
            "training_time_sec": "TODO_unavailable_after_timeout_resume",
            "best_epoch": "TODO",
            "best_val_loss": "TODO",
        }
    best_epoch: int | str = "TODO"
    best_val = math.inf
    for line in log_path.read_text(encoding="utf-8").splitlines():
        parts = {}
        for item in line.split(","):
            if "=" in item:
                key, value = item.split("=", 1)
                parts[key.strip()] = value.strip()
        try:
            epoch = int(parts["epoch"])
            val_loss = float(parts["val_loss"])
        except (KeyError, ValueError):
            continue
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
    return {
        "training_time_sec": "TODO_unavailable_after_timeout_resume",
        "best_epoch": best_epoch,
        "best_val_loss": best_val if math.isfinite(best_val) else "TODO",
    }


def prediction_distribution_row(model: str, fault: str, y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, Any]:
    return {
        "model": model,
        "fault": fault,
        "pred_mean": float(np.nanmean(y_pred)),
        "pred_std": float(np.nanstd(y_pred)),
        "pred_min": float(np.nanmin(y_pred)),
        "pred_max": float(np.nanmax(y_pred)),
        "target_mean": float(np.nanmean(y_true)),
        "target_std": float(np.nanstd(y_true)),
    }


def build_rdr_rows(metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_by_model = {row["model"]: float(row["mae"]) for row in metrics_rows if row["fault"] == "clean"}
    rows = []
    for row in metrics_rows:
        clean = clean_by_model.get(row["model"], math.nan)
        fault_mae = float(row["mae"])
        rows.append(
            {
                "dataset": row["dataset"],
                "run_id": row["run_id"],
                "model": row["model"],
                "role": row["role"],
                "fault": row["fault"],
                "fault_type": row["fault_type"],
                "severity_group": row["severity_group"],
                "clean_mae": clean,
                "fault_mae": fault_mae,
                "rdr_mae": (fault_mae - clean) / clean if math.isfinite(clean) and clean != 0 else math.nan,
            }
        )
    return rows


def row_value(rows: list[dict[str, Any]], model: str, fault: str, key: str) -> float:
    return float(next(row[key] for row in rows if row["model"] == model and row["fault"] == fault))


def build_ablation_summary(metrics_rows: list[dict[str, Any]], rdr_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants = [spec["model"] for spec in MODEL_SPECS if spec.get("role") == "ablation"]
    rows = []
    for variant in variants:
        for fault in sorted(ABLATION_FAULTS):
            if not any(row["model"] == variant and row["fault"] == fault for row in metrics_rows):
                continue
            full_mae = row_value(metrics_rows, "SRAF-time-full", fault, "mae")
            variant_mae = row_value(metrics_rows, variant, fault, "mae")
            full_rdr = row_value(rdr_rows, "SRAF-time-full", fault, "rdr_mae")
            variant_rdr = row_value(rdr_rows, variant, fault, "rdr_mae")
            rows.append(
                {
                    "dataset": "METR-LA",
                    "run_id": "formal-v4-time-ablation",
                    "full_model": "SRAF-time-full",
                    "ablation_model": variant,
                    "fault": fault,
                    "full_mae": full_mae,
                    "ablation_mae": variant_mae,
                    "delta_mae_ablation_minus_full": variant_mae - full_mae,
                    "full_rdr_mae": full_rdr,
                    "ablation_rdr_mae": variant_rdr,
                    "delta_rdr_ablation_minus_full": variant_rdr - full_rdr,
                    "full_better_mae": full_mae < variant_mae,
                    "full_better_rdr": full_rdr < variant_rdr,
                }
            )
    return rows


def fault_type(label: str) -> str:
    if label == "clean":
        return "clean"
    for setting in FAULT_SETTINGS:
        if setting["label"] == label:
            return setting["fault"]
    return "unknown"


def severity_group(label: str) -> str:
    if label == "clean":
        return "clean"
    for setting in FAULT_SETTINGS:
        if setting["label"] == label:
            return setting.get("severity_group", "unknown")
    return "unknown"


def build_manifest(args: argparse.Namespace, data_dir: Path, out_dir: Path, shapes: dict[str, int]) -> dict[str, Any]:
    return {
        "run_id": "metr-la-formal-v4-time-ablation",
        "gate": "METR-LA FORMAL-V4 TIME-AWARE SRAF FULL FAULT MATRIX AND ABLATION GATE",
        "created_at": "2026-05-17",
        "seed": args.seed,
        "data_dir": str(data_dir),
        "output_dir": str(out_dir),
        "dataset": {"name": "METR-LA", "L": 12, "H": 12, "N": 207, "F_target": 1, **shapes},
        "training": {
            "train_samples": "full train split unless --smoke is set",
            "val_samples": "full validation split unless --smoke is set",
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "loss": "MSE",
            "hidden_dim": args.hidden_dim,
            "sensor_embedding_dim": args.sensor_embedding_dim,
            "best_checkpoint": "saved by validation MSE",
        },
        "time_feature_construction": "For input sample window index s and input step l, append sin(2*pi*(s+l+split_start)/288) and cos(...). Only input-window indices are used.",
        "target_leakage_check": "Target Y is never corrupted and no target-horizon time features are appended.",
        "fault_protocol": "Corrupt only input X. Save identical test corruption masks once per fault setting and reuse across models.",
        "fault_settings": FAULT_SETTINGS,
        "models": MODEL_SPECS,
        "integrity_note": "Do not write final manuscript conclusions from this gate. Missing evidence must be marked failed/skipped.",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_x_base, train_y = load_split(data_dir, "train")
    val_x_base, val_y = load_split(data_dir, "val")
    test_x_base, test_y = load_split(data_dir, "test")
    if args.smoke:
        train_x_base, train_y = train_x_base[:128], train_y[:128]
        val_x_base, val_y = val_x_base[:64], val_y[:64]
        test_x_base, test_y = test_x_base[:128], test_y[:128]
    mean, std = load_scale(data_dir)
    adjacency = torch.from_numpy(np.load(data_dir / "adjacency.npy").astype(np.float32))
    train_x = add_time_of_day_features(train_x_base, 0)
    val_x = add_time_of_day_features(val_x_base, train_x_base.shape[0])
    split_start = train_x_base.shape[0] + val_x_base.shape[0]

    manifest = build_manifest(
        args,
        data_dir,
        out_dir,
        {
            "train_samples_used": int(train_x_base.shape[0]),
            "val_samples_used": int(val_x_base.shape[0]),
            "test_samples_used": int(test_x_base.shape[0]),
        },
    )
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    fault_dir = out_dir / "fault_masks"
    fault_dir.mkdir(parents=True, exist_ok=True)
    fault_inputs: dict[str, np.ndarray] = {}
    for idx, setting in enumerate(FAULT_SETTINGS):
        label = setting["label"]
        cx, mask, meta = apply_fault(test_x_base, setting, seed=args.seed + idx, train_std=1.0)
        fault_inputs[label] = add_time_of_day_features(cx, split_start)
        meta = {**setting, **meta, "label": label, "target_corrupted": False, "mask_path": str(fault_dir / f"{label}_mask.npz")}
        np.savez_compressed(fault_dir / f"{label}_mask.npz", mask=mask)
        (fault_dir / f"{label}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    specs = MODEL_SPECS if not args.smoke else MODEL_SPECS[:3]
    metrics_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []

    if args.smoke:
        failed_rows.extend({"model": spec["model"], "status": "skipped", "reason": "smoke mode"} for spec in MODEL_SPECS[3:])

    for spec in specs:
        model_name = spec["model"]
        run_dir = out_dir / "models" / model_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config_resolved.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        try:
            train_extra = {"training_time_sec": 0.0, "best_epoch": 0.0, "best_val_loss": 0.0}
            param_count: int | float = 0
            model: nn.Module | None = None
            if spec["kind"] in {"residual", "sraf"}:
                model = make_model(spec, train_x_base.shape[2], train_y.shape[1], args)
                output_checkpoint = run_dir / "best_checkpoint.pt"
                source_checkpoint = ROOT / spec["checkpoint"] if "checkpoint" in spec and not args.smoke else None
                if output_checkpoint.exists():
                    model.load_state_dict(torch.load(output_checkpoint, map_location="cpu"))
                    train_extra = reused_training_meta(model_name)
                    if train_extra["training_time_sec"] == 0.0 and spec.get("train_protocol"):
                        train_extra = training_meta_from_log(run_dir)
                elif source_checkpoint is not None and source_checkpoint.exists():
                    model.load_state_dict(torch.load(source_checkpoint, map_location="cpu"))
                    torch.save(model.state_dict(), output_checkpoint)
                    train_extra = reused_training_meta(model_name)
                    (run_dir / "train_log.txt").write_text(f"reused_checkpoint: {source_checkpoint}\n", encoding="utf-8")
                else:
                    train_extra, curves = train_model(spec, model, train_x, train_y, val_x, val_y, args, run_dir, adjacency)
                    curve_rows.extend(curves)
                param_count = model_param_count(model)

            labels = [setting["label"] for setting in FAULT_SETTINGS]
            if spec.get("eval") == "ablation":
                labels = [label for label in labels if label in ABLATION_FAULTS]
            for label in labels:
                if spec["kind"] in {"persistence", "historical_average"}:
                    pred, inference_time = timed_baseline_predict(spec["kind"], fault_inputs[label], test_y.shape[1])
                else:
                    assert model is not None
                    pred, inference_time = predict_model(model, fault_inputs[label], args.batch_size, adjacency)
                m = safe_metrics(test_y, pred, mean, std)
                if label == "clean":
                    np.savez_compressed(run_dir / "clean_predictions.npz", y_pred=pred, y_true=test_y)
                row = {
                    "dataset": "METR-LA",
                    "run_id": "formal-v4-time-ablation",
                    "metrics_scale": "original",
                    "model": model_name,
                    "role": spec["role"],
                    "train_protocol": spec.get("train_protocol", "none"),
                    "uses_time_features": spec["kind"] in {"residual", "sraf"},
                    "fault": label,
                    "fault_type": fault_type(label),
                    "severity_group": severity_group(label),
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "mape": m["mape"],
                    "mae_h3": m.get("mae_h3", math.nan),
                    "mae_h6": m.get("mae_h6", math.nan),
                    "mae_h12": m.get("mae_h12", math.nan),
                    "parameter_count": param_count,
                    "inference_time_sec": inference_time,
                    "training_time_sec": train_extra["training_time_sec"],
                    "best_epoch": train_extra["best_epoch"],
                    "best_val_loss": train_extra["best_val_loss"],
                }
                metrics_rows.append(row)
                horizon_rows.append(
                    {
                        "dataset": "METR-LA",
                        "run_id": "formal-v4-time-ablation",
                        "model": model_name,
                        "role": spec["role"],
                        "fault": label,
                        "mae_15min_h3": row["mae_h3"],
                        "mae_30min_h6": row["mae_h6"],
                        "mae_60min_h12": row["mae_h12"],
                    }
                )
                distribution_rows.append(prediction_distribution_row(model_name, label, pred, test_y))
            clean_row = next((row for row in metrics_rows if row["model"] == model_name and row["fault"] == "clean"), None)
            complexity_rows.append(
                {
                    "dataset": "METR-LA",
                    "run_id": "formal-v4-time-ablation",
                    "model": model_name,
                    "role": spec["role"],
                    "parameter_count": param_count,
                    "training_time_sec": train_extra["training_time_sec"],
                    "clean_inference_time_sec": clean_row["inference_time_sec"] if clean_row else math.nan,
                    "best_epoch": train_extra["best_epoch"],
                    "best_val_loss": train_extra["best_val_loss"],
                }
            )
        except Exception as exc:
            failed_rows.append({"model": model_name, "status": "failed", "reason": repr(exc)})
            print(f"FAILED {model_name}: {exc!r}", flush=True)

    rdr_rows = build_rdr_rows(metrics_rows)
    ablation_rows = build_ablation_summary(metrics_rows, rdr_rows)

    write_csv(out_dir / "metrics_by_model_fault.csv", metrics_rows)
    write_csv(out_dir / "horizon_metrics.csv", horizon_rows)
    write_csv(out_dir / "robustness_rdr.csv", rdr_rows)
    write_csv(out_dir / "ablation_summary.csv", ablation_rows)
    write_csv(out_dir / "complexity_metrics.csv", complexity_rows)
    write_csv(out_dir / "prediction_distribution.csv", distribution_rows)
    write_csv(out_dir / "failed_or_skipped_runs.csv", failed_rows)
    write_csv(out_dir / "training_curves.csv", curve_rows)

    tables_dir = ROOT / "paper" / "tables"
    write_csv(tables_dir / "table_metr_la_v4_time_main.csv", [row for row in metrics_rows if row["role"] == "main"])
    write_csv(tables_dir / "table_metr_la_v4_time_rdr.csv", [row for row in rdr_rows if row["role"] == "main"])
    write_csv(tables_dir / "table_metr_la_v4_time_ablation.csv", ablation_rows)
    write_csv(tables_dir / "table_metr_la_v4_time_horizon.csv", horizon_rows)
    write_csv(tables_dir / "table_metr_la_v4_time_complexity.csv", complexity_rows)

    summary = summarize_gate(metrics_rows, rdr_rows, ablation_rows, failed_rows)
    (out_dir / "formal_v4_time_ablation_summary.md").write_text(summary, encoding="utf-8")
    return {"status": "completed", "output_dir": str(out_dir), "metrics_rows": len(metrics_rows), "failed_or_skipped": len(failed_rows)}


def summarize_gate(
    metrics_rows: list[dict[str, Any]],
    rdr_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
) -> str:
    medium_high = [row for row in rdr_rows if row["role"] == "main" and row["fault"] != "clean" and row["severity_group"] in {"medium", "high"}]
    faults = sorted({row["fault"] for row in medium_high})
    sraf_better = 0
    for fault in faults:
        has_pair = all(
            any(row["model"] == model and row["fault"] == fault for row in rdr_rows)
            for model in ("SRAF-time-full", "ResidualGRU-time-corruption-aware")
        )
        if has_pair and row_value(rdr_rows, "SRAF-time-full", fault, "rdr_mae") < row_value(
            rdr_rows, "ResidualGRU-time-corruption-aware", fault, "rdr_mae"
        ):
            sraf_better += 1
    ablation_positive = sum(1 for row in ablation_rows if row["fault"] != "clean" and row["full_better_rdr"])
    clean = [row for row in metrics_rows if row["fault"] == "clean"]
    lines = [
        "# Formal-v4 Time Ablation Summary",
        "",
        "Clean MAE:",
        *[f"- `{row['model']}`: {float(row['mae']):.6f}" for row in sorted(clean, key=lambda r: float(r["mae"]))],
        "",
        f"SRAF-time-full lower RDR than ResidualGRU-time-corruption-aware on {sraf_better}/{len(faults)} medium/high faults.",
        f"Ablation rows where full SRAF has lower faulty-fault RDR than an ablation: {ablation_positive}.",
        f"Failed/skipped rows: {len(failed_rows)}.",
        "",
        "No final manuscript conclusions are written by this gate.",
    ]
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
