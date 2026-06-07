"""Run METR-LA strong-baseline and horizon audit gate."""

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
from src.models.baselines import persistence_predict  # noqa: E402
from src.models.residual_models import ResidualGRU, SRAFResidualGRU  # noqa: E402


FAULT_SETTINGS = [
    {"fault": "clean", "label": "clean", "severity_group": "clean"},
    {"fault": "random_missing", "rate": 0.40, "label": "random_missing_40", "severity_group": "high"},
    {"fault": "continuous_outage", "length": 24, "label": "continuous_outage_24", "severity_group": "high"},
    {"fault": "gaussian_noise", "severity": "high", "label": "gaussian_noise_high", "severity_group": "high"},
    {"fault": "linear_drift", "severity": "high", "label": "linear_drift_high", "severity_group": "high"},
    {"fault": "stuck_at_last_value", "severity": "high", "label": "stuck_at_last_value_high", "severity_group": "high"},
]

MODEL_SPECS = [
    {"model": "Persistence", "kind": "persistence", "source": "evaluated_in_this_gate"},
    {
        "model": "ResidualGRU-time-corruption-aware-current",
        "kind": "residual",
        "hidden_dim": 32,
        "checkpoint": "experiments/metr-la-time-feature-parity/models/ResidualGRU-time-corruption-aware/best_checkpoint.pt",
        "source": "reused_time_feature_parity_checkpoint",
    },
    {
        "model": "ResidualGRU-time-corruption-aware-strong",
        "kind": "residual",
        "hidden_dim": 32,
        "train_protocol": "corruption_aware_strong",
        "source": "trained_in_this_gate",
    },
    {
        "model": "SRAF-time-current",
        "kind": "sraf",
        "hidden_dim": 32,
        "checkpoint": "experiments/metr-la-formal-v36-final-polish/configs/time_of_day_features_h32/best_checkpoint.pt",
        "source": "reused_v36_time_of_day_features_h32_checkpoint",
    },
    {
        "model": "Repair-Persistence",
        "kind": "repair_persistence",
        "hidden_dim": 32,
        "checkpoint": "experiments/metr-la-formal-v36-final-polish/configs/time_of_day_features_h32/best_checkpoint.pt",
        "source": "uses_sraf_time_current_repair_then_repeats_repaired_last_observation",
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/metr-la")
    parser.add_argument("--output-dir", default="experiments/metr-la-strong-baseline-audit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--strong-hidden-dim", type=int, default=32)
    parser.add_argument("--sensor-embedding-dim", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--smoke", action="store_true")
    return parser


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if path.name == "failed_or_skipped_runs.csv":
            path.write_text("model,status,reason\n", encoding="utf-8")
        else:
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


def make_residual(hidden_dim: int, sensors: int, horizon: int, sensor_embedding_dim: int) -> ResidualGRU:
    return ResidualGRU(
        sensors=sensors,
        features=3,
        output_features=1,
        horizon=horizon,
        hidden_dim=hidden_dim,
        sensor_embedding_dim=sensor_embedding_dim,
    )


def make_sraf(hidden_dim: int, sensors: int, horizon: int, sensor_embedding_dim: int) -> SRAFResidualGRU:
    return SRAFResidualGRU(
        sensors=sensors,
        features=3,
        output_features=1,
        horizon=horizon,
        hidden_dim=hidden_dim,
        sensor_embedding_dim=sensor_embedding_dim,
    )


def model_param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def evaluate_loss(model: nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int, adjacency: torch.Tensor) -> float:
    model.eval()
    loss_fn = nn.MSELoss()
    losses: list[float] = []
    with torch.no_grad():
        for xb, yb in iter_batches(x, y, batch_size, shuffle=False, seed=0, epoch=0):
            pred = model(torch.from_numpy(xb.astype(np.float32)), adjacency=adjacency)
            losses.append(float(loss_fn(pred, torch.from_numpy(yb.astype(np.float32))).detach().cpu()))
    return float(np.mean(losses))


def train_strong_residual(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    args: argparse.Namespace,
    run_dir: Path,
    adjacency: torch.Tensor,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
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
            xb = train_corruption_batch(xb, args.seed, batch_step)
            pred = model(torch.from_numpy(xb.astype(np.float32)), adjacency=adjacency)
            loss = loss_fn(pred, torch.from_numpy(yb.astype(np.float32)))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            batch_step += 1
        train_loss = float(np.mean(losses))
        val_loss = evaluate_loss(model, val_x, val_y, args.batch_size, adjacency)
        scheduler.step(val_loss)
        lr = float(optimizer.param_groups[0]["lr"])
        improved = val_loss < best_val - 1.0e-6
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        row = {
            "run_id": "strong-baseline-audit",
            "model": "ResidualGRU-time-corruption-aware-strong",
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val,
            "learning_rate": lr,
            "improved": improved,
            "early_stop_triggered": False,
            "has_nan_or_inf": (not math.isfinite(train_loss)) or (not math.isfinite(val_loss)),
        }
        rows.append(row)
        print(f"ResidualGRU-time-corruption-aware-strong epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={lr:.6g}", flush=True)
        if no_improve >= args.patience:
            rows[-1]["early_stop_triggered"] = True
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, run_dir / "best_checkpoint.pt")
    (run_dir / "train_log.txt").write_text(
        "\n".join(
            f"epoch={r['epoch']},train_loss={r['train_loss']:.6f},val_loss={r['val_loss']:.6f},"
            f"best_val_loss={r['best_val_loss']:.6f},learning_rate={r['learning_rate']:.8f},"
            f"early_stop_triggered={r['early_stop_triggered']}"
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


def predict_persistence(x: np.ndarray, horizon: int) -> tuple[np.ndarray, float]:
    start = perf_counter()
    pred = persistence_predict(np.nan_to_num(x[..., :1], nan=0.0).astype(np.float32), horizon)
    return pred, perf_counter() - start


def predict_repair_persistence(model: SRAFResidualGRU, x: np.ndarray, horizon: int, batch_size: int, adjacency: torch.Tensor) -> tuple[np.ndarray, float]:
    model.eval()
    preds = []
    start = perf_counter()
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[i : i + batch_size].astype(np.float32))
            repaired, _, _ = model.repair_input(xb, adjacency=adjacency)
            last = repaired[:, -1:, :, :1]
            preds.append(last.repeat(1, horizon, 1, 1).cpu().numpy())
    return np.concatenate(preds, axis=0), perf_counter() - start


def metric_rows_to_rdr(metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = {row["model"]: float(row["mae"]) for row in metrics_rows if row["fault"] == "clean"}
    rows = []
    for row in metrics_rows:
        clean_mae = clean.get(row["model"], math.nan)
        fault_mae = float(row["mae"])
        rows.append(
            {
                "dataset": "METR-LA",
                "run_id": "strong-baseline-audit",
                "model": row["model"],
                "fault": row["fault"],
                "fault_type": row["fault_type"],
                "severity_group": row["severity_group"],
                "clean_mae": clean_mae,
                "fault_mae": fault_mae,
                "rdr_mae": (fault_mae - clean_mae) / clean_mae if math.isfinite(clean_mae) and clean_mae != 0 else math.nan,
            }
        )
    return rows


def row_value(rows: list[dict[str, Any]], model: str, fault: str, key: str) -> float:
    return float(next(row[key] for row in rows if row["model"] == model and row["fault"] == fault))


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
    return "high"


def build_persistence_comparison(metrics_rows: list[dict[str, Any]], rdr_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model in [
        "ResidualGRU-time-corruption-aware-current",
        "ResidualGRU-time-corruption-aware-strong",
        "SRAF-time-current",
        "Repair-Persistence",
    ]:
        for setting in FAULT_SETTINGS:
            fault = setting["label"]
            if not any(row["model"] == model and row["fault"] == fault for row in metrics_rows):
                continue
            model_mae = row_value(metrics_rows, model, fault, "mae")
            persistence_mae = row_value(metrics_rows, "Persistence", fault, "mae")
            model_rdr = row_value(rdr_rows, model, fault, "rdr_mae")
            persistence_rdr = row_value(rdr_rows, "Persistence", fault, "rdr_mae")
            rows.append(
                {
                    "model": model,
                    "fault": fault,
                    "model_mae": model_mae,
                    "persistence_mae": persistence_mae,
                    "delta_mae_model_minus_persistence": model_mae - persistence_mae,
                    "model_beats_persistence_mae": model_mae < persistence_mae,
                    "model_rdr_mae": model_rdr,
                    "persistence_rdr_mae": persistence_rdr,
                    "model_lower_rdr_than_persistence": model_rdr < persistence_rdr,
                }
            )
    return rows


def load_reused_meta(model_name: str) -> dict[str, Any]:
    if model_name == "ResidualGRU-time-corruption-aware-current":
        path = ROOT / "experiments/metr-la-time-feature-parity/complexity_metrics.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("model") == "ResidualGRU-time-corruption-aware":
                        return {
                            "training_time_sec": float(row.get("training_time_sec", 0.0)),
                            "best_epoch": float(row.get("best_epoch", 0.0)),
                            "best_val_loss": float(row.get("best_val_loss", 0.0)),
                        }
    if model_name in {"SRAF-time-current", "Repair-Persistence"}:
        path = ROOT / "experiments/metr-la-formal-v36-final-polish/complexity_metrics.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("config") == "time_of_day_features_h32":
                        return {
                            "training_time_sec": float(row.get("training_time_sec", 0.0)),
                            "best_epoch": float(row.get("best_epoch", 0.0)),
                            "best_val_loss": float(row.get("best_val_loss", 0.0)),
                        }
    return {"training_time_sec": 0.0, "best_epoch": 0.0, "best_val_loss": 0.0}


def write_summaries(
    out_dir: Path,
    metrics_rows: list[dict[str, Any]],
    horizon_rows: list[dict[str, Any]],
    rdr_rows: list[dict[str, Any]],
    persistence_rows: list[dict[str, Any]],
) -> None:
    clean = sorted([row for row in metrics_rows if row["fault"] == "clean"], key=lambda row: float(row["mae"]))
    high_faults = [setting["label"] for setting in FAULT_SETTINGS if setting["label"] != "clean"]
    def has_pair(a: str, b: str, fault: str) -> bool:
        return all(any(row["model"] == model and row["fault"] == fault for row in rdr_rows) for model in (a, b))

    strong_faults = [fault for fault in high_faults if has_pair("SRAF-time-current", "ResidualGRU-time-corruption-aware-strong", fault)]
    current_faults = [fault for fault in high_faults if has_pair("SRAF-time-current", "ResidualGRU-time-corruption-aware-current", fault)]
    sraf_vs_strong_better = sum(
        row_value(rdr_rows, "SRAF-time-current", fault, "rdr_mae")
        < row_value(rdr_rows, "ResidualGRU-time-corruption-aware-strong", fault, "rdr_mae")
        for fault in strong_faults
    )
    sraf_vs_current_better = sum(
        row_value(rdr_rows, "SRAF-time-current", fault, "rdr_mae")
        < row_value(rdr_rows, "ResidualGRU-time-corruption-aware-current", fault, "rdr_mae")
        for fault in current_faults
    )
    sraf_beats_persistence = [
        row["fault"]
        for row in persistence_rows
        if row["model"] == "SRAF-time-current" and row["model_beats_persistence_mae"] == True
    ]
    summary = [
        "# Strong Baseline Summary",
        "",
        "Clean MAE:",
        *[f"- `{row['model']}`: {float(row['mae']):.6f}" for row in clean],
        "",
        f"SRAF-time-current lower RDR than current ResidualGRU-time on {sraf_vs_current_better}/{len(current_faults)} high-severity faults.",
        f"SRAF-time-current lower RDR than strong ResidualGRU-time on {sraf_vs_strong_better}/{len(strong_faults)} high-severity faults.",
        f"SRAF-time-current beats Persistence by MAE on: {', '.join(sraf_beats_persistence) if sraf_beats_persistence else 'none'}.",
        "",
        "No paper conclusions are written by this gate.",
    ]
    (out_dir / "strong_baseline_summary.md").write_text("\n".join(summary), encoding="utf-8")

    horizon_lines = ["# Horizon Audit Summary", ""]
    for fault in ["clean", *high_faults]:
        horizon_lines.append(f"## {fault}")
        for model in ["Persistence", "ResidualGRU-time-corruption-aware-strong", "SRAF-time-current", "Repair-Persistence"]:
            row = next((r for r in horizon_rows if r["model"] == model and r["fault"] == fault), None)
            if row:
                horizon_lines.append(
                    f"- `{model}`: h3={float(row['mae_15min_h3']):.6f}, h6={float(row['mae_30min_h6']):.6f}, h12={float(row['mae_60min_h12']):.6f}"
                )
        horizon_lines.append("")
    (out_dir / "horizon_audit_summary.md").write_text("\n".join(horizon_lines), encoding="utf-8")

    repair_lines = ["# Repair-Persistence Summary", ""]
    for fault in ["clean", *high_faults]:
        if not any(row["model"] == "Repair-Persistence" and row["fault"] == fault for row in metrics_rows):
            repair_lines.append(f"- `{fault}`: TODO - Repair-Persistence not evaluated in this run.")
            continue
        repair_mae = row_value(metrics_rows, "Repair-Persistence", fault, "mae")
        persistence_mae = row_value(metrics_rows, "Persistence", fault, "mae")
        sraf_mae = row_value(metrics_rows, "SRAF-time-current", fault, "mae") if any(
            row["model"] == "SRAF-time-current" and row["fault"] == fault for row in metrics_rows
        ) else math.nan
        repair_lines.append(
            f"- `{fault}`: Repair-Persistence MAE {repair_mae:.6f}; Persistence delta {repair_mae - persistence_mae:.6f}; SRAF delta {repair_mae - sraf_mae:.6f}."
        )
    repair_lines.append("")
    repair_lines.append("Repair-Persistence uses the trained SRAF-time repair/gating path and repeats the repaired last speed observation over all horizons.")
    (out_dir / "repair_persistence_summary.md").write_text("\n".join(repair_lines), encoding="utf-8")


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
        args.epochs = min(args.epochs, 1)
        args.patience = 1
    mean, std = load_scale(data_dir)
    adjacency = torch.from_numpy(np.load(data_dir / "adjacency.npy").astype(np.float32))
    train_x = add_time_of_day_features(train_x_base, 0)
    val_x = add_time_of_day_features(val_x_base, train_x_base.shape[0])
    split_start = train_x_base.shape[0] + val_x_base.shape[0]

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

    manifest = {
        "run_id": "metr-la-strong-baseline-audit",
        "gate": "STRONG_BASELINE_AND_HORIZON_AUDIT_GATE",
        "created_at": "2026-05-18",
        "seed": args.seed,
        "data_dir": str(data_dir),
        "output_dir": str(out_dir),
        "dataset": {
            "name": "METR-LA",
            "L": 12,
            "H": 12,
            "N": int(train_x_base.shape[2]),
            "F_target": 1,
            "train_samples_used": int(train_x_base.shape[0]),
            "val_samples_used": int(val_x_base.shape[0]),
            "test_samples_used": int(test_x_base.shape[0]),
        },
        "time_feature_construction": "For input sample window index s and input step l, append sin(2*pi*(s+l+split_start)/288) and cos(...). Only input-window indices are used.",
        "target_leakage_check": "Target Y is never corrupted and no target-horizon time features are appended.",
        "strong_residual_config": {
            "hidden_dim": args.strong_hidden_dim,
            "max_epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "reduce_lr_on_plateau": True,
            "gradient_clipping": args.grad_clip,
            "weight_decay": args.weight_decay,
            "loss": "MSE",
        },
        "models": MODEL_SPECS,
        "fault_settings": FAULT_SETTINGS,
        "integrity_note": "No PEMS-BAY, no manuscript changes, and no final paper conclusions.",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    metrics_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []

    specs = MODEL_SPECS if not args.smoke else MODEL_SPECS[:3]
    if args.smoke:
        failed_rows.extend({"model": spec["model"], "status": "skipped", "reason": "smoke mode"} for spec in MODEL_SPECS[3:])

    for spec in specs:
        model_name = spec["model"]
        run_dir = out_dir / "models" / model_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config_resolved.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        try:
            param_count: int | float = 0
            train_extra: dict[str, Any] = {"training_time_sec": 0.0, "best_epoch": 0.0, "best_val_loss": 0.0}
            predictor_kind = spec["kind"]
            model: nn.Module | None = None
            if predictor_kind == "residual":
                hidden_dim = args.strong_hidden_dim if model_name.endswith("-strong") else int(spec["hidden_dim"])
                model = make_residual(hidden_dim, train_x_base.shape[2], train_y.shape[1], args.sensor_embedding_dim)
                checkpoint = run_dir / "best_checkpoint.pt"
                source = ROOT / spec["checkpoint"] if "checkpoint" in spec else None
                if checkpoint.exists():
                    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
                    train_extra = load_reused_meta(model_name)
                elif source is not None and source.exists():
                    model.load_state_dict(torch.load(source, map_location="cpu"))
                    torch.save(model.state_dict(), checkpoint)
                    train_extra = load_reused_meta(model_name)
                else:
                    train_extra, curves = train_strong_residual(model, train_x, train_y, val_x, val_y, args, run_dir, adjacency)
                    curve_rows.extend(curves)
                param_count = model_param_count(model)
            elif predictor_kind in {"sraf", "repair_persistence"}:
                model = make_sraf(int(spec["hidden_dim"]), train_x_base.shape[2], train_y.shape[1], args.sensor_embedding_dim)
                source = ROOT / spec["checkpoint"]
                if not source.exists():
                    raise FileNotFoundError(str(source))
                model.load_state_dict(torch.load(source, map_location="cpu"))
                torch.save(model.state_dict(), run_dir / "best_checkpoint.pt")
                train_extra = load_reused_meta(model_name)
                param_count = model_param_count(model) if predictor_kind == "sraf" else 0

            for setting in FAULT_SETTINGS:
                label = setting["label"]
                if predictor_kind == "persistence":
                    pred, inference_time = predict_persistence(fault_inputs[label], test_y.shape[1])
                elif predictor_kind == "repair_persistence":
                    assert isinstance(model, SRAFResidualGRU)
                    pred, inference_time = predict_repair_persistence(model, fault_inputs[label], test_y.shape[1], args.batch_size, adjacency)
                else:
                    assert model is not None
                    pred, inference_time = predict_model(model, fault_inputs[label], args.batch_size, adjacency)
                m = safe_metrics(test_y, pred, mean, std)
                if label == "clean":
                    np.savez_compressed(run_dir / "clean_predictions.npz", y_pred=pred, y_true=test_y)
                row = {
                    "dataset": "METR-LA",
                    "run_id": "strong-baseline-audit",
                    "metrics_scale": "original",
                    "model": model_name,
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
                    "source": spec["source"],
                }
                metrics_rows.append(row)
                horizon_rows.append(
                    {
                        "dataset": "METR-LA",
                        "run_id": "strong-baseline-audit",
                        "model": model_name,
                        "fault": label,
                        "fault_type": row["fault_type"],
                        "mae_15min_h3": row["mae_h3"],
                        "mae_30min_h6": row["mae_h6"],
                        "mae_60min_h12": row["mae_h12"],
                    }
                )
            clean_row = next(row for row in metrics_rows if row["model"] == model_name and row["fault"] == "clean")
            complexity_rows.append(
                {
                    "dataset": "METR-LA",
                    "run_id": "strong-baseline-audit",
                    "model": model_name,
                    "parameter_count": param_count,
                    "training_time_sec": train_extra["training_time_sec"],
                    "clean_inference_time_sec": clean_row["inference_time_sec"],
                    "best_epoch": train_extra["best_epoch"],
                    "best_val_loss": train_extra["best_val_loss"],
                    "source": spec["source"],
                }
            )
        except Exception as exc:
            failed_rows.append({"model": model_name, "status": "failed", "reason": repr(exc)})
            print(f"FAILED {model_name}: {exc!r}", flush=True)

    rdr_rows = metric_rows_to_rdr(metrics_rows)
    persistence_rows = build_persistence_comparison(metrics_rows, rdr_rows)
    write_csv(out_dir / "metrics_by_model_fault.csv", metrics_rows)
    write_csv(out_dir / "horizon_metrics.csv", horizon_rows)
    write_csv(out_dir / "robustness_rdr.csv", rdr_rows)
    write_csv(out_dir / "persistence_comparison.csv", persistence_rows)
    write_csv(out_dir / "complexity_metrics.csv", complexity_rows)
    write_csv(out_dir / "training_curves.csv", curve_rows)
    write_csv(out_dir / "failed_or_skipped_runs.csv", failed_rows)
    write_summaries(out_dir, metrics_rows, horizon_rows, rdr_rows, persistence_rows)
    return {"status": "completed", "output_dir": str(out_dir), "metrics_rows": len(metrics_rows), "failed_or_skipped": len(failed_rows)}


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
