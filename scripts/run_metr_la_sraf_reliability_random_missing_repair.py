"""Run METR-LA SRAF reliability random-missing repair gate."""

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
from src.models.residual_models import ResidualGRU, SRAFResidualGRU  # noqa: E402


FAULT_SETTINGS = [
    {"fault": "clean", "label": "clean", "severity_group": "clean"},
    {"fault": "random_missing", "rate": 0.20, "label": "random_missing_20", "severity_group": "medium"},
    {"fault": "random_missing", "rate": 0.40, "label": "random_missing_40", "severity_group": "high"},
    {"fault": "continuous_outage", "length": 24, "label": "continuous_outage_24", "severity_group": "high"},
    {"fault": "gaussian_noise", "severity": "high", "label": "gaussian_noise_high", "severity_group": "high"},
    {"fault": "linear_drift", "severity": "high", "label": "linear_drift_high", "severity_group": "high"},
    {"fault": "stuck_at_last_value", "severity": "high", "label": "stuck_at_last_value_high", "severity_group": "high"},
]

DIAGNOSTIC_FAULTS = {"random_missing_40", "gaussian_noise_high", "stuck_at_last_value_high"}

CANDIDATES = [
    {
        "candidate": "current_sraf_time",
        "priority": 1,
        "mode": "load",
        "checkpoint": "experiments/metr-la-formal-v36-final-polish/configs/time_of_day_features_h32/best_checkpoint.pt",
        "description": "Traceable current SRAF-time checkpoint.",
        "model_kwargs": {},
        "lambda_repair": 0.0,
    },
    {
        "candidate": "hard_missing_gate",
        "priority": 2,
        "mode": "train",
        "description": "Force missing observations to rely on repair: r_final = mask * r_learned.",
        "model_kwargs": {"hard_missing_gate": True},
        "lambda_repair": 0.0,
    },
    {
        "candidate": "repair_consistency_loss",
        "priority": 3,
        "mode": "train",
        "description": "Add repair consistency loss on corrupted input positions where clean input is known.",
        "model_kwargs": {},
        "lambda_repair": 0.05,
    },
    {
        "candidate": "fault_type_reliability_features",
        "priority": 4,
        "mode": "skip",
        "description": "Lightweight missing/jump/deviation/stuck/spatial-disagreement reliability indicators.",
        "model_kwargs": {"enhanced_reliability_features": True},
        "lambda_repair": 0.0,
        "skip_reason": "Runtime priority limits this gate to A/B/E first; C is implemented but not run.",
    },
    {
        "candidate": "adaptive_repair_blending",
        "priority": 5,
        "mode": "skip",
        "description": "Learn bounded temporal/spatial alpha from reliability indicators.",
        "model_kwargs": {"enhanced_reliability_features": True, "adaptive_repair_blending": True},
        "lambda_repair": 0.0,
        "skip_reason": "Runtime priority limits this gate to A/B/E first; D is implemented but not run.",
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/metr-la")
    parser.add_argument("--output-dir", default="experiments/metr-la-sraf-reliability-random-missing-repair")
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
        if path.name == "failed_or_skipped_candidates.csv":
            path.write_text("candidate,status,reason\n", encoding="utf-8")
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


def apply_fault_speed(x_speed: np.ndarray, setting: dict[str, Any], seed: int, train_std: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if setting["fault"] == "clean":
        return x_speed.copy(), np.zeros_like(x_speed, dtype=bool), {"fault": "clean", "seed": seed, "target_corrupted": False}
    if setting["fault"] == "random_missing":
        return random_missing(x_speed, rate=setting["rate"], seed=seed)
    if setting["fault"] == "continuous_outage":
        return continuous_outage(x_speed, length=setting["length"], seed=seed)
    if setting["fault"] == "gaussian_noise":
        return gaussian_noise(x_speed, severity=setting["severity"], train_std=train_std, seed=seed)
    if setting["fault"] == "linear_drift":
        return linear_drift(x_speed, severity=setting["severity"], train_std=train_std, seed=seed)
    if setting["fault"] == "stuck_at_last_value":
        return stuck_at_last_value(x_speed, severity=setting["severity"], seed=seed)
    raise ValueError(f"Unknown fault setting: {setting}")


def apply_fault_with_time(x_time: np.ndarray, setting: dict[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    corrupted_speed, speed_mask, meta = apply_fault_speed(x_time[..., :1], setting, seed=seed, train_std=1.0)
    return np.concatenate([corrupted_speed, x_time[..., 1:]], axis=-1).astype(np.float32), speed_mask, meta


def iter_batches(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int, epoch: int) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(x.shape[0])
    if shuffle:
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(indices)
    return [(x[idx], y[idx]) for idx in np.array_split(indices, math.ceil(len(indices) / batch_size))]


def train_corruption_batch(x_time: np.ndarray, seed: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    choices = [setting for setting in FAULT_SETTINGS if setting["fault"] != "clean"]
    setting = choices[(seed + step) % len(choices)]
    corrupted, mask, _ = apply_fault_with_time(x_time, setting, seed=seed + step)
    return corrupted, mask.astype(np.float32)


def make_sraf(spec: dict[str, Any], sensors: int, horizon: int, args: argparse.Namespace) -> SRAFResidualGRU:
    return SRAFResidualGRU(
        sensors=sensors,
        features=3,
        output_features=1,
        horizon=horizon,
        hidden_dim=args.hidden_dim,
        sensor_embedding_dim=args.sensor_embedding_dim,
        **spec.get("model_kwargs", {}),
    )


def make_residual(sensors: int, horizon: int, args: argparse.Namespace) -> ResidualGRU:
    return ResidualGRU(
        sensors=sensors,
        features=3,
        output_features=1,
        horizon=horizon,
        hidden_dim=args.hidden_dim,
        sensor_embedding_dim=args.sensor_embedding_dim,
    )


def model_param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def training_meta_from_log(run_dir: Path) -> dict[str, Any]:
    log_path = run_dir / "train_log.txt"
    best_epoch: int | str = "TODO"
    best_val = math.inf
    if log_path.exists():
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
        "training_time_sec": "TODO_unavailable_after_checkpoint_regeneration",
        "best_epoch": best_epoch,
        "best_val_loss": best_val if math.isfinite(best_val) else "TODO",
    }


def evaluate_loss(model: nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int, adjacency: torch.Tensor) -> float:
    model.eval()
    loss_fn = nn.MSELoss()
    losses: list[float] = []
    with torch.no_grad():
        for xb, yb in iter_batches(x, y, batch_size, shuffle=False, seed=0, epoch=0):
            pred = model(torch.from_numpy(xb.astype(np.float32)), adjacency=adjacency)
            losses.append(float(loss_fn(pred, torch.from_numpy(yb.astype(np.float32))).detach().cpu()))
    return float(np.mean(losses))


def repair_loss_from_components(components: dict[str, torch.Tensor], clean_x: torch.Tensor, corrupt_mask: torch.Tensor) -> torch.Tensor:
    mask = corrupt_mask.to(clean_x.device, dtype=clean_x.dtype)
    denom = mask.sum().clamp_min(1.0)
    repaired_speed = components["repaired_input"][..., :1]
    clean_speed = clean_x[..., :1]
    return torch.sum(torch.abs(repaired_speed - clean_speed) * mask) / denom


def train_candidate(
    spec: dict[str, Any],
    model: SRAFResidualGRU,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    args: argparse.Namespace,
    run_dir: Path,
    adjacency: torch.Tensor,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    best_val = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0
    batch_step = 0
    rows: list[dict[str, Any]] = []
    start = perf_counter()
    lambda_repair = float(spec.get("lambda_repair", 0.0))
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        forecast_losses: list[float] = []
        repair_losses: list[float] = []
        for xb_clean, yb in iter_batches(train_x, train_y, args.batch_size, shuffle=True, seed=args.seed, epoch=epoch):
            xb_corrupt, corrupt_mask = train_corruption_batch(xb_clean, args.seed, batch_step)
            x_t = torch.from_numpy(xb_corrupt.astype(np.float32))
            y_t = torch.from_numpy(yb.astype(np.float32))
            pred, components = model(x_t, adjacency=adjacency, return_components=True)
            forecast_loss = loss_fn(pred, y_t)
            repair_loss = torch.tensor(0.0)
            if lambda_repair > 0:
                repair_loss = repair_loss_from_components(
                    components,
                    torch.from_numpy(xb_clean.astype(np.float32)),
                    torch.from_numpy(corrupt_mask.astype(np.float32)),
                )
            loss = forecast_loss + lambda_repair * repair_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            forecast_losses.append(float(forecast_loss.detach().cpu()))
            repair_losses.append(float(repair_loss.detach().cpu()))
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
            "run_id": "sraf-reliability-random-missing-repair",
            "candidate": spec["candidate"],
            "epoch": epoch,
            "train_loss": train_loss,
            "forecast_loss": float(np.mean(forecast_losses)),
            "repair_loss": float(np.mean(repair_losses)),
            "val_loss": val_loss,
            "best_val_loss": best_val,
            "lambda_repair": lambda_repair,
            "improved": improved,
            "early_stop_triggered": False,
            "has_nan_or_inf": (not math.isfinite(train_loss)) or (not math.isfinite(val_loss)),
        }
        rows.append(row)
        print(f"{spec['candidate']} epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
        if no_improve >= args.patience:
            rows[-1]["early_stop_triggered"] = True
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, run_dir / "best_checkpoint.pt")
    (run_dir / "train_log.txt").write_text(
        "\n".join(
            f"epoch={r['epoch']},train_loss={r['train_loss']:.6f},forecast_loss={r['forecast_loss']:.6f},"
            f"repair_loss={r['repair_loss']:.6f},val_loss={r['val_loss']:.6f},best_val_loss={r['best_val_loss']:.6f},"
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


def component_arrays(model: SRAFResidualGRU, x: np.ndarray, batch_size: int, adjacency: torch.Tensor) -> dict[str, np.ndarray]:
    model.eval()
    parts: dict[str, list[np.ndarray]] = {
        "reliability": [],
        "repaired_input": [],
        "temporal_repair": [],
        "spatial_repair": [],
        "repair_blend": [],
    }
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[i : i + batch_size].astype(np.float32))
            comps = model.repair_components(xb, adjacency=adjacency)
            for key in parts:
                parts[key].append(comps[key].cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in parts.items()}


def masked_stats(values: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    if not np.any(mask):
        return math.nan, math.nan
    selected = values[mask]
    return float(np.nanmean(selected)), float(np.nanstd(selected))


def mae_on_mask(clean: np.ndarray, repaired: np.ndarray, mask: np.ndarray, mean: float, std: float) -> float:
    if not np.any(mask):
        return math.nan
    clean_orig = inverse_scale(clean[..., :1], mean, std)
    repaired_orig = inverse_scale(repaired[..., :1], mean, std)
    return float(np.nanmean(np.abs(clean_orig[mask] - repaired_orig[mask])))


def diagnostics_for_candidate(
    candidate: str,
    model: SRAFResidualGRU,
    clean_x_time: np.ndarray,
    fault_inputs: dict[str, np.ndarray],
    fault_masks: dict[str, np.ndarray],
    mean: float,
    std: float,
    batch_size: int,
    adjacency: torch.Tensor,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reliability_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    for fault in sorted(DIAGNOSTIC_FAULTS):
        comps = component_arrays(model, fault_inputs[fault], batch_size, adjacency)
        corrupt_mask = fault_masks[fault].astype(bool)
        clean_mask = ~corrupt_mask
        rel = comps["reliability"][..., :1]
        clean_mean, clean_std = masked_stats(rel, clean_mask)
        corrupt_mean, corrupt_std = masked_stats(rel, corrupt_mask)
        reliability_rows.append(
            {
                "candidate": candidate,
                "fault": fault,
                "clean_position_reliability_mean": clean_mean,
                "clean_position_reliability_std": clean_std,
                "corrupted_position_reliability_mean": corrupt_mean,
                "corrupted_position_reliability_std": corrupt_std,
                "corrupted_lower_than_clean": corrupt_mean < clean_mean if math.isfinite(corrupt_mean) and math.isfinite(clean_mean) else "TODO",
            }
        )
        repair_rows.append(
            {
                "candidate": candidate,
                "fault": fault,
                "corrupted_positions": int(np.sum(corrupt_mask)),
                "temporal_repair_mae": mae_on_mask(clean_x_time, comps["temporal_repair"], corrupt_mask, mean, std),
                "spatial_repair_mae": mae_on_mask(clean_x_time, comps["spatial_repair"], corrupt_mask, mean, std),
                "blend_repair_mae": mae_on_mask(clean_x_time, comps["repair_blend"], corrupt_mask, mean, std),
                "final_repair_mae": mae_on_mask(clean_x_time, comps["repaired_input"], corrupt_mask, mean, std),
            }
        )
    return reliability_rows, repair_rows


def build_rdr(metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = {row["candidate"]: float(row["mae"]) for row in metrics_rows if row["fault"] == "clean"}
    rows = []
    for row in metrics_rows:
        clean_mae = clean.get(row["candidate"], math.nan)
        fault_mae = float(row["mae"])
        rows.append(
            {
                "dataset": "METR-LA",
                "run_id": "sraf-reliability-random-missing-repair",
                "candidate": row["candidate"],
                "fault": row["fault"],
                "fault_type": row["fault_type"],
                "severity_group": row["severity_group"],
                "clean_mae": clean_mae,
                "fault_mae": fault_mae,
                "rdr_mae": (fault_mae - clean_mae) / clean_mae if math.isfinite(clean_mae) and clean_mae != 0 else math.nan,
            }
        )
    return rows


def row_value(rows: list[dict[str, Any]], candidate: str, fault: str, key: str) -> float:
    return float(next(row[key] for row in rows if row["candidate"] == candidate and row["fault"] == fault))


def load_reference_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = ROOT / "experiments/metr-la-strong-baseline-audit/metrics_by_model_fault.csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["model"] in {"ResidualGRU-time-corruption-aware-strong", "SRAF-time-current"}:
                    rows.append(row)
    return rows


def write_summaries(
    out_dir: Path,
    metrics_rows: list[dict[str, Any]],
    rdr_rows: list[dict[str, Any]],
    reliability_rows: list[dict[str, Any]],
    repair_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
) -> None:
    sraf_names = {spec["candidate"] for spec in CANDIDATES}
    completed = sorted({row["candidate"] for row in metrics_rows if row["candidate"] in sraf_names})
    current_rm40 = row_value(metrics_rows, "current_sraf_time", "random_missing_40", "mae") if "current_sraf_time" in completed else math.nan
    reference = load_reference_rows()
    strong_rm = {
        row["fault"]: float(row["mae"])
        for row in metrics_rows
        if row["candidate"] == "ResidualGRU-time-strong-reference" and row["fault"] in {"random_missing_20", "random_missing_40"}
    }
    if not strong_rm:
        strong_rm = {
            row["fault"]: float(row["mae"])
            for row in reference
            if row["model"] == "ResidualGRU-time-corruption-aware-strong" and row["fault"] in {"random_missing_20", "random_missing_40"}
        }
    summary_lines = [
        "# Random Missing Repair Summary",
        "",
        "Random missing MAE:",
    ]
    for candidate in completed:
        rm20 = row_value(metrics_rows, candidate, "random_missing_20", "mae")
        rm40 = row_value(metrics_rows, candidate, "random_missing_40", "mae")
        summary_lines.append(f"- `{candidate}`: random_missing_20={rm20:.6f}, random_missing_40={rm40:.6f}, delta_rm40_vs_current={rm40 - current_rm40:.6f}")
    if strong_rm:
        summary_lines.append("")
        summary_lines.append(
            f"Strong ResidualGRU reference: random_missing_20={strong_rm.get('random_missing_20', math.nan):.6f}, random_missing_40={strong_rm.get('random_missing_40', math.nan):.6f}."
        )
    summary_lines.append("")
    summary_lines.append("No paper conclusions are written by this gate.")
    (out_dir / "random_missing_repair_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    selection_lines = ["# Candidate Selection Summary", ""]
    clean_limit = 6.0
    eligible_candidates: list[tuple[float, float, str]] = []
    for candidate in completed:
        if candidate == "current_sraf_time":
            continue
        clean_mae = row_value(metrics_rows, candidate, "clean", "mae")
        rm20 = row_value(metrics_rows, candidate, "random_missing_20", "mae")
        rm40 = row_value(metrics_rows, candidate, "random_missing_40", "mae")
        current_rm20 = row_value(metrics_rows, "current_sraf_time", "random_missing_20", "mae")
        improves = rm20 < current_rm20 or rm40 < current_rm40
        preserves = all(
            row_value(metrics_rows, candidate, fault, "mae") <= row_value(metrics_rows, "current_sraf_time", fault, "mae") + 0.25
            for fault in ["continuous_outage_24", "gaussian_noise_high", "stuck_at_last_value_high"]
        )
        rel_rows = [row for row in reliability_rows if row["candidate"] == candidate]
        reliability_clear = any(row["corrupted_lower_than_clean"] is True for row in rel_rows)
        eligible = improves and clean_mae <= clean_limit and preserves and reliability_clear
        selection_lines.append(
            f"- `{candidate}`: clean={clean_mae:.6f}, improves_random_missing={improves}, preserves_key_faults={preserves}, reliability_clear={reliability_clear}, eligible={eligible}."
        )
        if eligible:
            eligible_candidates.append((rm40, clean_mae, candidate))
    selected = sorted(eligible_candidates)[0][2] if eligible_candidates else "none"
    selection_lines.append("")
    selection_lines.append(f"Selected candidate: `{selected}`.")
    selection_lines.append(f"Failed/skipped candidates: {len(failed_rows)}.")
    (out_dir / "candidate_selection_summary.md").write_text("\n".join(selection_lines), encoding="utf-8")


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
            return setting["severity_group"]
    return "unknown"


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
    test_x_clean = add_time_of_day_features(test_x_base, train_x_base.shape[0] + val_x_base.shape[0])

    fault_dir = out_dir / "fault_masks"
    fault_dir.mkdir(parents=True, exist_ok=True)
    fault_inputs: dict[str, np.ndarray] = {}
    fault_masks: dict[str, np.ndarray] = {}
    for idx, setting in enumerate(FAULT_SETTINGS):
        label = setting["label"]
        cx, mask, meta = apply_fault_with_time(test_x_clean, setting, seed=args.seed + idx)
        fault_inputs[label] = cx
        fault_masks[label] = mask.astype(bool)
        meta = {**setting, **meta, "label": label, "target_corrupted": False, "mask_path": str(fault_dir / f"{label}_mask.npz")}
        np.savez_compressed(fault_dir / f"{label}_mask.npz", mask=mask)
        (fault_dir / f"{label}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    manifest = {
        "run_id": "metr-la-sraf-reliability-random-missing-repair",
        "gate": "SRAF_RELIABILITY_RANDOM_MISSING_REPAIR_GATE",
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
        "training": {
            "hidden_dim": args.hidden_dim,
            "sensor_embedding_dim": args.sensor_embedding_dim,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "loss": "MSE plus optional repair consistency loss",
            "corrupt_speed_only": True,
        },
        "time_feature_construction": "Speed plus sin/cos time-of-day; faults corrupt only speed observations, not time features.",
        "target_leakage_check": "Target Y is never corrupted and no future target-horizon features are used.",
        "candidates": CANDIDATES,
        "fault_settings": FAULT_SETTINGS,
        "integrity_note": "No PEMS-BAY, no manuscript conclusions, no change to main claim.",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    metrics_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    rdr_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []

    candidates = CANDIDATES[:3] if args.smoke else CANDIDATES
    for spec in candidates:
        candidate = spec["candidate"]
        run_dir = out_dir / "candidates" / candidate
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config_resolved.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        if spec["mode"] == "skip":
            failed_rows.append({"candidate": candidate, "status": "skipped", "reason": spec.get("skip_reason", "skipped")})
            continue
        try:
            model = make_sraf(spec, train_x_base.shape[2], train_y.shape[1], args)
            train_extra: dict[str, Any] = {"training_time_sec": 0.0, "best_epoch": 0.0, "best_val_loss": 0.0}
            if spec["mode"] == "load":
                checkpoint = ROOT / spec["checkpoint"]
                if not checkpoint.exists():
                    raise FileNotFoundError(str(checkpoint))
                model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
                torch.save(model.state_dict(), run_dir / "best_checkpoint.pt")
                (run_dir / "train_log.txt").write_text(f"reused_checkpoint: {checkpoint}\n", encoding="utf-8")
            else:
                checkpoint = run_dir / "best_checkpoint.pt"
                if checkpoint.exists():
                    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
                    train_extra = training_meta_from_log(run_dir)
                else:
                    train_extra, curves = train_candidate(spec, model, train_x, train_y, val_x, val_y, args, run_dir, adjacency)
                    curve_rows.extend(curves)
            param_count = model_param_count(model)
            for setting in FAULT_SETTINGS:
                label = setting["label"]
                pred, inference_time = predict_model(model, fault_inputs[label], args.batch_size, adjacency)
                m = safe_metrics(test_y, pred, mean, std)
                if label == "clean":
                    np.savez_compressed(run_dir / "clean_predictions.npz", y_pred=pred, y_true=test_y)
                row = {
                    "dataset": "METR-LA",
                    "run_id": "sraf-reliability-random-missing-repair",
                    "candidate": candidate,
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
                    "lambda_repair": spec.get("lambda_repair", 0.0),
                    "description": spec["description"],
                }
                metrics_rows.append(row)
                horizon_rows.append(
                    {
                        "dataset": "METR-LA",
                        "run_id": "sraf-reliability-random-missing-repair",
                        "candidate": candidate,
                        "fault": label,
                        "mae_15min_h3": row["mae_h3"],
                        "mae_30min_h6": row["mae_h6"],
                        "mae_60min_h12": row["mae_h12"],
                    }
                )
            rel, rep = diagnostics_for_candidate(candidate, model, test_x_clean, fault_inputs, fault_masks, mean, std, args.batch_size, adjacency)
            reliability_rows.extend(rel)
            repair_rows.extend(rep)
            clean_row = next(row for row in metrics_rows if row["candidate"] == candidate and row["fault"] == "clean")
            complexity_rows.append(
                {
                    "dataset": "METR-LA",
                    "run_id": "sraf-reliability-random-missing-repair",
                    "candidate": candidate,
                    "parameter_count": param_count,
                    "training_time_sec": train_extra["training_time_sec"],
                    "clean_inference_time_sec": clean_row["inference_time_sec"],
                    "best_epoch": train_extra["best_epoch"],
                    "best_val_loss": train_extra["best_val_loss"],
                }
            )
        except Exception as exc:
            failed_rows.append({"candidate": candidate, "status": "failed", "reason": repr(exc)})
            print(f"FAILED {candidate}: {exc!r}", flush=True)

    reference_path = ROOT / "experiments/metr-la-strong-baseline-audit/models/ResidualGRU-time-corruption-aware-strong/best_checkpoint.pt"
    if reference_path.exists() and not args.smoke:
        reference_name = "ResidualGRU-time-strong-reference"
        reference_model = make_residual(train_x_base.shape[2], train_y.shape[1], args)
        reference_model.load_state_dict(torch.load(reference_path, map_location="cpu"))
        reference_param_count = model_param_count(reference_model)
        for setting in FAULT_SETTINGS:
            label = setting["label"]
            pred, inference_time = predict_model(reference_model, fault_inputs[label], args.batch_size, adjacency)
            m = safe_metrics(test_y, pred, mean, std)
            row = {
                "dataset": "METR-LA",
                "run_id": "sraf-reliability-random-missing-repair",
                "candidate": reference_name,
                "fault": label,
                "fault_type": fault_type(label),
                "severity_group": severity_group(label),
                "mae": m["mae"],
                "rmse": m["rmse"],
                "mape": m["mape"],
                "mae_h3": m.get("mae_h3", math.nan),
                "mae_h6": m.get("mae_h6", math.nan),
                "mae_h12": m.get("mae_h12", math.nan),
                "parameter_count": reference_param_count,
                "inference_time_sec": inference_time,
                "training_time_sec": "reused_strong_baseline_audit_checkpoint",
                "best_epoch": "see_strong_baseline_audit",
                "best_val_loss": "see_strong_baseline_audit",
                "lambda_repair": 0.0,
                "description": "Strong ResidualGRU-time reference evaluated on this gate's masks.",
            }
            metrics_rows.append(row)
            horizon_rows.append(
                {
                    "dataset": "METR-LA",
                    "run_id": "sraf-reliability-random-missing-repair",
                    "candidate": reference_name,
                    "fault": label,
                    "mae_15min_h3": row["mae_h3"],
                    "mae_30min_h6": row["mae_h6"],
                    "mae_60min_h12": row["mae_h12"],
                }
            )
        clean_row = next(row for row in metrics_rows if row["candidate"] == reference_name and row["fault"] == "clean")
        complexity_rows.append(
            {
                "dataset": "METR-LA",
                "run_id": "sraf-reliability-random-missing-repair",
                "candidate": reference_name,
                "parameter_count": reference_param_count,
                "training_time_sec": "reused_strong_baseline_audit_checkpoint",
                "clean_inference_time_sec": clean_row["inference_time_sec"],
                "best_epoch": "see_strong_baseline_audit",
                "best_val_loss": "see_strong_baseline_audit",
            }
        )
    elif not args.smoke:
        failed_rows.append({"candidate": "ResidualGRU-time-strong-reference", "status": "skipped", "reason": f"missing checkpoint: {reference_path}"})

    rdr_rows = build_rdr(metrics_rows)
    write_csv(out_dir / "metrics_by_candidate_fault.csv", metrics_rows)
    write_csv(out_dir / "horizon_metrics.csv", horizon_rows)
    write_csv(out_dir / "robustness_rdr.csv", rdr_rows)
    write_csv(out_dir / "reliability_score_diagnostics.csv", reliability_rows)
    write_csv(out_dir / "repair_quality_diagnostics.csv", repair_rows)
    write_csv(out_dir / "complexity_metrics.csv", complexity_rows)
    write_csv(out_dir / "training_curves.csv", curve_rows)
    write_csv(out_dir / "failed_or_skipped_candidates.csv", failed_rows)
    write_summaries(out_dir, metrics_rows, rdr_rows, reliability_rows, repair_rows, failed_rows)
    return {"status": "completed", "output_dir": str(out_dir), "metrics_rows": len(metrics_rows), "failed_or_skipped": len(failed_rows)}


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
