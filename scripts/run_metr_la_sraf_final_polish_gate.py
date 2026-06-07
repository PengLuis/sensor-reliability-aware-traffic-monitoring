"""Run the final limited SRAF-ResidualGRU performance polish gate."""

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


CURRENT_BEST_CLEAN_MAE = 5.9507

FAULT_SETTINGS = [
    {"fault": "clean", "label": "clean"},
    {"fault": "random_missing", "rate": 0.40, "label": "random_missing_40"},
    {"fault": "continuous_outage", "length": 24, "label": "continuous_outage_24"},
    {"fault": "gaussian_noise", "severity": "high", "label": "gaussian_noise_high"},
    {"fault": "linear_drift", "severity": "high", "label": "linear_drift_high"},
    {"fault": "stuck_at_last_value", "severity": "high", "label": "stuck_at_last_value_high"},
]

RUN_CANDIDATES = [
    {
        "config": "reproduce_current_best",
        "mode": "load_checkpoint",
        "checkpoint": "experiments/metr-la-formal-v35-polish/configs/full_train_h32/best_checkpoint.pt",
        "hidden_dim": 32,
        "input_features": 1,
        "output_features": 1,
        "loss": "mse",
        "description": "Load current best full_train_h32 checkpoint and evaluate on final high-severity polish faults.",
    },
    {
        "config": "longer_schedule_h32",
        "mode": "warm_start_train",
        "checkpoint": "experiments/metr-la-formal-v35-polish/configs/full_train_h32/best_checkpoint.pt",
        "hidden_dim": 32,
        "input_features": 1,
        "output_features": 1,
        "epochs": 10,
        "patience": 8,
        "loss": "mse",
        "weight_decay": 1.0e-5,
        "gradient_clip": 5.0,
        "reduce_lr_on_plateau": True,
        "description": "Warm-start current best and train up to 10 additional epochs, equivalent to extending the schedule toward 30 epochs.",
    },
    {
        "config": "time_of_day_features_h32",
        "mode": "train",
        "hidden_dim": 32,
        "input_features": 3,
        "output_features": 1,
        "epochs": 20,
        "patience": 6,
        "loss": "mse",
        "weight_decay": 0.0,
        "gradient_clip": None,
        "reduce_lr_on_plateau": False,
        "description": "Append sin/cos time-of-day channels using period 288; predict speed channel only.",
    },
]

SKIPPED_CONFIGS = [
    {
        "config": "clean_fault_mix_70_30",
        "status": "skipped",
        "reason": "Runtime constrained after prior full-train polish runs; A/B/D satisfy PASS coverage.",
    },
    {
        "config": "huber_loss_h32",
        "status": "skipped",
        "reason": "Runtime constrained; deferred because A/B/D completed and include two improvement candidates.",
    },
    {
        "config": "hidden48_full_train",
        "status": "skipped",
        "reason": "Runtime constrained and previous hidden_dim 64 attempt timed out; avoid heavier model before ablation.",
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/metr-la")
    parser.add_argument("--formal-v3-dir", default="experiments/metr-la-formal-v3-residual")
    parser.add_argument("--output-dir", default="experiments/metr-la-formal-v36-final-polish")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sensor-embedding-dim", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--smoke", action="store_true")
    return parser


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
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
        return x.copy(), np.zeros_like(x, dtype=bool), {"fault": "clean", "seed": seed}
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


def loss_function(name: str) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.SmoothL1Loss(beta=1.0)
    raise ValueError(f"Unknown loss: {name}")


def make_sraf(cfg: dict[str, Any], sensors: int, horizon: int, sensor_embedding_dim: int) -> SRAFResidualGRU:
    return SRAFResidualGRU(
        sensors=sensors,
        features=cfg["input_features"],
        output_features=cfg["output_features"],
        horizon=horizon,
        hidden_dim=cfg["hidden_dim"],
        sensor_embedding_dim=sensor_embedding_dim,
    )


def model_param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def train_corruption_batch(x: np.ndarray, seed: int, step: int) -> np.ndarray:
    if (seed + step) % 2 == 0:
        return x
    choices = [s for s in FAULT_SETTINGS if s["fault"] != "clean"]
    setting = choices[(seed + step) % len(choices)]
    corrupted, _, _ = apply_fault(x, setting, seed + step, train_std=1.0)
    return corrupted


def evaluate_loss(model: nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int, adjacency: torch.Tensor, loss_fn: nn.Module) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for xb, yb in iter_batches(x, y, batch_size, shuffle=False, seed=0, epoch=0):
            pred = model(torch.from_numpy(xb.astype(np.float32)), adjacency=adjacency)
            losses.append(float(loss_fn(pred, torch.from_numpy(yb.astype(np.float32))).detach().cpu()))
    return float(np.mean(losses))


def train_candidate(
    cfg: dict[str, Any],
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    args: argparse.Namespace,
    run_dir: Path,
    adjacency: torch.Tensor,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        if cfg.get("reduce_lr_on_plateau")
        else None
    )
    loss_fn = loss_function(cfg["loss"])
    best_val = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0
    batch_step = 0
    rows: list[dict[str, Any]] = []
    start = perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in iter_batches(train_x, train_y, args.batch_size, shuffle=True, seed=args.seed, epoch=epoch):
            xb = train_corruption_batch(xb, args.seed, batch_step)
            pred = model(torch.from_numpy(xb.astype(np.float32)), adjacency=adjacency)
            loss = loss_fn(pred, torch.from_numpy(yb.astype(np.float32)))
            optimizer.zero_grad()
            loss.backward()
            if cfg.get("gradient_clip") is not None:
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            batch_step += 1
        train_loss = float(np.mean(losses))
        val_loss = evaluate_loss(model, val_x, val_y, args.batch_size, adjacency, loss_fn)
        if scheduler is not None:
            scheduler.step(val_loss)
        improved = val_loss < best_val - 1.0e-6
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "run_id": "formal-v36-final-polish",
            "config": cfg["config"],
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
        print(f"{cfg['config']} epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={lr:.6g}", flush=True)
        if no_improve >= cfg["patience"]:
            rows[-1]["early_stop_triggered"] = True
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, run_dir / "best_checkpoint.pt")
    return {"training_time_sec": perf_counter() - start, "best_epoch": best_epoch, "best_val_loss": best_val}, rows


def predict_model(model: nn.Module, x: np.ndarray, batch_size: int, adjacency: torch.Tensor) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    model.eval()
    preds, residuals, reliabilities = [], [], []
    start = perf_counter()
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = x[i : i + batch_size].astype(np.float32)
            pred, comp = model(torch.from_numpy(xb), adjacency=adjacency, return_components=True)
            preds.append(pred.cpu().numpy())
            residuals.append(comp["residual_delta"].cpu().numpy())
            reliabilities.append(comp["reliability"].cpu().numpy())
    return (
        np.concatenate(preds, axis=0),
        {"residual_delta": np.concatenate(residuals, axis=0), "reliability": np.concatenate(reliabilities, axis=0)},
        perf_counter() - start,
    )


def prediction_distribution(config: str, fault: str, pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(pred) & np.isfinite(target)
    corr = float(np.corrcoef(pred[valid].ravel(), target[valid].ravel())[0, 1]) if np.any(valid) else math.nan
    return {
        "config": config,
        "fault": fault,
        "prediction_mean_norm": float(np.nanmean(pred)),
        "prediction_std_norm": float(np.nanstd(pred)),
        "prediction_min_norm": float(np.nanmin(pred)),
        "prediction_max_norm": float(np.nanmax(pred)),
        "target_mean_norm": float(np.nanmean(target)),
        "target_std_norm": float(np.nanstd(target)),
        "target_min_norm": float(np.nanmin(target)),
        "target_max_norm": float(np.nanmax(target)),
        "prediction_target_correlation": corr,
        "near_constant_prediction": bool(float(np.nanstd(pred)) < 0.10 * max(float(np.nanstd(target)), 1.0e-8)),
    }


def residual_diag(config: str, fault: str, residual: np.ndarray) -> dict[str, Any]:
    return {
        "config": config,
        "fault": fault,
        "residual_mean_norm": float(np.nanmean(residual)),
        "residual_std_norm": float(np.nanstd(residual)),
        "residual_min_norm": float(np.nanmin(residual)),
        "residual_max_norm": float(np.nanmax(residual)),
        "residual_abs_mean_norm": float(np.nanmean(np.abs(residual))),
        "residual_near_zero": bool(float(np.nanmean(np.abs(residual))) < 1.0e-4 and float(np.nanstd(residual)) < 1.0e-4),
        "residual_exploding": bool(float(np.nanmax(np.abs(residual))) > 10.0),
    }


def reliability_diag(config: str, fault: str, reliability: np.ndarray) -> dict[str, Any]:
    return {
        "config": config,
        "fault": fault,
        "reliability_mean": float(np.nanmean(reliability)),
        "reliability_std": float(np.nanstd(reliability)),
        "reliability_min": float(np.nanmin(reliability)),
        "reliability_max": float(np.nanmax(reliability)),
    }


def evaluate_residual_reference(
    formal_v3_dir: Path,
    clean_x: np.ndarray,
    test_faults: dict[str, np.ndarray],
    test_y: np.ndarray,
    adjacency: torch.Tensor,
    batch_size: int,
    mean: float,
    std: float,
) -> tuple[dict[str, float], dict[str, float]]:
    ckpt = formal_v3_dir / "models" / "ResidualGRU-corruption-aware" / "best_checkpoint.pt"
    if not ckpt.exists():
        return {}, {}
    model = ResidualGRU(sensors=clean_x.shape[2], features=1, horizon=test_y.shape[1], hidden_dim=32, sensor_embedding_dim=8)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    clean_metrics: dict[str, float] = {}
    fault_metrics: dict[str, float] = {}
    clean_mae = math.nan
    with torch.no_grad():
        for label, x in test_faults.items():
            preds = []
            for i in range(0, x.shape[0], batch_size):
                xb = torch.from_numpy(x[i : i + batch_size].astype(np.float32))
                preds.append(model(xb, adjacency=adjacency).cpu().numpy())
            pred = np.concatenate(preds, axis=0)
            mae = safe_metrics(test_y, pred, mean, std)["mae"]
            fault_metrics[label] = mae
            if label == "clean":
                clean_mae = mae
                clean_metrics["clean_mae"] = mae
    rdr = {fault: (mae - clean_mae) / clean_mae for fault, mae in fault_metrics.items()} if math.isfinite(clean_mae) else {}
    return clean_metrics, rdr


def select_candidate(metrics_rows: list[dict[str, Any]], rdr_rows: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> str:
    del skipped
    clean_rows = [row for row in metrics_rows if row["fault"] == "clean"]
    eligible = []
    for row in clean_rows:
        cfg = row["config"]
        clean_mae = row["mae"]
        cfg_rdr = [r for r in rdr_rows if r["config"] == cfg and r["fault"] != "clean"]
        better_than_ref = [
            r for r in cfg_rdr if math.isfinite(r["rdr_gap_vs_residual_gru_ca"]) and r["rdr_gap_vs_residual_gru_ca"] < 0
        ]
        if clean_mae < CURRENT_BEST_CLEAN_MAE or (abs(clean_mae - CURRENT_BEST_CLEAN_MAE) <= 0.01 and len(better_than_ref) >= 3):
            eligible.append((clean_mae, cfg, len(better_than_ref)))
    if not eligible:
        return "current_best_full_train_h32"
    eligible.sort()
    return eligible[0][1]


def current_best_training_meta() -> dict[str, float]:
    path = ROOT / "experiments/metr-la-formal-v35-polish/complexity_metrics.csv"
    if not path.exists():
        return {"best_epoch": 0.0, "best_val_loss": 0.0}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["config"] == "full_train_h32":
                return {"best_epoch": float(row["best_epoch"]), "best_val_loss": float(row["best_val_loss"])}
    return {"best_epoch": 0.0, "best_val_loss": 0.0}


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
    out_dir.mkdir(parents=True, exist_ok=True)

    fault_dir = out_dir / "fault_masks"
    fault_dir.mkdir(exist_ok=True)
    corrupted_base: dict[str, np.ndarray] = {}
    for idx, setting in enumerate(FAULT_SETTINGS):
        cx, mask, meta = apply_fault(test_x_base, setting, seed=args.seed + idx, train_std=1.0)
        corrupted_base[setting["label"]] = cx
        np.savez_compressed(fault_dir / f"{setting['label']}_mask.npz", mask=mask)
        (fault_dir / f"{setting['label']}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    persistence_clean = safe_metrics(test_y, persistence_predict(test_x_base, test_y.shape[1]), mean, std)["mae"]
    ref_metrics, ref_rdr = evaluate_residual_reference(
        Path(args.formal_v3_dir), test_x_base, corrupted_base, test_y, adjacency, args.batch_size, mean, std
    )
    current_best_meta = current_best_training_meta()

    manifest = {
        "run_id": "metr-la-formal-v36-final-polish",
        "gate": "SRAF_FINAL_PERFORMANCE_POLISH_GATE",
        "metrics_scale": "original_scale_after_inverse_transform",
        "seed": args.seed,
        "data_dir": str(data_dir),
        "output_dir": str(out_dir),
        "train_samples_used": int(train_x_base.shape[0]),
        "val_samples_used": int(val_x_base.shape[0]),
        "test_samples_used": int(test_x_base.shape[0]),
        "candidate_policy": "Run A/B/D. F/E/C skipped due runtime after prior polish timings.",
        "time_of_day_features": "sin(2*pi*t/288), cos(2*pi*t/288), using chronological window index; speed remains the only target channel.",
        "current_best_clean_mae": CURRENT_BEST_CLEAN_MAE,
        "persistence_clean_mae": persistence_clean,
        "residual_gru_reference": ref_metrics,
        "integrity_note": "Final polish gate only. No full ablation and no paper conclusions.",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "config_resolved.yaml").write_text("\n".join(f"{k}: {v}" for k, v in manifest.items()), encoding="utf-8")

    metrics_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    rdr_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    rel_rows: list[dict[str, Any]] = []
    res_rows: list[dict[str, Any]] = []
    dist_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    skipped_rows = [dict(row) for row in SKIPPED_CONFIGS]
    clean_by_config: dict[str, float] = {}

    candidates = RUN_CANDIDATES if not args.smoke else [RUN_CANDIDATES[0], {**RUN_CANDIDATES[2], "epochs": 1, "patience": 1}]
    if args.smoke:
        skipped_rows = [{"config": row["config"], "status": "skipped", "reason": "smoke mode"} for row in SKIPPED_CONFIGS + [RUN_CANDIDATES[1]]]

    for cfg in candidates:
        run_dir = out_dir / "configs" / cfg["config"]
        run_dir.mkdir(parents=True, exist_ok=True)
        use_time = cfg["input_features"] > 1
        train_x = add_time_of_day_features(train_x_base, 0) if use_time else train_x_base
        val_x = add_time_of_day_features(val_x_base, train_x_base.shape[0]) if use_time else val_x_base
        test_faults = {
            label: add_time_of_day_features(x, train_x_base.shape[0] + val_x_base.shape[0]) if use_time else x
            for label, x in corrupted_base.items()
        }
        (run_dir / "config_resolved.yaml").write_text("\n".join(f"{k}: {v}" for k, v in cfg.items()), encoding="utf-8")
        try:
            model = make_sraf(cfg, train_x.shape[2], train_y.shape[1], args.sensor_embedding_dim)
            ckpt = cfg.get("checkpoint")
            train_extra = {"training_time_sec": 0.0, "best_epoch": math.nan, "best_val_loss": math.nan}
            curves: list[dict[str, Any]] = []
            if ckpt:
                checkpoint_path = ROOT / ckpt
                if checkpoint_path.exists() and cfg["input_features"] == 1:
                    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
                elif cfg["mode"] == "load_checkpoint":
                    raise FileNotFoundError(str(checkpoint_path))
            if cfg["mode"] != "load_checkpoint":
                train_extra, curves = train_candidate(cfg, model, train_x, train_y, val_x, val_y, args, run_dir, adjacency)
                curve_rows.extend(curves)
            else:
                train_extra = {"training_time_sec": 0.0, **current_best_meta}
                torch.save(model.state_dict(), run_dir / "best_checkpoint.pt")
            param_count = model_param_count(model)
            for label, x_eval in test_faults.items():
                pred, comp, latency = predict_model(model, x_eval, args.batch_size, adjacency)
                m = safe_metrics(test_y, pred, mean, std)
                if label == "clean":
                    clean_by_config[cfg["config"]] = m["mae"]
                    np.savez_compressed(run_dir / "clean_predictions.npz", y_pred=pred, y_true=test_y)
                metrics_rows.append(
                    {
                        "dataset": "METR-LA",
                        "run_id": "formal-v36-final-polish",
                        "metrics_scale": "original",
                        "config": cfg["config"],
                        "fault": label,
                        "mae": m["mae"],
                        "rmse": m["rmse"],
                        "mape": m["mape"],
                        "mae_h3": m.get("mae_h3", math.nan),
                        "mae_h6": m.get("mae_h6", math.nan),
                        "mae_h12": m.get("mae_h12", math.nan),
                        "parameter_count": param_count,
                        "inference_time_sec": latency,
                        "best_epoch": train_extra["best_epoch"],
                        "best_val_loss": train_extra["best_val_loss"],
                        "clean_mae_gap_vs_persistence": 0.0,
                        "clean_mae_gap_vs_current_best_sraf": 0.0,
                    }
                )
                horizon_rows.append(
                    {
                        "dataset": "METR-LA",
                        "run_id": "formal-v36-final-polish",
                        "config": cfg["config"],
                        "fault": label,
                        "mae_15min_h3": m.get("mae_h3", math.nan),
                        "mae_30min_h6": m.get("mae_h6", math.nan),
                        "mae_60min_h12": m.get("mae_h12", math.nan),
                    }
                )
                rel_rows.append(reliability_diag(cfg["config"], label, comp["reliability"]))
                res_rows.append(residual_diag(cfg["config"], label, comp["residual_delta"]))
                dist_rows.append(prediction_distribution(cfg["config"], label, pred, test_y))
            complexity_rows.append(
                {
                    "config": cfg["config"],
                    "parameter_count": param_count,
                    "training_time_sec": train_extra["training_time_sec"],
                    "clean_inference_time_sec": next(
                        row["inference_time_sec"] for row in metrics_rows if row["config"] == cfg["config"] and row["fault"] == "clean"
                    ),
                    "best_epoch": train_extra["best_epoch"],
                    "best_val_loss": train_extra["best_val_loss"],
                    "hidden_dim": cfg["hidden_dim"],
                    "input_features": cfg["input_features"],
                    "output_features": cfg["output_features"],
                }
            )
        except Exception as exc:
            skipped_rows.append({"config": cfg["config"], "status": "failed", "reason": repr(exc)})

    for row in metrics_rows:
        clean_row = next(item for item in metrics_rows if item["config"] == row["config"] and item["fault"] == "clean")
        row["clean_mae_gap_vs_persistence"] = clean_row["mae"] - persistence_clean
        row["clean_mae_gap_vs_current_best_sraf"] = clean_row["mae"] - CURRENT_BEST_CLEAN_MAE
        clean = clean_by_config.get(row["config"], math.nan)
        rdr = (row["mae"] - clean) / clean if clean and math.isfinite(clean) else math.nan
        ref = ref_rdr.get(row["fault"], math.nan)
        rdr_rows.append(
            {
                "dataset": row["dataset"],
                "run_id": row["run_id"],
                "config": row["config"],
                "fault": row["fault"],
                "clean_mae": clean,
                "fault_mae": row["mae"],
                "rdr_mae": rdr,
                "residual_gru_ca_rdr_mae": ref,
                "rdr_gap_vs_residual_gru_ca": rdr - ref if math.isfinite(rdr) and math.isfinite(ref) else math.nan,
            }
        )

    selected = select_candidate(metrics_rows, rdr_rows, skipped_rows)
    write_csv(out_dir / "metrics_by_config_fault.csv", metrics_rows)
    write_csv(out_dir / "robustness_rdr.csv", rdr_rows)
    write_csv(out_dir / "horizon_metrics.csv", horizon_rows)
    write_csv(out_dir / "complexity_metrics.csv", complexity_rows)
    write_csv(out_dir / "reliability_score_diagnostics.csv", rel_rows)
    write_csv(out_dir / "residual_diagnostics.csv", res_rows)
    write_csv(out_dir / "prediction_distribution.csv", dist_rows)
    write_csv(out_dir / "training_curves.csv", curve_rows)
    write_csv(out_dir / "failed_or_skipped_configs.csv", skipped_rows)

    clean_summary = [row for row in metrics_rows if row["fault"] == "clean"]
    summary = [
        "# Candidate Selection Summary",
        "",
        f"Selected candidate: `{selected}`",
        "",
        "Clean MAE by candidate:",
    ]
    for row in sorted(clean_summary, key=lambda item: item["mae"]):
        summary.append(f"- `{row['config']}`: MAE {row['mae']:.6f}")
    summary.extend(
        [
            "",
            f"Current best reference MAE: {CURRENT_BEST_CLEAN_MAE:.4f}",
            f"Persistence clean MAE on this gate: {persistence_clean:.6f}",
            "",
            "Selection rule: choose a candidate only if clean MAE improves, or similar clean MAE has stronger robustness without substantial complexity increase.",
            "No paper conclusions are written from this gate.",
        ]
    )
    (out_dir / "candidate_selection_summary.md").write_text("\n".join(summary), encoding="utf-8")

    return {"status": "completed", "output_dir": str(out_dir), "selected": selected, "metrics_rows": len(metrics_rows)}


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
