"""Run a limited SRAF-ResidualGRU performance polish gate."""

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
    random_missing,
    stuck_at_last_value,
)
from src.metrics.regression import regression_metrics  # noqa: E402
from src.models.residual_models import SRAFResidualGRU  # noqa: E402


FAULT_SETTINGS = [
    {"fault": "clean", "label": "clean", "formal_v3_seed_offset": 0},
    {"fault": "random_missing", "rate": 0.40, "label": "random_missing_40", "formal_v3_seed_offset": 2},
    {"fault": "gaussian_noise", "severity": "medium", "label": "gaussian_noise_medium", "formal_v3_seed_offset": 4},
    {
        "fault": "stuck_at_last_value",
        "severity": "medium",
        "label": "stuck_at_last_value_medium",
        "formal_v3_seed_offset": 6,
    },
    {"fault": "continuous_outage", "length": 12, "label": "continuous_outage_12", "formal_v3_seed_offset": 3},
]


PRIORITY_CANDIDATES = [
    {
        "config": "baseline_formal_v3_reproduce",
        "train_samples": 16000,
        "hidden_dim": 32,
        "epochs": 20,
        "patience": 6,
        "loss": "mse",
        "reliability_bias_init": None,
        "description": "Reproduce formal-v3 SRAF-ResidualGRU settings on the polish fault set.",
    },
    {
        "config": "full_train_h32",
        "train_samples": 0,
        "hidden_dim": 32,
        "epochs": 20,
        "patience": 6,
        "loss": "mse",
        "reliability_bias_init": None,
        "description": "Use all available METR-LA training samples with hidden_dim=32.",
    },
    {
        "config": "reliability_clean_bias",
        "train_samples": 16000,
        "hidden_dim": 32,
        "epochs": 20,
        "patience": 6,
        "loss": "mse",
        "reliability_bias_init": 2.0,
        "description": "Initialize final reliability layer to sigmoid(2.0), about 0.88 trust for clean observations.",
    },
]


SKIPPED_CANDIDATES = [
    {
        "config": "hidden48_16000",
        "status": "skipped",
        "reason": "Runtime constrained after formal-v3 hidden_dim=64 timeout; priority set A/B/D run first.",
    },
    {
        "config": "mae_or_huber_loss",
        "status": "skipped",
        "reason": "Runtime constrained; loss-function polish deferred unless A/B/D fail the tradeoff.",
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/metr-la")
    parser.add_argument("--formal-v3-dir", default="experiments/metr-la-formal-v3-residual")
    parser.add_argument("--output-dir", default="experiments/metr-la-formal-v35-polish")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sensor-embedding-dim", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--val-samples", type=int, default=2048)
    parser.add_argument("--test-samples", type=int, default=0, help="0 means full test set.")
    parser.add_argument("--smoke", action="store_true", help="Run tiny A/D-only smoke settings for code validation.")
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


def load_split(data_dir: Path, split: str, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(data_dir / f"{split}.npz")
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.float32)
    if limit is not None and limit > 0:
        x = x[:limit]
        y = y[:limit]
    return x, y


def load_scale(data_dir: Path) -> tuple[float, float]:
    stats = json.loads((data_dir / "dataset_stats.json").read_text(encoding="utf-8"))
    return float(stats["mean"]), float(stats["std"])


def inverse_scale(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return x * std + mean


def safe_metrics(y_true_norm: np.ndarray, y_pred_norm: np.ndarray, mean: float, std: float) -> dict[str, float]:
    return regression_metrics(inverse_scale(y_true_norm, mean, std), inverse_scale(y_pred_norm, mean, std))


def apply_fault(
    x: np.ndarray,
    setting: dict[str, Any],
    seed: int,
    train_std: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if setting["fault"] == "clean":
        return x.copy(), np.zeros_like(x, dtype=bool), {"fault": "clean", "seed": seed}
    if setting["fault"] == "random_missing":
        return random_missing(x, rate=setting["rate"], seed=seed)
    if setting["fault"] == "continuous_outage":
        return continuous_outage(x, length=setting["length"], seed=seed)
    if setting["fault"] == "gaussian_noise":
        return gaussian_noise(x, severity=setting["severity"], train_std=train_std, seed=seed)
    if setting["fault"] == "stuck_at_last_value":
        return stuck_at_last_value(x, severity=setting["severity"], seed=seed)
    raise ValueError(f"Unknown fault setting: {setting}")


def corruption_aware_batch(x: np.ndarray, seed: int, step: int, train_std: float) -> np.ndarray:
    choices = [s for s in FAULT_SETTINGS if s["fault"] != "clean"]
    setting = choices[(seed + step) % len(choices)]
    corrupted, _, _ = apply_fault(x, setting, seed + step, train_std)
    return corrupted


def iter_batches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    epoch: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(x.shape[0])
    if shuffle:
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(indices)
    return [(x[idx], y[idx]) for idx in np.array_split(indices, math.ceil(len(indices) / batch_size))]


def loss_function(name: str) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    if name == "huber":
        return nn.SmoothL1Loss(beta=0.5)
    raise ValueError(f"Unknown loss: {name}")


def model_param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def evaluate_loss(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    adjacency: torch.Tensor,
    loss_fn: nn.Module,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for xb, yb in iter_batches(x, y, batch_size, shuffle=False, seed=0, epoch=0):
            pred = model(torch.from_numpy(xb.astype(np.float32)), adjacency=adjacency)
            losses.append(float(loss_fn(pred, torch.from_numpy(yb.astype(np.float32))).detach().cpu()))
    return float(np.mean(losses))


def train_candidate(
    cfg: dict[str, Any],
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    args: argparse.Namespace,
    run_dir: Path,
    adjacency: torch.Tensor,
) -> tuple[nn.Module, dict[str, float], list[dict[str, Any]]]:
    _, _, sensors, features = train_x.shape
    model = SRAFResidualGRU(
        sensors=sensors,
        features=features,
        horizon=train_y.shape[1],
        hidden_dim=cfg["hidden_dim"],
        sensor_embedding_dim=args.sensor_embedding_dim,
        reliability_bias_init=cfg["reliability_bias_init"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
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
            xb = corruption_aware_batch(xb, args.seed, batch_step, train_std=1.0)
            pred = model(torch.from_numpy(xb.astype(np.float32)), adjacency=adjacency)
            loss = loss_fn(pred, torch.from_numpy(yb.astype(np.float32)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            batch_step += 1
        train_loss = float(np.mean(losses))
        val_loss = evaluate_loss(model, val_x, val_y, args.batch_size, adjacency, loss_fn)
        improved = val_loss < best_val - 1.0e-6
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        row = {
            "run_id": "formal-v35-polish",
            "config": cfg["config"],
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val,
            "loss": cfg["loss"],
            "improved": improved,
            "early_stop_triggered": False,
            "has_nan_or_inf": (not math.isfinite(train_loss)) or (not math.isfinite(val_loss)),
        }
        rows.append(row)
        print(f"{cfg['config']} epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
        if no_improve >= cfg["patience"]:
            rows[-1]["early_stop_triggered"] = True
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, run_dir / "best_checkpoint.pt")
    training_time = perf_counter() - start
    (run_dir / "train_log.txt").write_text(
        "\n".join(
            f"epoch={r['epoch']},train_loss={r['train_loss']:.6f},val_loss={r['val_loss']:.6f},"
            f"best_val_loss={r['best_val_loss']:.6f},early_stop_triggered={r['early_stop_triggered']}"
            for r in rows
        ),
        encoding="utf-8",
    )
    return model, {"training_time_sec": training_time, "best_epoch": best_epoch, "best_val_loss": best_val}, rows


def predict_candidate(
    model: nn.Module,
    x: np.ndarray,
    y_shape_template: tuple[int, ...],
    batch_size: int,
    adjacency: torch.Tensor,
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    del y_shape_template
    model.eval()
    preds = []
    residuals = []
    reliabilities = []
    start = perf_counter()
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = x[i : i + batch_size].astype(np.float32)
            pred, components = model(torch.from_numpy(xb), adjacency=adjacency, return_components=True)
            preds.append(pred.cpu().numpy())
            residuals.append(components["residual_delta"].cpu().numpy())
            reliabilities.append(components["reliability"].cpu().numpy())
    return (
        np.concatenate(preds, axis=0),
        {
            "residual_delta": np.concatenate(residuals, axis=0),
            "reliability": np.concatenate(reliabilities, axis=0),
        },
        perf_counter() - start,
    )


def distribution_row(config: str, fault: str, pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
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


def residual_row(config: str, fault: str, residual: np.ndarray) -> dict[str, Any]:
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


def reliability_row(config: str, fault: str, reliability: np.ndarray) -> dict[str, Any]:
    return {
        "config": config,
        "fault": fault,
        "reliability_mean": float(np.nanmean(reliability)),
        "reliability_std": float(np.nanstd(reliability)),
        "reliability_min": float(np.nanmin(reliability)),
        "reliability_max": float(np.nanmax(reliability)),
    }


def load_formal_v3_reference(formal_v3_dir: Path) -> dict[str, Any]:
    metrics_path = formal_v3_dir / "metrics_by_model_fault.csv"
    rdr_path = formal_v3_dir / "robustness_rdr.csv"
    if not metrics_path.exists() or not rdr_path.exists():
        return {}
    metrics = []
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    rdr = []
    with rdr_path.open(newline="", encoding="utf-8") as handle:
        rdr = list(csv.DictReader(handle))
    residual_clean = next(
        float(row["mae"])
        for row in metrics
        if row["model"] == "ResidualGRU-corruption-aware" and row["fault"] == "clean"
    )
    sraf_clean = next(
        float(row["mae"])
        for row in metrics
        if row["model"] == "SRAF-ResidualGRU-corruption-aware" and row["fault"] == "clean"
    )
    residual_rdr = {
        row["fault"]: float(row["rdr_mae"])
        for row in rdr
        if row["model"] == "ResidualGRU-corruption-aware"
    }
    sraf_rdr = {
        row["fault"]: float(row["rdr_mae"])
        for row in rdr
        if row["model"] == "SRAF-ResidualGRU-corruption-aware"
    }
    return {
        "residual_gru_ca_clean_mae": residual_clean,
        "formal_v3_sraf_clean_mae": sraf_clean,
        "residual_gru_ca_rdr": residual_rdr,
        "formal_v3_sraf_rdr": sraf_rdr,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mean, std = load_scale(data_dir)
    train_full_x, train_full_y = load_split(data_dir, "train", None)
    val_x, val_y = load_split(data_dir, "val", args.val_samples)
    test_limit = None if args.test_samples == 0 else args.test_samples
    test_x, test_y = load_split(data_dir, "test", test_limit)
    adjacency = torch.from_numpy(np.load(data_dir / "adjacency.npy").astype(np.float32))
    reference = load_formal_v3_reference(Path(args.formal_v3_dir))

    manifest = {
        "run_id": "metr-la-formal-v35-polish",
        "gate": "SRAF_RESIDUAL_PERFORMANCE_POLISH_GATE",
        "metrics_scale": "original_scale_after_inverse_transform",
        "seed": args.seed,
        "data_dir": str(data_dir),
        "output_dir": str(out_dir),
        "test_samples_used": int(test_x.shape[0]),
        "val_samples_used": int(val_x.shape[0]),
        "batch_size": args.batch_size,
        "sensor_embedding_dim": args.sensor_embedding_dim,
        "learning_rate": args.learning_rate,
        "fault_settings": [s["label"] for s in FAULT_SETTINGS],
        "candidate_policy": "Runtime-limited priority set A/B/D. C/E are explicitly skipped.",
        "reference": reference,
        "integrity_note": "Polish gate only. No full ablation and no paper conclusions.",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "config_resolved.yaml").write_text("\n".join(f"{k}: {v}" for k, v in manifest.items()), encoding="utf-8")

    fault_dir = out_dir / "fault_masks"
    fault_dir.mkdir(exist_ok=True)
    corrupted_test: dict[str, np.ndarray] = {}
    for setting in FAULT_SETTINGS:
        seed = args.seed + setting["formal_v3_seed_offset"]
        cx, mask, meta = apply_fault(test_x, setting, seed=seed, train_std=1.0)
        label = setting["label"]
        corrupted_test[label] = cx
        np.savez_compressed(fault_dir / f"{label}_mask.npz", mask=mask)
        (fault_dir / f"{label}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    metrics_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    rdr_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    train_curve_rows: list[dict[str, Any]] = []
    candidate_list = PRIORITY_CANDIDATES
    failed_or_skipped_rows: list[dict[str, Any]] = [dict(row) for row in SKIPPED_CANDIDATES]
    if args.smoke:
        candidate_list = [
            {**PRIORITY_CANDIDATES[0], "train_samples": 128, "epochs": 1, "patience": 1},
            {**PRIORITY_CANDIDATES[2], "train_samples": 128, "epochs": 1, "patience": 1},
        ]
        failed_or_skipped_rows = [
            {"config": "full_train_h32", "status": "skipped", "reason": "smoke mode"},
            {"config": "hidden48_16000", "status": "skipped", "reason": "smoke mode"},
            {"config": "mae_or_huber_loss", "status": "skipped", "reason": "smoke mode"},
        ]
    clean_by_config: dict[str, float] = {}

    for cfg in candidate_list:
        config_name = cfg["config"]
        run_dir = out_dir / "configs" / config_name
        run_dir.mkdir(parents=True, exist_ok=True)
        train_limit = None if cfg["train_samples"] == 0 else cfg["train_samples"]
        train_x = train_full_x if train_limit is None else train_full_x[:train_limit]
        train_y = train_full_y if train_limit is None else train_full_y[:train_limit]
        resolved_cfg = {
            **cfg,
            "train_samples_used": int(train_x.shape[0]),
            "val_samples_used": int(val_x.shape[0]),
            "test_samples_used": int(test_x.shape[0]),
            "seed": args.seed,
        }
        (run_dir / "config_resolved.yaml").write_text(
            "\n".join(f"{key}: {value}" for key, value in resolved_cfg.items()),
            encoding="utf-8",
        )
        try:
            model, train_extra, curves = train_candidate(cfg, train_x, train_y, val_x, val_y, args, run_dir, adjacency)
            train_curve_rows.extend(curves)
            param_count = model_param_count(model)
            for setting in FAULT_SETTINGS:
                label = setting["label"]
                pred, components, inference_time = predict_candidate(
                    model,
                    corrupted_test[label],
                    test_y.shape,
                    args.batch_size,
                    adjacency,
                )
                m = safe_metrics(test_y, pred, mean, std)
                if label == "clean":
                    clean_by_config[config_name] = m["mae"]
                    np.savez_compressed(run_dir / "clean_predictions.npz", y_pred=pred, y_true=test_y)
                row = {
                    "dataset": "METR-LA",
                    "run_id": "formal-v35-polish",
                    "metrics_scale": "original",
                    "config": config_name,
                    "fault": label,
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "mape": m["mape"],
                    "mae_h3": m.get("mae_h3", math.nan),
                    "mae_h6": m.get("mae_h6", math.nan),
                    "mae_h12": m.get("mae_h12", math.nan),
                    "parameter_count": param_count,
                    "inference_time_sec": inference_time,
                    "best_epoch": train_extra["best_epoch"],
                    "best_val_loss": train_extra["best_val_loss"],
                    "loss": cfg["loss"],
                    "hidden_dim": cfg["hidden_dim"],
                    "train_samples_used": int(train_x.shape[0]),
                    "clean_mae_gap_vs_residual_gru_ca": m["mae"] - reference.get("residual_gru_ca_clean_mae", math.nan)
                    if label == "clean"
                    else math.nan,
                }
                metrics_rows.append(row)
                horizon_rows.append(
                    {
                        "dataset": "METR-LA",
                        "run_id": "formal-v35-polish",
                        "config": config_name,
                        "fault": label,
                        "mae_15min_h3": row["mae_h3"],
                        "mae_30min_h6": row["mae_h6"],
                        "mae_60min_h12": row["mae_h12"],
                    }
                )
                distribution_rows.append(distribution_row(config_name, label, pred, test_y))
                residual_rows.append(residual_row(config_name, label, components["residual_delta"]))
                reliability_rows.append(reliability_row(config_name, label, components["reliability"]))
            complexity_rows.append(
                {
                    "config": config_name,
                    "parameter_count": param_count,
                    "training_time_sec": train_extra["training_time_sec"],
                    "clean_inference_time_sec": next(
                        row["inference_time_sec"]
                        for row in metrics_rows
                        if row["config"] == config_name and row["fault"] == "clean"
                    ),
                    "best_epoch": train_extra["best_epoch"],
                    "best_val_loss": train_extra["best_val_loss"],
                    "hidden_dim": cfg["hidden_dim"],
                    "train_samples_used": int(train_x.shape[0]),
                    "loss": cfg["loss"],
                    "reliability_bias_init": cfg["reliability_bias_init"],
                }
            )
        except Exception as exc:
            failed_or_skipped_rows.append({"config": config_name, "status": "failed", "reason": repr(exc)})

    for row in metrics_rows:
        clean = clean_by_config.get(row["config"], math.nan)
        rdr = (row["mae"] - clean) / clean if clean and math.isfinite(clean) else math.nan
        formal_v3_sraf_rdr = reference.get("formal_v3_sraf_rdr", {}).get(row["fault"], math.nan)
        residual_ca_rdr = reference.get("residual_gru_ca_rdr", {}).get(row["fault"], math.nan)
        rdr_rows.append(
            {
                "dataset": row["dataset"],
                "run_id": row["run_id"],
                "config": row["config"],
                "fault": row["fault"],
                "clean_mae": clean,
                "fault_mae": row["mae"],
                "rdr_mae": rdr,
                "formal_v3_sraf_rdr_mae": formal_v3_sraf_rdr,
                "residual_gru_ca_rdr_mae": residual_ca_rdr,
                "rdr_gap_vs_formal_v3_sraf": rdr - formal_v3_sraf_rdr if math.isfinite(rdr) and math.isfinite(formal_v3_sraf_rdr) else math.nan,
                "rdr_gap_vs_residual_gru_ca": rdr - residual_ca_rdr if math.isfinite(rdr) and math.isfinite(residual_ca_rdr) else math.nan,
            }
        )

    write_csv(out_dir / "metrics_by_config_fault.csv", metrics_rows)
    write_csv(out_dir / "horizon_metrics.csv", horizon_rows)
    write_csv(out_dir / "robustness_rdr.csv", rdr_rows)
    write_csv(out_dir / "complexity_metrics.csv", complexity_rows)
    write_csv(out_dir / "reliability_score_diagnostics.csv", reliability_rows)
    write_csv(out_dir / "residual_diagnostics.csv", residual_rows)
    write_csv(out_dir / "prediction_distribution.csv", distribution_rows)
    write_csv(out_dir / "training_curves.csv", train_curve_rows)
    write_csv(out_dir / "failed_or_skipped_configs.csv", failed_or_skipped_rows)
    return {
        "status": "completed",
        "output_dir": str(out_dir),
        "metrics_rows": len(metrics_rows),
        "failed_or_skipped": failed_or_skipped_rows,
    }


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
