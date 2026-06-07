"""Run the METR-LA residual spatiotemporal model repair gate."""

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
    {"fault": "clean", "label": "clean"},
    {"fault": "random_missing", "rate": 0.20, "label": "random_missing_20"},
    {"fault": "random_missing", "rate": 0.40, "label": "random_missing_40"},
    {"fault": "continuous_outage", "length": 12, "label": "continuous_outage_12"},
    {"fault": "gaussian_noise", "severity": "medium", "label": "gaussian_noise_medium"},
    {"fault": "linear_drift", "severity": "medium", "label": "linear_drift_medium"},
    {"fault": "stuck_at_last_value", "severity": "medium", "label": "stuck_at_last_value_medium"},
]


MODEL_SPECS = [
    {"name": "GRU-corruption-aware-reference", "kind": "flat_gru", "train": "corruption_aware"},
    {"name": "ResidualGRU-clean", "kind": "residual_gru", "train": "clean"},
    {"name": "ResidualGRU-corruption-aware", "kind": "residual_gru", "train": "corruption_aware"},
    {"name": "SRAF-ResidualGRU-corruption-aware", "kind": "sraf_residual_gru", "train": "corruption_aware"},
]


class FlatGRUForecast(nn.Module):
    """Previous flattened full-network GRU reference."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.head(hidden[-1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/metr-la")
    parser.add_argument("--output-dir", default="experiments/metr-la-formal-v3-residual")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--sensor-embedding-dim", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--train-samples", type=int, default=20000)
    parser.add_argument("--val-samples", type=int, default=2048)
    parser.add_argument("--test-samples", type=int, default=0, help="0 means full test set.")
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


def model_param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


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
    if setting["fault"] == "linear_drift":
        return linear_drift(x, severity=setting["severity"], train_std=train_std, seed=seed)
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


def make_model(spec: dict[str, str], x_shape: tuple[int, ...], horizon: int, args: argparse.Namespace) -> nn.Module:
    _, _, sensors, features = x_shape
    if spec["kind"] == "flat_gru":
        flat_dim = sensors * features
        return FlatGRUForecast(flat_dim, args.hidden_dim, horizon * flat_dim)
    if spec["kind"] == "residual_gru":
        return ResidualGRU(
            sensors=sensors,
            features=features,
            horizon=horizon,
            hidden_dim=args.hidden_dim,
            sensor_embedding_dim=args.sensor_embedding_dim,
        )
    if spec["kind"] == "sraf_residual_gru":
        return SRAFResidualGRU(
            sensors=sensors,
            features=features,
            horizon=horizon,
            hidden_dim=args.hidden_dim,
            sensor_embedding_dim=args.sensor_embedding_dim,
        )
    raise ValueError(f"Unknown model kind: {spec['kind']}")


def forward_model(
    model: nn.Module,
    spec: dict[str, str],
    xb: np.ndarray,
    y_shape: tuple[int, ...],
    adjacency: torch.Tensor,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if spec["kind"] == "flat_gru":
        xb = np.nan_to_num(xb, nan=0.0).astype(np.float32)
        x_t = torch.from_numpy(xb)
        batch, length, sensors, features = xb.shape
        pred = model(x_t.reshape(batch, length, sensors * features)).reshape(y_shape)
        if return_components:
            return pred, {}
        return pred
    x_t = torch.from_numpy(xb.astype(np.float32))
    return model(x_t, adjacency=adjacency, return_components=return_components)


def evaluate_loss(
    model: nn.Module,
    spec: dict[str, str],
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    adjacency: torch.Tensor,
) -> float:
    model.eval()
    loss_fn = nn.MSELoss()
    losses: list[float] = []
    with torch.no_grad():
        for xb, yb in iter_batches(x, y, batch_size, shuffle=False, seed=0, epoch=0):
            pred = forward_model(model, spec, xb, yb.shape, adjacency)
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = loss_fn(pred, torch.from_numpy(yb.astype(np.float32)))
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def train_model(
    spec: dict[str, str],
    model: nn.Module,
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
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in iter_batches(train_x, train_y, args.batch_size, shuffle=True, seed=args.seed, epoch=epoch):
            if spec["train"] == "corruption_aware":
                xb = corruption_aware_batch(xb, args.seed, batch_step, train_std=1.0)
            pred = forward_model(model, spec, xb, yb.shape, adjacency)
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = loss_fn(pred, torch.from_numpy(yb.astype(np.float32)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            batch_step += 1
        train_loss = float(np.mean(losses))
        val_loss = evaluate_loss(model, spec, val_x, val_y, args.batch_size, adjacency)
        improved = val_loss < best_val - 1.0e-6
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        row = {
            "run_id": "formal-v3-residual",
            "model": spec["name"],
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val,
            "improved": improved,
            "early_stop_triggered": False,
            "has_nan_or_inf": (not math.isfinite(train_loss)) or (not math.isfinite(val_loss)),
        }
        rows.append(row)
        print(f"{spec['name']} epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
        if no_improve >= args.patience:
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
    return {"training_time_sec": training_time, "best_epoch": best_epoch, "best_val_loss": best_val}, rows


def predict_model(
    model: nn.Module,
    spec: dict[str, str],
    x: np.ndarray,
    y_shape_template: tuple[int, ...],
    batch_size: int,
    adjacency: torch.Tensor,
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    model.eval()
    preds = []
    residuals = []
    bases = []
    start = perf_counter()
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = x[i : i + batch_size]
            pred_shape = (xb.shape[0],) + y_shape_template[1:]
            out = forward_model(model, spec, xb, pred_shape, adjacency, return_components=True)
            if isinstance(out, tuple):
                pred, components = out
            else:
                pred, components = out, {}
            preds.append(pred.cpu().numpy())
            if "residual_delta" in components:
                residuals.append(components["residual_delta"].cpu().numpy())
            if "base" in components:
                bases.append(components["base"].cpu().numpy())
    components_np: dict[str, np.ndarray] = {}
    if residuals:
        components_np["residual_delta"] = np.concatenate(residuals, axis=0)
    if bases:
        components_np["base"] = np.concatenate(bases, axis=0)
    return np.concatenate(preds, axis=0), components_np, perf_counter() - start


def distribution_row(
    model: str,
    fault: str,
    pred: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    valid = np.isfinite(pred) & np.isfinite(target)
    corr = float(np.corrcoef(pred[valid].ravel(), target[valid].ravel())[0, 1]) if np.any(valid) else math.nan
    return {
        "model": model,
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


def residual_stats(components: dict[str, np.ndarray]) -> dict[str, Any]:
    residual = components.get("residual_delta")
    if residual is None:
        return {
            "residual_mean_norm": math.nan,
            "residual_std_norm": math.nan,
            "residual_min_norm": math.nan,
            "residual_max_norm": math.nan,
            "residual_abs_mean_norm": math.nan,
            "residual_near_zero": math.nan,
            "residual_exploding": math.nan,
        }
    std = float(np.nanstd(residual))
    abs_mean = float(np.nanmean(np.abs(residual)))
    max_abs = float(np.nanmax(np.abs(residual)))
    return {
        "residual_mean_norm": float(np.nanmean(residual)),
        "residual_std_norm": std,
        "residual_min_norm": float(np.nanmin(residual)),
        "residual_max_norm": float(np.nanmax(residual)),
        "residual_abs_mean_norm": abs_mean,
        "residual_near_zero": bool(abs_mean < 1.0e-4 and std < 1.0e-4),
        "residual_exploding": bool(max_abs > 10.0),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_y = load_split(data_dir, "train", args.train_samples)
    val_x, val_y = load_split(data_dir, "val", args.val_samples)
    test_limit = None if args.test_samples == 0 else args.test_samples
    test_x, test_y = load_split(data_dir, "test", test_limit)
    mean, std = load_scale(data_dir)
    adjacency = torch.from_numpy(np.load(data_dir / "adjacency.npy").astype(np.float32))

    manifest = {
        "run_id": "metr-la-formal-v3-residual",
        "gate": "RESIDUAL_SPATIOTEMPORAL_MODEL_REPAIR_GATE",
        "status": "formal_v3_residual_diagnostic",
        "metrics_scale": "original_scale_after_inverse_transform",
        "seed": args.seed,
        "data_dir": str(data_dir),
        "output_dir": str(out_dir),
        "train_samples_used": int(train_x.shape[0]),
        "val_samples_used": int(val_x.shape[0]),
        "test_samples_used": int(test_x.shape[0]),
        "epochs_max": args.epochs,
        "early_stopping_patience": args.patience,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "sensor_embedding_dim": args.sensor_embedding_dim,
        "learning_rate": args.learning_rate,
        "fault_settings": [s["label"] for s in FAULT_SETTINGS],
        "method_note": "Residual models predict Y_hat[h,i] = repaired_persistence_base[i] + delta_theta[h,i].",
        "integrity_note": "Diagnostic run only. No ablation and no paper conclusions.",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "config_resolved.yaml").write_text("\n".join(f"{k}: {v}" for k, v in manifest.items()), encoding="utf-8")

    fault_dir = out_dir / "fault_masks"
    fault_dir.mkdir(exist_ok=True)
    corrupted_test: dict[str, np.ndarray] = {}
    for idx, setting in enumerate(FAULT_SETTINGS):
        cx, mask, meta = apply_fault(test_x, setting, args.seed + idx, train_std=1.0)
        label = setting["label"]
        corrupted_test[label] = cx
        np.savez_compressed(fault_dir / f"{label}_mask.npz", mask=mask)
        (fault_dir / f"{label}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    metrics_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    rdr_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    clean_by_model: dict[str, float] = {}

    persistence_metrics_by_fault: dict[str, dict[str, float]] = {}
    persistence_clean_inference_time = math.nan
    for setting in FAULT_SETTINGS:
        label = setting["label"]
        start = perf_counter()
        pred = persistence_predict(np.nan_to_num(corrupted_test[label], nan=0.0), test_y.shape[1])
        inference_time = perf_counter() - start
        m = safe_metrics(test_y, pred, mean, std)
        persistence_metrics_by_fault[label] = m
        row = {
            "dataset": "METR-LA",
            "run_id": "formal-v3-residual",
            "metrics_scale": "original",
            "model": "Persistence",
            "fault": label,
            "mae": m["mae"],
            "rmse": m["rmse"],
            "mape": m["mape"],
            "mae_h3": m.get("mae_h3", math.nan),
            "mae_h6": m.get("mae_h6", math.nan),
            "mae_h12": m.get("mae_h12", math.nan),
            "parameter_count": 0,
            "inference_time_sec": inference_time,
            "best_epoch": math.nan,
            "best_val_loss": math.nan,
        }
        metrics_rows.append(row)
        horizon_rows.append(
            {
                "dataset": "METR-LA",
                "run_id": "formal-v3-residual",
                "model": "Persistence",
                "fault": label,
                "mae_15min_h3": row["mae_h3"],
                "mae_30min_h6": row["mae_h6"],
                "mae_60min_h12": row["mae_h12"],
            }
        )
        if label == "clean":
            clean_by_model["Persistence"] = row["mae"]
            persistence_clean_inference_time = inference_time
            np.savez_compressed(out_dir / "persistence_clean_predictions.npz", y_pred=pred, y_true=test_y)
        distribution_rows.append(distribution_row("Persistence", label, pred, test_y))
    complexity_rows.append(
        {
            "model": "Persistence",
            "parameter_count": 0,
            "training_time_sec": 0.0,
            "clean_inference_time_sec": persistence_clean_inference_time,
            "best_epoch": math.nan,
            "best_val_loss": math.nan,
        }
    )

    for spec in MODEL_SPECS:
        model_name = spec["name"]
        run_dir = out_dir / "models" / model_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config_resolved.yaml").write_text(
            "\n".join(
                [
                    f"model: {model_name}",
                    f"kind: {spec['kind']}",
                    f"train_protocol: {spec['train']}",
                    f"seed: {args.seed}",
                    f"hidden_dim: {args.hidden_dim}",
                    f"sensor_embedding_dim: {args.sensor_embedding_dim}",
                    f"learning_rate: {args.learning_rate}",
                ]
            ),
            encoding="utf-8",
        )
        try:
            model = make_model(spec, train_x.shape, train_y.shape[1], args)
            train_extra, model_curves = train_model(spec, model, train_x, train_y, val_x, val_y, args, run_dir, adjacency)
            curve_rows.extend(model_curves)
            param_count = model_param_count(model)
            clean_inference_time = math.nan
            for setting in FAULT_SETTINGS:
                label = setting["label"]
                pred, components, inference_time = predict_model(
                    model,
                    spec,
                    corrupted_test[label],
                    test_y.shape,
                    args.batch_size,
                    adjacency,
                )
                m = safe_metrics(test_y, pred, mean, std)
                if label == "clean":
                    clean_by_model[model_name] = m["mae"]
                    clean_inference_time = inference_time
                    np.savez_compressed(run_dir / "clean_predictions.npz", y_pred=pred, y_true=test_y)
                row = {
                    "dataset": "METR-LA",
                    "run_id": "formal-v3-residual",
                    "metrics_scale": "original",
                    "model": model_name,
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
                }
                metrics_rows.append(row)
                horizon_rows.append(
                    {
                        "dataset": "METR-LA",
                        "run_id": "formal-v3-residual",
                        "model": model_name,
                        "fault": label,
                        "mae_15min_h3": row["mae_h3"],
                        "mae_30min_h6": row["mae_h6"],
                        "mae_60min_h12": row["mae_h12"],
                    }
                )
                distribution_rows.append(distribution_row(model_name, label, pred, test_y))
                pers = persistence_metrics_by_fault[label]
                rstats = residual_stats(components)
                residual_rows.append(
                    {
                        "dataset": "METR-LA",
                        "run_id": "formal-v3-residual",
                        "model": model_name,
                        "fault": label,
                        "model_mae": m["mae"],
                        "persistence_mae": pers["mae"],
                        "mae_gap_vs_persistence": m["mae"] - pers["mae"],
                        **rstats,
                    }
                )
            complexity_rows.append(
                {
                    "model": model_name,
                    "parameter_count": param_count,
                    "training_time_sec": train_extra["training_time_sec"],
                    "clean_inference_time_sec": clean_inference_time,
                    "best_epoch": train_extra["best_epoch"],
                    "best_val_loss": train_extra["best_val_loss"],
                }
            )
        except Exception as exc:
            failed_rows.append({"model": model_name, "error": repr(exc)})

    for row in metrics_rows:
        clean = clean_by_model.get(row["model"], math.nan)
        persistence_clean = clean_by_model["Persistence"]
        persistence_fault = persistence_metrics_by_fault[row["fault"]]["mae"]
        model_rdr = (row["mae"] - clean) / clean if clean and math.isfinite(clean) else math.nan
        persistence_rdr = (persistence_fault - persistence_clean) / persistence_clean
        rdr_rows.append(
            {
                "dataset": row["dataset"],
                "run_id": row["run_id"],
                "model": row["model"],
                "fault": row["fault"],
                "clean_mae": clean,
                "fault_mae": row["mae"],
                "rdr_mae": model_rdr,
                "persistence_fault_mae": persistence_fault,
                "persistence_rdr_mae": persistence_rdr,
                "mae_gap_vs_persistence": row["mae"] - persistence_fault,
                "rdr_gap_vs_persistence": model_rdr - persistence_rdr if math.isfinite(model_rdr) else math.nan,
            }
        )

    write_csv(out_dir / "metrics_by_model_fault.csv", metrics_rows)
    write_csv(out_dir / "horizon_metrics.csv", horizon_rows)
    write_csv(out_dir / "robustness_rdr.csv", rdr_rows)
    write_csv(out_dir / "complexity_metrics.csv", complexity_rows)
    write_csv(out_dir / "residual_diagnostics.csv", residual_rows)
    write_csv(out_dir / "prediction_distribution.csv", distribution_rows)
    write_csv(out_dir / "training_curves.csv", curve_rows)
    write_csv(out_dir / "failed_runs.csv", failed_rows)

    return {
        "status": "completed",
        "output_dir": str(out_dir),
        "metrics_rows": len(metrics_rows),
        "failed_runs": failed_rows,
    }


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
