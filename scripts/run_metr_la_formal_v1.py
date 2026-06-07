"""Run METR-LA formal-v1 preliminary experiments.

This script writes preliminary real-data evidence only. It does not draft paper
claims and does not use synthetic smoke outputs.
"""

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
from src.models.sraf import SRAFModel  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


FAULT_SETTINGS = [
    {"fault": "clean", "setting": "clean", "label": "clean"},
    {"fault": "random_missing", "rate": 0.10, "label": "random_missing_10"},
    {"fault": "random_missing", "rate": 0.20, "label": "random_missing_20"},
    {"fault": "random_missing", "rate": 0.40, "label": "random_missing_40"},
    {"fault": "continuous_outage", "length": 6, "label": "continuous_outage_6"},
    {"fault": "continuous_outage", "length": 12, "label": "continuous_outage_12"},
    {"fault": "continuous_outage", "length": 24, "label": "continuous_outage_24"},
    {"fault": "gaussian_noise", "severity": "low", "label": "gaussian_noise_low"},
    {"fault": "gaussian_noise", "severity": "medium", "label": "gaussian_noise_medium"},
    {"fault": "gaussian_noise", "severity": "high", "label": "gaussian_noise_high"},
    {"fault": "linear_drift", "severity": "low", "label": "linear_drift_low"},
    {"fault": "linear_drift", "severity": "medium", "label": "linear_drift_medium"},
    {"fault": "linear_drift", "severity": "high", "label": "linear_drift_high"},
    {"fault": "stuck_at_last_value", "severity": "low", "label": "stuck_at_last_value_low"},
    {"fault": "stuck_at_last_value", "severity": "medium", "label": "stuck_at_last_value_medium"},
    {"fault": "stuck_at_last_value", "severity": "high", "label": "stuck_at_last_value_high"},
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/metr-la")
    parser.add_argument("--output-dir", default="experiments/metr-la-formal-v1")
    parser.add_argument("--tables-dir", default="paper/tables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--val-samples", type=int, default=1024)
    parser.add_argument("--test-samples", type=int, default=0, help="0 means full test set.")
    return parser


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


class GRUForecast(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.head(hidden[-1])


class TCNForecast(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.net(x.transpose(1, 2))[..., : x.shape[1]]
        return self.head(features.mean(dim=-1))


def model_param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def make_model(model_name: str, x_shape: tuple[int, ...], horizon: int, hidden_dim: int) -> nn.Module:
    _, length, sensors, features = x_shape
    flat_dim = sensors * features
    output_dim = horizon * flat_dim
    if model_name.startswith("GRU"):
        return GRUForecast(flat_dim, hidden_dim, output_dim)
    if model_name.startswith("TCN"):
        return TCNForecast(flat_dim, hidden_dim, output_dim)
    if model_name.startswith("SRAF-GRU"):
        return SRAFModel(sensors=sensors, features=features, horizon=horizon, hidden_dim=hidden_dim, backbone="GRU")
    raise ValueError(f"Unsupported model: {model_name}")


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


def corruption_aware_batch(x: np.ndarray, seed: int, step: int, train_std: float) -> np.ndarray:
    choices = [s for s in FAULT_SETTINGS if s["fault"] != "clean"]
    setting = choices[(seed + step) % len(choices)]
    corrupted, _, _ = apply_fault(x, setting, seed + step, train_std)
    return corrupted


def iter_batches(x: np.ndarray, y: np.ndarray, batch_size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(x[i : i + batch_size], y[i : i + batch_size]) for i in range(0, x.shape[0], batch_size)]


def train_model(
    model_name: str,
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    args: argparse.Namespace,
    run_dir: Path,
    corruption_aware: bool,
    train_std: float,
) -> dict[str, float]:
    torch.manual_seed(args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    loss_fn = nn.MSELoss()
    logs: list[str] = []
    start = perf_counter()
    batch_step = 0
    batches = iter_batches(train_x, train_y, args.batch_size)
    for epoch in range(args.epochs):
        model.train()
        losses: list[float] = []
        for xb, yb in batches:
            if corruption_aware:
                xb = corruption_aware_batch(xb, args.seed, batch_step, train_std)
            xb = np.nan_to_num(xb, nan=0.0).astype(np.float32)
            x_t = torch.from_numpy(xb)
            y_t = torch.from_numpy(yb.astype(np.float32))
            optimizer.zero_grad()
            if model_name.startswith("SRAF"):
                adjacency = torch.from_numpy(np.load(Path(args.data_dir) / "adjacency.npy").astype(np.float32))
                pred = model(x_t, adjacency=adjacency)
            else:
                batch, length, sensors, features = xb.shape
                pred = model(x_t.reshape(batch, length, sensors * features)).reshape(yb.shape)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            batch_step += 1
        logs.append(f"epoch={epoch + 1},loss={sum(losses) / max(len(losses), 1):.6f}")
    training_time = perf_counter() - start
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_log.txt").write_text("\n".join(logs), encoding="utf-8")
    return {"training_time_sec": training_time}


def predict_model(
    model_name: str,
    model: nn.Module,
    x: np.ndarray,
    batch_size: int,
    adjacency_path: Path,
) -> tuple[np.ndarray, float]:
    model.eval()
    preds = []
    start = perf_counter()
    adjacency = torch.from_numpy(np.load(adjacency_path).astype(np.float32)) if model_name.startswith("SRAF") else None
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = np.nan_to_num(x[i : i + batch_size], nan=0.0).astype(np.float32)
            x_t = torch.from_numpy(xb)
            if model_name.startswith("SRAF"):
                pred = model(x_t, adjacency=adjacency).cpu().numpy()
            else:
                batch, length, sensors, features = xb.shape
                pred = model(x_t.reshape(batch, length, sensors * features)).cpu().numpy().reshape(
                    batch, -1, sensors, features
                )
            preds.append(pred)
    return np.concatenate(preds, axis=0), perf_counter() - start


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    train_x, train_y = load_split(data_dir, "train", args.train_samples)
    val_x, val_y = load_split(data_dir, "val", args.val_samples)
    test_limit = None if args.test_samples == 0 else args.test_samples
    test_x, test_y = load_split(data_dir, "test", test_limit)
    mean, std = load_scale(data_dir)
    train_std_norm = 1.0
    manifest = {
        "run_id": "metr-la-formal-v1",
        "status": "formal_v1_preliminary",
        "metrics_scale": "original_scale_after_inverse_transform",
        "seed": args.seed,
        "data_dir": str(data_dir),
        "output_dir": str(out_dir),
        "train_samples_used": int(train_x.shape[0]),
        "val_samples_used": int(val_x.shape[0]),
        "test_samples_used": int(test_x.shape[0]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "note": "Formal-v1 preliminary uses reduced train/val samples for trainable models; test evaluation uses configured test sample count.",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "config_resolved.yaml").write_text("\n".join(f"{k}: {v}" for k, v in manifest.items()), encoding="utf-8")

    fault_dir = out_dir / "fault_masks"
    fault_dir.mkdir(exist_ok=True)
    corrupted_test: dict[str, np.ndarray] = {}
    for idx, setting in enumerate(FAULT_SETTINGS):
        cx, mask, meta = apply_fault(test_x, setting, args.seed + idx, train_std_norm)
        label = setting["label"]
        corrupted_test[label] = cx
        np.savez_compressed(fault_dir / f"{label}_mask.npz", mask=mask)
        (fault_dir / f"{label}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    model_specs = [
        {"name": "HistoricalAverage", "train": "none"},
        {"name": "Persistence", "train": "none"},
        {"name": "GRU-clean", "train": "clean"},
        {"name": "TCN-clean", "train": "clean"},
        {"name": "GRU-corruption-aware", "train": "corruption_aware"},
        {"name": "TCN-corruption-aware", "train": "corruption_aware"},
        {"name": "SRAF-GRU-corruption-aware", "train": "corruption_aware"},
    ]
    metrics_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    clean_by_model: dict[str, float] = {}

    for spec in model_specs:
        model_name = spec["name"]
        run_dir = out_dir / "models" / model_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config_resolved.yaml").write_text(
            "\n".join([f"model: {model_name}", f"train_protocol: {spec['train']}", f"seed: {args.seed}"]),
            encoding="utf-8",
        )
        try:
            train_extra: dict[str, float] = {}
            model: nn.Module | None = None
            if spec["train"] != "none":
                base_name = "SRAF-GRU" if model_name.startswith("SRAF") else model_name.split("-")[0]
                model = make_model(base_name, train_x.shape, train_y.shape[1], args.hidden_dim)
                train_extra = train_model(
                    base_name,
                    model,
                    train_x,
                    train_y,
                    args,
                    run_dir,
                    corruption_aware=spec["train"] == "corruption_aware",
                    train_std=train_std_norm,
                )
                complexity_rows.append(
                    {
                        "model": model_name,
                        "parameter_count": model_param_count(model),
                        "training_time_sec": train_extra["training_time_sec"],
                    }
                )
            else:
                (run_dir / "train_log.txt").write_text("no neural training required\n", encoding="utf-8")

            for setting in FAULT_SETTINGS:
                label = setting["label"]
                x_eval = corrupted_test[label]
                if model_name == "HistoricalAverage":
                    y_pred = historical_average_predict(x_eval, test_y.shape[1])
                    inference_time = math.nan
                    parameter_count = math.nan
                elif model_name == "Persistence":
                    y_pred = persistence_predict(np.nan_to_num(x_eval, nan=0.0), test_y.shape[1])
                    inference_time = math.nan
                    parameter_count = math.nan
                else:
                    assert model is not None
                    base_name = "SRAF-GRU" if model_name.startswith("SRAF") else model_name.split("-")[0]
                    y_pred, inference_time = predict_model(base_name, model, x_eval, args.batch_size, data_dir / "adjacency.npy")
                    parameter_count = model_param_count(model)
                m = safe_metrics(test_y, y_pred, mean, std)
                row = {
                    "dataset": "METR-LA",
                    "run_id": "formal-v1",
                    "metrics_scale": "original",
                    "model": model_name,
                    "fault": label,
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "mape": m["mape"],
                    "mae_h3": m.get("mae_h3", math.nan),
                    "mae_h6": m.get("mae_h6", math.nan),
                    "mae_h12": m.get("mae_h12", math.nan),
                    "parameter_count": parameter_count,
                    "inference_time_sec": inference_time,
                }
                metrics_rows.append(row)
                horizon_rows.append(
                    {
                        "dataset": "METR-LA",
                        "run_id": "formal-v1",
                        "model": model_name,
                        "fault": label,
                        "mae_15min_h3": row["mae_h3"],
                        "mae_30min_h6": row["mae_h6"],
                        "mae_60min_h12": row["mae_h12"],
                    }
                )
                if label == "clean":
                    clean_by_model[model_name] = row["mae"]
            # Save clean predictions only to keep storage feasible.
            clean_x = corrupted_test["clean"]
            if model_name == "HistoricalAverage":
                clean_pred = historical_average_predict(clean_x, test_y.shape[1])
            elif model_name == "Persistence":
                clean_pred = persistence_predict(clean_x, test_y.shape[1])
            else:
                assert model is not None
                base_name = "SRAF-GRU" if model_name.startswith("SRAF") else model_name.split("-")[0]
                clean_pred, _ = predict_model(base_name, model, clean_x, args.batch_size, data_dir / "adjacency.npy")
            np.savez_compressed(run_dir / "clean_predictions.npz", y_pred=clean_pred, y_true=test_y)
        except Exception as exc:  # keep gate honest
            failed_rows.append({"model": model_name, "error": repr(exc)})

    rdr_rows: list[dict[str, Any]] = []
    for row in metrics_rows:
        clean = clean_by_model.get(row["model"], math.nan)
        rdr = (row["mae"] - clean) / clean if clean and not math.isnan(clean) else math.nan
        rdr_rows.append(
            {
                "dataset": row["dataset"],
                "run_id": row["run_id"],
                "model": row["model"],
                "fault": row["fault"],
                "clean_mae": clean,
                "fault_mae": row["mae"],
                "rdr_mae": rdr,
            }
        )

    # Add inference summaries for trainable models into complexity rows.
    for c_row in complexity_rows:
        model_metrics = [r for r in metrics_rows if r["model"] == c_row["model"] and r["fault"] == "clean"]
        c_row["clean_inference_time_sec"] = model_metrics[0]["inference_time_sec"] if model_metrics else math.nan

    write_csv(out_dir / "metrics_by_model_fault.csv", metrics_rows)
    write_csv(out_dir / "horizon_metrics.csv", horizon_rows)
    write_csv(out_dir / "robustness_rdr.csv", rdr_rows)
    write_csv(out_dir / "complexity_metrics.csv", complexity_rows)
    write_csv(out_dir / "failed_runs.csv", failed_rows)

    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    write_csv(tables_dir / "table_metr_la_formal_v1_main.csv", metrics_rows)
    write_csv(tables_dir / "table_metr_la_formal_v1_rdr.csv", rdr_rows)
    write_csv(tables_dir / "table_metr_la_formal_v1_horizon.csv", horizon_rows)
    write_csv(tables_dir / "table_metr_la_formal_v1_complexity.csv", complexity_rows)

    return {
        "status": "completed",
        "output_dir": str(out_dir),
        "metrics_rows": len(metrics_rows),
        "failed_runs": failed_rows,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
