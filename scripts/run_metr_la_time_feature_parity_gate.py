"""Run METR-LA time-feature parity check before formal-v4 ablation."""

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
    {"fault": "clean", "label": "clean"},
    {"fault": "random_missing", "rate": 0.40, "label": "random_missing_40"},
    {"fault": "continuous_outage", "length": 24, "label": "continuous_outage_24"},
    {"fault": "gaussian_noise", "severity": "high", "label": "gaussian_noise_high"},
    {"fault": "linear_drift", "severity": "high", "label": "linear_drift_high"},
    {"fault": "stuck_at_last_value", "severity": "high", "label": "stuck_at_last_value_high"},
]

MODEL_SPECS = [
    {
        "model": "ResidualGRU-corruption-aware",
        "uses_time_features": False,
        "family": "residual",
        "mode": "train",
        "source": "trained_in_this_gate",
    },
    {
        "model": "ResidualGRU-time-corruption-aware",
        "uses_time_features": True,
        "family": "residual",
        "mode": "train",
        "source": "trained_in_this_gate",
    },
    {
        "model": "SRAF-ResidualGRU-full_train_h32",
        "uses_time_features": False,
        "family": "sraf",
        "mode": "load",
        "checkpoint": "experiments/metr-la-formal-v35-polish/configs/full_train_h32/best_checkpoint.pt",
        "source": "reused_v35_full_train_h32_checkpoint",
    },
    {
        "model": "SRAF-ResidualGRU-time-h32",
        "uses_time_features": True,
        "family": "sraf",
        "mode": "load",
        "checkpoint": "experiments/metr-la-formal-v36-final-polish/configs/time_of_day_features_h32/best_checkpoint.pt",
        "source": "reused_v36_time_of_day_features_h32_checkpoint",
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/metr-la")
    parser.add_argument("--output-dir", default="experiments/metr-la-time-feature-parity")
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


def train_corruption_batch(x: np.ndarray, seed: int, step: int) -> np.ndarray:
    if (seed + step) % 2 == 0:
        return x
    choices = [s for s in FAULT_SETTINGS if s["fault"] != "clean"]
    setting = choices[(seed + step) % len(choices)]
    corrupted, _, _ = apply_fault(x, setting, seed + step, train_std=1.0)
    return corrupted


def model_param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def make_model(spec: dict[str, Any], sensors: int, horizon: int, args: argparse.Namespace) -> nn.Module:
    features = 3 if spec["uses_time_features"] else 1
    if spec["family"] == "residual":
        return ResidualGRU(
            sensors=sensors,
            features=features,
            output_features=1,
            horizon=horizon,
            hidden_dim=args.hidden_dim,
            sensor_embedding_dim=args.sensor_embedding_dim,
        )
    return SRAFResidualGRU(
        sensors=sensors,
        features=features,
        output_features=1,
        horizon=horizon,
        hidden_dim=args.hidden_dim,
        sensor_embedding_dim=args.sensor_embedding_dim,
    )


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
            "run_id": "time-feature-parity",
            "model": spec["model"],
            "epoch": epoch,
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


def load_reused_training_meta(model_name: str) -> dict[str, float]:
    if model_name == "SRAF-ResidualGRU-full_train_h32":
        path = ROOT / "experiments/metr-la-formal-v35-polish/complexity_metrics.csv"
        key = "full_train_h32"
    else:
        path = ROOT / "experiments/metr-la-formal-v36-final-polish/complexity_metrics.csv"
        key = "time_of_day_features_h32"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["config"] == key:
                    return {
                        "training_time_sec": float(row["training_time_sec"]),
                        "best_epoch": float(row["best_epoch"]),
                        "best_val_loss": float(row["best_val_loss"]),
                    }
    return {"training_time_sec": 0.0, "best_epoch": 0.0, "best_val_loss": 0.0}


def predict_model(model: nn.Module, x: np.ndarray, batch_size: int, adjacency: torch.Tensor) -> tuple[np.ndarray, float]:
    model.eval()
    preds = []
    start = perf_counter()
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[i : i + batch_size].astype(np.float32))
            preds.append(model(xb, adjacency=adjacency).cpu().numpy())
    return np.concatenate(preds, axis=0), perf_counter() - start


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

    train_x_time = add_time_of_day_features(train_x_base, 0)
    val_x_time = add_time_of_day_features(val_x_base, train_x_base.shape[0])
    test_x_time_clean = add_time_of_day_features(test_x_base, train_x_base.shape[0] + val_x_base.shape[0])

    fault_dir = out_dir / "fault_masks"
    fault_dir.mkdir(exist_ok=True)
    fault_base: dict[str, np.ndarray] = {}
    fault_time: dict[str, np.ndarray] = {}
    for idx, setting in enumerate(FAULT_SETTINGS):
        cx, mask, meta = apply_fault(test_x_base, setting, seed=args.seed + idx, train_std=1.0)
        label = setting["label"]
        fault_base[label] = cx
        fault_time[label] = test_x_time_clean if label == "clean" else add_time_of_day_features(cx, train_x_base.shape[0] + val_x_base.shape[0])
        np.savez_compressed(fault_dir / f"{label}_mask.npz", mask=mask)
        (fault_dir / f"{label}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    manifest = {
        "run_id": "metr-la-time-feature-parity",
        "gate": "TIME_FEATURE_PARITY_CHECK_GATE",
        "metrics_scale": "original_scale_after_inverse_transform",
        "seed": args.seed,
        "data_dir": str(data_dir),
        "output_dir": str(out_dir),
        "train_samples_used": int(train_x_base.shape[0]),
        "val_samples_used": int(val_x_base.shape[0]),
        "test_samples_used": int(test_x_base.shape[0]),
        "time_feature_construction": "For sample window index s and input step l, use sin(2*pi*(s+l+split_start)/288) and cos(...). Features are input-window only; target Y is unchanged.",
        "target_leakage_check": "No future horizon features are used; only input-window time indices are appended.",
        "models": MODEL_SPECS,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    metrics_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    rdr_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    clean_by_model: dict[str, float] = {}

    specs = MODEL_SPECS if not args.smoke else MODEL_SPECS[:2]
    if args.smoke:
        failed_rows.extend({"model": spec["model"], "status": "skipped", "reason": "smoke mode"} for spec in MODEL_SPECS[2:])

    for spec in specs:
        run_dir = out_dir / "models" / spec["model"]
        run_dir.mkdir(parents=True, exist_ok=True)
        model = make_model(spec, train_x_base.shape[2], train_y.shape[1], args)
        train_x = train_x_time if spec["uses_time_features"] else train_x_base
        val_x = val_x_time if spec["uses_time_features"] else val_x_base
        eval_faults = fault_time if spec["uses_time_features"] else fault_base
        train_extra = {"training_time_sec": 0.0, "best_epoch": 0.0, "best_val_loss": 0.0}
        try:
            if spec["mode"] == "load":
                checkpoint = ROOT / spec["checkpoint"]
                if not checkpoint.exists():
                    raise FileNotFoundError(str(checkpoint))
                model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
                train_extra = load_reused_training_meta(spec["model"])
                torch.save(model.state_dict(), run_dir / "best_checkpoint.pt")
                (run_dir / "train_log.txt").write_text(f"reused_checkpoint: {checkpoint}\n", encoding="utf-8")
            else:
                train_extra, curves = train_model(spec, model, train_x, train_y, val_x, val_y, args, run_dir, adjacency)
                curve_rows.extend(curves)
            param_count = model_param_count(model)
            for setting in FAULT_SETTINGS:
                label = setting["label"]
                pred, inference_time = predict_model(model, eval_faults[label], args.batch_size, adjacency)
                m = safe_metrics(test_y, pred, mean, std)
                if label == "clean":
                    clean_by_model[spec["model"]] = m["mae"]
                    np.savez_compressed(run_dir / "clean_predictions.npz", y_pred=pred, y_true=test_y)
                row = {
                    "dataset": "METR-LA",
                    "run_id": "time-feature-parity",
                    "metrics_scale": "original",
                    "model": spec["model"],
                    "family": spec["family"],
                    "uses_time_features": spec["uses_time_features"],
                    "source": spec["source"],
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
                        "run_id": "time-feature-parity",
                        "model": spec["model"],
                        "fault": label,
                        "mae_15min_h3": row["mae_h3"],
                        "mae_30min_h6": row["mae_h6"],
                        "mae_60min_h12": row["mae_h12"],
                    }
                )
            complexity_rows.append(
                {
                    "model": spec["model"],
                    "family": spec["family"],
                    "uses_time_features": spec["uses_time_features"],
                    "parameter_count": param_count,
                    "training_time_sec": train_extra["training_time_sec"],
                    "clean_inference_time_sec": next(
                        row["inference_time_sec"]
                        for row in metrics_rows
                        if row["model"] == spec["model"] and row["fault"] == "clean"
                    ),
                    "best_epoch": train_extra["best_epoch"],
                    "best_val_loss": train_extra["best_val_loss"],
                }
            )
        except Exception as exc:
            failed_rows.append({"model": spec["model"], "status": "failed", "reason": repr(exc)})

    for row in metrics_rows:
        clean = clean_by_model[row["model"]]
        rdr_rows.append(
            {
                "dataset": row["dataset"],
                "run_id": row["run_id"],
                "model": row["model"],
                "fault": row["fault"],
                "clean_mae": clean,
                "fault_mae": row["mae"],
                "rdr_mae": (row["mae"] - clean) / clean,
            }
        )

    write_csv(out_dir / "metrics_by_model_fault.csv", metrics_rows)
    write_csv(out_dir / "horizon_metrics.csv", horizon_rows)
    write_csv(out_dir / "robustness_rdr.csv", rdr_rows)
    write_csv(out_dir / "complexity_metrics.csv", complexity_rows)
    write_csv(out_dir / "training_curves.csv", curve_rows)
    write_csv(out_dir / "failed_or_skipped_runs.csv", failed_rows)

    summary = build_summary(metrics_rows, rdr_rows, failed_rows)
    (out_dir / "time_feature_parity_summary.md").write_text(summary, encoding="utf-8")
    return {"status": "completed", "output_dir": str(out_dir), "metrics_rows": len(metrics_rows), "failed_or_skipped": failed_rows}


def value(rows: list[dict[str, Any]], model: str, fault: str, key: str) -> float:
    return float(next(row[key] for row in rows if row["model"] == model and row["fault"] == fault))


def build_summary(metrics_rows: list[dict[str, Any]], rdr_rows: list[dict[str, Any]], failed_rows: list[dict[str, Any]]) -> str:
    clean = {row["model"]: row["mae"] for row in metrics_rows if row["fault"] == "clean"}
    residual_gain = clean.get("ResidualGRU-corruption-aware", math.nan) - clean.get("ResidualGRU-time-corruption-aware", math.nan)
    sraf_gap = clean.get("SRAF-ResidualGRU-time-h32", math.nan) - clean.get("ResidualGRU-time-corruption-aware", math.nan)
    faults = [setting["label"] for setting in FAULT_SETTINGS if setting["label"] != "clean"]
    sraf_better = 0
    robustness_lines = []
    for fault in faults:
        sraf = value(rdr_rows, "SRAF-ResidualGRU-time-h32", fault, "rdr_mae") if any(r["model"] == "SRAF-ResidualGRU-time-h32" and r["fault"] == fault for r in rdr_rows) else math.nan
        residual = value(rdr_rows, "ResidualGRU-time-corruption-aware", fault, "rdr_mae") if any(r["model"] == "ResidualGRU-time-corruption-aware" and r["fault"] == fault for r in rdr_rows) else math.nan
        if math.isfinite(sraf) and math.isfinite(residual) and sraf < residual:
            sraf_better += 1
        robustness_lines.append(f"- `{fault}`: SRAF-time RDR {sraf:.6f}, ResidualGRU-time RDR {residual:.6f}")
    recommendation = "B. SRAF with time features"
    if math.isfinite(sraf_gap) and sraf_gap > 0.25 and sraf_better < 3:
        recommendation = "C. both as reported variants"
    return "\n".join(
        [
            "# Time Feature Parity Summary",
            "",
            "Time index construction: for sample window index `s` and input step `l`, use `sin(2*pi*(s+l+split_start)/288)` and `cos(...)`.",
            "Only input-window time features are appended. Target `Y` is not modified, and no horizon/future target features are used.",
            "",
            "Clean MAE:",
            *[f"- `{model}`: {mae:.6f}" for model, mae in sorted(clean.items(), key=lambda item: item[1])],
            "",
            f"ResidualGRU clean MAE improvement from time features: {residual_gain:.6f}",
            f"SRAF-time clean MAE gap vs ResidualGRU-time: {sraf_gap:.6f}",
            "",
            "High-severity RDR comparison:",
            *robustness_lines,
            "",
            f"SRAF-time has lower RDR than ResidualGRU-time on {sraf_better} of {len(faults)} high-severity faults.",
            "",
            f"Recommended formal-v4 reporting choice: {recommendation}.",
            "No paper conclusions are written from this gate.",
            "",
            f"Failed/skipped rows: {len(failed_rows)}",
        ]
    )


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
