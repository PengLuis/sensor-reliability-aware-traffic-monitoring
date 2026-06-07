"""Run STRONG_CLEAN_BACKBONE_INTEGRATION_GATE on METR-LA."""

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
from src.models.strong_backbones import (  # noqa: E402
    OfficialStyleSTID,
    SRAFBackboneWrapper,
    STIDBackbone,
    TCNTimeStrongBackbone,
)


FAULT_SETTINGS = [
    {"fault": "clean", "label": "clean", "severity_group": "clean"},
    {"fault": "random_missing", "rate": 0.20, "label": "random_missing_20", "severity_group": "medium"},
    {"fault": "random_missing", "rate": 0.40, "label": "random_missing_40", "severity_group": "high"},
    {"fault": "continuous_outage", "length": 24, "label": "continuous_outage_24", "severity_group": "high"},
    {"fault": "gaussian_noise", "severity": "high", "label": "gaussian_noise_high", "severity_group": "high"},
    {"fault": "linear_drift", "severity": "high", "label": "linear_drift_high", "severity_group": "high"},
    {"fault": "stuck_at_last_value", "severity": "high", "label": "stuck_at_last_value_high", "severity_group": "high"},
]

LITERATURE_HINT = {
    "h3_mae_range": "2.6-3.6",
    "h6_mae_range": "2.6-3.6",
    "h12_mae_range": "2.6-3.6",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/metr-la")
    parser.add_argument("--output-dir", default="experiments/metr-la-strong-clean-backbone-integration")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--sensor-embedding-dim", type=int, default=16)
    parser.add_argument("--stid-hidden-dim", type=int, default=128)
    parser.add_argument("--tcn-hidden-dim", type=int, default=96)
    parser.add_argument("--repair-hidden-dim", type=int, default=32)
    parser.add_argument("--lambda-repair", type=float, default=0.05)
    parser.add_argument("--lambda-rel", type=float, default=0.01)
    parser.add_argument("--model-set", choices=["default", "official_stid_precheck"], default="default")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--loss", choices=["mse", "mae"], default="mse")
    parser.add_argument("--smoke", action="store_true")
    return parser


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if path.name == "failed_or_skipped_models.csv":
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


def add_stid_identity_features(x: np.ndarray, start_index: int = 0) -> np.ndarray:
    samples, length, sensors, _ = x.shape
    offsets = np.arange(samples, dtype=np.int64)[:, None] + np.arange(length, dtype=np.int64)[None, :] + int(start_index)
    tod_index = offsets % 288
    dow_index = (offsets // 288) % 7
    tod = (tod_index.astype(np.float32) / 288.0)[:, :, None, None].repeat(sensors, axis=2)
    dow = (dow_index.astype(np.float32) / 7.0)[:, :, None, None].repeat(sensors, axis=2)
    return np.concatenate([x[..., :1], tod, dow], axis=-1).astype(np.float32)


def stid_time_feature_audit(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    train_start: int,
    val_start: int,
    test_start: int,
) -> dict[str, Any]:
    def split_audit(name: str, x: np.ndarray, start: int) -> dict[str, Any]:
        samples, length, _, _ = x.shape
        offsets = np.arange(samples, dtype=np.int64)[:, None] + np.arange(length, dtype=np.int64)[None, :] + int(start)
        tod = (offsets % 288).reshape(-1)
        dow = ((offsets // 288) % 7).reshape(-1)
        return {
            "split": name,
            "start_index": int(start),
            "sample_count": int(samples),
            "tod_min": int(tod.min()) if tod.size else None,
            "tod_max": int(tod.max()) if tod.size else None,
            "tod_unique_count": int(np.unique(tod).size),
            "first_50_tod_index_values": [int(v) for v in tod[:50]],
            "dow_min": int(dow.min()) if dow.size else None,
            "dow_max": int(dow.max()) if dow.size else None,
            "dow_unique_count": int(np.unique(dow).size),
            "first_50_dow_index_values": [int(v) for v in dow[:50]],
        }

    return {
        "time_feature_type": "official_style_stid_identity_features",
        "train_start_index": int(train_start),
        "val_start_index": int(val_start),
        "test_start_index": int(test_start),
        "splits": [
            split_audit("train", train_x, train_start),
            split_audit("val", val_x, val_start),
            split_audit("test", test_x, test_start),
        ],
        "real_timestamp_metadata_exists": False,
        "note": "Day-of-week identity is inferred from contiguous 5-minute sample indices, not from real timestamp metadata.",
    }


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false.")
    return torch.device(device_arg)


def make_loss(loss_name: str) -> nn.Module:
    if loss_name == "mae":
        return nn.L1Loss()
    if loss_name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unknown loss: {loss_name}")


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


def model_param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def predict_model(
    model: nn.Module,
    x: np.ndarray,
    batch_size: int,
    adjacency: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    model.eval()
    preds: list[np.ndarray] = []
    start = perf_counter()
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[i : i + batch_size].astype(np.float32)).to(device)
            preds.append(model(xb, adjacency=adjacency).detach().cpu().numpy())
    return np.concatenate(preds, axis=0), perf_counter() - start


def evaluate_loss(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    adjacency: torch.Tensor,
    device: torch.device,
    loss_fn: nn.Module,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for xb, yb in iter_batches(x, y, batch_size, shuffle=False, seed=0, epoch=0):
            xb_t = torch.from_numpy(xb.astype(np.float32)).to(device)
            yb_t = torch.from_numpy(yb.astype(np.float32)).to(device)
            pred = model(xb_t, adjacency=adjacency)
            losses.append(float(loss_fn(pred, yb_t).detach().cpu()))
    return float(np.mean(losses))


def train_clean_model(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    args: argparse.Namespace,
    run_dir: Path,
    adjacency: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
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
        epoch_losses: list[float] = []
        for xb, yb in iter_batches(train_x, train_y, args.batch_size, shuffle=True, seed=args.seed, epoch=epoch):
            xb_t = torch.from_numpy(xb.astype(np.float32)).to(device)
            yb_t = torch.from_numpy(yb.astype(np.float32)).to(device)
            pred = model(xb_t, adjacency=adjacency)
            loss = loss_fn(pred, yb_t)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(epoch_losses))
        val_loss = evaluate_loss(model, val_x, val_y, args.batch_size, adjacency, device, loss_fn)
        scheduler.step(val_loss)
        lr = float(optimizer.param_groups[0]["lr"])
        improved = val_loss < best_val - 1.0e-6
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val,
                "learning_rate": lr,
                "improved": improved,
                "early_stop_triggered": False,
            }
        )
        print(f"{run_dir.name} epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={lr:.6g}", flush=True)
        if no_improve >= args.patience:
            rows[-1]["early_stop_triggered"] = True
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, run_dir / "best_checkpoint.pt")
    (run_dir / "train_log.txt").write_text(
        "\n".join(
            f"epoch={r['epoch']},train_loss={r['train_loss']:.6f},val_loss={r['val_loss']:.6f},"
            f"best_val_loss={r['best_val_loss']:.6f},learning_rate={r['learning_rate']:.8f},early_stop_triggered={r['early_stop_triggered']}"
            for r in rows
        ),
        encoding="utf-8",
    )
    return {"training_time_sec": perf_counter() - start, "best_epoch": best_epoch, "best_val_loss": best_val}, rows


def corruption_aware_batch(x: np.ndarray, setting_idx: int, seed: int, train_std: float) -> tuple[np.ndarray, np.ndarray]:
    choices = [s for s in FAULT_SETTINGS if s["fault"] != "clean"]
    setting = choices[setting_idx % len(choices)]
    speed_corrupt, mask, _ = apply_fault(x[..., :1], setting, seed, train_std)
    x_corrupt = x.copy()
    x_corrupt[..., :1] = speed_corrupt
    return x_corrupt, mask.astype(np.float32)


def train_sraf_integrated_model(
    model: SRAFBackboneWrapper,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    args: argparse.Namespace,
    run_dir: Path,
    adjacency: torch.Tensor,
    train_std: float,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    loss_fn = make_loss(args.loss)
    best_val = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0
    step = 0
    rows: list[dict[str, Any]] = []
    start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        forecast_losses: list[float] = []
        repair_losses: list[float] = []
        rel_losses: list[float] = []
        total_losses: list[float] = []
        for xb, yb in iter_batches(train_x, train_y, args.batch_size, shuffle=True, seed=args.seed, epoch=epoch):
            x_corrupt, mask = corruption_aware_batch(xb, setting_idx=step, seed=args.seed + step, train_std=train_std)
            xb_t = torch.from_numpy(x_corrupt.astype(np.float32)).to(device)
            yb_t = torch.from_numpy(yb.astype(np.float32)).to(device)
            mask_t = torch.from_numpy(mask.astype(np.float32)).to(device)
            pred, comps = model(xb_t, adjacency=adjacency, return_components=True)
            forecast = loss_fn(pred, yb_t)

            repaired_speed = comps["repaired_input"][..., :1]
            clean_speed = torch.from_numpy(xb[..., :1].astype(np.float32)).to(device)
            denom = mask_t.sum().clamp_min(1.0)
            repair = torch.sum(torch.abs(repaired_speed - clean_speed) * mask_t) / denom

            reliability = comps["reliability"][..., :1]
            target_rel = 1.0 - mask_t
            rel = torch.mean((reliability - target_rel) ** 2)

            total = forecast + args.lambda_repair * repair + args.lambda_rel * rel
            optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            forecast_losses.append(float(forecast.detach().cpu()))
            repair_losses.append(float(repair.detach().cpu()))
            rel_losses.append(float(rel.detach().cpu()))
            total_losses.append(float(total.detach().cpu()))
            step += 1

        train_loss = float(np.mean(total_losses))
        val_loss = evaluate_loss(model, val_x, val_y, args.batch_size, adjacency, device, loss_fn)
        scheduler.step(val_loss)
        lr = float(optimizer.param_groups[0]["lr"])
        improved = val_loss < best_val - 1.0e-6
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "forecast_loss": float(np.mean(forecast_losses)),
                "repair_loss": float(np.mean(repair_losses)),
                "reliability_loss": float(np.mean(rel_losses)),
                "val_loss": val_loss,
                "best_val_loss": best_val,
                "learning_rate": lr,
                "improved": improved,
                "early_stop_triggered": False,
            }
        )
        print(
            f"{run_dir.name} epoch={epoch} train={train_loss:.6f} forecast={np.mean(forecast_losses):.6f} "
            f"repair={np.mean(repair_losses):.6f} rel={np.mean(rel_losses):.6f} val={val_loss:.6f}",
            flush=True,
        )
        if no_improve >= args.patience:
            rows[-1]["early_stop_triggered"] = True
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, run_dir / "best_checkpoint.pt")
    (run_dir / "train_log.txt").write_text(
        "\n".join(
            f"epoch={r['epoch']},train_loss={r['train_loss']:.6f},forecast_loss={r['forecast_loss']:.6f},"
            f"repair_loss={r['repair_loss']:.6f},reliability_loss={r['reliability_loss']:.6f},"
            f"val_loss={r['val_loss']:.6f},best_val_loss={r['best_val_loss']:.6f},"
            f"learning_rate={r['learning_rate']:.8f},early_stop_triggered={r['early_stop_triggered']}"
            for r in rows
        ),
        encoding="utf-8",
    )
    return {"training_time_sec": perf_counter() - start, "best_epoch": best_epoch, "best_val_loss": best_val}, rows


def build_backbone(name: str, sensors: int, horizon: int, args: argparse.Namespace) -> nn.Module:
    if name == "STID-style-clean-backbone":
        return STIDBackbone(
            sensors=sensors,
            input_length=12,
            input_features=3,
            horizon=horizon,
            hidden_dim=args.stid_hidden_dim,
            sensor_embedding_dim=args.sensor_embedding_dim,
            horizon_embedding_dim=8,
        )
    if name == "OfficialStyleSTID-clean-backbone":
        return OfficialStyleSTID(
            sensors=sensors,
            input_length=12,
            input_dim=3,
            horizon=horizon,
            embed_dim=32,
            node_dim=32,
            temp_dim_tid=32,
            temp_dim_diw=32,
            num_layers=3,
            time_of_day_size=288,
            day_of_week_size=7,
            use_node=True,
            use_time_in_day=True,
            use_day_in_week=True,
            dropout=0.15,
        )
    if name == "TCN-time-strong-clean-backbone":
        return TCNTimeStrongBackbone(
            sensors=sensors,
            input_features=3,
            horizon=horizon,
            hidden_dim=args.tcn_hidden_dim,
            sensor_embedding_dim=args.sensor_embedding_dim,
            horizon_embedding_dim=8,
        )
    raise ValueError(f"Unknown backbone: {name}")


def row_value(rows: list[dict[str, Any]], model: str, fault: str, key: str) -> float:
    return float(next(r[key] for r in rows if r["model"] == model and r["fault"] == fault))


def metric_rows_to_rdr(metrics_rows: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    clean = {row["model"]: float(row["mae"]) for row in metrics_rows if row["fault"] == "clean"}
    rows = []
    for row in metrics_rows:
        clean_mae = clean.get(row["model"], math.nan)
        fault_mae = float(row["mae"])
        rows.append(
            {
                "dataset": "METR-LA",
                "run_id": run_id,
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = (
        "metr-la-official-style-stid-code-repair"
        if args.model_set == "official_stid_precheck"
        else "strong-clean-backbone-integration"
    )

    train_x_base, train_y = load_split(data_dir, "train")
    val_x_base, val_y = load_split(data_dir, "val")
    test_x_base, test_y = load_split(data_dir, "test")
    full_train_count = train_x_base.shape[0]
    full_val_count = val_x_base.shape[0]
    full_test_count = test_x_base.shape[0]
    if args.smoke:
        train_x_base, train_y = train_x_base[:512], train_y[:512]
        val_x_base, val_y = val_x_base[:128], val_y[:128]
        test_x_base, test_y = test_x_base[:256], test_y[:256]
        args.epochs = min(args.epochs, 2)
        args.patience = 1
    elif args.model_set == "official_stid_precheck":
        train_x_base, train_y = train_x_base[:8000], train_y[:8000]
        val_x_base, val_y = val_x_base[:1024], val_y[:1024]

    mean, std = load_scale(data_dir)
    adjacency = torch.from_numpy(np.load(data_dir / "adjacency.npy").astype(np.float32)).to(device)
    train_x = add_time_of_day_features(train_x_base, 0)
    val_x = add_time_of_day_features(val_x_base, full_train_count)
    split_start = full_train_count + full_val_count
    train_x_stid = add_stid_identity_features(train_x_base, 0)
    val_x_stid = add_stid_identity_features(val_x_base, full_train_count)
    if args.model_set == "official_stid_precheck":
        audit_train_x, _ = load_split(data_dir, "train")
        audit_val_x, _ = load_split(data_dir, "val")
        audit_test_x, _ = load_split(data_dir, "test")
        time_audit = stid_time_feature_audit(
            audit_train_x,
            audit_val_x,
            audit_test_x,
            train_start=0,
            val_start=full_train_count,
            test_start=split_start,
        )
        time_audit["status"] = "PASS"
        time_audit["precheck_note"] = "Clean precheck trains on a bounded subset but uses these full chronological split offsets."
        (out_dir / "time_feature_audit.json").write_text(json.dumps(time_audit, indent=2), encoding="utf-8")

    # Shared fault masks for this gate
    fault_dir = out_dir / "fault_masks"
    fault_dir.mkdir(parents=True, exist_ok=True)
    active_fault_settings = [FAULT_SETTINGS[0]] if args.model_set == "official_stid_precheck" else FAULT_SETTINGS
    fault_inputs_time: dict[str, np.ndarray] = {}
    fault_inputs_stid: dict[str, np.ndarray] = {}
    for idx, setting in enumerate(active_fault_settings):
        label = setting["label"]
        cx_speed, mask, meta = apply_fault(test_x_base, setting, seed=args.seed + idx, train_std=std)
        fault_inputs_time[label] = add_time_of_day_features(cx_speed, split_start)
        fault_inputs_stid[label] = add_stid_identity_features(cx_speed, split_start)
        meta = {**setting, **meta, "label": label, "target_corrupted": False, "mask_path": str(fault_dir / f"{label}_mask.npz")}
        np.savez_compressed(fault_dir / f"{label}_mask.npz", mask=mask)
        (fault_dir / f"{label}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Stage 1 clean backbone precheck
    clean_backbone_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    train_curve_rows: list[dict[str, Any]] = []

    # A. current strong residual reference (reuse existing metric)
    strong_existing = pd_read(Path("experiments/metr-la-strong-baseline-audit/metrics_by_model_fault.csv"))
    strong_clean = strong_existing[
        (strong_existing["model"] == "ResidualGRU-time-corruption-aware-strong") & (strong_existing["fault"] == "clean")
    ]
    if strong_clean.empty:
        raise RuntimeError("Missing strong residual clean reference metrics.")
    strong_clean_row = strong_clean.iloc[0]
    clean_backbone_rows.append(
        {
            "model": "Strong ResidualGRU-time reference",
            "stage": "clean_precheck",
            "source": "reused_existing_traceable_metrics",
            "mae": float(strong_clean_row["mae"]),
            "rmse": float(strong_clean_row["rmse"]),
            "mape": float(strong_clean_row["mape"]),
            "h3_mae": float(strong_clean_row["mae_h3"]),
            "h6_mae": float(strong_clean_row["mae_h6"]),
            "h12_mae": float(strong_clean_row["mae_h12"]),
        }
    )

    # B and D candidates
    if args.model_set == "official_stid_precheck":
        candidate_names = ["STID-style-clean-backbone", "OfficialStyleSTID-clean-backbone"]
    else:
        candidate_names = ["STID-style-clean-backbone", "TCN-time-strong-clean-backbone"]
    trained_clean_models: dict[str, nn.Module] = {}
    clean_train_meta: dict[str, dict[str, float]] = {}
    for name in candidate_names:
        run_dir = out_dir / "models" / name
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            model = build_backbone(name, sensors=train_x_base.shape[2], horizon=train_y.shape[1], args=args)
            candidate_train_x = train_x_stid if name == "OfficialStyleSTID-clean-backbone" else train_x
            candidate_val_x = val_x_stid if name == "OfficialStyleSTID-clean-backbone" else val_x
            candidate_fault_inputs = fault_inputs_stid if name == "OfficialStyleSTID-clean-backbone" else fault_inputs_time
            meta, curves = train_clean_model(
                model,
                candidate_train_x,
                train_y,
                candidate_val_x,
                val_y,
                args,
                run_dir,
                adjacency,
                device,
            )
            train_curve_rows.extend([{**c, "model": name, "stage": "clean_precheck"} for c in curves])
            pred_clean, infer_t = predict_model(model, candidate_fault_inputs["clean"], args.batch_size, adjacency, device)
            m = safe_metrics(test_y, pred_clean, mean, std)
            clean_backbone_rows.append(
                {
                    "model": name,
                    "stage": "clean_precheck",
                    "source": "trained_in_this_gate",
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "mape": m["mape"],
                    "h3_mae": m["mae_h3"],
                    "h6_mae": m["mae_h6"],
                    "h12_mae": m["mae_h12"],
                }
            )
            complexity_rows.append(
                {
                    "dataset": "METR-LA",
                    "run_id": run_id,
                    "model": name,
                    "parameter_count": model_param_count(model),
                    "training_time_sec": meta["training_time_sec"],
                    "clean_inference_time_sec": infer_t,
                    "best_epoch": meta["best_epoch"],
                    "best_val_loss": meta["best_val_loss"],
                }
            )
            trained_clean_models[name] = model
            clean_train_meta[name] = meta
            np.savez_compressed(run_dir / "clean_predictions.npz", y_pred=pred_clean, y_true=test_y)
        except Exception as exc:
            failed_rows.append({"model": name, "status": "failed", "reason": repr(exc)})
            print(f"FAILED {name}: {exc!r}", flush=True)

    # C graphwavenet-lite explicitly skipped
    if args.model_set != "official_stid_precheck":
        failed_rows.append(
            {
                "model": "GraphWaveNet-lite-clean-backbone",
                "status": "skipped",
                "reason": "Not implemented in this gate to keep scope lightweight and finish STID/TCN integration first.",
            }
        )

    write_csv(out_dir / "clean_backbone_metrics.csv", clean_backbone_rows)

    # Select strongest clean backbone
    strong_ref_mae = float(strong_clean_row["mae"])
    eligible = [row for row in clean_backbone_rows if row["model"] in trained_clean_models and float(row["mae"]) < strong_ref_mae]
    selected_backbone = min(eligible, key=lambda r: float(r["mae"]))["model"] if eligible else None
    if args.model_set == "official_stid_precheck":
        selected_backbone = None

    # Stage 2 integration and full fault evaluation
    metrics_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []

    # Always evaluate baseline references (from checkpoints) on shared masks
    eval_models: dict[str, tuple[str, Any]] = {}
    eval_models["Persistence"] = ("persistence", None)

    strong_model = ResidualGRU(
        sensors=train_x_base.shape[2],
        features=3,
        output_features=1,
        horizon=train_y.shape[1],
        hidden_dim=32,
        sensor_embedding_dim=8,
    )
    strong_ckpt = ROOT / "experiments/metr-la-strong-baseline-audit/models/ResidualGRU-time-corruption-aware-strong/best_checkpoint.pt"
    if not strong_ckpt.exists():
        raise FileNotFoundError(str(strong_ckpt))
    strong_model.load_state_dict(torch.load(strong_ckpt, map_location="cpu"))
    strong_model.to(device)
    eval_models["Strong ResidualGRU-time"] = ("nn", strong_model)

    if args.model_set != "official_stid_precheck":
        sraf_current = SRAFResidualGRU(
            sensors=train_x_base.shape[2],
            features=3,
            output_features=1,
            horizon=train_y.shape[1],
            hidden_dim=32,
            sensor_embedding_dim=8,
            horizon_aware_decoder=True,
        )
        sraf_ckpt = ROOT / "experiments/metr-la-sraf-rc-v2-horizon-targeted-dominance/candidates/horizon_reference/best_checkpoint.pt"
        if not sraf_ckpt.exists():
            raise FileNotFoundError(str(sraf_ckpt))
        sraf_current.load_state_dict(torch.load(sraf_ckpt, map_location="cpu"))
        sraf_current.to(device)
        eval_models["current SRAF-RC-V2-Horizon"] = ("nn", sraf_current)

    # best clean backbone model (if trained)
    if selected_backbone is not None:
        eval_models["best clean backbone"] = ("nn", trained_clean_models[selected_backbone])
        # Train SRAF+best backbone corruption-aware
        backbone_for_sraf = build_backbone(selected_backbone, sensors=train_x_base.shape[2], horizon=train_y.shape[1], args=args)
        sraf_plus = SRAFBackboneWrapper(
            sensors=train_x_base.shape[2],
            input_features=3,
            horizon=train_y.shape[1],
            repair_hidden_dim=args.repair_hidden_dim,
            repair_sensor_embedding_dim=8,
            backbone=backbone_for_sraf,
        )
        sraf_run_dir = out_dir / "models" / "SRAF+best-backbone"
        sraf_run_dir.mkdir(parents=True, exist_ok=True)
        meta_sraf, curves = train_sraf_integrated_model(
            sraf_plus, train_x, train_y, val_x, val_y, args, sraf_run_dir, adjacency, train_std=std, device=device
        )
        train_curve_rows.extend([{**c, "model": "SRAF+best-backbone", "stage": "integration"} for c in curves])
        clean_pred_sraf, clean_infer_sraf = predict_model(sraf_plus, fault_inputs_time["clean"], args.batch_size, adjacency, device)
        np.savez_compressed(sraf_run_dir / "clean_predictions.npz", y_pred=clean_pred_sraf, y_true=test_y)
        complexity_rows.append(
            {
                "dataset": "METR-LA",
                "run_id": run_id,
                "model": "SRAF+best-backbone",
                "parameter_count": model_param_count(sraf_plus),
                "training_time_sec": meta_sraf["training_time_sec"],
                "clean_inference_time_sec": clean_infer_sraf,
                "best_epoch": meta_sraf["best_epoch"],
                "best_val_loss": meta_sraf["best_val_loss"],
            }
        )
        eval_models["SRAF+best-backbone"] = ("nn", sraf_plus)
    else:
        failed_rows.append(
            {
                "model": "SRAF+best-backbone",
                "status": "skipped",
                "reason": "No SRAF integration is run in OFFICIAL_STYLE_STID_CODE_REPAIR_GATE."
                if args.model_set == "official_stid_precheck"
                else "No clean backbone beat Strong ResidualGRU-time in Stage 1.",
            }
        )

    if args.model_set == "official_stid_precheck":
        for name, model in trained_clean_models.items():
            eval_models[name] = ("official_stid" if name == "OfficialStyleSTID-clean-backbone" else "nn", model)

    # evaluate all selected models on shared faults
    for model_name, (kind, model_obj) in eval_models.items():
        try:
            for setting in active_fault_settings:
                label = setting["label"]
                x_fault = fault_inputs_stid[label] if kind == "official_stid" else fault_inputs_time[label]
                if kind == "persistence":
                    start = perf_counter()
                    pred = persistence_predict(np.nan_to_num(x_fault[..., :1], nan=0.0).astype(np.float32), test_y.shape[1])
                    infer_t = perf_counter() - start
                else:
                    pred, infer_t = predict_model(model_obj, x_fault, args.batch_size, adjacency, device)  # type: ignore[arg-type]
                m = safe_metrics(test_y, pred, mean, std)
                metrics_rows.append(
                    {
                        "dataset": "METR-LA",
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
                        "inference_time_sec": infer_t,
                    }
                )
                horizon_rows.append(
                    {
                        "dataset": "METR-LA",
                        "run_id": run_id,
                        "model": model_name,
                        "fault": label,
                        "h3_mae": m["mae_h3"],
                        "h6_mae": m["mae_h6"],
                        "h12_mae": m["mae_h12"],
                    }
                )
        except Exception as exc:
            failed_rows.append({"model": model_name, "status": "failed", "reason": repr(exc)})
            print(f"FAILED eval {model_name}: {exc!r}", flush=True)

    rdr_rows = metric_rows_to_rdr(metrics_rows, run_id)
    write_csv(out_dir / "metrics_by_model_fault.csv", metrics_rows)
    write_csv(out_dir / "horizon_metrics.csv", horizon_rows)
    write_csv(out_dir / "robustness_rdr.csv", rdr_rows)
    write_csv(out_dir / "complexity_metrics.csv", complexity_rows)
    write_csv(out_dir / "training_curves.csv", train_curve_rows)
    write_csv(out_dir / "failed_or_skipped_models.csv", failed_rows)
    if args.model_set == "official_stid_precheck":
        write_csv(out_dir / "official_style_stid_clean_precheck_metrics.csv", metrics_rows)
        write_csv(out_dir / "official_style_stid_horizon_metrics.csv", horizon_rows)
        write_csv(out_dir / "official_style_stid_training_curves.csv", train_curve_rows)
        write_csv(out_dir / "official_style_stid_complexity.csv", complexity_rows)

    # Summaries
    backbone_lines = [
        "# Backbone Selection Summary",
        "",
        f"Strong ResidualGRU-time clean MAE reference: {strong_ref_mae:.6f}.",
    ]
    for row in clean_backbone_rows:
        if row["model"] == "Strong ResidualGRU-time reference":
            continue
        backbone_lines.append(
            f"- {row['model']}: clean MAE={float(row['mae']):.6f}, h3={float(row['h3_mae']):.6f}, h6={float(row['h6_mae']):.6f}, h12={float(row['h12_mae']):.6f}."
        )
    backbone_lines.append("")
    backbone_lines.append(f"Selected backbone: {selected_backbone if selected_backbone is not None else 'NONE'}")
    (out_dir / "backbone_selection_summary.md").write_text("\n".join(backbone_lines), encoding="utf-8")

    integration_lines = ["# SRAF Backbone Integration Summary", ""]
    if selected_backbone is None:
        integration_lines.append("Integration skipped because no clean backbone beat Strong ResidualGRU-time.")
    else:
        for fault in [s["label"] for s in FAULT_SETTINGS]:
            if not any(r["model"] == "SRAF+best-backbone" and r["fault"] == fault for r in metrics_rows):
                continue
            sraf_mae = row_value(metrics_rows, "SRAF+best-backbone", fault, "mae")
            backbone_mae = row_value(metrics_rows, "best clean backbone", fault, "mae")
            integration_lines.append(
                f"- {fault}: SRAF+backbone MAE={sraf_mae:.6f}, backbone-alone MAE={backbone_mae:.6f}, delta={sraf_mae - backbone_mae:.6f}."
            )
    (out_dir / "sraf_backbone_integration_summary.md").write_text("\n".join(integration_lines), encoding="utf-8")

    literature_lines = [
        "# Literature Clean Gap Summary",
        "",
        "The following literature ranges are contextual references only and not strict fair comparisons unless preprocessing and protocol are exactly aligned.",
        f"- Typical reported METR-LA MAE range (user-provided context): h3/h6/h12 around {LITERATURE_HINT['h3_mae_range']}.",
    ]
    for model in ["Strong ResidualGRU-time", "current SRAF-RC-V2-Horizon", "best clean backbone", "SRAF+best-backbone"]:
        row = next((r for r in metrics_rows if r["model"] == model and r["fault"] == "clean"), None)
        if row is None:
            literature_lines.append(f"- {model}: TODO (not available).")
        else:
            literature_lines.append(
                f"- {model}: clean h3={float(row['mae_h3']):.6f}, h6={float(row['mae_h6']):.6f}, h12={float(row['mae_h12']):.6f}, overall MAE={float(row['mae']):.6f}."
            )
    (out_dir / "literature_clean_gap_summary.md").write_text("\n".join(literature_lines), encoding="utf-8")

    if args.model_set == "official_stid_precheck":
        official_row = next(
            (r for r in clean_backbone_rows if r["model"] == "OfficialStyleSTID-clean-backbone"),
            None,
        )
        legacy_row = next((r for r in clean_backbone_rows if r["model"] == "STID-style-clean-backbone"), None)
        code_repair_lines = [
            "# Official-Style STID Code Repair Summary",
            "",
            "- Legacy `STIDBackbone` was preserved for comparison.",
            "- `OfficialStyleSTID` uses Conv2d time-series embedding, node identity embedding, time-in-day embedding, day-in-week embedding, residual 1x1 Conv2d MLP blocks, and a Conv2d regression head.",
            "- `OfficialStyleSTID` ignores adjacency and does not use horizon embedding or adjacency smoothing.",
            "- SRAF integration was not run in this repair gate.",
            "",
            f"- Strong ResidualGRU-time reference clean MAE: {strong_ref_mae:.6f}.",
        ]
        if legacy_row is not None:
            code_repair_lines.append(f"- Legacy STIDBackbone clean MAE: {float(legacy_row['mae']):.6f}.")
        if official_row is not None:
            code_repair_lines.append(f"- OfficialStyleSTID clean MAE: {float(official_row['mae']):.6f}.")
        (out_dir / "official_style_stid_code_repair_summary.md").write_text(
            "\n".join(code_repair_lines),
            encoding="utf-8",
        )

    manifest = {
        "run_id": "metr-la-official-style-stid-code-repair"
        if args.model_set == "official_stid_precheck"
        else "metr-la-strong-clean-backbone-integration",
        "gate": "OFFICIAL_STYLE_STID_CODE_REPAIR_GATE"
        if args.model_set == "official_stid_precheck"
        else "STRONG_CLEAN_BACKBONE_INTEGRATION_GATE",
        "created_at": "2026-05-21",
        "seed": args.seed,
        "data_dir": str(data_dir),
        "output_dir": str(out_dir),
        "model_set": args.model_set,
        "device_requested": args.device,
        "device_resolved": str(device),
        "dataset": {
            "name": "METR-LA",
            "L": 12,
            "H": 12,
            "N": int(train_x_base.shape[2]),
            "F_target": 1,
            "train_samples_used": int(train_x_base.shape[0]),
            "val_samples_used": int(val_x_base.shape[0]),
            "test_samples_used": int(test_x_base.shape[0]),
            "full_train_samples": int(full_train_count),
            "full_val_samples": int(full_val_count),
            "full_test_samples": int(full_test_count),
        },
        "training": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "loss": args.loss,
            "stid_hidden_dim": args.stid_hidden_dim,
            "tcn_hidden_dim": args.tcn_hidden_dim,
            "repair_hidden_dim": args.repair_hidden_dim,
            "lambda_repair": args.lambda_repair,
            "lambda_rel": args.lambda_rel,
        },
        "time_feature_construction": "OfficialStyleSTID uses speed plus input-window discrete time-of-day/day-of-week identity features; legacy and residual references use speed plus input-window sin/cos time-of-day."
        if args.model_set == "official_stid_precheck"
        else "Speed plus input-window sin/cos time-of-day; no future target leakage.",
        "time_feature_audit_path": str(out_dir / "time_feature_audit.json") if args.model_set == "official_stid_precheck" else None,
        "target_leakage_check": "Target Y never corrupted; only input speed channel corrupted for faulted evaluation.",
        "selected_backbone": selected_backbone,
        "models_evaluated": list(eval_models.keys()),
        "fault_settings": active_fault_settings,
        "integrity_note": "No PEMS-BAY and no manuscript conclusions.",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "status": "completed",
        "selected_backbone": selected_backbone,
        "clean_candidate_count": len([r for r in clean_backbone_rows if r["model"] != "Strong ResidualGRU-time reference"]),
        "metrics_rows": len(metrics_rows),
        "failed_or_skipped": len(failed_rows),
    }


def pd_read(path: Path) -> Any:
    import pandas as pd

    return pd.read_csv(path)


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
