"""Run SRAF_STID_SAME_BACKBONE_GAIN_GATE on METR-LA."""

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

from scripts.run_metr_la_strong_clean_backbone_integration import (  # noqa: E402
    add_stid_identity_features,
    add_time_of_day_features,
    apply_fault,
    load_scale,
    load_split,
    resolve_device,
)
from src.metrics.regression import regression_metrics  # noqa: E402
from src.models.baselines import persistence_predict  # noqa: E402
from src.models.residual_models import ResidualGRU, SRAFResidualGRU  # noqa: E402
from src.models.strong_backbones import OfficialStyleSTID, SRAFOfficialStyleSTIDWrapper  # noqa: E402


FAULT_SETTINGS = [
    {"fault": "clean", "label": "clean", "severity_group": "clean"},
    {"fault": "random_missing", "rate": 0.20, "label": "random_missing_20", "severity_group": "medium"},
    {"fault": "random_missing", "rate": 0.40, "label": "random_missing_40", "severity_group": "high"},
    {"fault": "continuous_outage", "length": 24, "label": "continuous_outage_24", "severity_group": "high"},
    {"fault": "gaussian_noise", "severity": "high", "label": "gaussian_noise_high", "severity_group": "high"},
    {"fault": "linear_drift", "severity": "high", "label": "linear_drift_high", "severity_group": "high"},
    {"fault": "stuck_at_last_value", "severity": "high", "label": "stuck_at_last_value_high", "severity_group": "high"},
]

TRAIN_FAULTS = [s for s in FAULT_SETTINGS if s["fault"] != "clean"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/metr-la")
    parser.add_argument("--output-dir", default="experiments/metr-la-sraf-stid-same-backbone-gain")
    parser.add_argument("--train-limit", type=int, default=8000)
    parser.add_argument("--val-limit", type=int, default=1024)
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


def inverse_scale(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return x * std + mean


def safe_metrics(y_true_norm: np.ndarray, y_pred_norm: np.ndarray, mean: float, std: float) -> dict[str, float]:
    return regression_metrics(inverse_scale(y_true_norm, mean, std), inverse_scale(y_pred_norm, mean, std))


def make_loss(name: str) -> nn.Module:
    return nn.L1Loss() if name == "mae" else nn.MSELoss()


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


def model_param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def build_official_stid(sensors: int, input_length: int, horizon: int) -> OfficialStyleSTID:
    return OfficialStyleSTID(
        sensors=sensors,
        input_length=input_length,
        input_dim=3,
        horizon=horizon,
        embed_dim=32,
        node_dim=32,
        temp_dim_tid=32,
        temp_dim_diw=32,
        num_layers=3,
        dropout=0.15,
    )


def build_sraf_stid(sensors: int, input_length: int, horizon: int, use_reliability_gate: bool = True) -> SRAFOfficialStyleSTIDWrapper:
    return SRAFOfficialStyleSTIDWrapper(
        sensors=sensors,
        horizon=horizon,
        repair_hidden_dim=32,
        repair_sensor_embedding_dim=8,
        backbone=build_official_stid(sensors=sensors, input_length=input_length, horizon=horizon),
        use_reliability_gate=use_reliability_gate,
    )


def clean_input_for_backbone(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    out[..., :1] = np.nan_to_num(out[..., :1], nan=0.0)
    return out.astype(np.float32)


def corruption_aware_batch(
    x_aug: np.ndarray,
    setting: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    speed_corrupt, mask, _ = apply_fault(x_aug[..., :1], setting, seed=seed, train_std=1.0)
    x_corrupt = x_aug.copy()
    x_corrupt[..., :1] = speed_corrupt
    observed_mask = np.isfinite(speed_corrupt).astype(np.float32)
    return x_corrupt.astype(np.float32), mask.astype(np.float32), observed_mask


def fixed_corrupt_val_sets(val_x: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray, str]]:
    settings = [
        {"fault": "random_missing", "rate": 0.40, "label": "random_missing_40", "severity_group": "high"},
        {"fault": "gaussian_noise", "severity": "high", "label": "gaussian_noise_high", "severity_group": "high"},
        {"fault": "stuck_at_last_value", "severity": "high", "label": "stuck_at_last_value_high", "severity_group": "high"},
    ]
    out = []
    for idx, setting in enumerate(settings):
        x_corrupt, _, observed = corruption_aware_batch(val_x, setting, seed + 1000 + idx)
        out.append((x_corrupt, observed, setting["label"]))
    return out


def eval_loss(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
    loss_fn: nn.Module,
    sraf: bool = False,
    observed_mask: np.ndarray | None = None,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(clean_input_for_backbone(x[i : i + batch_size])).to(device)
            yb = torch.from_numpy(y[i : i + batch_size].astype(np.float32)).to(device)
            if sraf:
                om = None if observed_mask is None else torch.from_numpy(observed_mask[i : i + batch_size].astype(np.float32)).to(device)
                pred = model(xb, observed_mask=om)
            else:
                pred = model(xb)
            losses.append(float(loss_fn(pred, yb).detach().cpu()))
    return float(np.mean(losses))


def train_official_stid_ca(
    model: OfficialStyleSTID,
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
        corrupt_vals = [
            eval_loss(model, vx, val_y, args.batch_size, device, loss_fn)
            for vx, _, _ in fixed_val
        ]
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
                "model": "OfficialStyleSTID-corruption-aware",
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
            f"OfficialStyleSTID-CA epoch={epoch} train={np.mean(losses):.6f} clean_val={clean_val:.6f} corrupt_val={corrupt_val:.6f}",
            flush=True,
        )
        if no_improve >= args.patience:
            rows[-1]["early_stop_triggered"] = True
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, run_dir / "best_checkpoint.pt")
    return {"training_time_sec": perf_counter() - start, "best_epoch": best_epoch, "best_val_loss": best_val}, rows


def train_sraf_stid(
    model: SRAFOfficialStyleSTIDWrapper,
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
    adjacency: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
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
        forecast_losses: list[float] = []
        repair_losses: list[float] = []
        rel_losses: list[float] = []
        total_losses: list[float] = []
        for xb, yb in iter_batches(train_x, train_y, args.batch_size, shuffle=True, seed=args.seed, epoch=epoch):
            setting = TRAIN_FAULTS[step % len(TRAIN_FAULTS)]
            x_corrupt, mask, observed = corruption_aware_batch(xb, setting, args.seed + step)
            xb_t = torch.from_numpy(clean_input_for_backbone(x_corrupt)).to(device)
            yb_t = torch.from_numpy(yb.astype(np.float32)).to(device)
            mask_t = torch.from_numpy(mask.astype(np.float32)).to(device)
            observed_t = torch.from_numpy(observed.astype(np.float32)).to(device)
            clean_speed_t = torch.from_numpy(xb[..., :1].astype(np.float32)).to(device)
            pred, comps = model(xb_t, adjacency=adjacency, observed_mask=observed_t, return_components=True)
            forecast = loss_fn(pred, yb_t)
            denom = mask_t.sum().clamp_min(1.0)
            repair = torch.sum(torch.abs(comps["repaired_input_speed"] - clean_speed_t) * mask_t) / denom
            rel_target = 1.0 - mask_t
            rel = torch.mean((comps["reliability"] - rel_target) ** 2)
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
        clean_val = eval_loss(model, val_x, val_y, args.batch_size, device, loss_fn, sraf=True)
        corrupt_vals = [
            eval_loss(model, vx, val_y, args.batch_size, device, loss_fn, sraf=True, observed_mask=obs)
            for vx, obs, _ in fixed_val
        ]
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
                "train_loss": float(np.mean(total_losses)),
                "forecast_loss": float(np.mean(forecast_losses)),
                "repair_loss": float(np.mean(repair_losses)),
                "reliability_loss": float(np.mean(rel_losses)),
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
            f"{model_name} epoch={epoch} train={np.mean(total_losses):.6f} forecast={np.mean(forecast_losses):.6f} "
            f"repair={np.mean(repair_losses):.6f} rel={np.mean(rel_losses):.6f} clean_val={clean_val:.6f} corrupt_val={corrupt_val:.6f}",
            flush=True,
        )
        if no_improve >= args.patience:
            rows[-1]["early_stop_triggered"] = True
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, run_dir / "best_checkpoint.pt")
    return {"training_time_sec": perf_counter() - start, "best_epoch": best_epoch, "best_val_loss": best_val}, rows


def predict_model(
    model: nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
    sraf: bool = False,
    observed_mask: np.ndarray | None = None,
    adjacency: torch.Tensor | None = None,
    return_components: bool = False,
) -> tuple[np.ndarray, float, dict[str, np.ndarray] | None]:
    model.eval()
    preds: list[np.ndarray] = []
    reliabilities: list[np.ndarray] = []
    repaired: list[np.ndarray] = []
    start = perf_counter()
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(clean_input_for_backbone(x[i : i + batch_size])).to(device)
            if sraf:
                om = None if observed_mask is None else torch.from_numpy(observed_mask[i : i + batch_size].astype(np.float32)).to(device)
                if return_components:
                    pred, comps = model(xb, adjacency=adjacency, observed_mask=om, return_components=True)
                    reliabilities.append(comps["reliability"].detach().cpu().numpy())
                    repaired.append(comps["repaired_input_speed"].detach().cpu().numpy())
                else:
                    pred = model(xb, adjacency=adjacency, observed_mask=om)
            else:
                pred = model(xb)
            preds.append(pred.detach().cpu().numpy())
    comps_out = None
    if return_components and reliabilities:
        comps_out = {"reliability": np.concatenate(reliabilities, axis=0), "repaired_speed": np.concatenate(repaired, axis=0)}
    return np.concatenate(preds, axis=0), perf_counter() - start, comps_out


def load_reference_models(sensors: int, input_length: int, horizon: int, device: torch.device) -> dict[str, tuple[str, nn.Module | None]]:
    models: dict[str, tuple[str, nn.Module | None]] = {"Persistence": ("persistence", None)}
    strong = ResidualGRU(
        sensors=sensors,
        features=3,
        output_features=1,
        horizon=horizon,
        hidden_dim=32,
        sensor_embedding_dim=8,
    )
    strong_ckpt = ROOT / "experiments/metr-la-strong-baseline-audit/models/ResidualGRU-time-corruption-aware-strong/best_checkpoint.pt"
    strong.load_state_dict(torch.load(strong_ckpt, map_location="cpu"))
    strong.to(device)
    models["Strong ResidualGRU-time reference"] = ("sin_cos", strong)

    sraf = SRAFResidualGRU(
        sensors=sensors,
        features=3,
        output_features=1,
        horizon=horizon,
        hidden_dim=32,
        sensor_embedding_dim=8,
        horizon_aware_decoder=True,
    )
    sraf_ckpt = ROOT / "experiments/metr-la-sraf-rc-v2-horizon-targeted-dominance/candidates/horizon_reference/best_checkpoint.pt"
    sraf.load_state_dict(torch.load(sraf_ckpt, map_location="cpu"))
    sraf.to(device)
    models["current SRAF-RC-V2-Horizon reference"] = ("sin_cos", sraf)

    official_clean = build_official_stid(sensors=sensors, input_length=input_length, horizon=horizon)
    official_clean_ckpt = ROOT / "experiments/metr-la-official-style-stid-code-repair/models/OfficialStyleSTID-clean-backbone/best_checkpoint.pt"
    official_clean.load_state_dict(torch.load(official_clean_ckpt, map_location="cpu"))
    official_clean.to(device)
    models["OfficialStyleSTID-clean"] = ("stid", official_clean)
    return models


def reliability_stats(reliability: np.ndarray, mask: np.ndarray, repaired_speed: np.ndarray, clean_speed: np.ndarray) -> dict[str, Any]:
    rel = reliability[..., :1]
    corrupted = mask.astype(bool)
    clean = ~corrupted
    out: dict[str, Any] = {
        "mean_reliability": float(np.mean(rel)),
        "min_reliability": float(np.min(rel)),
        "fraction_positions_repaired": float(np.mean(corrupted)),
    }
    if corrupted.any():
        out["corrupted_position_reliability_mean"] = float(np.mean(rel[corrupted]))
        out["repair_loss_on_corrupted_positions"] = float(np.mean(np.abs(repaired_speed[corrupted] - clean_speed[corrupted])))
    else:
        out["corrupted_position_reliability_mean"] = "TODO"
        out["repair_loss_on_corrupted_positions"] = "TODO"
    if clean.any():
        out["clean_position_reliability_mean"] = float(np.mean(rel[clean]))
    else:
        out["clean_position_reliability_mean"] = "TODO"
    if corrupted.any() and clean.any():
        out["corrupted_lower_than_clean"] = bool(out["corrupted_position_reliability_mean"] < out["clean_position_reliability_mean"])
    else:
        out["corrupted_lower_than_clean"] = "TODO"
    return out


def main() -> None:
    args = build_parser().parse_args()
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    if args.smoke:
        args.train_limit = 256
        args.val_limit = 128
        args.epochs = min(args.epochs, 2)
        args.patience = 1

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "models"
    model_dir.mkdir(exist_ok=True)
    fault_dir = out_dir / "fault_masks"
    fault_dir.mkdir(exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    train_x_base_full, train_y_full = load_split(data_dir, "train")
    val_x_base_full, val_y_full = load_split(data_dir, "val")
    test_x_base, test_y = load_split(data_dir, "test")
    mean, std = load_scale(data_dir)
    adjacency = torch.from_numpy(np.load(data_dir / "adjacency.npy").astype(np.float32)).to(device)

    train_x_base = train_x_base_full[: args.train_limit]
    train_y = train_y_full[: args.train_limit]
    val_x_base = val_x_base_full[: args.val_limit]
    val_y = val_y_full[: args.val_limit]
    if args.smoke:
        test_x_base = test_x_base[:256]
        test_y = test_y[:256]

    full_train_count = train_x_base_full.shape[0]
    full_val_count = val_x_base_full.shape[0]
    test_start = full_train_count + full_val_count
    train_x_stid = add_stid_identity_features(train_x_base, 0)
    val_x_stid = add_stid_identity_features(val_x_base, full_train_count)
    test_x_stid_clean = add_stid_identity_features(test_x_base, test_start)
    test_x_time_clean = add_time_of_day_features(test_x_base, test_start)

    fault_inputs_stid: dict[str, np.ndarray] = {}
    fault_inputs_time: dict[str, np.ndarray] = {}
    fault_masks: dict[str, np.ndarray] = {}
    observed_masks: dict[str, np.ndarray] = {}
    for idx, setting in enumerate(FAULT_SETTINGS):
        label = setting["label"]
        speed_fault, mask, meta = apply_fault(test_x_base, setting, seed=args.seed + idx, train_std=1.0)
        fault_inputs_stid[label] = add_stid_identity_features(speed_fault, test_start)
        fault_inputs_time[label] = add_time_of_day_features(speed_fault, test_start)
        fault_masks[label] = mask.astype(bool)
        observed_masks[label] = np.isfinite(speed_fault).astype(np.float32)
        meta = {**setting, **meta, "label": label, "target_corrupted": False, "mask_path": str(fault_dir / f"{label}_mask.npz")}
        np.savez_compressed(fault_dir / f"{label}_mask.npz", mask=mask)
        (fault_dir / f"{label}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    failed_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []

    references = load_reference_models(train_x_base.shape[2], train_x_base.shape[1], train_y.shape[1], device)

    stid_ca = build_official_stid(train_x_base.shape[2], train_x_base.shape[1], train_y.shape[1])
    ca_dir = model_dir / "OfficialStyleSTID-corruption-aware"
    ca_dir.mkdir(exist_ok=True)
    ca_meta, ca_curves = train_official_stid_ca(stid_ca, train_x_stid, train_y, val_x_stid, val_y, args, ca_dir, device)
    training_rows.extend(ca_curves)
    complexity_rows.append(
        {
            "model": "OfficialStyleSTID-corruption-aware",
            "parameter_count": model_param_count(stid_ca),
            "training_time_sec": ca_meta["training_time_sec"],
            "best_epoch": ca_meta["best_epoch"],
            "best_val_loss": ca_meta["best_val_loss"],
        }
    )

    sraf_full = build_sraf_stid(train_x_base.shape[2], train_x_base.shape[1], train_y.shape[1], use_reliability_gate=True)
    sraf_dir = model_dir / "SRAF-OfficialStyleSTID-full"
    sraf_dir.mkdir(exist_ok=True)
    sraf_meta, sraf_curves = train_sraf_stid(
        sraf_full, "SRAF-OfficialStyleSTID-full", train_x_stid, train_y, val_x_stid, val_y, args, sraf_dir, device, adjacency
    )
    training_rows.extend(sraf_curves)
    complexity_rows.append(
        {
            "model": "SRAF-OfficialStyleSTID-full",
            "parameter_count": model_param_count(sraf_full),
            "training_time_sec": sraf_meta["training_time_sec"],
            "best_epoch": sraf_meta["best_epoch"],
            "best_val_loss": sraf_meta["best_val_loss"],
        }
    )

    sraf_no_gate: SRAFOfficialStyleSTIDWrapper | None = None
    try:
        sraf_no_gate = build_sraf_stid(train_x_base.shape[2], train_x_base.shape[1], train_y.shape[1], use_reliability_gate=False)
        no_gate_dir = model_dir / "SRAF-OfficialStyleSTID-no-reliability-gate"
        no_gate_dir.mkdir(exist_ok=True)
        ng_meta, ng_curves = train_sraf_stid(
            sraf_no_gate,
            "SRAF-OfficialStyleSTID-no-reliability-gate",
            train_x_stid,
            train_y,
            val_x_stid,
            val_y,
            args,
            no_gate_dir,
            device,
            adjacency,
        )
        training_rows.extend(ng_curves)
        complexity_rows.append(
            {
                "model": "SRAF-OfficialStyleSTID-no-reliability-gate",
                "parameter_count": model_param_count(sraf_no_gate),
                "training_time_sec": ng_meta["training_time_sec"],
                "best_epoch": ng_meta["best_epoch"],
                "best_val_loss": ng_meta["best_val_loss"],
            }
        )
    except Exception as exc:
        failed_rows.append({"model": "SRAF-OfficialStyleSTID-no-reliability-gate", "status": "failed", "reason": repr(exc)})

    models: dict[str, tuple[str, nn.Module | None]] = {
        **references,
        "OfficialStyleSTID-corruption-aware": ("stid", stid_ca),
        "SRAF-OfficialStyleSTID-full": ("sraf_stid", sraf_full),
    }
    if sraf_no_gate is not None:
        models["SRAF-OfficialStyleSTID-no-reliability-gate"] = ("sraf_stid", sraf_no_gate)
    else:
        failed_rows.append(
            {
                "model": "SRAF-OfficialStyleSTID-no-reliability-gate",
                "status": "skipped",
                "reason": "No trained no-gate model available.",
            }
        )
    failed_rows.append({"model": "SRAF-OfficialStyleSTID-temporal-only", "status": "skipped", "reason": "Optional ablation skipped to avoid delaying main full model."})
    failed_rows.append({"model": "SRAF-OfficialStyleSTID-spatial-only", "status": "skipped", "reason": "Optional ablation skipped to avoid delaying main full model."})

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
                pred = persistence_predict(clean_input_for_backbone(fault_inputs_stid[label])[..., :1], test_y.shape[1])
                infer_time = perf_counter() - start
                comps = None
            elif kind == "sin_cos":
                pred, infer_time, comps = predict_model(
                    model, fault_inputs_time[label], args.batch_size, device, sraf=False
                )
            elif kind == "stid":
                pred, infer_time, comps = predict_model(
                    model, fault_inputs_stid[label], args.batch_size, device, sraf=False
                )
            elif kind == "sraf_stid":
                pred, infer_time, comps = predict_model(
                    model,
                    fault_inputs_stid[label],
                    args.batch_size,
                    device,
                    sraf=True,
                    observed_mask=observed_masks[label],
                    adjacency=adjacency,
                    return_components=True,
                )
                if comps is not None:
                    diag = reliability_stats(comps["reliability"], fault_masks[label], comps["repaired_speed"], test_x_base[..., :1])
                    repair_rows.append({"model": model_name, "fault": label, **diag})
                    reliability_rows.append({"model": model_name, "fault": label, **diag})
            else:
                raise ValueError(kind)
            inference_times[(model_name, label)] = infer_time
            m = safe_metrics(test_y, pred, mean, std)
            metrics_rows.append(
                {
                    "dataset": "METR-LA",
                    "run_id": "metr-la-sraf-stid-same-backbone-gain",
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
                    "dataset": "METR-LA",
                    "run_id": "metr-la-sraf-stid-same-backbone-gain",
                    "model": model_name,
                    "fault": label,
                    "h3_mae": m["mae_h3"],
                    "h6_mae": m["mae_h6"],
                    "h12_mae": m["mae_h12"],
                }
            )
            if label == "clean" and model_name in {"OfficialStyleSTID-corruption-aware", "SRAF-OfficialStyleSTID-full"}:
                np.savez_compressed(pred_dir / f"{model_name}_clean_predictions.npz", y_pred=pred, y_true=test_y)

    clean_by_model = {r["model"]: float(r["mae"]) for r in metrics_rows if r["fault"] == "clean"}
    rdr_rows = []
    for row in metrics_rows:
        clean_mae = clean_by_model.get(row["model"], math.nan)
        fault_mae = float(row["mae"])
        rdr_rows.append(
            {
                "dataset": "METR-LA",
                "run_id": "metr-la-sraf-stid-same-backbone-gain",
                "model": row["model"],
                "fault": row["fault"],
                "fault_type": row["fault_type"],
                "severity_group": row["severity_group"],
                "clean_mae": clean_mae,
                "fault_mae": fault_mae,
                "rdr_mae": (fault_mae - clean_mae) / clean_mae if clean_mae else "TODO",
            }
        )

    stid_clean_mae = clean_by_model.get("OfficialStyleSTID-clean", math.nan)
    clp_rows = []
    for model_name, clean_mae in clean_by_model.items():
        clp_rows.append(
            {
                "model": model_name,
                "official_stid_clean_mae": stid_clean_mae,
                "model_clean_mae": clean_mae,
                "clean_loss_penalty": (clean_mae - stid_clean_mae) / stid_clean_mae if stid_clean_mae else "TODO",
            }
        )

    rg_rows = []
    for setting in FAULT_SETTINGS:
        label = setting["label"]
        ca = next(r for r in metrics_rows if r["model"] == "OfficialStyleSTID-corruption-aware" and r["fault"] == label)
        sraf = next(r for r in metrics_rows if r["model"] == "SRAF-OfficialStyleSTID-full" and r["fault"] == label)
        ca_mae = float(ca["mae"])
        sraf_mae = float(sraf["mae"])
        rg_rows.append(
            {
                "fault": label,
                "official_stid_ca_mae": ca_mae,
                "sraf_stid_full_mae": sraf_mae,
                "same_backbone_robustness_gain": (ca_mae - sraf_mae) / ca_mae,
                "sraf_better": sraf_mae < ca_mae,
            }
        )

    same_gain_rows = []
    for row in rg_rows:
        fault = row["fault"]
        rdr_ca = next(r for r in rdr_rows if r["model"] == "OfficialStyleSTID-corruption-aware" and r["fault"] == fault)
        rdr_sraf = next(r for r in rdr_rows if r["model"] == "SRAF-OfficialStyleSTID-full" and r["fault"] == fault)
        same_gain_rows.append({**row, "official_stid_ca_rdr": rdr_ca["rdr_mae"], "sraf_stid_full_rdr": rdr_sraf["rdr_mae"]})

    for row in complexity_rows:
        model_name = row["model"]
        row["clean_inference_time_sec"] = inference_times.get((model_name, "clean"), "TODO")
        ca_latency = inference_times.get(("OfficialStyleSTID-corruption-aware", "clean"))
        if isinstance(row["clean_inference_time_sec"], float) and ca_latency:
            row["latency_overhead_vs_stid_ca"] = row["clean_inference_time_sec"] - ca_latency
        else:
            row["latency_overhead_vs_stid_ca"] = "TODO"

    write_csv(out_dir / "metrics_by_model_fault.csv", metrics_rows)
    write_csv(out_dir / "horizon_metrics.csv", horizon_rows)
    write_csv(out_dir / "robustness_rdr.csv", rdr_rows)
    write_csv(out_dir / "clean_loss_penalty.csv", clp_rows)
    write_csv(out_dir / "robustness_gain_vs_stid_ca.csv", rg_rows)
    write_csv(out_dir / "same_backbone_gain_summary.csv", same_gain_rows)
    write_csv(out_dir / "repair_diagnostics_by_fault.csv", repair_rows)
    write_csv(out_dir / "reliability_diagnostics.csv", reliability_rows)
    write_csv(out_dir / "complexity_metrics.csv", complexity_rows)
    write_csv(out_dir / "training_curves.csv", training_rows)
    write_csv(out_dir / "failed_or_skipped_models.csv", failed_rows)

    faulty = [r for r in rg_rows if r["fault"] != "clean"]
    improved_faults = sum(bool(r["sraf_better"]) for r in faulty)
    severe_labels = {"random_missing_40", "continuous_outage_24", "gaussian_noise_high", "linear_drift_high"}
    severe_comparable = 0
    for label in severe_labels:
        ca_rdr = float(next(r for r in same_gain_rows if r["fault"] == label)["official_stid_ca_rdr"])
        sraf_rdr = float(next(r for r in same_gain_rows if r["fault"] == label)["sraf_stid_full_rdr"])
        if sraf_rdr <= ca_rdr + 0.01:
            severe_comparable += 1
    sraf_clp = float(next(r for r in clp_rows if r["model"] == "SRAF-OfficialStyleSTID-full")["clean_loss_penalty"])
    reliability_directional = []
    for row in reliability_rows:
        if row["model"] == "SRAF-OfficialStyleSTID-full" and row["corrupted_lower_than_clean"] != "TODO":
            reliability_directional.append(bool(row["corrupted_lower_than_clean"]))
    if improved_faults >= 4 and severe_comparable >= 3 and sraf_clp <= 0.15 and any(reliability_directional):
        status = "PASS"
    elif improved_faults > 0 or severe_comparable >= 2:
        status = "PARTIAL"
    else:
        status = "FAIL"

    summary_lines = [
        "# SRAF-STID Same-Backbone Gain Summary",
        "",
        f"- Gate status: `{status}`",
        f"- Improved faulty settings versus OfficialStyleSTID-corruption-aware: `{improved_faults}/6`",
        f"- SRAF clean loss penalty versus OfficialStyleSTID-clean: `{sraf_clp:.6f}`",
        f"- SRAF integration used speed-channel repair only; tod/dow identity features were preserved.",
        "",
        "## Same-Backbone Fault Results",
    ]
    for row in rg_rows:
        summary_lines.append(
            f"- {row['fault']}: CA MAE={row['official_stid_ca_mae']:.6f}, "
            f"SRAF-STID MAE={row['sraf_stid_full_mae']:.6f}, RG={row['same_backbone_robustness_gain']:.6f}."
        )
    (out_dir / "sraf_stid_same_backbone_gain_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    selection_lines = [
        "# Candidate Selection Summary",
        "",
        f"- Selected clean backbone candidate entering this gate: `OfficialStyleSTID`.",
        f"- Gate status: `{status}`.",
        f"- OfficialStyleSTID-corruption-aware and SRAF-OfficialStyleSTID-full both completed.",
        f"- No manuscript conclusions are written here.",
    ]
    (out_dir / "candidate_selection_summary.md").write_text("\n".join(selection_lines), encoding="utf-8")

    manifest = {
        "run_id": "metr-la-sraf-stid-same-backbone-gain",
        "gate": "SRAF_STID_SAME_BACKBONE_GAIN_GATE",
        "created_at": "2026-05-21",
        "status": status,
        "seed": args.seed,
        "data_dir": str(data_dir),
        "output_dir": str(out_dir),
        "device_requested": args.device,
        "device_resolved": str(device),
        "dataset": {
            "name": "METR-LA",
            "L": 12,
            "H": 12,
            "N": int(train_x_base_full.shape[2]),
            "target_F": 1,
            "full_train_samples": int(train_x_base_full.shape[0]),
            "full_val_samples": int(val_x_base_full.shape[0]),
            "full_test_samples": int(test_x_base.shape[0]) if args.smoke else int(load_split(data_dir, "test")[0].shape[0]),
            "train_samples_used": int(train_x_base.shape[0]),
            "val_samples_used": int(val_x_base.shape[0]),
            "test_samples_used": int(test_x_base.shape[0]),
        },
        "training": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "loss": args.loss,
            "lambda_repair": args.lambda_repair,
            "lambda_rel": args.lambda_rel,
        },
        "time_feature_construction": "OfficialStyleSTID models use [speed_norm,tod_norm,dow_norm] identity features; SRAF repair touches only speed_norm.",
        "target_leakage_check": "Target Y is never corrupted. Faults are applied only to input speed channel.",
        "identity_preservation": "SRAFOfficialStyleSTIDWrapper concatenates original tod/dow features after speed repair and never repairs identity channels.",
        "fault_settings": FAULT_SETTINGS,
        "models_evaluated": list(models.keys()),
        "integrity_note": "No PEMS-BAY, no MoE, no manuscript conclusions, no previous outputs deleted.",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "improved_faults": improved_faults, "sraf_clp": sraf_clp}, indent=2), flush=True)


if __name__ == "__main__":
    main()
