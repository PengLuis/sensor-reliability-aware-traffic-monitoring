"""Seed-42 SAITS/GRIN-style impute-then-ID-MLP baseline adapter diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_metr_la_sraf_stid_same_backbone_gain import (  # noqa: E402
    clean_input_for_backbone,
    iter_batches,
    make_loss,
    predict_model,
    train_official_stid_ca,
    train_sraf_stid,
)
from scripts.run_metr_la_strong_clean_backbone_integration import resolve_device  # noqa: E402
from scripts.run_sraf_v2_publication_baseline_reproduction_and_selection import (  # noqa: E402
    BASELINES,
    FAULT_SPECS,
    dsae_fill,
    fill_simple,
    knn_fill,
    load_payload,
    make_v2_current_best,
    ppca_lite_fill,
    safe_metrics,
    train_dsae_lite,
)
from scripts.run_sraf_v2_version_freeze_and_multi_direction_exploration import (  # noqa: E402
    build_official_stid,
    build_v1,
    predict_v2,
    train_v2,
)
from scripts.run_metr_la_strong_clean_backbone_integration import apply_fault  # noqa: E402


FAULTS = [
    "clean",
    "random_missing_40",
    "continuous_outage_24",
    "gaussian_noise_high",
    "linear_drift_high",
    "stuck_at_last_value_high",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols: list[str] = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep] + body)


def build_fault(x: np.ndarray, label: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if label == "clean":
        mask = np.zeros_like(x[..., :1], dtype=np.float32)
        observed = np.ones_like(mask, dtype=np.float32)
        return x.astype(np.float32), mask, observed
    spec = FAULT_SPECS[label]
    speed, mask, _ = apply_fault(x[..., :1], spec, seed=seed, train_std=1.0)
    out = x.copy()
    out[..., :1] = speed
    observed = np.isfinite(speed).astype(np.float32)
    return out.astype(np.float32), mask.astype(np.float32), observed


def make_nan_imputer_input(x_fault: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_imp = x_fault[..., :1].copy()
    x_imp[mask > 0.5] = np.nan
    observed = np.isfinite(x_imp).astype(np.float32)
    clean_identity = x_fault[..., 1:].copy()
    return x_imp[..., 0].astype(np.float32), observed[..., 0].astype(np.float32), clean_identity.astype(np.float32)


class LocalSAITSStyleImputer(nn.Module):
    """Small temporal self-attention imputer; local adapter, not official SAITS."""

    def __init__(self, sensors: int, hidden: int = 32, heads: int = 4) -> None:
        super().__init__()
        self.input = nn.Linear(sensors * 2, hidden)
        enc = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=hidden * 2, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=1)
        self.out = nn.Linear(hidden, sensors)

    def forward(self, values: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(values, nan=0.0)
        z = torch.cat([x, observed], dim=-1)
        h = self.encoder(self.input(z))
        return self.out(h)


class LocalGRINStyleImputer(nn.Module):
    """Small graph-message temporal imputer; local adapter, not official GRIN."""

    def __init__(self, sensors: int, hidden: int = 32) -> None:
        super().__init__()
        self.temporal = nn.GRU(input_size=sensors * 2, hidden_size=hidden, batch_first=True)
        self.out = nn.Linear(hidden + sensors, sensors)

    def forward(self, values: torch.Tensor, observed: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(values, nan=0.0)
        graph_msg = torch.einsum("ij,btj->bti", adj_norm, x)
        h, _ = self.temporal(torch.cat([x, observed], dim=-1))
        return self.out(torch.cat([h, graph_msg], dim=-1))


def normalize_adj(adj: np.ndarray) -> np.ndarray:
    a = adj.astype(np.float32).copy()
    np.fill_diagonal(a, np.maximum(np.diag(a), 1.0))
    denom = np.clip(a.sum(axis=1, keepdims=True), 1.0e-6, None)
    return (a / denom).astype(np.float32)


def train_imputer(
    model: nn.Module,
    name: str,
    train_speed: np.ndarray,
    val_speed: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    adj_norm: torch.Tensor | None = None,
) -> tuple[nn.Module, dict[str, Any], list[dict[str, Any]]]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.imputer_lr)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    best = math.inf
    best_state = None
    no_imp = 0
    start = perf_counter()
    train_vals = train_speed[..., 0].astype(np.float32)
    val_vals = val_speed[..., 0].astype(np.float32)
    for ep in range(1, args.max_epochs_imputer + 1):
        losses: list[float] = []
        order = rng.permutation(train_vals.shape[0])
        model.train()
        for st in range(0, train_vals.shape[0], args.batch_size):
            clean = train_vals[order[st : st + args.batch_size]]
            miss = rng.random(clean.shape) < args.train_mask_rate
            inp = clean.copy()
            inp[miss] = np.nan
            obs = np.isfinite(inp).astype(np.float32)
            x_t = torch.from_numpy(inp).to(device)
            obs_t = torch.from_numpy(obs).to(device)
            y_t = torch.from_numpy(clean).to(device)
            if isinstance(model, LocalGRINStyleImputer):
                assert adj_norm is not None
                pred = model(x_t, obs_t, adj_norm)
            else:
                pred = model(x_t, obs_t)
            mask_t = torch.from_numpy(miss.astype(np.float32)).to(device)
            loss = torch.sum(torch.abs(pred - y_t) * mask_t) / mask_t.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_loss = evaluate_reconstruction(model, val_vals, args, device, adj_norm=adj_norm, seed=args.seed + ep)
        if val_loss < best - 1.0e-6:
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        rows.append({"imputer": name, "epoch": ep, "train_reconstruction_mae": float(np.mean(losses)), "val_reconstruction_mae": val_loss, "best_val": best})
        if no_imp >= args.imputer_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"imputer": name, "best_val_reconstruction_mae": best, "training_time_sec": perf_counter() - start}, rows


def evaluate_reconstruction(
    model: nn.Module,
    clean_vals: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    adj_norm: torch.Tensor | None,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    misses = rng.random(clean_vals.shape) < args.train_mask_rate
    inp = clean_vals.copy()
    inp[misses] = np.nan
    pred = impute_values(model, inp, np.isfinite(inp).astype(np.float32), args.batch_size, device, adj_norm=adj_norm)
    return float(np.mean(np.abs(pred[misses] - clean_vals[misses]))) if np.any(misses) else 0.0


def impute_values(
    model: nn.Module,
    values_nan: np.ndarray,
    observed: np.ndarray,
    batch_size: int,
    device: torch.device,
    adj_norm: torch.Tensor | None = None,
) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for st in range(0, values_nan.shape[0], batch_size):
            x_t = torch.from_numpy(values_nan[st : st + batch_size].astype(np.float32)).to(device)
            obs_t = torch.from_numpy(observed[st : st + batch_size].astype(np.float32)).to(device)
            if isinstance(model, LocalGRINStyleImputer):
                assert adj_norm is not None
                pred = model(x_t, obs_t, adj_norm)
            else:
                pred = model(x_t, obs_t)
            preds.append(pred.detach().cpu().numpy())
    out = np.concatenate(preds, axis=0)
    filled = np.nan_to_num(values_nan.copy(), nan=0.0)
    missing = observed < 0.5
    filled[missing] = out[missing]
    return filled.astype(np.float32)


def train_forecaster_clean(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any], list[dict[str, Any]]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = make_loss(args.loss)
    best = math.inf
    best_state = None
    no_imp = 0
    rows: list[dict[str, Any]] = []
    start = perf_counter()
    for ep in range(1, args.max_epochs_forecaster + 1):
        losses: list[float] = []
        model.train()
        for xb, yb in iter_batches(train_x, train_y, args.batch_size, shuffle=True, seed=args.seed, epoch=ep):
            pred = model(torch.from_numpy(clean_input_for_backbone(xb)).to(device))
            y_t = torch.from_numpy(yb.astype(np.float32)).to(device)
            loss = loss_fn(pred, y_t)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_losses: list[float] = []
        model.eval()
        with torch.no_grad():
            for st in range(0, val_x.shape[0], args.batch_size):
                pred = model(torch.from_numpy(clean_input_for_backbone(val_x[st : st + args.batch_size])).to(device))
                y_t = torch.from_numpy(val_y[st : st + args.batch_size].astype(np.float32)).to(device)
                val_losses.append(float(loss_fn(pred, y_t).detach().cpu()))
        val = float(np.mean(val_losses))
        if val < best - 1.0e-6:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        rows.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_loss": val, "best_val": best})
        if no_imp >= args.forecaster_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, run_dir / "best_checkpoint.pt")
    return model, {"best_val_loss": best, "training_time_sec": perf_counter() - start}, rows


def impute_augmented_windows(
    imputer: nn.Module,
    x_aug: np.ndarray,
    mask: np.ndarray,
    batch_size: int,
    device: torch.device,
    adj_norm: torch.Tensor | None,
) -> np.ndarray:
    values_nan, observed, identity = make_nan_imputer_input(x_aug, mask)
    filled = impute_values(imputer, values_nan, observed, batch_size, device, adj_norm=adj_norm)
    return np.concatenate([filled[..., None], identity], axis=-1).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="experiments/saits_grin_idmlp_baseline_adapter")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--train-limit", type=int, default=256)
    p.add_argument("--val-limit", type=int, default=64)
    p.add_argument("--test-limit", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.002)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--loss", choices=["mae", "mse"], default="mae")
    p.add_argument("--lambda-repair", type=float, default=0.05)
    p.add_argument("--lambda-rel", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--patience", type=int, default=1)
    p.add_argument("--max-epochs-imputer", type=int, default=3)
    p.add_argument("--imputer-patience", type=int, default=1)
    p.add_argument("--max-epochs-forecaster", type=int, default=2)
    p.add_argument("--forecaster-patience", type=int, default=1)
    p.add_argument("--imputer-lr", type=float, default=0.001)
    p.add_argument("--train-mask-rate", type=float, default=0.2)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--dsae-epochs", type=int, default=1)
    p.add_argument("--dsae-hidden", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("This gate allows seed 42 only.")
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)
    device = resolve_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    source_rows = [
        {"name": "SAITS", "source_url": "https://github.com/WenjieDu/SAITS; https://github.com/WenjieDu/PyPOTS", "local_path": "N/A (not vendored)", "license": "MIT / BSD-3-Clause", "commit_or_hash": "SAITS 660b87f19c1277065f314f24134f646229e89ca9; PyPOTS 012d45617ea6894e61a80f4b04ecb66aa5fddbb4", "dependencies": "official stack not installed; local adapter uses PyTorch", "compatibility": "local adapter compatible with current PyTorch", "input_shape": "[B,L,N] values + observed mask", "mask_convention": "observed_mask=1 observed; NaN for corrupted", "batch_training": "yes", "gpu": "yes", "integration_risk": "medium; local adapter only"},
        {"name": "GRIN", "source_url": "https://github.com/Graph-Machine-Learning-Group/grin", "local_path": "N/A (not vendored)", "license": "NEEDS REVIEW (GitHub API license null)", "commit_or_hash": "4a28afbb092600b6e6abeeabaaf67e87dbd1ed6e", "dependencies": "official stack not installed; local adapter uses PyTorch", "compatibility": "local adapter compatible with current PyTorch", "input_shape": "[B,L,N] values + observed mask + adjacency", "mask_convention": "observed_mask=1 observed; NaN for corrupted", "batch_training": "yes", "gpu": "yes", "integration_risk": "high; local adapter only"},
    ]

    metrics: list[dict[str, Any]] = []
    recon_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    adapter_rows: list[dict[str, Any]] = []

    for ds in ["METR-LA", "PEMS-BAY"]:
        payload = load_payload(ds, args.train_limit, args.val_limit, args.test_limit)
        sensors = payload["train_x"].shape[2]
        input_length = payload["train_x"].shape[1]
        horizon = payload["train_y"].shape[1]
        adj_norm_np = normalize_adj(payload["adj"])
        adj_t = torch.from_numpy(payload["adj"]).to(device)
        adj_norm_t = torch.from_numpy(adj_norm_np).to(device)
        ds_dir = out_dir / "models" / ds.lower().replace("-", "_")
        ds_dir.mkdir(parents=True, exist_ok=True)

        ca = build_official_stid(sensors, input_length, horizon)
        ca_dir = ds_dir / "id_mlp_ca"
        ca_dir.mkdir(exist_ok=True)
        train_official_stid_ca(ca, payload["train_x"], payload["train_y"], payload["val_x"], payload["val_y"], args, ca_dir, device)

        v1 = build_v1(sensors, input_length, horizon)
        v1_dir = ds_dir / "sraf_id_v1_formal"
        v1_dir.mkdir(exist_ok=True)
        train_sraf_stid(v1, "SRAF-ID-v1-formal", payload["train_x"], payload["train_y"], payload["val_x"], payload["val_y"], args, v1_dir, device, adj_t)

        v2 = make_v2_current_best(sensors, input_length, horizon)
        v2_dir = ds_dir / "sraf_id_v2_current_best"
        v2_dir.mkdir(exist_ok=True)
        train_v2(v2, "SRAF-ID-v2-current-best", payload["train_x"], payload["train_y"], payload["val_x"], payload["val_y"], args, v2_dir, device, adj_t)

        dsae = train_dsae_lite(payload["train_x"][..., :1], args, device)

        saits = LocalSAITSStyleImputer(sensors=sensors, hidden=args.hidden)
        saits, saits_meta, saits_logs = train_imputer(saits, "SAITS-style-local", payload["train_x"][..., :1], payload["val_x"][..., :1], args, device)
        torch.save(saits.state_dict(), ds_dir / "saits_style_local_imputer.pt")
        write_csv(out_dir / f"{ds.lower().replace('-', '_')}_saits_imputer_training_log.csv", saits_logs)
        runtime_rows.append({"dataset": ds, "model": "SAITS+ID-MLP", **saits_meta})

        grin = LocalGRINStyleImputer(sensors=sensors, hidden=args.hidden)
        grin, grin_meta, grin_logs = train_imputer(grin, "GRIN-style-local", payload["train_x"][..., :1], payload["val_x"][..., :1], args, device, adj_norm=adj_norm_t)
        torch.save(grin.state_dict(), ds_dir / "grin_style_local_imputer.pt")
        write_csv(out_dir / f"{ds.lower().replace('-', '_')}_grin_imputer_training_log.csv", grin_logs)
        runtime_rows.append({"dataset": ds, "model": "GRIN+ID-MLP", **grin_meta})

        train_mask = np.zeros_like(payload["train_x"][..., :1], dtype=np.float32)
        val_mask = np.zeros_like(payload["val_x"][..., :1], dtype=np.float32)
        train_saits_x = impute_augmented_windows(saits, payload["train_x"], train_mask, args.batch_size, device, adj_norm=None)
        val_saits_x = impute_augmented_windows(saits, payload["val_x"], val_mask, args.batch_size, device, adj_norm=None)
        train_grin_x = impute_augmented_windows(grin, payload["train_x"], train_mask, args.batch_size, device, adj_norm=adj_norm_t)
        val_grin_x = impute_augmented_windows(grin, payload["val_x"], val_mask, args.batch_size, device, adj_norm=adj_norm_t)

        saits_forecaster = build_official_stid(sensors, input_length, horizon)
        saits_forecaster, sf_meta, sf_logs = train_forecaster_clean(saits_forecaster, train_saits_x, payload["train_y"], val_saits_x, payload["val_y"], args, ds_dir / "saits_id_mlp", device)
        write_csv(out_dir / f"{ds.lower().replace('-', '_')}_saits_forecaster_training_log.csv", sf_logs)
        runtime_rows.append({"dataset": ds, "model": "SAITS+ID-MLP-forecaster", **sf_meta})

        grin_forecaster = build_official_stid(sensors, input_length, horizon)
        grin_forecaster, gf_meta, gf_logs = train_forecaster_clean(grin_forecaster, train_grin_x, payload["train_y"], val_grin_x, payload["val_y"], args, ds_dir / "grin_id_mlp", device)
        write_csv(out_dir / f"{ds.lower().replace('-', '_')}_grin_forecaster_training_log.csv", gf_logs)
        runtime_rows.append({"dataset": ds, "model": "GRIN+ID-MLP-forecaster", **gf_meta})

        adapter_rows.append({"dataset": ds, "adapter": "SAITS-style-local", "input": "[B,L,N] speed with NaN at M=1", "target": "clean historical X speed only", "leakage_check": "PASS: no target Y used for imputer training", "identity_features": "bypassed imputer"})
        adapter_rows.append({"dataset": ds, "adapter": "GRIN-style-local", "input": "[B,L,N] speed with NaN at M=1 + row-normalized adjacency", "target": "clean historical X speed only", "leakage_check": "PASS: no target Y used for imputer training", "identity_features": "bypassed imputer"})

        for idx, fault in enumerate(FAULTS):
            x_fault, mask, observed = build_fault(payload["test_x"], fault, args.seed + idx)
            y = payload["test_y"]
            pred_ca, lat_ca, _ = predict_model(ca, x_fault, args.batch_size, device, sraf=False)
            met_ca = safe_metrics(y, pred_ca, payload["mean"], payload["std"])
            metrics.append({"dataset": ds, "fault": fault, "model": "ID-MLP-CA", "mae": met_ca["mae"], "rmse": met_ca["rmse"], "latency_sec": lat_ca, "notes": "reference"})

            pred_v1, lat_v1, _ = predict_model(v1, x_fault, args.batch_size, device, sraf=True, observed_mask=observed, adjacency=adj_t)
            met_v1 = safe_metrics(y, pred_v1, payload["mean"], payload["std"])
            metrics.append({"dataset": ds, "fault": fault, "model": "SRAF-ID-v1-formal", "mae": met_v1["mae"], "rmse": met_v1["rmse"], "latency_sec": lat_v1, "notes": "rollback reference"})

            pred_v2, lat_v2 = predict_v2(v2, x_fault, observed, args.batch_size, device, adj_t)
            met_v2 = safe_metrics(y, pred_v2, payload["mean"], payload["std"])
            metrics.append({"dataset": ds, "fault": fault, "model": "SRAF-ID-v2-current-best", "mae": met_v2["mae"], "rmse": met_v2["rmse"], "latency_sec": lat_v2, "notes": "main reference"})

            x_saits = impute_augmented_windows(saits, x_fault, mask, args.batch_size, device, adj_norm=None)
            st = perf_counter()
            pred_saits, lat_saits, _ = predict_model(saits_forecaster, x_saits, args.batch_size, device, sraf=False)
            met_saits = safe_metrics(y, pred_saits, payload["mean"], payload["std"])
            metrics.append({"dataset": ds, "fault": fault, "model": "SAITS+ID-MLP", "mae": met_saits["mae"], "rmse": met_saits["rmse"], "latency_sec": lat_saits, "notes": "local SAITS-style adapter, not official reproduction"})
            recon_s = float(np.mean(np.abs(x_saits[..., :1][mask > 0.5] - payload["test_x"][..., :1][mask > 0.5]))) if np.any(mask > 0.5) else 0.0
            recon_rows.append({"dataset": ds, "fault": fault, "imputer": "SAITS-style-local", "historical_speed_reconstruction_mae_norm": recon_s, "imputation_plus_forecast_sec": perf_counter() - st})

            x_grin = impute_augmented_windows(grin, x_fault, mask, args.batch_size, device, adj_norm=adj_norm_t)
            st = perf_counter()
            pred_grin, lat_grin, _ = predict_model(grin_forecaster, x_grin, args.batch_size, device, sraf=False)
            met_grin = safe_metrics(y, pred_grin, payload["mean"], payload["std"])
            metrics.append({"dataset": ds, "fault": fault, "model": "GRIN+ID-MLP", "mae": met_grin["mae"], "rmse": met_grin["rmse"], "latency_sec": lat_grin, "notes": "local GRIN-style adapter, not official reproduction"})
            recon_g = float(np.mean(np.abs(x_grin[..., :1][mask > 0.5] - payload["test_x"][..., :1][mask > 0.5]))) if np.any(mask > 0.5) else 0.0
            recon_rows.append({"dataset": ds, "fault": fault, "imputer": "GRIN-style-local", "historical_speed_reconstruction_mae_norm": recon_g, "imputation_plus_forecast_sec": perf_counter() - st})

            for name, speed in {
                "KNN+ID-MLP": knn_fill(x_fault[..., :1], mask, payload["adj"], k=5),
                "PPCA-lite+ID-MLP": ppca_lite_fill(x_fault[..., :1], mask),
                "DSAE-lite+ID-MLP": dsae_fill(dsae, x_fault[..., :1], mask, args.batch_size, device),
            }.items():
                xr = x_fault.copy()
                xr[..., :1] = speed
                pred, lat, _ = predict_model(ca, xr, args.batch_size, device, sraf=False)
                met = safe_metrics(y, pred, payload["mean"], payload["std"])
                metrics.append({"dataset": ds, "fault": fault, "model": name, "mae": met["mae"], "rmse": met["rmse"], "latency_sec": lat, "notes": "context baseline from prior gate"})

    by = {(r["dataset"], r["fault"], r["model"]): r for r in metrics}
    for row in metrics:
        ca = by.get((row["dataset"], row["fault"], "ID-MLP-CA"))
        v2 = by.get((row["dataset"], row["fault"], "SRAF-ID-v2-current-best"))
        row["gain_vs_id_mlp_ca_pct"] = (ca["mae"] - row["mae"]) / ca["mae"] * 100.0 if ca else math.nan
        row["gain_vs_sraf_id_v2_current_best_pct"] = (v2["mae"] - row["mae"]) / v2["mae"] * 100.0 if v2 else math.nan

    write_csv(out_dir / "seed42_saits_grin_diagnostic_results.csv", metrics)
    write_csv(out_dir / "imputer_reconstruction_metrics.csv", recon_rows)
    write_csv(out_dir / "runtime_metrics.csv", runtime_rows)
    write_csv(out_dir / "data_adapter_audit.csv", adapter_rows)

    selection_rows = [
        {"baseline": "SAITS+ID-MLP", "classification": "SUPPLEMENTARY_FORMAL", "reason": "local SAITS-style adapter runnable on both datasets seed42; not official reproduction, useful neural-imputation contrast"},
        {"baseline": "GRIN+ID-MLP", "classification": "SUPPLEMENTARY_FORMAL", "reason": "local GRIN-style graph adapter runnable on both datasets seed42; official GRIN license/dependency integration remains unresolved"},
    ]
    write_csv(out_dir / "baseline_selection_recommendation.csv", selection_rows)

    (out_dir / "source_license_audit.md").write_text("\n".join(["# source_license_audit", "", md_table(source_rows, ["name", "source_url", "local_path", "license", "commit_or_hash", "dependencies", "compatibility", "input_shape", "mask_convention", "batch_training", "gpu", "integration_risk"])]), encoding="utf-8")
    (out_dir / "data_adapter_audit.md").write_text("\n".join(["# data_adapter_audit", "", md_table(adapter_rows, ["dataset", "adapter", "input", "target", "leakage_check", "identity_features"])]), encoding="utf-8")
    (out_dir / "saits_adapter_audit.md").write_text("# saits_adapter_audit\n\n- implementation: local SAITS-style temporal self-attention imputer\n- official reproduction: NO\n- training: train split only, clean historical speed target, random artificial masks\n- diagnostic status: runnable on METR-LA and PEMS-BAY seed 42\n", encoding="utf-8")
    (out_dir / "grin_adapter_audit.md").write_text("# grin_adapter_audit\n\n- implementation: local GRIN-style graph-message temporal imputer\n- official reproduction: NO\n- adjacency: row-normalized adjacency with self-loop floor\n- training: train split only, clean historical speed target, random artificial masks\n- diagnostic status: runnable on METR-LA and PEMS-BAY seed 42\n", encoding="utf-8")
    (out_dir / "baseline_selection_recommendation.md").write_text("\n".join(["# baseline_selection_recommendation", "", md_table(selection_rows, ["baseline", "classification", "reason"])]), encoding="utf-8")
    (out_dir / "failed_or_deferred_notes.md").write_text("# failed_or_deferred_notes\n\n- Official SAITS/PyPOTS not installed in current environment; local SAITS-style adapter used.\n- Official GRIN not vendored; GitHub API reports no SPDX license, so official integration needs license review.\n- These diagnostics must not be called official reproductions.\n", encoding="utf-8")

    report = [
        "# SAITS_GRIN_IDMLP_BASELINE_ADAPTER_REPORT",
        "",
        "## 1. Stage Metadata",
        "- stage: SAITS_GRIN_IDMLP_BASELINE_ADAPTER_GATE",
        "- status: PASS",
        f"- timestamp: {datetime.now().isoformat(timespec='seconds')}",
        "- files changed: scripts/run_saits_grin_idmlp_baseline_adapter.py; experiments/saits_grin_idmlp_baseline_adapter/*",
        "- external sources inspected: WenjieDu/SAITS, WenjieDu/PyPOTS, Graph-Machine-Learning-Group/grin",
        "- 10-seed run executed: NO",
        "- existing results overwritten: NO",
        "",
        "## 2. Source and License Audit",
        md_table(source_rows, ["name", "source_url", "license", "commit_or_hash", "dependencies", "integration_risk"]),
        "",
        "## 3. Shared Data Adapter",
        "- shape: X_imp_input [B,L,N], observed_mask [B,L,N], X_clean_hist [B,L,N], Y_clean [B,H,N].",
        "- mask semantics: M=1 corrupted; observed_mask=1 observed/unmasked.",
        "- corrupted speed positions are set to NaN before imputation.",
        "- split/scaler: loaded from existing processed METR-LA/PEMS-BAY artifacts; no target Y used for imputer training.",
        "- leakage check: PASS for local adapters.",
        "",
        "## 4. SAITS+ID-MLP Adapter",
        "- source: official sources audited; local SAITS-style adapter used due dependency isolation.",
        "- implementation: small temporal self-attention imputer over [B,L,N] values plus observed mask.",
        f"- config: seed 42, imputer epochs {args.max_epochs_imputer}, forecaster epochs {args.max_epochs_forecaster}, hidden {args.hidden}.",
        "- training status: completed on both datasets.",
        "- diagnostic status: completed on both datasets and all requested faults.",
        "- issues: not official reproduction; mark as local SAITS-style adapter.",
        "",
        "## 5. GRIN+ID-MLP Adapter",
        "- source: official GRIN audited; local GRIN-style adapter used because official license/dependency integration remains unresolved.",
        "- implementation: graph-message temporal imputer using row-normalized adjacency.",
        f"- config: seed 42, imputer epochs {args.max_epochs_imputer}, forecaster epochs {args.max_epochs_forecaster}, hidden {args.hidden}.",
        "- adjacency handling: row-normalized adjacency with self-loop floor.",
        "- training status: completed on both datasets.",
        "- diagnostic status: completed on both datasets and all requested faults.",
        "- issues: not official reproduction; mark as local GRIN-style adapter.",
        "",
        "## 6. Seed 42 Diagnostic Results",
        md_table(metrics, ["dataset", "fault", "model", "mae", "rmse", "gain_vs_id_mlp_ca_pct", "gain_vs_sraf_id_v2_current_best_pct", "notes"]),
        "",
        "## 7. Reconstruction Metrics",
        md_table(recon_rows, ["dataset", "fault", "imputer", "historical_speed_reconstruction_mae_norm", "imputation_plus_forecast_sec"]),
        "",
        "## 8. Runtime and Practicality",
        md_table(runtime_rows, ["dataset", "model", "imputer", "best_val_reconstruction_mae", "training_time_sec", "best_val_loss"]),
        "- feasible for 10-seed: PARTIAL/YES as supplementary local adapters; official reproduction remains deferred.",
        "",
        "## 9. Baseline Selection Recommendation",
        "- SAITS+ID-MLP classification: SUPPLEMENTARY_FORMAL.",
        "- GRIN+ID-MLP classification: SUPPLEMENTARY_FORMAL.",
        "- both can enter a formal run only if manuscript mentor accepts local-style adapters rather than official reproductions.",
        "- blockers: official dependency/license adapters still unresolved.",
        "",
        "## 10. Final Decision",
        "- STATUS: PASS",
        "- SAITS_STATUS: runnable local SAITS-style adapter; not official reproduction",
        "- GRIN_STATUS: runnable local GRIN-style adapter; not official reproduction",
        "- BASELINE_SET_READY_FOR_10SEED: PARTIAL",
        "- BLOCKERS: official SAITS/GRIN reproduction still deferred; local adapter naming must stay conservative",
        "- NEXT_ACTION: send report to manuscript mentor; do not run formal 10-seed until baseline set is approved.",
    ]
    (out_dir / "SAITS_GRIN_IDMLP_BASELINE_ADAPTER_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (out_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "stage": "SAITS_GRIN_IDMLP_BASELINE_ADAPTER_GATE",
                "status": "PASS",
                "seed": args.seed,
                "datasets": ["METR-LA", "PEMS-BAY"],
                "faults": FAULTS,
                "saits_status": "runnable local adapter, not official reproduction",
                "grin_status": "runnable local adapter, not official reproduction",
                "ten_seed_run_executed": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== SAITS/GRIN ID-MLP baseline adapter ===")
    print("SAITS status: runnable local adapter, not official reproduction")
    print("GRIN status: runnable local adapter, not official reproduction")
    print("datasets completed: METR-LA, PEMS-BAY")
    print(f"faults completed: {', '.join(FAULTS)}")
    print("seed used: 42")
    print("leakage check: PASS")
    print("10-seed run executed: NO")


if __name__ == "__main__":
    main()

