"""Seed-42 baseline reproduction/selection diagnostics for SRAF-ID-v2 publication planning."""

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
    predict_model,
    train_official_stid_ca,
    train_sraf_stid,
)
from scripts.run_metr_la_strong_clean_backbone_integration import (  # noqa: E402
    add_stid_identity_features,
    apply_fault,
    load_scale,
    load_split,
    resolve_device,
)
from scripts.run_pems_bay_sraf_id_transfer import add_pems_identity_features, load_json, safe_metrics as safe_metrics_pems  # noqa: E402
from scripts.run_sraf_v2_version_freeze_and_multi_direction_exploration import build_official_stid, build_v1, build_v2, predict_v2, train_v2  # noqa: E402


FAULT_SPECS = {
    "clean": {"fault": "clean", "label": "clean"},
    "random_missing_40": {"fault": "random_missing", "rate": 0.40, "label": "random_missing_40"},
    "continuous_outage_24": {"fault": "continuous_outage", "length": 24, "label": "continuous_outage_24"},
    "gaussian_noise_high": {"fault": "gaussian_noise", "severity": "high", "label": "gaussian_noise_high"},
    "linear_drift_high": {"fault": "linear_drift", "severity": "high", "label": "linear_drift_high"},
    "stuck_at_last_value_high": {"fault": "stuck_at_last_value", "severity": "high", "label": "stuck_at_last_value_high"},
}

BASELINES = [
    "ID-MLP-CA",
    "SRAF-ID-v1-formal",
    "SRAF-ID-v2-current-best",
    "MeanFill+ID-MLP",
    "ForwardFill+ID-MLP",
    "SpatialAvg+ID-MLP",
    "TemporalSpatialAvg+ID-MLP",
    "KNN+ID-MLP",
    "PPCA-lite+ID-MLP",
    "PMM-lite+ID-MLP",
    "DSAE-lite+ID-MLP",
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


def safe_metrics(y_true_norm: np.ndarray, y_pred_norm: np.ndarray, mean: float, std: float) -> dict[str, float]:
    return safe_metrics_pems(y_true_norm, y_pred_norm, mean, std)


def load_payload(dataset: str, train_limit: int, val_limit: int, test_limit: int | None) -> dict[str, Any]:
    if dataset == "METR-LA":
        data_dir = ROOT / "data/processed/metr-la"
        train_x, train_y = load_split(data_dir, "train")
        val_x, val_y = load_split(data_dir, "val")
        test_x, test_y = load_split(data_dir, "test")
        mean, std = load_scale(data_dir)
        adj = np.load(data_dir / "adjacency.npy").astype(np.float32)
        train_x, train_y = train_x[:train_limit], train_y[:train_limit]
        val_x, val_y = val_x[:val_limit], val_y[:val_limit]
        if test_limit:
            test_x, test_y = test_x[:test_limit], test_y[:test_limit]
        return {
            "dataset": dataset,
            "train_x": add_stid_identity_features(train_x, 0),
            "train_y": train_y,
            "val_x": add_stid_identity_features(val_x, train_x.shape[0]),
            "val_y": val_y,
            "test_x": add_stid_identity_features(test_x, train_x.shape[0] + val_x.shape[0]),
            "test_y": test_y,
            "mean": float(mean),
            "std": float(std),
            "adj": adj,
        }
    data_dir = ROOT / "data/processed/pems-bay"
    meta = load_json(data_dir / "time_metadata.json")
    offsets = meta.get("split_start_indices", {"train": 0, "val": 36465, "test": 41674})
    train_x, train_y = load_split(data_dir, "train")
    val_x, val_y = load_split(data_dir, "val")
    test_x, test_y = load_split(data_dir, "test")
    train_x, train_y = train_x[:train_limit], train_y[:train_limit]
    val_x, val_y = val_x[:val_limit], val_y[:val_limit]
    if test_limit:
        test_x, test_y = test_x[:test_limit], test_y[:test_limit]
    stats = load_json(data_dir / "dataset_stats.json")
    return {
        "dataset": dataset,
        "train_x": add_pems_identity_features(train_x, int(offsets["train"]), meta),
        "train_y": train_y,
        "val_x": add_pems_identity_features(val_x, int(offsets["val"]), meta),
        "val_y": val_y,
        "test_x": add_pems_identity_features(test_x, int(offsets["test"]), meta),
        "test_y": test_y,
        "mean": float(stats["mean"]),
        "std": float(stats["std"]),
        "adj": np.load(data_dir / "adjacency.npy").astype(np.float32),
    }


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


def fill_simple(speed_fault: np.ndarray, mask: np.ndarray, mode: str, adjacency: np.ndarray, train_mean: float = 0.0) -> np.ndarray:
    out = speed_fault.copy()
    if mode == "mean":
        out = np.nan_to_num(out, nan=train_mean)
        out[mask > 0.5] = train_mean
        return out.astype(np.float32)
    if mode == "ffill":
        out = np.nan_to_num(out, nan=train_mean)
        for t in range(1, out.shape[1]):
            m = mask[:, t] > 0.5
            out[:, t][m] = out[:, t - 1][m]
        return out.astype(np.float32)
    if mode == "spatial":
        x0 = np.nan_to_num(out, nan=0.0)
        denom = np.clip(adjacency.sum(axis=1, keepdims=True), 1.0e-6, None).reshape(1, 1, adjacency.shape[0], 1)
        sp = np.einsum("ij,btjf->btif", adjacency, x0) / denom
        x0[mask > 0.5] = sp[mask > 0.5]
        return x0.astype(np.float32)
    if mode == "temp_spatial":
        return (0.5 * fill_simple(speed_fault, mask, "ffill", adjacency, train_mean) + 0.5 * fill_simple(speed_fault, mask, "spatial", adjacency, train_mean)).astype(np.float32)
    raise ValueError(mode)


def knn_fill(speed_fault: np.ndarray, mask: np.ndarray, adjacency: np.ndarray, k: int = 5) -> np.ndarray:
    x = np.nan_to_num(speed_fault.copy(), nan=0.0)
    neighbors = np.argsort(-adjacency, axis=1)[:, :k]
    for i in range(adjacency.shape[0]):
        nbr = neighbors[i]
        weights = adjacency[i, nbr]
        weights = weights / max(float(np.sum(weights)), 1.0e-6)
        fill = np.sum(x[:, :, nbr, 0] * weights.reshape(1, 1, -1), axis=-1)
        mi = mask[:, :, i, 0] > 0.5
        x[:, :, i, 0][mi] = fill[mi]
    return x.astype(np.float32)


def ppca_lite_fill(speed_fault: np.ndarray, mask: np.ndarray, rank: int = 8, iters: int = 5) -> np.ndarray:
    shape = speed_fault.shape
    x = speed_fault[..., 0].reshape(-1, shape[2]).astype(np.float32)
    m = mask[..., 0].reshape(-1, shape[2]) > 0.5
    x[~np.isfinite(x)] = np.nan
    col_mean = np.nanmean(x, axis=0)
    col_mean = np.nan_to_num(col_mean, nan=0.0)
    filled = np.where(np.isnan(x) | m, col_mean.reshape(1, -1), x)
    observed = ~(np.isnan(x) | m)
    r = min(rank, max(1, min(filled.shape) - 1))
    for _ in range(iters):
        mu = filled.mean(axis=0, keepdims=True)
        centered = filled - mu
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        recon = (u[:, :r] * s[:r]) @ vt[:r] + mu
        filled[~observed] = recon[~observed]
    return filled.reshape(shape[0], shape[1], shape[2], 1).astype(np.float32)


def pmm_lite_fill(speed_fault: np.ndarray, mask: np.ndarray, seed: int, donor_k: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.nan_to_num(speed_fault.copy(), nan=0.0)
    b, l, n, _ = x.shape
    sensor_mean = np.nanmean(np.where(mask > 0.5, np.nan, speed_fault), axis=(0, 1, 3))
    sensor_mean = np.nan_to_num(sensor_mean, nan=0.0)
    for i in range(n):
        similar = np.argsort(np.abs(sensor_mean - sensor_mean[i]))[:donor_k]
        donor_values = x[:, :, similar, 0]
        donor_mean = donor_values.mean(axis=-1)
        jitter = rng.normal(0.0, 0.01, size=donor_mean.shape)
        mi = mask[:, :, i, 0] > 0.5
        x[:, :, i, 0][mi] = (donor_mean + jitter)[mi]
    return x.astype(np.float32)


class DSAELite(nn.Module):
    def __init__(self, sensors: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sensors, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, sensors),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_dsae_lite(train_speed: np.ndarray, args: argparse.Namespace, device: torch.device) -> DSAELite:
    sensors = train_speed.shape[2]
    model = DSAELite(sensors=sensors, hidden=args.dsae_hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    clean = train_speed[..., 0].reshape(-1, sensors).astype(np.float32)
    clean = np.nan_to_num(clean, nan=0.0)
    rng = np.random.default_rng(args.seed)
    for ep in range(args.dsae_epochs):
        idx = rng.permutation(clean.shape[0])
        for start in range(0, clean.shape[0], args.batch_size):
            rows = clean[idx[start : start + args.batch_size]]
            corrupt = rows.copy()
            drop = rng.random(corrupt.shape) < 0.2
            corrupt[drop] = 0.0
            xb = torch.from_numpy(corrupt).to(device)
            yb = torch.from_numpy(rows).to(device)
            pred = model(xb)
            loss = torch.mean(torch.abs(pred - yb))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def dsae_fill(model: DSAELite, speed_fault: np.ndarray, mask: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    shape = speed_fault.shape
    x = np.nan_to_num(speed_fault[..., 0], nan=0.0).reshape(-1, shape[2]).astype(np.float32)
    m = (mask[..., 0].reshape(-1, shape[2]) > 0.5)
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            pred = model(torch.from_numpy(x[start : start + batch_size]).to(device))
            preds.append(pred.detach().cpu().numpy())
    recon = np.concatenate(preds, axis=0)
    x[m] = recon[m]
    return x.reshape(shape[0], shape[1], shape[2], 1).astype(np.float32)


def make_v2_current_best(sensors: int, input_length: int, horizon: int) -> nn.Module:
    return build_v2(
        sensors,
        input_length,
        horizon,
        {
            "rel_hidden": 64,
            "alpha_hidden": 16,
            "adaptive_alpha": True,
            "stuck_features": True,
            "flatness": False,
            "second_delta": True,
            "repair_disagreement": True,
            "base_rel_only": False,
        },
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="experiments/sraf_v2_publication_baseline_reproduction_and_selection")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.002)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--lambda-repair", type=float, default=0.05)
    p.add_argument("--lambda-rel", type=float, default=0.01)
    p.add_argument("--loss", choices=["mae", "mse"], default="mae")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--train-limit", type=int, default=512)
    p.add_argument("--val-limit", type=int, default=128)
    p.add_argument("--test-limit", type=int, default=256)
    p.add_argument("--dsae-epochs", type=int, default=2)
    p.add_argument("--dsae-hidden", type=int, default=64)
    return p.parse_args()


def md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep] + body)


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
        {"baseline": "GRIN+ID-MLP", "repo": "https://github.com/Graph-Machine-Learning-Group/grin", "commit": "4a28afbb092600b6e6abeeabaaf67e87dbd1ed6e", "license": "NEEDS REVIEW (GitHub API license null)", "status": "DEFERRED", "reason": "official graph-imputation training stack not integrated; needs dependency/license review"},
        {"baseline": "SAITS+ID-MLP", "repo": "https://github.com/WenjieDu/SAITS; https://github.com/WenjieDu/PyPOTS", "commit": "SAITS 660b87f19c1277065f314f24134f646229e89ca9; PyPOTS 012d45617ea6894e61a80f4b04ecb66aa5fddbb4", "license": "MIT / BSD-3-Clause", "status": "DEFERRED", "reason": "PyPOTS not installed; official package integration requires separate environment validation"},
        {"baseline": "BRITS+ID-MLP", "repo": "https://github.com/caow13/BRITS", "commit": "fc0a3a472a6d99a6934e471d40c25ed0b029b501", "license": "MIT", "status": "DEFERRED", "reason": "legacy recurrent imputer input format needs adapter; not integrated in this seed42 gate"},
        {"baseline": "Official STID-clean/STID-CA", "repo": "https://github.com/GestaltCogTeam/STID", "commit": "e8b313bc591bdd0101a1619962c9b503e75127c0", "license": "Apache-2.0", "status": "DEFERRED", "reason": "official repo not cloned/adapted; current ID-MLP is manuscript-facing identity backbone, not official reproduction"},
        {"baseline": "DCNN-GAN/DSAE/GAN/BGCP", "repo": "literature-specific sources pending", "commit": "N/A", "license": "NEEDS REVIEW", "status": "PARTIAL", "reason": "implemented DSAE-lite internal diagnostic only; no official DCNN-GAN/BGCP reproduction"},
    ]
    write_csv(out_dir / "external_code_source_and_license_audit.csv", source_rows)

    payloads = {ds: load_payload(ds, args.train_limit, args.val_limit, args.test_limit) for ds in ["METR-LA", "PEMS-BAY"]}
    adj_t = {ds: torch.from_numpy(payload["adj"]).to(device) for ds, payload in payloads.items()}
    metrics: list[dict[str, Any]] = []
    train_meta: list[dict[str, Any]] = []

    for ds, payload in payloads.items():
        sensors = payload["train_x"].shape[2]
        input_length = payload["train_x"].shape[1]
        horizon = payload["train_y"].shape[1]
        ds_dir = out_dir / "models" / ds.lower().replace("-", "_")
        ds_dir.mkdir(parents=True, exist_ok=True)

        ca = build_official_stid(sensors, input_length, horizon)
        start = perf_counter()
        ca_dir = ds_dir / "id_mlp_ca"
        ca_dir.mkdir(parents=True, exist_ok=True)
        meta_ca, _ = train_official_stid_ca(ca, payload["train_x"], payload["train_y"], payload["val_x"], payload["val_y"], args, ca_dir, device)
        train_meta.append({"dataset": ds, "model": "ID-MLP-CA", **meta_ca, "wall_sec": perf_counter() - start})

        v1 = build_v1(sensors, input_length, horizon)
        start = perf_counter()
        v1_dir = ds_dir / "sraf_id_v1_formal"
        v1_dir.mkdir(parents=True, exist_ok=True)
        meta_v1, _ = train_sraf_stid(v1, "SRAF-ID-v1-formal", payload["train_x"], payload["train_y"], payload["val_x"], payload["val_y"], args, v1_dir, device, adj_t[ds])
        train_meta.append({"dataset": ds, "model": "SRAF-ID-v1-formal", **meta_v1, "wall_sec": perf_counter() - start})

        v2 = make_v2_current_best(sensors, input_length, horizon)
        start = perf_counter()
        v2_dir = ds_dir / "sraf_id_v2_current_best"
        v2_dir.mkdir(parents=True, exist_ok=True)
        meta_v2, _ = train_v2(v2, "SRAF-ID-v2-current-best", payload["train_x"], payload["train_y"], payload["val_x"], payload["val_y"], args, v2_dir, device, adj_t[ds])
        train_meta.append({"dataset": ds, "model": "SRAF-ID-v2-current-best", **meta_v2, "wall_sec": perf_counter() - start})

        dsae = train_dsae_lite(payload["train_x"][..., :1], args, device)

        for fault_idx, fault in enumerate(FAULT_SPECS):
            x_fault, mask, observed = build_fault(payload["test_x"], fault, args.seed + fault_idx)
            y = payload["test_y"]

            pred_ca, lat_ca, _ = predict_model(ca, x_fault, args.batch_size, device, sraf=False)
            met_ca = safe_metrics(y, pred_ca, payload["mean"], payload["std"])
            row_ca = {"dataset": ds, "fault": fault, "baseline": "ID-MLP-CA", "mae": met_ca["mae"], "rmse": met_ca["rmse"], "latency_sec": lat_ca, "notes": "reference in-gate diagnostic model"}
            metrics.append(row_ca)

            pred_v1, lat_v1, _ = predict_model(v1, x_fault, args.batch_size, device, sraf=True, observed_mask=observed, adjacency=adj_t[ds])
            met_v1 = safe_metrics(y, pred_v1, payload["mean"], payload["std"])
            metrics.append({"dataset": ds, "fault": fault, "baseline": "SRAF-ID-v1-formal", "mae": met_v1["mae"], "rmse": met_v1["rmse"], "latency_sec": lat_v1, "notes": "rollback reference; in-gate diagnostic model"})

            pred_v2, lat_v2 = predict_v2(v2, x_fault, observed, args.batch_size, device, adj_t[ds])
            met_v2 = safe_metrics(y, pred_v2, payload["mean"], payload["std"])
            metrics.append({"dataset": ds, "fault": fault, "baseline": "SRAF-ID-v2-current-best", "mae": met_v2["mae"], "rmse": met_v2["rmse"], "latency_sec": lat_v2, "notes": "main reference; in-gate diagnostic model"})

            fill_map = {
                "MeanFill+ID-MLP": fill_simple(x_fault[..., :1], mask, "mean", payload["adj"]),
                "ForwardFill+ID-MLP": fill_simple(x_fault[..., :1], mask, "ffill", payload["adj"]),
                "SpatialAvg+ID-MLP": fill_simple(x_fault[..., :1], mask, "spatial", payload["adj"]),
                "TemporalSpatialAvg+ID-MLP": fill_simple(x_fault[..., :1], mask, "temp_spatial", payload["adj"]),
                "KNN+ID-MLP": knn_fill(x_fault[..., :1], mask, payload["adj"], k=5),
                "PPCA-lite+ID-MLP": ppca_lite_fill(x_fault[..., :1], mask),
                "PMM-lite+ID-MLP": pmm_lite_fill(x_fault[..., :1], mask, seed=args.seed + fault_idx),
                "DSAE-lite+ID-MLP": dsae_fill(dsae, x_fault[..., :1], mask, args.batch_size, device),
            }
            for name, speed_rep in fill_map.items():
                xr = x_fault.copy()
                xr[..., :1] = speed_rep
                pred, lat, _ = predict_model(ca, xr, args.batch_size, device, sraf=False)
                met = safe_metrics(y, pred, payload["mean"], payload["std"])
                metrics.append({"dataset": ds, "fault": fault, "baseline": name, "mae": met["mae"], "rmse": met["rmse"], "latency_sec": lat, "notes": "impute-then-ID-MLP diagnostic"})

    by = {(r["dataset"], r["fault"], r["baseline"]): r for r in metrics}
    for r in metrics:
        ca = by.get((r["dataset"], r["fault"], "ID-MLP-CA"))
        v2 = by.get((r["dataset"], r["fault"], "SRAF-ID-v2-current-best"))
        r["gain_vs_id_mlp_ca_pct"] = (ca["mae"] - r["mae"]) / ca["mae"] * 100.0 if ca else math.nan
        r["gain_vs_sraf_id_v2_current_best_pct"] = (v2["mae"] - r["mae"]) / v2["mae"] * 100.0 if v2 else math.nan

    write_csv(out_dir / "seed42_baseline_diagnostic_results.csv", metrics)
    write_csv(out_dir / "training_meta.csv", train_meta)

    decisions = [
        {"baseline": "ID-MLP-CA", "class": "MAIN_FORMAL", "reason": "same-backbone corruption-aware reference"},
        {"baseline": "SRAF-ID-v1-formal", "class": "MAIN_FORMAL", "reason": "rollback and v1-v2 delta reference"},
        {"baseline": "SRAF-ID-v2-current-best", "class": "MAIN_FORMAL", "reason": "main candidate"},
        {"baseline": "KNN+ID-MLP", "class": "MAIN_FORMAL", "reason": "low-cost traffic sensor imputation baseline, runnable"},
        {"baseline": "PPCA-lite+ID-MLP", "class": "SUPPLEMENTARY_FORMAL", "reason": "PPCA-style low-rank imputation attempted and runnable; lite implementation should be labeled internal"},
        {"baseline": "MeanFill+ID-MLP", "class": "SUPPLEMENTARY_FORMAL", "reason": "simple lower-bound fill baseline"},
        {"baseline": "ForwardFill+ID-MLP", "class": "SUPPLEMENTARY_FORMAL", "reason": "temporal repair contrast"},
        {"baseline": "SpatialAvg+ID-MLP", "class": "SUPPLEMENTARY_FORMAL", "reason": "spatial repair contrast"},
        {"baseline": "TemporalSpatialAvg+ID-MLP", "class": "SUPPLEMENTARY_FORMAL", "reason": "fixed temporal-spatial repair contrast"},
        {"baseline": "PMM-lite+ID-MLP", "class": "DIAGNOSTIC_ONLY", "reason": "runnable but simplified PMM approximation; needs stronger literature-matched implementation for formal use"},
        {"baseline": "DSAE-lite+ID-MLP", "class": "DIAGNOSTIC_ONLY", "reason": "runnable internal neural imputer; not official DCNN-GAN/DSAE reproduction"},
        {"baseline": "GRIN+ID-MLP", "class": "DEFERRED", "reason": "official integration/license/dependency review pending"},
        {"baseline": "SAITS+ID-MLP", "class": "DEFERRED", "reason": "official/PyPOTS package unavailable in current environment"},
        {"baseline": "BRITS+ID-MLP", "class": "DEFERRED", "reason": "legacy adapter pending"},
        {"baseline": "Official STID-clean/STID-CA", "class": "DEFERRED", "reason": "official repo not adapted; avoid official reproduction claim"},
        {"baseline": "DCNN-GAN/BGCP", "class": "DEFERRED", "reason": "source/license/implementation unresolved"},
    ]
    write_csv(out_dir / "baseline_selection_decision.csv", decisions)

    ledger_lines = [
        "# BASELINE_REPRODUCTION_LEDGER",
        "",
        f"- timestamp: {datetime.now().isoformat(timespec='seconds')}",
        "- seed: 42 only",
        f"- train_limit: {args.train_limit}",
        f"- val_limit: {args.val_limit}",
        f"- test_limit: {args.test_limit}",
        "- formal 10-seed run executed: NO",
        "",
        md_table(decisions, ["baseline", "class", "reason"]),
    ]
    (out_dir / "BASELINE_REPRODUCTION_LEDGER.md").write_text("\n".join(ledger_lines), encoding="utf-8")

    (out_dir / "EXTERNAL_CODE_SOURCE_AND_LICENSE_AUDIT.md").write_text(
        "\n".join(["# EXTERNAL_CODE_SOURCE_AND_LICENSE_AUDIT", "", md_table(source_rows, ["baseline", "repo", "commit", "license", "status", "reason"])]),
        encoding="utf-8",
    )
    (out_dir / "BASELINE_ADAPTATION_PROTOCOL.md").write_text(
        "\n".join(
            [
                "# BASELINE_ADAPTATION_PROTOCOL",
                "",
                "- All runnable baselines use processed METR-LA/PEMS-BAY splits from `data/processed`.",
                "- Faults are applied only to input speed channel; target Y remains clean.",
                "- Identity features remain clean.",
                "- L=12 and H=12 are inherited from processed arrays.",
                "- Metrics are MAE/RMSE using dataset scaler.",
                "- This gate uses capped seed-42 diagnostic data only; outputs are not manuscript claims.",
                "- Impute-then-forecast baselines feed repaired speed plus clean identity features into the same ID-MLP-CA forecaster trained in this gate.",
            ]
        ),
        encoding="utf-8",
    )
    (out_dir / "baseline_selection_decision.md").write_text(
        "\n".join(["# baseline_selection_decision", "", md_table(decisions, ["baseline", "class", "reason"])]),
        encoding="utf-8",
    )
    deferred = [d for d in decisions if d["class"] in {"DEFERRED", "EXCLUDE"}]
    (out_dir / "failed_or_deferred_baselines.md").write_text(
        "\n".join(["# failed_or_deferred_baselines", "", md_table(deferred, ["baseline", "class", "reason"])]),
        encoding="utf-8",
    )

    status = "PARTIAL"
    report = [
        "# SRAF_V2_PUBLICATION_BASELINE_REPRODUCTION_AND_SELECTION_REPORT",
        "",
        "## 1. Stage Metadata",
        "- stage: SRAF_V2_PUBLICATION_BASELINE_REPRODUCTION_AND_SELECTION_GATE",
        f"- status: {status}",
        f"- timestamp: {datetime.now().isoformat(timespec='seconds')}",
        "- files changed: scripts/run_sraf_v2_publication_baseline_reproduction_and_selection.py; experiments/sraf_v2_publication_baseline_reproduction_and_selection/*",
        "- external repos inspected: GRIN, SAITS/PyPOTS, BRITS, official STID",
        f"- baselines attempted: {len(BASELINES)} runnable/internal plus external audit set",
        "- baselines runnable: ID-MLP-CA, SRAF-ID-v1-formal, SRAF-ID-v2-current-best, MeanFill, ForwardFill, SpatialAvg, TemporalSpatialAvg, KNN, PPCA-lite, PMM-lite, DSAE-lite",
        "- 10-seed run executed: NO",
        "- existing results overwritten: NO",
        "",
        "## 2. Sensors/MDPI Baseline Motivation",
        "- PPCA-style low-rank imputation covers classical traffic missing-data reconstruction.",
        "- KNN/PMM-style methods cover simple sensor-neighbor and donor-value imputation traditions.",
        "- DCNN-GAN/DSAE/GAN/BGCP-style methods are relevant to Sensors missing-data baselines; this gate only ran DSAE-lite internally.",
        "- FCM-like temporal/spatial repair modes are represented by ForwardFill, SpatialAvg, and TemporalSpatialAvg.",
        "- GRIN/SAITS/BRITS represent stronger neural imputation families; official integration remains deferred rather than fabricated.",
        "- Official STID is relevant for identity forecasting, but current code should not be called official STID reproduction.",
        "",
        "## 3. Baseline Implementation Status",
        md_table(
            [
                {"baseline": d["baseline"], "source/repo": next((s["repo"] for s in source_rows if s["baseline"] == d["baseline"]), "internal"), "license": next((s["license"] for s in source_rows if s["baseline"] == d["baseline"]), "project internal"), "implemented": "yes" if d["class"] != "DEFERRED" else "no", "runnable": "yes" if d["class"] not in {"DEFERRED", "EXCLUDE"} else "no", "adaptation_summary": d["reason"], "fairness_risk": "medium" if "lite" in d["baseline"] else "low", "runtime_risk": "medium" if d["class"] == "DEFERRED" else "low", "decision": d["class"]}
                for d in decisions
            ],
            ["baseline", "source/repo", "license", "implemented", "runnable", "adaptation_summary", "fairness_risk", "runtime_risk", "decision"],
        ),
        "",
        "## 4. Seed 42 Diagnostic Results",
        md_table(metrics, ["dataset", "fault", "baseline", "mae", "rmse", "gain_vs_id_mlp_ca_pct", "gain_vs_sraf_id_v2_current_best_pct", "notes"]),
        "",
        "## 5. Baseline Selection for Formal Run",
        md_table(decisions, ["baseline", "class", "reason"]),
        "",
        "## 6. Formal 10-Seed Readiness",
        "- baseline set ready for formal 10-seed: PARTIAL",
        "- recommended required model list: ID-MLP-CA, SRAF-ID-v1-formal, SRAF-ID-v2-current-best, KNN+ID-MLP, PPCA-lite+ID-MLP.",
        "- supplementary candidates: MeanFill, ForwardFill, SpatialAvg, TemporalSpatialAvg; PMM-lite and DSAE-lite diagnostic only unless mentor accepts simplified baselines.",
        "- estimated required run count without supplementary simple baselines: 2 datasets x 6 faults x 5 models x 10 seeds = 600 runs.",
        "- blockers: no official GRIN/SAITS/BRITS runnable adapter yet; official STID not adapted; PPCA is lite/internal rather than official ppca-em package.",
        "",
        "## 7. Final Decision",
        "- STATUS: PARTIAL",
        "- BASELINE_SET_READY_FOR_10SEED: PARTIAL",
        "- REQUIRED_MAIN_BASELINES: ID-MLP-CA; SRAF-ID-v1-formal; SRAF-ID-v2-current-best; KNN+ID-MLP; PPCA-lite+ID-MLP",
        "- SUPPLEMENTARY_BASELINES: MeanFill+ID-MLP; ForwardFill+ID-MLP; SpatialAvg+ID-MLP; TemporalSpatialAvg+ID-MLP",
        "- DEFERRED_BASELINES: GRIN+ID-MLP; SAITS+ID-MLP; BRITS+ID-MLP; Official STID-clean/STID-CA; DCNN-GAN/BGCP",
        "- BLOCKERS: external neural baseline official adapters not runnable in this gate; official STID not adapted; formal 10-seed still not authorized.",
        "- NEXT_ACTION: manuscript mentor reviews baseline selection; do not run formal 10-seed until approved.",
    ]
    (out_dir / "SRAF_V2_PUBLICATION_BASELINE_REPRODUCTION_AND_SELECTION_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    manifest = {
        "stage": "SRAF_V2_PUBLICATION_BASELINE_REPRODUCTION_AND_SELECTION_GATE",
        "status": status,
        "seed": args.seed,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "test_limit": args.test_limit,
        "epochs": args.epochs,
        "dsae_epochs": args.dsae_epochs,
        "baselines": BASELINES,
        "formal_10seed_run_executed": False,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    runnable = [d["baseline"] for d in decisions if d["class"] not in {"DEFERRED", "EXCLUDE"}]
    print("=== SRAF V2 publication baseline selection ===")
    print(f"baselines attempted: {len(decisions)}")
    print(f"baselines runnable: {len(runnable)}")
    print("PPCA status: PPCA-lite runnable; official ppca-em deferred")
    print("neural imputation baseline status: DSAE-lite runnable; GRIN/SAITS/BRITS deferred")
    print("official STID status: deferred")
    print("10-seed run executed: NO")
    print(f"status: {status}")


if __name__ == "__main__":
    main()
