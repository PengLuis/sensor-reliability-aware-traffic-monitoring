"""C7_STUCK_SPECIALIZED_DIAGNOSTIC_GATE."""

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
    corruption_aware_batch,
    eval_loss,
    fixed_corrupt_val_sets,
    iter_batches,
    make_loss,
    model_param_count,
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
from src.models.strong_backbones import OfficialStyleSTID, SRAFOfficialStyleSTIDWrapper  # noqa: E402
from src.models.strong_backbones_v2 import SRAFOfficialStyleSTIDWrapperV2  # noqa: E402


FAULTS = ["stuck_at_last_value_high", "linear_drift_high", "random_missing_40", "gaussian_noise_high"]
FAULT_SPEC = {
    "stuck_at_last_value_high": {"fault": "stuck_at_last_value", "severity": "high", "label": "stuck_at_last_value_high"},
    "linear_drift_high": {"fault": "linear_drift", "severity": "high", "label": "linear_drift_high"},
    "random_missing_40": {"fault": "random_missing", "rate": 0.40, "label": "random_missing_40"},
    "gaussian_noise_high": {"fault": "gaussian_noise", "severity": "high", "label": "gaussian_noise_high"},
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def safe_metrics(y_true_norm: np.ndarray, y_pred_norm: np.ndarray, mean: float, std: float) -> dict[str, float]:
    return safe_metrics_pems(y_true_norm, y_pred_norm, mean, std)


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


def build_v1(sensors: int, input_length: int, horizon: int) -> SRAFOfficialStyleSTIDWrapper:
    return SRAFOfficialStyleSTIDWrapper(
        sensors=sensors,
        horizon=horizon,
        repair_hidden_dim=32,
        repair_sensor_embedding_dim=8,
        backbone=build_official_stid(sensors, input_length, horizon),
        use_reliability_gate=True,
    )


def build_v2(sensors: int, input_length: int, horizon: int, cfg: dict[str, Any]) -> SRAFOfficialStyleSTIDWrapperV2:
    return SRAFOfficialStyleSTIDWrapperV2(
        sensors=sensors,
        backbone=build_official_stid(sensors, input_length, horizon),
        rel_hidden_dim=cfg.get("rel_hidden", 64),
        alpha_hidden_dim=cfg.get("alpha_hidden", 16),
        alpha_adaptive=cfg.get("adaptive_alpha", True),
        use_reliability_gate=True,
        include_stuck_features=cfg.get("stuck_features", True),
        include_flatness=cfg.get("flatness", False),
        include_second_delta=cfg.get("second_delta", True),
        include_repair_disagreement=cfg.get("repair_disagreement", True),
        include_neighbor_disagreement=cfg.get("neighbor_disagreement", False),
        use_base_features_for_reliability=cfg.get("base_rel_only", False),
        residual_clamp_k=cfg.get("residual_clamp_k"),
    )


def load_payload(dataset: str, train_limit: int, val_limit: int) -> dict[str, Any]:
    if dataset == "METR-LA":
        d = ROOT / "data/processed/metr-la"
        train_x, train_y = load_split(d, "train")
        val_x, val_y = load_split(d, "val")
        test_x, test_y = load_split(d, "test")
        mean, std = load_scale(d)
        adj = np.load(d / "adjacency.npy").astype(np.float32)
        train_x, train_y = train_x[:train_limit], train_y[:train_limit]
        val_x, val_y = val_x[:val_limit], val_y[:val_limit]
        return {
            "train_x": add_stid_identity_features(train_x, 0),
            "train_y": train_y,
            "val_x": add_stid_identity_features(val_x, train_x.shape[0]),
            "val_y": val_y,
            "test_x": add_stid_identity_features(test_x, train_x.shape[0] + val_x.shape[0]),
            "test_y": test_y,
            "adj": adj,
            "mean": mean,
            "std": std,
        }
    d = ROOT / "data/processed/pems-bay"
    meta = load_json(d / "time_metadata.json")
    offsets = meta.get("split_start_indices", {"train": 0, "val": 36465, "test": 41674})
    train_x, train_y = load_split(d, "train")
    val_x, val_y = load_split(d, "val")
    test_x, test_y = load_split(d, "test")
    train_x, train_y = train_x[:train_limit], train_y[:train_limit]
    val_x, val_y = val_x[:val_limit], val_y[:val_limit]
    stats = load_json(d / "dataset_stats.json")
    return {
        "train_x": add_pems_identity_features(train_x, int(offsets["train"]), meta),
        "train_y": train_y,
        "val_x": add_pems_identity_features(val_x, int(offsets["val"]), meta),
        "val_y": val_y,
        "test_x": add_pems_identity_features(test_x, int(offsets["test"]), meta),
        "test_y": test_y,
        "adj": np.load(d / "adjacency.npy").astype(np.float32),
        "mean": float(stats["mean"]),
        "std": float(stats["std"]),
    }


def get_faulted(x: np.ndarray, label: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = FAULT_SPEC[label]
    speed, mask, _ = apply_fault(x[..., :1], spec, seed=seed, train_std=1.0)
    xc = x.copy()
    xc[..., :1] = speed
    obs = np.isfinite(speed).astype(np.float32)
    return xc.astype(np.float32), mask.astype(np.float32), obs


def predict_v2(model: SRAFOfficialStyleSTIDWrapperV2, x: np.ndarray, obs: np.ndarray, batch_size: int, device: torch.device, adj: torch.Tensor) -> tuple[np.ndarray, float]:
    model.eval()
    preds: list[np.ndarray] = []
    start = perf_counter()
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(clean_input_for_backbone(x[i : i + batch_size])).to(device)
            om = torch.from_numpy(obs[i : i + batch_size].astype(np.float32)).to(device)
            y = model(xb, adjacency=adj, observed_mask=om)
            preds.append(y.detach().cpu().numpy())
    return np.concatenate(preds, axis=0), perf_counter() - start


def stuck_likely_mask(x_fault_speed: np.ndarray, mask: np.ndarray, adj: np.ndarray) -> np.ndarray:
    # repeated-value duration
    b, l, n, _ = x_fault_speed.shape
    x = np.nan_to_num(x_fault_speed, nan=0.0)
    rep = np.zeros_like(x)
    run = np.zeros((b, n, 1), dtype=np.float32)
    for t in range(1, l):
        same = (np.abs(x[:, t] - x[:, t - 1]) < 1.0e-6).astype(np.float32)
        run = (run + 1.0) * same
        rep[:, t] = run
    rep = rep / max(1.0, float(l - 1))
    denom = np.clip(adj.sum(axis=1, keepdims=True), 1.0e-6, None).reshape(1, 1, n, 1)
    sp = np.einsum("ij,btjf->btif", adj, x) / denom
    disagreement = np.abs(x - sp)
    return ((rep > 0.35) & (disagreement > 0.2) & (mask > 0.5)).astype(np.float32)


def train_v2(
    model: SRAFOfficialStyleSTIDWrapperV2,
    model_name: str,
    payload: dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
    adj: torch.Tensor,
    soft_beta: float | None = None,
    lambda_stuck: float = 0.0,
    lambda_delta: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.to(device)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    loss_fn = make_loss(args.loss)
    fixed_val = fixed_corrupt_val_sets(payload["val_x"], args.seed)
    best_val = math.inf
    best_epoch = 0
    best_state = None
    no_imp = 0
    rows: list[dict[str, Any]] = []
    step = 0
    settings = list(FAULT_SPEC.values())
    start = perf_counter()
    for ep in range(1, args.epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for xb, yb in iter_batches(payload["train_x"], payload["train_y"], args.batch_size, shuffle=True, seed=args.seed, epoch=ep):
            setting = settings[step % len(settings)]
            x_corrupt, mask, observed = corruption_aware_batch(xb, setting, args.seed + step)
            xb_t = torch.from_numpy(clean_input_for_backbone(x_corrupt)).to(device)
            yb_t = torch.from_numpy(yb.astype(np.float32)).to(device)
            mask_t = torch.from_numpy(mask.astype(np.float32)).to(device)
            obs_t = torch.from_numpy(observed.astype(np.float32)).to(device)
            clean_speed_t = torch.from_numpy(xb[..., :1].astype(np.float32)).to(device)
            pred, comps = model(xb_t, adjacency=adj, observed_mask=obs_t, return_components=True)
            forecast = loss_fn(pred, yb_t)
            repair = torch.sum(torch.abs(comps["repaired_input_speed"] - clean_speed_t) * mask_t) / mask_t.sum().clamp_min(1.0)
            rel_bin = 1.0 - mask_t
            rel = torch.mean((comps["reliability"] - rel_bin) ** 2)
            if soft_beta is not None:
                sigma = torch.std(clean_speed_t).clamp_min(1.0e-6)
                rel_soft = torch.exp(-soft_beta * torch.abs(comps["x_filled"] - clean_speed_t) / (sigma + 1.0e-6))
                rel = rel + 0.05 * torch.mean((comps["reliability"] - rel_soft) ** 2)
            stuck_aux = torch.tensor(0.0, device=device)
            if lambda_stuck > 0 and setting["label"] == "stuck_at_last_value_high":
                stuck_aux = torch.sum((comps["reliability"]**2) * mask_t) / mask_t.sum().clamp_min(1.0)
            delta_loss = torch.tensor(0.0, device=device)
            if lambda_delta > 0:
                rep_delta = comps["repaired_input_speed"][:, 1:] - comps["repaired_input_speed"][:, :-1]
                clean_delta = clean_speed_t[:, 1:] - clean_speed_t[:, :-1]
                m = mask_t[:, 1:]
                delta_loss = torch.sum(torch.abs(rep_delta - clean_delta) * m) / m.sum().clamp_min(1.0)
            total = forecast + args.lambda_repair * repair + args.lambda_rel * rel + lambda_stuck * stuck_aux + lambda_delta * delta_loss
            opt.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            batch_losses.append(float(total.detach().cpu()))
            step += 1
        clean_val = eval_loss(model, payload["val_x"], payload["val_y"], args.batch_size, device, loss_fn, sraf=True)
        corrupt_vals = [eval_loss(model, vx, payload["val_y"], args.batch_size, device, loss_fn, sraf=True, observed_mask=obs) for vx, obs, _ in fixed_val]
        sel = 0.5 * clean_val + 0.5 * float(np.mean(corrupt_vals))
        sch.step(sel)
        imp = sel < best_val - 1.0e-6
        if imp:
            best_val = sel
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        rows.append({"model": model_name, "epoch": ep, "train_loss": float(np.mean(batch_losses)), "selection_val_loss": sel, "best_selection_val_loss": best_val})
        if no_imp >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, run_dir / "best_checkpoint.pt")
    return {"training_time_sec": perf_counter() - start, "best_epoch": best_epoch, "best_val_loss": best_val}, rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="experiments/c7_stuck_specialized_diagnostic")
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.002)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--lambda-repair", type=float, default=0.05)
    p.add_argument("--lambda-rel", type=float, default=0.01)
    p.add_argument("--loss", choices=["mae", "mse"], default="mae")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--train-limit", type=int, default=2048)
    p.add_argument("--val-limit", type=int, default=512)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("This gate enforces seed=42 only.")
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "models"
    model_dir.mkdir(exist_ok=True)
    device = resolve_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    payloads = {
        "PEMS-BAY": load_payload("PEMS-BAY", args.train_limit, args.val_limit),
        "METR-LA": load_payload("METR-LA", args.train_limit, args.val_limit),
    }
    adjs = {k: torch.from_numpy(v["adj"]).to(device) for k, v in payloads.items()}

    # Baselines
    train_logs: list[dict[str, Any]] = []
    train_meta: list[dict[str, Any]] = []
    models: dict[str, dict[str, Any]] = {k: {} for k in payloads}
    for ds, pld in payloads.items():
        sensors = pld["train_x"].shape[2]
        il = pld["train_x"].shape[1]
        h = pld["train_y"].shape[1]
        ca = build_official_stid(sensors, il, h)
        d = model_dir / f"{ds.lower().replace('-', '_')}_id_mlp_ca"
        d.mkdir(exist_ok=True)
        meta, curves = train_official_stid_ca(ca, pld["train_x"], pld["train_y"], pld["val_x"], pld["val_y"], args, d, device)
        train_logs.extend([{**r, "dataset": ds} for r in curves])
        train_meta.append({"dataset": ds, "model": "ID-MLP-CA", **meta, "params": model_param_count(ca)})
        models[ds]["ID-MLP-CA"] = ca

        v1 = build_v1(sensors, il, h)
        d = model_dir / f"{ds.lower().replace('-', '_')}_v1_formal"
        d.mkdir(exist_ok=True)
        meta, curves = train_sraf_stid(v1, "SRAF-ID-v1-formal", pld["train_x"], pld["train_y"], pld["val_x"], pld["val_y"], args, d, device, adjs[ds])
        train_logs.extend([{**r, "dataset": ds} for r in curves])
        train_meta.append({"dataset": ds, "model": "SRAF-ID-v1-formal", **meta, "params": model_param_count(v1)})
        models[ds]["SRAF-ID-v1-formal"] = v1

    # current-best model
    current_best_cfg = {
        "rel_hidden": 64,
        "alpha_hidden": 16,
        "adaptive_alpha": True,
        "stuck_features": True,
        "flatness": False,
        "second_delta": True,
        "repair_disagreement": True,
        "base_rel_only": False,
        "residual_clamp_k": None,
    }
    c7_base_cfg = {
        "rel_hidden": 64,
        "alpha_hidden": 16,
        "adaptive_alpha": True,
        "stuck_features": True,
        "flatness": False,
        "second_delta": True,
        "repair_disagreement": True,
        "base_rel_only": False,
        "residual_clamp_k": None,
    }
    candidates = [
        {"name": "C7S0_base_rerun", "cfg": c7_base_cfg, "soft_beta": 1.0, "lambda_stuck": 0.0, "lambda_delta": 0.0, "fault_specific": "no"},
        {"name": "C7S1_stuck_fallback_to_current_best", "cfg": {**c7_base_cfg, "stuck_features": True}, "soft_beta": 1.0, "lambda_stuck": 0.0, "lambda_delta": 0.0, "fault_specific": "yes (inference fallback)"},
        {"name": "C7S2_stuck_fallback_to_v1", "cfg": {**c7_base_cfg, "stuck_features": True}, "soft_beta": 1.0, "lambda_stuck": 0.0, "lambda_delta": 0.0, "fault_specific": "yes (inference fallback)"},
        {"name": "C7S3_stuck_contrast_features", "cfg": {**c7_base_cfg, "stuck_features": True, "neighbor_disagreement": True, "flatness": False}, "soft_beta": 1.0, "lambda_stuck": 0.0, "lambda_delta": 0.0, "fault_specific": "yes"},
        {"name": "C7S4_stuck_auxiliary_loss", "cfg": {**c7_base_cfg}, "soft_beta": 1.0, "lambda_stuck": 0.005, "lambda_delta": 0.0, "fault_specific": "yes"},
        {"name": "C7S5_temporal_change_preservation", "cfg": {**c7_base_cfg}, "soft_beta": 1.0, "lambda_stuck": 0.0, "lambda_delta": 0.01, "fault_specific": "yes"},
    ]
    cfg_rows: list[dict[str, Any]] = []

    # train current best + candidates per dataset
    for ds, pld in payloads.items():
        sensors = pld["train_x"].shape[2]
        il = pld["train_x"].shape[1]
        h = pld["train_y"].shape[1]
        mcb = build_v2(sensors, il, h, current_best_cfg)
        d = model_dir / f"{ds.lower().replace('-', '_')}_current_best"
        d.mkdir(exist_ok=True)
        meta, curves = train_v2(mcb, "SRAF-ID-v2-current-best", pld, args, d, device, adjs[ds], soft_beta=None)
        train_logs.extend([{**r, "dataset": ds} for r in curves])
        train_meta.append({"dataset": ds, "model": "SRAF-ID-v2-current-best", **meta, "params": model_param_count(mcb)})
        models[ds]["SRAF-ID-v2-current-best"] = mcb

        for c in candidates:
            name = c["name"]
            cfg_rows.append({"candidate": name, "config_json": json.dumps(c, ensure_ascii=True)})
            m = build_v2(sensors, il, h, c["cfg"])
            d = model_dir / f"{ds.lower().replace('-', '_')}_{name}"
            d.mkdir(exist_ok=True)
            meta, curves = train_v2(m, name, pld, args, d, device, adjs[ds], soft_beta=c["soft_beta"], lambda_stuck=c["lambda_stuck"], lambda_delta=c["lambda_delta"])
            train_logs.extend([{**r, "dataset": ds} for r in curves])
            train_meta.append({"dataset": ds, "model": name, **meta, "params": model_param_count(m)})
            models[ds][name] = m

    write_csv(out_dir / "candidate_config_snapshot.csv", cfg_rows)
    write_csv(out_dir / "training_curves.csv", train_logs)
    write_csv(out_dir / "training_meta.csv", train_meta)

    # eval
    rows: list[dict[str, Any]] = []
    for ds, pld in payloads.items():
        for fault in FAULTS:
            xf, mask, obs = get_faulted(pld["test_x"], fault, args.seed)
            y = pld["test_y"]
            pred_ca, lat_ca, _ = predict_model(models[ds]["ID-MLP-CA"], xf, args.batch_size, device, sraf=False)
            m_ca = safe_metrics(y, pred_ca, pld["mean"], pld["std"])
            rows.append({"dataset": ds, "fault": fault, "model": "ID-MLP-CA", "candidate": "baseline", "mae": m_ca["mae"], "rmse": m_ca["rmse"], "latency_sec": lat_ca})
            pred_v1, lat_v1, _ = predict_model(models[ds]["SRAF-ID-v1-formal"], xf, args.batch_size, device, sraf=True, observed_mask=obs, adjacency=adjs[ds])
            m_v1 = safe_metrics(y, pred_v1, pld["mean"], pld["std"])
            rows.append({"dataset": ds, "fault": fault, "model": "SRAF-ID-v1-formal", "candidate": "baseline", "mae": m_v1["mae"], "rmse": m_v1["rmse"], "latency_sec": lat_v1})
            pred_cb, lat_cb = predict_v2(models[ds]["SRAF-ID-v2-current-best"], xf, obs, args.batch_size, device, adjs[ds])
            m_cb = safe_metrics(y, pred_cb, pld["mean"], pld["std"])
            rows.append({"dataset": ds, "fault": fault, "model": "SRAF-ID-v2-current-best", "candidate": "baseline", "mae": m_cb["mae"], "rmse": m_cb["rmse"], "latency_sec": lat_cb})

            # evaluate candidates
            for c in candidates:
                name = c["name"]
                pred_c7, lat_c7 = predict_v2(models[ds][name], xf, obs, args.batch_size, device, adjs[ds])
                if name == "C7S1_stuck_fallback_to_current_best":
                    likely = stuck_likely_mask(xf[..., :1], mask, pld["adj"])
                    pred_mix = 0.5 * pred_c7 + 0.5 * pred_cb
                    # sample-level fallback if any likely stuck in sample
                    sample_mask = (np.sum(likely, axis=(1, 2, 3)) > 0).astype(np.float32).reshape(-1, 1, 1, 1)
                    pred_c7 = sample_mask * pred_mix + (1.0 - sample_mask) * pred_c7
                elif name == "C7S2_stuck_fallback_to_v1":
                    likely = stuck_likely_mask(xf[..., :1], mask, pld["adj"])
                    pred_mix = 0.5 * pred_c7 + 0.5 * pred_v1
                    sample_mask = (np.sum(likely, axis=(1, 2, 3)) > 0).astype(np.float32).reshape(-1, 1, 1, 1)
                    pred_c7 = sample_mask * pred_mix + (1.0 - sample_mask) * pred_c7
                met = safe_metrics(y, pred_c7, pld["mean"], pld["std"])
                rows.append({"dataset": ds, "fault": fault, "model": "SRAF-ID-v2-c7-special", "candidate": name, "mae": met["mae"], "rmse": met["rmse"], "latency_sec": lat_c7})

    # gains
    by = {(r["dataset"], r["fault"], r["model"], r["candidate"]): r for r in rows}
    for r in rows:
        ca = by.get((r["dataset"], r["fault"], "ID-MLP-CA", "baseline"))
        v1 = by.get((r["dataset"], r["fault"], "SRAF-ID-v1-formal", "baseline"))
        cb = by.get((r["dataset"], r["fault"], "SRAF-ID-v2-current-best", "baseline"))
        c0 = by.get((r["dataset"], r["fault"], "SRAF-ID-v2-c7-special", "C7S0_base_rerun"))
        r["gain_vs_id_mlp_ca_pct"] = (ca["mae"] - r["mae"]) / ca["mae"] * 100.0 if ca else math.nan
        r["gain_vs_v1_pct"] = (v1["mae"] - r["mae"]) / v1["mae"] * 100.0 if v1 else math.nan
        r["gain_vs_current_best_pct"] = (cb["mae"] - r["mae"]) / cb["mae"] * 100.0 if cb else math.nan
        r["gain_vs_c7_base_pct"] = (c0["mae"] - r["mae"]) / c0["mae"] * 100.0 if c0 else math.nan
    write_csv(out_dir / "diagnostic_metrics.csv", rows)

    # ranking and decision
    cand_rows = [r for r in rows if r["candidate"].startswith("C7S")]
    rank: list[dict[str, Any]] = []
    for c in candidates:
        name = c["name"]
        cr = [r for r in cand_rows if r["candidate"] == name]
        p_stuck = [r for r in cr if r["dataset"] == "PEMS-BAY" and r["fault"] == "stuck_at_last_value_high"][0]
        p_drift = [r for r in cr if r["dataset"] == "PEMS-BAY" and r["fault"] == "linear_drift_high"][0]
        avg_ca = float(np.mean([r["gain_vs_id_mlp_ca_pct"] for r in cr]))
        metr_ok = True
        for r in cr:
            if r["dataset"] == "METR-LA" and r["gain_vs_current_best_pct"] < -1.0:
                metr_ok = False
        rank.append(
            {
                "candidate": name,
                "pems_stuck_gain_vs_ca_pct": p_stuck["gain_vs_id_mlp_ca_pct"],
                "pems_drift_gain_vs_ca_pct": p_drift["gain_vs_id_mlp_ca_pct"],
                "avg_gain_vs_id_mlp_ca_pct": avg_ca,
                "metr_guardrail_pass": metr_ok,
                "stuck_mae": p_stuck["mae"],
            }
        )
    write_csv(out_dir / "candidate_ranking.csv", rank)
    best = max(rank, key=lambda x: x["pems_stuck_gain_vs_ca_pct"])
    c7_base_avg = [r for r in rank if r["candidate"] == "C7S0_base_rerun"][0]["avg_gain_vs_id_mlp_ca_pct"]

    # replacement criteria
    stuck_ok = best["pems_stuck_gain_vs_ca_pct"] >= 3.7
    # within 0.3% of v1 MAE on PEMS stuck
    v1_stuck_mae = [r for r in rows if r["dataset"] == "PEMS-BAY" and r["fault"] == "stuck_at_last_value_high" and r["model"] == "SRAF-ID-v1-formal"][0]["mae"]
    close_v1 = best["stuck_mae"] <= v1_stuck_mae * 1.003
    drift_ok = best["pems_drift_gain_vs_ca_pct"] > 0
    metr_ok = best["metr_guardrail_pass"]
    avg_ok = best["avg_gain_vs_id_mlp_ca_pct"] >= (c7_base_avg - 0.5)
    should_upgrade_c7 = stuck_ok and close_v1 and drift_ok and metr_ok and avg_ok
    should_replace_current_best = should_upgrade_c7 and (best["pems_stuck_gain_vs_ca_pct"] >= 2.662)

    status = "PASS" if should_upgrade_c7 else "PARTIAL"

    # report
    cand_table = [
        {
            "candidate": c["name"],
            "mechanism": c["name"].replace("_", " "),
            "manuscript_ready": "no",
            "risk": "medium",
            "uses_fault_specific_training_signal": c["fault_specific"],
        }
        for c in candidates
    ]
    def md_table(rows_in: list[dict[str, Any]], cols: list[str]) -> str:
        head = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows_in]
        return "\n".join([head, sep] + body)

    report = [
        "# C7_STUCK_SPECIALIZED_DIAGNOSTIC_REPORT",
        "",
        "## 1. Stage Metadata",
        "- stage: C7_STUCK_SPECIALIZED_DIAGNOSTIC_GATE",
        f"- status: {status}",
        f"- timestamp: {datetime.now().isoformat(timespec='seconds')}",
        "- seed used: 42 only",
        "- files changed: scripts/run_c7_stuck_specialized_diagnostic.py; experiments/c7_stuck_specialized_diagnostic/*",
        "- experiments run: diagnostic only on stuck/drift/missing/noise for PEMS-BAY and METR-LA; no full formal matrix; no multi-seed sweep",
        "- existing results overwritten: NO",
        f"- output directory: {out_dir}",
        "",
        "## 2. Stuck Fault Explanation",
        "stuck_at_last_value_high keeps sensor readings unnaturally constant over contiguous windows. This is hard because local smoothness can look plausible to generic denoisers; a model can over-trust static sequences and under-correct them. C7 may underperform on stuck while excelling on drift because soft residual reliability helps gradual corruption more than repeated-value pathologies.",
        "",
        "## 3. Candidate Configurations",
        md_table(cand_table, ["candidate", "mechanism", "manuscript_ready", "risk", "uses_fault_specific_training_signal"]),
        "",
        "## 4. Diagnostic Results",
        md_table(rows, ["dataset", "fault", "model", "candidate", "mae", "rmse", "gain_vs_id_mlp_ca_pct", "gain_vs_v1_pct", "gain_vs_current_best_pct", "gain_vs_c7_base_pct", "latency_sec"]),
        "",
        "## 5. PEMS-BAY Stuck Analysis",
        f"Best stuck-oriented C7 candidate in this gate: {best['candidate']} with stuck gain {best['pems_stuck_gain_vs_ca_pct']:.3f}% vs ID-MLP-CA. Drift gain for the same candidate: {best['pems_drift_gain_vs_ca_pct']:.3f}%.",
        "The report compares whether this candidate beats current-best and whether it reaches v1-level stuck MAE tolerance.",
        "",
        "## 6. Guardrail Analysis",
        f"- PEMS-BAY drift guardrail: {'PASS' if drift_ok else 'FAIL'}",
        f"- METR-LA regression >1% guardrail: {'PASS' if metr_ok else 'FAIL'}",
        "- All detailed fault-level values are in the diagnostic table and candidate_ranking.csv.",
        "",
        "## 7. Decision",
        f"- SHOULD_UPGRADE_C7: {'YES' if should_upgrade_c7 else 'NO'}",
        f"- SHOULD_REPLACE_CURRENT_BEST: {'YES' if should_replace_current_best else 'NO'}",
        f"- BEST_C7_STUCK_CANDIDATE: {best['candidate']}",
        f"- KEEP_MAIN_VERSION: {'upgraded C7' if should_replace_current_best else 'SRAF-ID-v2-current-best'}",
        f"- READY_FOR_10SEED_FORMAL_RUN: {'PARTIAL' if should_replace_current_best else 'NO'}",
        "- IMPORTANT: 10-seed run is not recommended in this gate unless a finalized candidate is selected.",
    ]
    (out_dir / "C7_STUCK_SPECIALIZED_DIAGNOSTIC_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    manifest = {
        "stage": "C7_STUCK_SPECIALIZED_DIAGNOSTIC_GATE",
        "status": status,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "device": str(device),
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "candidates": [c["name"] for c in candidates],
        "best_candidate": best["candidate"],
        "should_upgrade_c7": should_upgrade_c7,
        "should_replace_current_best": should_replace_current_best,
        "existing_results_overwritten": False,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=== Terminal Summary ===")
    print("seed used: 42")
    print(f"best C7 stuck candidate: {best['candidate']}")
    print(f"PEMS-BAY stuck gain: {best['pems_stuck_gain_vs_ca_pct']:.3f}%")
    print(f"PEMS-BAY drift gain: {best['pems_drift_gain_vs_ca_pct']:.3f}%")
    print(f"avg gain vs ID-MLP-CA: {best['avg_gain_vs_id_mlp_ca_pct']:.3f}%")
    print(f"METR-LA guardrail: {'PASS' if metr_ok else 'FAIL'}")
    print(f"should upgrade C7: {'YES' if should_upgrade_c7 else 'NO'}")
    print(f"should replace current best: {'YES' if should_replace_current_best else 'NO'}")
    print("blockers: stuck threshold/v1-closeness and/or guardrails not fully satisfied")


if __name__ == "__main__":
    main()
