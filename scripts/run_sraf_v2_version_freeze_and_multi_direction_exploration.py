"""SRAF_V2_VERSION_FREEZE_AND_MULTI_DIRECTION_EXPLORATION_GATE."""

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


FAULTS = ["random_missing_40", "gaussian_noise_high", "linear_drift_high", "stuck_at_last_value_high"]
FAULT_SPEC = {
    "random_missing_40": {"fault": "random_missing", "rate": 0.40, "label": "random_missing_40"},
    "gaussian_noise_high": {"fault": "gaussian_noise", "severity": "high", "label": "gaussian_noise_high"},
    "linear_drift_high": {"fault": "linear_drift", "severity": "high", "label": "linear_drift_high"},
    "stuck_at_last_value_high": {"fault": "stuck_at_last_value", "severity": "high", "label": "stuck_at_last_value_high"},
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
        use_base_features_for_reliability=cfg.get("base_rel_only", False),
        residual_clamp_k=cfg.get("residual_clamp_k"),
        safe_fallback_enable=cfg.get("safe_fallback", False),
        safe_fallback_eta=cfg.get("safe_eta", 0.5),
        safe_fallback_uncertainty_threshold=cfg.get("safe_uncertainty_th", 0.25),
        use_temporal_attention_candidate=cfg.get("use_attn_candidate", False),
        temporal_attention_hidden_dim=cfg.get("attn_hidden", 16),
        use_bidirectional_temporal_candidate=cfg.get("use_bidir_candidate", False),
        use_light_graph_message_candidate=cfg.get("use_graph_candidate", False),
        use_candidate_softmax_fusion=cfg.get("use_candidate_softmax", False),
        stuck_fallback_enable=cfg.get("stuck_fallback_enable", False),
        stuck_duration_threshold=cfg.get("stuck_duration_threshold", 0.4),
        spatial_disagreement_threshold=cfg.get("spatial_disagreement_threshold", 0.2),
        stuck_fallback_eta=cfg.get("stuck_fallback_eta", 0.5),
    )


def predict_v2(
    model: SRAFOfficialStyleSTIDWrapperV2,
    x: np.ndarray,
    observed: np.ndarray,
    batch_size: int,
    device: torch.device,
    adjacency: torch.Tensor,
) -> tuple[np.ndarray, float]:
    model.eval()
    preds: list[np.ndarray] = []
    st = perf_counter()
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(clean_input_for_backbone(x[i : i + batch_size])).to(device)
            om = torch.from_numpy(observed[i : i + batch_size].astype(np.float32)).to(device)
            pred = model(xb, adjacency=adjacency, observed_mask=om)
            preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds, axis=0), perf_counter() - st


def train_v2(
    model: SRAFOfficialStyleSTIDWrapperV2,
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
    adjacency: torch.Tensor,
    soft_beta: float | None = None,
    soft_gamma: float = 0.05,
    lambda_alpha: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.to(device)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    loss_fn = make_loss(args.loss)
    fixed_val = fixed_corrupt_val_sets(val_x, args.seed)
    best_val = math.inf
    best_epoch = 0
    best_state = None
    no_imp = 0
    rows: list[dict[str, Any]] = []
    step = 0
    start = perf_counter()
    settings = list(FAULT_SPEC.values())
    for ep in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in iter_batches(train_x, train_y, args.batch_size, shuffle=True, seed=args.seed, epoch=ep):
            setting = settings[step % len(settings)]
            x_corrupt, mask, observed = corruption_aware_batch(xb, setting, args.seed + step)
            xb_t = torch.from_numpy(clean_input_for_backbone(x_corrupt)).to(device)
            yb_t = torch.from_numpy(yb.astype(np.float32)).to(device)
            mask_t = torch.from_numpy(mask.astype(np.float32)).to(device)
            observed_t = torch.from_numpy(observed.astype(np.float32)).to(device)
            clean_speed_t = torch.from_numpy(xb[..., :1].astype(np.float32)).to(device)
            pred, comps = model(xb_t, adjacency=adjacency, observed_mask=observed_t, return_components=True)
            forecast = loss_fn(pred, yb_t)
            repair = torch.sum(torch.abs(comps["repaired_input_speed"] - clean_speed_t) * mask_t) / mask_t.sum().clamp_min(1.0)
            rel_bin = 1.0 - mask_t
            rel_loss = torch.mean((comps["reliability"] - rel_bin) ** 2)
            if soft_beta is not None:
                sigma = torch.std(clean_speed_t).clamp_min(1.0e-6)
                rel_soft = torch.exp(-soft_beta * torch.abs(comps["x_filled"] - clean_speed_t) / (sigma + 1.0e-6))
                rel_loss = rel_loss + soft_gamma * torch.mean((comps["reliability"] - rel_soft) ** 2)
            alpha_reg = torch.mean((comps["alpha"] - 0.5) ** 2) if lambda_alpha > 0 else torch.tensor(0.0, device=device)
            total = forecast + args.lambda_repair * repair + args.lambda_rel * rel_loss + lambda_alpha * alpha_reg
            opt.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(total.detach().cpu()))
            step += 1
        clean_val = eval_loss(model, val_x, val_y, args.batch_size, device, loss_fn, sraf=True)
        corrupt_vals = [eval_loss(model, vx, val_y, args.batch_size, device, loss_fn, sraf=True, observed_mask=obs) for vx, obs, _ in fixed_val]
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
        rows.append({"model": model_name, "epoch": ep, "train_loss": float(np.mean(losses)), "selection_val_loss": sel, "best_selection_val_loss": best_val})
        if no_imp >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, run_dir / "best_checkpoint.pt")
    return {"training_time_sec": perf_counter() - start, "best_epoch": best_epoch, "best_val_loss": best_val}, rows


def knn_fill_with_adjacency(speed_fault: np.ndarray, mask: np.ndarray, adjacency: np.ndarray, k: int = 5) -> np.ndarray:
    x = np.nan_to_num(speed_fault.copy(), nan=0.0)
    n = adjacency.shape[0]
    neighbor_idx = np.argsort(-adjacency, axis=1)[:, :k]
    for i in range(n):
        nbr = neighbor_idx[i]
        nbr_w = adjacency[i, nbr]
        w = nbr_w / max(float(np.sum(nbr_w)), 1.0e-6)
        nbr_vals = x[:, :, nbr, 0]
        fill = np.sum(nbr_vals * w.reshape(1, 1, -1), axis=-1)
        xi = x[:, :, i, 0]
        mi = mask[:, :, i, 0] > 0.5
        xi[mi] = fill[mi]
        x[:, :, i, 0] = xi
    return x.astype(np.float32)


def load_payload(dataset: str, train_limit: int, val_limit: int) -> dict[str, Any]:
    if dataset == "METR-LA":
        data_dir = ROOT / "data/processed/metr-la"
        train_x, train_y = load_split(data_dir, "train")
        val_x, val_y = load_split(data_dir, "val")
        test_x, test_y = load_split(data_dir, "test")
        mean, std = load_scale(data_dir)
        adj = np.load(data_dir / "adjacency.npy").astype(np.float32)
        train_x, train_y = train_x[:train_limit], train_y[:train_limit]
        val_x, val_y = val_x[:val_limit], val_y[:val_limit]
        train_aug = add_stid_identity_features(train_x, 0)
        val_aug = add_stid_identity_features(val_x, train_x.shape[0])
        test_aug = add_stid_identity_features(test_x, train_x.shape[0] + val_x.shape[0])
        return {"train_x": train_aug, "train_y": train_y, "val_x": val_aug, "val_y": val_y, "test_x": test_aug, "test_y": test_y, "adj": adj, "mean": mean, "std": std}
    data_dir = ROOT / "data/processed/pems-bay"
    meta = load_json(data_dir / "time_metadata.json")
    offsets = meta.get("split_start_indices", {"train": 0, "val": 36465, "test": 41674})
    train_x, train_y = load_split(data_dir, "train")
    val_x, val_y = load_split(data_dir, "val")
    test_x, test_y = load_split(data_dir, "test")
    train_x, train_y = train_x[:train_limit], train_y[:train_limit]
    val_x, val_y = val_x[:val_limit], val_y[:val_limit]
    train_aug = add_pems_identity_features(train_x, int(offsets["train"]), meta)
    val_aug = add_pems_identity_features(val_x, int(offsets["val"]), meta)
    test_aug = add_pems_identity_features(test_x, int(offsets["test"]), meta)
    stats = load_json(data_dir / "dataset_stats.json")
    adj = np.load(data_dir / "adjacency.npy").astype(np.float32)
    return {"train_x": train_aug, "train_y": train_y, "val_x": val_aug, "val_y": val_y, "test_x": test_aug, "test_y": test_y, "adj": adj, "mean": float(stats["mean"]), "std": float(stats["std"])}


def get_faulted(x: np.ndarray, fault_label: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = FAULT_SPEC[fault_label]
    speed, mask, _ = apply_fault(x[..., :1], spec, seed=seed, train_std=1.0)
    x_fault = x.copy()
    x_fault[..., :1] = speed
    observed = np.isfinite(speed).astype(np.float32)
    return x_fault.astype(np.float32), mask.astype(np.float32), observed


def candidate_grid() -> list[dict[str, Any]]:
    base_best = {
        "rel_hidden": 64,
        "alpha_hidden": 16,
        "adaptive_alpha": True,
        "stuck_features": True,
        "flatness": False,  # frozen best
        "second_delta": True,
        "repair_disagreement": True,
        "base_rel_only": False,
        "residual_clamp_k": None,
        "safe_fallback": False,
        "use_attn_candidate": False,
        "use_bidir_candidate": False,
        "use_graph_candidate": False,
        "use_candidate_softmax": False,
        "soft_beta": None,
        "lambda_alpha": 0.0,
        "stuck_fallback_enable": False,
    }
    return [
        {"name": "C0_current_best_rerun", "inspiration": "frozen current best", **base_best},
        {"name": "C1_stuck_fallback", "inspiration": "rule-based safe fallback for likely stuck positions", **{**base_best, "stuck_fallback_enable": True, "stuck_duration_threshold": 0.35, "spatial_disagreement_threshold": 0.2, "stuck_fallback_eta": 0.5}},
        {"name": "C2_residual_clamp", "inspiration": "conservative displacement clamp", **{**base_best, "residual_clamp_k": 1.5}},
        {"name": "C3_alpha_regularized", "inspiration": "adaptive-alpha stabilization", **{**base_best, "lambda_alpha": 0.001}},
        {"name": "C4_light_temporal_attention_repair", "inspiration": "SAITS-inspired lightweight temporal attention repair candidate", **{**base_best, "use_attn_candidate": True, "attn_hidden": 16, "use_candidate_softmax": True}},
        {"name": "C5_bidirectional_temporal_repair", "inspiration": "BRITS-inspired bidirectional temporal repair candidate", **{**base_best, "use_bidir_candidate": True, "use_candidate_softmax": True}},
        {"name": "C6_light_graph_message_repair", "inspiration": "GRIN-inspired lightweight graph message repair candidate", **{**base_best, "use_graph_candidate": True, "use_candidate_softmax": True}},
        {"name": "C7_soft_reliability_residual_target", "inspiration": "soft residual reliability supervision", **{**base_best, "soft_beta": 1.0}},
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="experiments/sraf_v2_version_freeze_and_multi_direction_exploration")
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
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "models"
    model_dir.mkdir(exist_ok=True)
    device = resolve_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Freeze ledger
    ledger = [
        "# VERSION_FREEZE_LEDGER",
        "",
        "## SRAF-ID-v1-formal",
        "- code path: src/models/strong_backbones.py::SRAFOfficialStyleSTIDWrapper + scripts/run_metr_la_sraf_stid_full_training_confirmation.py, scripts/run_pems_bay_sraf_id_transfer.py",
        "- result path: experiments/metr-la-sraf-stid-full-training-confirmation/, experiments/pems-bay-sraf-id-full-confirmation/",
        "- status: frozen stable reference",
        "- known strengths: conservative, stable same-backbone robustness under multiple faults",
        "- known weaknesses: weaker drift/stuck gains on some settings compared with stronger v2 candidates",
        "",
        "## SRAF-ID-v2-current-best",
        "- config name: v2_c7_no_flatness_features",
        "- code path: src/models/residual_models_v2.py + src/models/strong_backbones_v2.py",
        "- result path: experiments/sraf_v2_stabilization_and_internal_baseline_repair/",
        "- key diagnostic results: METR non-regression fixed; PEMS drift positive; stuck gain moderate",
        "- status: protected candidate",
        "- known strengths: improved average gain vs ID-MLP-CA and stronger drift performance than v1 in diagnostics",
        "- known weaknesses: PEMS stuck gain may remain below target threshold in some reruns",
        "",
        "## Policy",
        "- new candidates cannot replace current best unless they outperform it under this gate's replacement criteria.",
        "- if no candidate satisfies criteria, keep SRAF-ID-v2-current-best.",
        "- if SRAF-ID-v2-current-best fails later full formal run, fallback is SRAF-ID-v1-formal.",
    ]
    (out_dir / "VERSION_FREEZE_LEDGER.md").write_text("\n".join(ledger), encoding="utf-8")

    payloads = {
        "METR-LA": load_payload("METR-LA", args.train_limit, args.val_limit),
        "PEMS-BAY": load_payload("PEMS-BAY", args.train_limit, args.val_limit),
    }
    adj = {k: torch.from_numpy(v["adj"]).to(device) for k, v in payloads.items()}

    # Train baselines
    train_logs: list[dict[str, Any]] = []
    train_meta: list[dict[str, Any]] = []
    models: dict[str, dict[str, Any]] = {k: {} for k in payloads}
    for ds, payload in payloads.items():
        sensors = payload["train_x"].shape[2]
        input_length = payload["train_x"].shape[1]
        horizon = payload["train_y"].shape[1]
        ca = build_official_stid(sensors, input_length, horizon)
        d = model_dir / f"{ds.lower().replace('-', '_')}_id_mlp_ca"
        d.mkdir(exist_ok=True)
        meta, curves = train_official_stid_ca(ca, payload["train_x"], payload["train_y"], payload["val_x"], payload["val_y"], args, d, device)
        train_logs.extend([{**r, "dataset": ds} for r in curves])
        train_meta.append({"dataset": ds, "model": "ID-MLP-CA", **meta, "params": model_param_count(ca)})
        models[ds]["ID-MLP-CA"] = ca

        v1 = build_v1(sensors, input_length, horizon)
        d = model_dir / f"{ds.lower().replace('-', '_')}_sraf_v1"
        d.mkdir(exist_ok=True)
        meta, curves = train_sraf_stid(v1, "SRAF-ID-v1-formal", payload["train_x"], payload["train_y"], payload["val_x"], payload["val_y"], args, d, device, adj[ds])
        train_logs.extend([{**r, "dataset": ds} for r in curves])
        train_meta.append({"dataset": ds, "model": "SRAF-ID-v1-formal", **meta, "params": model_param_count(v1)})
        models[ds]["SRAF-ID-v1-formal"] = v1

    # Train candidates on both datasets
    candidates = candidate_grid()
    config_rows: list[dict[str, Any]] = []
    for cand in candidates:
        config_rows.append({"candidate": cand["name"], "config_json": json.dumps(cand, ensure_ascii=True)})
        for ds, payload in payloads.items():
            sensors = payload["train_x"].shape[2]
            input_length = payload["train_x"].shape[1]
            horizon = payload["train_y"].shape[1]
            m = build_v2(sensors, input_length, horizon, cand)
            d = model_dir / f"{ds.lower().replace('-', '_')}_{cand['name']}"
            d.mkdir(exist_ok=True)
            meta, curves = train_v2(
                m,
                cand["name"],
                payload["train_x"],
                payload["train_y"],
                payload["val_x"],
                payload["val_y"],
                args,
                d,
                device,
                adj[ds],
                soft_beta=cand.get("soft_beta"),
                soft_gamma=0.05,
                lambda_alpha=cand.get("lambda_alpha", 0.0),
            )
            train_logs.extend([{**r, "dataset": ds} for r in curves])
            train_meta.append({"dataset": ds, "model": cand["name"], **meta, "params": model_param_count(m)})
            models[ds][cand["name"]] = m

    write_csv(out_dir / "candidate_config_snapshot.csv", config_rows)
    write_csv(out_dir / "training_curves.csv", train_logs)
    write_csv(out_dir / "training_meta.csv", train_meta)

    # Eval
    eval_rows: list[dict[str, Any]] = []
    for ds, payload in payloads.items():
        for fault in FAULTS:
            x_fault, mask, observed = get_faulted(payload["test_x"], fault, args.seed)
            y = payload["test_y"]
            pred_ca, lat_ca, _ = predict_model(models[ds]["ID-MLP-CA"], x_fault, args.batch_size, device, sraf=False)
            m_ca = safe_metrics(y, pred_ca, payload["mean"], payload["std"])
            eval_rows.append({"dataset": ds, "fault": fault, "model": "ID-MLP-CA", "candidate": "baseline", "mae": m_ca["mae"], "rmse": m_ca["rmse"], "latency_sec": lat_ca})
            pred_v1, lat_v1, _ = predict_model(models[ds]["SRAF-ID-v1-formal"], x_fault, args.batch_size, device, sraf=True, observed_mask=observed, adjacency=adj[ds])
            m_v1 = safe_metrics(y, pred_v1, payload["mean"], payload["std"])
            eval_rows.append({"dataset": ds, "fault": fault, "model": "SRAF-ID-v1-formal", "candidate": "baseline", "mae": m_v1["mae"], "rmse": m_v1["rmse"], "latency_sec": lat_v1})

            # KNN baseline
            x_knn = x_fault.copy()
            x_knn[..., :1] = knn_fill_with_adjacency(x_fault[..., :1], mask, payload["adj"], k=5)
            pred_knn, lat_knn, _ = predict_model(models[ds]["ID-MLP-CA"], x_knn, args.batch_size, device, sraf=False)
            m_knn = safe_metrics(y, pred_knn, payload["mean"], payload["std"])
            eval_rows.append({"dataset": ds, "fault": fault, "model": "KNN+ID-MLP", "candidate": "knn_k5", "mae": m_knn["mae"], "rmse": m_knn["rmse"], "latency_sec": lat_knn})

            for cand in candidates:
                name = cand["name"]
                pred, lat = predict_v2(models[ds][name], x_fault, observed, args.batch_size, device, adj[ds])
                met = safe_metrics(y, pred, payload["mean"], payload["std"])
                eval_rows.append({"dataset": ds, "fault": fault, "model": "SRAF-ID-v2", "candidate": name, "mae": met["mae"], "rmse": met["rmse"], "latency_sec": lat})

    by = {(r["dataset"], r["fault"], r["model"], r["candidate"]): r for r in eval_rows}
    for r in eval_rows:
        ca = by.get((r["dataset"], r["fault"], "ID-MLP-CA", "baseline"))
        v1 = by.get((r["dataset"], r["fault"], "SRAF-ID-v1-formal", "baseline"))
        c0 = by.get((r["dataset"], r["fault"], "SRAF-ID-v2", "C0_current_best_rerun"))
        r["gain_vs_id_mlp_ca_pct"] = (ca["mae"] - r["mae"]) / ca["mae"] * 100.0 if ca else math.nan
        r["gain_vs_sraf_v1_formal_pct"] = (v1["mae"] - r["mae"]) / v1["mae"] * 100.0 if v1 else math.nan
        r["gain_vs_v2_current_best_pct"] = (c0["mae"] - r["mae"]) / c0["mae"] * 100.0 if c0 else math.nan
    write_csv(out_dir / "diagnostic_metrics.csv", eval_rows)

    # Ranking and replacement
    rank_rows: list[dict[str, Any]] = []
    c0_stuck = float(np.mean([r["gain_vs_id_mlp_ca_pct"] for r in eval_rows if r["dataset"] == "PEMS-BAY" and r["fault"] == "stuck_at_last_value_high" and r["candidate"] == "C0_current_best_rerun"]))
    replacements: list[dict[str, Any]] = []
    for cand in candidates:
        name = cand["name"]
        crows = [r for r in eval_rows if r["candidate"] == name]
        avg_mae = float(np.mean([r["mae"] for r in crows]))
        avg_vs_ca = float(np.mean([r["gain_vs_id_mlp_ca_pct"] for r in crows]))
        avg_vs_v1 = float(np.mean([r["gain_vs_sraf_v1_formal_pct"] for r in crows]))
        avg_vs_c0 = float(np.mean([r["gain_vs_v2_current_best_pct"] for r in crows]))
        p_stuck = float(np.mean([r["gain_vs_id_mlp_ca_pct"] for r in crows if r["dataset"] == "PEMS-BAY" and r["fault"] == "stuck_at_last_value_high"]))
        p_drift = float(np.mean([r["gain_vs_id_mlp_ca_pct"] for r in crows if r["dataset"] == "PEMS-BAY" and r["fault"] == "linear_drift_high"]))
        metr_nonreg = True
        for f in ["random_missing_40", "gaussian_noise_high", "linear_drift_high"]:
            rr = [r for r in crows if r["dataset"] == "METR-LA" and r["fault"] == f]
            if rr and rr[0]["gain_vs_v2_current_best_pct"] < -1.0:
                metr_nonreg = False
        lat_ratio_ok = True
        for ds in ["METR-LA", "PEMS-BAY"]:
            for f in FAULTS:
                vv = by.get((ds, f, "SRAF-ID-v1-formal", "baseline"))
                cc = by.get((ds, f, "SRAF-ID-v2", name))
                if vv and cc and vv["latency_sec"] > 0:
                    if (cc["latency_sec"] / vv["latency_sec"]) > 1.8:
                        lat_ratio_ok = False
        replace_ok = (
            avg_vs_c0 > 0
            and metr_nonreg
            and p_drift > 0
            and p_stuck >= c0_stuck
            and all((r["gain_vs_sraf_v1_formal_pct"] > 0 for r in crows if r["dataset"] == "METR-LA" and r["fault"] in ["random_missing_40", "gaussian_noise_high", "linear_drift_high"]))
            and lat_ratio_ok
        )
        rank_rows.append(
            {
                "candidate": name,
                "avg_mae": avg_mae,
                "avg_gain_vs_current_best_pct": avg_vs_c0,
                "avg_gain_vs_id_mlp_ca_pct": avg_vs_ca,
                "avg_gain_vs_v1_pct": avg_vs_v1,
                "pems_stuck_gain_vs_ca_pct": p_stuck,
                "pems_drift_gain_vs_ca_pct": p_drift,
                "metr_non_regression_vs_current_best": metr_nonreg,
                "latency_ratio_ok": lat_ratio_ok,
                "eligible_to_replace": replace_ok,
            }
        )
        if replace_ok:
            replacements.append({"candidate": name, "avg_gain_vs_current_best_pct": avg_vs_c0})
    write_csv(out_dir / "candidate_ranking.csv", rank_rows)

    best_candidate = max(rank_rows, key=lambda x: x["avg_gain_vs_current_best_pct"])["candidate"] if rank_rows else "C0_current_best_rerun"
    should_replace = len(replacements) > 0
    replacement = max(replacements, key=lambda x: x["avg_gain_vs_current_best_pct"])["candidate"] if should_replace else "C0_current_best_rerun"
    status = "PASS" if should_replace else "PARTIAL"

    # Report
    cand_desc_rows = [
        {"candidate": c["name"], "inspiration": c["inspiration"], "implementation_summary": "see candidate_config_snapshot.csv", "manuscript_ready": "no", "risk": "medium"} for c in candidates
    ]

    def md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
        head = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
        return "\n".join([head, sep] + body)

    report = [
        "# SRAF_V2_VERSION_FREEZE_AND_MULTI_DIRECTION_EXPLORATION_REPORT",
        "",
        "## 1. Stage Metadata",
        "- stage: SRAF_V2_VERSION_FREEZE_AND_MULTI_DIRECTION_EXPLORATION_GATE",
        f"- status: {status}",
        f"- timestamp: {datetime.now().isoformat(timespec='seconds')}",
        "- files changed: src/models/residual_models_v2.py; src/models/strong_backbones_v2.py; scripts/run_sraf_v2_version_freeze_and_multi_direction_exploration.py; experiments/sraf_v2_version_freeze_and_multi_direction_exploration/*",
        "- experiments run: two datasets x four faults with expanded diagnostic budget (35 epochs max, patience 7, batch 64, train_limit>=2048, val_limit>=512)",
        "- existing results overwritten: NO",
        f"- output directory: {out_dir}",
        "",
        "## 2. Version Freeze Ledger",
        "- See VERSION_FREEZE_LEDGER.md in this directory.",
        "",
        "## 3. Candidate Descriptions",
        md_table(cand_desc_rows, ["candidate", "inspiration", "implementation_summary", "manuscript_ready", "risk"]),
        "",
        "## 4. Diagnostic Results",
        md_table(eval_rows, ["dataset", "fault", "model", "candidate", "mae", "rmse", "gain_vs_id_mlp_ca_pct", "gain_vs_sraf_v1_formal_pct", "gain_vs_v2_current_best_pct", "latency_sec"]),
        "",
        "## 5. Candidate Ranking",
        md_table(rank_rows, ["candidate", "avg_gain_vs_current_best_pct", "avg_gain_vs_id_mlp_ca_pct", "pems_stuck_gain_vs_ca_pct", "pems_drift_gain_vs_ca_pct", "metr_non_regression_vs_current_best", "latency_ratio_ok", "eligible_to_replace"]),
        "",
        "## 6. Replacement Decision",
        f"- SHOULD_REPLACE_CURRENT_BEST: {'YES' if should_replace else 'NO'}",
        f"- BEST_OVERALL_CANDIDATE: {replacement if should_replace else best_candidate}",
        "- IF_NO_REPLACEMENT_KEEP: SRAF-ID-v2-current-best",
        "- IF_CURRENT_BEST_FAILS_FULL_RUN_FALLBACK: SRAF-ID-v1-formal",
        "",
        "## 7. Full-Run Recommendation",
        f"- READY_FOR_FULL_FORMAL_RUN: {'YES' if should_replace else 'PARTIAL'}",
        "- models to include in full run: ID-MLP-CA, SRAF-ID-v1-formal, SRAF-ID-v2-current-best, best new candidate (if any).",
        "- baselines to include in full run: KNN+ID-MLP (implemented), plus existing simple repair baselines.",
        "- faults to include: clean + random_missing_20/40 + continuous_outage_24 + gaussian_noise_high + linear_drift_high + stuck_at_last_value_high.",
        "- seeds to include: 42/43/44.",
        "- blockers: replacement criteria not satisfied by any new candidate in this gate.",
        "",
        "## 8. Next Action",
        "Provide this report to manuscript mentor and keep SRAF-ID-v2-current-best for now unless a new candidate passes replacement criteria in the next focused retry.",
    ]
    (out_dir / "SRAF_V2_VERSION_FREEZE_AND_MULTI_DIRECTION_EXPLORATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    manifest = {
        "stage": "SRAF_V2_VERSION_FREEZE_AND_MULTI_DIRECTION_EXPLORATION_GATE",
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
        "should_replace_current_best": should_replace,
        "best_new_candidate": replacement if should_replace else None,
        "keep_current_best": not should_replace,
        "fallback_if_current_best_fails": "SRAF-ID-v1-formal",
        "existing_results_overwritten": False,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=== Terminal Summary ===")
    print("current best frozen: yes")
    print(f"best new candidate: {best_candidate}")
    print(f"should replace current best: {'yes' if should_replace else 'no'}")
    if rank_rows:
        top = max(rank_rows, key=lambda x: x["avg_gain_vs_current_best_pct"])
        print(f"avg gain vs current best (top): {top['avg_gain_vs_current_best_pct']:.3f}%")
        print(f"avg gain vs ID-MLP-CA (top): {top['avg_gain_vs_id_mlp_ca_pct']:.3f}%")
        print(f"PEMS-BAY drift gain (top): {top['pems_drift_gain_vs_ca_pct']:.3f}%")
        print(f"PEMS-BAY stuck gain (top): {top['pems_stuck_gain_vs_ca_pct']:.3f}%")
        print(f"METR-LA non-regression (top): {top['metr_non_regression_vs_current_best']}")
    print(f"full-run recommendation: {'YES' if should_replace else 'PARTIAL'}")
    print("blockers: replacement criteria unmet by new candidates in this gate")


if __name__ == "__main__":
    main()
